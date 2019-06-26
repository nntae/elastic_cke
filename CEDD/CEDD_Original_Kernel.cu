// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>

// Files
#include <sstream>
#include <string>
#include <fstream>
#include <iostream>

// Vectors
#include <vector>

#include <iterator>
#include <algorithm> // for std::copy

using namespace std;

#include "../elastic_kernel.h"
#include "CEDD.h"

extern t_tqueue *tqueues;

__constant__ float gaus[3][3] = {{0.0625f, 0.125f, 0.0625f}, {0.1250f, 0.250f, 0.1250f}, {0.0625f, 0.125f, 0.0625f}};
__constant__ int   sobx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
__constant__ int   soby[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

const int GPU_IN_PROXY = 0;
const int GPU_OUT_PROXY = 1;
// unsigned char *    h_in_out[2];
// unsigned char *data_CEDD, *out_CEDD, *theta_CEDD;
int rows_CEDD, cols_CEDD, in_size;
int l_mem_size;

 __device__ uint get_smid_CEDD(void) {
	uint ret;

	asm("mov.u32 %0, %smid;" : "=r"(ret) );

	return ret;
}

/** 
 * Gaussian Canny (CUDA Kernel)
 */
__global__ void
original_gaussianCannyCUDA(unsigned char *data_CEDD, unsigned char *out_CEDD, int rows_CEDD, int cols_CEDD, 
						int gridDimY)
{
    extern __shared__ int l_mem[];
    int* l_data = l_mem;

    const int L_SIZE = blockDim.x;
    int sum         = 0;
    // const int   g_row = blockIdx.y * blockDim.y + threadIdx.y + 1;
    // const int   g_col = blockIdx.x * blockDim.x + threadIdx.x + 1;
	const int   g_row = (blockIdx.x % gridDimY) * blockDim.y + threadIdx.y + 1;
	const int   g_col = (blockIdx.x / gridDimY) * blockDim.x + threadIdx.x + 1;
    const int l_row = threadIdx.y + 1;
    const int l_col = threadIdx.x + 1;

    const int pos = g_row * cols_CEDD + g_col;

    // copy to local
    l_data[l_row * (L_SIZE + 2) + l_col] = data_CEDD[pos];

    // top most row
    if(l_row == 1) {
        l_data[0 * (L_SIZE + 2) + l_col] = data_CEDD[pos - cols_CEDD];
        // top left
        if(l_col == 1)
            l_data[0 * (L_SIZE + 2) + 0] = data_CEDD[pos - cols_CEDD - 1];

        // top right
        else if(l_col == L_SIZE)
            l_data[0 * (L_SIZE + 2) + L_SIZE + 1] = data_CEDD[pos - cols_CEDD + 1];
    }
    // bottom most row
    else if(l_row == L_SIZE) {
        l_data[(L_SIZE + 1) * (L_SIZE + 2) + l_col] = data_CEDD[pos + cols_CEDD];
        // bottom left
        if(l_col == 1)
            l_data[(L_SIZE + 1) * (L_SIZE + 2) + 0] = data_CEDD[pos + cols_CEDD - 1];

        // bottom right
        else if(l_col == L_SIZE)
            l_data[(L_SIZE + 1) * (L_SIZE + 2) + L_SIZE + 1] = data_CEDD[pos + cols_CEDD + 1];
    }

    if(l_col == 1)
        l_data[l_row * (L_SIZE + 2) + 0] = data_CEDD[pos - 1];
    else if(l_col == L_SIZE)
        l_data[l_row * (L_SIZE + 2) + L_SIZE + 1] = data_CEDD[pos + 1];

    __syncthreads();

    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            sum += gaus[i][j] * l_data[(i + l_row - 1) * (L_SIZE + 2) + j + l_col - 1];
        }
    }

    out_CEDD[pos] = min(255, max(0, sum));
}

__global__ void
SMT_gaussianCannyCUDA(unsigned char *data_CEDD, unsigned char *out_CEDD, int rows_CEDD, int cols_CEDD,
					int gridDimY,
					int SIMD_min, int SIMD_max,
					int num_subtask, int iter_per_subtask, int *cont_subtask, State *status)
{
	
	__shared__ int s_bid;
	
	unsigned int SM_id = get_smid_CEDD();
	
	if (SM_id <SIMD_min || SM_id > SIMD_max) /* Only blocks executing within SIMD_min and SIMD_max can progress */ 
			return;
	
	while (1){
		
		/********** Task Id calculation *************/
		
		if (threadIdx.x == 0 && threadIdx.y == 0) { 
			if (*status == TOEVICT)
				s_bid = -1;
			else {
				s_bid = atomicAdd(cont_subtask, 1);				//subtask_id
				//printf("Blq=%d cont=%d\n", blockIdx.x, s_bid);
			}
		}
		
		__syncthreads();
		
		//if (s_bid[warpid] >= num_subtask || s_bid[warpid] == -1)
		if (s_bid >=num_subtask || s_bid ==-1) /* If all subtasks have been executed */{
			//if (threadIdx.x == 0)  printf("El bloque %d se sale con %d\n", blockIdx.x, s_bid); 
			return;
		}
		
		extern __shared__ int l_mem[];
		int* l_data = l_mem;

		const int L_SIZE = blockDim.x;
		int sum         = 0;
		// const int   g_row = blockIdx.y * blockDim.y + threadIdx.y + 1;
		// const int   g_col = blockIdx.x * blockDim.x + threadIdx.x + 1;
		const int   g_row = (s_bid % gridDimY) * blockDim.y + threadIdx.y + 1;
		const int   g_col = (s_bid / gridDimY) * blockDim.x + threadIdx.x + 1;
		const int l_row = threadIdx.y + 1;
		const int l_col = threadIdx.x + 1;

		const int pos = g_row * cols_CEDD + g_col;

		// copy to local
		l_data[l_row * (L_SIZE + 2) + l_col] = data_CEDD[pos];

		// top most row
		if(l_row == 1) {
			l_data[0 * (L_SIZE + 2) + l_col] = data_CEDD[pos - cols_CEDD];
			// top left
			if(l_col == 1)
				l_data[0 * (L_SIZE + 2) + 0] = data_CEDD[pos - cols_CEDD - 1];

			// top right
			else if(l_col == L_SIZE)
				l_data[0 * (L_SIZE + 2) + L_SIZE + 1] = data_CEDD[pos - cols_CEDD + 1];
		}
		// bottom most row
		else if(l_row == L_SIZE) {
			l_data[(L_SIZE + 1) * (L_SIZE + 2) + l_col] = data_CEDD[pos + cols_CEDD];
			// bottom left
			if(l_col == 1)
				l_data[(L_SIZE + 1) * (L_SIZE + 2) + 0] = data_CEDD[pos + cols_CEDD - 1];

			// bottom right
			else if(l_col == L_SIZE)
				l_data[(L_SIZE + 1) * (L_SIZE + 2) + L_SIZE + 1] = data_CEDD[pos + cols_CEDD + 1];
		}

		if(l_col == 1)
			l_data[l_row * (L_SIZE + 2) + 0] = data_CEDD[pos - 1];
		else if(l_col == L_SIZE)
			l_data[l_row * (L_SIZE + 2) + L_SIZE + 1] = data_CEDD[pos + 1];

		__syncthreads();

		for(int i = 0; i < 3; i++) {
			for(int j = 0; j < 3; j++) {
				sum += gaus[i][j] * l_data[(i + l_row - 1) * (L_SIZE + 2) + j + l_col - 1];
			}
		}

		out_CEDD[pos] = min(255, max(0, sum));
	}
}

__global__ void
SMK_gaussianCannyCUDA(unsigned char *data_CEDD, unsigned char *out_CEDD, int rows_CEDD, int cols_CEDD,
					int gridDimY,
					int max_blocks_per_SM,
					int num_subtask,
					int iter_per_subtask,
					int *cont_SM,
					int *cont_subtask,
					State *status
)
{
	__shared__ int s_bid, s_index;
	
	unsigned int SM_id = get_smid_CEDD();
	
	if (threadIdx.x == 0 && threadIdx.y == 0)  
		s_index = atomicAdd(&cont_SM[SM_id],1);
	
	__syncthreads();

	if (s_index > max_blocks_per_SM)
		return;
	
	while (1){
		
		/********** Task Id calculation *************/
		if (threadIdx.x == 0 && threadIdx.y == 0) {
			if (*status == TOEVICT)
				s_bid = -1;
			else
				s_bid = atomicAdd(cont_subtask, 1);
		}
		
		__syncthreads();
		
		if (s_bid >= num_subtask || s_bid == -1) /* If all subtasks have been executed */
			return;
		
		extern __shared__ int l_mem[];
		int* l_data = l_mem;

		const int L_SIZE = blockDim.x;
		int sum         = 0;
		// const int   g_row = blockIdx.y * blockDim.y + threadIdx.y + 1;
		// const int   g_col = blockIdx.x * blockDim.x + threadIdx.x + 1;
		const int   g_row = (s_bid % gridDimY) * blockDim.y + threadIdx.y + 1;
		const int   g_col = (s_bid / gridDimY) * blockDim.x + threadIdx.x + 1;
		const int l_row = threadIdx.y + 1;
		const int l_col = threadIdx.x + 1;

		const int pos = g_row * cols_CEDD + g_col;

		// copy to local
		l_data[l_row * (L_SIZE + 2) + l_col] = data_CEDD[pos];

		// top most row
		if(l_row == 1) {
			l_data[0 * (L_SIZE + 2) + l_col] = data_CEDD[pos - cols_CEDD];
			// top left
			if(l_col == 1)
				l_data[0 * (L_SIZE + 2) + 0] = data_CEDD[pos - cols_CEDD - 1];

			// top right
			else if(l_col == L_SIZE)
				l_data[0 * (L_SIZE + 2) + L_SIZE + 1] = data_CEDD[pos - cols_CEDD + 1];
		}
		// bottom most row
		else if(l_row == L_SIZE) {
			l_data[(L_SIZE + 1) * (L_SIZE + 2) + l_col] = data_CEDD[pos + cols_CEDD];
			// bottom left
			if(l_col == 1)
				l_data[(L_SIZE + 1) * (L_SIZE + 2) + 0] = data_CEDD[pos + cols_CEDD - 1];

			// bottom right
			else if(l_col == L_SIZE)
				l_data[(L_SIZE + 1) * (L_SIZE + 2) + L_SIZE + 1] = data_CEDD[pos + cols_CEDD + 1];
		}

		if(l_col == 1)
			l_data[l_row * (L_SIZE + 2) + 0] = data_CEDD[pos - 1];
		else if(l_col == L_SIZE)
			l_data[l_row * (L_SIZE + 2) + L_SIZE + 1] = data_CEDD[pos + 1];

		__syncthreads();

		for(int i = 0; i < 3; i++) {
			for(int j = 0; j < 3; j++) {
				sum += gaus[i][j] * l_data[(i + l_row - 1) * (L_SIZE + 2) + j + l_col - 1];
			}
		}

		out_CEDD[pos] = min(255, max(0, sum));
	}
}

int launch_orig_GCEDD(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	
	t_CEDD_params * params = (t_CEDD_params *)kstub->params;
	
	dim3 dimGrid(kstub->kconf.gridsize.x);
    dim3 threads(kstub->kconf.blocksize.x, kstub->kconf.blocksize.y);
	
	original_gaussianCannyCUDA<<<dimGrid, threads, l_mem_size>>>(
		params->out_CEDD, params->data_CEDD, rows_CEDD, cols_CEDD,
		params->gridDimY);

	return 0;
}

int launch_preemp_GCEDD(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	
	t_CEDD_params * params = (t_CEDD_params *)kstub->params;
	
	//dim3 dimGrid((cols_CEDD-2)/threads, (rows_CEDD-2)/threads);
    dim3 threads(kstub->kconf.blocksize.x, kstub->kconf.blocksize.y);
	
	#ifdef SMT
		SMT_gaussianCannyCUDA<<< kstub->kconf.numSMs * kstub->kconf.max_persistent_blocks, threads, l_mem_size, *(kstub->execution_s) >>>(
			params->out_CEDD, params->data_CEDD, rows_CEDD, cols_CEDD,
			
			params->gridDimY,
			
			kstub->idSMs[0],
			kstub->idSMs[1],
			kstub->total_tasks,
			kstub->kconf.coarsening,
			kstub->d_executed_tasks,
			&(kstub->gm_state[kstub->stream_index]));
	#else
		SMK_gaussianCannyCUDA<<< kstub->kconf.numSMs * kstub->kconf.max_persistent_blocks, threads, l_mem_size, *(kstub->execution_s) >>>(
			params->out_CEDD, params->data_CEDD, rows_CEDD, cols_CEDD,
			
			params->gridDimY,
			
			kstub->num_blocks_per_SM,
			kstub->total_tasks,
			kstub->kconf.coarsening,
			kstub->d_SMs_cont,
			kstub->d_executed_tasks,
			&(kstub->gm_state[kstub->stream_index]));
	#endif
	
	return 0;
}

/**
 * Sobel Canny (CUDA Kernel)
 */
__global__ void
original_sobelCannyCUDA(unsigned char *data_CEDD, unsigned char *out_CEDD, unsigned char *theta_CEDD, int rows_CEDD, int cols_CEDD, 
						int gridDimY, int iter_per_block)
{
    extern __shared__ int l_mem[];
    int* l_data_CEDD = l_mem;

    // collect sums separately. we're storing them into floats because that
    // is what hypot and atan2 will expect.
    const int L_SIZE = blockDim.x;
    const float PI    = 3.14159265f;
    // const int   g_row = blockIdx.y * blockDim.y + threadIdx.y + 1;
    // const int   g_col = blockIdx.x * blockDim.x + threadIdx.x + 1;
	
	for(int k = 0; k < iter_per_block; k++){
	
		const int   g_row = (blockIdx.x % gridDimY) * blockDim.y * iter_per_block + k * blockDim.y + threadIdx.y + 1;
		const int   g_col = (blockIdx.x / gridDimY) * blockDim.x * iter_per_block + k * blockDim.x + threadIdx.x + 1;
		const int   l_row = threadIdx.y + 1;
		const int   l_col = threadIdx.x + 1;
		
		if(g_row < rows_CEDD && g_col < cols_CEDD){

			const int pos = g_row * cols_CEDD + g_col;

			// copy to local
			l_data_CEDD[l_row * (L_SIZE + 2) + l_col] = data_CEDD[pos];

			// top most row
			if(l_row == 1) {
				l_data_CEDD[0 * (L_SIZE + 2) + l_col] = data_CEDD[pos - cols_CEDD];
				// top left
				if(l_col == 1)
					l_data_CEDD[0 * (L_SIZE + 2) + 0] = data_CEDD[pos - cols_CEDD - 1];

				// top right
				else if(l_col == L_SIZE)
					l_data_CEDD[0 * (L_SIZE + 2) + (L_SIZE + 1)] = data_CEDD[pos - cols_CEDD + 1];
			}
			// bottom most row
			else if(l_row == L_SIZE) {
				l_data_CEDD[(L_SIZE + 1) * (L_SIZE + 2) + l_col] = data_CEDD[pos + cols_CEDD];
				// bottom left
				if(l_col == 1)
					l_data_CEDD[(L_SIZE + 1) * (L_SIZE + 2) + 0] = data_CEDD[pos + cols_CEDD - 1];

				// bottom right
				else if(l_col == L_SIZE)
					l_data_CEDD[(L_SIZE + 1) * (L_SIZE + 2) + (L_SIZE + 1)] = data_CEDD[pos + cols_CEDD + 1];
			}

			// left
			if(l_col == 1)
				l_data_CEDD[l_row * (L_SIZE + 2) + 0] = data_CEDD[pos - 1];
			// right
			else if(l_col == L_SIZE)
				l_data_CEDD[l_row * (L_SIZE + 2) + (L_SIZE + 1)] = data_CEDD[pos + 1];

			__syncthreads();

			float sumx = 0, sumy = 0, angle = 0;
			// find x and y derivatives
			for(int i = 0; i < 3; i++) {
				for(int j = 0; j < 3; j++) {
					sumx += sobx[i][j] * l_data_CEDD[(i + l_row - 1) * (L_SIZE + 2) + j + l_col - 1];
					sumy += soby[i][j] * l_data_CEDD[(i + l_row - 1) * (L_SIZE + 2) + j + l_col - 1];
				}
			}

			// The out_CEDDput is now the square root of their squares, but they are
			// constrained to 0 <= value <= 255. Note that hypot is a built in function
			// defined as: hypot(x,y) = sqrt(x*x, y*y).
			out_CEDD[pos] = min(255, max(0, (int)hypot(sumx, sumy)));

			// Compute the direction angle theta_CEDD in radians
			// atan2 has a range of (-PI, PI) degrees
			angle = atan2(sumy, sumx);

			// If the angle is negative,
			// shift the range to (0, 2PI) by adding 2PI to the angle,
			// then perform modulo operation of 2PI
			if(angle < 0) {
				angle = fmod((angle + 2 * PI), (2 * PI));
			}

			// Round the angle to one of four possibilities: 0, 45, 90, 135 degrees
			// then store it in the theta_CEDD buffer at the proper position
			//theta_CEDD[pos] = ((int)(degrees(angle * (PI/8) + PI/8-0.0001) / 45) * 45) % 180;
			if(angle <= PI / 8)
				theta_CEDD[pos] = 0;
			else if(angle <= 3 * PI / 8)
				theta_CEDD[pos] = 45;
			else if(angle <= 5 * PI / 8)
				theta_CEDD[pos] = 90;
			else if(angle <= 7 * PI / 8)
				theta_CEDD[pos] = 135;
			else if(angle <= 9 * PI / 8)
				theta_CEDD[pos] = 0;
			else if(angle <= 11 * PI / 8)
				theta_CEDD[pos] = 45;
			else if(angle <= 13 * PI / 8)
				theta_CEDD[pos] = 90;
			else if(angle <= 15 * PI / 8)
				theta_CEDD[pos] = 135;
			else
				theta_CEDD[pos] = 0; // (angle <= 16*PI/8)
		}
	}
}

__global__ void
__launch_bounds__(256, 8)
SMT_sobelCannyCUDA(unsigned char *data_CEDD, unsigned char *out_CEDD, unsigned char *theta_CEDD, int rows_CEDD, int cols_CEDD,
					int gridDimY,
					int SIMD_min, int SIMD_max,
					int num_subtask, int iter_per_subtask, int *cont_subtask, State *status)
{
	
	__shared__ int s_bid;
	
	unsigned int SM_id = get_smid_CEDD();
	
	if (SM_id <SIMD_min || SM_id > SIMD_max) /* Only blocks executing within SIMD_min and SIMD_max can progress */ 
			return;
	
	while (1){
		
		/********** Task Id calculation *************/
		
		if (threadIdx.x == 0 && threadIdx.y == 0) { 
			if (*status == TOEVICT)
				s_bid = -1;
			else {
				s_bid = atomicAdd(cont_subtask, 1);				//subtask_id
				//printf("Blq=%d cont=%d\n", blockIdx.x, s_bid);
			}
		}
		
		__syncthreads();
		
		//if (s_bid[warpid] >= num_subtask || s_bid[warpid] == -1)
		if (s_bid >=num_subtask || s_bid ==-1) /* If all subtasks have been executed */{
			//if (threadIdx.x == 0)  printf("El bloque %d se sale con %d\n", blockIdx.x, s_bid); 
			return;
		}
		
		extern __shared__ int l_mem[];
		int* l_data_CEDD = l_mem;

		// collect sums separately. we're storing them into floats because that
		// is what hypot and atan2 will expect.
		const int L_SIZE = blockDim.x;
		const float PI    = 3.14159265f;
		
		for(int j = 0; j < iter_per_subtask; j++){
		
			// const int   g_row = blockIdx.y * blockDim.y + threadIdx.y + 1;
			// const int   g_col = blockIdx.x * blockDim.x + threadIdx.x + 1;
			const int   g_row = (s_bid % gridDimY) * blockDim.y * iter_per_subtask + j * blockDim.y + threadIdx.y + 1;
			const int   g_col = (s_bid / gridDimY) * blockDim.x * iter_per_subtask + j * blockDim.x + threadIdx.x + 1;
			const int   l_row = threadIdx.y + 1;
			const int   l_col = threadIdx.x + 1;

			if(g_row < rows_CEDD && g_col < cols_CEDD){
				const int pos = g_row * cols_CEDD + g_col;

				// copy to local
				l_data_CEDD[l_row * (L_SIZE + 2) + l_col] = data_CEDD[pos];

				// top most row
				if(l_row == 1) {
					l_data_CEDD[0 * (L_SIZE + 2) + l_col] = data_CEDD[pos - cols_CEDD];
					// top left
					if(l_col == 1)
						l_data_CEDD[0 * (L_SIZE + 2) + 0] = data_CEDD[pos - cols_CEDD - 1];

					// top right
					else if(l_col == L_SIZE)
						l_data_CEDD[0 * (L_SIZE + 2) + (L_SIZE + 1)] = data_CEDD[pos - cols_CEDD + 1];
				}
				// bottom most row
				else if(l_row == L_SIZE) {
					l_data_CEDD[(L_SIZE + 1) * (L_SIZE + 2) + l_col] = data_CEDD[pos + cols_CEDD];
					// bottom left
					if(l_col == 1)
						l_data_CEDD[(L_SIZE + 1) * (L_SIZE + 2) + 0] = data_CEDD[pos + cols_CEDD - 1];

					// bottom right
					else if(l_col == L_SIZE)
						l_data_CEDD[(L_SIZE + 1) * (L_SIZE + 2) + (L_SIZE + 1)] = data_CEDD[pos + cols_CEDD + 1];
				}

				// left
				if(l_col == 1)
					l_data_CEDD[l_row * (L_SIZE + 2) + 0] = data_CEDD[pos - 1];
				// right
				else if(l_col == L_SIZE)
					l_data_CEDD[l_row * (L_SIZE + 2) + (L_SIZE + 1)] = data_CEDD[pos + 1];

				__syncthreads();

				float sumx = 0, sumy = 0, angle = 0;
				// find x and y derivatives
				for(int i = 0; i < 3; i++) {
					for(int j = 0; j < 3; j++) {
						sumx += sobx[i][j] * l_data_CEDD[(i + l_row - 1) * (L_SIZE + 2) + j + l_col - 1];
						sumy += soby[i][j] * l_data_CEDD[(i + l_row - 1) * (L_SIZE + 2) + j + l_col - 1];
					}
				}

				// The out_CEDDput is now the square root of their squares, but they are
				// constrained to 0 <= value <= 255. Note that hypot is a built in function
				// defined as: hypot(x,y) = sqrt(x*x, y*y).
				out_CEDD[pos] = min(255, max(0, (int)hypot(sumx, sumy)));

				// Compute the direction angle theta_CEDD in radians
				// atan2 has a range of (-PI, PI) degrees
				angle = atan2(sumy, sumx);

				// If the angle is negative,
				// shift the range to (0, 2PI) by adding 2PI to the angle,
				// then perform modulo operation of 2PI
				if(angle < 0) {
					angle = fmod((angle + 2 * PI), (2 * PI));
				}

				// Round the angle to one of four possibilities: 0, 45, 90, 135 degrees
				// then store it in the theta_CEDD buffer at the proper position
				//theta_CEDD[pos] = ((int)(degrees(angle * (PI/8) + PI/8-0.0001) / 45) * 45) % 180;
				if(angle <= PI / 8)
					theta_CEDD[pos] = 0;
				else if(angle <= 3 * PI / 8)
					theta_CEDD[pos] = 45;
				else if(angle <= 5 * PI / 8)
					theta_CEDD[pos] = 90;
				else if(angle <= 7 * PI / 8)
					theta_CEDD[pos] = 135;
				else if(angle <= 9 * PI / 8)
					theta_CEDD[pos] = 0;
				else if(angle <= 11 * PI / 8)
					theta_CEDD[pos] = 45;
				else if(angle <= 13 * PI / 8)
					theta_CEDD[pos] = 90;
				else if(angle <= 15 * PI / 8)
					theta_CEDD[pos] = 135;
				else
					theta_CEDD[pos] = 0; // (angle <= 16*PI/8)
			}
		}
	}
}

__global__ void
__launch_bounds__(256, 8)
SMK_sobelCannyCUDA(unsigned char *data_CEDD, unsigned char *out_CEDD, unsigned char *theta_CEDD, int rows_CEDD, int cols_CEDD,
					int gridDimY,
					int max_blocks_per_SM,
					int num_subtask,
					int iter_per_subtask,
					int *cont_SM,
					int *cont_subtask,
					State *status
)
{
	__shared__ int s_bid, s_index;
	
	unsigned int SM_id = get_smid_CEDD();
	
	if (threadIdx.x == 0 && threadIdx.y == 0)  
		s_index = atomicAdd(&cont_SM[SM_id],1);
	
	__syncthreads();

	if (s_index > max_blocks_per_SM)
		return;
	
	while (1){
		
		/********** Task Id calculation *************/
		if (threadIdx.x == 0 && threadIdx.y == 0) {
			if (*status == TOEVICT)
				s_bid = -1;
			else
				s_bid = atomicAdd(cont_subtask, 1);
		}
		
		__syncthreads();
		
		if (s_bid >= num_subtask || s_bid == -1) /* If all subtasks have been executed */
			return;
		
		extern __shared__ int l_mem[];
		int* l_data_CEDD = l_mem;

		// collect sums separately. we're storing them into floats because that
		// is what hypot and atan2 will expect.
		const int L_SIZE = blockDim.x;
		const float PI    = 3.14159265f;
		
		for(int j = 0; j < iter_per_subtask; j++){
			// const int   g_row = blockIdx.y * blockDim.y + threadIdx.y + 1;
			// const int   g_col = blockIdx.x * blockDim.x + threadIdx.x + 1;
			const int   g_row = (s_bid % gridDimY) * blockDim.y * iter_per_subtask + j * blockDim.y + threadIdx.y + 1;
			const int   g_col = (s_bid / gridDimY) * blockDim.x * iter_per_subtask + j * blockDim.x + threadIdx.x + 1;
			const int   l_row = threadIdx.y + 1;
			const int   l_col = threadIdx.x + 1;

			if(g_row < rows_CEDD && g_col < cols_CEDD){
				const int pos = g_row * cols_CEDD + g_col;

				// copy to local
				l_data_CEDD[l_row * (L_SIZE + 2) + l_col] = data_CEDD[pos];

				// top most row
				if(l_row == 1) {
					l_data_CEDD[0 * (L_SIZE + 2) + l_col] = data_CEDD[pos - cols_CEDD];
					// top left
					if(l_col == 1)
						l_data_CEDD[0 * (L_SIZE + 2) + 0] = data_CEDD[pos - cols_CEDD - 1];

					// top right
					else if(l_col == L_SIZE)
						l_data_CEDD[0 * (L_SIZE + 2) + (L_SIZE + 1)] = data_CEDD[pos - cols_CEDD + 1];
				}
				// bottom most row
				else if(l_row == L_SIZE) {
					l_data_CEDD[(L_SIZE + 1) * (L_SIZE + 2) + l_col] = data_CEDD[pos + cols_CEDD];
					// bottom left
					if(l_col == 1)
						l_data_CEDD[(L_SIZE + 1) * (L_SIZE + 2) + 0] = data_CEDD[pos + cols_CEDD - 1];

					// bottom right
					else if(l_col == L_SIZE)
						l_data_CEDD[(L_SIZE + 1) * (L_SIZE + 2) + (L_SIZE + 1)] = data_CEDD[pos + cols_CEDD + 1];
				}

				// left
				if(l_col == 1)
					l_data_CEDD[l_row * (L_SIZE + 2) + 0] = data_CEDD[pos - 1];
				// right
				else if(l_col == L_SIZE)
					l_data_CEDD[l_row * (L_SIZE + 2) + (L_SIZE + 1)] = data_CEDD[pos + 1];

				__syncthreads();

				float sumx = 0, sumy = 0, angle = 0;
				// find x and y derivatives
				for(int i = 0; i < 3; i++) {
					for(int j = 0; j < 3; j++) {
						sumx += sobx[i][j] * l_data_CEDD[(i + l_row - 1) * (L_SIZE + 2) + j + l_col - 1];
						sumy += soby[i][j] * l_data_CEDD[(i + l_row - 1) * (L_SIZE + 2) + j + l_col - 1];
					}
				}

				// The out_CEDDput is now the square root of their squares, but they are
				// constrained to 0 <= value <= 255. Note that hypot is a built in function
				// defined as: hypot(x,y) = sqrt(x*x, y*y).
				out_CEDD[pos] = min(255, max(0, (int)hypot(sumx, sumy)));

				// Compute the direction angle theta_CEDD in radians
				// atan2 has a range of (-PI, PI) degrees
				angle = atan2(sumy, sumx);

				// If the angle is negative,
				// shift the range to (0, 2PI) by adding 2PI to the angle,
				// then perform modulo operation of 2PI
				if(angle < 0) {
					angle = fmod((angle + 2 * PI), (2 * PI));
				}

				// Round the angle to one of four possibilities: 0, 45, 90, 135 degrees
				// then store it in the theta_CEDD buffer at the proper position
				//theta_CEDD[pos] = ((int)(degrees(angle * (PI/8) + PI/8-0.0001) / 45) * 45) % 180;
				if(angle <= PI / 8)
					theta_CEDD[pos] = 0;
				else if(angle <= 3 * PI / 8)
					theta_CEDD[pos] = 45;
				else if(angle <= 5 * PI / 8)
					theta_CEDD[pos] = 90;
				else if(angle <= 7 * PI / 8)
					theta_CEDD[pos] = 135;
				else if(angle <= 9 * PI / 8)
					theta_CEDD[pos] = 0;
				else if(angle <= 11 * PI / 8)
					theta_CEDD[pos] = 45;
				else if(angle <= 13 * PI / 8)
					theta_CEDD[pos] = 90;
				else if(angle <= 15 * PI / 8)
					theta_CEDD[pos] = 135;
				else
					theta_CEDD[pos] = 0; // (angle <= 16*PI/8)
			}
		}
	}
}

int launch_orig_SCEDD(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	
	t_CEDD_params * params = (t_CEDD_params *)kstub->params;
	
	dim3 dimGrid(kstub->kconf.gridsize.x);
    dim3 threads(kstub->kconf.blocksize.x, kstub->kconf.blocksize.y);
	
	original_sobelCannyCUDA<<<dimGrid, threads, l_mem_size>>>(
		params->data_CEDD, params->out_CEDD, params->theta_CEDD, rows_CEDD, cols_CEDD,
		params->gridDimY,
		
		kstub->kconf.coarsening);

	return 0;
}

int launch_preemp_SCEDD(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	
	t_CEDD_params * params = (t_CEDD_params *)kstub->params;
	
	//dim3 dimGrid((cols_CEDD-2)/threads, (rows_CEDD-2)/threads);
    dim3 threads(kstub->kconf.blocksize.x, kstub->kconf.blocksize.y);
	
	#ifdef SMT
		SMT_sobelCannyCUDA<<< kstub->kconf.numSMs * kstub->kconf.max_persistent_blocks, threads, l_mem_size, *(kstub->execution_s) >>>(
			params->data_CEDD, params->out_CEDD, params->theta_CEDD, rows_CEDD, cols_CEDD,
			
			params->gridDimY,
			
			kstub->idSMs[0],
			kstub->idSMs[1],
			kstub->total_tasks,
			kstub->kconf.coarsening,
			kstub->d_executed_tasks,
			&(kstub->gm_state[kstub->stream_index]));
	#else
		SMK_sobelCannyCUDA<<< kstub->kconf.numSMs * kstub->kconf.max_persistent_blocks, threads, l_mem_size, *(kstub->execution_s) >>>(
			params->data_CEDD, params->out_CEDD, params->theta_CEDD, rows_CEDD, cols_CEDD,
			
			params->gridDimY,
			
			kstub->num_blocks_per_SM,
			kstub->total_tasks,
			kstub->kconf.coarsening,
			kstub->d_SMs_cont,
			kstub->d_executed_tasks,
			&(kstub->gm_state[kstub->stream_index]));
	#endif
	
	return 0;
}

/**
 * Non_max_supp Canny (CUDA Kernel)
 */
__global__ void
original_nonCannyCUDA(unsigned char *data_CEDD, unsigned char *out_CEDD, unsigned char *theta_CEDD, int rows_CEDD, int cols_CEDD, 
						int gridDimY)
{
    extern __shared__ int l_mem[];
    int* l_data = l_mem;

    // These variables are offset by one to avoid seg. fault errors
    // As such, this kernel ignores the outside ring of pixels
    const int L_SIZE = blockDim.x;
    // const int   g_row = blockIdx.y * blockDim.y + threadIdx.y + 1;
	// const int   g_col = blockIdx.x * blockDim.x + threadIdx.x + 1;
	const int   g_row = (blockIdx.x % gridDimY) * blockDim.y + threadIdx.y + 1;
	const int   g_col = (blockIdx.x / gridDimY) * blockDim.x + threadIdx.x + 1;
    const int l_row = threadIdx.y + 1;
    const int l_col = threadIdx.x + 1;

    const int pos = g_row * cols_CEDD + g_col;

    // copy to l_data
    l_data[l_row * (L_SIZE + 2) + l_col] = data_CEDD[pos];

    // top most row
    if(l_row == 1) {
        l_data[0 * (L_SIZE + 2) + l_col] = data_CEDD[pos - cols_CEDD];
        // top left
        if(l_col == 1)
            l_data[0 * (L_SIZE + 2) + 0] = data_CEDD[pos - cols_CEDD - 1];

        // top right
        else if(l_col == L_SIZE)
            l_data[0 * (L_SIZE + 2) + (L_SIZE + 1)] = data_CEDD[pos - cols_CEDD + 1];
    }
    // bottom most row
    else if(l_row == L_SIZE) {
        l_data[(L_SIZE + 1) * (L_SIZE + 2) + l_col] = data_CEDD[pos + cols_CEDD];
        // bottom left
        if(l_col == 1)
            l_data[(L_SIZE + 1) * (L_SIZE + 2) + 0] = data_CEDD[pos + cols_CEDD - 1];

        // bottom right
        else if(l_col == L_SIZE)
            l_data[(L_SIZE + 1) * (L_SIZE + 2) + (L_SIZE + 1)] = data_CEDD[pos + cols_CEDD + 1];
    }

    if(l_col == 1)
        l_data[l_row * (L_SIZE + 2) + 0] = data_CEDD[pos - 1];
    else if(l_col == L_SIZE)
        l_data[l_row * (L_SIZE + 2) + (L_SIZE + 1)] = data_CEDD[pos + 1];

    __syncthreads();

    unsigned char my_magnitude = l_data[l_row * (L_SIZE + 2) + l_col];

    // The following variables are used to address the matrices more easily
    switch(theta_CEDD[pos]) {
    // A gradient angle of 0 degrees = an edge that is North/South
    // Check neighbors to the East and West
    case 0:
        // supress me if my neighbor has larger magnitude
        if(my_magnitude <= l_data[l_row * (L_SIZE + 2) + l_col + 1] || // east
            my_magnitude <= l_data[l_row * (L_SIZE + 2) + l_col - 1]) // west
        {
            out_CEDD[pos] = 0;
        }
        // otherwise, copy my value to the output buffer
        else {
            out_CEDD[pos] = my_magnitude;
        }
        break;

    // A gradient angle of 45 degrees = an edge that is NW/SE
    // Check neighbors to the NE and SW
    case 45:
        // supress me if my neighbor has larger magnitude
        if(my_magnitude <= l_data[(l_row - 1) * (L_SIZE + 2) + l_col + 1] || // north east
            my_magnitude <= l_data[(l_row + 1) * (L_SIZE + 2) + l_col - 1]) // south west
        {
            out_CEDD[pos] = 0;
        }
        // otherwise, copy my value to the output buffer
        else {
            out_CEDD[pos] = my_magnitude;
        }
        break;

    // A gradient angle of 90 degrees = an edge that is E/W
    // Check neighbors to the North and South
    case 90:
        // supress me if my neighbor has larger magnitude
        if(my_magnitude <= l_data[(l_row - 1) * (L_SIZE + 2) + l_col] || // north
            my_magnitude <= l_data[(l_row + 1) * (L_SIZE + 2) + l_col]) // south
        {
            out_CEDD[pos] = 0;
        }
        // otherwise, copy my value to the output buffer
        else {
            out_CEDD[pos] = my_magnitude;
        }
        break;

    // A gradient angle of 135 degrees = an edge that is NE/SW
    // Check neighbors to the NW and SE
    case 135:
        // supress me if my neighbor has larger magnitude
        if(my_magnitude <= l_data[(l_row - 1) * (L_SIZE + 2) + l_col - 1] || // north west
            my_magnitude <= l_data[(l_row + 1) * (L_SIZE + 2) + l_col + 1]) // south east
        {
            out_CEDD[pos] = 0;
        }
        // otherwise, copy my value to the output buffer
        else {
            out_CEDD[pos] = my_magnitude;
        }
        break;

    default: out_CEDD[pos] = my_magnitude; break;
    }
}

__global__ void
SMT_nonCannyCUDA(unsigned char *data_CEDD, unsigned char *out_CEDD, unsigned char *theta_CEDD, int rows_CEDD, int cols_CEDD,
					int gridDimY,
					int SIMD_min, int SIMD_max,
					int num_subtask, int iter_per_subtask, int *cont_subtask, State *status)
{
	
	__shared__ int s_bid;
	
	unsigned int SM_id = get_smid_CEDD();
	
	if (SM_id <SIMD_min || SM_id > SIMD_max) /* Only blocks executing within SIMD_min and SIMD_max can progress */ 
			return;
	
	while (1){
		
		/********** Task Id calculation *************/
		
		if (threadIdx.x == 0 && threadIdx.y == 0) { 
			if (*status == TOEVICT)
				s_bid = -1;
			else {
				s_bid = atomicAdd(cont_subtask, 1);				//subtask_id
				//printf("Blq=%d cont=%d\n", blockIdx.x, s_bid);
			}
		}
		
		__syncthreads();
		
		//if (s_bid[warpid] >= num_subtask || s_bid[warpid] == -1)
		if (s_bid >=num_subtask || s_bid ==-1) /* If all subtasks have been executed */{
			//if (threadIdx.x == 0)  printf("El bloque %d se sale con %d\n", blockIdx.x, s_bid); 
			return;
		}
		
		extern __shared__ int l_mem[];
		int* l_data = l_mem;

		// These variables are offset by one to avoid seg. fault errors
		// As such, this kernel ignores the outside ring of pixels
		const int L_SIZE = blockDim.x;
		// const int   g_row = blockIdx.y * blockDim.y + threadIdx.y + 1;
		// const int   g_col = blockIdx.x * blockDim.x + threadIdx.x + 1;
		const int   g_row = (s_bid % gridDimY) * blockDim.y + threadIdx.y + 1;
		const int   g_col = (s_bid / gridDimY) * blockDim.x + threadIdx.x + 1;
		const int l_row = threadIdx.y + 1;
		const int l_col = threadIdx.x + 1;

		const int pos = g_row * cols_CEDD + g_col;

		// copy to l_data
		l_data[l_row * (L_SIZE + 2) + l_col] = data_CEDD[pos];

		// top most row
		if(l_row == 1) {
			l_data[0 * (L_SIZE + 2) + l_col] = data_CEDD[pos - cols_CEDD];
			// top left
			if(l_col == 1)
				l_data[0 * (L_SIZE + 2) + 0] = data_CEDD[pos - cols_CEDD - 1];

			// top right
			else if(l_col == L_SIZE)
				l_data[0 * (L_SIZE + 2) + (L_SIZE + 1)] = data_CEDD[pos - cols_CEDD + 1];
		}
		// bottom most row
		else if(l_row == L_SIZE) {
			l_data[(L_SIZE + 1) * (L_SIZE + 2) + l_col] = data_CEDD[pos + cols_CEDD];
			// bottom left
			if(l_col == 1)
				l_data[(L_SIZE + 1) * (L_SIZE + 2) + 0] = data_CEDD[pos + cols_CEDD - 1];

			// bottom right
			else if(l_col == L_SIZE)
				l_data[(L_SIZE + 1) * (L_SIZE + 2) + (L_SIZE + 1)] = data_CEDD[pos + cols_CEDD + 1];
		}

		if(l_col == 1)
			l_data[l_row * (L_SIZE + 2) + 0] = data_CEDD[pos - 1];
		else if(l_col == L_SIZE)
			l_data[l_row * (L_SIZE + 2) + (L_SIZE + 1)] = data_CEDD[pos + 1];

		__syncthreads();

		unsigned char my_magnitude = l_data[l_row * (L_SIZE + 2) + l_col];

		// The following variables are used to address the matrices more easily
		switch(theta_CEDD[pos]) {
		// A gradient angle of 0 degrees = an edge that is North/South
		// Check neighbors to the East and West
		case 0:
			// supress me if my neighbor has larger magnitude
			if(my_magnitude <= l_data[l_row * (L_SIZE + 2) + l_col + 1] || // east
				my_magnitude <= l_data[l_row * (L_SIZE + 2) + l_col - 1]) // west
			{
				out_CEDD[pos] = 0;
			}
			// otherwise, copy my value to the output buffer
			else {
				out_CEDD[pos] = my_magnitude;
			}
			break;

		// A gradient angle of 45 degrees = an edge that is NW/SE
		// Check neighbors to the NE and SW
		case 45:
			// supress me if my neighbor has larger magnitude
			if(my_magnitude <= l_data[(l_row - 1) * (L_SIZE + 2) + l_col + 1] || // north east
				my_magnitude <= l_data[(l_row + 1) * (L_SIZE + 2) + l_col - 1]) // south west
			{
				out_CEDD[pos] = 0;
			}
			// otherwise, copy my value to the output buffer
			else {
				out_CEDD[pos] = my_magnitude;
			}
			break;

		// A gradient angle of 90 degrees = an edge that is E/W
		// Check neighbors to the North and South
		case 90:
			// supress me if my neighbor has larger magnitude
			if(my_magnitude <= l_data[(l_row - 1) * (L_SIZE + 2) + l_col] || // north
				my_magnitude <= l_data[(l_row + 1) * (L_SIZE + 2) + l_col]) // south
			{
				out_CEDD[pos] = 0;
			}
			// otherwise, copy my value to the output buffer
			else {
				out_CEDD[pos] = my_magnitude;
			}
			break;

		// A gradient angle of 135 degrees = an edge that is NE/SW
		// Check neighbors to the NW and SE
		case 135:
			// supress me if my neighbor has larger magnitude
			if(my_magnitude <= l_data[(l_row - 1) * (L_SIZE + 2) + l_col - 1] || // north west
				my_magnitude <= l_data[(l_row + 1) * (L_SIZE + 2) + l_col + 1]) // south east
			{
				out_CEDD[pos] = 0;
			}
			// otherwise, copy my value to the output buffer
			else {
				out_CEDD[pos] = my_magnitude;
			}
			break;

		default: out_CEDD[pos] = my_magnitude; break;
		}
	}
}

__global__ void
SMK_nonCannyCUDA(unsigned char *data_CEDD, unsigned char *out_CEDD, unsigned char *theta_CEDD, int rows_CEDD, int cols_CEDD,
					int gridDimY,
					int max_blocks_per_SM,
					int num_subtask,
					int iter_per_subtask,
					int *cont_SM,
					int *cont_subtask,
					State *status
)
{
	__shared__ int s_bid, s_index;
	
	unsigned int SM_id = get_smid_CEDD();
	
	if (threadIdx.x == 0 && threadIdx.y == 0)  
		s_index = atomicAdd(&cont_SM[SM_id],1);
	
	__syncthreads();

	if (s_index > max_blocks_per_SM)
		return;
	
	while (1){
		
		/********** Task Id calculation *************/
		if (threadIdx.x == 0 && threadIdx.y == 0) {
			if (*status == TOEVICT)
				s_bid = -1;
			else
				s_bid = atomicAdd(cont_subtask, 1);
		}
		
		__syncthreads();
		
		if (s_bid >= num_subtask || s_bid == -1) /* If all subtasks have been executed */
			return;
		
		extern __shared__ int l_mem[];
		int* l_data = l_mem;

		// These variables are offset by one to avoid seg. fault errors
		// As such, this kernel ignores the outside ring of pixels
		const int L_SIZE = blockDim.x;
		// const int   g_row = blockIdx.y * blockDim.y + threadIdx.y + 1;
		// const int   g_col = blockIdx.x * blockDim.x + threadIdx.x + 1;
		const int   g_row = (s_bid % gridDimY) * blockDim.y + threadIdx.y + 1;
		const int   g_col = (s_bid / gridDimY) * blockDim.x + threadIdx.x + 1;
		const int l_row = threadIdx.y + 1;
		const int l_col = threadIdx.x + 1;

		const int pos = g_row * cols_CEDD + g_col;

		// copy to l_data
		l_data[l_row * (L_SIZE + 2) + l_col] = data_CEDD[pos];

		// top most row
		if(l_row == 1) {
			l_data[0 * (L_SIZE + 2) + l_col] = data_CEDD[pos - cols_CEDD];
			// top left
			if(l_col == 1)
				l_data[0 * (L_SIZE + 2) + 0] = data_CEDD[pos - cols_CEDD - 1];

			// top right
			else if(l_col == L_SIZE)
				l_data[0 * (L_SIZE + 2) + (L_SIZE + 1)] = data_CEDD[pos - cols_CEDD + 1];
		}
		// bottom most row
		else if(l_row == L_SIZE) {
			l_data[(L_SIZE + 1) * (L_SIZE + 2) + l_col] = data_CEDD[pos + cols_CEDD];
			// bottom left
			if(l_col == 1)
				l_data[(L_SIZE + 1) * (L_SIZE + 2) + 0] = data_CEDD[pos + cols_CEDD - 1];

			// bottom right
			else if(l_col == L_SIZE)
				l_data[(L_SIZE + 1) * (L_SIZE + 2) + (L_SIZE + 1)] = data_CEDD[pos + cols_CEDD + 1];
		}

		if(l_col == 1)
			l_data[l_row * (L_SIZE + 2) + 0] = data_CEDD[pos - 1];
		else if(l_col == L_SIZE)
			l_data[l_row * (L_SIZE + 2) + (L_SIZE + 1)] = data_CEDD[pos + 1];

		__syncthreads();

		unsigned char my_magnitude = l_data[l_row * (L_SIZE + 2) + l_col];

		// The following variables are used to address the matrices more easily
		switch(theta_CEDD[pos]) {
		// A gradient angle of 0 degrees = an edge that is North/South
		// Check neighbors to the East and West
		case 0:
			// supress me if my neighbor has larger magnitude
			if(my_magnitude <= l_data[l_row * (L_SIZE + 2) + l_col + 1] || // east
				my_magnitude <= l_data[l_row * (L_SIZE + 2) + l_col - 1]) // west
			{
				out_CEDD[pos] = 0;
			}
			// otherwise, copy my value to the output buffer
			else {
				out_CEDD[pos] = my_magnitude;
			}
			break;

		// A gradient angle of 45 degrees = an edge that is NW/SE
		// Check neighbors to the NE and SW
		case 45:
			// supress me if my neighbor has larger magnitude
			if(my_magnitude <= l_data[(l_row - 1) * (L_SIZE + 2) + l_col + 1] || // north east
				my_magnitude <= l_data[(l_row + 1) * (L_SIZE + 2) + l_col - 1]) // south west
			{
				out_CEDD[pos] = 0;
			}
			// otherwise, copy my value to the output buffer
			else {
				out_CEDD[pos] = my_magnitude;
			}
			break;

		// A gradient angle of 90 degrees = an edge that is E/W
		// Check neighbors to the North and South
		case 90:
			// supress me if my neighbor has larger magnitude
			if(my_magnitude <= l_data[(l_row - 1) * (L_SIZE + 2) + l_col] || // north
				my_magnitude <= l_data[(l_row + 1) * (L_SIZE + 2) + l_col]) // south
			{
				out_CEDD[pos] = 0;
			}
			// otherwise, copy my value to the output buffer
			else {
				out_CEDD[pos] = my_magnitude;
			}
			break;

		// A gradient angle of 135 degrees = an edge that is NE/SW
		// Check neighbors to the NW and SE
		case 135:
			// supress me if my neighbor has larger magnitude
			if(my_magnitude <= l_data[(l_row - 1) * (L_SIZE + 2) + l_col - 1] || // north west
				my_magnitude <= l_data[(l_row + 1) * (L_SIZE + 2) + l_col + 1]) // south east
			{
				out_CEDD[pos] = 0;
			}
			// otherwise, copy my value to the output buffer
			else {
				out_CEDD[pos] = my_magnitude;
			}
			break;

		default: out_CEDD[pos] = my_magnitude; break;
		}
	}
}

int launch_orig_NCEDD(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	
	t_CEDD_params * params = (t_CEDD_params *)kstub->params;
	
	dim3 dimGrid(kstub->kconf.gridsize.x);
    dim3 threads(kstub->kconf.blocksize.x, kstub->kconf.blocksize.y);
	
	original_nonCannyCUDA<<<dimGrid, threads, l_mem_size>>>(
		params->out_CEDD, params->data_CEDD, params->theta_CEDD, rows_CEDD, cols_CEDD,
		params->gridDimY);

	return 0;
}

int launch_preemp_NCEDD(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	
	t_CEDD_params * params = (t_CEDD_params *)kstub->params;
	
	//dim3 dimGrid((cols_CEDD-2)/threads, (rows_CEDD-2)/threads);
    dim3 threads(kstub->kconf.blocksize.x, kstub->kconf.blocksize.y);
	
	#ifdef SMT
		SMT_nonCannyCUDA<<< kstub->kconf.numSMs * kstub->kconf.max_persistent_blocks, threads, l_mem_size, *(kstub->execution_s) >>>(
			params->out_CEDD, params->data_CEDD, params->theta_CEDD, rows_CEDD, cols_CEDD,
			
			params->gridDimY,
			
			kstub->idSMs[0],
			kstub->idSMs[1],
			kstub->total_tasks,
			kstub->kconf.coarsening,
			kstub->d_executed_tasks,
			&(kstub->gm_state[kstub->stream_index]));
	#else
		SMK_nonCannyCUDA<<< kstub->kconf.numSMs * kstub->kconf.max_persistent_blocks, threads, l_mem_size, *(kstub->execution_s) >>>(
			params->out_CEDD, params->data_CEDD, params->theta_CEDD, rows_CEDD, cols_CEDD,
			
			params->gridDimY,
			
			kstub->num_blocks_per_SM,
			kstub->total_tasks,
			kstub->kconf.coarsening,
			kstub->d_SMs_cont,
			kstub->d_executed_tasks,
			&(kstub->gm_state[kstub->stream_index]));
	#endif		
	
	return 0;
}

/**
 * Hyst Canny (CUDA Kernel)
 */
__global__ void
original_hystCannyCUDA(unsigned char *data_CEDD, unsigned char *out_CEDD, int rows_CEDD, int cols_CEDD, 
						int gridDimY)
{
    // Establish our high and low thresholds as floats
    float lowThresh  = 10;
    float highThresh = 70;

    // These variables are offset by one to avoid seg. fault errors
    // As such, this kernel ignores the outside ring of pixels
    // const int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
    // const int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
	const int   row = (blockIdx.x % gridDimY) * blockDim.y + threadIdx.y + 1;
	const int   col = (blockIdx.x / gridDimY) * blockDim.x + threadIdx.x + 1;
    const int pos = row * cols_CEDD + col;

    const unsigned char EDGE = 255;

    unsigned char magnitude = data_CEDD[pos];

    if(magnitude >= highThresh)
        out_CEDD[pos] = EDGE;
    else if(magnitude <= lowThresh)
        out_CEDD[pos] = 0;
    else {
        float med = (highThresh + lowThresh) / 2;

        if(magnitude >= med)
            out_CEDD[pos] = EDGE;
        else
            out_CEDD[pos] = 0;
    }
}

__global__ void
SMT_hystCannyCUDA(unsigned char *data_CEDD, unsigned char *out_CEDD, int rows_CEDD, int cols_CEDD,
					int gridDimY,
					int SIMD_min, int SIMD_max,
					int num_subtask, int iter_per_subtask, int *cont_subtask, State *status)
{
	
	__shared__ int s_bid;
	
	unsigned int SM_id = get_smid_CEDD();
	
	if (SM_id <SIMD_min || SM_id > SIMD_max) /* Only blocks executing within SIMD_min and SIMD_max can progress */ 
			return;
	
	while (1){
		
		/********** Task Id calculation *************/
		
		if (threadIdx.x == 0 && threadIdx.y == 0) { 
			if (*status == TOEVICT)
				s_bid = -1;
			else {
				s_bid = atomicAdd(cont_subtask, 1);				//subtask_id
				//printf("Blq=%d cont=%d\n", blockIdx.x, s_bid);
			}
		}
		
		__syncthreads();
		
		//if (s_bid[warpid] >= num_subtask || s_bid[warpid] == -1)
		if (s_bid >=num_subtask || s_bid ==-1) /* If all subtasks have been executed */{
			//if (threadIdx.x == 0)  printf("El bloque %d se sale con %d\n", blockIdx.x, s_bid); 
			return;
		}
		
		// Establish our high and low thresholds as floats
		float lowThresh  = 10;
		float highThresh = 70;

		// These variables are offset by one to avoid seg. fault errors
		// As such, this kernel ignores the outside ring of pixels
		// const int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
		// const int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
		const int   row = (s_bid % gridDimY) * blockDim.y + threadIdx.y + 1;
		const int   col = (s_bid / gridDimY) * blockDim.x + threadIdx.x + 1;
		const int pos = row * cols_CEDD + col;

		const unsigned char EDGE = 255;

		unsigned char magnitude = data_CEDD[pos];

		if(magnitude >= highThresh)
			out_CEDD[pos] = EDGE;
		else if(magnitude <= lowThresh)
			out_CEDD[pos] = 0;
		else {
			float med = (highThresh + lowThresh) / 2;

			if(magnitude >= med)
				out_CEDD[pos] = EDGE;
			else
				out_CEDD[pos] = 0;
		}
	}
}

__global__ void
SMK_hystCannyCUDA(unsigned char *data_CEDD, unsigned char *out_CEDD, int rows_CEDD, int cols_CEDD,
					int gridDimY,
					int max_blocks_per_SM,
					int num_subtask,
					int iter_per_subtask,
					int *cont_SM,
					int *cont_subtask,
					State *status
)
{
	__shared__ int s_bid, s_index;
	
	unsigned int SM_id = get_smid_CEDD();
	
	if (threadIdx.x == 0 && threadIdx.y == 0)  
		s_index = atomicAdd(&cont_SM[SM_id],1);
	
	__syncthreads();

	if (s_index > max_blocks_per_SM)
		return;
	
	while (1){
		
		/********** Task Id calculation *************/
		if (threadIdx.x == 0 && threadIdx.y == 0) {
			if (*status == TOEVICT)
				s_bid = -1;
			else
				s_bid = atomicAdd(cont_subtask, 1);
		}
		
		__syncthreads();
		
		if (s_bid >= num_subtask || s_bid == -1) /* If all subtasks have been executed */
			return;
		
		// Establish our high and low thresholds as floats
		float lowThresh  = 10;
		float highThresh = 70;

		// These variables are offset by one to avoid seg. fault errors
		// As such, this kernel ignores the outside ring of pixels
		// const int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
		// const int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
		const int   row = (s_bid % gridDimY) * blockDim.y + threadIdx.y + 1;
		const int   col = (s_bid / gridDimY) * blockDim.x + threadIdx.x + 1;
		const int pos = row * cols_CEDD + col;

		const unsigned char EDGE = 255;

		unsigned char magnitude = data_CEDD[pos];

		if(magnitude >= highThresh)
			out_CEDD[pos] = EDGE;
		else if(magnitude <= lowThresh)
			out_CEDD[pos] = 0;
		else {
			float med = (highThresh + lowThresh) / 2;

			if(magnitude >= med)
				out_CEDD[pos] = EDGE;
			else
				out_CEDD[pos] = 0;
		}
	}
}

int launch_orig_HCEDD(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	
	t_CEDD_params * params = (t_CEDD_params *)kstub->params;
	
	dim3 dimGrid(kstub->kconf.gridsize.x);
    dim3 threads(kstub->kconf.blocksize.x, kstub->kconf.blocksize.y);
	
	original_hystCannyCUDA<<<dimGrid, threads>>>(
		params->data_CEDD, params->out_CEDD, rows_CEDD, cols_CEDD,
		params->gridDimY);

	return 0;
}

int launch_preemp_HCEDD(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	
	t_CEDD_params * params = (t_CEDD_params *)kstub->params;
	
	//dim3 dimGrid((cols_CEDD-2)/threads, (rows_CEDD-2)/threads);
    dim3 threads(kstub->kconf.blocksize.x, kstub->kconf.blocksize.y);
	
	#ifdef SMT
		SMT_hystCannyCUDA<<< kstub->kconf.numSMs * kstub->kconf.max_persistent_blocks, threads, 0, *(kstub->execution_s) >>>(
			params->data_CEDD, params->out_CEDD, rows_CEDD, cols_CEDD,
			
			params->gridDimY,
			
			kstub->idSMs[0],
			kstub->idSMs[1],
			kstub->total_tasks,
			kstub->kconf.coarsening,
			kstub->d_executed_tasks,
			&(kstub->gm_state[kstub->stream_index]));
	#else
		SMK_hystCannyCUDA<<< kstub->kconf.numSMs * kstub->kconf.max_persistent_blocks, threads, 0, *(kstub->execution_s) >>>(
			params->data_CEDD, params->out_CEDD, rows_CEDD, cols_CEDD,
			
			params->gridDimY,
			
			kstub->num_blocks_per_SM,
			kstub->total_tasks,
			kstub->kconf.coarsening,
			kstub->d_SMs_cont,
			kstub->d_executed_tasks,
			&(kstub->gm_state[kstub->stream_index]));
	#endif	
	
	return 0;
}

void loadVector(unsigned char *vector, string file, int size)
{
	srand(time(NULL));

	for(int i = 0; i < size; i++)
		vector[i] = rand() % 256;

	//FILE *fp = fopen(("./CEDD/" + file).c_str(), "r");
	//if(fp == NULL)
	//	exit(EXIT_FAILURE);
	//
	//for(int i = 0; i < size; i++) {
	//	fscanf(fp, "%u ", (unsigned int *)&vector[i]);
	//}
	
	//fclose(fp);	
}

int GCEDD_start_kernel(void *arg) 
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	
	t_CEDD_params * params = (t_CEDD_params *)kstub->params;
	rows_CEDD = params->nRows;
	cols_CEDD = params->nCols;
	
	in_size = rows_CEDD * cols_CEDD * sizeof(unsigned char);
	
    // h_in_out[GPU_IN_PROXY] = (unsigned char *)malloc(in_size);
	// h_in_out[GPU_OUT_PROXY] = (unsigned char *)malloc(in_size);
	cudaMallocHost(&params->h_in_out[GPU_IN_PROXY], in_size);
	cudaMallocHost(&params->h_in_out[GPU_OUT_PROXY], in_size);
	
	int size = rows_CEDD * cols_CEDD;
	
	loadVector(params->h_in_out[GPU_IN_PROXY], "input.txt", size);
	
    cudaMalloc((void**)&params->data_CEDD, in_size);
	cudaMalloc((void**)&params->out_CEDD, in_size);
	cudaMalloc((void**)&params->theta_CEDD, in_size);
	
	l_mem_size = (kstub->kconf.blocksize.x + 2) * (kstub->kconf.blocksize.y + 2) * sizeof(int);
	
	cudaMemcpy(params->out_CEDD, params->h_in_out[GPU_IN_PROXY], in_size, cudaMemcpyHostToDevice);
	
	return 0;
}

int GCEDD_start_mallocs(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	
	t_CEDD_params * params = (t_CEDD_params *)kstub->params;
	rows_CEDD = params->nRows;
	cols_CEDD = params->nCols;
	
	in_size = rows_CEDD * cols_CEDD * sizeof(unsigned char);
	
#if defined(MEMCPY_SYNC) || defined(MEMCPY_ASYNC)
	cudaMallocHost(&params->h_in_out[GPU_IN_PROXY], in_size);
	cudaMallocHost(&params->h_in_out[GPU_OUT_PROXY], in_size);

	cudaMalloc((void**)&params->data_CEDD, in_size);
	cudaMalloc((void**)&params->out_CEDD, in_size);
	cudaMalloc((void**)&params->theta_CEDD, in_size);
#else
	#ifdef MANAGED_MEM

	cudaMallocManaged(&params->h_in_out[GPU_IN_PROXY], in_size);
	cudaMallocManaged(&params->h_in_out[GPU_OUT_PROXY], in_size);
	
	loadVector(params->h_in_out[GPU_IN_PROXY], "input.txt", size);
	
	params->out_CEDD = params->h_in_out[GPU_IN_PROXY];
	#else
		printf("No transfer model: Exiting ...\n");
		exit(-1);
	#endif
#endif

	// Verify that allocations succeeded
    if (params->h_in_out[GPU_IN_PROXY] == NULL || params->h_in_out[GPU_OUT_PROXY] == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    int size = rows_CEDD * cols_CEDD;
	
	loadVector(params->h_in_out[GPU_IN_PROXY], "input.txt", size);
	
	l_mem_size = (kstub->kconf.blocksize.x + 2) * (kstub->kconf.blocksize.y + 2) * sizeof(int);

	return 0;
}

int GCEDD_start_transfers(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	
	t_CEDD_params * params = (t_CEDD_params *)kstub->params;
	rows_CEDD = params->nRows;
	cols_CEDD = params->nCols;
	
	in_size = rows_CEDD * cols_CEDD * sizeof(unsigned char);
	
#ifdef MEMCPY_SYNC
	enqueue_tcomamnd(tqueues, params->out_CEDD, params->h_in_out[GPU_IN_PROXY], in_size, cudaMemcpyHostToDevice, 0, BLOCKING, DATA, LOW, kstub);
	
	kstub->HtD_tranfers_finished = 1;

	
#else
	
	#ifdef MEMCPY_ASYNC
	
	//enqueue_tcomamnd(tqueues, out_CEDD, h_in_out[GPU_IN_PROXY], in_size, cudaMemcpyHostToDevice, 0, NONBLOCKING, LAST_TRANSFER, MEDIUM, kstub);
	cudaMemcpyAsync(params->out_CEDD, params->h_in_out[GPU_IN_PROXY], in_size, cudaMemcpyHostToDevice, kstub->transfer_s[0]);
	#else
	#ifdef MANAGED_MEM

	cudaDeviceProp p;
    cudaGetDeviceProperties(&p, kstub->deviceId);
	
	if (p.concurrentManagedAccess)
	{
		err = cudaMemPrefetchAsync(params->h_in_out[GPU_IN_PROXY], in_size, kstub->deviceId);
		if ( err != cudaSuccess) {
			printf("Error in vAdd:cudaMemPrefetchAsync\n");
			exit(EXIT_FAILURE);
		}
		err = cudaMemPrefetchAsync(params->h_in_out[GPU_OUT_PROXY], in_size, kstub->deviceId);
		if ( err != cudaSuccess) {
			printf("Error in vAdd:cudaMemPrefetchAsync\n");
			exit(EXIT_FAILURE);
		}
	}
	//cudaEventRecord(kstub->end_HtD, kstub->transfer_s[0]);
	
	//cudaStreamSynchronize(kstub->transfer_s[0]);
	kstub->HtD_tranfers_finished = 1;

	#endif
	#endif
#endif

	return 0;
}

int GCEDD_end_kernel(void *arg)
{
/*	
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	t_CEDD_params * params = (t_CEDD_params *)kstub->params;
	
#ifdef MEMCPY_SYNC

	cudaEventSynchronize(kstub->end_Exec);

	enqueue_tcomamnd(tqueues, params->h_in_out[GPU_OUT_PROXY], params->out_CEDD, in_size, cudaMemcpyDeviceToHost, 0, BLOCKING, DATA, LOW, kstub);
	 
#else
	#ifdef MEMCPY_ASYNC
	printf("-->Comienzo de DtH para tarea %d\n", kstub->id);

	//enqueue_tcomamnd(tqueues, h_in_out[GPU_OUT_PROXY], out_CEDD, in_size, cudaMemcpyDeviceToHost, kstub->transfer_s[1] , NONBLOCKING, LAST_TRANSFER, MEDIUM, kstub);
	cudaMemcpyAsync(params->h_in_out[GPU_OUT_PROXY], params->out_CEDD, in_size, cudaMemcpyDeviceToHost, kstub->transfer_s[1]);

	#else
		#ifdef MANAGED_MEM
			cudaStreamSynchronize(*(kstub->execution_s)); // To be sure kernel execution has finished before processing output data
		#endif
	#endif
#endif
*/
	return 0;
}	

/*int GCEDD_end_kernel(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	
	t_CEDD_params * params = (t_CEDD_params *)kstub->params;
	
	dim3 dimGrid(kstub->kconf.gridsize.x);
    dim3 threads(kstub->kconf.blocksize.x, kstub->kconf.blocksize.y);
	
	original_sobelCannyCUDA<<<dimGrid, threads, l_mem_size>>>(
		data_CEDD, out_CEDD, theta_CEDD, rows_CEDD, cols_CEDD,
		params->gridDimY,
		
		kstub->kconf.coarsening);
		
	original_nonCannyCUDA<<<dimGrid, threads, l_mem_size>>>(
		out_CEDD, data_CEDD, theta_CEDD, rows_CEDD, cols_CEDD,
		params->gridDimY);
		
	original_hystCannyCUDA<<<dimGrid, threads>>>(
		data_CEDD, out_CEDD, rows_CEDD, cols_CEDD,
		params->gridDimY);

	cudaMemcpy(h_in_out[GPU_OUT_PROXY], out_CEDD, in_size, cudaMemcpyDeviceToHost);
	
	// int size = rows_CEDD * cols_CEDD;
	
	// loadVector(h_in_out[GPU_IN_PROXY], "output.txt", size);
	
	// check(h_in_out, arg);

	cudaFree(data_CEDD);
	cudaFree(out_CEDD);
	cudaFree(theta_CEDD);
	
	// free(h_in_out[GPU_IN_PROXY]);
	// free(h_in_out[GPU_OUT_PROXY]);
	cudaFreeHost(h_in_out[GPU_IN_PROXY]);
	cudaFreeHost(h_in_out[GPU_OUT_PROXY]);

    return 0;
}*/

int SCEDD_start_kernel(void *arg) 
{
	return 0;
}

int SCEDD_start_mallocs(void *arg)
{
	return 0;
}

int SCEDD_start_transfers(void *arg)
{
	return 0;
}

int SCEDD_end_kernel(void *arg)
{
	return 0;
}	

int NCEDD_start_kernel(void *arg) 
{
	return 0;
}

int NCEDD_start_mallocs(void *arg)
{
	return 0;
}

int NCEDD_start_transfers(void *arg)
{
	return 0;
}

int NCEDD_end_kernel(void *arg)
{
	return 0;
}	

int HCEDD_start_kernel(void *arg) 
{
	return 0;
}

int HCEDD_start_mallocs(void *arg)
{
	return 0;
}

int HCEDD_start_transfers(void *arg)
{
	return 0;
}

int HCEDD_end_kernel(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	t_CEDD_params * params = (t_CEDD_params *)kstub->params;
	
#ifdef MEMCPY_SYNC

	cudaEventSynchronize(kstub->end_Exec);

	enqueue_tcomamnd(tqueues, params->h_in_out[GPU_OUT_PROXY], params->out_CEDD, in_size, cudaMemcpyDeviceToHost, 0, BLOCKING, DATA, LOW, kstub);
	 
#else
	#ifdef MEMCPY_ASYNC
	printf("-->Comienzo de DtH para tarea %d\n", kstub->id);

	//enqueue_tcomamnd(tqueues, h_in_out[GPU_OUT_PROXY], out_CEDD, in_size, cudaMemcpyDeviceToHost, kstub->transfer_s[1] , NONBLOCKING, LAST_TRANSFER, MEDIUM, kstub);
	cudaMemcpyAsync(params->h_in_out[GPU_OUT_PROXY], params->out_CEDD, in_size, cudaMemcpyDeviceToHost, kstub->transfer_s[1]);

	#else
		#ifdef MANAGED_MEM
			cudaStreamSynchronize(*(kstub->execution_s)); // To be sure kernel execution has finished before processing output data
		#endif
	#endif
#endif

	return 0;
}	
