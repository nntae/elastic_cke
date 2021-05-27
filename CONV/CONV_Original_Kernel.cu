// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

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
#include "CONV.h"
#include "../memaddrcnt.cuh"

extern t_tqueue *tqueues;

// float *h_Kernel, *h_Input, *h_Buffer, *h_OutputCPU, *h_OutputGPU;
// float *d_Input, *d_Output, *d_Buffer;

int imageW;
int imageH;

__constant__ float c_Kernel[KERNEL_LENGTH];

 __device__ uint get_smid_CONV(void) {
	uint ret;

	asm("mov.u32 %0, %smid;" : "=r"(ret) );

	return ret;
}

void setConvolutionKernel(float *h_Kernel)
{
    cudaMemcpyToSymbol(c_Kernel, h_Kernel, KERNEL_LENGTH * sizeof(float));
}

/**
 * Rows Convolution Separable (CUDA Kernel)
 */
__global__ void
original_rowsConvolutionCUDA(float *d_Dst_p, float *d_Src_p, int imageW, int imageH, int pitch)
{
	__shared__ float s_Data[ROWS_BLOCKDIM_Y][(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X];

	//Offset to the left halo edge
	const int baseX = (blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
	const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;	
	// const int baseX = ((blockIdx.x / gridDimY)* ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
	// const int baseY = (blockIdx.x % gridDimY) * ROWS_BLOCKDIM_Y + threadIdx.y;
	float *d_Src, *d_Dst;
	//d_Src += baseY * pitch + baseX;
	//d_Dst += baseY * pitch + baseX;
	
	d_Src = d_Src_p + baseY * pitch + baseX;
	d_Dst = d_Dst_p + baseY * pitch + baseX;
	//Load main data
	#pragma unroll

	for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
	{
		s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = d_Src[i * ROWS_BLOCKDIM_X];
	}

	//Load left halo
	#pragma unroll

	for (int i = 0; i < ROWS_HALO_STEPS; i++)
	{
		s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (baseX >= -i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
	}

	//Load right halo
	#pragma unroll

	for (int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++)
	{
		s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (imageW - baseX > i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
	}

	//Compute and store results
	__syncthreads();
	#pragma unroll

	for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
	{
		float sum = 0;

		#pragma unroll

		for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
		{
			sum += c_Kernel[KERNEL_RADIUS - j] * s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j];
		}

		d_Dst[i * ROWS_BLOCKDIM_X] = sum;
	}
}

__global__ void
//__launch_bounds__(64, 8)
slicing_rowsConvolutionCUDA(float *d_Dst_p, float *d_Src_p, int imageW, int imageH, int pitch, int coarsening, int gridDimY, int gridDimX, int init_blkIdx, int *zc_slc)
{
	__shared__ float s_Data[ROWS_BLOCKDIM_Y][(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X];

	if (threadIdx.x == 0 && threadIdx.y == 0) atomicAdd(zc_slc, 1);

	int blkIdx_x = (blockIdx.x + init_blkIdx) % gridDimX;
	int blkIdx_y = (blockIdx.x + init_blkIdx) / gridDimX;

	for (int coar = 0; coar < coarsening; coar++) {

	//Offset to the left halo edge
	const int baseX = ((coar * gridDimX + blkIdx_x) * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
	const int baseY = blkIdx_y * ROWS_BLOCKDIM_Y + threadIdx.y;	
	// const int baseX = ((blockIdx.x / gridDimY)* ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
	// const int baseY = (blockIdx.x % gridDimY) * ROWS_BLOCKDIM_Y + threadIdx.y;
	float *d_Src, *d_Dst;
	//d_Src += baseY * pitch + baseX;
	//d_Dst += baseY * pitch + baseX;

	
	
	d_Src = d_Src_p + baseY * pitch + baseX; // 
	d_Dst = d_Dst_p + baseY * pitch + baseX;
		
	#pragma unroll

	for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
	{
		s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = d_Src[i * ROWS_BLOCKDIM_X];
	}

	//Load left halo
	#pragma unroll

	for (int i = 0; i < ROWS_HALO_STEPS; i++)
	{
		s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (baseX >= -i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
	}

	//Load right halo
	#pragma unroll

	for (int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++)
	{
		s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (imageW - baseX > i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
	}

	//Compute and store results
	__syncthreads();
	#pragma unroll

	for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
	{
		float sum = 0;

		#pragma unroll

		for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
		{
			sum += c_Kernel[KERNEL_RADIUS - j] * s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j];
		}

		d_Dst[i * ROWS_BLOCKDIM_X] = sum;
	}
	}
}

__global__ void
profiling_rowsConvolutionCUDA(float *d_Dst, float *d_Src, int imageW, int imageH, int pitch,
					int gridDimY,
					
					int num_subtask,
					int iter_per_subtask,
					int *cont_SM,
					int *cont_subtask,
					State *status)
{
	 
	__shared__ int s_bid, CTA_cont;
	
	unsigned int SM_id = get_smid_CONV();
	
	if (SM_id >= 8){ /* Only blocks executing in first 8 SM  are used for profiling */ 
		//delay();
		return;
	}
	
	if (threadIdx.x == 0 && threadIdx.y== 0) {
		CTA_cont = atomicAdd(&cont_SM[SM_id], 1);
	//	if (SM_id == 7 && CTA_cont == 8)
	//		printf("Aqui\n");
	}
	
	__syncthreads();
	
	if (CTA_cont > SM_id) {/* Only one block makes computation in SM0, two blocks in SM1 and so on */
		//delay();
		return;
	}
	
	int cont_task = 0;
			
	float *o_Src = d_Src;
	float *o_Dst = d_Dst;
	
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
			if (threadIdx.x == 0 && threadIdx.y== 0)
				printf ("SM=%d CTA=%d Executed_tasks= %d \n", SM_id, CTA_cont, cont_task);	
			return;
		}
		
		if (threadIdx.x == 0 && threadIdx.y== 0) // Acumula numeor de tareas ejecutadas
			 cont_task++;
			 
		for(int iter = 0; iter < iter_per_subtask; iter++){
			d_Src = o_Src;
			d_Dst = o_Dst;
		
			__shared__ float s_Data[ROWS_BLOCKDIM_Y][(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X];

			//Offset to the left halo edge
			// const int baseX = (blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
			// const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;	
			const int baseX = ((s_bid / gridDimY)* ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
			const int baseY = (s_bid % gridDimY) * ROWS_BLOCKDIM_Y + threadIdx.y;

			d_Src += baseY * pitch + baseX;
			d_Dst += baseY * pitch + baseX;

			//Load main data
			#pragma unroll

			for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
			{
				s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = d_Src[i * ROWS_BLOCKDIM_X];
			}

			//Load left halo
			#pragma unroll

			for (int i = 0; i < ROWS_HALO_STEPS; i++)
			{
				s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (baseX >= -i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
			}

			//Load right halo
			#pragma unroll

			for (int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++)
			{
				s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (imageW - baseX > i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
			}

			//Compute and store results
			__syncthreads();
			#pragma unroll

			for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
			{
				float sum = 0;

				#pragma unroll

				for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
				{
					sum += c_Kernel[KERNEL_RADIUS - j] * s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j];
				}

				d_Dst[i * ROWS_BLOCKDIM_X] = sum;
			}
		}
	}
}

__global__ void
SMT_rowsConvolutionCUDA(float *d_Dst_p, float *d_Src_p, int imageW, int imageH, int pitch,
					int gridDimX,
					int SIMD_min, int SIMD_max,
					int num_subtask, int iter_per_subtask, int *cont_subtask, State *status)
{
	
	__shared__ float s_Data[ROWS_BLOCKDIM_Y][(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X];
	__shared__ int s_bid;
	
	unsigned int SM_id = get_smid_CONV();
	
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
		
		for(int iter = 0; iter < iter_per_subtask; iter++){
			
			float *d_Src, *d_Dst;

			//Offset to the left halo edge
			
			//const int baseX = (((s_bid + iter * blockDim.x) / gridDimY)* ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
			//const int baseY = ((s_bid + iter * blockDim.x) - ((s_bid / gridDimY) * gridDimY)) * ROWS_BLOCKDIM_Y + threadIdx.y;
			
			int row = (s_bid * iter_per_subtask + iter ) / gridDimX;
			int col = (s_bid * iter_per_subtask + iter ) - col * gridDimX;  
			
			const int baseX = (col * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
			const int baseY = row * ROWS_BLOCKDIM_Y + threadIdx.y;	

			d_Src = d_Src_p + baseY * pitch + baseX;
			d_Dst = d_Dst_p + baseY * pitch + baseX;

			//Load main data
			#pragma unroll

			for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
			{
				s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = d_Src[i * ROWS_BLOCKDIM_X];
			}

			//Load left halo
			#pragma unroll

			for (int i = 0; i < ROWS_HALO_STEPS; i++)
			{
				s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (baseX >= -i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
			}

			//Load right halo
			#pragma unroll

			for (int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++)
			{
				s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (imageW - baseX > i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
			}

			//Compute and store results
			__syncthreads();
			#pragma unroll

			for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
			{
				float sum = 0;

				#pragma unroll

				for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
				{
					sum += c_Kernel[KERNEL_RADIUS - j] * s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j];
				}

				d_Dst[i * ROWS_BLOCKDIM_X] = sum;
			}
		}
	}
}

__global__ void
memaddr_SMT_rowsConvolutionCUDA(float *d_Dst_p, float *d_Src_p, int imageW, int imageH, int pitch,
					int gridDimX,
					int *numUniqueAddr, int SIMD_min, int SIMD_max,
					int num_subtask, int iter_per_subtask, int *cont_subtask, State *status)
{
	
	__shared__ float s_Data[ROWS_BLOCKDIM_Y][(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X];
	__shared__ int s_bid;
	
	unsigned int SM_id = get_smid_CONV();
	
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
		
		for(int iter = 0; iter < iter_per_subtask; iter++){
			
			float *d_Src, *d_Dst;

			//Offset to the left halo edge
			
			//const int baseX = (((s_bid + iter * blockDim.x) / gridDimY)* ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
			//const int baseY = ((s_bid + iter * blockDim.x) - ((s_bid / gridDimY) * gridDimY)) * ROWS_BLOCKDIM_Y + threadIdx.y;
			
			int row = (s_bid * iter_per_subtask + iter ) / gridDimX;
			int col = (s_bid * iter_per_subtask + iter ) - col * gridDimX;  
			
			const int baseX = (col * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
			const int baseY = row * ROWS_BLOCKDIM_Y + threadIdx.y;	

			d_Src = d_Src_p + baseY * pitch + baseX;
			d_Dst = d_Dst_p + baseY * pitch + baseX;

			//Load main data
			#pragma unroll

			for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
			{
#if defined(COUNT_ALL_TASKS)
				if ( s_bid == 0 )
#endif
				{
					get_unique_lines((intptr_t) &d_Src[i * ROWS_BLOCKDIM_X], numUniqueAddr);
					get_conflicting_banks( (intptr_t) &s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X], &numUniqueAddr[1] );				
				}
				s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = d_Src[i * ROWS_BLOCKDIM_X];
			}

			//Load left halo
			#pragma unroll

			for (int i = 0; i < ROWS_HALO_STEPS; i++)
			{
				// Cambiamos la expresion condicional para usar if
//				s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (baseX >= -i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
				if ( baseX >= -i * ROWS_BLOCKDIM_X )
				{
#if defined(COUNT_ALL_TASKS)
				if ( s_bid == 0 )
#endif
					{
						get_unique_lines((intptr_t) &d_Src[i * ROWS_BLOCKDIM_X], numUniqueAddr);
						get_conflicting_banks( (intptr_t) &s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X], &numUniqueAddr[1] );
					}				
					s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = d_Src[i * ROWS_BLOCKDIM_X];
				}
				else
				{
#if defined(COUNT_ALL_TASKS)
				if ( s_bid == 0 )
#endif
					{
						get_conflicting_banks( (intptr_t) &s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X], &numUniqueAddr[1] );
					}				
					s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = 0;
				}
			}

			//Load right halo
			#pragma unroll

			for (int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++)
			{
//				s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (imageW - baseX > i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
				if (imageW - baseX > i * ROWS_BLOCKDIM_X)
				{
#if defined(COUNT_ALL_TASKS)
				if ( s_bid == 0 )
#endif
					{
						get_unique_lines((intptr_t) &d_Src[i * ROWS_BLOCKDIM_X], numUniqueAddr);
						get_conflicting_banks( (intptr_t) &s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X], &numUniqueAddr[1] );
					}				
					s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = d_Src[i * ROWS_BLOCKDIM_X];
				}
				else
				{
#if defined(COUNT_ALL_TASKS)
				if ( s_bid == 0 )
#endif
					{
						get_conflicting_banks( (intptr_t) &s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X], &numUniqueAddr[1] );
					}				
					s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = 0;
				}
			}

			//Compute and store results
			__syncthreads();
			#pragma unroll

			for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
			{
				float sum = 0;

				#pragma unroll

				for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
				{
#if defined(COUNT_ALL_TASKS)
				if ( s_bid == 0 )
#endif
					{
						get_conflicting_banks( (intptr_t) &s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j], &numUniqueAddr[1] );
					}
					sum += c_Kernel[KERNEL_RADIUS - j] * s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j];
				}

#if defined(COUNT_ALL_TASKS)
				if ( s_bid == 0 )
#endif
				{
					get_unique_lines((intptr_t) &d_Dst[i * ROWS_BLOCKDIM_X], numUniqueAddr);
				}				
				d_Dst[i * ROWS_BLOCKDIM_X] = sum;
			}
		}
	}
}

__global__ void
SMK_rowsConvolutionCUDA(float *d_Dst, float *d_Src, int imageW, int imageH, int pitch,
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
	
	unsigned int SM_id = get_smid_CONV();
	
	if (threadIdx.x == 0 && threadIdx.y == 0)  
		s_index = atomicAdd(&cont_SM[SM_id],1);
	
	__syncthreads();

	if (s_index > max_blocks_per_SM)
		return;
		
	float *o_Src = d_Src;
	float *o_Dst = d_Dst;
	
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
			
		for(int iter = 0; iter < iter_per_subtask; iter++){
			d_Src = o_Src;
			d_Dst = o_Dst;
		
			__shared__ float s_Data[ROWS_BLOCKDIM_Y][(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X];

			//Offset to the left halo edge
			// const int baseX = (blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
			// const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;	
			const int baseX = ((s_bid / gridDimY)* ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
			const int baseY = (s_bid % gridDimY) * ROWS_BLOCKDIM_Y + threadIdx.y;

			d_Src += baseY * pitch + baseX;
			d_Dst += baseY * pitch + baseX;

			//Load main data
			#pragma unroll

			for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
			{
				s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = d_Src[i * ROWS_BLOCKDIM_X];
			}

			//Load left halo
			#pragma unroll

			for (int i = 0; i < ROWS_HALO_STEPS; i++)
			{
				s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (baseX >= -i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
			}

			//Load right halo
			#pragma unroll

			for (int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++)
			{
				s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (imageW - baseX > i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
			}

			//Compute and store results
			__syncthreads();
			#pragma unroll

			for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
			{
				float sum = 0;

				#pragma unroll

				for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
				{
					sum += c_Kernel[KERNEL_RADIUS - j] * s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j];
				}

				d_Dst[i * ROWS_BLOCKDIM_X] = sum;
			}
		}
	}
}

int launch_orig_RCONV(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	
	t_CONV_params * params = (t_CONV_params *)kstub->params;
	
	dim3 blocks(kstub->kconf.gridsize.x, kstub->kconf.gridsize.y);
    dim3 threads(kstub->kconf.blocksize.x, kstub->kconf.blocksize.y);
	
	original_rowsConvolutionCUDA<<<blocks, threads>>>(
		params->d_Buffer, params->d_Input, imageW, imageH, imageW);

	return 0;
}

int launch_slc_RCONV(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	
	t_CONV_params * params = (t_CONV_params *)kstub->params;
	
	dim3 blocks(kstub->kconf.gridsize.x, kstub->kconf.gridsize.y);
    dim3 threads(kstub->kconf.blocksize.x, kstub->kconf.blocksize.y);
	
	slicing_rowsConvolutionCUDA<<<kstub->total_tasks, threads,  0, *(kstub->execution_s)>>>(
		params->d_Buffer, params->d_Input, imageW, imageH, imageW, kstub->kconf.coarsening, params->gridDimY[0], params->gridDimX[0], kstub->kconf.initial_blockID, params->zc_slc);

	return 0;
}

int prof_RCONV(void *arg)
{
	
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	
	t_CONV_params * params = (t_CONV_params *)kstub->params;
	
	//dim3 blocks(kstub->kconf.gridsize.x);
    dim3 threads(kstub->kconf.blocksize.x, kstub->kconf.blocksize.y);

	profiling_rowsConvolutionCUDA<<< kstub->kconf.numSMs * kstub->kconf.max_persistent_blocks, threads, 0, *(kstub->execution_s) >>>(
			params->d_Buffer, params->d_Input, imageW, imageH, imageW,
			kstub->kconf.gridsize.x,
			kstub->total_tasks,
			kstub->kconf.coarsening,
			kstub->d_SMs_cont,
			kstub->d_executed_tasks,
			&kstub->gm_state[kstub->stream_index]);

	return 0;
}

int launch_preemp_RCONV(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	
	t_CONV_params * params = (t_CONV_params *)kstub->params;
	
	//dim3 blocks(kstub->kconf.gridsize.x);
    dim3 threads(kstub->kconf.blocksize.x, kstub->kconf.blocksize.y);
	
	#ifdef SMT
	if ( !(kstub->memaddr_profile) )	
		SMT_rowsConvolutionCUDA<<< kstub->kconf.numSMs * kstub->kconf.max_persistent_blocks, threads, 0, *(kstub->execution_s) >>>(
			params->d_Buffer, params->d_Input, imageW, imageH, imageW,
			
			kstub->kconf.gridsize.y,
			
			kstub->idSMs[0],
			kstub->idSMs[1],
			kstub->total_tasks,
			kstub->kconf.coarsening,
			kstub->d_executed_tasks,
			kstub->gm_state);
	else
		memaddr_SMT_rowsConvolutionCUDA<<< kstub->kconf.numSMs * kstub->kconf.max_persistent_blocks, threads, 0, *(kstub->execution_s) >>>(
			params->d_Buffer, params->d_Input, imageW, imageH, imageW,
			
			kstub->kconf.gridsize.y,
			
			kstub->d_numUniqueAddr,				
			kstub->idSMs[0],
			kstub->idSMs[1],
			kstub->total_tasks,
			kstub->kconf.coarsening,
			kstub->d_executed_tasks,
			kstub->gm_state);
				
	#else
		SMK_rowsConvolutionCUDA<<< kstub->kconf.numSMs * kstub->kconf.max_persistent_blocks, threads, 0, *(kstub->execution_s) >>>(
			params->d_Buffer, params->d_Input, imageW, imageH, imageW,
			
			kstub->kconf.gridsize.y,
			
			kstub->num_blocks_per_SM,
			kstub->total_tasks,
			kstub->kconf.coarsening,
			kstub->d_SMs_cont,
			kstub->d_executed_tasks,
			kstub->gm_state);
	#endif		
	
	return 0;
}

/**
 * Cols Convolution Separable (CUDA Kernel)
 */
__global__ void
original_colsConvolutionCUDA(float *d_Dst, float *d_Src, int imageW, int imageH, int pitch)
{
	__shared__ float s_Data[COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];

	//Offset to the upper halo edge
	const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
	const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
	// const int baseX = (blockIdx.x / gridDimY) * COLUMNS_BLOCKDIM_X + threadIdx.x;
	// const int baseY = ((blockIdx.x % gridDimY) * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
	d_Src += baseY * pitch + baseX;
	d_Dst += baseY * pitch + baseX;

	//Main data
	#pragma unroll

	for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
	{
		s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = d_Src[i * COLUMNS_BLOCKDIM_Y * pitch];
	}

	//Upper halo
	#pragma unroll

	for (int i = 0; i < COLUMNS_HALO_STEPS; i++)
	{
		s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
	}

	//Lower halo
	#pragma unroll

	for (int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++)
	{
		s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y]= (imageH - baseY > i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
	}

	//Compute and store results
	__syncthreads();
	#pragma unroll

	for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
	{
		float sum = 0;
		#pragma unroll

		for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
		{
			sum += c_Kernel[KERNEL_RADIUS - j] * s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j];
		}

		d_Dst[i * COLUMNS_BLOCKDIM_Y * pitch] = sum;
	}
}

__global__ void
__launch_bounds__(128, 16)
slicing_colsConvolutionCUDA(float *d_Dst_start, float *d_Src_start, int imageW, int imageH, int pitch, int coarsening, int gridDimY, int gridDimX, int init_blkIdx, int *zc_slc)
{
	__shared__ float s_Data[COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];

	if (threadIdx.x == 0 && threadIdx.y == 0) atomicAdd(zc_slc, 1);
	
	//Offset to the upper halo edge
	//int blkIdx_x = (blockIdx.x + init_blkIdx) % gridDimX;
	//int blkIdx_y = (blockIdx.x + init_blkIdx) / gridDimX;
	float *d_Dst, *d_Src;

	for (int coar=0; coar < coarsening; coar++) {
			
		int blkIdx_x = ((blockIdx.x  + init_blkIdx) * coarsening + coar) % gridDimX;
		int blkIdx_y = ((blockIdx.x  + init_blkIdx) * coarsening + coar) / gridDimX;

		const int baseX = blkIdx_x  * COLUMNS_BLOCKDIM_X + threadIdx.x;
		const int baseY = (blkIdx_y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;

		d_Src = d_Src_start + baseY * pitch + baseX ;
		d_Dst = d_Dst_start + baseY * pitch + baseX ;

		//if (threadIdx.x==0 && threadIdx.y==0) 
		//	printf("blkIdx=%d blkIdxy=%d c=%d pitch=%d coar=%d bx =%d by=%d off_scr=%d off_dst=%d\n", blkIdx_x, blkIdx_y, coarsening, pitch, coar, baseX, baseY, baseY * pitch + baseX, baseY * pitch + baseX);

		//Main data
		#pragma unroll
	
		for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
		{
			s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] ;
		}
	
		//Upper halo
		#pragma unroll
	
		for (int i = 0; i < COLUMNS_HALO_STEPS; i++)
		{
			s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
		}
	
		//Lower halo
		#pragma unroll
	
		for (int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++)
		{
			s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y]= (imageH - baseY > i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
		}

	
		//Compute and store results
		__syncthreads();
		#pragma unroll
	
		for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
		{
			float sum = 0;
			#pragma unroll
	
			for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
			{
				sum += c_Kernel[KERNEL_RADIUS - j] * s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j];
			}
	
			d_Dst[i * COLUMNS_BLOCKDIM_Y * pitch] = sum;
		}
	}		
}

__global__ void
SMT_colsConvolutionCUDA(float *d_Dst_p, float *d_Src_p, int imageW, int imageH, int pitch,
					int gridDimX,
					int SIMD_min, int SIMD_max,
					int num_subtask, int iter_per_subtask, int *cont_subtask, State *status)
{
	__shared__ float s_Data[COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];
	__shared__ int s_bid;
	
	unsigned int SM_id = get_smid_CONV();
	
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
		
		for(int iter = 0; iter < iter_per_subtask; iter++){
			float *d_Src, *d_Dst;
			
			int row = (s_bid * iter_per_subtask + iter ) / gridDimX;
			int col = (s_bid * iter_per_subtask + iter ) - col * gridDimX;  
			
			const int baseX = col * COLUMNS_BLOCKDIM_X + threadIdx.x;
			const int baseY = (row * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;	

			//Offset to the upper halo edge
			// const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
			// const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
			//const int baseX = ((s_bid + iter * blockDim.x) / gridDimY) * COLUMNS_BLOCKDIM_X + threadIdx.x;
			//const int baseY = (((s_bid + iter * blockDim.x) - ((s_bid / gridDimY) * gridDimY)) * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
			
			d_Src = d_Src_p + baseY * pitch + baseX;
			d_Dst = d_Dst_p + baseY * pitch + baseX;

			//Main data
			#pragma unroll

			for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
			{
				s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = d_Src[i * COLUMNS_BLOCKDIM_Y * pitch];
			}

			//Upper halo
			#pragma unroll

			for (int i = 0; i < COLUMNS_HALO_STEPS; i++)
			{
				s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
			}

			//Lower halo
			#pragma unroll

			for (int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++)
			{
				s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y]= (imageH - baseY > i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
			}

			//Compute and store results
			__syncthreads();
			#pragma unroll

			for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
			{
				float sum = 0;
				#pragma unroll

				for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
				{
					sum += c_Kernel[KERNEL_RADIUS - j] * s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j];
				}

				d_Dst[i * COLUMNS_BLOCKDIM_Y * pitch] = sum;
			}
		}
	}
}

__global__ void
memaddr_SMT_colsConvolutionCUDA(float *d_Dst_p, float *d_Src_p, int imageW, int imageH, int pitch,
					int gridDimX,
					int *numUniqueAddr, int SIMD_min, int SIMD_max,
					int num_subtask, int iter_per_subtask, int *cont_subtask, State *status)
{
	__shared__ float s_Data[COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];
	__shared__ int s_bid;
	
	unsigned int SM_id = get_smid_CONV();
	
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
		
		for(int iter = 0; iter < iter_per_subtask; iter++){
			float *d_Src, *d_Dst;
			
			int row = (s_bid * iter_per_subtask + iter ) / gridDimX;
			int col = (s_bid * iter_per_subtask + iter ) - col * gridDimX;  
			
			const int baseX = col * COLUMNS_BLOCKDIM_X + threadIdx.x;
			const int baseY = (row * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;	

			//Offset to the upper halo edge
			// const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
			// const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
			//const int baseX = ((s_bid + iter * blockDim.x) / gridDimY) * COLUMNS_BLOCKDIM_X + threadIdx.x;
			//const int baseY = (((s_bid + iter * blockDim.x) - ((s_bid / gridDimY) * gridDimY)) * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
			
			d_Src = d_Src_p + baseY * pitch + baseX;
			d_Dst = d_Dst_p + baseY * pitch + baseX;

			//Main data
			#pragma unroll

			for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
			{
#if defined(COUNT_ALL_TASKS)
				if ( s_bid == 0 )
#endif
				{
					get_unique_lines((intptr_t) &d_Src[i * COLUMNS_BLOCKDIM_Y * pitch], numUniqueAddr);
					get_conflicting_banks( (intptr_t) &s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y], &numUniqueAddr[1] );			
				}
				s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = d_Src[i * COLUMNS_BLOCKDIM_Y * pitch];
			}

			//Upper halo
			#pragma unroll

			for (int i = 0; i < COLUMNS_HALO_STEPS; i++)
			{
				// s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
				if (baseY >= -i * COLUMNS_BLOCKDIM_Y)
				{
#if defined(COUNT_ALL_TASKS)
				if ( s_bid == 0 )
#endif
					{
						get_unique_lines((intptr_t) &d_Src[i * COLUMNS_BLOCKDIM_Y * pitch], numUniqueAddr);
						get_conflicting_banks( (intptr_t) &s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y], &numUniqueAddr[1] );
					}				
					s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = d_Src[i * COLUMNS_BLOCKDIM_Y * pitch];
				}
				else
				{
#if defined(COUNT_ALL_TASKS)
				if ( s_bid == 0 )
#endif
					{
						get_conflicting_banks( (intptr_t) &s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y], &numUniqueAddr[1] );
					}	
					s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = 0;		
				}
			}

			//Lower halo
			#pragma unroll

			for (int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++)
			{
				//s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y]= (imageH - baseY > i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
				if (imageH - baseY > i * COLUMNS_BLOCKDIM_Y)
				{
#if defined(COUNT_ALL_TASKS)
				if ( s_bid == 0 )
#endif
					{
						get_unique_lines((intptr_t) &d_Src[i * COLUMNS_BLOCKDIM_Y * pitch], numUniqueAddr);
						get_conflicting_banks( (intptr_t) &s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y], &numUniqueAddr[1] );
					}				
					s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = d_Src[i * COLUMNS_BLOCKDIM_Y * pitch];
				}
				else
				{
#if defined(COUNT_ALL_TASKS)
				if ( s_bid == 0 )
#endif
					{
						get_conflicting_banks( (intptr_t) &s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y], &numUniqueAddr[1] );
					}
					s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = 0;					}
			}

			//Compute and store results
			__syncthreads();
			#pragma unroll

			for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
			{
				float sum = 0;
				#pragma unroll

				for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
				{
#if defined(COUNT_ALL_TASKS)
				if ( s_bid == 0 )
#endif
					{
						get_conflicting_banks( (intptr_t) &s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j], &numUniqueAddr[1] );
					}
					sum += c_Kernel[KERNEL_RADIUS - j] * s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j];
				}
				
#if defined(COUNT_ALL_TASKS)
				if ( s_bid == 0 )
#endif
				{
					get_unique_lines((intptr_t) &d_Dst[i * COLUMNS_BLOCKDIM_Y * pitch], numUniqueAddr);
				}	
				d_Dst[i * COLUMNS_BLOCKDIM_Y * pitch] = sum;
			}
		}
	}
}


__global__ void
SMK_colsConvolutionCUDA(float *d_Dst, float *d_Src, int imageW, int imageH, int pitch,
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
	
	unsigned int SM_id = get_smid_CONV();
	
	if (threadIdx.x == 0 && threadIdx.y == 0)  
		s_index = atomicAdd(&cont_SM[SM_id],1);
	
	__syncthreads();

	if (s_index > max_blocks_per_SM)
		return;
		
	float *o_Src = d_Src;
	float *o_Dst = d_Dst;
	
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
			
		for(int iter = 0; iter < iter_per_subtask; iter++){
			d_Src = o_Src;
			d_Dst = o_Dst;
		
			__shared__ float s_Data[COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];

			//Offset to the upper halo edge
			// const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
			// const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
			const int baseX = (s_bid / gridDimY) * COLUMNS_BLOCKDIM_X + threadIdx.x;
			const int baseY = ((s_bid % gridDimY) * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
			d_Src += baseY * pitch + baseX;
			d_Dst += baseY * pitch + baseX;

			//Main data
			#pragma unroll

			for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
			{
				s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = d_Src[i * COLUMNS_BLOCKDIM_Y * pitch];
			}

			//Upper halo
			#pragma unroll

			for (int i = 0; i < COLUMNS_HALO_STEPS; i++)
			{
				s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
			}

			//Lower halo
			#pragma unroll

			for (int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++)
			{
				s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y]= (imageH - baseY > i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
			}

			//Compute and store results
			__syncthreads();
			#pragma unroll

			for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
			{
				float sum = 0;
				#pragma unroll

				for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
				{
					sum += c_Kernel[KERNEL_RADIUS - j] * s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j];
				}

				d_Dst[i * COLUMNS_BLOCKDIM_Y * pitch] = sum;
			}
		}
	}
}

int launch_orig_CCONV(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	
	t_CONV_params * params = (t_CONV_params *)kstub->params;
	
	dim3 blocks(kstub->kconf.gridsize.x, kstub->kconf.gridsize.y);
    dim3 threads(kstub->kconf.blocksize.x, kstub->kconf.blocksize.y);
	
	original_colsConvolutionCUDA<<<blocks, threads>>>(
		params->d_Output, params->d_Buffer, imageW, imageH, imageW);

	return 0;
}


int launch_slc_CCONV(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	
	t_CONV_params * params = (t_CONV_params *)kstub->params;
	
	dim3 blocks(kstub->kconf.gridsize.x, kstub->kconf.gridsize.y);
    dim3 threads(kstub->kconf.blocksize.x, kstub->kconf.blocksize.y);
	
	slicing_colsConvolutionCUDA<<<kstub->total_tasks, threads, 0, *(kstub->execution_s)>>>(
		params->d_Output, params->d_Buffer, imageW, imageH, imageW,  kstub->kconf.coarsening, params->gridDimY[1], params->gridDimX[1], kstub->kconf.initial_blockID, params->zc_slc);

	return 0;
}

int launch_preemp_CCONV(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	
	t_CONV_params * params = (t_CONV_params *)kstub->params;
	
	//dim3 blocks(kstub->kconf.gridsize.x);
    dim3 threads(kstub->kconf.blocksize.x, kstub->kconf.blocksize.y);
	
	#ifdef SMT
	if ( !(kstub->memaddr_profile) )	
		SMT_colsConvolutionCUDA<<< kstub->kconf.numSMs * kstub->kconf.max_persistent_blocks, threads, 0, *(kstub->execution_s) >>>(
			params->d_Output, params->d_Buffer, imageW, imageH, imageW,
			
			kstub->kconf.gridsize.x,
			
			kstub->idSMs[0],
			kstub->idSMs[1],
			kstub->total_tasks,
			kstub->kconf.coarsening,
			kstub->d_executed_tasks,
			&(kstub->gm_state[kstub->stream_index]));
	else
		memaddr_SMT_colsConvolutionCUDA<<< kstub->kconf.numSMs * kstub->kconf.max_persistent_blocks, threads, 0, *(kstub->execution_s) >>>(
			params->d_Output, params->d_Buffer, imageW, imageH, imageW,
			
			kstub->kconf.gridsize.x,

			kstub->d_numUniqueAddr,				
			kstub->idSMs[0],
			kstub->idSMs[1],
			kstub->total_tasks,
			kstub->kconf.coarsening,
			kstub->d_executed_tasks,
			&(kstub->gm_state[kstub->stream_index]));
	#else
		SMK_colsConvolutionCUDA<<< kstub->kconf.numSMs * kstub->kconf.max_persistent_blocks, threads, 0, *(kstub->execution_s) >>>(
			params->d_Output, params->d_Buffer, imageW, imageH, imageW,
			
			kstub->kconf.gridsize.x,
			
			kstub->num_blocks_per_SM,
			kstub->total_tasks,
			kstub->kconf.coarsening,
			kstub->d_SMs_cont,
			kstub->d_executed_tasks,
			);
	#endif
	
	return 0;
}

int RCONV_start_kernel(void *arg) 
{	
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	
	t_CONV_params * params = (t_CONV_params *)kstub->params;
	
	imageH = params->conv_rows;
	imageW = params->conv_cols;
	
	// h_Kernel    = (float *)malloc(KERNEL_LENGTH * sizeof(float));
    // h_Input     = (float *)malloc(imageW * imageH * sizeof(float));
    // h_Buffer    = (float *)malloc(imageW * imageH * sizeof(float));
    // h_OutputCPU = (float *)malloc(imageW * imageH * sizeof(float));
    // h_OutputGPU = (float *)malloc(imageW * imageH * sizeof(float));
	
	cudaMallocHost(&params->h_Kernel, KERNEL_LENGTH * sizeof(float));
	cudaMallocHost(&params->h_Input, imageW * imageH * sizeof(float));
	cudaMallocHost(&params->h_Buffer, imageW * imageH * sizeof(float));
	cudaMallocHost(&params->h_OutputCPU, imageW * imageH * sizeof(float));
	cudaMallocHost(&params->h_OutputGPU, imageW * imageH * sizeof(float));
    srand(200);
	
	for (unsigned int i = 0; i < KERNEL_LENGTH; i++)
    {
        params->h_Kernel[i] = (float)(rand() % 16);
    }

    for (unsigned i = 0; i < imageW * imageH; i++)
    {
        params->h_Input[i] = (float)(rand() % 16);
    }
	
	checkCudaErrors(cudaMalloc((void **)&params->d_Input,   imageW * imageH * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&params->d_Output,  imageW * imageH * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&params->d_Buffer , imageW * imageH * sizeof(float)));
	
	setConvolutionKernel(params->h_Kernel);
    checkCudaErrors(cudaMemcpy(params->d_Input, params->h_Input, imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));
	
	return 0;
}

int RCONV_start_mallocs(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	
	t_CONV_params * params = (t_CONV_params *)kstub->params;
	params->conv_cols *= kstub->kconf.coarsening;
	kstub->kconf.gridsize.x = params->conv_cols / (8 * kstub->kconf.blocksize.x * kstub->kconf.coarsening );
	kstub->kconf.gridsize.y = params->conv_rows / kstub->kconf.blocksize.y;
	kstub->total_tasks = (kstub->kconf.gridsize.x * kstub->kconf.gridsize.y)/kstub->kconf.coarsening;
	params->gridDimX[0] = kstub->kconf.gridsize.x;
	params->gridDimY[0] = kstub->kconf.gridsize.y;

	// globalmemory position for launched ctas counter
	cudaMalloc((void **)&params->zc_slc, sizeof(int));
	
	imageH = params->conv_rows;
	imageW = params->conv_cols;
	
#if defined(MEMCPY_SYNC) || defined(MEMCPY_ASYNC)
	cudaMallocHost(&params->h_Kernel, KERNEL_LENGTH * sizeof(float));
	cudaMallocHost(&params->h_Input, imageW * imageH * sizeof(float));
	cudaMallocHost(&params->h_Buffer, imageW * imageH * sizeof(float));
	cudaMallocHost(&params->h_OutputCPU, imageW * imageH * sizeof(float));
	cudaMallocHost(&params->h_OutputGPU, imageW * imageH * sizeof(float));

	checkCudaErrors(cudaMalloc((void **)&params->d_Input,   imageW * imageH * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&params->d_Output,  imageW * imageH * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&params->d_Buffer , imageW * imageH * sizeof(float)));
#else
	#ifdef MANAGED_MEM

	cudaMallocManaged(&params->h_Kernel, KERNEL_LENGTH * sizeof(float));
	cudaMallocManaged(&params->h_Input, imageW * imageH * sizeof(float));
	cudaMallocManaged(&params->h_Buffer, imageW * imageH * sizeof(float));
	cudaMallocManaged(&params->h_OutputCPU, imageW * imageH * sizeof(float));
	cudaMallocManaged(&params->h_OutputGPU, imageW * imageH * sizeof(float));
	
	d_Input = h_Input;
	#else
		printf("No transfer model: Exiting ...\n");
		exit(-1);
	#endif
#endif

	// Verify that allocations succeeded
    if (params->h_Kernel == NULL || params->h_Input == NULL || params->h_Buffer == NULL || params->h_OutputCPU == NULL || params->h_OutputGPU == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    srand(200);
	
	for (unsigned int i = 0; i < KERNEL_LENGTH; i++)
    {
        params->h_Kernel[i] = (float)(rand() % 16);
    }

    for (unsigned i = 0; i < imageW * imageH; i++)
    {
        params->h_Input[i] = (float)(rand() % 16);
    }
	
	setConvolutionKernel(params->h_Kernel);

	return 0;
}

int RCONV_start_transfers(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	
	t_CONV_params * params = (t_CONV_params *)kstub->params;
	
	imageH = params->conv_rows;
	imageW = params->conv_cols;
	
#ifdef MEMCPY_SYNC
	enqueue_tcomamnd(tqueues, params->d_Input, params->h_Input, imageW * imageH * sizeof(float), cudaMemcpyHostToDevice, 0, BLOCKING, DATA, LOW, kstub);
	
	kstub->HtD_tranfers_finished = 1;

	
#else
	
	#ifdef MEMCPY_ASYNC
	
	//enqueue_tcomamnd(tqueues, d_Input, h_Input, imageW * imageH * sizeof(float), cudaMemcpyHostToDevice, 0, NONBLOCKING, LAST_TRANSFER, MEDIUM, kstub);
	cudaMemcpyAsync(params->d_Input, params->h_Input, imageW * imageH * sizeof(float), cudaMemcpyHostToDevice, kstub->transfer_s[0]);
	#else
	#ifdef MANAGED_MEM

	cudaDeviceProp p;
    cudaGetDeviceProperties(&p, kstub->deviceId);
	
	if (p.concurrentManagedAccess)
	{
		err = cudaMemPrefetchAsync(params->h_Kernel, KERNEL_LENGTH * sizeof(float), kstub->deviceId);
		if ( err != cudaSuccess) {
			printf("Error in vAdd:cudaMemPrefetchAsync\n");
			exit(EXIT_FAILURE);
		}
		err = cudaMemPrefetchAsync(params->h_Input, imageW * imageH * sizeof(float), kstub->deviceId);
		if ( err != cudaSuccess) {
			printf("Error in vAdd:cudaMemPrefetchAsync\n");
			exit(EXIT_FAILURE);
		}
		err = cudaMemPrefetchAsync(params->h_Buffer, imageW * imageH * sizeof(float), kstub->deviceId);
		if ( err != cudaSuccess) {
			printf("Error in vAdd:cudaMemPrefetchAsync\n");
			exit(EXIT_FAILURE);
		}
		err = cudaMemPrefetchAsync(params->h_OutputCPU, imageW * imageH * sizeof(float), kstub->deviceId);
		if ( err != cudaSuccess) {
			printf("Error in vAdd:cudaMemPrefetchAsync\n");
			exit(EXIT_FAILURE);
		}
		err = cudaMemPrefetchAsync(params->h_OutputGPU, imageW * imageH * sizeof(float), kstub->deviceId);
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

int RCONV_end_kernel(void *arg)
{
	/*
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	t_CONV_params * params = (t_CONV_params *)kstub->params;
	
#ifdef MEMCPY_SYNC

	cudaEventSynchronize(kstub->end_Exec);

	enqueue_tcomamnd(tqueues, params->h_OutputGPU, params->d_Output, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost, 0, BLOCKING, DATA, LOW, kstub);
	 
#else
	#ifdef MEMCPY_ASYNC
	printf("-->Comienzo de DtH para tarea %d\n", kstub->id);

	//enqueue_tcomamnd(tqueues, h_OutputGPU, d_Output, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost, kstub->transfer_s[1] , NONBLOCKING, LAST_TRANSFER, MEDIUM, kstub);
	cudaMemcpyAsync(params->h_OutputGPU, params->d_Output, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost, kstub->transfer_s[1]);
	#else
		#ifdef MANAGED_MEM
			cudaStreamSynchronize(*(kstub->execution_s)); // To be sure kernel execution has finished before processing output data
		#endif
	#endif
#endif
*/
	return 0;
}	

/*int RCONV_end_kernel(void *arg)
{


    checkCudaErrors(cudaFree(d_Buffer));
    checkCudaErrors(cudaFree(d_Output));
    checkCudaErrors(cudaFree(d_Input));
    // free(h_OutputGPU);
    // free(h_OutputCPU);
    // free(h_Buffer);
    // free(h_Input);
    // free(h_Kernel);
	
	cudaFreeHost(h_OutputGPU);
	cudaFreeHost(h_OutputCPU);
    cudaFreeHost(h_Buffer);
    cudaFreeHost(h_Input);
    cudaFreeHost(h_Kernel);

    return 0;
}*/

int CCONV_start_kernel(void *arg) 
{	
	return 0;
}

int CCONV_start_mallocs(void *arg)
{
	return 0;
}

int CCONV_start_transfers(void *arg)
{
	return 0;
}

int CCONV_end_kernel(void *arg)
{
	
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	t_CONV_params * params = (t_CONV_params *)kstub->params;
	
#ifdef MEMCPY_SYNC

	cudaEventSynchronize(kstub->end_Exec);

	enqueue_tcomamnd(tqueues, params->h_OutputGPU, params->d_Output, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost, 0, BLOCKING, DATA, LOW, kstub);
	 
#else
	#ifdef MEMCPY_ASYNC
	printf("-->Comienzo de DtH para tarea %d\n", kstub->id);

	//enqueue_tcomamnd(tqueues, h_OutputGPU, d_Output, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost, kstub->transfer_s[1] , NONBLOCKING, LAST_TRANSFER, MEDIUM, kstub);
	cudaMemcpyAsync(params->h_OutputGPU, params->d_Output, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost, kstub->transfer_s[1]);
	#else
		#ifdef MANAGED_MEM
			cudaStreamSynchronize(*(kstub->execution_s)); // To be sure kernel execution has finished before processing output data
		#endif
	#endif
#endif

	return 0;
}	