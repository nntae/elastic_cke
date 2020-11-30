#define _CUDA_COMPILER_

#include <math.h>
#include "support/common.h"
#include "support/partitioner.h"
#include "support/cuda-setup.h"
#include "support/verify.h"
#include "../elastic_kernel.h"

#include "RSC.h"

__device__ uint get_smid_RSC(void) {

     uint ret;

     asm("mov.u32 %0, %smid;" : "=r"(ret) );

     return ret;

}

// CUDA baseline kernel for model generation ---------------------------------------------------------------------------------
// Generate model on GPU side
__device__ int gen_model_paramGPU(int x1, int y1, int vx1, int vy1, int x2, int y2, int vx2, int vy2, float *model_param, int N, float X) {
    float temp;
    // xc -> model_param[0], yc -> model_param[1], D -> model_param[2], R -> model_param[3]
    temp = (float)((vx1 * (vx1 - (2 * vx2))) + (vx2 * vx2) + (vy1 * vy1) - (vy2 * ((2 * vy1) - vy2)));
    if(temp == 0) { // Check to prevent division by zero
        return (0);
    }
    model_param[0] = (((vx1 * ((-vx2 * x1) + (vx1 * x2) - (vx2 * x2) + (vy2 * y1) - (vy2 * y2))) +
                          (vy1 * ((-vy2 * x1) + (vy1 * x2) - (vy2 * x2) - (vx2 * y1) + (vx2 * y2))) +
                          (x1 * ((vy2 * vy2) + (vx2 * vx2)))) /
                      temp);
    model_param[1] = (((vx2 * ((vy1 * x1) - (vy1 * x2) - (vx1 * y1) + (vx2 * y1) - (vx1 * y2))) +
                          (vy2 * ((-vx1 * x1) + (vx1 * x2) - (vy1 * y1) + (vy2 * y1) - (vy1 * y2))) +
                          (y2 * ((vx1 * vx1) + (vy1 * vy1)))) /
                      temp);

    temp = (float)((x1 * (x1 - (2 * x2))) + (x2 * x2) + (y1 * (y1 - (2 * y2))) + (y2 * y2));
    if(temp == 0) { // Check to prevent division by zero
        return (0);
    }
    model_param[2] = ((((x1 - x2) * (vx1 - vx2)) + ((y1 - y2) * (vy1 - vy2))) / temp);
    model_param[3] = ((((x1 - x2) * (vy1 - vy2)) + ((y2 - y1) * (vx1 - vx2))) / temp);
	
	float a=1.2;
	for (int i=0; i<N; i++)
		a*=(float)i;
	
	model_param[0] = model_param[0] + X * a;
    return (1);
}

__global__ void original_RANSAC_kernel_block_model(int flowvector_count, int max_iter, int error_threshold, float convergence_threshold,
    int n_tasks, float alpha, float *model_param_local, flowvector *flowvectors,
    int *random_numbers, int *model_candidate, int *outliers_candidate, int *launch_gpu, int N, float X
    ) 
	
{
    extern __shared__ int l_mem[];
#ifdef CUDA_8_0
    int* outlier_block_count = l_mem;
    int* l_tmp = &outlier_block_count[1];
#endif
    
#ifdef CUDA_8_0
    Partitioner p = partitioner_create(n_tasks, alpha, worklist, l_tmp);
#else
    Partitioner p = partitioner_create(n_tasks, alpha);
#endif
    
    //const int tx         = threadIdx.x;
    //const int bx         = blockIdx.x;
    //const int num_blocks = gridDim.x;

    // Each block performs one iteration
    for(int iter = gpu_first(&p); gpu_more(&p); iter = gpu_next(&p)) {

        float *model_param =
            &model_param_local
                [4 * iter]; // xc=model_param_sh[0], yc=model_param_sh[1], D=model_param_sh[2], R=model_param_sh[3]

        // Thread (any) computes F-o-F model (SISD phase)
        // Select two random flow vectors
        int        rand_num = random_numbers[iter * 2 + 0];
        flowvector fv[2];
        fv[0]    = flowvectors[rand_num];
        rand_num = random_numbers[iter * 2 + 1];
        fv[1]    = flowvectors[rand_num];

        int ret = 0;
        int vx1 = fv[0].vx - fv[0].x;
        int vy1 = fv[0].vy - fv[0].y;
        int vx2 = fv[1].vx - fv[1].x;
        int vy2 = fv[1].vy - fv[1].y;

        // Function to generate model parameters according to F-o-F (xc, yc, D and R)
        ret = gen_model_paramGPU(fv[0].x, fv[0].y, vx1, vy1, fv[1].x, fv[1].y, vx2, vy2, model_param, N, X);
        if(ret == 0)
            model_param[0] = -2011;

        // Set launch_gpu flag
#ifdef CUDA_8_0
        atomicAdd_system(&launch_gpu[iter], 1);
#else
        atomicAdd(&launch_gpu[iter], 1);
#endif

        if(model_param[0] == -2011)
            continue;

    } 
}

__global__ void SMT_RANSAC_kernel_block_model(int flowvector_count, int max_iter, int error_threshold, float convergence_threshold,
    int n_tasks, float alpha, float *model_param_local, flowvector *flowvectors,
    int *random_numbers, int *model_candidate, int *outliers_candidate, int *launch_gpu, int N, float X,
	int griddim, 
	int SIMD_min, int SIMD_max,
	int num_subtask, int iter_per_subtask, int *cont_subtask, State *status
    ) 
{

    extern __shared__ int l_mem[];
	
	__shared__ int s_bid;
	
	unsigned int SM_id = get_smid_RSC();
	
	if (SM_id <SIMD_min || SM_id > SIMD_max) /* Only blocks executing within SIMD_min and SIMD_max can progress */ 
			return;
		
	
	while (1){
	
		// Block index
		
		
		/********** Task Id calculation *************/
		
		if (threadIdx.x==0 && threadIdx.y== 0) { 
			if (*status == TOEVICT)
				s_bid = -1;
			else {
				s_bid = atomicAdd(cont_subtask, 1);				//subtask_id
				//printf("Blq=%d cont=%d\n", blockIdx.x, s_bid);
			}
		}
		
		__syncthreads();
	
		if (s_bid >=num_subtask || s_bid ==-1) /* If all subtasks have been executed */{
			return;
		}
		
	 
	
#ifdef CUDA_8_0
		int* outlier_block_count = l_mem;
		int* l_tmp = &outlier_block_count[1];
#endif

    
#ifdef CUDA_8_0
		Partitioner p = partitioner_create(n_tasks, alpha, worklist, l_tmp);
#else
		Partitioner p = partitioner_create(n_tasks, alpha);
#endif
    
		//const int tx         = threadIdx.x;
		//const int bx         = s_bid;
		//const int num_blocks = gridDim.x;

		// Each block performs one iteration
		//for(int iter = preemp_gpu_first(&p, griddim, s_bid); gpu_more(&p); iter = preemp_gpu_next(&p, griddim, s_bid)) {

		int iter = s_bid * blockDim.x + threadIdx.x;
			
		if (iter < n_tasks) {
		
			float *model_param =
				&model_param_local
                [4 * iter]; // xc=model_param_sh[0], yc=model_param_sh[1], D=model_param_sh[2], R=model_param_sh[3]

			// Thread (any) computes F-o-F model (SISD phase)
			// Select two random flow vectors
			int        rand_num = random_numbers[iter * 2 + 0];
			flowvector fv[2];
			fv[0]    = flowvectors[rand_num];
			rand_num = random_numbers[iter * 2 + 1];
			fv[1]    = flowvectors[rand_num];

			int ret = 0;
			int vx1 = fv[0].vx - fv[0].x;
			int vy1 = fv[0].vy - fv[0].y;
			int vx2 = fv[1].vx - fv[1].x;
			int vy2 = fv[1].vy - fv[1].y;

			// Function to generate model parameters according to F-o-F (xc, yc, D and R)
			ret = gen_model_paramGPU(fv[0].x, fv[0].y, vx1, vy1, fv[1].x, fv[1].y, vx2, vy2, model_param, N, X);
			if(ret == 0)
				model_param[0] = -2011;

        // Set launch_gpu flag
#ifdef CUDA_8_0
			atomicAdd_system(&launch_gpu[iter], 1);
#else
			atomicAdd(&launch_gpu[iter], 1);
#endif

			if(model_param[0] == -2011)
				continue;
		}

		//}
	}
}

__global__ void SMK_RANSAC_kernel_block_model(int flowvector_count, int max_iter, int error_threshold, float convergence_threshold,
    int n_tasks, float alpha, float *model_param_local, flowvector *flowvectors,
    int *random_numbers, int *model_candidate, int *outliers_candidate, int *launch_gpu, int N, float X,
	int griddim,
	int max_blocks_per_SM,
	int num_subtask,
	int iter_per_subtask,
	int *cont_SM,
	int *cont_subtask,
	State *status
   ) 
	
{
    extern __shared__ int l_mem[];
	
	__shared__ int s_bid, s_index;
	
	unsigned int SM_id = get_smid_RSC();
	
	if (threadIdx.x == 0)  
		s_index = atomicAdd(&cont_SM[SM_id],1);
	
	__syncthreads();

	if (s_index > max_blocks_per_SM)
		return;
		
	while (1){
		
		/********** Task Id calculation *************/
		if (threadIdx.x == 0) {
			if (*status == TOEVICT)
				s_bid = -1;
			else
				s_bid = atomicAdd(cont_subtask, 1);
		}
		
		__syncthreads();
		
		if (s_bid >= num_subtask || s_bid == -1) /* If all subtasks have been executed */
			return;
	
	
#ifdef CUDA_8_0
		int* outlier_block_count = l_mem;
		int* l_tmp = &outlier_block_count[1];
#endif
    
#ifdef CUDA_8_0
		Partitioner p = partitioner_create(n_tasks, alpha, worklist, l_tmp);
#else
		Partitioner p = partitioner_create(n_tasks, alpha);
#endif
    
		//const int tx         = threadIdx.x;
		//const int bx         = s_bid;
		//const int num_blocks = gridDim.x;

    // Each block performs one iteration
		for(int iter = preemp_gpu_first(&p, griddim, s_bid); gpu_more(&p); iter = preemp_gpu_next(&p, griddim, s_bid)) {

			float *model_param =
				&model_param_local
					[4 * iter]; // xc=model_param_sh[0], yc=model_param_sh[1], D=model_param_sh[2], R=model_param_sh[3]

        // Thread (any) computes F-o-F model (SISD phase)
        // Select two random flow vectors
			int        rand_num = random_numbers[iter * 2 + 0];
			flowvector fv[2];
			fv[0]    = flowvectors[rand_num];
			rand_num = random_numbers[iter * 2 + 1];
			fv[1]    = flowvectors[rand_num];

			int ret = 0;
			int vx1 = fv[0].vx - fv[0].x;
			int vy1 = fv[0].vy - fv[0].y;
			int vx2 = fv[1].vx - fv[1].x;
			int vy2 = fv[1].vy - fv[1].y;

			// Function to generate model parameters according to F-o-F (xc, yc, D and R)
			ret = gen_model_paramGPU(fv[0].x, fv[0].y, vx1, vy1, fv[1].x, fv[1].y, vx2, vy2, model_param, N, X);
			if(ret == 0)
				model_param[0] = -2011;

        // Set launch_gpu flag
#ifdef CUDA_8_0
			atomicAdd_system(&launch_gpu[iter], 1);
#else
			atomicAdd(&launch_gpu[iter], 1);
#endif

			if(model_param[0] == -2011)
				continue;
		}

    }
}

// CUDA heterogeneous kernel ------------------------------------------------------------------------------------------
__global__ void original_RANSAC_kernel_block_evaluate(float *model_param_local, flowvector *flowvectors,
    int flowvector_count, int *random_numbers, int max_iter, int error_threshold, float convergence_threshold,
    int *g_out_id, int *model_candidate, int *outliers_candidate, int *launch_gpu) {

    extern __shared__ int l_mem[];
    int* outlier_block_count = l_mem;

    const int tx         = threadIdx.x;
    const int bx         = blockIdx.x;
    const int num_blocks = gridDim.x;

    float vx_error, vy_error;
    int   outlier_local_count = 0;

    // Each block performs one iteration
    for(int loop_count = bx; loop_count < max_iter; loop_count += num_blocks) {

        float *model_param =
            &model_param_local
                [4 *
                    loop_count]; // xc=model_param_sh[0], yc=model_param_sh[1], D=model_param_sh[2], R=model_param_sh[3]
        // Wait until CPU computes F-o-F model
        if(tx == 0) {
            outlier_block_count[0] = 0;
            while(atomicAdd(&launch_gpu[loop_count], 0) == 0) {
            }
        }
        __syncthreads();

        if(model_param[0] == -2011)
            continue;

        // Reset local outlier counter
        outlier_local_count = 0;

        // Compute number of outliers
        for(int i = tx; i < flowvector_count; i += blockDim.x) {
            flowvector fvreg = flowvectors[i]; // x, y, vx, vy
            vx_error         = fvreg.x + ((int)((fvreg.x - model_param[0]) * model_param[2]) -
                                     (int)((fvreg.y - model_param[1]) * model_param[3])) -
                       fvreg.vx;
            vy_error = fvreg.y + ((int)((fvreg.y - model_param[1]) * model_param[2]) +
                                     (int)((fvreg.x - model_param[0]) * model_param[3])) -
                       fvreg.vy;
            if((fabs(vx_error) >= error_threshold) || (fabs(vy_error) >= error_threshold)) {
                outlier_local_count++;
            }
        }

        atomicAdd(&outlier_block_count[0], outlier_local_count);

        __syncthreads();
        if(tx == 0) {
            // Compare to threshold
            if(outlier_block_count[0] < flowvector_count * convergence_threshold) {
                int index                 = atomicAdd(g_out_id, 1);
                model_candidate[index]    = loop_count;
                outliers_candidate[index] = outlier_block_count[0];
            }
        }
    }
}

__global__ void SMT_RANSAC_kernel_block_evaluate(float *model_param_local, flowvector *flowvectors,
    int flowvector_count, int *random_numbers, int max_iter, int error_threshold, float convergence_threshold,
    int *g_out_id, int *model_candidate, int *outliers_candidate, int *launch_gpu,
	int griddim,
	int SIMD_min, int SIMD_max,
	int num_subtask, int iter_per_subtask, int *cont_subtask, State *status
	) 
{

    extern __shared__ int l_mem[];
	
	__shared__ int s_bid;
	
	unsigned int SM_id = get_smid_RSC();
	
	if (SM_id <SIMD_min || SM_id > SIMD_max) /* Only blocks executing within SIMD_min and SIMD_max can progress */ 
			return;
		
	
	while (1){
	
		// Block index
		
		
		/********** Task Id calculation *************/
		
		if (threadIdx.x==0 && threadIdx.y== 0) { 
			if (*status == TOEVICT)
				s_bid = -1;
			else {
				s_bid = atomicAdd(cont_subtask, 1);				//subtask_id
				//printf("Blq=%d cont=%d\n", blockIdx.x, s_bid);
			}
		}
		
		__syncthreads();
	
		if (s_bid >=num_subtask || s_bid ==-1) /* If all subtasks have been executed */{
			return;
		}
		
	
		int* outlier_block_count = l_mem;

		const int tx         = threadIdx.x;
		const int bx         = s_bid;
		const int num_blocks = griddim;

		const int loop_count = s_bid;
		
		float vx_error, vy_error;
		int   outlier_local_count = 0;

		// Each block performs one iteration
		//for(int loop_count = bx; loop_count < max_iter; loop_count += num_blocks) {

			float *model_param =
				&model_param_local
                [4 *
                    loop_count]; // xc=model_param_sh[0], yc=model_param_sh[1], D=model_param_sh[2], R=model_param_sh[3]
			// Wait until CPU computes F-o-F model
			if(tx == 0) {
				outlier_block_count[0] = 0;
				while(atomicAdd(&launch_gpu[loop_count], 0) == 0) {
				}
			}
			__syncthreads();

			if(model_param[0] == -2011)
				continue;

			// Reset local outlier counter
			outlier_local_count = 0;

			// Compute number of outliers
			for(int i = tx; i < flowvector_count; i += blockDim.x) {
				flowvector fvreg = flowvectors[i]; // x, y, vx, vy
				vx_error         = fvreg.x + ((int)((fvreg.x - model_param[0]) * model_param[2]) -
                                     (int)((fvreg.y - model_param[1]) * model_param[3])) -
                       fvreg.vx;
				vy_error = fvreg.y + ((int)((fvreg.y - model_param[1]) * model_param[2]) +
                                     (int)((fvreg.x - model_param[0]) * model_param[3])) -
                       fvreg.vy;
				if((fabs(vx_error) >= error_threshold) || (fabs(vy_error) >= error_threshold)) {
					outlier_local_count++;
				}
			}

			atomicAdd(&outlier_block_count[0], outlier_local_count);

			__syncthreads();
			if(tx == 0) {
				// Compare to threshold
				if(outlier_block_count[0] < flowvector_count * convergence_threshold) {
					int index                 = atomicAdd(g_out_id, 1);
					model_candidate[index]    = loop_count;
					outliers_candidate[index] = outlier_block_count[0];
				}
			}
		//}
    }
}


__global__ void SMK_RANSAC_kernel_block_evaluate(float *model_param_local, flowvector *flowvectors,
    int flowvector_count, int *random_numbers, int max_iter, int error_threshold, float convergence_threshold,
    int *g_out_id, int *model_candidate, int *outliers_candidate, int *launch_gpu,
	int griddim,
	int max_blocks_per_SM,
	int num_subtask,
	int iter_per_subtask,
	int *cont_SM,
	int *cont_subtask,
	State *status
   ) 
	
{
    extern __shared__ int l_mem[];
	
	__shared__ int s_bid, s_index;
	
	unsigned int SM_id = get_smid_RSC();
	
	if (threadIdx.x == 0)  
		s_index = atomicAdd(&cont_SM[SM_id],1);
	
	__syncthreads();

	if (s_index > max_blocks_per_SM)
		return;
		
	while (1){
		
		/********** Task Id calculation *************/
		if (threadIdx.x == 0) {
			if (*status == TOEVICT)
				s_bid = -1;
			else
				s_bid = atomicAdd(cont_subtask, 1);
		}
		
		__syncthreads();
		
		if (s_bid >= num_subtask || s_bid == -1) /* If all subtasks have been executed */
			return;
	
		int* outlier_block_count = l_mem;

		const int tx         = threadIdx.x;
		const int bx         = s_bid;
		const int num_blocks = griddim;

		float vx_error, vy_error;
		int   outlier_local_count = 0;

		// Each block performs one iteration
		for(int loop_count = bx; loop_count < max_iter; loop_count += num_blocks) {

			float *model_param =
				&model_param_local
                [4 *
                    loop_count]; // xc=model_param_sh[0], yc=model_param_sh[1], D=model_param_sh[2], R=model_param_sh[3]
			// Wait until CPU computes F-o-F model
			if(tx == 0) {
				outlier_block_count[0] = 0;
				while(atomicAdd(&launch_gpu[loop_count], 0) == 0) {
				}
			}
			__syncthreads();

			if(model_param[0] == -2011)
				continue;

			// Reset local outlier counter
			outlier_local_count = 0;

			// Compute number of outliers
			for(int i = tx; i < flowvector_count; i += blockDim.x) {
				flowvector fvreg = flowvectors[i]; // x, y, vx, vy
				vx_error         = fvreg.x + ((int)((fvreg.x - model_param[0]) * model_param[2]) -
                                     (int)((fvreg.y - model_param[1]) * model_param[3])) -
                       fvreg.vx;
				vy_error = fvreg.y + ((int)((fvreg.y - model_param[1]) * model_param[2]) +
                                     (int)((fvreg.x - model_param[0]) * model_param[3])) -
                       fvreg.vy;
				if((fabs(vx_error) >= error_threshold) || (fabs(vy_error) >= error_threshold)) {
					outlier_local_count++;
				}
			}

			atomicAdd(&outlier_block_count[0], outlier_local_count);

			__syncthreads();
			if(tx == 0) {
				// Compare to threshold
				if(outlier_block_count[0] < flowvector_count * convergence_threshold) {
					int index                 = atomicAdd(g_out_id, 1);
					model_candidate[index]    = loop_count;
					outliers_candidate[index] = outlier_block_count[0];
				}
			}
		}
    }
}



// Input Data -----------------------------------------------------------------
int read_input_size(char *file_name) {
    FILE *File = NULL;
    File       = fopen(file_name, "r");
    if(File == NULL) {
        puts("Error al abrir el fichero");
        exit(-1);
    }

    int n;
    fscanf(File, "%d", &n);

    fclose(File);

    return n;
}

void read_input(flowvector *v, int *r, char *file_name, int max_iter) {

    int ic = 0;

    // Open input file
    FILE *File = NULL;
    File       = fopen(file_name, "r");
    if(File == NULL) {
        puts("Error opening file!");
        exit(-1);
    }

    int n;
    fscanf(File, "%d", &n);

    while(fscanf(File, "%d,%d,%d,%d", &v[ic].x, &v[ic].y, &v[ic].vx, &v[ic].vy) == 4) {
        ic++;
        if(ic > n) {
            puts("Error: inconsistent file data!");
            exit(-1);
        }
    }
    if(ic < n) {
        puts("Error: inconsistent file data!");
        exit(-1);
    }

    srand(time(NULL));
    for(int i = 0; i < 2 * max_iter; i++) {
        r[i] = ((int)rand()) % n;
    }
}

int RSC_model_start_kernel(void *arg) 
{
	cudaError_t  cudaStatus;
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	
	t_RSC_params *params= (t_RSC_params *)kstub->params;
	
	int max_iter = 20000;		
	CUDASetup    setcuda(0); // Device id
		
	int         n_flow_vectors = read_input_size("RSC/input/vectors.csv");
    int         best_model     = -1;
    int         best_outliers  = n_flow_vectors;
    flowvector *     h_flow_vector_array  = (flowvector *)malloc(n_flow_vectors * sizeof(flowvector));
    int *            h_random_numbers     = (int *)malloc(2 * max_iter * sizeof(int));
    int *            h_model_candidate    = (int *)malloc(max_iter * sizeof(int));
    int *            h_outliers_candidate = (int *)malloc(max_iter * sizeof(int));
    int h_g_out_id = 0;
    flowvector *flow_vector_array;
    cudaStatus = cudaMalloc(&flow_vector_array, n_flow_vectors * sizeof(flowvector));
    int *random_numbers;
    cudaStatus = cudaMalloc(&random_numbers, 2 * max_iter * sizeof(int));
    int *model_candidate;
    cudaStatus = cudaMalloc(&model_candidate, max_iter * sizeof(int));
    int *outliers_candidate;
    cudaStatus = cudaMalloc(&outliers_candidate, max_iter * sizeof(int));
    float *model_param_local;
    cudaStatus = cudaMalloc(&model_param_local, 4 * max_iter * sizeof(float));
    int *g_out_id;
    cudaStatus = cudaMalloc(&g_out_id, sizeof(int));
    int *launch_gpu;
    cudaStatus = cudaMalloc(&launch_gpu, (max_iter + kstub->kconf.gridsize.x) * sizeof(int));
    ALLOC_ERR(h_flow_vector_array, h_random_numbers, h_model_candidate, h_outliers_candidate);
    CUDA_ERR();
    cudaDeviceSynchronize();

    // Initialize
    const int max_gpu_threads = setcuda.max_gpu_threads();
    read_input(h_flow_vector_array, h_random_numbers, "RSC/input/vectors.csv", max_iter);
    cudaDeviceSynchronize();

    // Copy to device
    cudaStatus = cudaMemcpy(flow_vector_array, h_flow_vector_array, n_flow_vectors * sizeof(flowvector), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(random_numbers, h_random_numbers, 2 * max_iter * sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    CUDA_ERR();
	
	params->best_model = -1;
	params->h_model_candidate=h_model_candidate;
    params->h_outliers_candidate=h_outliers_candidate;
	params->outliers_candidate=outliers_candidate;
    params->h_flow_vector_array=h_flow_vector_array;
    params->h_random_numbers=h_random_numbers;
	params->h_g_out_id = h_g_out_id;
	params->best_outliers = best_outliers;
	params->flow_vector_count =n_flow_vectors;
	params->max_iter = max_iter;
	params->convergence_threshold = 0.75;
	params->n_tasks = params->max_iter;
	params->alpha = 0;
	params->model_params_local = model_param_local;
	params->flow_vectors = flow_vector_array;
	params->random_numbers = random_numbers;
	params->model_candidate = model_candidate;
	params->launch_gpu = launch_gpu;
	params->l_mem_size = sizeof(int);
	
	/* Specific parameters for evaluation kernel */
	
	params->error_threshold = 3;
	params->convergence_threshold = 0.75;
	params->g_out_id = g_out_id;
	

   cudaMemset((void *)model_candidate, 0, max_iter * sizeof(int));
   cudaMemset((void *)outliers_candidate, 0, max_iter * sizeof(int));
   cudaMemset((void *)model_param_local, 0, 4 * max_iter * sizeof(float));
   cudaMemset((void *)launch_gpu, 0, (max_iter + kstub->kconf.gridsize.x) * sizeof(int));
   cudaMemset((void *)g_out_id, 0, sizeof(int));
   
   return 0;
}


int RSC_model_end_kernel(void *arg)
{		
		return 0;
}

int RSC_evaluate_start_kernel(void *arg)
{
	return 0;
}



int RSC_evaluate_end_kernel(void *arg)
{
	cudaError_t  cudaStatus;
	
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	
	t_RSC_params *params= (t_RSC_params *)kstub->params;
	int otro;
	
    cudaStatus = cudaMemcpy(/*&(params->h_g_out_id)*/&otro, params->g_out_id, sizeof(int), cudaMemcpyDeviceToHost);
	cudaStatus = cudaMemcpy(&(params->h_g_out_id), params->g_out_id, sizeof(int), cudaMemcpyDeviceToHost);
    cudaStatus = cudaMemcpy(params->h_model_candidate, params->model_candidate, params->h_g_out_id * sizeof(int), cudaMemcpyDeviceToHost);
    cudaStatus = cudaMemcpy(params->h_outliers_candidate, params->outliers_candidate, params->h_g_out_id * sizeof(int), cudaMemcpyDeviceToHost);
    CUDA_ERR();
 
    for(int i = 0; i < params->h_g_out_id; i++) {
        if(params->h_outliers_candidate[i] < params->best_outliers) {
            params->best_outliers = params->h_outliers_candidate[i];
            params->best_model    = params->h_model_candidate[i];
        }
    }
        
	// Verify answer
    verify(params->h_flow_vector_array, params->flow_vector_count, params->h_random_numbers, params->max_iter, params->error_threshold, params->convergence_threshold,
        params->h_g_out_id, params->best_outliers);
	

    // Free memory
    free(params->h_model_candidate);
    free(params->h_outliers_candidate);
    free(params->h_flow_vector_array);
    free(params->h_random_numbers);
    cudaStatus = cudaFree(params->model_candidate);
    cudaStatus = cudaFree(params->outliers_candidate);
    cudaStatus = cudaFree(params->model_params_local);
    cudaStatus = cudaFree(params->g_out_id);
    cudaStatus = cudaFree(params->flow_vectors);
    cudaStatus = cudaFree(params->random_numbers);
    cudaStatus = cudaFree(params->launch_gpu);
	
	return 0;
}


int launch_orig_RSC_model(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	
	t_RSC_params *params = (t_RSC_params *)kstub->params;

	dim3 dimGrid(kstub->kconf.gridsize.x);
	dim3 dimBlock(kstub->kconf.blocksize.x);
	
  cudaMemset((void *)params->model_candidate, 0, params->max_iter * sizeof(int));
   cudaMemset((void *)params->outliers_candidate, 0, params->max_iter * sizeof(int));
   cudaMemset((void *)params->model_params_local, 0, 4 * params->max_iter * sizeof(float));
   cudaMemset((void *)params->launch_gpu, 0, (params->max_iter + kstub->kconf.gridsize.x) * sizeof(int));
   cudaMemset((void *)params->g_out_id, 0, sizeof(int));
    
    
	original_RANSAC_kernel_block_model<<<dimGrid, dimBlock, params->l_mem_size>>>(params->flow_vector_count, params->max_iter, params->error_threshold, 
        params->convergence_threshold, params->n_tasks, params->alpha, params->model_params_local, params->flow_vectors,
        params->random_numbers, params->model_candidate, params->outliers_candidate, params->launch_gpu, 100, 0);
    
	cudaError_t err = cudaGetLastError();
    
	return 0;
}

int launch_preemp_RSC_model(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	
	t_RSC_params *params = (t_RSC_params *)kstub->params;

	dim3 dimGrid(kstub->kconf.max_persistent_blocks*kstub->kconf.numSMs);
	dim3 dimBlock(kstub->kconf.blocksize.x);
	
	params->gridDimX = kstub->kconf.gridsize.x ; //Number of executing blocks
	kstub->total_tasks = (int)(ceil((double)params->max_iter/(float)kstub->kconf.blocksize.x));
	
	cudaMemset((void *)params->model_candidate, 0, params->max_iter * sizeof(int));
   cudaMemset((void *)params->outliers_candidate, 0, params->max_iter * sizeof(int));
   cudaMemset((void *)params->model_params_local, 0, 4 * params->max_iter * sizeof(float));
   cudaMemset((void *)params->launch_gpu, 0, (params->max_iter + kstub->kconf.gridsize.x) * sizeof(int));
   cudaMemset((void *)params->g_out_id, 0, sizeof(int));
	 
	#ifdef SMT

	SMT_RANSAC_kernel_block_model<<<dimGrid, dimBlock, params->l_mem_size, kstub->execution_s>>>(params->flow_vector_count, params->max_iter, params->error_threshold, 
        params->convergence_threshold, params->n_tasks, params->alpha, params->model_params_local, params->flow_vectors,
        params->random_numbers, params->model_candidate, params->outliers_candidate, params->launch_gpu, 100, 0,
		params->gridDimX,
		kstub->idSMs[0],
		kstub->idSMs[1],
		kstub->total_tasks,
		kstub->kconf.coarsening,
		kstub->d_executed_tasks,
		kstub->gm_state);
	#else
		SMK_RANSAC_kernel_block_model<<<dimGrid, dimBlock, params->l_mem_size, kstub->execution_s>>>(params->flow_vector_count, params->max_iter, params->error_threshold, 
        params->convergence_threshold, params->n_tasks, params->alpha, params->model_params_local, params->flow_vectors,
        params->random_numbers, params->model_candidate, params->outliers_candidate, params->launch_gpu, 100, 0,
		params->gridDimX,
		kstub->num_blocks_per_SM,
		kstub->total_tasks,
		kstub->kconf.coarsening,
		kstub->d_SMs_cont,
		kstub->d_executed_tasks,
		kstub->gm_state);
	#endif
	
    
	return 0;

}

int launch_orig_RSC_evaluate (void *arg)
{
	
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	
	t_RSC_params *params = (t_RSC_params *)kstub->params;

	dim3 dimGrid(kstub->kconf.gridsize.x);
	dim3 dimBlock(kstub->kconf.blocksize.x);
    
	original_RANSAC_kernel_block_evaluate<<<dimGrid, dimBlock, params->l_mem_size>>>(params->model_params_local, params->flow_vectors, 
	params->flow_vector_count, params->random_numbers, params->max_iter, params->error_threshold, params->convergence_threshold,
    params->g_out_id, params->model_candidate, params->outliers_candidate, params->launch_gpu
	);
	
    cudaError_t err = cudaGetLastError();
    
	return err;
}

int launch_preemp_RSC_evaluate (void *arg)
{
	
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	
	t_RSC_params *params = (t_RSC_params *)kstub->params;

	dim3 dimGrid(kstub->kconf.max_persistent_blocks*kstub->kconf.numSMs);
	dim3 dimBlock(kstub->kconf.blocksize.x);
	
	kstub->total_tasks = params->max_iter;
	
	params->gridDimX = kstub->kconf.gridsize.x ; //Number of executing blocks
	
	#ifdef SMT
    
	SMT_RANSAC_kernel_block_evaluate<<<dimGrid, dimBlock, params->l_mem_size, kstub->execution_s>>>(params->model_params_local, params->flow_vectors, 
		params->flow_vector_count, params->random_numbers, params->max_iter, params->error_threshold, params->convergence_threshold,
		params->g_out_id, params->model_candidate, params->outliers_candidate, params->launch_gpu,
		params->gridDimX,
		kstub->idSMs[0],
		kstub->idSMs[1],
		kstub->total_tasks,
		kstub->kconf.coarsening,
		kstub->d_executed_tasks,
		kstub->gm_state);	
	#else
	SMK_RANSAC_kernel_block_evaluate<<<dimGrid, dimBlock, params->l_mem_size, kstub->execution_s>>>(params->model_params_local, params->flow_vectors, 
		params->flow_vector_count, params->random_numbers, params->max_iter, params->error_threshold, params->convergence_threshold,
		params->g_out_id, params->model_candidate, params->outliers_candidate, params->launch_gpu,
		params->gridDimX,
		kstub->num_blocks_per_SM,
		kstub->total_tasks,
		kstub->kconf.coarsening,
		kstub->d_SMs_cont,
		kstub->d_executed_tasks,
		kstub->gm_state);
	#endif
	
    cudaError_t err = cudaGetLastError();
    
	return err;
}


	