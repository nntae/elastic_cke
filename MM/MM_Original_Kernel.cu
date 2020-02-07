/*This sample implements matrix multiplication as described in Chapter 3
 * of the programming guide.
 * It has been written for clarity of exposition to illustrate various CUDA
 * programming principles, not with the goal of providing the most
 * performant generic kernel for matrix multiplication.
 *
 * See also:
 * V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
 * in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
 * Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
 */


// System includes 
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>

#include "../memaddrcnt.cuh"
#include "../elastic_kernel.h"
#include "MM.h"

extern t_tqueue *tqueues;

 __device__ uint get_smid_MM(void) {

     uint ret; 

     asm("mov.u32 %0, %smid;" : "=r"(ret) );

     return ret;

}



/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
template <int BLOCK_SIZE> __global__ void
original_matrixMulCUDA(float *C, float *A, float *B, int wA, int wB, int gridDimX)
{
    // Block index
    //int bx = blockIdx.x;
    //int by = blockIdx.y;
	int bx = blockIdx.x % gridDimX; // This is necessary because of the grid linearization
	int by = blockIdx.x / gridDimX;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
         a <= aEnd;
         a += aStep, b += bStep)
    {

        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll

        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

template <int BLOCK_SIZE> __global__ void
__launch_bounds__(256, 8)
profiling_matrixMulCUDA(float *C, float *A, float *B, int wA, int wB, int gridDimX,
										int num_subtask,
										int iter_per_subtask,
										int *cont_SM,
										int *cont_subtask,
										State *status)
{
	__shared__ int s_bid, CTA_cont;
	
	unsigned int SM_id = get_smid_MM();
	
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
	
	while (1){
	
		// Block index
		// int bx = blockIdx.x;
		// int by = blockIdx.y;
		
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
		
		//if (s_bid[warpid] >= num_subtask || s_bid[warpid] == -1)
		if (s_bid >=num_subtask || s_bid ==-1) /* If all subtasks have been executed */{
			//if (threadIdx.x == 0 && threadIdx.y== 0)
			//	printf ("SM=%d CTA=%d Executed_tasks= %d \n", SM_id, CTA_cont, cont_task);	
			return;
		}
		
		if (threadIdx.x == 0 && threadIdx.y== 0) // Acumula numeor de tareas ejecutadas
			 cont_task++;

		int bx = s_bid % gridDimX;
		int by = s_bid / gridDimX;
	
		// Thread index
		int tx = threadIdx.x;
		int ty = threadIdx.y;

		// Index of the first sub-matrix of A processed by the block
		int aBegin = wA * BLOCK_SIZE * by;

		// Index of the last sub-matrix of A processed by the block
		int aEnd   = aBegin + wA - 1;

		// Step size used to iterate through the sub-matrices of A
		int aStep  = BLOCK_SIZE;

		// Index of the first sub-matrix of B processed by the block
		int bBegin = BLOCK_SIZE * bx;

		// Step size used to iterate through the sub-matrices of B
		int bStep  = BLOCK_SIZE * wB;

		// Csub is used to store the element of the block sub-matrix
		// that is computed by the thread
		float Csub = 0;

		// Loop over all the sub-matrices of A and B
		// required to compute the block sub-matrix
		for (int a = aBegin, b = bBegin;
			a <= aEnd;
			a += aStep, b += bStep)
		{

			// Declaration of the shared memory array As used to
			// store the sub-matrix of A
			__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

			// Declaration of the shared memory array Bs used to
			// store the sub-matrix of B
			__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

			// Load the matrices from device memory
			// to shared memory; each thread loads
			// one element of each matrix
			As[ty][tx] = A[a + wA * ty + tx];
			Bs[ty][tx] = B[b + wB * ty + tx];

			// Synchronize to make sure the matrices are loaded
			__syncthreads();

			// Multiply the two matrices together;
			// each thread computes one element
			// of the block sub-matrix
		#pragma unroll

			for (int k = 0; k < BLOCK_SIZE; ++k)
			{
				Csub += As[ty][k] * Bs[k][tx];
			}

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
			__syncthreads();
		}

    // Write the block sub-matrix to device memory;
    // each thread writes one element
		int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
		C[c + wB * ty + tx] = Csub;
	}
}

template <int BLOCK_SIZE> __global__ void
__launch_bounds__(256, 8)
SMT_matrixMulCUDA(float *C, float *A, float *B, int wA, int wB, int gridDimX,
					int SIMD_min, int SIMD_max,
					int num_subtask, int iter_per_subtask, int *cont_subtask, State *status)
{
	__shared__ int s_bid;
	
	unsigned int SM_id = get_smid_MM();
	
	if (SM_id <SIMD_min || SM_id > SIMD_max) /* Only blocks executing within SIMD_min and SIMD_max can progress */ 
			return;
		
	//if (threadIdx.x==0 && threadIdx.y== 0) // Ojo, esto es una prueba. Habría que tener en cuenta iteraciones entre distintos bloques
	//	printf("SMID=%d \n", SM_id);
	
	#ifdef SHOW_SM
		if (threadIdx.x==0 && threadIdx.y== 0)
			printf("%d, MM\n", SM_id);
	#endif
	
	while (1){
	
		// Block index
		// int bx = blockIdx.x;
		// int by = blockIdx.y;
		
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
		
		//if (s_bid[warpid] >= num_subtask || s_bid[warpid] == -1)
		if (s_bid >=num_subtask || s_bid ==-1) /* If all subtasks have been executed */{
			//if (threadIdx.x == 0)  printf("El bloque %d se sale con %d\n", blockIdx.x, s_bid); 
			return;
		}

		int bx = s_bid % gridDimX;
		int by = s_bid / gridDimX;
	
		// Thread index
		int tx = threadIdx.x;
		int ty = threadIdx.y;

		// Index of the first sub-matrix of A processed by the block
		int aBegin = wA * BLOCK_SIZE * by;

		// Index of the last sub-matrix of A processed by the block
		int aEnd   = aBegin + wA - 1;

		// Step size used to iterate through the sub-matrices of A
		int aStep  = BLOCK_SIZE;

		// Index of the first sub-matrix of B processed by the block
		int bBegin = BLOCK_SIZE * bx;

		// Step size used to iterate through the sub-matrices of B
		int bStep  = BLOCK_SIZE * wB;

		// Csub is used to store the element of the block sub-matrix
		// that is computed by the thread
		float Csub = 0;

		// Loop over all the sub-matrices of A and B
		// required to compute the block sub-matrix
		for (int a = aBegin, b = bBegin;
			a <= aEnd;
			a += aStep, b += bStep)
		{

			// Declaration of the shared memory array As used to
			// store the sub-matrix of A
			__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

			// Declaration of the shared memory array Bs used to
			// store the sub-matrix of B
			__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

			// Load the matrices from device memory
			// to shared memory; each thread loads
			// one element of each matrix
			As[ty][tx] = A[a + wA * ty + tx];
			Bs[ty][tx] = B[b + wB * ty + tx];

			// Synchronize to make sure the matrices are loaded
			__syncthreads();

			// Multiply the two matrices together;
			// each thread computes one element
			// of the block sub-matrix
		#pragma unroll

			for (int k = 0; k < BLOCK_SIZE; ++k)
			{
				Csub += As[ty][k] * Bs[k][tx];
			}

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
			__syncthreads();
		}

    // Write the block sub-matrix to device memory;
    // each thread writes one element
		int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
		C[c + wB * ty + tx] = Csub;
	}
}

template <int BLOCK_SIZE> __global__ void
__launch_bounds__(256, 8)
memaddr_SMT_matrixMulCUDA(float *C, float *A, float *B, int wA, int wB, int gridDimX,
					int *numUniqueAddr, int SIMD_min, int SIMD_max,
					int num_subtask, int iter_per_subtask, int *cont_subtask, State *status)
{
	__shared__ int s_bid;
	
	unsigned int SM_id = get_smid_MM();
	
	if (SM_id <SIMD_min || SM_id > SIMD_max) /* Only blocks executing within SIMD_min and SIMD_max can progress */ 
			return;
		
	//if (threadIdx.x==0 && threadIdx.y== 0) // Ojo, esto es una prueba. Habría que tener en cuenta iteraciones entre distintos bloques
	//	*status = RUNNING;
	
	while (1){
	
		// Block index
		// int bx = blockIdx.x;
		// int by = blockIdx.y;
		
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
		
		//if (s_bid[warpid] >= num_subtask || s_bid[warpid] == -1)
		if (s_bid >=num_subtask || s_bid ==-1) /* If all subtasks have been executed */{
			//if (threadIdx.x == 0)  printf("El bloque %d se sale con %d\n", blockIdx.x, s_bid); 
			return;
		}

		int bx = s_bid % gridDimX;
		int by = s_bid / gridDimX;
	
		// Thread index
		int tx = threadIdx.x;
		int ty = threadIdx.y;

		// Index of the first sub-matrix of A processed by the block
		int aBegin = wA * BLOCK_SIZE * by;

		// Index of the last sub-matrix of A processed by the block
		int aEnd   = aBegin + wA - 1;

		// Step size used to iterate through the sub-matrices of A
		int aStep  = BLOCK_SIZE;

		// Index of the first sub-matrix of B processed by the block
		int bBegin = BLOCK_SIZE * bx;

		// Step size used to iterate through the sub-matrices of B
		int bStep  = BLOCK_SIZE * wB;

		// Csub is used to store the element of the block sub-matrix
		// that is computed by the thread
		float Csub = 0;

		// Loop over all the sub-matrices of A and B
		// required to compute the block sub-matrix
		for (int a = aBegin, b = bBegin;
			a <= aEnd;
			a += aStep, b += bStep)
		{

			// Declaration of the shared memory array As used to
			// store the sub-matrix of A
			__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

			// Declaration of the shared memory array Bs used to
			// store the sub-matrix of B
			__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

			// Load the matrices from device memory
			// to shared memory; each thread loads
			// one element of each matrix
#if defined(COUNT_ALL_TASKS)
				if ( s_bid == 0 )
#endif
			{
				get_unique_lines((intptr_t) &A[a + wA * ty + tx], numUniqueAddr);
				get_unique_lines((intptr_t) &B[b + wB * ty + tx], numUniqueAddr);
				get_conflicting_banks( (intptr_t) &As[ty][tx], &numUniqueAddr[1] );					
				get_conflicting_banks( (intptr_t) &Bs[ty][tx], &numUniqueAddr[1] );					
			}
			As[ty][tx] = A[a + wA * ty + tx];
			Bs[ty][tx] = B[b + wB * ty + tx];

			// Synchronize to make sure the matrices are loaded
			__syncthreads();

			// Multiply the two matrices together;
			// each thread computes one element
			// of the block sub-matrix
		#pragma unroll

			for (int k = 0; k < BLOCK_SIZE; ++k)
			{
#if defined(COUNT_ALL_TASKS)
				if ( s_bid == 0 )
#endif
				{
					get_conflicting_banks( (intptr_t) &As[ty][k], &numUniqueAddr[1] );					
					get_conflicting_banks( (intptr_t) &Bs[k][tx], &numUniqueAddr[1] );					
				}
				Csub += As[ty][k] * Bs[k][tx];
			}

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
			__syncthreads();
		}

    // Write the block sub-matrix to device memory;
    // each thread writes one element
		int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
#if defined(COUNT_ALL_TASKS)
				if ( s_bid == 0 )
#endif
{
			get_unique_lines((intptr_t) &C[c + wB * ty + tx], numUniqueAddr);
}
		C[c + wB * ty + tx] = Csub;
	}
}

template <int BLOCK_SIZE> __global__ void
__launch_bounds__(256, 8)
SMK_matrixMulCUDA(float *C, float *A, float *B, int wA, int wB, int gridDimX,
					int max_blocks_per_SM,
					int num_subtask,
					int iter_per_subtask,
					int *cont_SM,
					int *cont_subtask,
					State *status
)
{
	__shared__ int s_bid, s_index;
	
	unsigned int SM_id = get_smid_MM();
	
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
		
		int bx = s_bid % gridDimX;
		int by = s_bid / gridDimX;
	
		// Thread index
		int tx = threadIdx.x;
		int ty = threadIdx.y;

		// Index of the first sub-matrix of A processed by the block
		int aBegin = wA * BLOCK_SIZE * by;

		// Index of the last sub-matrix of A processed by the block
		int aEnd   = aBegin + wA - 1;

		// Step size used to iterate through the sub-matrices of A
		int aStep  = BLOCK_SIZE;

		// Index of the first sub-matrix of B processed by the block
		int bBegin = BLOCK_SIZE * bx;

		// Step size used to iterate through the sub-matrices of B
		int bStep  = BLOCK_SIZE * wB;

		// Csub is used to store the element of the block sub-matrix
		// that is computed by the thread
		float Csub = 0;

		// Loop over all the sub-matrices of A and B
		// required to compute the block sub-matrix
		for (int a = aBegin, b = bBegin;
			a <= aEnd;
			a += aStep, b += bStep)
		{

			// Declaration of the shared memory array As used to
			// store the sub-matrix of A
			__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

			// Declaration of the shared memory array Bs used to
			// store the sub-matrix of B
			__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

			// Load the matrices from device memory
			// to shared memory; each thread loads
			// one element of each matrix
			As[ty][tx] = A[a + wA * ty + tx];
			Bs[ty][tx] = B[b + wB * ty + tx];

			// Synchronize to make sure the matrices are loaded
			__syncthreads();

			// Multiply the two matrices together;
			// each thread computes one element
			// of the block sub-matrix
		#pragma unroll

			for (int k = 0; k < BLOCK_SIZE; ++k)
			{
				Csub += As[ty][k] * Bs[k][tx];
			}

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
			__syncthreads();
		}

    // Write the block sub-matrix to device memory;
    // each thread writes one element
		int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
		C[c + wB * ty + tx] = Csub;
	}
}


void constantInit(float *data, int size, float val)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] = val;
    }
}

// float *h_MMA, *h_MMB, *h_MMC;
// float *d_MMA, *d_MMB, *d_MMC;
// int size_A, size_B, size_C;
// int dimA_x, dimB_x;

int MM_start_kernel(void *arg) 
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	t_MM_params * params = (t_MM_params *)kstub->params;
	
	dim3 dimsA, dimsB;
	
	dimsA = params->Asize;
	dimsB = params->Bsize;

    // Allocate host memory for matrices A and B
	
    params->size_A = dimsA.x * dimsA.y;
	params->dimA_x = dimsA.x;
    unsigned int mem_size_A = sizeof(float) * params->size_A;
    params->h_MMA = (float *)malloc(mem_size_A);
    params->size_B = dimsB.x * dimsB.y;
	params->dimB_x=dimsB.x;
    unsigned int mem_size_B = sizeof(float) * params->size_B;
    params->h_MMB = (float *)malloc(mem_size_B);

    // Initialize host memory
    const float valB = 0.01f;
    constantInit(params->h_MMA, params->size_A, 1.0f);
    constantInit(params->h_MMB, params->size_B, valB);

    // Allocate host matrix C
    dim3 dimsC(dimsB.x, dimsA.y, 1);
	params->size_C = dimsC.x * dimsC.y;
    unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
    params->h_MMC = (float *) malloc(mem_size_C);

    if (params->h_MMC == NULL)
    {
        fprintf(stderr, "Failed to allocate host matrix C!\n");
        exit(EXIT_FAILURE);
    }

    cudaError_t error;

    error = cudaMalloc((void **) &params->d_MMA, mem_size_A);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc((void **) &params->d_MMB, mem_size_B);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_B returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc((void **) &params->d_MMC, mem_size_C);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_C returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    // copy host memory to device
    error = cudaMemcpy(params->d_MMA, params->h_MMA, mem_size_A, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_A,h_A) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMemcpy(params->d_MMB, params->h_MMB, mem_size_B, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_B,h_B) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
	
	return 0;
}

int MM_start_mallocs(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	cudaMemcpyAsync(&kstub->gm_state[0], &kstub->h_state[0], sizeof(State), cudaMemcpyHostToDevice, *(kstub->preemp_s)); 
	cudaDeviceSynchronize();
	dim3 dimsA, dimsB;
	
	t_MM_params * params = (t_MM_params *)kstub->params;
	dimsA = params->Asize;
	dimsB = params->Bsize;
	
    // Allocate host memory for matrices A and B
	
    params->size_A = dimsA.x * dimsA.y;
	params->dimA_x = dimsA.x;
    unsigned int mem_size_A = sizeof(float) * params->size_A;
   
    params->size_B = dimsB.x * dimsB.y;
	params->dimB_x=dimsB.x;
    unsigned int mem_size_B = sizeof(float) * params->size_B;
   
	// Allocate host matrix C
    dim3 dimsC(dimsB.x, dimsA.y, 1);
	params->size_C = dimsC.x * dimsC.y;
    unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
	
	
 
#if defined(MEMCPY_SYNC) || defined(MEMCPY_ASYNC)
	
	cudaMallocHost(&params->h_MMA, mem_size_A);
	cudaMallocHost(&params->h_MMB, mem_size_A);
	cudaMallocHost(&params->h_MMC, mem_size_A);
	
	if (params->h_MMC == NULL)
    {
        fprintf(stderr, "Failed to allocate host matrix C!\n");
        exit(EXIT_FAILURE);
    }

    cudaError_t error;

    error = cudaMalloc((void **) &params->d_MMA, mem_size_A);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc((void **) &params->d_MMB, mem_size_B);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_B returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc((void **) &params->d_MMC, mem_size_C);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_C returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
#else
	
	#ifdef MANAGED_MEM
	
	checkCudaErrors(cudaMallocManaged(&params->h_MMA, mem_size_A*sizeof(float)));
	checkCudaErrors(cudaMallocManaged(&params->h_MMB, mem_size_B*sizeof(float)));
	checkCudaErrors(cudaMallocManaged(&params->h_MMC, mem_size_C*sizeof(float)));
	
	params->d_MMA = params->h_MMA;
	params->d_MMB = params->h_MMB;
	params->d_MMC = params->h_MMC;
	
	#else
		printf("No transfer model: Exiting ...\n");
		exit(-1);
	#endif
#endif
	
	 // Initialize host memory
    const float valB = 0.01f;
    constantInit(params->h_MMA, params->size_A, 1.0f);
    constantInit(params->h_MMB, params->size_B, valB);
	constantInit(params->h_MMC, params->size_C, 0.0f);
	
	return 0;
}

int MM_start_transfers(void *arg)
{

	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	
	dim3 dimsA, dimsB;
	
	t_MM_params * params = (t_MM_params *)kstub->params;
	dimsA = params->Asize;
	dimsB = params->Bsize;
	 
    // Allocate host memory for matrices A and B
	
    params->size_A = dimsA.x * dimsA.y;
	params->dimA_x = dimsA.x;
    unsigned int mem_size_A = sizeof(float) * params->size_A;
   
    params->size_B = dimsB.x * dimsB.y;
	params->dimB_x=dimsB.x;
    unsigned int mem_size_B = sizeof(float) * params->size_B;
	
	dim3 dimsC(dimsB.x, dimsA.y, 1);
	params->size_C = dimsC.x * dimsC.y;
    unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
 

#ifdef MEMCPY_SYNC

	cudaError_t error;

	// copy host memory to device
	//HtD_data_transfer(d_MMA, h_MMA, mem_size_A, C_S);
	printf("llamando enqueue\n");
	enqueue_tcomamnd(tqueues, params->d_MMA, params->h_MMA, mem_size_A, cudaMemcpyHostToDevice, 0, BLOCKING, DATA, LOW, kstub);
    /*error = cudaMemcpy(d_MMA, h_MMA, mem_size_A, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_A,h_A) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }*/

	//HtD_data_transfer(d_MMB, h_MMB, mem_size_B, C_S);
	enqueue_tcomamnd(tqueues, params->d_MMB, params->h_MMB, mem_size_B, cudaMemcpyHostToDevice, 0, BLOCKING, DATA, LOW,  kstub);

    /*error = cudaMemcpy(d_MMB, h_MMB, mem_size_B, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_B,h_B) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }*/
	
	//cudaEventRecord(kstub->end_HtD); // Stream 0
	
	kstub->HtD_tranfers_finished = 1;
	
#else
	#ifdef MEMCPY_ASYNC

	cudaError_t error;

	// copy host memoray to device
    error = cudaMemcpyAsync(params->d_MMA, params->h_MMA, mem_size_A, cudaMemcpyHostToDevice, kstub->transfer_s[0]);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_A,h_A) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
	//enqueue_tcomamnd(tqueues, d_MMA, h_MMA, mem_size_A, cudaMemcpyHostToDevice, kstub->transfer_s[0], NONBLOCKING, DATA, MEDIUM, kstub);

    error = cudaMemcpyAsync(params->d_MMB, params->h_MMB, mem_size_B, cudaMemcpyHostToDevice, kstub->transfer_s[0]);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_B,h_B) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
	
	cudaStreamSynchronize(kstub->transfer_s[0]);
	kstub->HtD_tranfers_finished = 1;
	
	//t_tcommand *com = enqueue_tcomamnd(tqueues, d_MMB, h_MMB, mem_size_B, cudaMemcpyHostToDevice, kstub->transfer_s[0], NONBLOCKING, LAST_TRANSFER, MEDIUM, kstub);
	//cudaEventSynchronize(com->end_transfers);

	/*pthread_mutex_lock(&com->lock);
	pthread_cond_wait( &com->cond, &com->lock); 
	pthread_mutex_unlock(&com->lock);*/
	
	
	//enqueue_tcomamnd(tqueues, NULL, NULL, 0, cudaMemcpyHostToDevice, kstub->transfer_s[0], STREAM_SYNCHRO, DATA, MEDIUM);
	//cudaEventSynchronize(com->end_transfers);
	//printf("Sincro HtD\n");
	
	//cudaEventRecord(kstub->end_HtD, kstub->transfer_s[0]);
	//cudaStreamSynchronize(kstub->transfer_s[0]);
	
	//kstub->HtD_tranfers_finished = 1;
	
	#else
	#ifdef MANAGED_MEM
	cudaDeviceProp p;
    cudaGetDeviceProperties(&p, kstub->deviceId);
	cudaError_t err;

	if (p.concurrentManagedAccess)
	{
		err = cudaMemPrefetchAsync(params->h_MMA, mem_size_A*sizeof(float), kstub->deviceId, kstub->transfer_s[0]);
		if ( err != cudaSuccess) {
			printf("Error in MM:cudaMemPrefetchAsync\n");
			exit(EXIT_FAILURE);
		}
		err = cudaMemPrefetchAsync(params->h_MMB, mem_size_B*sizeof(float), kstub->deviceId, kstub->transfer_s[0]);
		if ( err != cudaSuccess) {
			printf("Error in MM:cudaMemPrefetchAsync\n");
			exit(EXIT_FAILURE);
		}
		err = cudaMemPrefetchAsync(params->h_MMC, mem_size_C*sizeof(float), kstub->deviceId, kstub->transfer_s[0]);
		if ( err != cudaSuccess) {
			printf("Error in MM:cudaMemPrefetchAsync\n");
			exit(EXIT_FAILURE);
		}
	}
	
	//cudaEventRecord(kstub->end_HtD, kstub->transfer_s[0]);
	cudaStreamSynchronize(kstub->transfer_s[0]);
	kstub->HtD_tranfers_finished = 1;
	
	
	#endif
	#endif
#endif	

	return 0;
	
}

int MM_end_kernel(void *arg)
{
	cudaError_t error;
	
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	t_MM_params * params = (t_MM_params *)kstub->params;
	
#ifdef MEMCPY_SYNC

	cudaEventSynchronize(kstub->end_Exec);
	
	//DtH_data_transfer(h_MMC, d_MMC, size_C*sizeof(float), C_S);
	
	t_tcommand *com = enqueue_tcomamnd(tqueues, params->h_MMC, params->d_MMC, params->size_C*sizeof(float), cudaMemcpyDeviceToHost, 0, BLOCKING, LAST_TRANSFER, LOW, kstub);
	cudaEventSynchronize(com->end_transfers);
       // Copy result from device to host
    /*error = cudaMemcpy(h_MMC, d_MMC, size_C*sizeof(float), cudaMemcpyDeviceToHost);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (h_C,d_C) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }*/
#else
	#ifdef MEMCPY_ASYNC
		
	//error = cudaEventSynchronize(kstub->end_Exec);
	
	   // Copy result from device to host
    error = cudaMemcpyAsync(params->h_MMC, params->d_MMC, params->size_C*sizeof(float), cudaMemcpyDeviceToHost, kstub->transfer_s[1]);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (h_C,d_C) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
	cudaStreamSynchronize(kstub->transfer_s[1]);
	
	/*error = cudaMemcpy(h_MMC, d_MMC, size_C*sizeof(float), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess)
    {
        printf("cudaMemcpy (h_C,d_C) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }*/
	
	
	//		printf("-->Comienzo de DtH para tarea %d\n", kstub->id);

	//t_tcommand *com = enqueue_tcomamnd(tqueues, h_MMC, d_MMC, size_C*sizeof(float), cudaMemcpyDeviceToHost, kstub->transfer_s[1] , NONBLOCKING, LAST_TRANSFER, MEDIUM, kstub);
	//enqueue_tcomamnd(tqueues, NULL, NULL, 0, cudaMemcpyDeviceToHost, kstub->transfer_s[1], STREAM_SYNCHRO, DATA, MEDIUM);
	
	//cudaEventSynchronize(com->end_transfers);
	kstub->DtH_tranfers_finished = 1;
	
	//printf("-->Fin de DtH para tarea %d\n", kstub->id);


	//cudaEventRecord(kstub->end_DtH, kstub->transfer_s[1]);
	
	/*cudaEventSynchronize(kstub->end_DtH);*/
	
	#else
		#ifdef MANAGED_MEM
			cudaStreamSynchronize(*(kstub->execution_s)); // To be sure kernel execution has finished before processing output data
		#endif
	#endif
#endif
	/*
    printf("Checking computed result for correctness: ");
    bool correct = true;

    // test relative error by the formula
    //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
    double eps = 1.e-6 ; // machine zero
    const float valB = 0.01f;

    for (int i = 0; i < size_C; i++)
    {
        double abs_err = fabs(h_MMC[i] - (dimA_x * valB));
        double dot_length = dimA_x;
        double abs_val = fabs(h_MMC[i]);
        double rel_err = abs_err/abs_val/dot_length ;

        if (rel_err > eps)
        {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i, h_MMC[i], dimA_x*valB, eps);
            correct = false;
        }
    }

    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");
	*/
/*
#if defined(MEMCPY_SYNC) || defined(MEMCPY_ASYNC)
	
    // Clean up memory
    cudaFreeHost(h_MMA);
    cudaFreeHost(h_MMB);
    cudaFreeHost(h_MMC);
    cudaFree(d_MMA);
    cudaFree(d_MMB);
    cudaFree(d_MMC);
#else
	cudaFree(h_MMA);
    cudaFree(h_MMB);
    cudaFree(h_MMC);
#endif
*/
    return 0;
}
 
int launch_orig_MM(void *arg)
{
	
	t_kernel_stub *kstub = (t_kernel_stub *) arg;
		
	dim3 dimsA, dimsB;
	
	t_MM_params * params = (t_MM_params *)kstub->params;
	dimsA = params->Asize;
	dimsB = params->Bsize;
	
	
	// Setup execution parameters
    dim3 threads(kstub->kconf.blocksize.x, kstub->kconf.blocksize.y);
    dim3 grid(kstub->kconf.gridsize.x, kstub->kconf.gridsize.y);

    // Create and start timer

    // Performs warmup operation using matrixMul CUDA kernel
    if (kstub->kconf.blocksize.x == 16)
    {
        original_matrixMulCUDA<16><<< kstub->kconf.gridsize.x, threads, 0, *(kstub->execution_s) >>>(params->d_MMC, params->d_MMA, params->d_MMB, dimsA.x, dimsB.x, params->gridDimX);
    }
    else
    {
        original_matrixMulCUDA<32><<< kstub->kconf.gridsize.x, threads, 0, *(kstub->execution_s) >>>(params->d_MMC, params->d_MMA, params->d_MMB, dimsA.x, dimsB.x, params->gridDimX);
    }
		
		return 0;
}

int prof_MM(void * arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	
	dim3 dimsA, dimsB;
	
	t_MM_params * params = (t_MM_params *)kstub->params;
	dimsA = params->Asize;
	dimsB = params->Bsize;
	
	// Setup execution parameters
    dim3 threads(kstub->kconf.blocksize.x, kstub->kconf.blocksize.y);
	
	if (kstub->kconf.blocksize.x == 16)
		profiling_matrixMulCUDA<16><<<kstub->kconf.numSMs * kstub->kconf.max_persistent_blocks, threads, 0, *(kstub->execution_s)>>>(params->d_MMC, params->d_MMA, params->d_MMB, dimsA.x, dimsB.x, params->gridDimX,
						kstub->total_tasks,
						kstub->kconf.coarsening,
						kstub->d_SMs_cont,
						kstub->d_executed_tasks,
						&kstub->gm_state[kstub->stream_index]);
	else
		profiling_matrixMulCUDA<32><<< kstub->kconf.numSMs * kstub->kconf.max_persistent_blocks, threads, 0, *(kstub->execution_s)>>>(params->d_MMC, params->d_MMA, params->d_MMB, dimsA.x, dimsB.x, params->gridDimX,
						kstub->total_tasks,
						kstub->kconf.coarsening,
						kstub->d_SMs_cont,
						kstub->d_executed_tasks,
						&kstub->gm_state[kstub->stream_index]);
}	

int launch_preemp_MM(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	
	dim3 dimsA, dimsB;
	
	t_MM_params * params = (t_MM_params *)kstub->params;
	dimsA = params->Asize;
	dimsB = params->Bsize;
	
	// Setup execution parameters
    dim3 threads(kstub->kconf.blocksize.x, kstub->kconf.blocksize.y);
//	printf("Launching %d blocks with %d threads and %d tasks\n", kstub->kconf.numSMs * kstub->kconf.max_persistent_blocks, kstub->kconf.blocksize.x * kstub->kconf.blocksize.y, kstub->total_tasks);
	
	#ifdef SMT

	if ( !(kstub->memaddr_profile) )
		if (kstub->kconf.blocksize.x == 16)
			SMT_matrixMulCUDA<16><<<kstub->kconf.numSMs * kstub->kconf.max_persistent_blocks, threads, 0, *(kstub->execution_s)>>>(params->d_MMC,
						params->d_MMA, params->d_MMB, dimsA.x, dimsB.x, params->gridDimX,
						kstub->idSMs[0],
						kstub->idSMs[1],
						kstub->total_tasks,
						kstub->kconf.coarsening,
						kstub->d_executed_tasks,
						&(kstub->gm_state[kstub->stream_index]));
		else 
			SMT_matrixMulCUDA<32><<< kstub->kconf.numSMs * kstub->kconf.max_persistent_blocks, threads, 0, *(kstub->execution_s)>>>(params->d_MMC,
						params->d_MMA, params->d_MMB, dimsA.x, dimsB.x, params->gridDimX,
						kstub->idSMs[0],
						kstub->idSMs[1],
						kstub->total_tasks,
						kstub->kconf.coarsening,
						kstub->d_executed_tasks,
						&kstub->gm_state[kstub->stream_index]);
	else
		if (kstub->kconf.blocksize.x == 16)
			memaddr_SMT_matrixMulCUDA<16><<<kstub->kconf.numSMs * kstub->kconf.max_persistent_blocks, threads, 0, *(kstub->execution_s)>>>(params->d_MMC,
						params->d_MMA, params->d_MMB, dimsA.x, dimsB.x, params->gridDimX,
						kstub->d_numUniqueAddr,
						kstub->idSMs[0],
						kstub->idSMs[1],
						kstub->total_tasks,
						kstub->kconf.coarsening,
						kstub->d_executed_tasks,
						&(kstub->gm_state[kstub->stream_index]));
		else 
			memaddr_SMT_matrixMulCUDA<32><<< kstub->kconf.numSMs * kstub->kconf.max_persistent_blocks, threads, 0, *(kstub->execution_s)>>>(params->d_MMC,
						params->d_MMA, params->d_MMB, dimsA.x, dimsB.x, params->gridDimX,
						kstub->d_numUniqueAddr,
						kstub->idSMs[0],
						kstub->idSMs[1],
						kstub->total_tasks,
						kstub->kconf.coarsening,
						kstub->d_executed_tasks,
						&kstub->gm_state[kstub->stream_index]);
	
	#else
		
	
	if (kstub->kconf.blocksize.x == 16)
		SMK_matrixMulCUDA<16><<<kstub->kconf.numSMs * kstub->kconf.max_persistent_blocks, threads, 0, *(kstub->execution_s)>>(params->d_MMC, params->d_MMA, params->d_MMB, dimsA.x, dimsB.x, params->gridDimX,
						kstub->num_blocks_per_SM,
						kstub->total_tasks,
						kstub->kconf.coarsening,
						kstub->d_SMs_cont,
						kstub->d_executed_tasks,
						&kstub->gm_state[kstub->stream_index]
						);
	else
		SMK_matrixMulCUDA<32><<<kstub->kconf.numSMs * kstub->kconf.max_persistent_blocks, threads, 0, *(kstub->execution_s)>>>(params->d_MMC, params->d_MMA, params->d_MMB, dimsA.x, dimsB.x, params->gridDimX,
						kstub->num_blocks_per_SM,
						kstub->total_tasks,
						kstub->kconf.coarsening,
						kstub->d_SMs_cont,
						kstub->d_executed_tasks,
						&kstub->gm_state[kstub->stream_index]
						);
		
	#endif
	
	return 0;
}
 



 