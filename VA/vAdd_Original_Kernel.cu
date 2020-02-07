/**
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */
 
#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_") 
#include <cuda_runtime.h>
#include <helper_functions.h>   // helper functions for string parsing
#include <helper_cuda.h>        // helper functions CUDA error checking and initialization
#include <semaphore.h>
#include "../elastic_kernel.h"
#include "VA.h"
#include "../memaddrcnt.cuh"

//extern t_tqueue *tqueues;

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */

 
  __device__ uint get_smid_VA(void) {

     uint ret;

     asm("mov.u32 %0, %smid;" : "=r"(ret) );

     return ret;

}

__global__ void
original_vectorAdd(const float *A, const float *B, float *C, int iter_per_block, int numelements)
{
	
	for (int k=0; k<iter_per_block; k++) {
			
		//const int i = k * gridDim.x * blockDim.x + j * gridDim.x * blockDim.x +  
		//blockIdx.x * blockDim.x + threadIdx.x;
			
		const int i = blockIdx.x * blockDim.x * iter_per_block + k * blockDim.x + threadIdx.x;
			
		if (i < numelements)
				C[i] = A[i] + B[i];
	}
	

}

__global__ void
preempt_SMK_vectorAdd(const float *A, const float *B, float *C, int numelements, 
			int max_blocks_per_SM, 
			int num_subtask, int iter_per_subtask, int *cont_SM, int *cont_subtask, State *status)
{
	int i;
	
	__shared__ int s_bid, s_index;
	
	unsigned int SM_id = get_smid_VA();
	
	if (threadIdx.x == 0)  
		s_index = atomicAdd(&cont_SM[SM_id],1);
	
	__syncthreads();

	if (s_index > max_blocks_per_SM)
		return;
	
	//unsigned int SM_id = get_smid_VA();
	
	//int warpid = threadIdx.x >> 5;
	
	//int thIdxwarp = threadIdx.x & 0x1F;
	
	//if (SM_id <SIMD_min || SM_id > SIMD_max) /* Only blocks executing within SIMD_min and SIMD_max can progress */ 
	//		return;
		
	while (1){
		
		/********** Task Id calculation *************/
		
		if (threadIdx.x == 0) {
			if (*status == TOEVICT)
				s_bid = -1;
			else
				s_bid = atomicAdd(cont_subtask, 1); //subtask_id
		}
		
		//if ( thIdxwarp == 0) {
		//	if (*status == TOEVICT)
		//		s_bid[warpid] = -1;
		//	else
		//		s_bid[warpid] = atomicAdd(cont_subtask + warpid, 1); //subtask_id
		//}
			
		__syncthreads();
		
		//if (s_bid[warpid] >= num_subtask || s_bid[warpid] == -1)
		if (s_bid >=num_subtask || s_bid ==-1) /* If all subtasks have been executed */
			return;
	
		for (int j=0; j<iter_per_subtask; j++) {
			
			 i = s_bid * blockDim.x * iter_per_subtask +  j * blockDim.x + threadIdx.x;
			
			if (i < numelements)
					C[i] = A[i] + B[i];
		}
		
	}

}

__global__ void
preempt_SMT_vectorAdd(const float *A, const float *B, float *C, int numelements, 
			int SIMD_min, int SIMD_max,
			int num_subtask, int iter_per_subtask, int *cont_subtask, State *status)
{
	int i;
	
	__shared__ int s_bid;
	
	unsigned int SM_id = get_smid_VA();
	
	//int warpid = threadIdx.x >> 5;
	
	//int thIdxwarp = threadIdx.x & 0x1F;
	
	if (SM_id <SIMD_min || SM_id > SIMD_max) /* Only blocks executing within SIMD_min and SIMD_max can progress */ 
			return;
			
	#ifdef SHOW_SM
		if (threadIdx.x == 0) 
			printf("%d, VA\n", SM_id);
	#endif
			
	//if (threadIdx.x==0) // Ojo, esto es una prueba. Habría que tener en cuenta iteraciones entre distintos bloques
	//	*status = RUNNING;
		
	while (1){
		
		/********** Task Id calculation *************/
		
		if (threadIdx.x == 0) {
			if (*status == TOEVICT)
				s_bid = -1;
			else
				s_bid = atomicAdd(cont_subtask, 1); //subtask_id
		}
		
		//if ( thIdxwarp == 0) {
		//	if (*status == TOEVICT)
		//		s_bid[warpid] = -1;
		//	else
		//		s_bid[warpid] = atomicAdd(cont_subtask + warpid, 1); //subtask_id
		//}
			
		__syncthreads();
		
		//if (s_bid[warpid] >= num_subtask || s_bid[warpid] == -1)
		if (s_bid >=num_subtask || s_bid ==-1) /* If all subtasks have been executed */
			return;
	
		for (int j=0; j<iter_per_subtask; j++) {
			
			 i = s_bid * blockDim.x * iter_per_subtask +  j * blockDim.x + threadIdx.x;
			
			if (i < numelements)
					C[i] = A[i] + B[i];
		}
		
	}

}

__global__ void
memaddr_preempt_SMT_vectorAdd(const float *A, const float *B, float *C, int numelements, 
			int *numUniqueAddr, int SIMD_min, int SIMD_max,
			int num_subtask, int iter_per_subtask, int *cont_subtask, State *status)
{
	int i;
	
	__shared__ int s_bid;
	
	unsigned int SM_id = get_smid_VA();
	
	//int warpid = threadIdx.x >> 5;
	
	//int thIdxwarp = threadIdx.x & 0x1F;
	
	if (SM_id <SIMD_min || SM_id > SIMD_max) /* Only blocks executing within SIMD_min and SIMD_max can progress */ 
			return;
			
	//if (threadIdx.x==0) // Ojo, esto es una prueba. Habría que tener en cuenta iteraciones entre distintos bloques
	//	*status = RUNNING;
		
	while (1){
		
		/********** Task Id calculation *************/
		
		if (threadIdx.x == 0) {
			if (*status == TOEVICT)
				s_bid = -1;
			else
				s_bid = atomicAdd(cont_subtask, 1); //subtask_id
		}
		
		//if ( thIdxwarp == 0) {
		//	if (*status == TOEVICT)
		//		s_bid[warpid] = -1;
		//	else
		//		s_bid[warpid] = atomicAdd(cont_subtask + warpid, 1); //subtask_id
		//}
			
		__syncthreads();
		
		//if (s_bid[warpid] >= num_subtask || s_bid[warpid] == -1)
		if (s_bid >=num_subtask || s_bid ==-1) /* If all subtasks have been executed */
			return;
	
		for (int j=0; j<iter_per_subtask; j++) {
			
			 i = s_bid * blockDim.x * iter_per_subtask +  j * blockDim.x + threadIdx.x;
			
			if (i < numelements)
			{
#if defined(COUNT_ALL_TASKS)

				if ( s_bid == 0 )
#endif
				{
					get_unique_lines((intptr_t) &A[i], numUniqueAddr);
					get_unique_lines((intptr_t) &B[i], numUniqueAddr);
					get_unique_lines((intptr_t) &C[i], numUniqueAddr);
				}
				C[i] = A[i] + B[i];
			}
		}
		
	}

}

__global__ void
profiling_SMT_vectorAdd(const float *A, const float *B, float *C, int numelements, 						
						int num_subtask,
						int iter_per_subtask,
						int *cont_SM,
						int *cont_subtask,
						State *status)
{
	int i;
	
	__shared__ int s_bid, CTA_cont;
	
	unsigned int SM_id = get_smid_VA();
	
	if (SM_id >= 8){ /* Only blocks executing in first 8 SM  are used for profiling */ 
		//delay();
		return;
	}
	
	if (threadIdx.x == 0) {
		CTA_cont = atomicAdd(&cont_SM[SM_id], 1);
	//	if (SM_id == 7 && CTA_cont == 8)
	//		printf("Aqui\n");
	}
	
	__syncthreads();
	
	if (CTA_cont > SM_id) {/* Only one block makes computation in SM0, two blocks in SM1 and so on */
		//delay();
		return;
	}
	
	//if (threadIdx.x == 0)
	//	printf ("SM=%d CTA = %d\n", SM_id, CTA_cont);

	int cont_task = 0;
		
	while (1){
		
		/********** Task Id calculation *************/
		
		if (threadIdx.x == 0) {
			if (*status == TOEVICT)
				s_bid = -1;
			else
				s_bid = atomicAdd(cont_subtask, 1); //subtask_id
		}
		
		//if ( thIdxwarp == 0) {
		//	if (*status == TOEVICT)
		//		s_bid[warpid] = -1;
		//	else
		//		s_bid[warpid] = atomicAdd(cont_subtask + warpid, 1); //subtask_id
		//}
			
		__syncthreads();
		
		//if (s_bid[warpid] >= num_subtask || s_bid[warpid] == -1)
		if (s_bid >=num_subtask || s_bid ==-1){ /* If all subtasks have been executed */
			if (threadIdx.x == 0)
				printf ("SM=%d CTA=%d Executed_tasks= %d \n", SM_id, CTA_cont, cont_task);
			return;
		}
		
		if (threadIdx.x == 0) // Acumula numeor de tareas ejecutadas
			 cont_task++;
	
		for (int j=0; j<iter_per_subtask; j++) {
			
			 i = s_bid * blockDim.x * iter_per_subtask +  j * blockDim.x + threadIdx.x;
			
			if (i < numelements)
				C[i] = A[i] + B[i];
		}
		
	}

}

/**
 * Host main routine
 */
 
//// Global variables

 // float *h_A;
 // float *h_B;
 // float *h_C;
 // float *d_A;
 // float *d_B;
 // float *d_C;
 
 // int numElements;
 
int VA_start_kernel_dummy(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	t_VA_params * params = (t_VA_params *)kstub->params;
	
	// Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    /*int*/ params->numElements = kstub->kconf.gridsize.x * kstub->kconf.blocksize.x * kstub->kconf.coarsening;
    size_t size = params->numElements * sizeof(float);
    //printf("[Vector addition of %d elements]\n", params->numElements);

    // Allocate the host input vector A
    params->h_A = (float *)malloc(size);

    // Allocate the host input vector B
    params->h_B = (float *)malloc(size);

    // Allocate the host output vector C
    params->h_C = (float *)malloc(size);

    // Verify that allocations succeeded
    if (params->h_A == NULL || params->h_B == NULL || params->h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < params->numElements; ++i)
    {
        params->h_A[i] = rand()/(float)RAND_MAX;
        params->h_B[i] = rand()/(float)RAND_MAX;
    }

    // Allocate the device input vector A
    params->d_A = NULL;
    err = cudaMalloc((void **)&params->d_A, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector B
    params->d_B = NULL;
    err = cudaMalloc((void **)&params->d_B, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    params->d_C = NULL;
    err = cudaMalloc((void **)&params->d_C, size);
    checkCudaErrors(cudaMemset(params->d_C, 0, size));


    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    //printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(params->d_A, params->h_A, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(params->d_B, params->h_B, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
	return 0;
}

int VA_start_mallocs(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	t_VA_params * params = (t_VA_params *)kstub->params;
	
	// Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    /*int*/ params->numElements = kstub->kconf.gridsize.x * kstub->kconf.blocksize.x * kstub->kconf.coarsening;
    size_t size = params->numElements * sizeof(float);
    //printf("[Vector addition of %d elements]\n", params->numElements);
	
#if defined(MEMCPY_SYNC) || defined(MEMCPY_ASYNC)

    // Allocate the host input vector A
	cudaMallocHost(&params->h_A, size);

    // Allocate the host input vector B
    cudaMallocHost(&params->h_B, size);

    // Allocate the host output vector C
    cudaMallocHost(&params->h_C, size);
	
	// Allocate the device input vector A
    params->d_A = NULL;
    err = cudaMalloc((void **)&params->d_A, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector B
    params->d_B = NULL;
    err = cudaMalloc((void **)&params->d_B, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    params->d_C = NULL;
    err = cudaMalloc((void **)&params->d_C, size);
    checkCudaErrors(cudaMemset(params->d_C, 0, size));


    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
#else
	#ifdef MANAGED_MEM

	cudaMallocManaged(&params->h_A, size);
	cudaMallocManaged(&params->h_B, size);
	cudaMallocManaged(&params->h_C, size);
	
	params->d_A = params->h_A;
	params->d_B = params->h_B;
	params->d_C = params->h_C;
	
	#else
		printf("No transfer model: Exiting ...\n");
		exit(-1);
	#endif
#endif


    // Verify that allocations succeeded
    if (params->h_A == NULL || params->h_B == NULL || params->h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < params->numElements; ++i)
    {
        params->h_A[i] = rand()/(float)RAND_MAX;
        params->h_B[i] = rand()/(float)RAND_MAX;
		params->h_C[i] = 0;
    }

	return 0;
}
	
int VA_start_transfers(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	t_VA_params * params = (t_VA_params *)kstub->params;

	size_t size = params->numElements * sizeof(float);
	
	// Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
	
#ifdef MEMCPY_SYNC
	// Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    //printf("Copy input data from the host memory to the CUDA device\n");
  /*  err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }*/
	
	/*HtD_data_transfer(d_A, h_A, size, C_S);*/
	
	enqueue_tcomamnd(tqueues, params->d_A, params->h_A, size, cudaMemcpyHostToDevice, 0, BLOCKING, DATA, LOW, kstub);

    /*err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }*/
	
	//HtD_data_transfer(d_B, h_B, size, C_S);
	enqueue_tcomamnd(tqueues, params->d_B, params->h_B, size, cudaMemcpyHostToDevice, 0, BLOCKING, DATA, LOW, kstub);
	
	kstub->HtD_tranfers_finished = 1;

	
#else
	
	#ifdef MEMCPY_ASYNC
	
	err = cudaMemcpyAsync(params->d_A, params->h_A, size, cudaMemcpyHostToDevice, kstub->transfer_s[0]);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpyAsync(params->d_B, params->h_B, size, cudaMemcpyHostToDevice, kstub->transfer_s[0]);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
	//cudaEventRecord(kstub->end_HtD, kstub->transfer_s[0]);
	cudaStreamSynchronize(kstub->transfer_s[0]);
	
	/*
	enqueue_tcomamnd(tqueues, d_A, h_A, size, cudaMemcpyHostToDevice, kstub->transfer_s[0], NONBLOCKING, DATA, MEDIUM, kstub);
	enqueue_tcomamnd(tqueues, d_B,  h_B, size, cudaMemcpyHostToDevice, kstub->transfer_s[0], NONBLOCKING, LAST_TRANSFER, MEDIUM, kstub);
*/
	//enqueue_tcomamnd(tqueues, NULL, NULL, 0, cudaMemcpyHostToDevice, kstub->transfer_s[0], STREAM_SYNCHRO, DATA, MEDIUM, kstub);
	
	
	//kstub->HtD_tranfers_finished = 1;

	#else
	#ifdef MANAGED_MEM

	cudaDeviceProp p;
    cudaGetDeviceProperties(&p, kstub->deviceId);
	
	if (p.concurrentManagedAccess)
	{
		err = cudaMemPrefetchAsync(params->h_A, size, kstub->deviceId);
		if ( err != cudaSuccess) {
			printf("Error in vAdd:cudaMemPrefetchAsync\n");
			exit(EXIT_FAILURE);
		}
		err = cudaMemPrefetchAsync(params->h_B, size, kstub->deviceId);
		if ( err != cudaSuccess) {
			printf("Error in vAdd:cudaMemPrefetchAsync\n");
			exit(EXIT_FAILURE);
		}
		err = cudaMemPrefetchAsync(params->h_C, size, kstub->deviceId);
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
	
 
int VA_end_kernel_dummy(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	t_VA_params * params = (t_VA_params *)kstub->params;
	
#ifdef MEMCPY_SYNC

	cudaEventSynchronize(kstub->end_Exec);

	// Copy the device result vector in device memory to the host result vector
    // in host memory.
    //printf("Copy output data from the CUDA device to the host memory\n");
    
	/* cudaError_t err = cudaSuccess;
	 err = cudaMemcpy(h_C, d_C, numElements*sizeof(float), cudaMemcpyDeviceToHost);

     if (err != cudaSuccess)
     {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
     }*/
	 
	 //DtH_data_transfer(h_C, d_C, numElements*sizeof(float), C_S);
	 enqueue_tcomamnd(tqueues, params->h_C, params->d_C, params->numElements*sizeof(float), cudaMemcpyDeviceToHost, 0, BLOCKING, DATA, LOW, kstub);
	 
#else
	#ifdef MEMCPY_ASYNC

	cudaError_t err = cudaSuccess;

	//err = cudaEventSynchronize(kstub->end_Exec);
	
	err = cudaMemcpyAsync(params->h_C, params->d_C, params->numElements*sizeof(float), cudaMemcpyDeviceToHost, kstub->transfer_s[1]);

     if (err != cudaSuccess)
     {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
     }
	 			printf("-->Comienzo de DtH para tarea %d\n", kstub->id);

	//enqueue_tcomamnd(tqueues, h_C, d_C, numElements*sizeof(float), cudaMemcpyDeviceToHost, kstub->transfer_s[1] , NONBLOCKING, LAST_TRANSFER, MEDIUM, kstub);
	//enqueue_tcomamnd(tqueues, NULL, NULL, 0, cudaMemcpyDeviceToHost, kstub->transfer_s[1], STREAM_SYNCHRO, DATA, MEDIUM, kstub);
	 
	//kstub->DtH_tranfers_finished = 1;
	
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
    // // Verify that the result vector is correct
     for (int i = 0; i < numElements; ++i)
     {
         if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
         {
             fprintf(stderr, "Result verification failed at element %d!\n", i);
             exit(EXIT_FAILURE);
         }
     }*/
    //printf("Test PASSED\n");
	
	/*
	
#if defined(MEMCPY_SYNC) || defined(MEMCPY_ASYNC)

    // Free device global memory
    err = cudaFree(d_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaFree(d_B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaFree(d_C);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);

    // Reset the device and exit
    //err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
#else
	
	cudaFree(h_A);
	cudaFree(h_B);
	cudaFree(h_C);
#endif
	*/
	return 0;

}	 



int launch_orig_VA(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	t_VA_params * params = (t_VA_params *)kstub->params;
	
	original_vectorAdd<<<kstub->kconf.gridsize.x, kstub->kconf.blocksize.x, 0, *(kstub->execution_s)>>>(
		params->d_A, params->d_B, params->d_C, 
		kstub->kconf.coarsening,
		params->numElements);
	
	return 0;
}

int prof_VA(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	t_VA_params * params = (t_VA_params *)kstub->params;
	
	profiling_SMT_vectorAdd<<<kstub->kconf.numSMs * kstub->kconf.max_persistent_blocks, kstub->kconf.blocksize.x, 0, *(kstub->execution_s)>>>/*(VAparams->d_A, VAparams->d_B, VAparams->d_C, VAparams->numElements, */
			(params->d_A, params->d_B, params->d_C, params->numElements,
			kstub->total_tasks,
			kstub->kconf.coarsening,
			kstub->d_SMs_cont,
			kstub->d_executed_tasks,
			&kstub->gm_state[kstub->stream_index]);
	return 0;
}

int launch_preemp_VA(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	t_VA_params * params = (t_VA_params *)kstub->params;
	
	#ifdef SMT

	if ( !(kstub->memaddr_profile) )
		preempt_SMT_vectorAdd<<<kstub->kconf.numSMs * kstub->kconf.max_persistent_blocks, kstub->kconf.blocksize.x, 0, *(kstub->execution_s)>>>/*(VAparams->d_A, VAparams->d_B, VAparams->d_C, VAparams->numElements, */
			(params->d_A, params->d_B, params->d_C, params->numElements,
			kstub->idSMs[0],
			kstub->idSMs[1],
			kstub->total_tasks,
			kstub->kconf.coarsening,
			kstub->d_executed_tasks,
			&(kstub->gm_state[kstub->stream_index]));
	else
		memaddr_preempt_SMT_vectorAdd<<<kstub->kconf.numSMs * kstub->kconf.max_persistent_blocks, kstub->kconf.blocksize.x, 0, *(kstub->execution_s)>>>/*(VAparams->d_A, VAparams->d_B, VAparams->d_C, VAparams->numElements, */
			(params->d_A, params->d_B, params->d_C, params->numElements,
			kstub->d_numUniqueAddr,
			kstub->idSMs[0],
			kstub->idSMs[1],
			kstub->total_tasks,
			kstub->kconf.coarsening,
			kstub->d_executed_tasks,
			&(kstub->gm_state[kstub->stream_index]));
	
	#else
		
	preempt_SMK_vectorAdd<<<kstub->kconf.numSMs * kstub->kconf.max_persistent_blocks, kstub->kconf.blocksize.x, 0, *(kstub->execution_s)>>>
			(params->d_A, params->d_B, params->d_C, params->numElements, 
			kstub->num_blocks_per_SM,
			kstub->total_tasks,
			kstub->kconf.coarsening,
			kstub->d_SMs_cont,
			kstub->d_executed_tasks,
			&(kstub->gm_state[kstub->stream_index])
	);	
		
	#endif
	
	return 0;
}