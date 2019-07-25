#include <stdio.h>
#include <algorithm>

#include "../cudacommon.h"
#include <cassert>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <iostream>
#include <semaphore.h>
#include "../elastic_kernel.h"
#include "reduction.h"

__device__ uint get_smid_reduction(void) {
	uint ret;

	asm("mov.u32 %0, %smid;" : "=r"(ret) );

	return ret;
}

// Reduction Kernel
// Kernel has been changed to increase the number of blocks. Now, one block
//   only reduces 2*blockDim.x elements
__global__ void
reduce(float *g_idata, float *g_odata, unsigned int n)
{
	extern __shared__ float sdata[];

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	//unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
	unsigned int gridSize = blockDim.x*2*gridDim.x;

	//float mySum = (i < n) ? g_idata[i] : 0;
	float mySum = 0;

	// if (i + blockDim.x < n)
		// mySum += g_idata[i+blockDim.x];
	
	while(i < n){
		mySum += g_idata[i];
		
		if(i + blockDim.x < n)
			mySum += g_idata[i+blockDim.x];

		i += gridSize;
	}

	sdata[tid] = mySum;
	__syncthreads();

	// do reduction in shared mem
	if ((blockDim.x >= 512) && (tid < 256))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 256];
	}

	__syncthreads();

	if ((blockDim.x >= 256) &&(tid < 128))
	{
			sdata[tid] = mySum = mySum + sdata[tid + 128];
	}

	 __syncthreads();

	if ((blockDim.x >= 128) && (tid <  64))
	{
	   sdata[tid] = mySum = mySum + sdata[tid +  64];
	}

	__syncthreads();

#if (__CUDA_ARCH__ >= 300 )
	if ( tid < 32 )
	{
		// Fetch final intermediate sum from 2nd warp
		if (blockDim.x >=  64) mySum += sdata[tid + 32];
		// Reduce final warp using shuffle
		for (int offset = warpSize/2; offset > 0; offset /= 2) 
		{
			mySum += __shfl_down(mySum, offset);
		}
	}
#else
	// fully unroll reduction within a single warp
	if ((blockDim.x >=  64) && (tid < 32))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 32];
	}

	__syncthreads();

	if ((blockDim.x >=  32) && (tid < 16))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 16];
	}

	__syncthreads();

	if ((blockDim.x >=  16) && (tid <  8))
	{
		sdata[tid] = mySum = mySum + sdata[tid +  8];
	}

	__syncthreads();

	if ((blockDim.x >=   8) && (tid <  4))
	{
		sdata[tid] = mySum = mySum + sdata[tid +  4];
	}

	__syncthreads();

	if ((blockDim.x >=   4) && (tid <  2))
	{
		sdata[tid] = mySum = mySum + sdata[tid +  2];
	}

	__syncthreads();

	if ((blockDim.x >=   2) && ( tid <  1))
	{
		sdata[tid] = mySum = mySum + sdata[tid +  1];
	}

	__syncthreads();
#endif

	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] += mySum;
}

__global__ void
profiling_reduce_kernel(float *g_idata, float *g_odata, unsigned int n,
		int num_subtask,
		int iter_per_subtask,
		int *cont_SM,
		int *cont_subtask,
		State *status)
{
	__shared__ int s_bid, CTA_cont;

	unsigned int SM_id = get_smid_reduction();
	
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
			
	//extern __shared__ float sdata[];
	//float mySum = 0;
		
	while (1){
		
		/********** Task Id calculation *************/
		if (threadIdx.x == 0) {
			if (*status == TOEVICT)
				s_bid = -1;
			else
				s_bid = atomicAdd(cont_subtask, 1);
		}
		
		__syncthreads();
		
		if (s_bid >= num_subtask || s_bid == -1){ /* If all subtasks have been executed */
			if (threadIdx.x == 0)
				printf ("SM=%d CTA=%d Executed_tasks= %d \n", SM_id, CTA_cont, cont_task);	
			return;
		}
		
		if (threadIdx.x == 0) // Acumula numeor de tareas ejecutadas
			 cont_task++;
		
		extern __shared__ float sdata[];

		// perform first level of reduction,
		// reading from global memory, writing to shared memory
		unsigned int tid = threadIdx.x;
		//unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
		unsigned int i = s_bid * (blockDim.x * 2) + threadIdx.x;
		unsigned int gridSize = blockDim.x * 2 * num_subtask;

		//float mySum = (i < n) ? g_idata[i] : 0;
		float mySum = 0;

		// if (i + blockDim.x < n)
			// mySum += g_idata[i+blockDim.x];
		
		int cont = 0;
		
		while(i < n && cont < iter_per_subtask){
			mySum += g_idata[i];
			
			if(i + blockDim.x < n)
				mySum += g_idata[i+blockDim.x];

			i += gridSize;
			cont++;
		}

		sdata[tid] = mySum;
		__syncthreads();

		// do reduction in shared mem
		if ((blockDim.x >= 512) && (tid < 256))
		{
			sdata[tid] = mySum = mySum + sdata[tid + 256];
		}

		__syncthreads();

		if ((blockDim.x >= 256) &&(tid < 128))
		{
			sdata[tid] = mySum = mySum + sdata[tid + 128];
		}

		 __syncthreads();

		if ((blockDim.x >= 128) && (tid <  64))
		{
		   sdata[tid] = mySum = mySum + sdata[tid +  64];
		}

		__syncthreads();

	#if (__CUDA_ARCH__ >= 300 )
		if ( tid < 32 )
		{
			// Fetch final intermediate sum from 2nd warp
			if (blockDim.x >=  64) mySum += sdata[tid + 32];
			// Reduce final warp using shuffle
			for (int offset = warpSize/2; offset > 0; offset /= 2) 
			{
				mySum += __shfl_down(mySum, offset);
			}
		}
	#else
		// fully unroll reduction within a single warp
		if ((blockDim.x >=  64) && (tid < 32))
		{
			sdata[tid] = mySum = mySum + sdata[tid + 32];
		}

		__syncthreads();

		if ((blockDim.x >=  32) && (tid < 16))
		{
			sdata[tid] = mySum = mySum + sdata[tid + 16];
		}

		__syncthreads();

		if ((blockDim.x >=  16) && (tid <  8))
		{
			sdata[tid] = mySum = mySum + sdata[tid +  8];
		}

		__syncthreads();

		if ((blockDim.x >=   8) && (tid <  4))
		{
			sdata[tid] = mySum = mySum + sdata[tid +  4];
		}

		__syncthreads();

		if ((blockDim.x >=   4) && (tid <  2))
		{
			sdata[tid] = mySum = mySum + sdata[tid +  2];
		}

		__syncthreads();

		if ((blockDim.x >=   2) && ( tid <  1))
		{
			sdata[tid] = mySum = mySum + sdata[tid +  1];
		}

		__syncthreads();
	#endif

		// write result for this block to global mem
		if (tid == 0) g_odata[s_bid % 448] += mySum;
	}
}


// Reduction Kernel
__global__ void
preemp_SMT_reduce_kernel(float *g_idata, float *g_odata, unsigned int n,
		int SIMD_min,
		int SIMD_max,
		unsigned long int num_subtask,
		int iter_per_subtask,
		int *cont_subtask,
		State *status)
{
	__shared__ int s_bid;

	unsigned int SM_id = get_smid_reduction();
	
	if (SM_id <SIMD_min || SM_id > SIMD_max) /* Only blocks executing within SIMD_min and SIMD_max can progress */ 
			return;
			
	//extern __shared__ float sdata[];
	//float mySum = 0;
		
	while (1){
		
		/********** Task Id calculation *************/
		if (threadIdx.x == 0) {
			if (*status == TOEVICT)
				s_bid = -1;
			else
				s_bid = atomicAdd(cont_subtask, 1);
		}
		
		__syncthreads();
		
		if (s_bid >= num_subtask || s_bid == -1){ /* If all subtasks have been executed */
			return;
		}
		
		extern __shared__ float sdata[];

		// perform first level of reduction,
		// reading from global memory, writing to shared memory
		unsigned int tid = threadIdx.x;
		//unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
		unsigned int i = s_bid * (blockDim.x * 2) + threadIdx.x;
		unsigned int gridSize = blockDim.x * 2 * num_subtask;

		//float mySum = (i < n) ? g_idata[i] : 0;
		float mySum = 0;

		// if (i + blockDim.x < n)
			// mySum += g_idata[i+blockDim.x];
		
		int cont = 0;
		
		while(i < n && cont < iter_per_subtask){
			mySum += g_idata[i];
			
			if(i + blockDim.x < n)
				mySum += g_idata[i+blockDim.x];

			i += gridSize;
			cont++;
		}

		sdata[tid] = mySum;
		__syncthreads();

		// do reduction in shared mem
		if ((blockDim.x >= 512) && (tid < 256))
		{
			sdata[tid] = mySum = mySum + sdata[tid + 256];
		}

		__syncthreads();

		if ((blockDim.x >= 256) &&(tid < 128))
		{
			sdata[tid] = mySum = mySum + sdata[tid + 128];
		}

		 __syncthreads();

		if ((blockDim.x >= 128) && (tid <  64))
		{
		   sdata[tid] = mySum = mySum + sdata[tid +  64];
		}

		__syncthreads();

	#if (__CUDA_ARCH__ >= 300 )
		if ( tid < 32 )
		{
			// Fetch final intermediate sum from 2nd warp
			if (blockDim.x >=  64) mySum += sdata[tid + 32];
			// Reduce final warp using shuffle
			for (int offset = warpSize/2; offset > 0; offset /= 2) 
			{
				mySum += __shfl_down(mySum, offset);
			}
		}
	#else
		// fully unroll reduction within a single warp
		if ((blockDim.x >=  64) && (tid < 32))
		{
			sdata[tid] = mySum = mySum + sdata[tid + 32];
		}

		__syncthreads();

		if ((blockDim.x >=  32) && (tid < 16))
		{
			sdata[tid] = mySum = mySum + sdata[tid + 16];
		}

		__syncthreads();

		if ((blockDim.x >=  16) && (tid <  8))
		{
			sdata[tid] = mySum = mySum + sdata[tid +  8];
		}

		__syncthreads();

		if ((blockDim.x >=   8) && (tid <  4))
		{
			sdata[tid] = mySum = mySum + sdata[tid +  4];
		}

		__syncthreads();

		if ((blockDim.x >=   4) && (tid <  2))
		{
			sdata[tid] = mySum = mySum + sdata[tid +  2];
		}

		__syncthreads();

		if ((blockDim.x >=   2) && ( tid <  1))
		{
			sdata[tid] = mySum = mySum + sdata[tid +  1];
		}

		__syncthreads();
	#endif

		// write result for this block to global mem
		if (tid == 0) g_odata[s_bid % 448] += mySum;
	}
}

// Reduction Kernel
__global__ void
preemp_SMK_reduce_kernel(float *g_idata, float *g_odata, unsigned int n,
		int max_blocks_per_SM, 
		unsigned long int num_subtask,
		int iter_per_subtask,
		int *cont_SM,
		int *cont_subtask,
		State *status)
{
	__shared__ int s_bid, s_index;
	
	unsigned int SM_id = get_smid_reduction();
	
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
			
		extern __shared__ float sdata[];

		// perform first level of reduction,
		// reading from global memory, writing to shared memory
		unsigned int tid = threadIdx.x;
		//unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
		unsigned int i = s_bid*(blockDim.x*2) + threadIdx.x;
		unsigned int gridSize = blockDim.x*2*num_subtask;

		//float mySum = (i < n) ? g_idata[i] : 0;
		float mySum = 0;

		// if (i + blockDim.x < n)
			// mySum += g_idata[i+blockDim.x];
		
		while(i < n){
			mySum += g_idata[i];
			
			if(i + blockDim.x < n)
				mySum += g_idata[i+blockDim.x];

			i += gridSize;
		}

		sdata[tid] = mySum;
		__syncthreads();

		// do reduction in shared mem
		if ((blockDim.x >= 512) && (tid < 256))
		{
			sdata[tid] = mySum = mySum + sdata[tid + 256];
		}

		__syncthreads();

		if ((blockDim.x >= 256) &&(tid < 128))
		{
				sdata[tid] = mySum = mySum + sdata[tid + 128];
		}

		 __syncthreads();

		if ((blockDim.x >= 128) && (tid <  64))
		{
		   sdata[tid] = mySum = mySum + sdata[tid +  64];
		}

		__syncthreads();

	#if (__CUDA_ARCH__ >= 300 )
		if ( tid < 32 )
		{
			// Fetch final intermediate sum from 2nd warp
			if (blockDim.x >=  64) mySum += sdata[tid + 32];
			// Reduce final warp using shuffle
			for (int offset = warpSize/2; offset > 0; offset /= 2) 
			{
				mySum += __shfl_down(mySum, offset);
			}
		}
	#else
		// fully unroll reduction within a single warp
		if ((blockDim.x >=  64) && (tid < 32))
		{
			sdata[tid] = mySum = mySum + sdata[tid + 32];
		}

		__syncthreads();

		if ((blockDim.x >=  32) && (tid < 16))
		{
			sdata[tid] = mySum = mySum + sdata[tid + 16];
		}

		__syncthreads();

		if ((blockDim.x >=  16) && (tid <  8))
		{
			sdata[tid] = mySum = mySum + sdata[tid +  8];
		}

		__syncthreads();

		if ((blockDim.x >=   8) && (tid <  4))
		{
			sdata[tid] = mySum = mySum + sdata[tid +  4];
		}

		__syncthreads();

		if ((blockDim.x >=   4) && (tid <  2))
		{
			sdata[tid] = mySum = mySum + sdata[tid +  2];
		}

		__syncthreads();

		if ((blockDim.x >=   2) && ( tid <  1))
		{
			sdata[tid] = mySum = mySum + sdata[tid +  1];
		}

		__syncthreads();
	#endif

		// write result for this block to global mem
		if (tid == 0) g_odata[s_bid] = mySum;
	}
}

int reduce_start_kernel(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	t_reduction_params * params = (t_reduction_params *)kstub->params;
	
	params->smem_size = (kstub->kconf.blocksize.x <= 32) ? 2 * kstub->kconf.blocksize.x * sizeof(float) : kstub->kconf.blocksize.x * sizeof(float);
	// params->size = 1<<24;
	// params->size *= 50;
	
	unsigned int bytes = params->size * sizeof(float);

	params->h_idata = (float *) malloc(bytes);

	for (int i=0; i<params->size; i++)
	{
		// Keep the numbers small so we don't get truncation error in the sum
		params->h_idata[i] = (rand() & 0xFF);
	}

	// allocate mem for the result on host side
	params->h_odata = (float *) malloc(kstub->kconf.gridsize.x*sizeof(float));

	// allocate device memory and data
	params->d_idata = NULL;
	params->d_odata = NULL;

	checkCudaErrors(cudaMalloc((float **) &params->d_idata, bytes));
	checkCudaErrors(cudaMalloc((float **) &params->d_odata, kstub->kconf.gridsize.x*sizeof(float)));

	return 0;
 }

int reduce_start_mallocs(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	t_reduction_params * params = (t_reduction_params *)kstub->params;
	
	params->smem_size = (kstub->kconf.blocksize.x <= 32) ? 2 * kstub->kconf.blocksize.x * sizeof(float) : kstub->kconf.blocksize.x * sizeof(float);
	// params->size = 1<<24;
	// params->size *= 50;
	
	unsigned int bytes = params->size * sizeof(float);

	params->h_idata = (float *)malloc(bytes);

	for (int i=0; i<params->size; i++)
	{
		// Keep the numbers small so we don't get truncation error in the sum
		params->h_idata[i] = (rand() & 0xFF);
	}

	// allocate mem for the result on host side
	params->h_odata = (float *) malloc(kstub->kconf.gridsize.x*sizeof(float));

	// allocate device memory and data
	params->d_idata = NULL;
	params->d_odata = NULL;

	checkCudaErrors(cudaMalloc((float **) &params->d_idata, bytes));
	checkCudaErrors(cudaMalloc((float **) &params->d_odata, kstub->kconf.gridsize.x*sizeof(float)));

	return 0;
}

int reduce_start_transfers(void *arg){
	
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	t_reduction_params * params = (t_reduction_params *)kstub->params;
	
	unsigned int bytes = params->size * sizeof(float);

#if defined(MEMCPY_ASYNC)	
	checkCudaErrors(cudaMemcpyAsync(params->d_idata, params->h_idata, bytes, cudaMemcpyHostToDevice, kstub->transfer_s[0]));
	checkCudaErrors(cudaMemcpyAsync(params->d_odata, params->h_odata, kstub->kconf.gridsize.x*sizeof(float), cudaMemcpyHostToDevice, kstub->transfer_s[0]));
#else
	checkCudaErrors(cudaMemcpy(params->d_idata, params->h_idata, bytes, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(params->d_odata, params->h_odata, kstub->kconf.gridsize.x*sizeof(float), cudaMemcpyHostToDevice));
#endif
      
	return 0;
}

int reduce_end_kernel(void *arg)
{

	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	t_reduction_params * params = (t_reduction_params *)kstub->params;
	
	unsigned int bytes = params->size * sizeof(float);

#if defined(MEMCPY_ASYNC)
	cudaEventSynchronize(kstub->end_Exec);

	//checkCudaErrors(cudaMemcpyAsync(params->h_idata, params->d_idata, bytes, cudaMemcpyDeviceToHost, kstub->transfer_s[1]));
	checkCudaErrors(cudaMemcpyAsync(params->h_odata, params->d_odata, kstub->kconf.gridsize.x*sizeof(float), cudaMemcpyDeviceToHost, kstub->transfer_s[1]));
#else	
	cudaEventSynchronize(kstub->end_Exec);
	
	//checkCudaErrors(cudaMemcpy(params->h_idata, params->d_idata, bytes, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(params->h_odata, params->d_odata, kstub->kconf.gridsize.x*sizeof(float), cudaMemcpyDeviceToHost));			 		 
#endif

	free(params->h_idata);
	free(params->h_odata);

	checkCudaErrors(cudaFree(params->d_idata));
	checkCudaErrors(cudaFree(params->d_odata));
	
	return 0;
}
	
int launch_orig_reduce(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	t_reduction_params * params = (t_reduction_params *)kstub->params;
	
	reduce<<<kstub->kconf.gridsize.x, kstub->kconf.blocksize.x, params->smem_size>>>
		(params->d_idata, params->d_odata, params->size);			
	
	return 0;
}

int prof_Reduce(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	t_reduction_params * params = (t_reduction_params *)kstub->params;
	
	profiling_reduce_kernel<<<kstub->kconf.numSMs * kstub->kconf.max_persistent_blocks, kstub->kconf.blocksize.x, params->smem_size, *(kstub->execution_s)>>>
		(params->d_idata, params->d_odata, params->size,
		
		kstub->total_tasks,
			kstub->kconf.coarsening,
			kstub->d_SMs_cont,
			kstub->d_executed_tasks,
			&kstub->gm_state[kstub->stream_index]);
			
	return 0;
}

int launch_preemp_reduce(void *arg)
{
	
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	t_reduction_params * params = (t_reduction_params *)kstub->params;

	#ifdef SMT

	preemp_SMT_reduce_kernel<<<kstub->kconf.numSMs * kstub->kconf.max_persistent_blocks, kstub->kconf.blocksize.x, params->smem_size, *(kstub->execution_s)>>>
		(params->d_idata, params->d_odata, params->size,
		kstub->idSMs[0],
		kstub->idSMs[1],
		kstub->total_tasks,
		kstub->kconf.coarsening,
		kstub->d_executed_tasks,
		&(kstub->gm_state[kstub->stream_index])
	);

	#else

	preemp_SMK_reduce_kernel<<<kstub->kconf.numSMs * kstub->kconf.max_persistent_blocks, kstub->kconf.blocksize.x, params->smem_size, *(kstub->execution_s)>>>
		(params->d_idata, params->d_odata, params->size, 
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