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

using namespace std;

#include "../elastic_kernel.h"
#include "HST256.h"

// uchar *h_Data256;
// uint  *h_HistogramCPU256, *h_HistogramGPU256;
// uchar *d_Data256;
// uint  *d_Histogram256;
// uint byteCount256;

static const uint PARTIAL_HISTOGRAM256_COUNT = 240 * 24;
//static uint *d_PartialHistograms256;

extern t_tqueue *tqueues;

 __device__ uint get_smid_HST256(void) {
	uint ret;

	asm("mov.u32 %0, %smid;" : "=r"(ret) );

	return ret;
}

inline __device__ void addByte(uint *s_WarpHist, uint data, uint threadTag)
{
    atomicAdd(s_WarpHist + data, 1);
}

inline __device__ void addWord(uint *s_WarpHist, uint data, uint tag)
{
    addByte(s_WarpHist, (data >>  0) & 0xFFU, tag);
    addByte(s_WarpHist, (data >>  8) & 0xFFU, tag);
    addByte(s_WarpHist, (data >> 16) & 0xFFU, tag);
    addByte(s_WarpHist, (data >> 24) & 0xFFU, tag);
}

/**
 * Histogram (CUDA Kernel)
 */
__global__ void
original_histogram256CUDA(uint *d_PartialHistograms256, uint *d_Data256, uint dataCount)
{
    //Per-warp subhistogram storage
    __shared__ uint s_Hist[HISTOGRAM256_THREADBLOCK_MEMORY];
    uint *s_WarpHist= s_Hist + (threadIdx.x >> LOG2_WARP_SIZE) * HISTOGRAM256_BIN_COUNT;

    //Clear shared memory storage for current threadblock before processing
#pragma unroll

    for (uint i = 0; i < (HISTOGRAM256_THREADBLOCK_MEMORY / HISTOGRAM256_THREADBLOCK_SIZE); i++)
    {
        s_Hist[threadIdx.x + i * HISTOGRAM256_THREADBLOCK_SIZE] = 0;
    }

    //Cycle through the entire data set, update subhistograms for each warp
    const uint tag = threadIdx.x << (UINT_BITS - LOG2_WARP_SIZE);

    __syncthreads();

    for (uint pos = UMAD(blockIdx.x, blockDim.x, threadIdx.x); pos < dataCount; pos += UMUL(blockDim.x, gridDim.x))
    {
        uint data = d_Data256[pos];
        addWord(s_WarpHist, data, tag);
    }

    //Merge per-warp histograms into per-block and write to global memory
    __syncthreads();

    for (uint bin = threadIdx.x; bin < HISTOGRAM256_BIN_COUNT; bin += HISTOGRAM256_THREADBLOCK_SIZE)
    {
        uint sum = 0;

        for (uint i = 0; i < WARP_COUNT; i++)
        {
            sum += s_Hist[bin + i * HISTOGRAM256_BIN_COUNT] & TAG_MASK;
        }

        d_PartialHistograms256[blockIdx.x * HISTOGRAM256_BIN_COUNT + bin] = sum;
    }
}

__global__ void
__launch_bounds__(192, 8)
SMT_histogram256CUDA(uint *d_PartialHistograms256, uint *d_Data256, uint dataCount,
					int SIMD_min, int SIMD_max,
					int num_subtask, int iter_per_subtask, int *cont_subtask, State *status)
{
	__shared__ int s_bid;
	
	unsigned int SM_id = get_smid_HST256();
	
	if (SM_id <SIMD_min || SM_id > SIMD_max) /* Only blocks executing within SIMD_min and SIMD_max can progress */ 
			return;
	
	while (1){
		
		/********** Task Id calculation *************/
		
		if (threadIdx.x == 0) { 
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

		//Per-warp subhistogram storage
		__shared__ uint s_Hist[HISTOGRAM256_THREADBLOCK_MEMORY];
		uint *s_WarpHist= s_Hist + (threadIdx.x >> LOG2_WARP_SIZE) * HISTOGRAM256_BIN_COUNT;

		//Clear shared memory storage for current threadblock before processing
	#pragma unroll

		for (uint i = 0; i < (HISTOGRAM256_THREADBLOCK_MEMORY / HISTOGRAM256_THREADBLOCK_SIZE); i++)
		{
			s_Hist[threadIdx.x + i * HISTOGRAM256_THREADBLOCK_SIZE] = 0;
		}

		//Cycle through the entire data set, update subhistograms for each warp
		const uint tag = threadIdx.x << (UINT_BITS - LOG2_WARP_SIZE);

		__syncthreads();

		for (uint pos = UMAD(s_bid, blockDim.x, threadIdx.x); pos < dataCount; pos += UMUL(blockDim.x, num_subtask))
		{
			uint data = d_Data256[pos];
			addWord(s_WarpHist, data, tag);
		}

		//Merge per-warp histograms into per-block and write to global memory
		__syncthreads();

		for (uint bin = threadIdx.x; bin < HISTOGRAM256_BIN_COUNT; bin += HISTOGRAM256_THREADBLOCK_SIZE)
		{
			uint sum = 0;

			for (uint i = 0; i < WARP_COUNT; i++)
			{
				sum += s_Hist[bin + i * HISTOGRAM256_BIN_COUNT] & TAG_MASK;
			}

			d_PartialHistograms256[s_bid * HISTOGRAM256_BIN_COUNT + bin] = sum;
		}
	}
}

__global__ void
__launch_bounds__(192, 8)
SMK_histogram256CUDA(uint *d_PartialHistograms256, uint *d_Data256, uint dataCount,
					int max_blocks_per_SM,
					int num_subtask,
					int iter_per_subtask,
					int *cont_SM,
					int *cont_subtask,
					State *status
)
{
	__shared__ int s_bid, s_index;
	
	unsigned int SM_id = get_smid_HST256();
	
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
		
		//Per-warp subhistogram storage
		__shared__ uint s_Hist[HISTOGRAM256_THREADBLOCK_MEMORY];
		uint *s_WarpHist= s_Hist + (threadIdx.x >> LOG2_WARP_SIZE) * HISTOGRAM256_BIN_COUNT;

		//Clear shared memory storage for current threadblock before processing
	#pragma unroll

		for (uint i = 0; i < (HISTOGRAM256_THREADBLOCK_MEMORY / HISTOGRAM256_THREADBLOCK_SIZE); i++)
		{
			s_Hist[threadIdx.x + i * HISTOGRAM256_THREADBLOCK_SIZE] = 0;
		}

		//Cycle through the entire data set, update subhistograms for each warp
		const uint tag = threadIdx.x << (UINT_BITS - LOG2_WARP_SIZE);

		__syncthreads();

		for (uint pos = UMAD(s_bid, blockDim.x, threadIdx.x); pos < dataCount; pos += UMUL(blockDim.x, num_subtask))
		{
			uint data = d_Data256[pos];
			addWord(s_WarpHist, data, tag);
		}

		//Merge per-warp histograms into per-block and write to global memory
		__syncthreads();

		for (uint bin = threadIdx.x; bin < HISTOGRAM256_BIN_COUNT; bin += HISTOGRAM256_THREADBLOCK_SIZE)
		{
			uint sum = 0;

			for (uint i = 0; i < WARP_COUNT; i++)
			{
				sum += s_Hist[bin + i * HISTOGRAM256_BIN_COUNT] & TAG_MASK;
			}

			d_PartialHistograms256[s_bid * HISTOGRAM256_BIN_COUNT + bin] = sum;
		}
	}
}

int HST256_start_kernel(void *arg) 
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	t_HST256_params * params = (t_HST256_params *)kstub->params;
	
	//byteCount256 = params->byteCount256;
	//Data set 1 
	//byteCount256 = 64 * 1048576 * 6;
	
	//Data set 2
	//byteCount256 = 64 * 1048576 * 6 * 6;

	// h_Data256         = (uchar *)malloc(byteCount256);
    // h_HistogramCPU256 = (uint *)malloc(HISTOGRAM256_BIN_COUNT * sizeof(uint));
    // h_HistogramGPU256 = (uint *)malloc(HISTOGRAM256_BIN_COUNT * sizeof(uint));
	
	cudaMallocHost(&params->h_Data256, params->byteCount256);
	cudaMallocHost(&params->h_HistogramCPU256, HISTOGRAM256_BIN_COUNT * sizeof(uint));
	cudaMallocHost(&params->h_HistogramGPU256, HISTOGRAM256_BIN_COUNT * sizeof(uint));
	
	srand(2009);

    for (uint i = 0; i < params->byteCount256; i++)
    {
        params->h_Data256[i] = rand() % 256;
    }
	
	checkCudaErrors(cudaMalloc((void **)&params->d_Data256, params->byteCount256));
    checkCudaErrors(cudaMalloc((void **)&params->d_Histogram256, HISTOGRAM256_BIN_COUNT * sizeof(uint)));
    checkCudaErrors(cudaMemcpy(params->d_Data256, params->h_Data256, params->byteCount256, cudaMemcpyHostToDevice));
	
	checkCudaErrors(cudaMalloc((void **)&params->d_PartialHistograms256, PARTIAL_HISTOGRAM256_COUNT * HISTOGRAM256_BIN_COUNT * sizeof(uint)));
	
	return 0;
}

int HST256_start_mallocs(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	t_HST256_params * params = (t_HST256_params *)kstub->params;
	
	//byteCount256 = params->byteCount256;
	
	//Data set 1 
	//byteCount256 = 64 * 1048576 * 6;
	
	//Data set 2
	//byteCount256 = 64 * 1048576 * 6 * 6;
	
#if defined(MEMCPY_SYNC) || defined(MEMCPY_ASYNC)
	cudaMallocHost(&params->h_Data256, params->byteCount256);
	cudaMallocHost(&params->h_HistogramCPU256, HISTOGRAM256_BIN_COUNT * sizeof(uint));
	cudaMallocHost(&params->h_HistogramGPU256, HISTOGRAM256_BIN_COUNT * sizeof(uint));

	checkCudaErrors(cudaMalloc((void **)&params->d_Data256, params->byteCount256));
    checkCudaErrors(cudaMalloc((void **)&params->d_Histogram256, HISTOGRAM256_BIN_COUNT * sizeof(uint)));	
	checkCudaErrors(cudaMalloc((void **)&params->d_PartialHistograms256, PARTIAL_HISTOGRAM256_COUNT * HISTOGRAM256_BIN_COUNT * sizeof(uint)));
#else
	#ifdef MANAGED_MEM

	cudaMallocManaged(&params->h_Data256, params->byteCount256);
	cudaMallocManaged(&params->h_HistogramCPU256, HISTOGRAM256_BIN_COUNT * sizeof(uint));
	cudaMallocManaged(&params->h_HistogramGPU256, HISTOGRAM256_BIN_COUNT * sizeof(uint));
	
	srand(2009);

    for (uint i = 0; i < params->byteCount256; i++)
    {
        params->h_Data256[i] = rand() % 256;
    }
	
	params->d_Data256 = params->h_Data256;
	#else
		printf("No transfer model: Exiting ...\n");
		exit(-1);
	#endif
#endif

	// Verify that allocations succeeded
    if (params->h_Data256 == NULL || params->h_HistogramCPU256 == NULL || params->h_HistogramGPU256 == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    srand(2009);

    for (uint i = 0; i < params->byteCount256; i++)
    {
        params->h_Data256[i] = rand() % 256;
    }

	return 0;
}

int HST256_start_transfers(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	t_HST256_params * params = (t_HST256_params *)kstub->params;
	
	//byteCount256 = params->byteCount256;
	
	//Data set 1 
	//byteCount256 = 64 * 1048576 * 6;
	
	//Data set 2
	//byteCount256 = 64 * 1048576 * 6 * 6;
	
#ifdef MEMCPY_SYNC
	enqueue_tcomamnd(tqueues, params->d_Data256, params->h_Data256, params->byteCount256, cudaMemcpyHostToDevice, 0, BLOCKING, DATA, LOW, kstub);
	kstub->HtD_tranfers_finished = 1;

	
#else
	
	#ifdef MEMCPY_ASYNC
	
	//enqueue_tcomamnd(tqueues, d_Data256, h_Data256, byteCount256, cudaMemcpyHostToDevice, 0, NONBLOCKING, LAST_TRANSFER, MEDIUM, kstub);
	cudaMemcpyAsync(params->d_Data256, params->h_Data256, params->byteCount256, cudaMemcpyHostToDevice, kstub->transfer_s[0]);

	#else
	#ifdef MANAGED_MEM

	cudaDeviceProp p;
    cudaGetDeviceProperties(&p, kstub->deviceId);
	
	if (p.concurrentManagedAccess)
	{
		err = cudaMemPrefetchAsync(params->h_Data256, params->byteCount256, kstub->deviceId);
		if ( err != cudaSuccess) {
			printf("Error in vAdd:cudaMemPrefetchAsync\n");
			exit(EXIT_FAILURE);
		}
		err = cudaMemPrefetchAsync(params->h_HistogramCPU256, HISTOGRAM256_BIN_COUNT * sizeof(uint), kstub->deviceId);
		if ( err != cudaSuccess) {
			printf("Error in vAdd:cudaMemPrefetchAsync\n");
			exit(EXIT_FAILURE);
		}
		err = cudaMemPrefetchAsync(params->h_HistogramGPU256, HISTOGRAM256_BIN_COUNT * sizeof(uint), kstub->deviceId);
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

int HST256_end_kernel(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	t_HST256_params * params = (t_HST256_params *)kstub->params;
	
#ifdef MEMCPY_SYNC

	cudaEventSynchronize(kstub->end_Exec);

	enqueue_tcomamnd(tqueues, params->h_HistogramGPU256, params->d_Histogram256, HISTOGRAM256_BIN_COUNT * sizeof(uint), cudaMemcpyDeviceToHost, 0, BLOCKING, DATA, LOW, kstub);
	 
#else
	#ifdef MEMCPY_ASYNC
	//enqueue_tcomamnd(tqueues, h_HistogramGPU256, d_Histogram256, HISTOGRAM256_BIN_COUNT * sizeof(uint), cudaMemcpyDeviceToHost, kstub->transfer_s[1] , NONBLOCKING, LAST_TRANSFER, MEDIUM, kstub);
	cudaMemcpyAsync(params->h_HistogramGPU256, params->d_Histogram256, HISTOGRAM256_BIN_COUNT * sizeof(uint), cudaMemcpyDeviceToHost, kstub->transfer_s[1]);
	
	#else
		#ifdef MANAGED_MEM
			cudaStreamSynchronize(*(kstub->execution_s)); // To be sure kernel execution has finished before processing output data
		#endif
	#endif
#endif

	return 0;
}

/*int HST256_end_kernel_dummy(void *arg)
{	
	checkCudaErrors(cudaMemcpy(h_HistogramGPU256, d_Histogram256, HISTOGRAM256_BIN_COUNT * sizeof(uint), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_PartialHistograms256));
	checkCudaErrors(cudaFree(d_Histogram256));
    checkCudaErrors(cudaFree(d_Data256));
    // free(h_HistogramGPU256);
    // free(h_HistogramCPU256);
    // free(h_Data256);
	
	cudaFreeHost(h_HistogramGPU256);
    cudaFreeHost(h_HistogramCPU256);
    cudaFreeHost(h_Data256);
	
    return 0;
}*/
 
int launch_orig_HST256(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	t_HST256_params * params = (t_HST256_params *)kstub->params;
	
	original_histogram256CUDA<<<kstub->kconf.gridsize.x, kstub->kconf.blocksize.x>>>(
        params->d_PartialHistograms256,
        (uint *)params->d_Data256,
        params->byteCount256 / sizeof(uint)
    );

	return 0;
}

int launch_preemp_HST256(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	t_HST256_params * params = (t_HST256_params *)kstub->params;
	
	#ifdef SMT
		SMT_histogram256CUDA<<<kstub->kconf.numSMs * kstub->kconf.max_persistent_blocks, kstub->kconf.blocksize.x, 0, *(kstub->execution_s)>>>(
			params->d_PartialHistograms256,
			(uint *)params->d_Data256,
			params->byteCount256 / sizeof(uint),
			
			kstub->idSMs[0],
			kstub->idSMs[1],
			kstub->total_tasks,
			kstub->kconf.coarsening,
			kstub->d_executed_tasks,
			&(kstub->gm_state[kstub->stream_index])
		);
	#else
		SMK_histogram256CUDA<<<kstub->kconf.numSMs * kstub->kconf.max_persistent_blocks, kstub->kconf.blocksize.x, 0, *(kstub->execution_s)>>>(
			params->d_PartialHistograms256,
			(uint *)params->d_Data256,
			params->byteCount256 / sizeof(uint),
			
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