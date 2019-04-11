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
#include "reduction_original_kernel.h" // Warning: needs scheduler.h because State definition

extern t_tqueue *tqueues;

// ****************************************************************************
// Function: reduceCPU
//
// Purpose:
//   Simple cpu reduce routine to verify device results
//
// Arguments:
//   data : the input data
//   size : size of the input data
//
// Returns:  sum of the data
//
// Programmer: Kyle Spafford
// Creation: August 13, 2009
//
// Modifications:
//
// ****************************************************************************
template <class T>
T reduceCPU(const T *data, unsigned long int size)
{
    T sum = 0;
    for (unsigned long int i = 0; i < size; i++)
    {
        sum += data[i];
    }
    return sum;
}


//*************************************************/
// Memory Allocation
//*************************************************/

float *h_idata, *h_odata;
float *d_idata, *d_odata;

unsigned long int nItems = 32*32*1024*1024; // Maximum size without overflow
int nBlocks = 64;
int blockSize = 256;
int smem_size = blockSize * sizeof(float);

int reduce_start_kernel(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	
	nBlocks = kstub->kconf.gridsize.x;
	blockSize =	kstub->kconf.blocksize.x;
    nItems = nBlocks*blockSize*2;
	smem_size = blockSize * sizeof(float);
	
	// Allocate and set up host data (assuming T data is float)
	CUDA_SAFE_CALL(cudaMallocHost((void**)&h_idata, nItems * sizeof(float)));
    CUDA_SAFE_CALL(cudaMallocHost((void**)&h_odata, nBlocks * sizeof(float)));

	for(long int i = 0; i < nItems; i++)
    {
        h_idata[i] = ((float) (i % 2)/100000); //Fill with some pattern
    }

	// Allocate device memory
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_idata, nItems * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_odata, nBlocks * sizeof(float)));
	
	// Transfer data from host to device
	CUDA_SAFE_CALL(cudaMemcpy(d_idata, h_idata, nItems*sizeof(float), cudaMemcpyHostToDevice));

  
	return 0;
 }

int reduce_start_mallocs(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	
	nBlocks = kstub->kconf.gridsize.x;
	blockSize =	kstub->kconf.blocksize.x;
    nItems = kstub->total_tasks*blockSize*2;
	smem_size = blockSize * sizeof(float);

	//printf("\tWorking with %d items, reduced with %d (%d) blocks\n", nItems, nBlocks, kstub->kconf.coarsening);

	// Allocate and set up host data (assuming T data is float)
	CUDA_SAFE_CALL(cudaMallocHost((void**)&h_idata, nItems * sizeof(float)));
    CUDA_SAFE_CALL(cudaMallocHost((void**)&h_odata, nBlocks * sizeof(float)));

	for(unsigned long int i = 0; i < nItems; i++)
    {
        h_idata[i] = ((float) (i % 2)/100000); //Fill with some pattern
        //h_idata[i] = (float) (i % 2); //Fill with some pattern
    }

	// Allocate device memory
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_idata, nItems * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_odata, nBlocks * sizeof(float)));

	return 0;
}

//*************************************************/
// HtD Transfers
//*************************************************/

int reduce_start_transfers(void *arg){
	
	t_kernel_stub *kstub = (t_kernel_stub *)arg;

#if defined(MEMCPY_ASYNC)

	//enqueue_tcomamnd(tqueues, d_idata, h_idata, nItems * sizeof(float), cudaMemcpyHostToDevice, 
	// kstub->transfer_s[0], NONBLOCKING, LAST_TRANSFER, MEDIUM, kstub);
	cudaMemcpyAsync(d_idata, h_idata, nItems * sizeof(float), cudaMemcpyHostToDevice, kstub->transfer_s[0]);
	
#else

	CUDA_SAFE_CALL(cudaMemcpy(d_idata, h_idata,   nItems * sizeof(float),
              cudaMemcpyHostToDevice));

#endif
      
	return 0;
}

//*************************************************/
// DtH transfers and deallocation
//*************************************************/

int reduce_end_kernel(void *arg)
{

	t_kernel_stub *kstub = (t_kernel_stub *)arg;

#if defined(MEMCPY_ASYNC)
	cudaEventSynchronize(kstub->end_Exec);

	//enqueue_tcomamnd(tqueues, h_odata, d_odata, nBlocks * sizeof(float), cudaMemcpyDeviceToHost, kstub->transfer_s[1] , NONBLOCKING, LAST_TRANSFER, MEDIUM, kstub);
	cudaMemcpyAsync(h_odata, d_odata, nBlocks * sizeof(float), cudaMemcpyDeviceToHost, kstub->transfer_s[1]);
#else
	
	cudaEventSynchronize(kstub->end_Exec);
	
	CUDA_SAFE_CALL(cudaMemcpy(h_odata, d_odata, nBlocks * sizeof(float),
                  cudaMemcpyDeviceToHost));
				 		 
	CUDA_SAFE_CALL(cudaFree(d_idata));
	CUDA_SAFE_CALL(cudaFree(d_odata));

#endif

	// Compute results on CPU
	// cudaDeviceSynchronize();
	// float dev_result = 0;
	// for (int i=0; i<nBlocks; i++)
	// {
		// dev_result += h_odata[i];
	// }

	// // compute reference solution
	// float cpu_result = reduceCPU<float>(h_idata, nItems);
	// double threshold = 1.0e-2;
	// float diff = fabs(dev_result - cpu_result);

	// if (diff < threshold)
		// printf("Test passed\n");
	// else
		// printf("Test failed: %f (%f - %f)\n", diff, dev_result, cpu_result);

	cudaFree(h_idata);
	cudaFree(h_odata);
	
	return 0;
}
	

int launch_orig_reduce(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;

	dim3 threads = dim3(blockSize, 1, 1);
	dim3 blocks = dim3(nBlocks, 1, 1);
	
	reduce<<<blocks, threads, smem_size>>>
                (d_idata, d_odata, nItems);			
	
	return 0;
}

int launch_preemp_reduce(void *arg)
{
	
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	dim3 threads = dim3(kstub->kconf.blocksize.x, 1, 1);
	dim3 blocks = dim3(kstub->kconf.numSMs * kstub->kconf.max_persistent_blocks, 1, 1);

	#ifdef SMT

	preemp_SMT_reduce_kernel<<<blocks, threads, smem_size, *(kstub->execution_s)>>>
			(d_idata, d_odata, nItems,
			kstub->idSMs[0],
			kstub->idSMs[1],
			kstub->total_tasks,
			kstub->kconf.coarsening,
			kstub->d_executed_tasks,
			&(kstub->gm_state[kstub->stream_index])
	);

	#else

	preemp_SMK_reduce_kernel<<<blocks, threads, smem_size, *(kstub->execution_s)>>>
			(d_idata, d_odata, nItems, 
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

// A testing routine
int reduce_test(t_Kernel *kernel_id, float *tHtD, float *tK, float *tDtH)
{
	
	int nsteps = 2;
	float t1 = 0, t2 = 0, t3 = 0;
	int devId = 5;
	
	cudaSetDevice(devId);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, devId);
	printf("Device=%s\n", deviceProp.name);
	
	t_kernel_stub **kstub = (t_kernel_stub **)calloc(1, sizeof(t_kernel_stub *));

	create_stubinfo(&kstub[0], devId, kernel_id[0], 0, 0);
	void *arg = kstub[0];
	

	cudaEvent_t profileStart1, profileEnd1, profileStart2, profileEnd2, profileStart3, profileEnd3;
	CUDA_SAFE_CALL(cudaEventCreate(&profileStart1));
    CUDA_SAFE_CALL(cudaEventCreate(&profileEnd1));
	CUDA_SAFE_CALL(cudaEventCreate(&profileStart2));
    CUDA_SAFE_CALL(cudaEventCreate(&profileEnd2));
	CUDA_SAFE_CALL(cudaEventCreate(&profileStart3));
    CUDA_SAFE_CALL(cudaEventCreate(&profileEnd3));

	tHtD[0] = 0;
	tK[0] = 0;
	tDtH[0] = 0;

    // Enqueue start event
	for ( int j = 0; j < nsteps; j++ )
	{
		reduce_start_mallocs(arg);

		cudaDeviceSynchronize();
		checkCudaErrors(cudaEventRecord(profileStart1, 0));
		// Transfer data from host to device
		CUDA_SAFE_CALL(cudaMemcpy(d_idata, h_idata,   nItems * sizeof(float), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaEventRecord(profileEnd1, 0));

	    // Wait for the kernel to complete
		cudaDeviceSynchronize();
        checkCudaErrors(cudaEventElapsedTime(&t1, profileStart1, profileEnd1));
		tHtD[0] += t1;
		
		checkCudaErrors(cudaEventRecord(profileStart2, 0));
		launch_orig_reduce(arg);
		checkCudaErrors(cudaEventRecord(profileEnd2, 0));

		cudaDeviceSynchronize();
        checkCudaErrors(cudaEventElapsedTime(&t2, profileStart2, profileEnd2));
		tK[0] += t2;

		checkCudaErrors(cudaEventRecord(profileStart3, 0));
		CUDA_SAFE_CALL(cudaMemcpy(h_odata, d_odata, nBlocks * sizeof(float), cudaMemcpyDeviceToHost));	
		checkCudaErrors(cudaEventRecord(profileEnd3, 0));

		cudaDeviceSynchronize();
        checkCudaErrors(cudaEventElapsedTime(&t3, profileStart3, profileEnd3));
		tDtH[0] += t3;
		
		// Compute results on CPU
		unsigned long int i = 0;
		float dev_result = 0;
		for (i=0; i<nBlocks; i++)
		{
			dev_result += h_odata[i];
		}
		
		float cpu_result = 0;
		for (i = 0; i < nItems; i++ )
			cpu_result += h_idata[i];
		//reduceCPU<float>(h_idata, nItems);
		double threshold = 5.0e-2;
		float diff = fabs(dev_result - cpu_result)/dev_result;

		if (diff < threshold)
			printf("Test passed\n");
		else
			printf("Test failed: %d items - %f (%f - %f)\n", i, diff, dev_result, cpu_result);

		cudaFree(h_idata);
		cudaFree(h_odata);
	}

	tHtD[0] /= nsteps;
	tK[0] /= nsteps;
	tDtH[0] /= nsteps;

	return 0;
}
