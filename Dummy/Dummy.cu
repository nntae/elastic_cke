#include "../cudacommon.h"
#include <cassert>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <iostream>
#include <semaphore.h>
#include "../elastic_kernel.h" // Warning: needs scheduler.h because State definition
#include "Dummy.h"
#include "Dummy_kernel.h" 

extern t_tqueue *tqueues; // For testing purposes

// A dummy kernel: cpu version
void dummy_cpu(const float *idata, float *odata, const unsigned long int n, const int iter_per_subtask)
{

	for ( unsigned long int i = 0; i < n; i++ )
	{
		odata[i] = 1;
		for ( int iter = 0; iter < iter_per_subtask; iter++ )
			odata[i] *= idata[i];
	}
}

//*************************************************/
// Memory Allocation
//*************************************************/

static float *h_idata, *h_odata;
static float *d_idata, *d_odata;
static unsigned long int nItems = 32*32*1024*1024;

// A dummy kernel: start routine, deprecated
int dummy_start_kernel(void *arg)
{

	dummy_start_mallocs(arg);
	
	// Transfer data from host to device
	CUDA_SAFE_CALL(cudaMemcpy(d_idata, h_idata, nItems*sizeof(float), cudaMemcpyHostToDevice));
  
	return 0;
 }

// A dummy kernel: memory allocation in host and device
int dummy_start_mallocs(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	
    nItems = kstub->total_tasks*kstub->kconf.blocksize.x*kstub->kconf.blocksize.y;
	
	printf("Working with %d items\n", nItems);
	// Allocate and set up host data
	CUDA_SAFE_CALL(cudaMallocHost((void**)&h_idata, nItems * sizeof(float)));
    CUDA_SAFE_CALL(cudaMallocHost((void**)&h_odata, nItems * sizeof(float)));

	for (unsigned long int i = 0; i < nItems; i++)
    {
        h_idata[i] = ((float) (i)/nItems); //Fill with some pattern
		h_odata[i] = 1;
    }

	// Allocate device memory
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_idata, nItems * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_odata, nItems * sizeof(float)));

	return 0;
}

// A dummy kernel: HtD Transfers

int dummy_start_transfers(void *arg){
	
	t_kernel_stub *kstub = (t_kernel_stub *)arg;

#if defined(MEMCPY_ASYNC)

	enqueue_tcomamnd(tqueues, d_idata, h_idata, nItems * sizeof(float), cudaMemcpyHostToDevice, 
						kstub->transfer_s[0], NONBLOCKING, LAST_TRANSFER, MEDIUM, kstub);
	
#else

	CUDA_SAFE_CALL(cudaMemcpy(d_idata, h_idata,   nItems * sizeof(float),
              cudaMemcpyHostToDevice));

#endif

	kstub->HtD_tranfers_finished = 1;
      
	return 0;
}

// A dummy kernel: DtH transfers and deallocation

int dummy_end_kernel(void *arg)
{

	t_kernel_stub *kstub = (t_kernel_stub *)arg;

#if defined(MEMCPY_ASYNC)
	cudaEventSynchronize(kstub->end_Exec);

	enqueue_tcomamnd(tqueues, h_odata, d_odata, nItems * sizeof(float), cudaMemcpyDeviceToHost, kstub->transfer_s[1] , NONBLOCKING, LAST_TRANSFER, MEDIUM, kstub);

#else
	
	cudaEventSynchronize(kstub->end_Exec);
	
	CUDA_SAFE_CALL(cudaMemcpy(h_odata, d_odata, nItems * sizeof(float),
                  cudaMemcpyDeviceToHost));
				 		 
	CUDA_SAFE_CALL(cudaFree(d_idata));
	CUDA_SAFE_CALL(cudaFree(d_odata));

#endif

	// Compute results on CPU
	cudaDeviceSynchronize();
	
	// float *cpu_odata = 0;
	// cpu_odata = (float *) malloc( nItems*sizeof(float) );	
	// dummy_cpu(h_idata, cpu_odata, nItems, kstub->kconf.coarsening);
	// float diff = 0, v1 = 0, v2 = 0;
	// for (unsigned long int i = 0; i < nItems; i++)
	// {
		// diff += fabs(h_odata[i]-cpu_odata[i]);
		// v1 += h_odata[i];
		// v2 += cpu_odata[i];
	// }

	// float threshold = 1.0e-4;

	// if (diff < threshold)
		// printf("Test passed\n");
	// else
		// printf("Test failed: %f\n", diff);

	cudaFree(h_idata);
	cudaFree(h_odata);
	
	return 0;
}
	
// A dummy kernel: launch original kernel using total_tasks blocks
int launch_orig_dummy(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;

	dim3 threads = dim3(kstub->kconf.blocksize.x, kstub->kconf.blocksize.y, 1);
	dim3 blocks = dim3(kstub->total_tasks, 1, 1);

	dummy_kernel<<<blocks, threads, 0, 0>>>(d_idata, d_odata, nItems, kstub->kconf.coarsening);
	
	return 0;
}

// A dummy kernel: launch preemption versions
int launch_preemp_dummy(void *arg)
{
	
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	dim3 threads = dim3(kstub->kconf.blocksize.x, kstub->kconf.blocksize.y, 1);
	dim3 blocks = dim3(kstub->kconf.numSMs * kstub->kconf.max_persistent_blocks, 1, 1);
	
	#ifdef SMT

	preemp_SMT_dummy_kernel<<<blocks, threads, 0, *(kstub->execution_s)>>>
			(d_idata, d_odata, nItems,
			kstub->idSMs[0],
			kstub->idSMs[1],
			kstub->total_tasks,
			kstub->kconf.coarsening,
			kstub->d_executed_tasks,
			&(kstub->gm_state[kstub->stream_index])
	);

	#else

	preemp_SMK_dummy_kernel<<<blocks, threads, 0, *(kstub->execution_s)>>>
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

// A dummy kernel: testing routine
int dummy_test(t_Kernel *kernel_id, float *tHtD, float *tK, float *tDtH)
{
	
	int nsteps = 5; // Number of tests
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
		dummy_start_mallocs(arg);

		cudaDeviceSynchronize();
		checkCudaErrors(cudaEventRecord(profileStart1, 0));
		// Transfer data from host to device
		CUDA_SAFE_CALL(cudaMemcpy(d_idata, h_idata, nItems * sizeof(float), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaEventRecord(profileEnd1, 0));

	    // Wait for the kernel to complete
		cudaDeviceSynchronize();
        checkCudaErrors(cudaEventElapsedTime(&t1, profileStart1, profileEnd1));
		tHtD[0] += t1;
		
		checkCudaErrors(cudaEventRecord(profileStart2, 0));
		launch_orig_dummy(arg);
		checkCudaErrors(cudaEventRecord(profileEnd2, 0));

		cudaDeviceSynchronize();
        checkCudaErrors(cudaEventElapsedTime(&t2, profileStart2, profileEnd2));
		tK[0] += t2;

		checkCudaErrors(cudaEventRecord(profileStart3, 0));
		CUDA_SAFE_CALL(cudaMemcpy(h_odata, d_odata, nItems * sizeof(float), cudaMemcpyDeviceToHost));	
		checkCudaErrors(cudaEventRecord(profileEnd3, 0));

		cudaDeviceSynchronize();
        checkCudaErrors(cudaEventElapsedTime(&t3, profileStart3, profileEnd3));
		tDtH[0] += t3;
		
		CUDA_SAFE_CALL(cudaFree(d_idata));
		CUDA_SAFE_CALL(cudaFree(d_odata));
		// Compute results on CPU
		// float diff = 0, v1 = 0, v2 = 0;
		// float *cpu_odata = 0;
		// cpu_odata = (float *) malloc( nItems*sizeof(float) );	
		// dummy_cpu(h_idata, cpu_odata, nItems, kstub[0]->kconf.coarsening);
		// for (unsigned long int i = 0; i < nItems; i++)
		// {
			// diff += fabs(h_odata[i]-cpu_odata[i]);
			// v1 += h_odata[i];
			// v2 += cpu_odata[i];
		// }

		// float threshold = 1.0e-4;

		// if (diff < threshold)
			// printf("Test passed\n");
		// else
			// printf("Test failed: %f, %f - %f\n", diff, v1, v2);

		cudaFree(h_idata);
		cudaFree(h_odata);
	}

	tHtD[0] /= nsteps;
	tK[0] /= nsteps;
	tDtH[0] /= nsteps;

	return 0;
}
