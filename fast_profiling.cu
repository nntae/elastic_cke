#include <unistd.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <helper_functions.h>   // helper functions for string parsing
#include <helper_cuda.h>   
#include "elastic_kernel.h"

int fast_profiling(int deviceId, t_Kernel kid)
{
	cudaError_t err;
	struct timespec now;
	double init_time, prev_time, curr_time, sample_time;
	
	int cont_tasks[10000];

	// Select device
	cudaSetDevice(deviceId);
	cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);	
	printf("Device=%s\n", deviceProp.name);

	/** Create commom streams for all kernels: two for asynchronous transfers, one for preemption commands*/
	cudaStream_t *transfers_s;
	transfers_s = (cudaStream_t *)calloc(2, sizeof(cudaStream_t));
	
	
	for (int i=0;i<2;i++){
		err = cudaStreamCreate(&transfers_s[i]);
		checkCudaErrors(err);
	} 
	
	cudaStream_t preemp_s;
	checkCudaErrors(cudaStreamCreateWithFlags(&preemp_s, cudaStreamNonBlocking)); 
	
	/** Create stub ***/
	t_kernel_stub *kstub=NULL;
	create_stubinfo(&kstub, deviceId, kid, transfers_s, &preemp_s);
	
	// Make allocation and HtD transfer for applications
	make_transfers(&kstub, 1);
	
	// Create streams kernel info for coexecution
	t_kstreams *kstr = (t_kstreams *)calloc(1, sizeof(t_kstreams));
	create_kstreams(kstub, kstr);
	
	// Coxecution info	
	t_kcoexec coexec;
	create_coexec(&coexec, 2);
	
	// Create sched structure
	
	t_sched sched;
	create_sched(&sched);
	
	// Launch proxy
	launch_generic_proxy((void *)&sched);	// Launch proxy
	
	double interval = 0.000015;  // in seconds
	
	// Get executed tasks during interval
	int curr_executed_tasks; // Read zero-copy variables
	int flag = 0;
	int cont_us = 0;
	memset(cont_tasks, 0, sizeof(int)*10000);
	
	// Select initial kernel
	int task_index = 0; // Index of the kernel in the array with ready kernels;
	//k_done[task_index] = 1; // Kernel removed from pending kernels*/
	 
	// add one stream (one CTA per SM) to kernel (index= 0 in ready list)
	add_kernel_for_coexecution(&coexec, &sched, &kstr[task_index], 1, 0);
	
	// Execute kernels (launching streams) in coexec structure
	launch_coexec(&coexec);		
	
	clock_gettime(CLOCK_REALTIME, &now);
 	prev_time = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
	init_time = prev_time;

	do {
		
			clock_gettime(CLOCK_REALTIME, &now);
			curr_time = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
			int dif = (int)((curr_time - init_time) * 1000000.0); // in us
			if (dif > cont_us){
				cont_us = dif;
				cont_tasks[cont_us] = *(sched.cont_tasks_zc);
			}
				
			if ((curr_time-prev_time) > 0.2 * interval && flag == 0) { // Fisr sample is taken when a minumum time has elapsed
				curr_executed_tasks= *(sched.cont_tasks_zc);
				sample_time = curr_time;
				flag = 1;
			}
	} while (curr_time-prev_time < interval);
	
	double tpms = (double)(*(sched.cont_tasks_zc)-curr_executed_tasks)/((curr_time-sample_time)*1000);
	printf("Executed task for CTA=1 tpms=%f\n", tpms);
	
	// Do it when more CTAs are added
	for (int cta=2; cta <= 8; cta++) {
		add_streams_to_kernel(&coexec, &sched, coexec.kstr[0], 1);
		launch_coexec(&coexec);
		
		clock_gettime(CLOCK_REALTIME, &now);
		prev_time = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
		/*sample_time = prev_time;
		
		curr_executed_tasks = *(sched.cont_tasks_zc); // Read zero-copy variables*/
		flag = 0;
		do {
			clock_gettime(CLOCK_REALTIME, &now);
			curr_time = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
			
			int dif = (int)((curr_time - init_time) * 1000000.0); // in us
			if (dif > cont_us){
				cont_us = dif;
				cont_tasks[cont_us] = *(sched.cont_tasks_zc);
			}
			
			if ((curr_time-prev_time) > 0.2 * interval && flag == 0) {
				curr_executed_tasks= *(sched.cont_tasks_zc);
				sample_time = curr_time;
				flag = 1;
			}
		} while (curr_time-prev_time < interval);
		
		tpms = (double)(*(sched.cont_tasks_zc)-curr_executed_tasks)/((curr_time-sample_time)*1000);
		printf("Executed task for CTA=%d tpms=%f\n", cta, tpms);
		prev_time = curr_time;
	}
	
	// Evict proxy
	sched.kernel_evict_zc[0] =  PROXY_EVICT;
	cudaDeviceSynchronize();
	
	/*for (int i=0; i<=cont_us; i++)
		printf("%d,", cont_tasks[i]);
	printf("\n");
		*/
	return 0;
}
	
		

	
	
	
	
	
	