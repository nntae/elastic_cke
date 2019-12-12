#include <unistd.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <helper_functions.h>   // helper functions for string parsing
#include <helper_cuda.h>   
#include "elastic_kernel.h"

// Table to store coexecution configurations
extern t_smk_coBlocks smk_info_coBlocks[Number_of_Kernels-1][Number_of_Kernels-1];

t_smk_coBlocks *create_coBlocks_entry(int num_configs)
{
	t_smk_coBlocks *info = (t_smk_coBlocks *)calloc(1, sizeof(t_smk_coBlocks));

	info->num_configs = num_configs;
	
	info->pairs = (int **) calloc(info->num_configs, sizeof(int *));
	info->tpms = (double **) calloc(info->num_configs, sizeof(double *));
	for (int i=0; i < info->num_configs; i++) {
		info->pairs[i] = (int *)calloc(2, sizeof(int));
		info->tpms[i] = (double *)calloc(2, sizeof(double));
	}
	
	return info;
}

int free_coBlocks_entry(t_smk_coBlocks *info)
{
	
	for (int i=0; i < info->num_configs; i++) {
		free(info->pairs[i]);
		free(info->tpms[i]);
	}
	
	free(info->pairs);
	free(info->tpms);
	
	free(info);
	
	return 0;
}




int fast_cke_profiling(int deviceId, t_Kernel *kid)
{
	cudaError_t err;
	struct timespec now;
	double  curr_time, init_time, prof_start, prof_end;
	
	// Load SMK initial tables
	smk_fill_coBlocks();
	smk_fill_solo();

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
	
	/** Create stubs ***/
	t_kernel_stub *kstub0=NULL, *kstub1=NULL;
	create_stubinfo(&kstub0, deviceId, kid[0], transfers_s, &preemp_s);
	create_stubinfo(&kstub1, deviceId, kid[1], transfers_s, &preemp_s);
	
	//Copy coexecution configs so that first kernel start with the minimum BSU
	
	t_smk_coBlocks *p = &smk_info_coBlocks[kstub0->id][kstub1->id]; // temporal pointer to table entry
	
	t_smk_coBlocks *info_coexec = create_coBlocks_entry(p->num_configs);
	info_coexec->kid[0] = p->kid[0]; info_coexec->kid[1] = p->kid[1];	
	if (p->pairs[0][0] > p->pairs[info_coexec->num_configs-1][0]) // Copy in reverse order
		for (int i=0; i < info_coexec->num_configs; i++) {
			info_coexec->pairs[i][0] = p->pairs[p->num_configs-1-i][0];
			info_coexec->pairs[i][1] = p->pairs[p->num_configs-1-i][1];
		}
	else
		for (int i=0; i < info_coexec->num_configs; i++) {
			info_coexec->pairs[i][0] = p->pairs[i][0];
			info_coexec->pairs[i][1] = p->pairs[i][1];
		}
		
	
	// Make allocation and HtD transfer for applications
	make_transfers(&kstub0, 1);
	make_transfers(&kstub1, 1);
	
	// Create streams kernel info for coexecution
	t_kstreams *kstr = (t_kstreams *)calloc(2, sizeof(t_kstreams));
	create_kstreams(kstub0, &kstr[0]);
	create_kstreams(kstub1, &kstr[1]);
	
	// Coxecution info	
	t_kcoexec coexec;
	create_coexec(&coexec, 2);
	
	// Create sched structure
	
	t_sched sched;
	create_sched(&sched);
	
	// Launch proxy
	launch_generic_proxy((void *)&sched);	// Launch proxy
	
	// Select initial kernel
	// Index of the kernel in the array with ready kernels;
	//k_done[task_index] = 1; // Kernel removed from pending kernels*/
	 
	// Initial BSUs (info_coexec->pairs[0][0] for first kernel and info_coexec->pairs[0][1] for second kernel)
	if (add_kernel_for_coexecution(&coexec, &sched, &kstr[0], info_coexec->pairs[0][0], 0) < 0)
		return -1;
	if (add_kernel_for_coexecution(&coexec, &sched, &kstr[1], info_coexec->pairs[0][1], 1) < 0)
		return -1;
	
	double start_interval = 0.000200;  // When both kernel are launched (8 BSUs) wait until all streams are running
	double intra_interval = 0.000040; //0.000040; // When a BSU is launched (1 BSU) until stream starts 
	double sampling_interval = 0.000100; //0.000100; // Interval between executed task samples
	double prev_tpms0, prev_tpms1;
	int flag = 1, save_partition=1;
	for (int conf =1; conf <= info_coexec->num_configs; conf++) {
		
		// Select the correct interval
		double interval;	
		if (conf ==1 )
			interval = start_interval;
		else
			interval = intra_interval;
	
		// Execute kernels (launching streams) in coexec structure
		launch_coexec(&coexec);		
		
		// Wait before starting sampling
		clock_gettime(CLOCK_REALTIME, &now);
		init_time = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
		if (conf == 1) 	prof_start = init_time; // Profiling start
		
		do {
		
			clock_gettime(CLOCK_REALTIME, &now);
			curr_time = (double)now.tv_sec+(double)now.tv_nsec*1e-9;

		} while (curr_time-init_time < interval);
	
		// Get initial value for executed tasks;
		int cont_tasks[2][2];
		cont_tasks[0][0] = *(sched.cont_tasks_zc);
		cont_tasks[0][1] = *(sched.cont_tasks_zc + 1);
	
		// Wait before evicting a stream
		clock_gettime(CLOCK_REALTIME, &now);
		init_time = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
		do {
			clock_gettime(CLOCK_REALTIME, &now);
			curr_time = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
		} while (curr_time-init_time < sampling_interval);
	
		// Evict stream of second kernel(BSU)
		if (conf < info_coexec->num_configs) { // In last iteration do no evict
			evict_streams(coexec.kstr[1], coexec.num_streams[1] - info_coexec->pairs[conf][1]); // Revisar
			coexec.num_streams[1] -= (coexec.num_streams[1] - info_coexec->pairs[conf][1]); // Discount evicted streams (second kernel) so it is not launched again
		}
		
		// After eviction (synchronous operation), sample executed tasks
		
		clock_gettime(CLOCK_REALTIME, &now);
		curr_time = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
		
		if (conf == info_coexec->num_configs )
			prof_end = curr_time; // profiling end
	
		cont_tasks[1][0] = *(sched.cont_tasks_zc);
		cont_tasks[1][1] = *(sched.cont_tasks_zc + 1);
		
		double tpms0 = (double)(cont_tasks[1][0]-cont_tasks[0][0])/((curr_time-init_time)*1000);
		double tpms1 = (double)(cont_tasks[1][1]-cont_tasks[0][1])/((curr_time-init_time)*1000);
	
		//printf("Tpms k0=%f (%d,%d) k1=%f (%d,%d)\n", tpms0, cont_tasks[1][0], cont_tasks[0][0], tpms1, cont_tasks[1][1], cont_tasks[0][1]);
		
		if (conf > 1 )
			if ((tpms0/prev_tpms0 + tpms1/prev_tpms1) < 2 && flag == 1) {
				save_partition = conf-1;
			//	printf("Aqui ***************************\n");
				flag = 0;
				prof_end = curr_time;
				break;
			}
			
		// Save current tpms
		prev_tpms0 = tpms0;
		prev_tpms1 = tpms1;
	
		if (conf < info_coexec->num_configs) { // In last iteration do no add stream
			//printf("*****Adding %d\n", info_coexec->pairs[conf][0] - coexec.num_streams[0]);
			add_streams_to_kernel(&coexec, &sched, coexec.kstr[0], info_coexec->pairs[conf][0] - coexec.num_streams[0] ); // Add a new stream (BSU) to first kernel
		}
	}
	
	printf("Mejor configuracion %d %d \n", info_coexec->pairs[save_partition-1][0],info_coexec->pairs[save_partition-1][1]);
	printf("Tiempo de profiling en us=%f\n", (prof_end-prof_start)*1000000); 
	
	// Evict kernels
	
	evict_streams(coexec.kstr[0], coexec.num_streams[0]);
	evict_streams(coexec.kstr[1], coexec.num_streams[1]);
	
	
	// Evict proxy
	sched.kernel_evict_zc[0] =  PROXY_EVICT;
	cudaDeviceSynchronize();
	
	free_coBlocks_entry(info_coexec);
	
	return 0;
}


int fast_solo_profiling(int deviceId, t_Kernel kid)
{
	cudaError_t err;
	struct timespec now;
	double curr_time, prof_start, prof_end;

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
	
	// Select initial kernel
	int task_index = 0; // Index of the kernel in the array with ready kernels;
	//k_done[task_index] = 1; // Kernel removed from pending kernels*/// Do it when more CTAs are added
	
	// add kernel to coexec  
	add_kernel_for_coexecution(&coexec, &sched, &kstr[task_index], 1, 0);
	
	for (int cta=1; cta <= 8; cta++) {
	
		// Execute kernel (launching streams) in coexec structure: a new stream is launched
		launch_coexec(&coexec);	
		
		// Waiting time when stream is launched: streams takes a time to start running
		clock_gettime(CLOCK_REALTIME, &now);
		double time0 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
		if (cta == 1) prof_start = time0;
		double initial_interval = 0.000200 ;
		do {
		
		
			clock_gettime(CLOCK_REALTIME, &now);
			curr_time = (double)now.tv_sec+(double)now.tv_nsec*1e-9;

		} while (curr_time-time0 < initial_interval);
		
		// Get current task counter	
		int cont_task0 = *(sched.cont_tasks_zc);
		
		// Waiting time for sampling task counter
		clock_gettime(CLOCK_REALTIME, &now);
		time0 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
		double sampling_interval = 0.000300 ;
		do {
		
			clock_gettime(CLOCK_REALTIME, &now);
			curr_time = (double)now.tv_sec+(double)now.tv_nsec*1e-9;

		}  while (curr_time-time0 < sampling_interval);
		
		// Get current task counter	
		int cont_task1 = *(sched.cont_tasks_zc);
		
		// Calculate tpms
		double tpms = double(cont_task1-cont_task0)/((curr_time-time0)*1000);
		printf("CTA=%d tpms=%f\n", cta, tpms);
		
		if (cta == 8) prof_end = curr_time;
		
		add_streams_to_kernel(&coexec, &sched, coexec.kstr[0], 1); // Add a new stream (BSU) to kernel (solo kernel)
	}

	printf("Profiling time =%f\n", prof_end-prof_start);
	
	// Evict proxy
	sched.kernel_evict_zc[0] =  PROXY_EVICT;
	cudaDeviceSynchronize();
	
	/*for (int i=0; i<=cont_us; i++)
		printf("%d,", cont_tasks[i]);
	printf("\n");
		*/
	return 0;
}


	
	
	
	
	
	