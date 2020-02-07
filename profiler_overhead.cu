#include <unistd.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <helper_functions.h>   // helper functions for string parsing
#include <helper_cuda.h>   
#include "elastic_kernel.h"
#include <cuda_profiler_api.h>

// Tables to store results for solo exectuions
extern t_smk_solo *smk_solo; // 
extern t_smt_solo *smt_solo;

// Tables to store coexecution results
extern t_smk_coBlocks **smk_conc;
extern t_smt_coBlocks **smt_conc; //tpms of each kernel in coexection

// Table to store better speedups in coexecution

extern t_co_speedup **smk_best_sp;
extern t_co_speedup **smt_best_sp;

int interleaved_launch_coexec(t_kcoexec *coexec)
{
	//Launch interleaved BSUs (streams) belonging to different kernels
	
	int new_streams_k0 = coexec->num_streams[0]-coexec->kstr[0]->num_streams;
	int new_streams_k1 = coexec->num_streams[1]-coexec->kstr[1]->num_streams;

	if (new_streams_k0 < new_streams_k1) {
		
		// launch interleaved streams
		for (int i=0; i< new_streams_k0; i++){
			launch_SMK_kernel(coexec->kstr[0], 1);
			launch_SMK_kernel(coexec->kstr[1], 1);
		}
		
		// Laiuch remainning streams of k1
		launch_SMK_kernel(coexec->kstr[1], new_streams_k1-new_streams_k0);
	}
	else {
		
		// launch interleaved streams
		for (int i=0; i< new_streams_k1; i++){
			launch_SMK_kernel(coexec->kstr[0], 1);
			launch_SMK_kernel(coexec->kstr[1], 1);
		}
	
		launch_SMK_kernel(coexec->kstr[0], new_streams_k0-new_streams_k1);
	}
	
	return 0;
}


typedef enum {START, FORWARD, BACKWARD} t_search; // Search in co-execution configuration set. START: first search using FORWARD direction.  


int pair_overhed(t_kernel_stub *kstub0, t_kernel_stub *kstub1)
{
	struct timespec now;
	char name0[20], name1[20]; // Kernel names
	// Temporal model variables: waiting times in seconds
	double w_launch = 	0.000200; //0.000120; // Wait until all streas are running
	double w_sample =	0.000030; // waiting time between samples (executed_tasks), that is, the inverse of sampling rate: fixed to the minimum eviction time divided by two
	
	// Load profiling tables: only set of coexecution configurations are necessary
	read_profling_tables();
	
	// Create streams kernel info for coexecution
	t_kstreams *kstr = (t_kstreams *)calloc(2, sizeof(t_kstreams));
	create_kstreams(kstub0, &kstr[0]);
	create_kstreams(kstub1, &kstr[1]);

	// Coxecution info	
	t_kcoexec coexec;
	create_coexec(&coexec, 2);
	
	// Launch proxy
	t_sched sched;
	create_sched(&sched);
	launch_generic_proxy((void *)&sched);	// Launch proxy
	
	// Kids
	int kid0 = kstub0->id;
	int kid1 = kstub1->id;
	kid_from_index(kid0, name0);
	kid_from_index(kid1, name1);

	// Get coexecution configuration located in central position 
	int num_confs = smk_conc[kid0][kid1].num_configs;
	int index = num_confs/2;
	int prev_index;
	int b0 = smk_conc[kid0][kid1].pairs[index][0];
	int b1 = smk_conc[kid0][kid1].pairs[index][1];
	int prev_b0, prev_b1;
	double prev_ter0 = -1, prev_ter1=-1;
	double ter0, ter1;
	t_search search = START;
	
	// Add streams (BSUs)
	add_kernel_for_coexecution(&coexec, &sched, &kstr[0], b0, 0); // Add b0 streams
	add_kernel_for_coexecution(&coexec, &sched, &kstr[1], b1, 1); // Add b1 streams
	
	// Execute kernels (launching streams) in coexec structure
	interleaved_launch_coexec(&coexec);
	
	clock_gettime(CLOCK_REALTIME, &now);
	double time1 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
	double time2 = time1;
	double start_time = time2;
	
	double prof_ini = time1;
	
	// Wait until launching ends
	while ((time2 - time1) < w_launch){
		clock_gettime(CLOCK_REALTIME, &now);
		time2 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
	}
	time1 = time2;	
	//printf("PROF: Conf(%d,%d)\n", b0, b1);
	while (1) {
		
		// Set start time: start of sampling of a co-execution configuration  
		clock_gettime(CLOCK_REALTIME, &now);
		start_time = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
		
		// Get number of task at the beginning of the sampling
		int executed_tasks0 = *(sched.cont_tasks_zc + 0);
		int executed_tasks1 = *(sched.cont_tasks_zc + 1);
	
		// Take samples until they are stable
		double lprev_ter0=0, lprev_ter1=0;
		while (1) { 
			
			// Interval between consecutive samples
			while ((time2 - time1) < w_sample){
				clock_gettime(CLOCK_REALTIME, &now);
				time2 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
			}
			time1 = time2;
			
			// Performance with WS
			
			int cont_task0 = *(sched.cont_tasks_zc + 0);
			int cont_task1 = *(sched.cont_tasks_zc + 1);
			
			if (cont_task0 >= kstub0->total_tasks || cont_task1 >= kstub1->total_tasks) {
				// Evict proxy
				sched.kernel_evict_zc[0] =  PROXY_EVICT;
				cudaDeviceSynchronize();
	
				// If a kernel finished during profiling, exit
				if (cont_task0 >= kstub0->total_tasks)
					printf("PROF:, %s-%s, %f, %f First kernel has finished\n",  name0, name1, ter0, ter1);
				else
					printf("PROF:, %s-%s, %f, %f Second kernel has finished\n",  name0, name1, ter0, ter1);

	
				// Free
				remove_kstreams(kstr);
				free(kstr);
				remove_coexec(&coexec);
				return -1;
			}
	
			ter0 = (double)(cont_task0-executed_tasks0)/(time2 - start_time);
			ter1 = (double)(cont_task1-executed_tasks1)/(time2 - start_time);
			
			// Compare previous samples with the current ones 
			if (ter0 !=0 && ter1 !=0) // Is some ter is zero, go on taking samples
				if (fabs(ter0-lprev_ter0)<0.2*ter0 && fabs(ter1-lprev_ter1)<0.2*ter1) { // Is samples are stable go to WS calculation
					printf("(ter0=%f ter1=%f) (prev_ter0=%f prev_ter1=%f) \n", ter0, ter1, lprev_ter0, lprev_ter1);
					printf("(ter0-prev_ter0=%f, %f ter1-prev_ter1=%f, %f) \n", fabs(ter0-lprev_ter0), 0.1*ter0, fabs(ter1-lprev_ter1), 0.1*ter1);

					break;
				}
				
			lprev_ter0=ter0;
			lprev_ter1=ter1;
			
		}
	
		if (prev_ter0 != -1){ // Flasg to indicate that TER for two configurations have been calculated 
			double ws = 0.5*(ter0/prev_ter0 + ter1/prev_ter1);
			printf("WS Conf(%d->%d) %f\n", index, prev_index, ws);
			if (search == START) { //If first ws calculated calculated
				if (ws < 1) { // change direction of the search
					search = BACKWARD;
					ter0 = prev_ter0;
					ter1 = prev_ter1;
					index = prev_index;
					b0 = prev_b0;
					b1 = prev_b1;
				}
				else 
					search = FORWARD;
			}
			else  
				if (ws < 1) {					
					printf("Best configuration achieved %s/%s=(%d,%d)\n", name0, name1, smk_conc[kid0][kid1].pairs[prev_index][0], smk_conc[kid0][kid1].pairs[prev_index][1]);
					break;
				}
		}
		
		// Save values from current configuration
		prev_index = index;
		prev_ter0 = ter0;
		prev_ter1 = ter1;
		prev_b0 = b0;
		prev_b1 = b1;
		
		// Change search config
		if (search == START || search == FORWARD)
			index = index +1; //New condifuracion
		else
			index = index-1;
		
		if (index >= num_confs || index < 0) {
			printf("Configuracion extrema alcanzada %s/%s=(%d,%d)\n", name0, name1, smk_conc[kid0][kid1].pairs[prev_index][0], smk_conc[kid0][kid1].pairs[prev_index][1]);
			break;
		}
		b0 = smk_conc[kid0][kid1].pairs[index][0];
		b1 = smk_conc[kid0][kid1].pairs[index][1];
		//printf("PROF: Conf(%d,%d)\n", b0, b1);

		if (b0 < coexec.num_streams[0]){ // If k0 has less BSUs
			evict_streams(coexec.kstr[0], coexec.num_streams[0] - b0); // Evict BSU(s)
			//printf("Conf=%d Eviciting  %d BSUs the kernel 0\n", index, coexec.num_streams[0] - b0);
			coexec.num_streams[0] -= (coexec.num_streams[0] - b0);
			
			//printf("Conf=%d Lanzando %d BSUs the kernel 1\n", index, b1 - coexec.num_streams[1]);
			add_streams_to_kernel(&coexec, &sched, coexec.kstr[1], b1-coexec.num_streams[1]);
			launch_coexec(&coexec);
		}
		else { // b1 has less BSUs
			evict_streams(coexec.kstr[1], coexec.num_streams[1] - b1); // Evict BSU(s)
			//printf("Conf=%d Eviciting  %d BSUs the kernel 1\n", index, coexec.num_streams[1] - b1);
			coexec.num_streams[1] -= (coexec.num_streams[1] - b1);
			
			//printf("Conf=%d Lanzando %d BSUs the kernel 0\n", index, b0 - coexec.num_streams[0]);
			add_streams_to_kernel(&coexec, &sched, coexec.kstr[0], b0-coexec.num_streams[0]);
			launch_coexec(&coexec);

		}
	}
	
	// Read executed tasks during profiling for each kernel
	int executed_task0  = *(sched.cont_tasks_zc + 0);
	int executed_task1  = *(sched.cont_tasks_zc + 1);
	
	// Get tpms achieved for each kernel during solo execution
	double tpms0 = smk_solo[kid0].tpms[smk_solo[kid0].num_configs-1];
	double tpms1 = smk_solo[kid1].tpms[smk_solo[kid1].num_configs-1];
	
	// Calculate the time taken for both kernels, sequentially executed, to run those tasks 
 	double seq_time = (double)executed_task0/tpms0 + (double)executed_task1/tpms1;
	
	
	// Get time to calculate profiling time;
	clock_gettime(CLOCK_REALTIME, &now);
	time1 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
	
	// Compare profiling time with sequential time executing the sae number of tasks
	printf("Label, K0/K1, Config, Ptime(ms), Stime(ms), HyperQtime (ms)\n");
	printf("PROF:, %s-%s, %d-%d, %f, %f",  name0, name1, smk_conc[kid0][kid1].pairs[prev_index][0], smk_conc[kid0][kid1].pairs[prev_index][1], (time1-prof_ini)*1000, seq_time);
	
	// Evict proxy
	sched.kernel_evict_zc[0] =  PROXY_EVICT;
	cudaDeviceSynchronize();
	
	
	// Check sequential time using two streams 

	kstub0->kconf.max_persistent_blocks = 8;
	int save_total_task0 = kstub0->total_tasks;
	kstub0->total_tasks = executed_task0;
	kstub1->kconf.max_persistent_blocks = 8;
	int save_total_task1 = kstub1->total_tasks;
	kstub1->total_tasks = executed_task1;

	for (int i=0; i<MAX_STREAMS_PER_KERNEL; i++)
		kstub0->h_state[i] = PREP;
	cudaMemcpy(kstub0->gm_state,  kstub0->h_state, sizeof(State) * MAX_STREAMS_PER_KERNEL, cudaMemcpyHostToDevice);
	kstub0->execution_s = &(kstr[0].str[0]); //Stream id
	kstub0->stream_index = 0; // Index used by kernel to test state of i-esimo stream
	
	for (int i=0; i<MAX_STREAMS_PER_KERNEL; i++)
		kstub1->h_state[i] = PREP;
	cudaMemcpy(kstub1->gm_state,  kstub1->h_state, sizeof(State) * MAX_STREAMS_PER_KERNEL, cudaMemcpyHostToDevice);
	kstub1->execution_s = &(kstr[1].str[0]); //Stream id
	kstub1->stream_index = 0; // Index used by kernel to test state of i-esimo stream
	
	// Use all the SMs
	int idSMs[2];
	idSMs[0] = 0;idSMs[1] = kstr->kstub->kconf.numSMs-1;
	kstub0->idSMs = idSMs;
	kstub1->idSMs = idSMs;
	
	cudaMemset(kstub0->d_executed_tasks, 0, sizeof(int));
	cudaMemset(kstub1->d_executed_tasks, 0, sizeof(int));

	// Get time to calculate profiling time;
	clock_gettime(CLOCK_REALTIME, &now);
	time1 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
	(kstub0->launchCKEkernel)(kstub0);
	(kstub1->launchCKEkernel)(kstub1);
	cudaDeviceSynchronize();
	clock_gettime(CLOCK_REALTIME, &now);
	time2 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
	
	
	printf(", %f\n", (time2-time1)*1000);
	
	// Restore original value
	kstub0->total_tasks = save_total_task0;
	kstub1->total_tasks = save_total_task1;
	
	// Free
	remove_kstreams(kstr);
	free(kstr);
	remove_coexec(&coexec);
	
	return 0;
}


// Applications is composed of one or several kernels
typedef struct{
	int num_kernels;
	int index;
	t_Kernel kid[8]; // Max: 8 kernels per application
	t_kernel_stub* kstubs[8]; // One kernel stub per kernel
}t_application;

int online_profiler_overhead(t_Kernel *kid, int num_kernels, int deviceId)
{

	struct timespec now;
    cudaError_t err;
	
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
	int max_num_kernels=13;
	t_kernel_stub **kstubs = (t_kernel_stub **)calloc(max_num_kernels, sizeof(t_kernel_stub*));
	int cont=0;
	for (int i=0; i<num_kernels; i++) { 
		create_stubinfo(&kstubs[cont++], deviceId, kid[i], transfers_s, &preemp_s);
		if (kid[i] == GCEDD) { 
			create_stubinfo_with_params(&kstubs[cont++], deviceId, SCEDD, transfers_s, &preemp_s, kstubs[i]->params);
			create_stubinfo_with_params(&kstubs[cont++], deviceId, NCEDD, transfers_s, &preemp_s, kstubs[i]->params);
			create_stubinfo_with_params(&kstubs[cont++], deviceId, HCEDD, transfers_s, &preemp_s, kstubs[i]->params);
		}
		if (kid[i] == RCONV){
			create_stubinfo_with_params(&kstubs[cont++], deviceId, CCONV, transfers_s, &preemp_s, kstubs[i]->params);
		}
	}

	// Make allocation and HtD transfer for kernels
	make_transfers(kstubs, cont);
	
	for (int i=0; i<cont; i++)
		for (int j=i+1; j<cont; j++)
			pair_overhed(kstubs[i], kstubs[j]);
	
	return 0;
	
	
}