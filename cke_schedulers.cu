#include <unistd.h>
#include <string.h>
#include <helper_functions.h>   // helper functions for string parsing
#include <helper_cuda.h>   
#include "elastic_kernel.h"


/* Scheduler that coexecutes GPU kernels from a list of pending kernels */  
int launch_tasks(int deviceId)
{
	t_Kernel *kid;
	t_kernel_stub **kstubs;
	cudaError_t err;
	
	// Select device
	cudaSetDevice(deviceId);
	cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);	
	printf("Device=%s\n", deviceProp.name);
	
	// Initializae performance of coexecution
	
	initialize_performance();
	
	// Select kernels
	int num_kernels=4;
	
	kid = (t_Kernel *)calloc(num_kernels, sizeof(t_Kernel));
	kid[0]=MM; kid[1]=VA; kid[2]=BS; kid[3]=PF;
	
	/** Create commom streams for all kernels: two for asynchronous transfers, one for preemption commands*/
	cudaStream_t *transfers_s;
	transfers_s = (cudaStream_t *)calloc(2, sizeof(cudaStream_t));
	
	for (int i=0;i<2;i++){
		err = cudaStreamCreate(&transfers_s[i]);
		checkCudaErrors(err);
	} 
	
	cudaStream_t preemp_s;
	checkCudaErrors(cudaStreamCreateWithFlags(&preemp_s, cudaStreamNonBlocking)); 
	
	// Create stream pool for kernel execution
	t_strPool pool;	
	int max_cke_kernels = 2;
	create_stream_pool(max_cke_kernels*MAX_STREAMS_PER_KERNEL, &pool);
	
	/** Create stubs ***/
	kstubs = (t_kernel_stub **)calloc(num_kernels, sizeof(t_kernel_stub*));
	for (int i=0; i<num_kernels; i++)
		create_stubinfo(&kstubs[i], deviceId, kid[i], transfers_s, &preemp_s);
	
	make_transfers(kstubs, num_kernels);
	
	// Create streams kernel info for coexecution
	t_kstreams *kstr = (t_kstreams *)calloc(num_kernels, sizeof(t_kstreams));
	for (int i=0; i<num_kernels; i++)
		create_kstreams(kstubs[i], &kstr[i]);
	
	// Coxecution info	
	t_kcoexec coexec;
	create_coexec(&coexec, 2);
	
	// Select first kernel to be launch	
	//t_Kernel curr_kid = MM; // Id of the first kernel in coexec
	
	int *k_select = (int *)calloc(num_kernels, sizeof(int));
	
	/*int i;
	for (i=0; i<num_kernels; i++)
		if (kid[i] == curr_kid && k_select[i] == 0){
			k_select[i] = 1; //Mask kernel to avoid the relaunch of a kernel
		}
	
	if (i>= num_kernels){
		printf("Error: first kernel not found\n");
		return -1;
	}
	*/
	// Get coexecution kernels
	int run_kernel_index; //= i; // index in kid array of the fist kernel of coexec structure (coexec.kstr[0])
	int b0, b1; // optimun number of blocks in kernels stored in coexec to achieve the best speedup
	int index_new_kernel; // index in kid array of the second kernel of coexec structure (coexec.kstr[0]
	t_Kernel curr_kid, next_kid; // Id of the second kernel of coexec
	int cont_kernels_finished = 0;
	while (cont_kernels_finished < num_kernels) {
		
		int i;
		if (coexec.kstr[0] == NULL) { // If no kernel in coexec selected 
			for (i=0;i<num_kernels;i++){
				if (k_select[i] == 0) // Select the first kernel available
					break;
			}
			if (i == num_kernels) {
				printf("error:  No kernel found\n");
				return -1;
			}
			else {
				k_select[i]=1;
				curr_kid = kid[i];
				run_kernel_index = i;
			}
		}

		if (cont_kernels_finished == num_kernels - 1 ){ // If only a kernel remains
				get_last_kernel(coexec.kstr[0]->kstub->id, &b0); // Give all resources to the last kernel
				next_kid = EMPTY;
			}
			else
				get_best_partner(kid, k_select, num_kernels, curr_kid, &next_kid, &index_new_kernel, &b0, &b1);	// Get the partner (next_kid) provinding the higher speepup in coexecution with curr_kid kernel 	
	
		if (next_kid == EMPTY) {		// It is better to execute kernel alone
			coexec.kstr[0] = &kstr[run_kernel_index]; // Fisrt kernel
			coexec.kstr[1] = NULL; // Second kernel null
			coexec.num_kernels = 1;
			if (b0 < kstr[0].num_streams){ // If number of streams of irst kernel must be reduced
				evict_streams(&pool, coexec.kstr[0], kstr[0].num_streams - b0); // Evict those streams (remaining streams continue in execution)
			}
			else {
				add_streams(&pool, coexec.kstr[0], b0 - kstr[0].num_streams); // More streams are added
				launch_SMK_kernel(&pool, coexec.kstr[0]); // Those new streams are launched
			}
			
		}
		else
		{
			k_select[index_new_kernel] = 1;
			coexec.kstr[0] = &kstr[run_kernel_index]; // Fist kernel is loaded in coexec
			coexec.kstr[1] = &kstr[index_new_kernel]; // Second kernel
			coexec.num_kernels = 2;
			if (b0 < kstr[0].num_streams) { // Fist kernel is analyzed either to evict some running streams or to launch the new ones
				evict_streams(&pool, coexec.kstr[0], kstr[0].num_streams - b0);
			}
			else{
				add_streams(&pool, coexec.kstr[0], b0 - kstr[0].num_streams);
				launch_SMK_kernel(&pool, coexec.kstr[0]);
			}
	
			add_streams(&pool, coexec.kstr[1], b1); // Second kernel is launched for the fist time: streams are added
			launch_SMK_kernel(&pool, coexec.kstr[1]);
		}
	
		int kernel_id;
		wait_for_kernel_termination(&pool, &coexec, &kernel_id); // Wait until one of the two runnunk kernels ends (all its streams finish)
		printf("Kernel %d finished\n", coexec.kstr[kernel_id]->kstub->id);
		cont_kernels_finished++; // Counter that indicates when all kernels have been processed
		coexec.num_kernels--;

		if (cont_kernels_finished >= num_kernels) // Check if last kernel has finished
			break; 
	
		if (kernel_id == 1){ /* If second kernel finishes*/
		
			//k_select[index_new_kernel] = 1; // Remove from further processing
		
			coexec.kstr[1]=NULL; // Eliminate it from coexec
			curr_kid = coexec.kstr[0]->kstub->id; // Update running kernel for next iteration
		}
	
		if (kernel_id == 0){
			
			//k_select[run_kernel_index] = 1; // Remove from further processing

			coexec.kstr[0] = coexec.kstr[1]; //Move info from coexec.kstr1 to coexec.kstr0
			coexec.kstr[1] = NULL; // Remove from further processing
			run_kernel_index = index_new_kernel; // Update index of running kernel
			curr_kid = coexec.kstr[0]->kstub->id;
		}
	}
	
	return 0;
}
