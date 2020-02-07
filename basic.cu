#include <unistd.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <helper_functions.h>   // helper functions for string parsing
#include <helper_cuda.h>   
#include "elastic_kernel.h"

/** Launch a kernel using a set of streams. Each stream will run NUM_SMs blocks (for SMK execution **/

extern t_smk_solo *smk_solo; 

int create_coexec(t_kcoexec *coexec, int num_kernels)
{
	coexec->num_kernels = 0;
	coexec->kstr = (t_kstreams **)calloc(num_kernels, sizeof(t_kstreams *));
	coexec->num_streams = (int *)calloc(num_kernels, sizeof(int));
	coexec->queue_index = (int *)calloc(num_kernels, sizeof(int));
	
	return 0;
}

int remove_coexec(t_kcoexec *coexec)
{
	free(coexec->kstr);
	free(coexec->num_streams);
	free(coexec->queue_index);
	
	return 0;
}

int create_kstreams(t_kernel_stub *kstub, t_kstreams *kstr)
{
	cudaError_t err;
	kstr->kstub = kstub;
	kstr->num_streams = 0;
	
	for (int i=0; i<MAX_STREAMS_PER_KERNEL; i++){
		err = cudaStreamCreate(&kstr->str[i]);
		checkCudaErrors(err);
	}
	
	return 0;
}

int remove_kstreams(t_kstreams *kstr)
{
	for (int i=0; i<MAX_STREAMS_PER_KERNEL; i++)
		cudaStreamDestroy(kstr->str[i]);
	
	return 0;
}

int add_streams(t_kstreams *kstr, int num_streams)
{ 
	if (num_streams + kstr->num_streams > MAX_STREAMS_PER_KERNEL){
		printf("Error: not enough available streams\n");
		return -1;
	}
	
	kstr->num_streams += num_streams;
	
	return 0;
}
	
int launch_SMK_kernel(t_kstreams *kstr, int new_streams)
{	
	
	// Use all the SMs
	int idSMs[2];
	idSMs[0] = 0;idSMs[1] = kstr->kstub->kconf.numSMs-1;
	kstr->kstub->idSMs = idSMs;
	
	if (new_streams + kstr->num_streams > MAX_STREAMS_PER_KERNEL){
		printf("Error: not enough available streams\n");
		return -1;
	}
	
	if (kstr->kstub->id == RCONV)   // It is supposed number of original streams have been divided ny two
		kstr->kstub->kconf.max_persistent_blocks = 2; // Two kernels per SM. So the number of streams will be lower
	else
		kstr->kstub->kconf.max_persistent_blocks = 1; // Only a block per SM will be launched by each stream
	
	for (int i=0 ; i<new_streams; i++)  /* Before launching set streams state to PREP */
		kstr->kstub->h_state[kstr->num_streams + i] = PREP;	
	//cudaMemcpy(kstr->kstub->gm_state+kstr->num_streams,  kstr->kstub->h_state+kstr->num_streams, new_streams * sizeof(State), cudaMemcpyHostToDevice);
  
	// Launch streams
	for (int i=0; i<new_streams; i++) {
		kstr->kstub->execution_s = &(kstr->str[i+kstr->num_streams]); //Stream id
		kstr->kstub->stream_index = i + kstr->num_streams; // Index used by kernel to test state of i-esimo stream
		(kstr->kstub->launchCKEkernel)(kstr->kstub);
		printf("Lanzado stream %d del kernel %d\n", i+kstr->num_streams, kstr->kstub->id);
	}
	
	kstr->num_streams += new_streams;
	
	return 0;
}

int wait_for_kernel_termination(t_kcoexec *info, int *kernelid)
{
	
	// Obtain the number of streams per kernel
	int *pending_streams = (int *) calloc(info->num_kernels, sizeof(int));
	int **mask = (int **)calloc(info->num_kernels, sizeof(int));

	for (int i=0; i<info->num_kernels; i++){
		pending_streams[i] = info->kstr[i]->num_streams;
		mask[i] = (int *)calloc(info->kstr[i]->num_streams, sizeof(int));
	}		
	
	while(1){
		// Check streams
		for (int i = 0; i < info->num_kernels; i++) { // For each kernel 
			for (int j=0; j < info->kstr[i]->num_streams; j++){ // For each stream
				if (mask[i][j] == 0) {
					if (cudaStreamQuery(info->kstr[i]->str[j]) == cudaSuccess) {
						//printf("stream %d de kernel %d finalizado\n", j, info->kstr[i]->kstub->id);
					
						mask[i][j] = 1; // Masking			
						pending_streams[i]--;
					}
				}
			}
			
			if (pending_streams[i] == 0 ) {//All the streams of a kernel have finished
				info->kstr[i]->num_streams=0;
				*kernelid = i;
				free(pending_streams);
				for (int i=0; i<info->num_kernels; i++)
					free(mask[i]);
				free(mask);
				return 0;
			}
		}
	}				
}

int wait_for_kernel_termination_with_proxy(t_sched *sched, t_kcoexec *info, int *kernelid, double *speedup)
{
	int prev_executed_tasks[MAX_NUM_COEXEC_KERNELS];
	int curr_executed_tasks[MAX_NUM_COEXEC_KERNELS];
	double  task_per_second[MAX_NUM_COEXEC_KERNELS];
	struct timespec now;
	double init_time, prev_time, curr_time;
	double s = 1.0;
	
	// Obtain the number of streams per kernel
	int *pending_streams = (int *) calloc(MAX_NUM_COEXEC_KERNELS, sizeof(int));
	int **mask = (int **)calloc(MAX_NUM_COEXEC_KERNELS, sizeof(int)); 
	//memset(task_per_second, 0, MAX_NUM_COEXEC_KERNELS * sizeof(double));

	for (int i=0; i<MAX_NUM_COEXEC_KERNELS; i++){
		if (info->kstr[i] != NULL) {		
			pending_streams[i] = info->kstr[i]->num_streams;
			mask[i] = (int *)calloc(info->kstr[i]->num_streams, sizeof(int));
		}
	}		
		
	// Current value of number of executed tasks
	clock_gettime(CLOCK_REALTIME, &now);
	init_time = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
	prev_time = init_time+0.0004;
	for (int i = 0; i < MAX_NUM_COEXEC_KERNELS; i++)
		if (info->kstr[i] != NULL)
			prev_executed_tasks[i] = *(sched->cont_tasks_zc+i);
	
	while(1){
		
		// Check streams
		
		// Check streams
		for (int i = 0; i < MAX_NUM_COEXEC_KERNELS; i++) { // For each kernel
			if (info->kstr[i] != NULL) {
				for (int j=0; j < info->kstr[i]->num_streams; j++){ // For each stream
					if (mask[i][j] == 0) {
						if (cudaStreamQuery(info->kstr[i]->str[j]) == cudaSuccess) {
							//printf("stream %d de kernel %d finalizado\n", j, info->kstr[i]->kstub->id);
					
							mask[i][j] = 1; // Masking			
							pending_streams[i]--;
						}
					}
				}
			
				if (pending_streams[i] == 0 ) {//All the streams of a kernel have finished
					//printf("Kernel %d terminado con pending %d. El otro %d\n", info->kstr[i]->kstub->id, pending_streams[i], pending_streams[(i+1) % 2]); 
					info->kstr[i]->num_streams=0;
					
					*kernelid = i;
					*speedup = s;
					free(pending_streams);
					for (int j=0; j<info->num_kernels; j++)
						free(mask[j]);
					free(mask);
					if (info->kstr[i]->kstub->id == 8)
						printf("Aqui\n");
					return 0;
					
				}
			}
		}
		
		#ifdef ONLINE_CHECK
		
		// Check speepup when two kernels are concurrently running
		clock_gettime(CLOCK_REALTIME, &now);
 		curr_time = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
		
		if (curr_time - prev_time > 0.0002){ // Check interval 
			
			for (int i = 0; i < MAX_NUM_COEXEC_KERNELS; i++)
				if (info->kstr[i] != NULL)
					curr_executed_tasks[i] = *(sched->cont_tasks_zc+i); // Read zero-copy variables
			for (int i = 0; i < MAX_NUM_COEXEC_KERNELS; i++) {
				if (info->kstr[i] != NULL) {
					task_per_second[i] = (curr_executed_tasks[i] - prev_executed_tasks[i]) / (curr_time - init_time);
					//printf("Kid=%d tpms=%f\n", info->kstr[i]->kstub->id, task_per_second[i]/1000);
				}
			}
			if (info->num_kernels == 2) { // Compare with serial ejecucion only when two kernels are running
				t_Kernel idk0, idk1;
				idk0 = info->kstr[0]->kstub->id;
				idk1 = info->kstr[1]->kstub->id;
				double t0 = task_per_second[0]/(smk_solo[idk0].tpms[smk_solo[idk0].num_configs-1] *1000);
				double t1 = task_per_second[1]/(smk_solo[idk1].tpms[smk_solo[idk1].num_configs-1] *1000);
				s = t0+t1;
				//printf("Speedup = %f \n", s); 
				
				if ( s < MIN_SPEEDUP) {
					free(pending_streams);
					for (int i=0; i<info->num_kernels; i++)
						free(mask[i]);
					free(mask);
					*kernelid = -1;
					*speedup = s;
					return 0;
				}
			}
			
			prev_time = curr_time;
		}
		#endif
	}
	
}

// Coexecution with a ls kernel
int ls_coexec_wait_for_kernel_termination_with_proxy(t_sched *sched, t_kcoexec *info, double min_tpms, 
									double start_time, double sampl_interval, double *acc_numstreams_time, int *kernelid, int *return_code)
{
	int prev_executed_tasks[MAX_NUM_COEXEC_KERNELS];
	int curr_executed_tasks[MAX_NUM_COEXEC_KERNELS];
	double  task_per_second[MAX_NUM_COEXEC_KERNELS];
	struct timespec now;
	double s = 1.0;
	
	/** Save executed task of LS kernel on function entry */
	int prev_tasks = *(sched->cont_tasks_zc + 0);
	
	// Get current time
	clock_gettime(CLOCK_REALTIME, &now);
	double prev_time = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
		
	// Obtain the number of streams per kernel
	int *pending_streams = (int *) calloc(MAX_NUM_COEXEC_KERNELS, sizeof(int));
	int **mask = (int **)calloc(MAX_NUM_COEXEC_KERNELS, sizeof(int)); 
	//memset(task_per_second, 0, MAX_NUM_COEXEC_KERNELS * sizeof(double));

	for (int i=0; i<MAX_NUM_COEXEC_KERNELS; i++){
		if (info->kstr[i] != NULL) {		
			pending_streams[i] = info->kstr[i]->num_streams;
			mask[i] = (int *)calloc(info->kstr[i]->num_streams, sizeof(int));
		}
	}		
	
	while(1){
		
		// Check streams
		for (int i = 0; i < MAX_NUM_COEXEC_KERNELS; i++) { // For each kernel
			if (info->kstr[i] != NULL) {
				for (int j=0; j < info->kstr[i]->num_streams; j++){ // For each stream
					if (mask[i][j] == 0) {
						if (cudaStreamQuery(info->kstr[i]->str[j]) == cudaSuccess) {
							printf("stream %d de kernel %d finalizado\n", j, info->kstr[i]->kstub->id);
					
							mask[i][j] = 1; // Masking			
							pending_streams[i]--;
						}
					}
				}
			
				if (pending_streams[i] == 0 ) {//All the streams of a kernel have finished
					printf("Kernel %d terminado con pending %d. El otro %d\n", info->kstr[i]->kstub->id, pending_streams[i], pending_streams[(i+1) % 2]); 
					info->kstr[i]->num_streams=0;
					
					*kernelid = i;
					*return_code = 0;
					free(pending_streams);
					for (int j=0; j<info->num_kernels; j++)
						free(mask[j]);
					free(mask);
					if (info->kstr[i]->kstub->id == 8)
						printf("Aqui\n");
					return 0;
					
				}
			}
		}
			
		if (sampl_interval > 0) { // profiling is deactivated when sampl_interval <=0
			
			// Get current time
			clock_gettime(CLOCK_REALTIME, &now);
			double curr_time = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
		
			// If enough time has passed, check
			if ((curr_time - prev_time) > sampl_interval){ // 
				*acc_numstreams_time += (curr_time - prev_time) * info->num_streams[1]; // Product time * num_streams of NON LS kernel
				curr_executed_tasks[0] = *(sched->cont_tasks_zc + 0); // In sched->cont_tasks_zc + 0 is mapped the task counter for LS kernel
				int curr_tasks = curr_executed_tasks[0] - prev_tasks;
				double tpms = (double)curr_tasks/ ((curr_time - prev_time)*1000.0);
				printf("tpms_total=%f tpms=%f tpms_min=%f\n", (double)curr_executed_tasks[0]/((curr_time -start_time)*1000.0), tpms, min_tpms);

				if (tpms < min_tpms || tpms > 1.1 * min_tpms){ // If current tpms of ls kernel is below min value, indicate s that more streams are added
					free(pending_streams);
					for (int j=0; j<info->num_kernels; j++)
						free(mask[j]);
					free(mask);
					if (tpms < min_tpms)
						*return_code = 1;
					if (tpms > 1.1 * min_tpms)
						*return_code = 2;
					return 0; 
				}
				prev_tasks = curr_executed_tasks[0];
				prev_time = curr_time;
			}
		}
	}
	
}


// Synchronous call: exit when evicted streams have finished
int evict_streams(t_kstreams *kstr, int num_streams)
{

	if (num_streams == 0)
		return 0;
	
	int total_streams = kstr->num_streams;

	if (num_streams > total_streams) {
		printf("Error: Too many streams to evict\n");
		return -1;
	}
	
	// Send evict commands
	int index;
	for (int i=0; i<num_streams; i++){
		index = total_streams-1-i; // Start with the last streams
		kstr->kstub->h_state[index] = TOEVICT;
		//cudaMemcpyAsync(&kstr->kstub->gm_state[index],  &kstr->kstub->h_state[index], sizeof(State), cudaMemcpyHostToDevice, *(kstr->kstub->preemp_s));  // Signal TOEVICT State
		printf("--> Evicting stream %d(%d) of kernel %d\n", index, total_streams, kstr->kstub->id);
	}
	cudaMemcpyAsync(&kstr->kstub->gm_state[index],  &kstr->kstub->h_state[index], num_streams * sizeof(State), cudaMemcpyHostToDevice, *(kstr->kstub->preemp_s));  // Signal TOEVICT State
	
	// Wait until stream finish
	for (int i=0; i<num_streams; i++){
		index = total_streams-1-i; 
		cudaStreamSynchronize(kstr->str[index]); // Wait for those streams
		//Update kstream info
		kstr->num_streams--;
		kstr->kstub->h_state[index] = PREP;
		cudaMemcpyAsync(&kstr->kstub->gm_state[index],  &kstr->kstub->h_state[index], sizeof(State), cudaMemcpyHostToDevice, *(kstr->kstub->preemp_s));  // Signal TOEVICT State
	}	
	cudaMemcpyAsync(&kstr->kstub->gm_state[index],  &kstr->kstub->h_state[index], num_streams * sizeof(State), cudaMemcpyHostToDevice, *(kstr->kstub->preemp_s)); // Restore PREP State (in case stream is relaunched)
			
	return 0; 
}

// Synchronous call: exit when evicted streams have finished
int evict_streams_ver2(t_kcoexec *coexec, t_sched *sched, t_kstreams *kstr, int num_streams)
{

	if (num_streams == 0)
		return 0;
	
	int index;
	for (index=0; index<MAX_NUM_COEXEC_KERNELS; index++)
		if (coexec->kstr[index] == kstr)
			break;
		
	if (index >= MAX_NUM_COEXEC_KERNELS){
		printf("Error: Kernel is noi in coexec\n");
		return -1;
	}
		
	int total_streams = kstr->num_streams;

	if (num_streams > total_streams) {
		printf("Error: Too many streams to evict\n");
		return -1;
	}
	
	for (int i=0; i<num_streams; i++)
		sched->kernel_evict_zc[index * MAX_STREAMS_PER_KERNEL + total_streams-1-i] = TOEVICT; 
	
	//Send evict commands
	//int index;
	//for (int i=0; i<num_streams; i++){
	//	index = total_streams-1-i; // Start with the last streams
	//	kstr->kstub->h_state[index] = TOEVICT;
	//	printf("--> Evicting stream %d(%d) of kernel %d\n", index, total_streams, kstr->kstub->id);
	//}
	//cudaMemcpyAsync(&kstr->kstub->gm_state[index],  &kstr->kstub->h_state[index], num_streams * sizeof(State), cudaMemcpyHostToDevice, *(kstr->kstub->preemp_s));  // Signal TOEVICT State
	
	// Wait until stream finish
	for (int i=0; i<num_streams; i++){
		index = total_streams-1-i; 
		cudaStreamSynchronize(kstr->str[index]); // Wait for those streams
		printf("Stream %d del kernel %d evicted\n", index, kstr->kstub->id);
		//Update kstream info
		kstr->num_streams--;
		kstr->kstub->h_state[index] = PREP;
	}	
	//cudaMemcpyAsync(&kstr->kstub->gm_state[index],  &kstr->kstub->h_state[index], num_streams * sizeof(State), cudaMemcpyHostToDevice, *(kstr->kstub->preemp_s)); // Restore PREP State (in case stream is relaunched)
			
	return 0;
}



int all_nocke_execution(t_kernel_stub **kstubs, int num_kernels)
{	
	//kstr->kstub->kconf.max_persistent_blocks = 1; // Only a block per SM will be launched by each stream
	int *idSMs = (int *)calloc(num_kernels*2, sizeof(int));
	// Launch streams
	for (int i=0; i<num_kernels; i++) {
		idSMs[2*i] = 0;
		idSMs[2*i+1] = kstubs[i]->kconf.numSMs-1;
		kstubs[i]->idSMs = idSMs+2*i;
		(kstubs[i]->launchCKEkernel)(kstubs[i]);
		cudaDeviceSynchronize();
	}
	
	
	// Reset task counter
	
	for (int i=0; i<num_kernels; i++)
		cudaMemset(kstubs[i]->d_executed_tasks, 0, sizeof(int));
	
	return 0;
}

int create_sched(t_sched *sched)
{
	cudaError_t err;
	sched->num_conc_kernels = 2; // Two concurrent kernels
	
	sched->proxy_s = (cudaStream_t *)malloc(sizeof(cudaStream_t));
	err = cudaStreamCreate(sched->proxy_s); // Stream to lauch proxy
	checkCudaErrors(err);
	
	/* Variables shared by proxy and running kernels and streams: signaling kernel evicition and reading executed tasks */ 
	checkCudaErrors(cudaMalloc((void **)&(sched->kernel_evict),  sched->num_conc_kernels * MAX_STREAMS_PER_KERNEL * sizeof(State*))); 
	checkCudaErrors(cudaMalloc((void **)&(sched->gm_cont_tasks), sched->num_conc_kernels * sizeof(State*)));
	cudaMemset(sched->kernel_evict, 0,  sched->num_conc_kernels * MAX_STREAMS_PER_KERNEL * sizeof(State *));
	cudaMemset(sched->gm_cont_tasks, 0,  sched->num_conc_kernels * sizeof(int *));
	
	/* Zero-copy memory variables of scheduler */
	checkCudaErrors(cudaHostAlloc((void **)&(sched->kernel_evict_zc), sched->num_conc_kernels * MAX_STREAMS_PER_KERNEL * sizeof(State), cudaHostAllocMapped));
	checkCudaErrors(cudaHostAlloc((void **)&(sched->cont_tasks_zc), sched->num_conc_kernels * sizeof(int), cudaHostAllocMapped));
	memset(sched->kernel_evict_zc, 0, sched->num_conc_kernels * MAX_STREAMS_PER_KERNEL * sizeof(State));
	memset(sched->cont_tasks_zc, 0, sched->num_conc_kernels * sizeof(int));
	
	return 0;
}

/*
int launch_tasks_with_proxy(int deviceId)
{	
	cudaError_t err;

	// Select device
	cudaSetDevice(deviceId);
	cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);	
	printf("Device=%s\n", deviceProp.name);

	// Create sched structure
	
	t_sched sched;
	create_sched(&sched);
	
	// Prepare coexecution
	
	t_Kernel *kid;
	t_kernel_stub **kstubs;

	
	char skid[20];	
	
	// Initializae performance of coexecution
	
	initialize_performance();
	initialize_solo_performance();
	
	// Select kernels
	int num_kernels=4;
	
	kid = (t_Kernel *)calloc(num_kernels, sizeof(t_Kernel));
	//kid[0]=MM; kid[1]=VA; kid[2]=BS; kid[3]=PF;
	kid[0]=BS; kid[1]=Reduction; kid[2]=VA; kid[3]=PF; //kid[4]=MM;
	// Create commom streams for all kernels: two for asynchronous transfers, one for preemption commands
	cudaStream_t *transfers_s;
	transfers_s = (cudaStream_t *)calloc(2, sizeof(cudaStream_t));
	
	for (int i=0;i<2;i++){
		err = cudaStreamCreate(&transfers_s[i]);
		checkCudaErrors(err);
	} 
	
	cudaStream_t preemp_s;
	checkCudaErrors(cudaStreamCreateWithFlags(&preemp_s, cudaStreamNonBlocking)); 
	
	// Create stream pool for kernel execution
	//t_strPool pool;	
	//int max_cke_kernels = 2;
	//create_stream_pool(max_cke_kernels*MAX_STREAMS_PER_KERNEL, &pool);
	
	// Create stubs 
	kstubs = (t_kernel_stub **)calloc(num_kernels, sizeof(t_kernel_stub*));
	for (int i=0; i<num_kernels; i++)
		create_stubinfo(&kstubs[i], deviceId, kid[i], transfers_s, &preemp_s);
	
	make_transfers(kstubs, num_kernels);
	
	all_nocke_execution(kstubs, num_kernels);

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
	
	// Launch proxy
	launch_generic_proxy((void *)&sched);	// Launch proxy
	
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
				coexec.kstr[0] = &kstr[run_kernel_index];
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
			
			kid_from_index(curr_kid, skid);
			printf("Solo kernel %s(%d)\n", skid, b0); 
			
			if (b0 < coexec.kstr[0]->num_streams){ // If number of streams of irst kernel must be reduced
				evict_streams(coexec.kstr[0], coexec.kstr[0]->num_streams - b0); // Evict those streams (remaining streams continue in execution)
			}
			else {
				//add_streams(coexec.kstr[0], b0 - coexec.kstr[0]->num_streams); // More streams are added
				coexec.kstr[0]->kstub->d_executed_tasks = sched.gm_cont_tasks; // Remaping to zero-copy
				coexec.kstr[0]->kstub->gm_state = sched.kernel_evict; // Remaping to zero-copy 
				launch_SMK_kernel(coexec.kstr[0], b0 - coexec.kstr[0]->num_streams); // Those new streams are launched
			}
			
		}
		else
		{
			k_select[index_new_kernel] = 1;
			coexec.kstr[0] = &kstr[run_kernel_index]; // First kernel is loaded in coexec
			coexec.kstr[1] = &kstr[index_new_kernel]; // Second kernel
			coexec.num_kernels = 2;
			
			kid_from_index(curr_kid, skid);
			printf("Two kernels %s(%d) ", skid, b0); 
			kid_from_index(next_kid, skid);
			printf(" %s(%d) \n", skid, b1); 
			
			if (b0 < coexec.kstr[0]->num_streams) { // Fist kernel is analyzed either to evict some running streams or to launch the new ones
				evict_streams(coexec.kstr[0], coexec.kstr[0]->num_streams - b0);
			}
			else{
				//add_streams(&pool, coexec.kstr[0], b0 - coexec.kstr[0]->num_streams);
				coexec.kstr[0]->kstub->d_executed_tasks = sched.gm_cont_tasks; // Remaping to zero-copy
				coexec.kstr[0]->kstub->gm_state = sched.kernel_evict; //Remaping to zero-copy 
				launch_SMK_kernel(coexec.kstr[0], b0 - coexec.kstr[0]->num_streams);
			}
	
			//add_streams(&pool, coexec.kstr[1], b1); // Second kernel is launched for the fist time: streams are added
			coexec.kstr[1]->kstub->d_executed_tasks = sched.gm_cont_tasks+1; // Remaping to zero-copy
			coexec.kstr[1]->kstub->gm_state = sched.kernel_evict+1; // Remaping to zero-copy 
			launch_SMK_kernel(coexec.kstr[1], b1);
		}
	
		int kernel_id;
		double speedup;
		wait_for_kernel_termination_with_proxy(&sched, &coexec, &kernel_id, &speedup); // Wait until one of the two runnunk kernels ends (all its streams finish)
		printf("Kernel %d finished\n", coexec.kstr[kernel_id]->kstub->id);
		cont_kernels_finished++; // Counter that indicates when all kernels have been processed
		coexec.num_kernels--;

		if (cont_kernels_finished >= num_kernels) // Check if last kernel has finished
			break; 
	
		if (kernel_id == 1){ // If second kernel finishes
		
			//k_select[index_new_kernel] = 1; // Remove from further processing
		
			coexec.kstr[1]=NULL; // Eliminate it from coexec
			curr_kid = coexec.kstr[0]->kstub->id; // Update running kernel for next iteration
		}
	
		if (kernel_id == 0){
			
			//k_select[run_kernel_index] = 1; // Remove from further processing

			coexec.kstr[0] = coexec.kstr[1]; //Move info from coexec.kstr1 to coexec.kstr0
			coexec.kstr[1] = NULL; // Remove from further processing
			run_kernel_index = index_new_kernel; // Update index of running kernel
			if (coexec.kstr[0] != NULL) // If a kernel is sill running (in coexec.kstr[0])
				curr_kid = coexec.kstr[0]->kstub->id;
		}
	}
	
	// Evict proxy
	sched.kernel_evict_zc[0] =  PROXY_EVICT;
	cudaDeviceSynchronize();
	return 0;
}*/

int kernel_in_coexec(t_kcoexec *coexec, t_kstreams *kstr, int *pos)
{
	int i;
	for (i=0; i<MAX_NUM_COEXEC_KERNELS; i++)
		if (kstr == coexec->kstr[i])
			break;

	if (i < MAX_NUM_COEXEC_KERNELS)
		*pos = i;
	else 
		*pos=-1;
	
	return 0;
}


int add_kernel_for_coexecution(t_kcoexec *coexec, t_sched * sched, t_kstreams *kstr, int num_streams, int pos)
{
	if (coexec->num_kernels >= MAX_NUM_COEXEC_KERNELS) {
		printf("Error: too many kernels for coexecution<ºn");
		return -1;
	}
	
	int i;
	for (i=0; i<MAX_NUM_COEXEC_KERNELS; i++)
		if (coexec->kstr[i] == NULL)
			break;
		
	if (i >= MAX_NUM_COEXEC_KERNELS){
		printf("Error: This could never happen\n");
		return -1;
	}
	
	if (num_streams > MAX_STREAMS_PER_KERNEL){
		printf("Too many streams\n");
		return -1;
	}
	
	coexec->kstr[i] = kstr; // add new kernel to coexec struct
	coexec->num_streams[i] = num_streams; // Number of streams for be launched for the kernel 
	coexec->queue_index[i] = pos;
	
	
	//coexec->kstr[i]->kstub->gm_state = sched->kernel_evict + i * MAX_NUM_COEXEC_KERNELS; /*Remaping to zero-copy eviction array*/
	//for (int j=0; j<num_streams; j++)
	//	sched->kernel_evict_zc[MAX_NUM_COEXEC_KERNELS *i + j] = PREP; // Se actualizará gm_state (via proxy) antes de lanzar el kernel?
	
	kstr->save_d_executed_tasks = kstr->kstub->d_executed_tasks; // Save original position of taks counter
	kstr->saved_max_persistent_blocks = kstr->kstub->kconf.max_persistent_blocks;
	coexec->kstr[i]->kstub->d_executed_tasks = sched->gm_cont_tasks+i; // Remaping kernel task counter to scheduler memory
	cudaMemcpyAsync(sched->gm_cont_tasks+i, &(kstr->save_cont_tasks),  sizeof(int),  cudaMemcpyHostToDevice, *(coexec->kstr[i]->kstub->preemp_s)); // Set counter task for this kernelid
	cudaStreamSynchronize(*(coexec->kstr[i]->kstub->preemp_s)); // This instruction could be delayed to increase concurrency
	
	coexec->num_kernels++;
	
	return 0;
}

int add_streams_to_kernel(t_kcoexec *coexec, t_sched *sched, t_kstreams *kstr, int num_streams)
{
	int i;
	for (i=0; i<MAX_NUM_COEXEC_KERNELS; i++)
		if (coexec->kstr[i] == kstr)
			break;
		
	if (i >= MAX_NUM_COEXEC_KERNELS){
		printf("Error: kernel not found\n");
		return -1;
	}
	
	//for (int j=0; j<num_streams; j++)
	//	sched->kernel_evict_zc[MAX_NUM_COEXEC_KERNELS * i + coexec->num_streams[i] + j] = PREP; // Se actualizará gm_state (via proxy) antes de lanzar el kernel?
	
	coexec->num_streams[i] += num_streams;
	
	return 0;
}

int rem_kernel_from_coexecution(t_kcoexec *coexec, t_sched *sched, t_kstreams *kstr)
{
	int i;
	for (i=0; i<MAX_NUM_COEXEC_KERNELS; i++)
		if (coexec->kstr[i] == kstr)
			break;
		
	if (i >= MAX_NUM_COEXEC_KERNELS){
		printf("Error: kernel not found\n");
		return -1;
	}
	printf("Removing kernel %d\n", kstr->kstub->id);
	
	kstr->save_cont_tasks = *(sched->cont_tasks_zc+i);
	coexec->kstr[i] = NULL; // remove kernel from coexec struct
	coexec->num_streams[i] = 0;
	coexec->queue_index[i] = -1;
	coexec->num_kernels--;
	
	//printf("Restaurando valores para kernel %d\n", kstr->kstub->id);
	kstr->kstub->d_executed_tasks = kstr->save_d_executed_tasks; // Restore original pointer to task counter
	kstr->kstub->kconf.max_persistent_blocks = kstr->saved_max_persistent_blocks; // Restore original number of persistent blocks
	
	return 0;
}

int launch_coexec(t_kcoexec *coexec)
{
	for (int i=0; i<MAX_NUM_COEXEC_KERNELS; i++){
		if (coexec->kstr[i] != NULL)
			launch_SMK_kernel(coexec->kstr[i], coexec->num_streams[i]-coexec->kstr[i]->num_streams); // Those new streams are launched
	}
	
	return 0;
}


int main(int argc, char **argv)
{
	
	//launch_tasks_with_proxy_theoretical(2);
	
	int num_kernels = 9;
	t_Kernel kid[13];
	
	/*int deviceId = atoi(argv[1]);
	kid[0] = kid_from_name(argv[2]);
	kid[1] = kid_from_name(argv[3]);
		
	fast_cke_profiling(deviceId, kid);

	return 0;
*/  
	kid[0]=VA;
	kid[1]=MM;
	kid[2]=BS;
	kid[3]=Reduction;
	kid[4]=PF;
	kid[5]=GCEDD; // Ojo: en profiling.cu se procesan tambien los tres kernels restantes de la aplicacion
	kid[6]=HST256;
	kid[7]=SPMV_CSRscalar;
	kid[8]=RCONV; // Ojo: en profiling se procesa tambien CCONV

	online_profiler_overhead(kid, 7/*num_kernels*/, 2); // SPMV y RCONV dan problemas por 
 	
	//all_profiling(kid, num_kernels, 2);
	
	//smk_check_CTA_allocation(kid, 2, 2);
	
	//greedy_coexecution(2);

	//fast_solo_profiling(2, MM);

	
	//rt_scheduler(2, atof(argv[2]), (t_Kernel)atoi(argv[1]));
	return 0;
}

