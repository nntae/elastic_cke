#include <unistd.h>
#include <string.h>
#include <helper_functions.h>   // helper functions for string parsing
#include <helper_cuda.h>   
#include "elastic_kernel.h"

/** Launch a kernel using a set of streams. Each stream will run NUM_SMs blocks (for SMK execution **/

typedef enum {Free, Busy} t_strStatus;

typedef struct{
	t_kernel_stub *kstub;
	int num_streams;
	int *index;
	State *status;
}t_kstreams;

typedef struct{
	int num_kernels;
	t_kstreams **kstr;
}t_kcoexec;

typedef struct{
	cudaStream_t *streams;
	t_strStatus *status;
	int total_streams;
	int avail_streams;
}t_strPool;

int create_coexec(t_kcoexec *coexec, int num_kernels)
{
	coexec->num_kernels = num_kernels;
	coexec->kstr = (t_kstreams **)calloc(num_kernels, sizeof(t_kstreams *));
	
	return 0;
}

int create_stream_pool(int num_streams, t_strPool *pool)
{
	cudaError_t err;
	
	// Create streams for kernels computation (streams are kernel agnostic)
	
	cudaStream_t *streams;
	streams = (cudaStream_t *)calloc(num_streams, sizeof(cudaStream_t));
	t_strStatus *status = (t_strStatus *)calloc(num_streams, sizeof(t_strStatus));
	
	for (int i=0; i < num_streams; i++){
		err = cudaStreamCreate(&streams[i]);
		checkCudaErrors(err);
		status[i] = Free;	
	}
	
	pool->streams = streams;
	pool->status = status;
	pool->total_streams = num_streams;
	pool->avail_streams = num_streams;
	
	return 0;
}

int create_kstreams(t_kernel_stub *kstub, t_kstreams *kstr)
{
	kstr->kstub = kstub;
	kstr->num_streams = 0;
	
	kstr->index = (int *)calloc(MAX_STREAMS_PER_KERNEL, sizeof(int));
	for (int i=0; i<MAX_STREAMS_PER_KERNEL; i++)
		kstr->index[i] = -1;
	kstr->status = (State *)calloc(MAX_STREAMS_PER_KERNEL, sizeof(State));
	
	for (int i=0; i<MAX_STREAMS_PER_KERNEL; i++)
		kstr->status[i] = NONE;
	
	return 0;
}

int remove_kstream(t_kstreams *kstr)
{
	free(kstr->status);
	free(kstr->index);
	
	return 0;
}

int add_streams(t_strPool *pool, t_kstreams *kstr, int num_streams)
{
	if (num_streams > pool->avail_streams){
		printf("Error: not enough available streams\n");
		return -1;
	}
	
	int curr_streams = kstr->num_streams; //offset
	
	int i,j;
	for (i=0,j=0; i<num_streams; i++) {
		while (pool->status[j] != Free && j < pool->total_streams){
			j++;
		}
		
		if (j >= pool->total_streams){
			printf("Error: available stream not found\n");
			return -1;
		}
		
		pool->status[j] = Busy;
		kstr->index[i+curr_streams] = j;
		kstr->status[i+curr_streams] = TORUN;
		
	}
	
	pool->avail_streams -= num_streams;
	
	kstr->num_streams += num_streams;
	
	return 0;
}
	
int launch_SMK_kernel(t_strPool *pool, t_kstreams *kstr)
{	
	
	// Use all the SMs
	int idSMs[2];
	idSMs[0] = 0;idSMs[1] = kstr->kstub->kconf.numSMs-1;
	kstr->kstub->idSMs = idSMs;
	
	kstr->kstub->kconf.max_persistent_blocks = 1; // Only a block per SM will be launched by each stream
	
	// Launch streams
	for (int i=0; i<kstr->num_streams; i++) {
		if (kstr->status[i] == TORUN){
			kstr->kstub->h_state[i] = PREP;
			kstr->kstub->execution_s = &(pool->streams[kstr->index[i]]);
			kstr->kstub->stream_index = i;
			(kstr->kstub->launchCKEkernel)(kstr->kstub);
			kstr->status[i] = RUNNING;
			printf("Lanzado stream %d del kernel %d\n", i, kstr->kstub->id);
		}
	}
	
	return 0;
}

int wait_for_kernel_termination(t_strPool *pool, t_kcoexec *info, int *kernelid)
{
	
	// Obtain the number of streams per kernel
	int *pending_streams = (int *) calloc(info->num_kernels, sizeof(int));
	for (int i = 0; i < info->num_kernels; i++){
		for (int j = 0; j < info->kstr[i]->num_streams; j++)
			if (info->kstr[i]->index[j] != -1)
				pending_streams[i]++;			
	}		
			
	
	while(1){
		
		// Check streams
		
		for (int i = 0; i < info->num_kernels; i++) { // For each kernel 
			for (int j=0; j < info->kstr[i]->num_streams; j++){ // For each stream
				if (info->kstr[i]->index[j] != -1)
				{
					if (cudaStreamQuery(pool->streams[info->kstr[i]->index[j]]) == cudaSuccess) {
						printf("stream %d de kernel %d finalizado\n", j, info->kstr[i]->kstub->id);
					
						/* Udpate pool info */
						pool->status[info->kstr[i]->index[j]] = Free;
						pool->avail_streams++;
						/* Update kstream info */
						info->kstr[i]->status[j] = NONE;
						info->kstr[i]->index[j] = -1;				
				
						pending_streams[i]--;
					}
				}
			}
			
			if (pending_streams[i] == 0 ) {//All the streams of a kernel have finished
				info->kstr[i]->num_streams=0;
				*kernelid = i;
				free(pending_streams);
				return 0;
			}
		}
	}				
}

int evict_streams(t_strPool *pool, t_kstreams *kstr, int num_streams)
{

	int total_streams = kstr->num_streams;

	if (num_streams > total_streams) {
		printf("Error: Too many streams to evict\n");
		return -1;
	}
	
	// Send evict commands
	for (int i=0; i<num_streams; i++){
		int index = total_streams-1-i;
		kstr->kstub->h_state[index] = TOEVICT;
		printf("--> Evicting stream %d(%d) of kernel %d\n", i, total_streams, kstr->kstub->id);
		cudaMemcpyAsync(&kstr->kstub->gm_state[index],  &kstr->kstub->h_state[index], sizeof(State), cudaMemcpyHostToDevice, *(kstr->kstub->preemp_s));
	}
	
	// Wait until stream finish
	for (int i=0; i<num_streams; i++){
		int index = total_streams-1-i;
		cudaStreamSynchronize(pool->streams[kstr->index[index]]);
		//Update pool info
		pool->status[kstr->index[index]] = Free;
		pool->avail_streams++;
		//Update kstream info
		kstr->status[i] = NONE;
		kstr->index[index] = -1;
		kstr->num_streams--;
		kstr->kstub->h_state[index] = PREP;
		cudaMemcpyAsync(&kstr->kstub->gm_state[index],  &kstr->kstub->h_state[index], sizeof(State), cudaMemcpyHostToDevice, *(kstr->kstub->preemp_s));
	}
			
	return 0;
}

int make_transfers(t_kernel_stub **kstubs, int num_kernels)
{
	for (int i=0; i<num_kernels; i++) {
		
		// Data allocation and transfers
	
		(kstubs[i]->startMallocs)((void *)(kstubs[i]));
		(kstubs[i]->startTransfers)((void *)(kstubs[i]));
	
	}
	
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
	}
	
	cudaDeviceSynchronize();
	
	// Reset task counter
	
	for (int i=0; i<num_kernels; i++)
		cudaMemset(kstubs[i]->d_executed_tasks, 0, sizeof(int));
	
	return 0;
}
	
int launch_tasks(int deviceId)
{
	t_Kernel *kid;
	t_kernel_stub **kstubs;
	cudaError_t err;
	
	char skid[20];
	
	// Select device
	cudaSetDevice(deviceId);
	cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);	
	printf("Device=%s\n", deviceProp.name);
	
	// Initializae performance of coexecution
	
	initialize_performance();
	
	// Select kernels
	int num_kernels=5;
	
	kid = (t_Kernel *)calloc(num_kernels, sizeof(t_Kernel));
	//kid[0]=MM; kid[1]=VA; kid[2]=BS; kid[3]=PF;
	kid[0]=BS; kid[1]=Reduction; kid[2]=VA; kid[3]=PF; kid[4]=MM;
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
				evict_streams(&pool, coexec.kstr[0], coexec.kstr[0]->num_streams - b0); // Evict those streams (remaining streams continue in execution)
			}
			else {
				add_streams(&pool, coexec.kstr[0], b0 - coexec.kstr[0]->num_streams); // More streams are added
				launch_SMK_kernel(&pool, coexec.kstr[0]); // Those new streams are launched
			}
			
		}
		else
		{
			k_select[index_new_kernel] = 1;
			coexec.kstr[0] = &kstr[run_kernel_index]; // Fist kernel is loaded in coexec
			coexec.kstr[1] = &kstr[index_new_kernel]; // Second kernel
			coexec.num_kernels = 2;
			
			kid_from_index(curr_kid, skid);
			printf("Two kernels %s(%d) ", skid, b0); 
			kid_from_index(next_kid, skid);
			printf(" %s(%d) \n", skid, b1); 
			
			if (b0 < coexec.kstr[0]->num_streams) { // Fist kernel is analyzed either to evict some running streams or to launch the new ones
				evict_streams(&pool, coexec.kstr[0], coexec.kstr[0]->num_streams - b0);
			}
			else{
				add_streams(&pool, coexec.kstr[0], b0 - coexec.kstr[0]->num_streams);
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
			if (coexec.kstr[0] != NULL) // If a kernel is sill running (in coexec.kstr[0])
				curr_kid = coexec.kstr[0]->kstub->id;
		}
	}
	
	return 0;
}

int main(int argc, char **argv)
{
	launch_tasks(4);
	
	return 0;
}

/*	
int main(int argc, char **argv)
{

	t_Kernel kid[2];
	
	kid[0]=MM; 
	kid[1]=BS; 
	
	cudaError_t err;
	
	// Select device
	int deviceId = 4;
	cudaSetDevice(deviceId);
	cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);	
	printf("Device=%s\n", deviceProp.name);
	
	// Create two streams for transfers (shared by all kerneels)
	
	cudaStream_t *transfers_s;
	transfers_s = (cudaStream_t *)calloc(2, sizeof(cudaStream_t));
	
	for (int i=0;i<2;i++){
		err = cudaStreamCreate(&transfers_s[i]);
		checkCudaErrors(err);
	} 
	
	// Create a stream for preemption commands
	
	cudaStream_t preemp_s;
	checkCudaErrors(cudaStreamCreateWithFlags(&preemp_s, cudaStreamNonBlocking)); 
	
	/// Create kstubs //
	t_kernel_stub *kstub0, *kstub1, *kstub2;
	create_stubinfo(&kstub0, deviceId, kid[0], transfers_s, &preemp_s);
	create_stubinfo(&kstub1, deviceId, kid[1], transfers_s, &preemp_s);
	create_stubinfo(&kstub2, deviceId, kid[1], transfers_s, &preemp_s);
	
	// Data allocation and transfers
	
	(kstub0->startMallocs)((void *)kstub0);
	(kstub0->startTransfers)((void *)kstub0);
	
	(kstub1->startMallocs)((void *)kstub1);
	(kstub1->startTransfers)((void *)kstub1);
	
	
	// Create stream pool for kernel execution
	t_strPool pool;
	
	int num_kernels = 2;
	create_stream_pool(num_kernels*MAX_STREAMS_PER_KERNEL, &pool);
	
	// Coxecution info
	
	t_kcoexec coexec;
	create_coexec(&coexec, num_kernels);
	
	
	// Create streams kernel info for coexecution
	t_kstreams kstr0, kstr1, kstr2;
	
	create_kstreams(kstub0, &kstr0);
	create_kstreams(kstub1, &kstr1);
	create_kstreams(kstub2, &kstr2);
	
	//Asssign kernel to coexecution
	
	coexec.kstr[0] = &kstr0;
	coexec.kstr[1] = &kstr1;
	
	// Assign streams to kernels
	
	add_streams(&pool, coexec.kstr[0], 4);
	add_streams(&pool, coexec.kstr[1], 2);

	// Execute kernels

	launch_SMK_kernel(&pool, coexec.kstr[0]);
	launch_SMK_kernel(&pool, coexec.kstr[1]);
	
	// Wait until all the streams of the fastest kernel finish
	
	int kernelid;
	wait_for_kernel_termination(&pool, &coexec, &kernelid);
	printf("Kernel %d finished\n", kernelid);
	
	remove_kstream(coexec.kstr[kernelid]);
	coexec.kstr[kernelid] = &kstr2;
	
	//add_streams(&pool, coexec.kstr[kernelid], 1);
	//launch_SMK_kernel(&pool, coexec.kstr[1]);
	
	
	evict_streams(&pool, coexec.kstr[0], 2);
	
	//add_streams(&pool, &coexec.kstr[0], 2);
	//launch_SMK_kernel(&pool, &coexec.kstr[0]);
	
	
	cudaDeviceSynchronize();

	return 0;
}*/