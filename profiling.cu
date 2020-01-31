#include <unistd.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <helper_functions.h>   // helper functions for string parsing
#include <helper_cuda.h>   
#include "elastic_kernel.h"
#include <math.h>

// Tables to store results for solo exectuions
extern t_smk_solo smk_info_solo[Number_of_Kernels-1]; // 
t_smt_solo smt_info_solo[Number_of_Kernels-1];

// Tables to store coexecution results

extern t_smk_coBlocks smk_info_coBlocks[Number_of_Kernels-1][Number_of_Kernels-1];
t_smt_coBlocks info_tpmsSMT[Number_of_Kernels-1][Number_of_Kernels-1]; //tpms of each kernel in coexection

// Table to store better speedups in coexecution

t_co_speedup smk_speedup[Number_of_Kernels-1][Number_of_Kernels-1];
t_co_speedup smt_speedup[Number_of_Kernels-1][Number_of_Kernels-1];


int save_profling_tables()
{
	FILE *fp;
	
	if ((fp = fopen("profiling_table.bin", "w")) == NULL) {
		printf("Cannot create file\n");
		return -1;
	}
	
	// Number of kernels 
	int n = Number_of_Kernels-1;
	fwrite (&n, 1, sizeof(int), fp);
	
	// Save t_smk_solo smk_info_solo[]
	for (int i=0; i<n; i++){
		fwrite(&smk_info_solo[i].num_configs, 1, sizeof(int), fp);
		fwrite(smk_info_solo[i].tpms, smk_info_solo[i].num_configs, sizeof(double), fp);
	}
	
	// Save t_smt_solo smt_info_solo
	for (int i=0; i<n; i++){
		fwrite(&smt_info_solo[i].num_configs, 1, sizeof(int), fp);
		fwrite(smt_info_solo[i].tpms, smt_info_solo[i].num_configs, sizeof(double), fp);
	}
	
	//Save t_smk_coBlocks smk_info_coBlocks
	
	for (int i=0; i<n; i++)
		for (int j=0; j<n; j++) {
			fwrite(smk_info_coBlocks[i][j].kid, 2, sizeof(t_Kernel), fp);
			fwrite(&smk_info_coBlocks[i][j].num_configs, 1, sizeof(int), fp);
			for (int k=0; k<smk_info_coBlocks[i][j].num_configs; k++)
				fwrite(smk_info_coBlocks[i][j].pairs[k], 2, sizeof(int), fp);
			for (int k=0; k<smk_info_coBlocks[i][j].num_configs; k++)
				fwrite(smk_info_coBlocks[i][j].tpms[k], 2, sizeof(double), fp);
		}
		
	// Save t_smt_coBlocks info_tpmsSMT
	
	for (int i=0; i<n; i++)
		for (int j=0; j<n; j++) {
			fwrite(info_tpmsSMT[i][j].kid, 2, sizeof(t_Kernel), fp);
			fwrite(&info_tpmsSMT[i][j].num_configs, 1, sizeof(int), fp);
			for (int k=0; k<info_tpmsSMT[i][j].num_configs; k++)
				fwrite(info_tpmsSMT[i][j].pairs[k], 2, sizeof(int), fp);
			for (int k=0; k<info_tpmsSMT[i][j].num_configs; k++)
				fwrite(info_tpmsSMT[i][j].tpms[k], 2, sizeof(double), fp);
		}
	
	// Save t_co_speedup smk_speedup

	for (int i=0; i<n; i++)
		for (int j=0; j<n; j++) {
			fwrite(smk_speedup[i][j].pairs, 2, sizeof(int), fp);
			fwrite(&smk_speedup[i][j].speedup, 1, sizeof(double), fp);
		}
		
	// Save t_co_speedup smt_speedup

	for (int i=0; i<n; i++)
		for (int j=0; j<n; j++) {
			fwrite(smt_speedup[i][j].pairs, 2, sizeof(int), fp);
			fwrite(&smt_speedup[i][j].speedup, 1, sizeof(double), fp);
		}
		
	fclose(fp);
	
	return 0;
}

int make_transfers(t_kernel_stub **kstubs, int num_kernels)
{
	for (int i=0; i<num_kernels; i++) {
		
		// Data allocation and transfers
	
		(kstubs[i]->startMallocs)((void *)(kstubs[i]));
		(kstubs[i]->startTransfers)((void *)(kstubs[i]));
	
	}
	
	cudaDeviceSynchronize();
	
	return 0;
}

int solo_original(t_kernel_stub *kstub, double *exectime_s)
{
	cudaEvent_t start, stop;
	float elapsedTime;
	
	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	
	kstub->launchORIkernel(kstub);
	
	cudaEventCreate(&stop);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsedTime, start, stop);
	
	*exectime_s = (double)elapsedTime/1000;
	
	return 0;
}

int smt_solo_prof(t_kernel_stub *kstub)
{
	struct timespec now;
	int idSMs[2];

	kstub->idSMs = idSMs;
	t_Kernel kid = kstub->id; 
	
	kstub->kconf.max_persistent_blocks = smk_info_solo[kstub->id].num_configs; //Max number of blocks per SM
	
	for (int sm=1; sm <=smt_info_solo[kid].num_configs; sm++) {
			
		idSMs[0]=0;idSMs[1]=sm-1;// Limit SMs for execution
			
		clock_gettime(CLOCK_REALTIME, &now);
		double time1 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
	
		(kstub->launchCKEkernel)(kstub);
		cudaDeviceSynchronize();

		clock_gettime(CLOCK_REALTIME, &now);
		double time2 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
	
		smt_info_solo[kid].tpms[sm-1] = (double)kstub->total_tasks/((time2-time1)*1000.0);
	
		int exec_tasks=0;
		cudaMemcpyAsync(kstub->d_executed_tasks, &exec_tasks, sizeof(int), cudaMemcpyHostToDevice, *kstub->preemp_s); // Reset task counter
		
		cudaDeviceSynchronize();
	
	}
	
	return 0;
}

int smk_solo_prof(t_kernel_stub *kstub)
{
	struct timespec now;
	int idSMs[2];
	double time1, time2;
		
	idSMs[0]=0;idSMs[1]=kstub->kconf.numSMs-1;
	kstub->idSMs = idSMs;	
	
	t_smk_solo *info = &smk_info_solo[kstub->id];
	
	for (int block=1; block <= info->num_configs; block++) {
			
		kstub->kconf.max_persistent_blocks = block; // Limit the max number of blocks per SM
			
		clock_gettime(CLOCK_REALTIME, &now);
		time1 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
	
		(kstub->launchCKEkernel)(kstub);
		cudaDeviceSynchronize();

		clock_gettime(CLOCK_REALTIME, &now);
		time2 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
	
		info->tpms[block-1] = (double)kstub->total_tasks/((time2-time1)*1000.0);
		
		//printf("Executinf %f\n", time2-time1);
	
		int exec_tasks=0;
		cudaMemcpyAsync(kstub->d_executed_tasks, &exec_tasks, sizeof(int), cudaMemcpyHostToDevice, *kstub->preemp_s); // Reset task counter
		
		cudaDeviceSynchronize();
	
	} 
	
	//printf("Kernel=%d time=%f\n", kstub->id, time2-time1);
	
	return 0;
}

int smt_coexec_prof(t_kernel_stub **kstub)
{
	
	struct timespec now;
	int idSMs[2][2];
	

	char skid0[20];
	char skid1[20];
	kid_from_index(kstub[0]->id, skid0);
	kid_from_index(kstub[1]->id, skid1);

	//printf("**** Profiling for kid0=%s kid1=%s ****\n", skid0, skid1);
	
	int flag = -1;
	
	int num_configs = info_tpmsSMT[kstub[0]->id][kstub[1]->id].num_configs;
		
	for (int i = 0; i<num_configs; i++) {
		
		idSMs[0][0]=0;		idSMs[0][1]=i;
		idSMs[1][0]=i+1;	idSMs[1][1]=num_configs;
		kstub[0]->idSMs = idSMs[0];
		kstub[1]->idSMs = idSMs[1];
		
		kstub[0]->kconf.max_persistent_blocks = smk_info_solo[kstub[0]->id].num_configs; //Maz number of blocks per SM
		kstub[1]->kconf.max_persistent_blocks = smk_info_solo[kstub[1]->id].num_configs;

		(kstub[0]->launchCKEkernel)(kstub[0]);
		(kstub[1]->launchCKEkernel)(kstub[1]);
		
		clock_gettime(CLOCK_REALTIME, &now);
		double time1 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;

		while (1) { // Wait until fastest kernel finishes 
			
			if (cudaStreamQuery(*kstub[0]->execution_s) == cudaSuccess) {
				flag = 0;
				break;
			}
				
			if (cudaStreamQuery(*kstub[1]->execution_s) == cudaSuccess) {
				flag = 1;
				break;
			}
			
		}

		clock_gettime(CLOCK_REALTIME, &now);
		double time2 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
		
		// Get number of executed tasks
		int exec_tasks[2];
		if (flag == 0) { 
			cudaMemcpyAsync(&exec_tasks[1], kstub[1]->d_executed_tasks, sizeof(int), cudaMemcpyDeviceToHost, *kstub[1]->preemp_s);
			exec_tasks[0] = kstub[0]->total_tasks;
		}
		else {
			cudaMemcpyAsync(&exec_tasks[0], kstub[0]->d_executed_tasks, sizeof(int), cudaMemcpyDeviceToHost, *kstub[0]->preemp_s);
			exec_tasks[1] = kstub[1]->total_tasks;
		}
		
		info_tpmsSMT[kstub[0]->id][kstub[1]->id].pairs[i][0]=i+1;
		info_tpmsSMT[kstub[0]->id][kstub[1]->id].pairs[i][1]= info_tpmsSMT[kstub[0]->id][kstub[1]->id].num_configs-i;
		
		info_tpmsSMT[kstub[0]->id][kstub[1]->id].tpms[i][0] = (double)exec_tasks[0] / ((time2 - time1) * 1000.0);
		info_tpmsSMT[kstub[0]->id][kstub[1]->id].tpms[i][1] = (double)exec_tasks[1] / ((time2 - time1) * 1000.0);
		
		// Reverse
		info_tpmsSMT[kstub[1]->id][kstub[0]->id].pairs[i][0] = info_tpmsSMT[kstub[0]->id][kstub[1]->id].pairs[i][1];
		info_tpmsSMT[kstub[1]->id][kstub[0]->id].pairs[i][1] = info_tpmsSMT[kstub[0]->id][kstub[1]->id].pairs[i][0];
		info_tpmsSMT[kstub[1]->id][kstub[0]->id].tpms[i][0] = info_tpmsSMT[kstub[0]->id][kstub[1]->id].tpms[i][1];
		info_tpmsSMT[kstub[1]->id][kstub[0]->id].tpms[i][1] = info_tpmsSMT[kstub[0]->id][kstub[1]->id].tpms[i][0];

		//printf("SMs0=%d SMs1=%d tpms0=%f, tmps1=%f\n", i, numSMs-i , info_tpmsSMT[kstub[0]->id][kstub[1]->id].tpms[i][0], info_tpmsSMT[kstub[0]->id][kstub[1]->id].tpms[i][1]);
		
		cudaDeviceSynchronize();
		
		exec_tasks[0]=0;
		exec_tasks[1]=0;
		cudaMemcpyAsync(kstub[0]->d_executed_tasks, &exec_tasks[0], sizeof(int), cudaMemcpyHostToDevice, *kstub[0]->preemp_s); // Reset task counter
		cudaMemcpyAsync(kstub[1]->d_executed_tasks, &exec_tasks[1], sizeof(int), cudaMemcpyHostToDevice, *kstub[0]->preemp_s);
		
		cudaDeviceSynchronize();

	}
	
	return 0;
}

int smk_coexec_prof(t_kernel_stub **kstub)
{
	
	struct timespec now;
	int idSMs[2];
	
	idSMs[0]=0;idSMs[1]=kstub[0]->kconf.numSMs-1;
	kstub[0]->idSMs = idSMs;
	kstub[1]->idSMs = idSMs;
	
	
	char skid0[20];
	char skid1[20];
	kid_from_index(kstub[0]->id, skid0);
	kid_from_index(kstub[1]->id, skid1);

	//printf("**** Profiling for kid0=%s kid1=%s ****\n", skid0, skid1);
	
	int flag = -1;
	
	t_smk_coBlocks *info = &smk_info_coBlocks[kstub[0]->id][kstub[1]->id];
	
	for (int i = 0; i<info->num_configs; i++) {
		
		kstub[0]->kconf.max_persistent_blocks = info->pairs[i][0]; // Only a block per SM will be launched by each stream
		kstub[1]->kconf.max_persistent_blocks = info->pairs[i][1]; // Only a block per SM will be launched by each stream

		(kstub[0]->launchCKEkernel)(kstub[0]);
		(kstub[1]->launchCKEkernel)(kstub[1]);
		
		clock_gettime(CLOCK_REALTIME, &now);
		double time1 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;

		while (1) { // Wait until fastest kernel finishes 
			
			if (cudaStreamQuery(*kstub[0]->execution_s) == cudaSuccess) {
				flag = 0;
				break;
			}
				
			if (cudaStreamQuery(*kstub[1]->execution_s) == cudaSuccess) {
				flag = 1;
				break;
			}
			
		}

		clock_gettime(CLOCK_REALTIME, &now);
		double time2 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
		
		// Get number of executed tasks
		int exec_tasks[2];
		if (flag == 0) { 
			cudaMemcpyAsync(&exec_tasks[1], kstub[1]->d_executed_tasks, sizeof(int), cudaMemcpyDeviceToHost, *kstub[1]->preemp_s);
			exec_tasks[0] = kstub[0]->total_tasks;
		}
		else {
			cudaMemcpyAsync(&exec_tasks[0], kstub[0]->d_executed_tasks, sizeof(int), cudaMemcpyDeviceToHost, *kstub[0]->preemp_s);
			exec_tasks[1] = kstub[1]->total_tasks;
		}
	
		info->tpms[i][0] = (double)exec_tasks[0] / ((time2 - time1) * 1000.0);
		info->tpms[i][1] = (double)exec_tasks[1] / ((time2 - time1) * 1000.0); 
		
		// Reverse annoation
		t_smk_coBlocks *info1 = &smk_info_coBlocks[kstub[1]->id][kstub[0]->id];
		info1->tpms[i][0] = info->tpms[i][1]; 
		info1->tpms[i][1] = info->tpms[i][0]; 
		info1->pairs[i][0] = info->pairs[i][1];
		info1->pairs[i][1] = info->pairs[i][0];
			
		printf(" *** bk0=%d bk1=%d time=%f exec_task0=%d exec_task1=%d tpms0=%f, tmps1=%f\n", info->pairs[i][0], info->pairs[i][1], (time2 - time1), exec_tasks[0], exec_tasks[1], info->tpms[i][0], info->tpms[i][1]);
		
		cudaDeviceSynchronize();
		
		exec_tasks[0]=0;
		exec_tasks[1]=0;
		cudaMemcpyAsync(kstub[0]->d_executed_tasks, &exec_tasks[0], sizeof(int), cudaMemcpyHostToDevice, *kstub[0]->preemp_s);
		cudaMemcpyAsync(kstub[1]->d_executed_tasks, &exec_tasks[1], sizeof(int), cudaMemcpyHostToDevice, *kstub[0]->preemp_s);
		
		cudaDeviceSynchronize();

	}
	
	return 0;
}

// Para como se ubican los CTAs entre los SMs
int smk_check_CTA_allocation(t_Kernel *kid, int num_kernels, int deviceId)
{
	cudaError_t err;

	// Select device
	cudaSetDevice(deviceId);
	cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);	
	printf("Device=%s\n", deviceProp.name);
	int numSMs = 28;
	
	// Load SMK initial tables
	smk_fill_coBlocks();
	smk_fill_solo();
	
	// Create SMT tables

	for (int i = 0; i < Number_of_Kernels-1; i++) {
		smt_info_solo[i].num_configs = numSMs;
		smt_info_solo[i].tpms = (double *)calloc(smt_info_solo[i].num_configs, sizeof(double));
	}


	for(int i=0; i < Number_of_Kernels-1; i++) {
		for (int j=0; j< Number_of_Kernels-1; j++) {
			info_tpmsSMT[i][j].num_configs = numSMs -1;
			int **pairs = (int **)calloc(numSMs-1, sizeof(int *));
			double **tpms = (double **)calloc(numSMs-1, sizeof(double *));
			for (int k=0; k< numSMs-1; k++) {
				pairs[k] = (int *)calloc(2, sizeof(int));
				tpms[k] = (double *)calloc(2, sizeof(double));
			}
			info_tpmsSMT[i][j].pairs = pairs;
			info_tpmsSMT[i][j].tpms = tpms;
		}
	}
	
	/** Create commom streams for all kernels: two for asynchronous transfers, one for preemption commands*/
	cudaStream_t *transfers_s;
	transfers_s = (cudaStream_t *)calloc(2, sizeof(cudaStream_t));
	
	for (int i=0;i<2;i++){
		err = cudaStreamCreate(&transfers_s[i]);
		checkCudaErrors(err);
	} 
	
	cudaStream_t preemp_s;
	checkCudaErrors(cudaStreamCreateWithFlags(&preemp_s, cudaStreamNonBlocking)); 
	
	int total_num_kernels = 0;
	for (int i=0; i<num_kernels; i++){
		total_num_kernels++;
		if (kid[i] == RCONV) total_num_kernels++;
		if (kid[i] == GCEDD) total_num_kernels += 3;
	}
	
	/** Create stubs ***/
	// Ojo la lista de kernels sólo debe ponerse el primero de una aplicacion. Los demás
	// son creados por el siguiente código
	t_kernel_stub **kstubs = (t_kernel_stub **)calloc(total_num_kernels, sizeof(t_kernel_stub*));
	for (int i=0, cont=0; i<num_kernels; i++) {	
		create_stubinfo(&kstubs[cont], deviceId, kid[i], transfers_s, &preemp_s);
		cont++;
		if (kid[i] == RCONV){ // RCONV params struct must be passed to CCONV 
			create_stubinfo_with_params(&kstubs[cont], deviceId, CCONV, transfers_s, &preemp_s, (void *)kstubs[cont-1]->params);
			cont++;
		}
		
		if (kid[i] == GCEDD){
			create_stubinfo_with_params(&kstubs[cont], deviceId, SCEDD, transfers_s, &preemp_s, (void *)kstubs[cont-1]->params);
			cont++;
			
			create_stubinfo_with_params(&kstubs[cont], deviceId, NCEDD, transfers_s, &preemp_s, (void *)kstubs[cont-2]->params);
			cont++;
			
			create_stubinfo_with_params(&kstubs[cont], deviceId, HCEDD, transfers_s, &preemp_s, (void *)kstubs[cont-3]->params);
			cont++;
		}
	}

	// make HtD transfers of all kernels
	make_transfers(kstubs, total_num_kernels);
	
	// Coexecution profiling
	t_kernel_stub *pair_kstubs[2];
	// Coexecute all tasls and extract performance
	for (int i=0; i<total_num_kernels; i++) {
		for (int j=i+1; j<total_num_kernels; j++) {
			pair_kstubs[0] = kstubs[i];
			pair_kstubs[1] = kstubs[j];
			printf("*****Profiling smk %d %d\n", kstubs[i]->id, kstubs[j]->id);
			smk_coexec_prof(pair_kstubs);
		}
	}
	
	return 0;
}

int all_profiling(t_Kernel *kid, int num_kernels, int deviceId)
{
	
	cudaError_t err;

	// Select device
	cudaSetDevice(deviceId);
	cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);	
	printf("Device=%s\n", deviceProp.name);
	int numSMs = 28;
	
	// Load SMK initial tables
	smk_fill_coBlocks();
	smk_fill_solo();
	
	// Create SMT tables

	for (int i = 0; i < Number_of_Kernels-1; i++) {
		smt_info_solo[i].num_configs = numSMs;
		smt_info_solo[i].tpms = (double *)calloc(smt_info_solo[i].num_configs, sizeof(double));
	}


	for(int i=0; i < Number_of_Kernels-1; i++) {
		for (int j=0; j< Number_of_Kernels-1; j++) {
			info_tpmsSMT[i][j].num_configs = numSMs -1;
			int **pairs = (int **)calloc(numSMs-1, sizeof(int *));
			double **tpms = (double **)calloc(numSMs-1, sizeof(double *));
			for (int k=0; k< numSMs-1; k++) {
				pairs[k] = (int *)calloc(2, sizeof(int));
				tpms[k] = (double *)calloc(2, sizeof(double));
			}
			info_tpmsSMT[i][j].pairs = pairs;
			info_tpmsSMT[i][j].tpms = tpms;
		}
	}	
	
	/** Create commom streams for all kernels: two for asynchronous transfers, one for preemption commands*/
	cudaStream_t *transfers_s;
	transfers_s = (cudaStream_t *)calloc(2, sizeof(cudaStream_t));
	
	for (int i=0;i<2;i++){
		err = cudaStreamCreate(&transfers_s[i]);
		checkCudaErrors(err);
	} 
	
	cudaStream_t preemp_s;
	checkCudaErrors(cudaStreamCreateWithFlags(&preemp_s, cudaStreamNonBlocking)); 
	
	int total_num_kernels = 0;
	for (int i=0; i<num_kernels; i++){
		total_num_kernels++;
		if (kid[i] == RCONV) total_num_kernels++;
		if (kid[i] == GCEDD) total_num_kernels += 3;
	}
	
	/** Create stubs ***/
	// Ojo la lista de kernels sólo debe ponerse el primero de una aplicacion. Los demás
	// son creados por el siguiente código
	t_kernel_stub **kstubs = (t_kernel_stub **)calloc(total_num_kernels, sizeof(t_kernel_stub*));
	for (int i=0, cont=0; i<num_kernels; i++) {	
		create_stubinfo(&kstubs[cont], deviceId, kid[i], transfers_s, &preemp_s);
		cont++;
		if (kid[i] == RCONV){ // RCONV params struct must be passed to CCONV 
			create_stubinfo_with_params(&kstubs[cont], deviceId, CCONV, transfers_s, &preemp_s, (void *)kstubs[cont-1]->params);
			cont++;
		}
		
		if (kid[i] == GCEDD){
			create_stubinfo_with_params(&kstubs[cont], deviceId, SCEDD, transfers_s, &preemp_s, (void *)kstubs[cont-1]->params);
			cont++;
			
			create_stubinfo_with_params(&kstubs[cont], deviceId, NCEDD, transfers_s, &preemp_s, (void *)kstubs[cont-2]->params);
			cont++;
			
			create_stubinfo_with_params(&kstubs[cont], deviceId, HCEDD, transfers_s, &preemp_s, (void *)kstubs[cont-3]->params);
			cont++;
		}
	}

	// make HtD transfers of all kernels
	make_transfers(kstubs, total_num_kernels);
	
	// Solo original profiling
	double *exectime_s = (double *)calloc(total_num_kernels, sizeof(double));
	for (int i=0; i<total_num_kernels; i++) // Important: is an application has several kernels, they must executed in order (data dependecies)
		solo_original(kstubs[i], &exectime_s[i]);
	
	// Solo SMK profiling and annoate results in smk_smk_info_solo table
	for (int i=0; i<total_num_kernels; i++)
		smk_solo_prof(kstubs[i]);
	
	printf("Overhead SMK wrt original\n");
	char kname[50];
	for (int i=0; i<total_num_kernels; i++) {
		t_smk_solo *info = &smk_info_solo[kstubs[i]->id];
		double smk_time = (double)kstubs[i]->total_tasks / info->tpms[info->num_configs-1] / 1000.0 ;
		kid_from_index(kstubs[i]->id, kname);
		printf("kid=%s Ori=%f SMK=%f Over=%f\n", kname, exectime_s[i], smk_time, smk_time/exectime_s[i]);
	}
	
	// Solo SMT profiling
	for (int i=0; i<total_num_kernels; i++)
		smt_solo_prof(kstubs[i]);
	
	printf("Overhead SMT wrt original\n");
	for (int i=0; i<total_num_kernels; i++) {
		t_smt_solo *info = &smt_info_solo[kstubs[i]->id];
		double smt_time = (double)kstubs[i]->total_tasks / info->tpms[info->num_configs-1] / 1000.0 ;
		kid_from_index(kstubs[i]->id, kname);
		printf("kid=%s Ori=%f SMT=%f Over=%f\n", kname, exectime_s[i], smt_time, smt_time/exectime_s[i]);
	}
	
	// Coexecution profiling
	t_kernel_stub *pair_kstubs[2];
	// Coexecute all tasls and extract performance
	for (int i=0; i<total_num_kernels; i++) {
		for (int j=i+1; j<total_num_kernels; j++) {
			pair_kstubs[0] = kstubs[i];
			pair_kstubs[1] = kstubs[j];
			//printf("Profiling smk %d %d\n", kstubs[i]->id, kstubs[j]->id);
			smk_coexec_prof(pair_kstubs);
		}
	}
	
	for (int i=0; i<total_num_kernels; i++) {
		for (int j=i+1; j<total_num_kernels; j++) {
			pair_kstubs[0] = kstubs[i];
			pair_kstubs[1] = kstubs[j];
			smt_coexec_prof(pair_kstubs);
		}
	}
	
	char skid0[20];
	char skid1[20];
	
	/* SMK: Extract better speedup */
	printf("**************** SMK Experiment *************\n"); 
	printf("Kernels \tBlocks_co	\tBlocks_solo \tBlocks_ibc \tBlocks_fair \tSpeedup_colo \tSpeedup_th \tSpeedup_real \tSpeedup_inc \tSpeedup_fair\n"); 
	double *sp_smk = (double *)calloc(100, sizeof(double));
	int b0, b1, t_b0, t_b1, t_k, f_b0, f_b1;
	for (int i=0; i<total_num_kernels; i++) {
		for (int j=i+1; j<total_num_kernels; j++) {
			t_smk_coBlocks *info_coexec= &smk_info_coBlocks[kstubs[i]->id][kstubs[j]->id];
			t_smk_solo *smk_info_solo1 = &smk_info_solo[kstubs[i]->id];
			t_smk_solo *smk_info_solo2 = &smk_info_solo[kstubs[j]->id];
			double speedup = 0, t_speedup = 0, f_speedup=0, f_d=10.0;
			int max_pos =-1, max_b0=0, max_b1=1;
			double save_speedup_inc=0;
			printf("K0=%d K1=%d\n", kstubs[i]->id, kstubs[j]->id);
			for (int k = 0; k<info_coexec->num_configs; k++) {
				double s1, s2;
				s1 = info_coexec->tpms[k][0]/smk_info_solo1->tpms[smk_info_solo1->num_configs-1];
				s2 = info_coexec->tpms[k][1]/smk_info_solo2->tpms[smk_info_solo2->num_configs-1];
				double s = s1  +s2;
				sp_smk[k] = s;
				double s_th = smk_info_solo1->tpms[info_coexec->pairs[k][0]-1]/smk_info_solo1->tpms[smk_info_solo1->num_configs-1]+smk_info_solo2->tpms[info_coexec->pairs[k][1]-1]/smk_info_solo2->tpms[smk_info_solo2->num_configs-1];
				//printf("Solo prediction %d %d %f %f %f \n", info_coexec->pairs[k][0], info_coexec->pairs[k][1], smk_info_solo1->tpms[info_coexec->pairs[k][0]-1], smk_info_solo2->tpms[info_coexec->pairs[k][1]-1], s_th);
				//printf("%f %f\n", s1, s2);
				if (s > speedup) {
					speedup = s;
					b0 = info_coexec->pairs[k][0];
					b1 = info_coexec->pairs[k][1];
				}
				if (f_d > fabs(s1-s2)){
					f_d = fabs(s1-s2);
					f_b0 = info_coexec->pairs[k][0];
					f_b1 = info_coexec->pairs[k][1];
					f_speedup = s;
				}
				
				//printf("b0=%d b1=%d s=%f ", info_coexec->pairs[k][0], info_coexec->pairs[k][1], s);
				if (s_th > t_speedup) {
					t_speedup = s_th;
					t_k = k;
					t_b0 = info_coexec->pairs[k][0];
					t_b1 = info_coexec->pairs[k][1];
				}
				if (k < info_coexec->num_configs-1){ 
					
					double val0 = info_coexec->tpms[k+1][0]/info_coexec->tpms[k][0];
					double val1 = info_coexec->tpms[k+1][1]/info_coexec->tpms[k][1];
					
					printf("%d, b0=%d, b1=%d, %f, %f\n", k+1, b0, b1, (val0+val1), (1.0/val0 + 1.0/val1));		
					
					if (val0+val1 < 2.0 && max_pos == -1){ // Chequeo de memjora global de las tpms de los dos kernels
						max_pos = k; 
						max_b0 = info_coexec->pairs[k][0];
						max_b1 = info_coexec->pairs[k][1];
						save_speedup_inc = s;
					}
					
					if (k == info_coexec->num_configs -2 && max_pos == -1) {
						max_pos = k; 
						max_b0 = info_coexec->pairs[k][0];
						max_b1 = info_coexec->pairs[k][1];
						save_speedup_inc = s;
					}
				}
				//printf("%d,%d, %f,%f\n", info_coexec->pairs[k][0], info_coexec->pairs[k][1], info_coexec->tpms[k][0], info_coexec->tpms[k][1]);
			}
			
			// Annotate best speedup 
			smk_speedup[kstubs[i]->id][kstubs[j]->id].pairs[0] = b0;
			smk_speedup[kstubs[i]->id][kstubs[j]->id].pairs[1] = b1;
			smk_speedup[kstubs[i]->id][kstubs[j]->id].speedup = speedup;
			
			// Reverse
			smk_speedup[kstubs[j]->id][kstubs[i]->id].pairs[0] = b1;
			smk_speedup[kstubs[j]->id][kstubs[i]->id].pairs[1] = b0;
			smk_speedup[kstubs[j]->id][kstubs[i]->id].speedup = speedup;
			
			kid_from_index(kstubs[i]->id, skid0);
			kid_from_index(kstubs[j]->id, skid1);
			//printf("Best SMK real configuration for (%s,%s) -> (%d, %d) con s=%f\n", skid0, skid1, b0, b1, speedup);
			//printf("Best SMK theoretical configuration for (%s,%s) -> (%d, %d) con s=%f\n\n", skid0, skid1, t_b0, t_b1, t_speedup);
			printf("%s/%s \t %d_%d \t %d_%d \t %d_%d \t %d_%d \t %f \t %f \t %f \t%f \t%f\n", skid0, skid1, b0, b1, t_b0, t_b1, max_b0, max_b1, f_b0, f_b1, speedup, t_speedup, sp_smk[t_k], save_speedup_inc, f_speedup);  

		}
		
		
	}
	
	free(sp_smk);
					
	/* SMT: Extract better speedup */
	printf("\n**************** SMT Experiment *************\n"); 
	printf("Kernels \tBlocks_co	\tBlocks_solo \tBlocks_inc \tSpeedup_colo \tSpeedup_th \tSpeedup_real \tSpeedup_inc\n");   
	double *sp = (double *)calloc(numSMs, sizeof(double));
	int sms=0, t_sms=0;
	for (int i=0; i<total_num_kernels; i++) {
		for (int j=i+1; j<total_num_kernels; j++) {
			double speedup = 0, t_speedup = 0;
			int num_configs = info_tpmsSMT[kstubs[i]->id][kstubs[j]->id].num_configs;
			int max_pos = -1;
			for (int k = 0; k<num_configs; k++) {
				
				double tpms0 = info_tpmsSMT[kstubs[i]->id][kstubs[j]->id].tpms[k][0];
				double tpms1 = info_tpmsSMT[kstubs[i]->id][kstubs[j]->id].tpms[k][1];
				
				double tpms0_solo = smt_info_solo[kstubs[i]->id].tpms[smt_info_solo[kstubs[i]->id].num_configs-1];
				double tpms1_solo = smt_info_solo[kstubs[j]->id].tpms[smt_info_solo[kstubs[j]->id].num_configs-1];
				
				double s = tpms0/tpms0_solo+tpms1/tpms1_solo;
				sp[k] = s;
				double s_th = smt_info_solo[kstubs[i]->id].tpms[k]/tpms0_solo+smt_info_solo[kstubs[j]->id].tpms[num_configs - k]/tpms1_solo;
				//printf("-->%d, %f\n", k, s);
				if (s > speedup) {
					speedup = s;
					sms = k+1;
				}
				if (s_th > t_speedup) {
					t_speedup = s_th;
					t_sms = k;
				}		
				 
				if ( k < num_configs -1){
				
					double val0 = info_tpmsSMT[kstubs[i]->id][kstubs[j]->id].tpms[k+1][0]/info_tpmsSMT[kstubs[i]->id][kstubs[j]->id].tpms[k][0];
					double val1 = info_tpmsSMT[kstubs[i]->id][kstubs[j]->id].tpms[k+1][1]/info_tpmsSMT[kstubs[i]->id][kstubs[j]->id].tpms[k][1];
					
					//printf("%d, %f\n", k+1, val0+val1);		
					
					if (max_pos == -1){ 
						if (val0+val1 <2.0)
						max_pos = k; 
					}
				}
				
				
				//printf("%d,%d, %f,%f\n", k+1, numSMs-k-1, tpms0, tpms1);

			}
			
			// Annotate best speedup 
			smt_speedup[kstubs[i]->id][kstubs[j]->id].pairs[0] = sms;
			smt_speedup[kstubs[i]->id][kstubs[j]->id].pairs[1] = numSMs-(sms);
			smt_speedup[kstubs[i]->id][kstubs[j]->id].speedup = speedup;
			
			// Reverse
			smt_speedup[kstubs[j]->id][kstubs[i]->id].pairs[0] = numSMs-(sms);
			smt_speedup[kstubs[j]->id][kstubs[i]->id].pairs[1] = sms;
			smt_speedup[kstubs[j]->id][kstubs[i]->id].speedup = speedup;
			
			kid_from_index(kstubs[i]->id, skid0);
			kid_from_index(kstubs[j]->id, skid1);
			//printf("Best SMT real configuration for (%s,%s) -> (%d, %d) con s=%f\n", skid0, skid1, sms, numSMs-sms, speedup);
			//printf("Best SMT theoretical configuration for (%s,%s) -> (%d, %d) con s=%f\n\n", skid0, skid1, t_sms, numSMs-t_sms, t_speedup);
			printf("%s/%s \t %d_%d \t %d_%d \t%d_%d \t %f \t %f \t %f %f\n", skid0, skid1, sms, numSMs-sms, t_sms, numSMs-t_sms, max_pos+1, numSMs-max_pos-1, speedup, t_speedup, sp[t_sms-1], sp[max_pos]);  
			//printf("Nuevo %s/%s \t %d_%d \t %f\n", skid0, skid1, max_pos+1, numSMs-max_pos-1, sp[max_pos]);  
			
		}
	}

	save_profling_tables();
	
	printf("\n\n");
	
	free(sp);
			
	return 0;
}
	

	
	