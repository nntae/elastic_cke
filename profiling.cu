#include <unistd.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <helper_functions.h>   // helper functions for string parsing
#include <helper_cuda.h>   
#include "elastic_kernel.h"

extern t_coBlocks info_coBlocks[Number_of_Kernels-1][Number_of_Kernels-1];
extern t_solo info_solo[Number_of_Kernels-1];
double info_tpmsSMT[Number_of_Kernels-1][Number_of_Kernels-1][28][2]; //tpms of each kernel in coexection
double SMTsolo_tpms[Number_of_Kernels-1][28];

int solo_execution_prof(t_kernel_stub *kstub, double *tpms)
{	
	struct timespec now;
	int idSMs[2];
	
	idSMs[0]=0;idSMs[1]=kstub->kconf.numSMs-1;
	kstub->idSMs = idSMs;
	
	clock_gettime(CLOCK_REALTIME, &now);
 	double time1 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
	
	(kstub->launchCKEkernel)(kstub);
	cudaDeviceSynchronize();

	clock_gettime(CLOCK_REALTIME, &now);
 	double time2 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
	
	*tpms = (double)kstub->total_tasks/((time2-time1)*1000.0);
	
	
	
	int exec_tasks=0;
	cudaMemcpyAsync(kstub->d_executed_tasks, &exec_tasks, sizeof(int), cudaMemcpyHostToDevice, *kstub->preemp_s);
	
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

int smt_solo_prof(t_kernel_stub *kstub, int numSMs)
{
	struct timespec now;
	int idSMs[2];

	kstub->idSMs = idSMs;	
	
	kstub->kconf.max_persistent_blocks = info_solo[kstub->id].num_configs; //Max number of blocks per SM
	
	for (int sm=1; sm <=numSMs; sm++) {
			
		idSMs[0]=0;idSMs[1]=sm-1;// Limit SMs for execution
			
		clock_gettime(CLOCK_REALTIME, &now);
		double time1 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
	
		(kstub->launchCKEkernel)(kstub);
		cudaDeviceSynchronize();

		clock_gettime(CLOCK_REALTIME, &now);
		double time2 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
	
		SMTsolo_tpms[kstub->id][sm-1] = (double)kstub->total_tasks/((time2-time1)*1000.0);
	
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
	
	t_solo *info = &info_solo[kstub->id];
	
	for (int block=1; block <= info->num_configs; block++) {
			
		kstub->kconf.max_persistent_blocks = block; // Limit the max number of blocks per SM
			
		clock_gettime(CLOCK_REALTIME, &now);
		time1 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
	
		(kstub->launchCKEkernel)(kstub);
		cudaDeviceSynchronize();

		clock_gettime(CLOCK_REALTIME, &now);
		time2 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
	
		info->tpms[block-1] = (double)kstub->total_tasks/((time2-time1)*1000.0);
	
		int exec_tasks=0;
		cudaMemcpyAsync(kstub->d_executed_tasks, &exec_tasks, sizeof(int), cudaMemcpyHostToDevice, *kstub->preemp_s); // Reset task counter
		
		cudaDeviceSynchronize();
	
	} 
	
	//printf("Kernel=%d time=%f\n", kstub->id, time2-time1);
	
	return 0;
}

int smt_coexec_prof(t_kernel_stub **kstub, int numSMs)
{
	
	struct timespec now;
	int idSMs[2][2];
	

	char skid0[20];
	char skid1[20];
	kid_from_index(kstub[0]->id, skid0);
	kid_from_index(kstub[1]->id, skid1);

	//printf("**** Profiling for kid0=%s kid1=%s ****\n", skid0, skid1);
	
	int flag = -1;
		
	for (int i = 1; i<numSMs; i++) {
		
		idSMs[0][0]=0;	idSMs[0][1]=i-1;
		idSMs[1][0]=i;	idSMs[1][1]=numSMs -1;
		kstub[0]->idSMs = idSMs[0];
		kstub[1]->idSMs = idSMs[1];
		
		kstub[0]->kconf.max_persistent_blocks = info_solo[kstub[0]->id].num_configs; //Maz number of blocks per SM
		kstub[1]->kconf.max_persistent_blocks = info_solo[kstub[1]->id].num_configs;

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
		
		info_tpmsSMT[kstub[0]->id][kstub[1]->id][i-1][0] = (double)exec_tasks[0] / ((time2 - time1) * 1000.0);
		info_tpmsSMT[kstub[0]->id][kstub[1]->id][i-1][1] = (double)exec_tasks[1] / ((time2 - time1) * 1000.0);
		
		//printf("SMs0=%d SMs1=%d tpms0=%f, tmps1=%f\n", i, numSMs-i , info_tpmsSMT[kstub[0]->id][kstub[1]->id][i-1][0], info_tpmsSMT[kstub[0]->id][kstub[1]->id][i-1][1]);
		
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
	
	t_coBlocks *info = &info_coBlocks[kstub[0]->id][kstub[1]->id];
	
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
		
		
		//printf("bk0=%d bk1=%d tpms0=%f, tmps1=%f\n", info->pairs[i][0], info->pairs[i][1], info->tpms[i][0], info->tpms[i][1]);
		
		cudaDeviceSynchronize();
		
		exec_tasks[0]=0;
		exec_tasks[1]=0;
		cudaMemcpyAsync(kstub[0]->d_executed_tasks, &exec_tasks[0], sizeof(int), cudaMemcpyHostToDevice, *kstub[0]->preemp_s);
		cudaMemcpyAsync(kstub[1]->d_executed_tasks, &exec_tasks[1], sizeof(int), cudaMemcpyHostToDevice, *kstub[0]->preemp_s);
		
		cudaDeviceSynchronize();

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
	
	// Load initial tables
	fill_coBlocks();
	fill_solo();
	
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
	t_kernel_stub **kstubs = (t_kernel_stub **)calloc(num_kernels, sizeof(t_kernel_stub*));
	for (int i=0; i<num_kernels; i++)
		create_stubinfo(&kstubs[i], deviceId, kid[i], transfers_s, &preemp_s);
	
	// make HtD transfers of all kernels
	make_transfers(kstubs, num_kernels);
	
	// Solo original profiling
	double *exectime_s = (double *)calloc(num_kernels, sizeof(double));
	for (int i=0; i<num_kernels; i++)
		solo_original(kstubs[i], &exectime_s[i]);
	
	// Solo SMK profiling
	for (int i=0; i<num_kernels; i++)
		smk_solo_prof(kstubs[i]);
	
	printf("Overhead SMK wrt original\n");
	for (int i=0; i<num_kernels; i++) {
		t_solo *info = &info_solo[kstubs[i]->id];
		double smk_time = (double)kstubs[i]->total_tasks / info->tpms[info->num_configs-1] / 1000.0 ;
		printf("K=%d Ori=%f SMK=%f Over=%f\n", i, exectime_s[i], smk_time, smk_time/exectime_s[i]);
	}
	
	return 0;
	
	// Solo SMT profiling
	for (int i=0; i<num_kernels; i++)
		smt_solo_prof(kstubs[i], 28);
	
	// Coexecution profiling
	t_kernel_stub *pair_kstubs[2];
	// Coexecute all tasls and extract performance
	for (int i=0; i<num_kernels; i++) {
		for (int j=i+1; j<num_kernels; j++) {
			pair_kstubs[0] = kstubs[i];
			pair_kstubs[1] = kstubs[j];
			//printf("Profiling smk %d %d\n", kstubs[i]->id, kstubs[j]->id);
			smk_coexec_prof(pair_kstubs);
		}
	}
	
	int numSMs = 28;
	for (int i=0; i<num_kernels; i++) {
		for (int j=i+1; j<num_kernels; j++) {
			pair_kstubs[0] = kstubs[i];
			pair_kstubs[1] = kstubs[j];
			smt_coexec_prof(pair_kstubs, numSMs);
		}
	}
	
	char skid0[20];
	char skid1[20];
	
	/* SMK: Extract better speedup */
	printf("**************** SMK Experiment *************\n"); 
	printf("Kernels \tBlocks_co	\tBlocks_solo \tSpeedup_colo \tSpeedup_th \tSpeedup_real\n"); 
	double *sp_smk = (double *)calloc(100, sizeof(double));
	int b0, b1, t_b0, t_b1, t_k;
	for (int i=0; i<num_kernels; i++) {
		for (int j=i+1; j<num_kernels; j++) {
			t_coBlocks *info_coexec= &info_coBlocks[kstubs[i]->id][kstubs[j]->id];
			t_solo *info_solo1 = &info_solo[kstubs[i]->id];
			t_solo *info_solo2 = &info_solo[kstubs[j]->id];
			double speedup = 0, t_speedup = 0;
			for (int k = 0; k<info_coexec->num_configs; k++) {
				double s = info_coexec->tpms[k][0]/info_solo1->tpms[info_solo1->num_configs-1]+info_coexec->tpms[k][1]/info_solo2->tpms[info_solo2->num_configs-1];
				sp_smk[k] = s;
				double s_th = info_solo1->tpms[info_coexec->pairs[k][0]-1]/info_solo1->tpms[info_solo1->num_configs-1]+info_solo2->tpms[info_coexec->pairs[k][1]-1]/info_solo2->tpms[info_solo2->num_configs-1];
				if (s > speedup) {
					speedup = s;
					b0 = info_coexec->pairs[k][0];
					b1 = info_coexec->pairs[k][1];
				}
				if (s_th > t_speedup) {
					t_speedup = s_th;
					t_k = k;
					t_b0 = info_coexec->pairs[k][0];
					t_b1 = info_coexec->pairs[k][1];
				}	
			}
			
			kid_from_index(kstubs[i]->id, skid0);
			kid_from_index(kstubs[j]->id, skid1);
			//printf("Best SMK real configuration for (%s,%s) -> (%d, %d) con s=%f\n", skid0, skid1, b0, b1, speedup);
			//printf("Best SMK theoretical configuration for (%s,%s) -> (%d, %d) con s=%f\n\n", skid0, skid1, t_b0, t_b1, t_speedup);
			printf("%s/%s \t %d_%d \t %d_%d \t %f \t %f \t %f\n", skid0, skid1, b0, b1, t_b0, t_b1, speedup, t_speedup, sp_smk[t_k]);  

		}
	}
	
	free(sp_smk);
					
	/* SMT: Extract better speedup */
	printf("\n\n**************** SMT Experiment *************\n"); 
	printf("Kernels \tBlocks_co	\tBlocks_solo \tSpeedup_colo \tSpeedup_th \tSpeedup_real\n");   
	printf("\n");
	double *sp = (double *)calloc(numSMs, sizeof(double));
	int sms=0, t_sms=0;
	for (int i=0; i<num_kernels; i++) {
		for (int j=i+1; j<num_kernels; j++) {
			double speedup = 0, t_speedup = 0;
			for (int k = 1; k<numSMs; k++) {
				
				double tmps0 = info_tpmsSMT[kstubs[i]->id][kstubs[j]->id][k-1][0];
				double tmps1 = info_tpmsSMT[kstubs[i]->id][kstubs[j]->id][k-1][1];
				
				t_solo *info_solo1 = &info_solo[kstubs[i]->id];
				t_solo *info_solo2 = &info_solo[kstubs[j]->id];
				
				double s = tmps0/info_solo1->tpms[info_solo1->num_configs-1]+tmps1/info_solo2->tpms[info_solo2->num_configs-1];
				sp[k-1] = s;
				double s_th = SMTsolo_tpms[kstubs[i]->id][k-1]/info_solo1->tpms[info_solo1->num_configs-1]+SMTsolo_tpms[kstubs[j]->id][numSMs-k-1]/info_solo2->tpms[info_solo2->num_configs-1];
				//printf("-->%d, %f\n", k, s);
				if (s > speedup) {
					speedup = s;
					sms = k;
				}
				if (s_th > t_speedup) {
					t_speedup = s_th;
					t_sms = k;
				}	
			}
			
			kid_from_index(kstubs[i]->id, skid0);
			kid_from_index(kstubs[j]->id, skid1);
			//printf("Best SMT real configuration for (%s,%s) -> (%d, %d) con s=%f\n", skid0, skid1, sms, numSMs-sms, speedup);
			//printf("Best SMT theoretical configuration for (%s,%s) -> (%d, %d) con s=%f\n\n", skid0, skid1, t_sms, numSMs-t_sms, t_speedup);
			printf("%s/%s \t %d_%d \t %d_%d \t %f \t %f \t %f\n", skid0, skid1, sms, numSMs-sms, t_sms, numSMs-t_sms, speedup, t_speedup, sp[t_sms-1]);  
			
		}
	}		
	
	free(sp);
			
	return 0;
}
	

	
	