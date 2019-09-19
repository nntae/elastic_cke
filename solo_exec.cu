#include <unistd.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <cuda_profiler_api.h>
#include <helper_functions.h>   // helper functions for string parsing
#include <helper_cuda.h>   
#include "elastic_kernel.h"

int preemp_overhead(t_kernel_stub *kstub, int iteraciones, double *over)
{
	cudaError_t err;
	struct timespec now;
	double time1, time2;
	int idSMs[2];
	//cudaMemcpyAsync(&kstub->gm_state[0], &kstub->h_state[0], sizeof(State), cudaMemcpyHostToDevice, *(kstub->preemp_s)); 
	//cudaDeviceSynchronize();

	// make HtD transfers 
	/*(kstub->startMallocs)((void *)(kstub));
	cudaMemcpyAsync(&kstub->gm_state[0], &kstub->h_state[0], sizeof(State), cudaMemcpyHostToDevice, *(kstub->preemp_s)); 
	cudaDeviceSynchronize();
	(kstub->startTransfers)((void *)(kstub));	
	cudaDeviceSynchronize();*/
	
	

	// Use max number of BpSM and all SMs
	idSMs[0]=0;idSMs[1]=kstub->kconf.numSMs-1;
	kstub->idSMs = idSMs;	
	
	// Index to indicate global memory postion with Ste value 
	kstub->stream_index = 0;
	
	int max_BpSM = kstub->kconf.max_persistent_blocks;
		
	double acc_over;
	for (int j=1; j <= max_BpSM; j++) { // Test different BpSM
	
		kstub->kconf.max_persistent_blocks = j;
		acc_over = 0;
		
		for (int i =0; i< iteraciones; i++) {
	
			// Launch kernel supporting prediction
			(kstub->launchCKEkernel)(kstub);
	
			// wait 500 us before eviction
			clock_gettime(CLOCK_REALTIME, &now);
			time1 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
			time2 = time1;
			while ((time2-time1) < 0.001){
				clock_gettime(CLOCK_REALTIME, &now);
				time2 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
			}
	
			// Eviction command
			kstub->h_state[0] = TOEVICT;
			err = cudaMemcpyAsync(&kstub->gm_state[0], &kstub->h_state[0], sizeof(State), cudaMemcpyHostToDevice, *(kstub->preemp_s));  // Signal TOEVICT State
			checkCudaErrors(err);
			cudaDeviceSynchronize(); // wait kernel eviction
	
			clock_gettime(CLOCK_REALTIME, &now);
			time1 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
		
			// Reset kernel status for reexecution
			kstub->h_state[0] = READY;
			err = cudaMemcpyAsync(&kstub->gm_state[0], &kstub->h_state[0], sizeof(State), cudaMemcpyHostToDevice, *(kstub->preemp_s));  
			cudaDeviceSynchronize();
		
			// Reset task counter for reexecution
			int exec_tasks = 0;
			cudaMemcpyAsync(kstub->d_executed_tasks, &exec_tasks, sizeof(int), cudaMemcpyHostToDevice, *kstub->preemp_s); // Reset task counter
			cudaDeviceSynchronize();
		
			acc_over += time1-time2;
		}
		
		char kname[50];
		kid_from_index(kstub->id, kname);
		printf("Overhead %s with %d BpSM = %f\n", kname, j, acc_over / (double)iteraciones);
		
	}
	
	*over = acc_over / (double)iteraciones ;

	return 0;
}

int execution_with_BpSM(t_kernel_stub *kstub, int BpSM, int iterations, double *tpms)
{
	
	
	

	// make HtD transfers 
	(kstub->startMallocs)((void *)(kstub));
	(kstub->startTransfers)((void *)(kstub));	
	cudaDeviceSynchronize();
	
	struct timespec now;
	int idSMs[2];
	double time1, time2;
		
	idSMs[0]=0;idSMs[1]=kstub->kconf.numSMs-1;
	kstub->idSMs = idSMs;	
	
	kstub->kconf.max_persistent_blocks = BpSM; // Limit the max number of blocks per SM
	
	double acc_t = 0;
	for (int i=0; i<iterations; i++) {
	
		cudaProfilerStart();
	
		clock_gettime(CLOCK_REALTIME, &now);
		time1 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
	
		(kstub->launchCKEkernel)((void *)kstub);
		cudaDeviceSynchronize();

		clock_gettime(CLOCK_REALTIME, &now);
		time2 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
		
		cudaProfilerStop();
	
		acc_t += (time2-time1);
		int exec_tasks=0;
		cudaMemcpyAsync(kstub->d_executed_tasks, &exec_tasks, sizeof(int), cudaMemcpyHostToDevice, *kstub->preemp_s); // Reset task counter
		
		cudaDeviceSynchronize();
	}

	printf("BSP=%d Time=%f Tpms=%f\n", BpSM, acc_t/(double)iterations, (double)kstub->total_tasks/(1000.0*(acc_t/(double)iterations)));
	
	*tpms = (double)kstub->total_tasks/(1000.0*(acc_t/(double)iterations));

	return 0;
}

// The way a kstub is created depends on the kernel id: first application kernel and following ones
t_kernel_stub *kstub_creation(t_Kernel kid, int deviceId)
{
	char kname[30];
	
	/** Create commom streams for all kernels: two for asynchronous transfers, one for preemption commands*/
	cudaStream_t *transfers_s;
	transfers_s = (cudaStream_t *)calloc(2, sizeof(cudaStream_t));
	
	for (int i=0;i<2;i++){
		cudaStreamCreate(&transfers_s[i]);
	} 
	
	cudaStream_t preemp_s;
	checkCudaErrors(cudaStreamCreateWithFlags(&preemp_s, cudaStreamNonBlocking));
	
	kid_from_index(kid, kname);
	
	int flag_create_kstub_with_params = 0;
	t_kernel_stub *kstub;
	
	if (strcmp(kname, "MM") == 0){
		kid = MM;
	}
	
	if (strcmp(kname, "BS") == 0){
		kid = BS;
	}
		
	if (strcmp(kname, "VA") == 0){
		kid = VA;
	}
	
	if (strcmp(kname, "PF") == 0){
		kid = PF;
	}
	
	if (strcmp(kname, "SPMV_CSRscalar") == 0){
		kid = SPMV_CSRscalar;
	}
	
	if (strcmp(kname, "PF") == 0){
		kid = PF;
	}
	
	if (strcmp(kname, "RCONV") == 0){
		kid = RCONV;
	}
	
	if (strcmp(kname, "CCONV") == 0){
		// First RCONV must be executed 
		kid = RCONV;
		
		/** Create stub ***/
		create_stubinfo(&kstub, deviceId, kid, transfers_s, &preemp_s);
	
		// make HtD transfers 
		(kstub->startMallocs)((void *)(kstub));
		(kstub->startTransfers)((void *)(kstub));	
		cudaDeviceSynchronize();
		
		// Exec 
		int idSMs[2];
		
		idSMs[0]=0;idSMs[1]=kstub->kconf.numSMs-1;
		kstub->idSMs = idSMs;
		(kstub->launchCKEkernel)(kstub);
		cudaDeviceSynchronize();
		
		flag_create_kstub_with_params = 1;
		kid = CCONV;
	}
	
	if (strcmp(kname, "GCEDD") == 0){
		kid = GCEDD;
	}
	
	if (strcmp(kname, "SCEDD") == 0){
		
		// First GCEDD must be executed 
		kid = GCEDD;
		/** Create stub ***/
		create_stubinfo(&kstub, deviceId, kid, transfers_s, &preemp_s);
	
		// make HtD transfers 
		(kstub->startMallocs)((void *)(kstub));
		(kstub->startTransfers)((void *)(kstub));	
		cudaDeviceSynchronize();
		
		// Exec 
		int idSMs[2];
		
		idSMs[0]=0;idSMs[1]=kstub->kconf.numSMs-1;
		kstub->idSMs = idSMs;
		(kstub->launchCKEkernel)(kstub);
		cudaDeviceSynchronize();
		
		flag_create_kstub_with_params = 1;
		kid = SCEDD;

	}
	
		if (strcmp(kname, "NCEDD") == 0){
		
		// First GCEDD must be executed 
		kid = GCEDD;
		/** Create stub ***/
		create_stubinfo(&kstub, deviceId, kid, transfers_s, &preemp_s);
	
		// make HtD transfers 
		(kstub->startMallocs)((void *)(kstub));
		(kstub->startTransfers)((void *)(kstub));	
		cudaDeviceSynchronize();
		
		// Exec 
		int idSMs[2];
		
		idSMs[0]=0;idSMs[1]=kstub->kconf.numSMs-1;
		kstub->idSMs = idSMs;
		(kstub->launchCKEkernel)(kstub);
		cudaDeviceSynchronize();
		
		// Then, next previous kernel
		t_Kernel kidt1 = SCEDD;
		t_kernel_stub * kstubt1;
		
		create_stubinfo_with_params(&kstubt1, deviceId, kidt1, transfers_s, &preemp_s, (void *)kstub->params);
		
		// make HtD transfers 
		(kstubt1->startMallocs)((void *)(kstubt1));
		(kstubt1->startTransfers)((void *)(kstubt1));	
		cudaDeviceSynchronize();
		
		// Exec
		idSMs[0]=0;idSMs[1]=kstubt1->kconf.numSMs-1;
		kstubt1->idSMs = idSMs;
		(kstubt1->launchCKEkernel)(kstubt1);
		cudaDeviceSynchronize();
		
		flag_create_kstub_with_params = 1;
		kid = NCEDD;
	}
	
	if (strcmp(kname, "HCEDD") == 0){
		
		// First GCEDD must be executed 
		kid = GCEDD;
		/** Create stub ***/
		create_stubinfo(&kstub, deviceId, kid, transfers_s, &preemp_s);
	
		// make HtD transfers 
		(kstub->startMallocs)((void *)(kstub));
		(kstub->startTransfers)((void *)(kstub));	
		cudaDeviceSynchronize();
		
		// Exec 
		int idSMs[2];
		
		idSMs[0]=0;idSMs[1]=kstub->kconf.numSMs-1;
		kstub->idSMs = idSMs;
		(kstub->launchCKEkernel)(kstub);
		cudaDeviceSynchronize();
		
		// Then, next previous kernel
		t_Kernel kidt1 = SCEDD;
		t_kernel_stub * kstubt1;
		
		create_stubinfo_with_params(&kstubt1, deviceId, kidt1, transfers_s, &preemp_s, (void *)kstub->params);
		
		// make HtD transfers 
		(kstubt1->startMallocs)((void *)(kstubt1));
		(kstubt1->startTransfers)((void *)(kstubt1));	
		cudaDeviceSynchronize();
		
		// Exec
		idSMs[0]=0;idSMs[1]=kstubt1->kconf.numSMs-1;
		kstubt1->idSMs = idSMs;
		(kstubt1->launchCKEkernel)(kstubt1);
		cudaDeviceSynchronize();
		
		// Finally, a third kernel
		
		t_Kernel kidt2 = NCEDD;
		t_kernel_stub * kstubt2;
		
		create_stubinfo_with_params(&kstubt2, deviceId, kidt2, transfers_s, &preemp_s, (void *)kstub->params);
		
		// make HtD transfers 
		(kstubt2->startMallocs)((void *)(kstubt2));
		(kstubt2->startTransfers)((void *)(kstubt2));	
		cudaDeviceSynchronize();
		
		// Exec
		idSMs[0]=0;idSMs[1]=kstubt2->kconf.numSMs-1;
		kstubt2->idSMs = idSMs;
		(kstubt2->launchCKEkernel)(kstubt2);
		cudaDeviceSynchronize();
		
		flag_create_kstub_with_params = 1;
		kid = HCEDD;
	}
		
	
	if (strcmp(kname, "Reduction") == 0){
		kid = Reduction;
	}
	
	if (strcmp(kname, "HST256") == 0){
		kid = HST256;
	}

	if (kid < 0){
		printf("Error: Wrong kernel name\n");
		return NULL;
	}
	
	/** Create stub ***/
	t_kernel_stub *kstub1;
	
	if (flag_create_kstub_with_params == 0)
		create_stubinfo(&kstub1, deviceId, kid, transfers_s, &preemp_s);
	else
		create_stubinfo_with_params(&kstub1, deviceId, kid, transfers_s, &preemp_s, (void *)kstub->params);
	
	return kstub1;
}


	
  
int main(int argc, char **argv)
{
	t_kernel_stub *kstub;
	
	if (argc <4) {
		printf("Error: program must run as follows: sheduler device_id kernel_name num_BpSM iterations\n");
		return -1;
	}
	
	int deviceId = atoi(argv[1]);
	
	cudaSetDevice(deviceId);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId); 
	
	/*kstub = kstub_creation((t_Kernel)atoi(argv[2]), atoi(argv[1]));
	cudaMemcpyAsync(&kstub->gm_state[0], &kstub->h_state[0], sizeof(State), cudaMemcpyHostToDevice, *(kstub->preemp_s)); 
	double over;
	preemp_overhead(kstub, atoi(argv[4]), &over);
	
	return 0;*/
	/** Create commom streams for all kernels: two for asynchronous transfers, one for preemption commands*/
	cudaStream_t *transfers_s;
	transfers_s = (cudaStream_t *)calloc(2, sizeof(cudaStream_t));
	
	for (int i=0;i<2;i++){
		cudaStreamCreate(&transfers_s[i]);
	} 
	
	cudaStream_t preemp_s;
	checkCudaErrors(cudaStreamCreateWithFlags(&preemp_s, cudaStreamNonBlocking)); 
	
	//cudaProfilerInitialize("profiling_options.txt", "sal_prof", cudaKeyValuePair );

	t_Kernel kid=(t_Kernel)-1;
	
	int flag_create_kstub_with_params = 0;
	
	if (strcmp(argv[2], "MM") == 0){
		kid = MM;
	}
	
	if (strcmp(argv[2], "BS") == 0){
		kid = BS;
	}
		
	if (strcmp(argv[2], "VA") == 0){
		kid = VA;
	}
	
	if (strcmp(argv[2], "PF") == 0){
		kid = PF;
	}
	
	if (strcmp(argv[2], "SPMV_CSRscalar") == 0){
		kid = SPMV_CSRscalar;
	}
	
	if (strcmp(argv[2], "PF") == 0){
		kid = PF;
	}
	
	if (strcmp(argv[2], "RCONV") == 0){
		kid = RCONV;
	}
	
	if (strcmp(argv[2], "CCONV") == 0){
		
		// First RCONV must be executed 
		kid = RCONV;
		
		/** Create stub ***/
		create_stubinfo(&kstub, deviceId, kid, transfers_s, &preemp_s);
	
		// make HtD transfers 
		(kstub->startMallocs)((void *)(kstub));
		(kstub->startTransfers)((void *)(kstub));	
		cudaDeviceSynchronize();
		
		// Exec 
		int idSMs[2];
		
		idSMs[0]=0;idSMs[1]=kstub->kconf.numSMs-1;
		kstub->idSMs = idSMs;
		(kstub->launchCKEkernel)(kstub);
		cudaDeviceSynchronize();
		
		flag_create_kstub_with_params = 1;
		kid = CCONV;
	}
	
	if (strcmp(argv[2], "GCEDD") == 0){
		kid = GCEDD;
	}
	
	if (strcmp(argv[2], "SCEDD") == 0){
		
		// First GCEDD must be executed 
		kid = GCEDD;
		/** Create stub ***/
		create_stubinfo(&kstub, deviceId, kid, transfers_s, &preemp_s);
	
		// make HtD transfers 
		(kstub->startMallocs)((void *)(kstub));
		(kstub->startTransfers)((void *)(kstub));	
		cudaDeviceSynchronize();
		
		// Exec 
		int idSMs[2];
		
		idSMs[0]=0;idSMs[1]=kstub->kconf.numSMs-1;
		kstub->idSMs = idSMs;
		(kstub->launchCKEkernel)(kstub);
		cudaDeviceSynchronize();
		
		flag_create_kstub_with_params = 1;
		kid = SCEDD;

	}
	
	if (strcmp(argv[2], "NCEDD") == 0){
		
		// First GCEDD must be executed 
		kid = GCEDD;
		/** Create stub ***/
		create_stubinfo(&kstub, deviceId, kid, transfers_s, &preemp_s);
	
		// make HtD transfers 
		(kstub->startMallocs)((void *)(kstub));
		(kstub->startTransfers)((void *)(kstub));	
		cudaDeviceSynchronize();
		
		// Exec 
		int idSMs[2];
		
		idSMs[0]=0;idSMs[1]=kstub->kconf.numSMs-1;
		kstub->idSMs = idSMs;
		(kstub->launchCKEkernel)(kstub);
		cudaDeviceSynchronize();
		
		// Then, next previous kernel
		t_Kernel kidt1 = SCEDD;
		t_kernel_stub * kstubt1;
		
		create_stubinfo_with_params(&kstubt1, deviceId, kidt1, transfers_s, &preemp_s, (void *)kstub->params);
		
		// make HtD transfers 
		(kstubt1->startMallocs)((void *)(kstubt1));
		(kstubt1->startTransfers)((void *)(kstubt1));	
		cudaDeviceSynchronize();
		
		// Exec
		idSMs[0]=0;idSMs[1]=kstubt1->kconf.numSMs-1;
		kstubt1->idSMs = idSMs;
		(kstubt1->launchCKEkernel)(kstubt1);
		cudaDeviceSynchronize();
		
		flag_create_kstub_with_params = 1;
		kid = NCEDD;
	}
	
	if (strcmp(argv[2], "HCEDD") == 0){
		
		// First GCEDD must be executed 
		kid = GCEDD;
		/** Create stub ***/
		create_stubinfo(&kstub, deviceId, kid, transfers_s, &preemp_s);
	
		// make HtD transfers 
		(kstub->startMallocs)((void *)(kstub));
		(kstub->startTransfers)((void *)(kstub));	
		cudaDeviceSynchronize();
		
		// Exec 
		int idSMs[2];
		
		idSMs[0]=0;idSMs[1]=kstub->kconf.numSMs-1;
		kstub->idSMs = idSMs;
		(kstub->launchCKEkernel)(kstub);
		cudaDeviceSynchronize();
		
		// Then, next previous kernel
		t_Kernel kidt1 = SCEDD;
		t_kernel_stub * kstubt1;
		
		create_stubinfo_with_params(&kstubt1, deviceId, kidt1, transfers_s, &preemp_s, (void *)kstub->params);
		
		// make HtD transfers 
		(kstubt1->startMallocs)((void *)(kstubt1));
		(kstubt1->startTransfers)((void *)(kstubt1));	
		cudaDeviceSynchronize();
		
		// Exec
		idSMs[0]=0;idSMs[1]=kstubt1->kconf.numSMs-1;
		kstubt1->idSMs = idSMs;
		(kstubt1->launchCKEkernel)(kstubt1);
		cudaDeviceSynchronize();
		
		// Finally, a third kernel
		
		t_Kernel kidt2 = NCEDD;
		t_kernel_stub * kstubt2;
		
		create_stubinfo_with_params(&kstubt2, deviceId, kidt2, transfers_s, &preemp_s, (void *)kstub->params);
		
		// make HtD transfers 
		(kstubt2->startMallocs)((void *)(kstubt2));
		(kstubt2->startTransfers)((void *)(kstubt2));	
		cudaDeviceSynchronize();
		
		// Exec
		idSMs[0]=0;idSMs[1]=kstubt2->kconf.numSMs-1;
		kstubt2->idSMs = idSMs;
		(kstubt2->launchCKEkernel)(kstubt2);
		cudaDeviceSynchronize();
		
		flag_create_kstub_with_params = 1;
		kid = HCEDD;
	}
		
	
	if (strcmp(argv[2], "Reduction") == 0){
		kid = Reduction;
	}
	
	if (strcmp(argv[2], "HST256") == 0){
		kid = HST256;
	}

	if (kid < 0){
		printf("Error: Wrong kernel name\n");
		return -1;
	}
	
	/** Create stub ***/
	t_kernel_stub *kstub1;
	
	if (flag_create_kstub_with_params == 0)
		create_stubinfo(&kstub1, deviceId, kid, transfers_s, &preemp_s);
	else
		create_stubinfo_with_params(&kstub1, deviceId, kid, transfers_s, &preemp_s, (void *)kstub->params);
	
	// make HtD transfers 
	(kstub1->startMallocs)((void *)(kstub1));
	(kstub1->startTransfers)((void *)(kstub1));
	
	cudaDeviceSynchronize();
	
	double over;
	preemp_overhead(kstub1, atoi(argv[4]), &over);
	
	return 0;

	// Solo execution
	
	int BpSM = atoi(argv[3]);
	int iterations = atoi(argv[4]);
	struct timespec now;
	int idSMs[2];
	double time1, time2;
		
	idSMs[0]=0;idSMs[1]=kstub1->kconf.numSMs-1;
	kstub1->idSMs = idSMs;	
	
	kstub1->kconf.max_persistent_blocks = BpSM; // Limit the max number of blocks per SM
	
	for (int i=0; i<iterations; i++) {
	
		cudaProfilerStart();
	
		clock_gettime(CLOCK_REALTIME, &now);
		time1 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
	
		(kstub1->launchCKEkernel)((void *)kstub1);
		cudaDeviceSynchronize();

		clock_gettime(CLOCK_REALTIME, &now);
		time2 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
		
		cudaProfilerStop();
	
		int exec_tasks=0;
		cudaMemcpyAsync(kstub1->d_executed_tasks, &exec_tasks, sizeof(int), cudaMemcpyHostToDevice, *kstub1->preemp_s); // Reset task counter
		
		cudaDeviceSynchronize();
	}
			
	printf("BSP=%d Time=%f Tpms=%f\n", BpSM, time2-time1, (double)kstub1->total_tasks/(1000.0*(time2-time1)));
	
	cudaProfilerStart();
	clock_gettime(CLOCK_REALTIME, &now);
	time1 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
	
	(kstub1->launchORIkernel)((void *)kstub1);
	cudaDeviceSynchronize();
	
	clock_gettime(CLOCK_REALTIME, &now);
	time2 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
	cudaProfilerStop();
	printf("Original Time=%f\n", time2-time1);

	
	return 0;
}