#include <unistd.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <helper_functions.h>   // helper functions for string parsing
#include <helper_cuda.h>   
#include "elastic_kernel.h"

int main(int argc, char **argv)
{
	if (argc <4) {
		printf("Error: program must run as follows: sheduler device_id kernel_name num_BpSM iterations\n");
		return -1;
	}
	
	int deviceId = atoi(argv[1]);
	
	cudaSetDevice(deviceId);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);

	t_Kernel kid=(t_Kernel)-1;
	
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
	
	if (strcmp(argv[2], "GCEDD") == 0){
		kid = GCEDD;
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
	
	/** Create commom streams for all kernels: two for asynchronous transfers, one for preemption commands*/
	cudaStream_t *transfers_s;
	transfers_s = (cudaStream_t *)calloc(2, sizeof(cudaStream_t));
	
	for (int i=0;i<2;i++){
		cudaStreamCreate(&transfers_s[i]);
	} 
	
	cudaStream_t preemp_s;
	checkCudaErrors(cudaStreamCreateWithFlags(&preemp_s, cudaStreamNonBlocking)); 
	
	/** Create stub ***/
	t_kernel_stub *kstub;
	create_stubinfo(&kstub, deviceId, kid, transfers_s, &preemp_s);
	
	// make HtD transfers 
	(kstub->startMallocs)((void *)(kstub));
	(kstub->startTransfers)((void *)(kstub));
	
	cudaDeviceSynchronize();
	
	// Solo execution
	
	int BpSM = atoi(argv[3]);
	int iterations = atoi(argv[4]);
	struct timespec now;
	int idSMs[2];
	double time1, time2;
		
	idSMs[0]=0;idSMs[1]=kstub->kconf.numSMs-1;
	kstub->idSMs = idSMs;	
	
	kstub->kconf.max_persistent_blocks = BpSM; // Limit the max number of blocks per SM
	
	for (int i=0; i<iterations; i++) {
	
		clock_gettime(CLOCK_REALTIME, &now);
		time1 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
	
		(kstub->launchCKEkernel)(kstub);
		cudaDeviceSynchronize();

		clock_gettime(CLOCK_REALTIME, &now);
		time2 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
	
		int exec_tasks=0;
		cudaMemcpyAsync(kstub->d_executed_tasks, &exec_tasks, sizeof(int), cudaMemcpyHostToDevice, *kstub->preemp_s); // Reset task counter
		
		cudaDeviceSynchronize();
	}
			
	printf("BSP=%d Time=%f Tpms=%f\n", BpSM, time2-time1, (double)kstub->total_tasks/(1000.0*(time2-time1)));
	
	
	return 0;
}