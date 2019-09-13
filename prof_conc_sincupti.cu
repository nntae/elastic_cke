#include <unistd.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <helper_functions.h>   // helper functions for string parsing
#include <helper_cuda.h>   
#include "elastic_kernel.h"

int main(int argc, char **argv)
{

	cudaError_t err;
	int deviceId = atoi(argv[1]);
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
	
	// Create kerbel stub
	t_kernel_stub *kstub;
	
	create_stubinfo(&kstub, deviceId, MM, transfers_s, &preemp_s);
	
	// Make transfoer
	(kstub->startMallocs)((void *)(kstub));
	(kstub->startTransfers)((void *)(kstub));
	cudaDeviceSynchronize();
	
	// Execute 
	
	int idSMs[2];
	idSMs[0]=0;idSMs[1]=kstub->kconf.numSMs-1;
	kstub->idSMs = idSMs;	
	//(kstub->launchCKEkernel)(kstub);
	
	struct timespec now;
	double time1, time2;
	
	clock_gettime(CLOCK_REALTIME, &now);
 	time1 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
	
	prof_MM(kstub);
	cudaDeviceSynchronize();
	
	clock_gettime(CLOCK_REALTIME, &now);
 	time2 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
	printf("Concurrent excution time=%f sec.\n", time2-time1);


	return 0;
}
	
	
	
	
	
	