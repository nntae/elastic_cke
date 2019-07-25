#include <unistd.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <helper_functions.h>   // helper functions for string parsing
#include <helper_cuda.h>   
#include "elastic_kernel.h"
#include "cupti_profiler.h"

int main(int argc, char **argv)
{

	cudaError_t err;
	int deviceId = atoi(argv[1]);
	// Select device
	/*
	cudaSetDevice(deviceId);
	cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);	
	printf("Device=%s\n", deviceProp.name);
	*/
	std::vector<std::string> curr_metric = init_cupti_profiler( deviceId );
	
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
	
	create_stubinfo(&kstub, deviceId, BS, transfers_s, &preemp_s);
	
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
	FILE *fp = open_metric_file( "metric_values.log" );
	fprintf(fp, "MetricName, EventName, Sum, TotalInstances, NumInstances, Normalized, Values, ...\n");

	for ( int i = 0; i < curr_metric.size(); i++ )
	{
		CUpti_EventGroupSets *passData = start_cupti_profiler(	curr_metric[i].c_str() );
		int num_passes = passData->numSets ;
		advance_cupti_profiler( passData, 0 );
		if ( num_passes > 1 )
		{
			stop_cupti_profiler( false );
			printf("Ignoring metric %s because it needs %d passes\n", curr_metric[i].c_str(), num_passes);
		}
		else
		{

			int exec_tasks=0;
			cudaMemcpyAsync(kstub->d_executed_tasks, &exec_tasks, sizeof(int), cudaMemcpyHostToDevice, *kstub->preemp_s); // Reset task counter
			printf("Profiling %s ...\n", curr_metric[i].c_str());

			clock_gettime(CLOCK_REALTIME, &now);
			time1 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
				
			prof_BS(kstub);

			cudaDeviceSynchronize();
			stop_cupti_profiler( true );

			clock_gettime(CLOCK_REALTIME, &now);
			time2 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
			
			printf("Concurrent execution time=%f sec.\n", time2-time1);
		}
	}
	
	close_metric_file();

	return 0;
}
	
	
	
	
	
	