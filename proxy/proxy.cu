// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include "../elastic_kernel.h"


//* One block of one thread executes the code **/

__global__ void generic_proxy(int num_conc_kernels, State *kernel_evict_zc, State *kernel_evict, int *cont_tasks_zc, int *gm_cont_tasks)
{
	clock_t start;

	while (1){
		
		//start = clock();
		//while ((clock() - start) < 10000)
		//	;
		
		for (int i=0;i<num_conc_kernels; i++) {	

			
			// Check if schedluer commands kernel eviction
			//for (int j=0; j < MAX_STREAMS_PER_KERNEL; j++)
			//	if (kernel_evict_zc[i*MAX_STREAMS_PER_KERNEL+j] == TOEVICT)
			//			kernel_evict[i*MAX_STREAMS_PER_KERNEL+j] = TOEVICT;
				
			// Update kernel task counter in scheduler memory space
			
			*(cont_tasks_zc+i) = *(gm_cont_tasks+i);
			
		}
		
		if (kernel_evict_zc[0] == PROXY_EVICT)
				return;
	}
}
	
__global__ void proxy(State *kernel_eviction, int *proxy_eviction, int *proxy_cont_tasks, int *cont_tasks)
{	
	__shared__ int end;
	end = 0;
	
	
	while (1) {

		if (threadIdx.x == 0) {
		
			if (*proxy_eviction == 1){
				*kernel_eviction = TOEVICT;
			}
			
			if (*proxy_eviction == 2){
				*proxy_cont_tasks = *cont_tasks;
				end =1;
			}
		}
		
		__syncthreads();
		
		if (end == 1) {
			*kernel_eviction = RUNNING;
			return;
		}
	}

}

int launch_proxy(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	
	proxy<<<1, 1, 0, *(kstub->proxy_s)>>>(&(kstub->gm_state[kstub->stream_index]), kstub->d_proxy_eviction, kstub->d_exec_tasks_proxy, kstub->d_executed_tasks);
	
	return 0;
}

int launch_generic_proxy(void *arg)
{ 
	t_sched *sched = (t_sched *)arg;
	
	generic_proxy<<<1, 1, 0, *(sched->proxy_s)>>>(sched->num_conc_kernels, sched->kernel_evict_zc, sched->kernel_evict, sched->cont_tasks_zc, sched->gm_cont_tasks);
	
	return 0;
}

	