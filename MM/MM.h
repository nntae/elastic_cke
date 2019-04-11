#include <helper_functions.h>   // helper functions for string parsing
#include <helper_cuda.h>  
#include <semaphore.h> 

typedef struct {
	dim3 Asize;
	dim3 Bsize;
	int gridDimX;
} t_MM_params;
			
			
int launch_preemp_MM(void *kstub);
int launch_orig_MM(void *kstub);

int MM_start_kernel(void *arg);
int MM_end_kernel(void *arg);

int MM_start_mallocs(void *arg);
int MM_start_transfers(void *arg);
