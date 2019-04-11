#include <helper_functions.h>   // helper functions for string parsing
#include <helper_cuda.h>  
#include <semaphore.h> 			

typedef struct {
	int nRows;
	int nCols;
	int param_pyramid_height;
} t_PF_params;
	
int launch_preemp_PF(void *kstub);
int launch_orig_PF(void *kstub);

int PF_start_kernel(void *arg);
int PF_end_kernel(void *arg);

int PF_start_mallocs(void *arg);
int PF_start_transfers(void *arg);
