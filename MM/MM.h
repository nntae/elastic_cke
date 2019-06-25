#include <helper_functions.h>   // helper functions for string parsing
#include <helper_cuda.h>  
#include <semaphore.h> 

typedef struct {
	float *h_MMA, *h_MMB, *h_MMC;
	float *d_MMA, *d_MMB, *d_MMC;
	int size_A, size_B, size_C;
	int dimA_x, dimB_x;

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
