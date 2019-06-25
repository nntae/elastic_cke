#include <helper_functions.h>   // helper functions for string parsing
#include <helper_cuda.h>  
#include <semaphore.h> 			

typedef struct {
	unsigned char * h_in_out[2];
	unsigned char * data_CEDD, *out_CEDD, *theta_CEDD;
	
	int nRows;
	int nCols;
	int gridDimX;
	int gridDimY;
} t_CEDD_params;

//Gaussian
int launch_preemp_GCEDD(void *kstub);
int launch_orig_GCEDD(void *kstub);
int GCEDD_start_kernel(void *arg);
int GCEDD_end_kernel(void *arg);
int GCEDD_start_mallocs(void *arg);
int GCEDD_start_transfers(void *arg);

//Sobel
int launch_preemp_SCEDD(void *kstub);
int launch_orig_SCEDD(void *kstub);

//Non max supp
int launch_preemp_NCEDD(void *kstub);
int launch_orig_NCEDD(void *kstub);

//Hyst
int launch_preemp_HCEDD(void *kstub);
int launch_orig_HCEDD(void *kstub);

//CPU
void run_cpu_threads(unsigned char *buffer0, unsigned char *buffer1, unsigned char *theta, int rows, int cols, int num_threads, int t_index);