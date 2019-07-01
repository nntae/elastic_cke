#include <helper_functions.h>   // helper functions for string parsing
#include <helper_cuda.h>  
#include <semaphore.h> 

#define KERNEL_RADIUS 8
#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)

#define   ROWS_BLOCKDIM_X 16
#define   ROWS_BLOCKDIM_Y 4
#define ROWS_RESULT_STEPS 8
#define   ROWS_HALO_STEPS 1	
#define   ROWS_GRIDDIM_X  24 * 2
#define   ROWS_GRIDDIM_Y  768 * 2		

#define   COLUMNS_BLOCKDIM_X 16
#define   COLUMNS_BLOCKDIM_Y 8
#define COLUMNS_RESULT_STEPS 8
#define   COLUMNS_HALO_STEPS 1
#define   COLUMNS_GRIDDIM_X  192 * 2
#define   COLUMNS_GRIDDIM_Y  48 * 2

typedef struct {
	float *h_Kernel, *h_Input, *h_Buffer, *h_OutputCPU, *h_OutputGPU;
	float *d_Input, *d_Output, *d_Buffer;
	
	int conv_rows;
	int conv_cols;
	int gridDimY[2];
} t_CONV_params;

//Rows
int launch_preemp_RCONV(void *kstub);
int launch_orig_RCONV(void *kstub);
int RCONV_start_kernel(void *arg);
int RCONV_end_kernel(void *arg);

int RCONV_start_mallocs(void *arg);
int RCONV_start_transfers(void *arg);

//Cols
int launch_preemp_CCONV(void *kstub);
int launch_orig_CCONV(void *kstub);

int CCONV_start_kernel(void *arg);
int CCONV_end_kernel(void *arg);

int CCONV_start_mallocs(void *arg);
int CCONV_start_transfers(void *arg);