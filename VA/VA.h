#include <helper_functions.h>   // helper functions for string parsing
#include <helper_cuda.h>  
#include <semaphore.h> 

int VA_start_kernel_dummy(void *arg);				
int VA_end_kernel_dummy(void *arg);

int launch_preemp_VA(void *arg);
int launch_orig_VA(void *arg);

int VA_start_mallocs(void *arg);
int VA_start_transfers(void *arg);
