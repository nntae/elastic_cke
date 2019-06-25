typedef struct {
	int numRows;
	int nItems;
} t_SPMV_params;

int launch_preemp_SPMVcsr(void *arg);
int launch_orig_SPMVcsr(void *arg);
int SPMVcsr_start_kernel(void *arg);
int SPMVcsr_end_kernel(void *arg);
int SPMVcsr_start_transfers(void *arg);
int SPMVcsr_start_mallocs(void *arg);