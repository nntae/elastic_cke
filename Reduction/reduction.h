typedef struct {
	float *h_idata, *h_odata;
	float *d_idata, *d_odata;
} t_reduction_params;

int launch_preemp_reduce(void *arg);
int launch_orig_reduce(void *arg);
int reduce_start_kernel(void *arg);
int reduce_end_kernel(void *arg);
int reduce_start_transfers(void *arg);
int reduce_start_mallocs(void *arg);

int reduce_test(t_Kernel *kernel_id, float *tHtD, float *tK, float *tDtH);