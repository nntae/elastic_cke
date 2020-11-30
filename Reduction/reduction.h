typedef struct {
	float *h_idata, *h_odata;
	float *d_idata, *d_odata;
	unsigned int size;
	int smem_size;
	int gridDimX;
	int *zc_slc;
} t_reduction_params;

int launch_preemp_reduce(void *arg);
int launch_orig_reduce(void *arg);
int launch_slc_reduce(void *arg);
int reduce_start_kernel(void *arg);
int reduce_end_kernel(void *arg);
int reduce_start_transfers(void *arg);
int reduce_start_mallocs(void *arg);