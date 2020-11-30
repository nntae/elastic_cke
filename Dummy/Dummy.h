#ifndef _DUMMY_H_
#define _DUMMY_H_

int launch_preemp_dummy(void *arg);
int launch_orig_dummy(void *arg);
int dummy_start_kernel(void *arg);
int dummy_end_kernel(void *arg);
int dummy_start_transfers(void *arg);
int dummy_start_mallocs(void *arg);

int dummy_test(t_Kernel *kernel_id, float *tHtD, float *tK, float *tDtH);

#endif