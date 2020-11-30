#include "support/common.h"

 typedef struct {
	int gridDimX;
	int flow_vector_count;
	int max_iter;
	int error_threshold; 
    float convergence_threshold;
	int n_tasks;
	float alpha;
	int best_model;
	float *model_params_local;
	flowvector *h_flow_vector_array;
	flowvector *flow_vectors;
	int *h_random_numbers;
    int *random_numbers;
	int *h_model_candidate;
	int *model_candidate; 
	int best_outliers;
	int *h_outliers_candidate;
	int *outliers_candidate; 
	int *launch_gpu; 
    int l_mem_size;
	int h_g_out_id;
    int *g_out_id;
} t_RSC_params;

int launch_preemp_RSC_model(void *kstub);
int launch_orig_RSC_model(void *kstub);
int launch_preemp_RSC_evaluate(void *kstub);
int launch_orig_RSC_evaluate(void *kstub);

int RSC_model_start_kernel(void *arg);
int RSC_model_end_kernel(void *arg);
int RSC_evaluate_start_kernel(void *arg);
int RSC_evaluate_end_kernel(void *arg);