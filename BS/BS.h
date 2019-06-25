#include <helper_functions.h>   // helper functions for string parsing
#include <helper_cuda.h>  
#include <semaphore.h> 

typedef struct {
	float
    //Results calculated by CPU for reference
    *h_CallResultCPU,
    *h_PutResultCPU,
    //CPU copy of GPU results
    *h_CallResultGPU,
    *h_PutResultGPU,
    //CPU instance of input data
    *h_StockPrice,
    *h_OptionStrike,
    *h_OptionYears;

    //'d_' prefix - GPU (device) memory space
    float
    //Results calculated by GPU
    *d_CallResult=NULL,
    *d_PutResult=NULL,
    //GPU instance of input data
    *d_StockPrice=NULL,
    *d_OptionStrike=NULL,
    *d_OptionYears=NULL;
} t_BS_params;

/*__global__ void original_BlackScholesGPU(
    float *d_CallResult,
    float *d_PutResult,
    float *d_StockPrice,
    float *d_OptionStrike,
    float *d_OptionYears,
    float Riskfree,
    float Volatility,
    int optN,
	int iter_per_block
);

int BS_preemp_start_kernel(int TB_Number, int Blk_Size, int num_subtasks, int iter_per_subtask, int numSMS,
					 t_kernel_stub *k_stub);

int preemp_BS_end_kernel(t_BS_params *params);

int set_BSparams(t_BS_params *BS_params, int numSMs, int BS_Blq_resi_perSM, int TB_BS, int Blk_size, int BScoarsening,
				int *idSMs_BS, int max_blocks_per_SM,
				cudaStream_t stream, t_kernel_stub *k_stubBS, State *VA_gm_state);

__global__ void preemp_SMT_BlackScholesGPU(
    float *d_CallResult_1,
    float *d_PutResult_1,
    float *d_StockPrice_1, 
    float *d_OptionStrike_1,
    float *d_OptionYears_1,
    float Riskfree,
    float Volatility,
    int optN,
	
	int SIMD_min,
	int SIMD_max,
	int num_subtask,
	int iter_per_subtask,
	int *cont_subtask,
	State *status
);

__global__ void preemp_SMK_BlackScholesGPU(
    float *d_CallResult_1,
    float *d_PutResult_1,
    float *d_StockPrice_1,
    float *d_OptionStrike_1,
    float *d_OptionYears_1,
    float Riskfree,
    float Volatility,
    int optN,
 
	int max_blocks_per_SM,
	int num_subtask,
	int iter_per_subtask,
	int *cont_SM,
	int *cont_subtask,
	State *status
);
*/

int BS_start_kernel_dummy(void *arg);
//int BS_start_kernel(int TB, int blkDim, int iter_per_block);
//int BS_execute_kernel(int TB, int blkDim, int iter_per_blockBS, float *time_VA);
int BS_end_kernel_dummy(void *arg);
//int BS_end_kernel();

int launch_preemp_BS(void *kstub);
int launch_orig_BS(void *kstub);

int BS_start_mallocs(void *arg);
int BS_start_transfers(void *arg);

					
