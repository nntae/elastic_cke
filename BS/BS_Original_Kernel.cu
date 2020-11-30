 #include <semaphore.h>
 #include <helper_functions.h>   // helper functions for string parsing
 #include <helper_cuda.h>   
 #include "../elastic_kernel.h"
 #include "BS.h"

 extern t_tqueue *tqueues;
 
/*
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
 
  __device__ uint get_smid(void) {

     uint ret;

     asm("mov.u32 %0, %smid;" : "=r"(ret) );

     return ret;

}

///////////////////////////////////////////////////////////////////////////////
// Polynomial approximation of cumulative normal distribution function
///////////////////////////////////////////////////////////////////////////////
__device__ inline float cndGPU(float d)
{
    const float       A1 = 0.31938153f;
    const float       A2 = -0.356563782f;
    const float       A3 = 1.781477937f;
    const float       A4 = -1.821255978f;
    const float       A5 = 1.330274429f;
    const float RSQRT2PI = 0.39894228040143267793994605993438f;
 
    float
    K = 1.0f / (1.0f + 0.2316419f * fabsf(d));

    float
    cnd = RSQRT2PI * __expf(- 0.5f * d * d) *
          (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if (d > 0)
        cnd = 1.0f - cnd;

    return cnd;
}


///////////////////////////////////////////////////////////////////////////////
// Black-Scholes formula for both call and put
///////////////////////////////////////////////////////////////////////////////
__device__ inline void BlackScholesBodyGPU(
    float &CallResult_1,
    float &PutResult_1,
    float S, //Stock price
    float X, //Option strike
    float T, //Option years
    float R, //Riskless rate
    float V  //Volatility rate
)
{
    float sqrtT, expRT;
    float d1, d2, CNDD1, CNDD2;

    sqrtT = sqrtf(T);
    d1 = (__logf(S / X) + (R + 0.5f * V * V) * T) / (V * sqrtT);
    d2 = d1 - V * sqrtT;

    CNDD1 = cndGPU(d1);
    CNDD2 = cndGPU(d2); 

    //Calculate Call and Put simultaneously
    expRT = __expf(- R * T);
	if (threadIdx.x == 0) {
    CallResult_1 = S * CNDD1 - X * expRT * CNDD2;
    PutResult_1  = X * expRT * (1.0f - CNDD2) - S * (1.0f - CNDD1);
	}
}

////////////////////////////////////////////////////////////////////////////////
//Process an array of optN options on GPU
////////////////////////////////////////////////////////////////////////////////
__global__ void original_BlackScholesGPU(
    float *d_CallResult,
    float *d_PutResult,
    float *d_StockPrice,
    float *d_OptionStrike,
    float *d_OptionYears,
    float Riskfree,
    float Volatility,
    int optN,
	int iter_per_block)
{
    ////Thread index
    //const int      tid = blockDim.x * blockIdx.x + threadIdx.x;
    ////Total number of threads in execution grid
    //const int THREAD_N = blockDim.x * gridDim.x;

    //const int opt = blockDim.x * blockIdx.x + threadIdx.x;

	
	for (int k=0; k<iter_per_block; k++) {
		
		const int opt =  blockIdx.x * blockDim.x * iter_per_block + k *  blockDim.x + threadIdx.x;
			
    //No matter how small is execution grid or how large OptN is,
    //exactly OptN indices will be processed with perfect memory coalescing
    //for (int opt = tid; opt < optN; opt += THREAD_N)
		if (opt < optN){
		//for (int i=0;i<10;i++)
			BlackScholesBodyGPU(
					d_CallResult[opt],
					d_PutResult[opt],
					d_StockPrice[opt], 
					d_OptionStrike[opt],
					d_OptionYears[opt],
					Riskfree,
					Volatility
			);
		}
	}
	
}

__global__ void slicing_BlackScholesGPU(
    float *d_CallResult,
    float *d_PutResult,
    float *d_StockPrice,
    float *d_OptionStrike,
    float *d_OptionYears,
    float Riskfree,
    float Volatility,
    int optN,
	int iter_per_block,
	int init_blkIdx,
	int *zc_slc)
{
    ////Thread index
    //const int      tid = blockDim.x * blockIdx.x + threadIdx.x;
    ////Total number of threads in execution grid
    //const int THREAD_N = blockDim.x * gridDim.x;

	//const int opt = blockDim.x * blockIdx.x + threadIdx.x;
	
	if (threadIdx.x == 0) atomicAdd(zc_slc, 1);
	
	for (int k=0; k<iter_per_block; k++) {
		
		const int opt =  (blockIdx.x + init_blkIdx) * blockDim.x * iter_per_block + k *  blockDim.x + threadIdx.x;
			
    //No matter how small is execution grid or how large OptN is,
    //exactly OptN indices will be processed with perfect memory coalescing
    //for (int opt = tid; opt < optN; opt += THREAD_N)
		if (opt < optN){
		//for (int i=0;i<10;i++)
			BlackScholesBodyGPU(
					d_CallResult[opt],
					d_PutResult[opt],
					d_StockPrice[opt], 
					d_OptionStrike[opt],
					d_OptionYears[opt],
					Riskfree,
					Volatility
			);
		}
	}
	
}

////////////////////////////////////////////////////////////////////////////////
//SMK version: Only works with resident kernels
////////////////////////////////////////////////////////////////////////////////
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
	
)
{
	__shared__ int s_bid, s_index;
	
	unsigned int SM_id = get_smid();
	
	if (threadIdx.x == 0)  
		s_index = atomicAdd(&cont_SM[SM_id],1);
	
	__syncthreads();

	if (s_index > max_blocks_per_SM)
		return;
		
	while (1){
		
		/********** Task Id calculation *************/
		if (threadIdx.x == 0) {
			if (*status == TOEVICT)
				s_bid = -1;
			else
				s_bid = atomicAdd(cont_subtask, 1);
		}
		
	//	if ( thIdxwarp == 0) {
	//		if (*status == TOEVICT)
	//			s_bid[warpid] = -1;
	//		else
	//			s_bid[warpid] = atomicAdd(cont_subtask + warpid, 1); //subtask_id
	//	}
		
		__syncthreads();
		
		//if (s_bid[warpid] >= num_subtask || s_bid[warpid] == -1)
		if (s_bid >= num_subtask || s_bid == -1) /* If all subtasks have been executed */
			return;
		
		for (int j=0; j<iter_per_subtask; j++) {
			
			//const int opt = s_bid[warpid] * blockDim.x * iter_per_subtask +  j * blockDim.x + threadIdx.x;
			const int opt = s_bid * blockDim.x * iter_per_subtask +  j * blockDim.x + threadIdx.x;
            //No matter how small is execution grid or how large OptN is,
            //exactly OptN indices will be processed with perfect memory coalescing
            //for (int opt = tid; opt < optN; opt += THREAD_N)
			if (opt < optN){
				BlackScholesBodyGPU(
					d_CallResult_1[opt],
					d_PutResult_1[opt],
					d_StockPrice_1[opt], 
					d_OptionStrike_1[opt],
					d_OptionYears_1[opt],
					Riskfree,
					Volatility
				);
			}
		}
		
		/*** Status cheking ****/
		//if (*status == TOEVICT)
		//	return;
	}
}

__device__ void delay()
{
	clock_t t_ini;
	
	t_ini= clock();
	
	while (clock()-t_ini <100000);
	
}

__global__ void profiling_BlackScholesGPU(
    float *d_CallResult_1,
    float *d_PutResult_1,
    float *d_StockPrice_1,
    float *d_OptionStrike_1,
    float *d_OptionYears_1,
    float Riskfree,
    float Volatility,
    int optN,
	int num_subtask,
	int iter_per_subtask,
	int *cont_SM,
	int *cont_subtask,
	State *status
)

{
	__shared__ int s_bid, CTA_cont;
	
	unsigned int SM_id = get_smid();
	
	if (SM_id >= 8){ /* Only blocks executing in first 8 SM  are used for profiling */ 
		//delay();
		return;
	}
	
	if (threadIdx.x == 0) {
		CTA_cont = atomicAdd(&cont_SM[SM_id], 1);
	//	if (SM_id == 7 && CTA_cont == 8)
	//		printf("Aqui\n");
	}
	
	__syncthreads();
	
	if (CTA_cont > SM_id) {/* Only one block makes computation in SM0, two blocks in SM1 and so on */
		delay();
		return;
	}
	
	//if (threadIdx.x == 0)
	//	printf ("SM=%d CTA = %d\n", SM_id, CTA_cont);

	int cont_task = 0;
	
	while (1){
		
		/********** Task Id calculation *************/
		if (threadIdx.x == 0) {
			if (*status == TOEVICT)
				s_bid = -1;
			else
				s_bid = atomicAdd(cont_subtask, 1);
		}
		
		__syncthreads();
		
		
		
		if (s_bid >= num_subtask || s_bid == -1){ /* If all subtasks have been executed */
			if (threadIdx.x == 0)
				printf ("SM=%d CTA=%d Executed_tasks= %d \n", SM_id, CTA_cont, cont_task);	
			return;
		}
		
		if (threadIdx.x == 0) // Acumula numeor de tareas ejecutadas
			 cont_task++;
		
		for (int j=0; j<iter_per_subtask; j++) {
			
			const int opt = s_bid * blockDim.x * iter_per_subtask +  j * blockDim.x + threadIdx.x;
			if (opt < optN){
				BlackScholesBodyGPU(
					d_CallResult_1[opt],
					d_PutResult_1[opt],
					d_StockPrice_1[opt], 
					d_OptionStrike_1[opt],
					d_OptionYears_1[opt],
					Riskfree,
					Volatility
				);
			}
		}
	}
}


////////////////////////////////////////////////////////////////////////////////
//Process an array of optN options on GPU
////////////////////////////////////////////////////////////////////////////////
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
	
)
{
	__shared__ int s_bid;
	
	unsigned int SM_id = get_smid();
	
	if (SM_id <SIMD_min || SM_id > SIMD_max) /* Only blocks executing within SIMD_min and SIMD_max can progress */ 
			return;
	
	//int warpid = threadIdx.x >> 5;
	
	//int thIdxwarp = threadIdx.x & 0x1F;
	
	//if (threadIdx.x == 0) printf("I am executing BS\n");
	
	//if (threadIdx.x==0) // Ojo, esto es una prueba. Habría que tener en cuenta iteraciones entre distintos bloques
	//	*status = RUNNING;
		
	while (1){
		
		/********** Task Id calculation *************/
		if (threadIdx.x == 0) {
			if (*status == TOEVICT)
				s_bid = -1;
			else
				s_bid = atomicAdd(cont_subtask, 1);
		}
		
	//	if ( thIdxwarp == 0) {
	//		if (*status == TOEVICT)
	//			s_bid[warpid] = -1;
	//		else
	//			s_bid[warpid] = atomicAdd(cont_subtask + warpid, 1); //subtask_id
	//	}
		
		__syncthreads();
		
		//if (s_bid[warpid] >= num_subtask || s_bid[warpid] == -1)
		if (s_bid >= num_subtask || s_bid == -1){ /* If all subtasks have been executed */
		/*if (blockIdx.x == 0 && threadIdx.x == 0)
			printf("BS finished: execute dtask =%d\n", *cont_subtask);*/
			return;
		}
			
		
		for (int j=0; j<iter_per_subtask; j++) {
			
			//const int opt = s_bid[warpid] * blockDim.x * iter_per_subtask +  j * blockDim.x + threadIdx.x;
			const int opt = s_bid * blockDim.x * iter_per_subtask +  j * blockDim.x + threadIdx.x;
            //No matter how small is execution grid or how large OptN is,
            //exactly OptN indices will be processed with perfect memory coalescing
            //for (int opt = tid; opt < optN; opt += THREAD_N)
			if (opt < optN){
				BlackScholesBodyGPU(
					d_CallResult_1[opt],
					d_PutResult_1[opt],
					d_StockPrice_1[opt], 
					d_OptionStrike_1[opt],
					d_OptionYears_1[opt],
					Riskfree,
					Volatility
				);
			}
		}
		
		/*** Status cheking ****/
		//if (*status == TOEVICT)
		//	return;
	}
}

/*
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
 
/*
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <math.h>

///////////////////////////////////////////////////////////////////////////////
// Polynomial approximation of cumulative normal distribution function
///////////////////////////////////////////////////////////////////////////////
static double CND(double d)
{
    const double       A1 = 0.31938153;
    const double       A2 = -0.356563782;
    const double       A3 = 1.781477937;
    const double       A4 = -1.821255978;
    const double       A5 = 1.330274429;
    const double RSQRT2PI = 0.39894228040143267793994605993438;

    double
    K = 1.0 / (1.0 + 0.2316419 * fabs(d));

    double
    cnd = RSQRT2PI * exp(- 0.5 * d * d) *
          (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if (d > 0)
        cnd = 1.0 - cnd;

    return cnd;
}


///////////////////////////////////////////////////////////////////////////////
// Black-Scholes formula for both call and put
///////////////////////////////////////////////////////////////////////////////
static void BlackScholesBodyCPU(
    float &callResult,
    float &putResult,
    float Sf, //Stock price
    float Xf, //Option strike
    float Tf, //Option years
    float Rf, //Riskless rate
    float Vf  //Volatility rate
)
{
    double S = Sf, X = Xf, T = Tf, R = Rf, V = Vf;

    double sqrtT = sqrt(T);
    double    d1 = (log(S / X) + (R + 0.5 * V * V) * T) / (V * sqrtT);
    double    d2 = d1 - V * sqrtT;
    double CNDD1 = CND(d1);
    double CNDD2 = CND(d2);

    //Calculate Call and Put simultaneously
    double expRT = exp(- R * T);
    callResult   = (float)(S * CNDD1 - X * expRT * CNDD2);
    putResult    = (float)(X * expRT * (1.0 - CNDD2) - S * (1.0 - CNDD1));
}


////////////////////////////////////////////////////////////////////////////////
// Process an array of optN options
////////////////////////////////////////////////////////////////////////////////
void BlackScholesCPU(
    float *h_CallResult,
    float *h_PutResult,
    float *h_StockPrice,
    float *h_OptionStrike,
    float *h_OptionYears,
    float Riskfree,
    float Volatility,
    int optN
)
{
    for (int opt = 0; opt < optN; opt++)
        BlackScholesBodyCPU(
            h_CallResult[opt],
            h_PutResult[opt],
            h_StockPrice[opt],
            h_OptionStrike[opt],
            h_OptionYears[opt],
            Riskfree,
            Volatility
        );
}



#include <helper_functions.h>   // helper functions for string parsing
#include <helper_cuda.h>        // helper functions CUDA error checking and initialization

////////////////////////////////////////////////////////////////////////////////
// Process an array of optN options on CPU
////////////////////////////////////////////////////////////////////////////////
// extern "C" void BlackScholesCPU(
    // float *h_CallResult,
    // float *h_PutResult,
    // float *h_StockPrice,
    // float *h_OptionStrike,
    // float *h_OptionYears,
    // float Riskfree,
    // float Volatility,
    // int optN
// );



////////////////////////////////////////////////////////////////////////////////
// Helper function, returning uniformly distributed
// random float in [low, high] range
////////////////////////////////////////////////////////////////////////////////
float RandFloat(float low, float high)
{
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}

////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
// const int OPT_N = 4000000;
// const int  NUM_ITERATIONS = 512;


// const int          OPT_SZ = OPT_N * sizeof(float);
// const float      RISKFREE = 0.02f;
// const float    VOLATILITY = 0.30f;

#define DIV_UP(a, b) ( ((a) + (b) - 1) / (b) )

/// Global memory //////

//'h_' prefix - CPU (host) memory space
    /*float
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
    *d_OptionYears=NULL;*/

    double
    delta, ref, sum_delta, sum_ref, max_delta, L1norm;
	
	float	RISKFREE = 0.02f;
	float   VOLATILITY = 0.30f;
	long int 	OPT_SZ;
	int OPT_N;

	


////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////

int BS_start_kernel_dummy(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	t_BS_params * params = (t_BS_params *)kstub->params;
	
	OPT_N = kstub->kconf.gridsize.x * kstub->kconf.blocksize.x * kstub->kconf.coarsening;
	OPT_SZ = OPT_N * sizeof(float);

    // Start logs

    //printf("Initializing data...\n");
    //printf("...allocating CPU memory for options.\n");
    params->h_CallResultCPU = (float *)malloc(OPT_SZ);
    params->h_PutResultCPU  = (float *)malloc(OPT_SZ);
    params->h_CallResultGPU = (float *)malloc(OPT_SZ);
    params->h_PutResultGPU  = (float *)malloc(OPT_SZ);
    params->h_StockPrice    = (float *)malloc(OPT_SZ);
    params->h_OptionStrike  = (float *)malloc(OPT_SZ);
    params->h_OptionYears   = (float *)malloc(OPT_SZ);
		
    //printf("...allocating GPU memory for options.\n");
    checkCudaErrors(cudaMalloc((void **)&params->d_CallResult,   OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&params->d_PutResult,    OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&params->d_StockPrice,   OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&params->d_OptionStrike, OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&params->d_OptionYears,  OPT_SZ));
	
    // printf("...generating input data in CPU mem.\n");
    srand(5347);

    //Generate options set
    for (int i = 0; i < OPT_N; i++)
    {
        params->h_CallResultCPU[i] = 0.0f;
        params->h_PutResultCPU[i]  = -1.0f;
        params->h_StockPrice[i]    = RandFloat(5.0f, 30.0f);
        params->h_OptionStrike[i]  = RandFloat(1.0f, 100.0f);
        params->h_OptionYears[i]   = RandFloat(0.25f, 10.0f);
    }

    //printf("...copying input data to GPU mem.\n");
    //Copy options data to GPU memory for further processing
    checkCudaErrors(cudaMemcpy(params->d_StockPrice,  params->h_StockPrice,   OPT_SZ, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(params->d_OptionStrike, params->h_OptionStrike,  OPT_SZ, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(params->d_OptionYears,  params->h_OptionYears,   OPT_SZ, cudaMemcpyHostToDevice));
    //printf("Data init done.\n\n");
	

    checkCudaErrors(cudaMemset(params->d_CallResult, 0, OPT_SZ));
    checkCudaErrors(cudaMemset(params->d_PutResult, 0, OPT_SZ));
    
	return 0;
}

int BS_start_mallocs(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	t_BS_params * params = (t_BS_params *)kstub->params;

	// globalmemory position for launched ctas counter
	cudaMalloc((void **)&params->zc_slc, sizeof(int));
	
	OPT_N = kstub->kconf.gridsize.x * kstub->kconf.blocksize.x * kstub->kconf.coarsening;
	OPT_SZ = OPT_N * sizeof(float);

    // Start logs

    //printf("Initializing data...\n");
    //printf("...allocating CPU memory for options.\n");
#if defined(MEMCPY_SYNC) || defined(MEMCPY_ASYNC)
	checkCudaErrors(cudaMallocHost(&params->h_CallResultCPU, OPT_SZ));
	checkCudaErrors(cudaMallocHost(&params->h_PutResultCPU, OPT_SZ));
	checkCudaErrors(cudaMallocHost(&params->h_CallResultGPU, OPT_SZ));
	checkCudaErrors(cudaMallocHost(&params->h_PutResultGPU, OPT_SZ));
	checkCudaErrors(cudaMallocHost(&params->h_StockPrice, OPT_SZ));
	checkCudaErrors(cudaMallocHost(&params->h_OptionStrike, OPT_SZ));
	checkCudaErrors(cudaMallocHost(&params->h_OptionYears, OPT_SZ));
	
	//printf("...allocating GPU memory for options.\n");
    checkCudaErrors(cudaMalloc((void **)&params->d_CallResult,   OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&params->d_PutResult,    OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&params->d_StockPrice,   OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&params->d_OptionStrike, OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&params->d_OptionYears,  OPT_SZ));  
	
	checkCudaErrors(cudaMemset(params->d_CallResult, 0, OPT_SZ/*, kstub->transfer_s[0]*/));
    checkCudaErrors(cudaMemset(params->d_PutResult, 0, OPT_SZ/*, kstub->transfer_s[0])*/));
		
#else
	#ifdef MANAGED_MEM
	params->h_CallResultCPU = (float *)malloc(OPT_SZ);
    params->h_PutResultCPU  = (float *)malloc(OPT_SZ);
	
	cudaMallocManaged(&params->h_CallResultGPU, OPT_SZ);
	cudaMallocManaged(&params->h_PutResultGPU, OPT_SZ);
	cudaMallocManaged(&params->h_StockPrice, OPT_SZ);
	cudaMallocManaged(&params->h_OptionStrike, OPT_SZ);
	cudaMallocManaged(&params->h_OptionYears, OPT_SZ);
	
	params->d_CallResult = params->h_CallResultGPU;
    params->d_PutResult = params->h_PutResultGPU;
    params->d_StockPrice = params->h_StockPrice;
    params->d_OptionStrike = params->h_OptionStrike;
	params->d_OptionYears = params->h_OptionYears;
	
	#else
	printf("No transfer model: Exiting ...\n");
		exit(-1);
	#endif
	
#endif
	                                         
    // printf("...generating input data in CPU mem.\n");
    srand(5347);

    //Generate options set
    for (int i = 0; i < OPT_N; i++)
    {
        params->h_CallResultCPU[i] = 0.0f;
        params->h_PutResultCPU[i]  = -1.0f;
        params->h_StockPrice[i]    = RandFloat(5.0f, 30.0f);
        params->h_OptionStrike[i]  = RandFloat(1.0f, 100.0f);
        params->h_OptionYears[i]   = RandFloat(0.25f, 10.0f);
    }

	return 0;
}

int BS_start_transfers(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	t_BS_params * params = (t_BS_params *)kstub->params;
	
#ifdef MEMCPY_SYNC
	//printf("...copying input data to GPU mem.\n");
    //Copy options data to GPU memory for further processing
    /*checkCudaErrors(cudaMemcpy(d_StockPrice,  h_StockPrice,   OPT_SZ, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_OptionStrike, h_OptionStrike,  OPT_SZ, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_OptionYears,  h_OptionYears,   OPT_SZ, cudaMemcpyHostToDevice));*/
    //printf("Data init done.\n\n");
	
	/*HtD_data_transfer(d_StockPrice,  h_StockPrice,   OPT_SZ, C_S);
	HtD_data_transfer(d_OptionStrike, h_OptionStrike,  OPT_SZ, C_S);
	HtD_data_transfer(d_OptionYears,  h_OptionYears,   OPT_SZ, C_S);*/
	
	enqueue_tcomamnd(tqueues, params->d_StockPrice, params->h_StockPrice, OPT_SZ, cudaMemcpyHostToDevice, 0, BLOCKING, DATA, LOW, kstub);
	enqueue_tcomamnd(tqueues, params->d_OptionStrike, params->h_OptionStrike, OPT_SZ, cudaMemcpyHostToDevice, 0, BLOCKING, DATA, LOW, kstub);
	enqueue_tcomamnd(tqueues, params->d_OptionYears, params->h_OptionYears, OPT_SZ, cudaMemcpyHostToDevice, 0, BLOCKING, DATA, LOW, kstub);

   // checkCudaErrors(cudaMemset(d_CallResult, 0, OPT_SZ));
   // checkCudaErrors(cudaMemset(d_PutResult, 0, OPT_SZ));
	
	kstub->HtD_tranfers_finished = 1;

	
#else
	#ifdef MEMCPY_ASYNC	

	
	checkCudaErrors(cudaMemcpyAsync(params->d_StockPrice,  params->h_StockPrice,   OPT_SZ, cudaMemcpyHostToDevice, kstub->transfer_s[0]));
    checkCudaErrors(cudaMemcpyAsync(params->d_OptionStrike, params->h_OptionStrike,  OPT_SZ, cudaMemcpyHostToDevice, kstub->transfer_s[0]));
    checkCudaErrors(cudaMemcpyAsync(params->d_OptionYears,  params->h_OptionYears,   OPT_SZ, cudaMemcpyHostToDevice, kstub->transfer_s[0]));
	/*enqueue_tcomamnd(tqueues, d_StockPrice, h_StockPrice, OPT_SZ, cudaMemcpyHostToDevice, kstub->transfer_s[0], NONBLOCKING, DATA, MEDIUM, kstub);
	enqueue_tcomamnd(tqueues, d_OptionStrike,  h_OptionStrike, OPT_SZ, cudaMemcpyHostToDevice, kstub->transfer_s[0], NONBLOCKING, DATA, MEDIUM, kstub);
	enqueue_tcomamnd(tqueues, d_OptionYears, h_OptionYears, OPT_SZ, cudaMemcpyHostToDevice, kstub->transfer_s[0], NONBLOCKING, LAST_TRANSFER, MEDIUM, kstub);*/
	//cudaEventSynchronize(com->end_transfers);

	//enqueue_tcomamnd(tqueues, NULL, NULL, 0, cudaMemcpyHostToDevice, kstub->transfer_s[0], STREAM_SYNCHRO, DATA, MEDIUM);

    //printf("Data init done.\n\n");

    //checkCudaErrors(cudaMemset(d_CallResult, 0, OPT_SZ/*, kstub->transfer_s[0]*/));
   // checkCudaErrors(cudaMemset(d_PutResult, 0, OPT_SZ/*, kstub->transfer_s[0])*/));
	
	//cudaEventRecord(kstub->end_HtD, kstub->transfer_s[0]);
	cudaStreamSynchronize(kstub->transfer_s[0]);
	kstub->HtD_tranfers_finished = 1;
	
	#else
	#ifdef MANAGED_MEM
	
	cudaDeviceProp p;
    cudaGetDeviceProperties(&p, kstub->deviceId);
	cudaError_t err;

	if (p.concurrentManagedAccess)
	{
		err = cudaMemPrefetchAsync(params->h_CallResultGPU, OPT_SZ, kstub->deviceId, kstub->transfer_s[0]);
		if ( err != cudaSuccess) {
			printf("Error in BS:cudaMemPrefetchAsync\n");
			exit(EXIT_FAILURE);
		}
		err = cudaMemPrefetchAsync(params->h_PutResultGPU, OPT_SZ, kstub->deviceId, kstub->transfer_s[0]);
		if ( err != cudaSuccess) {
			printf("Error in BS:cudaMemPrefetchAsync\n");
			exit(EXIT_FAILURE);
		}
		err = cudaMemPrefetchAsync(params->h_StockPrice, OPT_SZ, kstub->deviceId, kstub->transfer_s[0]);
		if ( err != cudaSuccess) {
			printf("Error in BS:cudaMemPrefetchAsync\n");
			exit(EXIT_FAILURE);
		}
		err = cudaMemPrefetchAsync(params->h_OptionStrike, OPT_SZ, kstub->deviceId, kstub->transfer_s[0]);
		if ( err != cudaSuccess) {
			printf("Error in BS:cudaMemPrefetchAsync\n");
			exit(EXIT_FAILURE);
		}
		err = cudaMemPrefetchAsync(params->h_OptionYears, OPT_SZ, kstub->deviceId, kstub->transfer_s[0]);
		if ( err != cudaSuccess) {
			printf("Error in BS:cudaMemPrefetchAsync\n");
			exit(EXIT_FAILURE);
		}
	}
	
	//cudaEventRecord(kstub->end_HtD, kstub->transfer_s[0]);
	
	//cudaStreamSynchronize(kstub->transfer_s[0]);
	kstub->HtD_tranfers_finished = 1;

	
	#endif
	#endif
	
#endif
	
	return 0;
}
	

int BS_end_kernel_dummy(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	t_BS_params * params = (t_BS_params *)kstub->params;
	
#ifdef MEMCPY_SYNC
	
	cudaEventSynchronize(kstub->end_Exec);
    // printf("\nReading back GPU results...\n");
    //Read back GPU results to compare them to CPU results
    /*checkCudaErrors(cudaMemcpy(h_CallResultGPU, d_CallResult, OPT_SZ, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_PutResultGPU,  d_PutResult,  OPT_SZ, cudaMemcpyDeviceToHost));*/
	
	/*DtH_data_transfer(h_CallResultGPU, d_CallResult, OPT_SZ, C_S);
	DtH_data_transfer(h_PutResultGPU,  d_PutResult, OPT_SZ, C_S); */
	
	enqueue_tcomamnd(tqueues, params->h_CallResultGPU, params->d_CallResult, OPT_SZ, cudaMemcpyDeviceToHost, 0, BLOCKING, DATA, LOW, kstub);
	enqueue_tcomamnd(tqueues, params->h_PutResultGPU, params->d_PutResult, OPT_SZ, cudaMemcpyDeviceToHost, 0, BLOCKING, DATA, LOW, kstub);

	
#else
	#ifdef MEMCPY_ASYNC

	//cudaEventSynchronize(kstub->end_Exec);

	checkCudaErrors(cudaMemcpyAsync(params->h_CallResultGPU, params->d_CallResult, OPT_SZ, cudaMemcpyDeviceToHost, kstub->transfer_s[1]));
    checkCudaErrors(cudaMemcpyAsync(params->h_PutResultGPU,  params->d_PutResult,  OPT_SZ, cudaMemcpyDeviceToHost, kstub->transfer_s[1]));
	cudaStreamSynchronize(kstub->transfer_s[1]);
	
	/*checkCudaErrors(cudaMemcpy(h_CallResultGPU, d_CallResult, OPT_SZ, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_PutResultGPU,  d_PutResult,  OPT_SZ, cudaMemcpyDeviceToHost));*/
	
	/*enqueue_tcomamnd(tqueues, h_CallResultGPU, d_CallResult, OPT_SZ, cudaMemcpyDeviceToHost, kstub->transfer_s[1] , NONBLOCKING, DATA, MEDIUM, kstub);
	t_tcommand *com = enqueue_tcomamnd(tqueues, h_PutResultGPU, d_PutResult, OPT_SZ, cudaMemcpyDeviceToHost, kstub->transfer_s[1] , NONBLOCKING, LAST_TRANSFER, MEDIUM, kstub);
	cudaEventSynchronize(com->end_transfers);*/

	//enqueue_tcomamnd(tqueues, NULL, NULL, 0, cudaMemcpyDeviceToHost, kstub->transfer_s[1], STREAM_SYNCHRO, DATA, MEDIUM);

	kstub->DtH_tranfers_finished = 1;
	//printf("-->Fin de DtH para tarea %d\n", kstub->id);


	//cudaEventRecord(kstub->end_DtH, kstub->transfer_s[1]);
	
	/*cudaEventSynchronize(kstub->end_DtH);*/
	
	#else
		#ifdef MANAGED_MEM
			cudaStreamSynchronize(*(kstub->execution_s)); // To be sure kernel execution has finished before processing output data
		#endif
	#endif
#endif

	//printf("Checking the results...\n");
   // printf("...running CPU calculations.\n\n");
    //Calculate options values on CPU
  /*  BlackScholesCPU(
        h_CallResultCPU,
        h_PutResultCPU,
        h_StockPrice,
        h_OptionStrike,
        h_OptionYears,
        RISKFREE,
        VOLATILITY,
        OPT_N
    );

    //printf("Comparing the results...\n");
    //Calculate max absolute difference and L1 distance
    //between CPU and GPU results
    sum_delta = 0;
    sum_ref   = 0;
    max_delta = 0;

    for (int i = 0; i < OPT_N; i++)
    {
        ref   = h_CallResultCPU[i];
        delta = fabs(h_CallResultCPU[i] - h_CallResultGPU[i]);

        if (delta > max_delta)
        {
            max_delta = delta;
        }

        sum_delta += delta;
        sum_ref   += fabs(ref);
    }

    L1norm = sum_delta / sum_ref;
   
//   printf("L1 norm: %E\n", L1norm);
//    printf("Max absolute error: %E\n\n", max_delta);

    //printf("Shutting down...\n");
    //printf("...releasing GPU memory.\n");
	
	   //printf("Shutdown done.\n");

    //printf("\n[BlackScholes] - Test Summary\n");
   // cudaDeviceReset();

    if (L1norm > 1e-3)
    {
        printf("Test failed!\n");
        //exit(EXIT_FAILURE);
    }

    //printf("Test passed\n");
	
	*/
	/*
#if defined(MEMCPY_SYNC) || defined(MEMCPY_ASYNC)
    checkCudaErrors(cudaFree(d_OptionYears));
    checkCudaErrors(cudaFree(d_OptionStrike));
    checkCudaErrors(cudaFree(d_StockPrice));
    checkCudaErrors(cudaFree(d_PutResult));
    checkCudaErrors(cudaFree(d_CallResult));
	

    //printf("...releasing CPU memory.\n");
    cudaFreeHost(h_OptionYears);
    cudaFreeHost(h_OptionStrike);
    cudaFreeHost(h_StockPrice);
    cudaFreeHost(h_PutResultGPU);
    cudaFreeHost(h_CallResultGPU);
    cudaFreeHost(h_PutResultCPU);
    cudaFreeHost(h_CallResultCPU);
	
#else
	cudaFree(h_OptionYears);
    cudaFree(h_OptionStrike);
    cudaFree(h_StockPrice);
    cudaFree(h_PutResultGPU);
    cudaFree(h_CallResultGPU);
    cudaFree(h_PutResultCPU);
    cudaFree(h_CallResultCPU);
	
#endif
    
	*/
	return 0;
}

int launch_orig_BS(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	t_BS_params * params = (t_BS_params *)kstub->params;
	
	original_BlackScholesGPU <<<kstub->kconf.gridsize.x, kstub->kconf.blocksize.x>>>(
			params->d_CallResult,
            params->d_PutResult,
            params->d_StockPrice,
            params->d_OptionStrike,
            params->d_OptionYears,
            RISKFREE,
            VOLATILITY,
            OPT_N,
			
		kstub->kconf.coarsening);
		
		return 0;
}

int launch_slc_BS(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	t_BS_params * params = (t_BS_params *)kstub->params;
	
	slicing_BlackScholesGPU <<<kstub->total_tasks, kstub->kconf.blocksize.x, 0, *(kstub->execution_s)>>>(
			params->d_CallResult,
            params->d_PutResult,
            params->d_StockPrice,
            params->d_OptionStrike,
            params->d_OptionYears,
            RISKFREE,
            VOLATILITY,
            OPT_N,
			
		kstub->kconf.coarsening, kstub->kconf.initial_blockID, params->zc_slc);
		
		return 0;
}

int prof_BS(void * arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	t_BS_params * params = (t_BS_params *)kstub->params;
	
	profiling_BlackScholesGPU<<<kstub->kconf.numSMs * kstub->kconf.max_persistent_blocks, kstub->kconf.blocksize.x, 0, *(kstub->execution_s)>>>(
			params->d_CallResult,
            params->d_PutResult,
            params->d_StockPrice,
            params->d_OptionStrike,
            params->d_OptionYears,
            RISKFREE,
            VOLATILITY,
            OPT_N,
			
			kstub->total_tasks,
			kstub->kconf.coarsening,
			kstub->d_SMs_cont,
			kstub->d_executed_tasks,
			&kstub->gm_state[kstub->stream_index]);
			
	return 0;
}	

int launch_preemp_BS(void *arg)
{
	
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	t_BS_params * params = (t_BS_params *)kstub->params;
	
	#ifdef SMT
	
	preemp_SMT_BlackScholesGPU<<<kstub->kconf.numSMs * kstub->kconf.max_persistent_blocks, kstub->kconf.blocksize.x, 0, *(kstub->execution_s)>>>(
			params->d_CallResult,
            params->d_PutResult,
            params->d_StockPrice,
            params->d_OptionStrike,
            params->d_OptionYears,
            RISKFREE,
            VOLATILITY,
            OPT_N,
			
			kstub->idSMs[0],
			kstub->idSMs[1],
			kstub->total_tasks,
			kstub->kconf.coarsening,
			kstub->d_executed_tasks,
			&kstub->gm_state[kstub->stream_index]
    );
	#else
	preemp_SMK_BlackScholesGPU<<<kstub->kconf.numSMs * kstub->kconf.max_persistent_blocks, kstub->kconf.blocksize.x, 0, *(kstub->execution)>>>(
			params->d_CallResult,
            params->d_PutResult,
            params->d_StockPrice,
            params->d_OptionStrike,
            params->d_OptionYears,
            RISKFREE,
            VOLATILITY,
            OPT_N,
			
			kstub->num_blocks_per_SM,
			kstub->total_tasks, 
			kstub->kconf.coarsening,
			kstub->d_SMs_cont,
			kstub->d_executed_tasks,
			&kstub->gm_state[kstub->stream_index]
    );
		
	#endif
	
	return 0;
}
