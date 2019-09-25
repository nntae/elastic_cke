#include <helper_functions.h>   // helper functions for string parsing
#include <helper_cuda.h>  
#include <semaphore.h>

#define HISTOGRAM256_BIN_COUNT 256
#define UINT_BITS 32
typedef unsigned int uint;
typedef unsigned char uchar;

////////////////////////////////////////////////////////////////////////////////
// GPU-specific common definitions
////////////////////////////////////////////////////////////////////////////////
#define LOG2_WARP_SIZE 5U
#define WARP_SIZE (1U << LOG2_WARP_SIZE)

//May change on future hardware, so better parametrize the code
#define SHARED_MEMORY_BANKS 16

//Warps ==subhistograms per threadblock
//#define WARP_COUNT 6

//Threadblock size
//#define HISTOGRAM256_THREADBLOCK_SIZE (WARP_COUNT * WARP_SIZE)

//Shared memory per threadblock
//#define HISTOGRAM256_THREADBLOCK_MEMORY (WARP_COUNT * HISTOGRAM256_BIN_COUNT)

#define UMUL(a, b) ( (a) * (b) )
#define UMAD(a, b, c) ( UMUL((a), (b)) + (c) )

#define MERGE_THREADBLOCK_SIZE 256

#define TAG_MASK 0xFFFFFFFFU

#define PARTIAL_HISTOGRAM256_COUNT 224

typedef struct {
	uchar *h_Data256;
	uint  *h_HistogramGPU256;
	uint  *h_PartialHistograms256;
	uchar *d_Data256;
	uint  *d_Histogram256;
	uint *d_PartialHistograms256;
	
	uint byteCount256;
	
	int warp_count;
	int histogram256_threadblock_size;
	int histogram256_threadblock_memory;
} t_HST256_params;

/*** histogram ***/	
int launch_preemp_HST256(void *kstub);
int launch_orig_HST256(void *kstub);

int HST256_start_kernel(void *arg);
int HST256_end_kernel(void *arg);

int HST256_start_mallocs(void *arg);
int HST256_start_transfers(void *arg);