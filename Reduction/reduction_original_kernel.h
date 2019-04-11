#ifndef REDUCTION_KERNEL_H_
#define REDUCTION_KERNEL_H_

#include <cuda.h>

// The following class is a workaround for using dynamically sized
// shared memory in templated code. Without this workaround, the
// compiler would generate two shared memory arrays (one for SP
// and one for DP) of the same name and would generate an error.
template <class T>
class SharedMem
{
    public:
      __device__ inline T* getPointer()
      {
          extern __shared__ T s[];
          return s;
      };
};

// Specialization for double
template <>
class SharedMem <double>
{
    public:
      __device__ inline double* getPointer()
      {
          extern __shared__ double s_double[];
          return s_double;
      }
};

// specialization for float
template <>
class SharedMem <float>
{
    public:
      __device__ inline float* getPointer()
      {
          extern __shared__ float s_float[];
          return s_float;
      }
};

// Reduction Kernel
template <class T, int blockSize>
__global__ void
templated_reduce(const T* __restrict__ g_idata, T* __restrict__ g_odata,
        const unsigned int n)
{

    const unsigned int tid = threadIdx.x;
    unsigned int i = (blockIdx.x*(blockDim.x*2)) + tid;
    const unsigned int gridSize = blockDim.x*2*gridDim.x;

    // Shared memory will be used for intrablock summation
    // NB: CC's < 1.3 seem incompatible with the templated dynamic
    // shared memory workaround.
    // Inspection with cuda-gdb shows sdata as a pointer to *global*
    // memory and incorrect results are obtained.  This explicit macro
    // works around this issue. Further, it is acceptable to specify
    // float, since CC's < 1.3 do not support double precision.
#if __CUDA_ARCH__ <= 130
    extern volatile __shared__ float sdata[];
#else
    SharedMem<T> shared;
    volatile T* sdata = shared.getPointer();
#endif

    sdata[tid] = 0.0f;

    // Reduce multiple elements per thread
    while (i < n)
    {
        sdata[tid] += g_idata[i] + g_idata[i+blockSize];
        i += gridSize;
    }
    __syncthreads();

    // Reduce the contents of shared memory
    // NB: This is an unrolled loop, and assumes warp-syncrhonous
    // execution.
    if (blockSize >= 512)
    {
        if (tid < 256)
        {
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
    }
    if (blockSize >= 256)
    {
        if (tid < 128)
        {
            sdata[tid] += sdata[tid + 128];
        }
        __syncthreads();
    }
    if (blockSize >= 128)
    {
        if (tid < 64)  { sdata[tid] += sdata[tid + 64]; }  __syncthreads();
    }
    if (tid < warpSize)
    {
        // NB2: This section would also need __sync calls if warp
        // synchronous execution were not assumed
        if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
        if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
        if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
        if (blockSize >= 8)  sdata[tid] += sdata[tid + 4];
        if (blockSize >= 4)  sdata[tid] += sdata[tid + 2];
        if (blockSize >= 2)  sdata[tid] += sdata[tid + 1];
    }

    // Write result for this block to global memory
    if (tid == 0)
    {
        g_odata[blockIdx.x] = sdata[0];
    }
}


__device__ uint get_smid_reduction(void) {

     uint ret;

     asm("mov.u32 %0, %smid;" : "=r"(ret) );

     return ret;

}

// Reduction Kernel
// Kernel has been changed to increase the number of blocks. Now, one block
//   only reduces 2*blockDim.x elements
__global__ void
reduce(const float* __restrict__ g_idata, float* __restrict__ g_odata,
        const unsigned long int n)
{

    const unsigned long int tid = threadIdx.x;
    unsigned long int i = (blockIdx.x*(blockDim.x*2)) + tid;
    const unsigned long int gridSize = blockDim.x*2*gridDim.x;
	const unsigned long int blockSize = blockDim.x;

    // Shared memory will be used for intrablock summation
    // NB: CC's < 1.3 seem incompatible with the templated dynamic
    // shared memory workaround.
    // Inspection with cuda-gdb shows sdata as a pointer to *global*
    // memory and incorrect results are obtained.  This explicit macro
    // works around this issue. Further, it is acceptable to specify
    // float, since CC's < 1.3 do not support double precision.
#if __CUDA_ARCH__ <= 130
    extern volatile __shared__ float sdata[];
#else
    SharedMem<float> shared;
    volatile float* sdata = shared.getPointer();
#endif

    sdata[tid] = 0.0f;

    // Reduce multiple elements per thread
    while (i < n)
    {
        sdata[tid] += g_idata[i] + g_idata[i+blockSize];
        i += gridSize;
    }
    __syncthreads();

    // Reduce the contents of shared memory
    // NB: This is an unrolled loop, and assumes warp-syncrhonous
    // execution.
    if (blockSize >= 512)
    {
        if (tid < 256)
        {
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
    }
    if (blockSize >= 256)
    {
        if (tid < 128)
        {
            sdata[tid] += sdata[tid + 128];
        }
        __syncthreads();
    }
    if (blockSize >= 128)
    {
        if (tid < 64)  { sdata[tid] += sdata[tid + 64]; }  __syncthreads();
    }
    if (tid < warpSize)
    {
        // NB2: This section would also need __sync calls if warp
        // synchronous execution were not assumed
        if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
        if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
        if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
        if (blockSize >= 8)  sdata[tid] += sdata[tid + 4];
        if (blockSize >= 4)  sdata[tid] += sdata[tid + 2];
        if (blockSize >= 2)  sdata[tid] += sdata[tid + 1];
    }

    // Write result for this block to global memory
    if (tid == 0)
    {
        g_odata[blockIdx.x] = sdata[0];
    }
}

// Reduction Kernel
__global__ void
preemp_SMT_reduce_kernel(const float* __restrict__ g_idata, float* __restrict__ g_odata,
        const unsigned long int n,
		int SIMD_min,
		int SIMD_max,
		unsigned long int num_subtask,
		int iter_per_subtask,
		int *cont_subtask,
		State *status)
{
	__shared__ int s_bid;
    // Shared memory will be used for intrablock summation
    // NB: CC's < 1.3 seem incompatible with the templated dynamic
    // shared memory workaround.
    // Inspection with cuda-gdb shows sdata as a pointer to *global*
    // memory and incorrect results are obtained.  This explicit macro
    // works around this issue. Further, it is acceptable to specify
    // float, since CC's < 1.3 do not support double precision.
#if __CUDA_ARCH__ <= 130
    extern volatile __shared__ float sdata[];
#else
    SharedMem<float> shared;
    volatile float* sdata = shared.getPointer();
#endif

	unsigned int SM_id = get_smid_reduction();
	
	if (SM_id <SIMD_min || SM_id > SIMD_max) /* Only blocks executing within SIMD_min and SIMD_max can progress */ 
			return;
		
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
			return;
		}
		
		const unsigned long int tid = threadIdx.x;
//		unsigned long int i = (s_bid * iter_per_subtask + iter * blockDim.x)*(blockDim.x*2) + tid;
		unsigned long int i = (unsigned long int) (s_bid * blockDim.x * 2) + tid;
		const unsigned long int gridSize = blockDim.x*2*num_subtask; //gridDim.x;
		const unsigned long int blockSize = blockDim.x;
		
		for (int iter=0; iter<iter_per_subtask; iter++) {
			sdata[tid] = 0.0f;

			// Reduce multiple elements per thread
			while (i < n)
			{
				sdata[tid] += g_idata[i] + g_idata[i+blockSize];
				i += gridSize;
			}
			__syncthreads();

			// Reduce the contents of shared memory
			// NB: This is an unrolled loop, and assumes warp-syncrhonous
			// execution.
			if (blockSize >= 512)
			{
				if (tid < 256)
				{
					sdata[tid] += sdata[tid + 256];
				}
				__syncthreads();
			}
			if (blockSize >= 256)
			{
				if (tid < 128)
				{
					sdata[tid] += sdata[tid + 128];
				}
				__syncthreads();
			}
			if (blockSize >= 128)
			{
				if (tid < 64)  { sdata[tid] += sdata[tid + 64]; }  __syncthreads();
			}
			if (tid < warpSize)
			{
				// NB2: This section would also need __sync calls if warp
				// synchronous execution were not assumed
				if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
				if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
				if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
				if (blockSize >= 8)  sdata[tid] += sdata[tid + 4];
				if (blockSize >= 4)  sdata[tid] += sdata[tid + 2];
				if (blockSize >= 2)  sdata[tid] += sdata[tid + 1];
			}

			// Write result for this block to global memory
			if (tid == 0)
			{
				g_odata[s_bid] = sdata[0];
			}
		}
	}
}

// Reduction Kernel
__global__ void
preemp_SMK_reduce_kernel(const float* __restrict__ g_idata, float* __restrict__ g_odata,
        const unsigned long int n,
		int max_blocks_per_SM, 
		unsigned long int num_subtask,
		int iter_per_subtask,
		int *cont_SM,
		int *cont_subtask,
		State *status)
{
	__shared__ int s_bid, s_index;

    // Shared memory will be used for intrablock summation
    // NB: CC's < 1.3 seem incompatible with the templated dynamic
    // shared memory workaround.
    // Inspection with cuda-gdb shows sdata as a pointer to *global*
    // memory and incorrect results are obtained.  This explicit macro
    // works around this issue. Further, it is acceptable to specify
    // float, since CC's < 1.3 do not support double precision.
#if __CUDA_ARCH__ <= 130
    extern volatile __shared__ float sdata[];
#else
    SharedMem<float> shared;
    volatile float* sdata = shared.getPointer();
#endif
	
	unsigned int SM_id = get_smid_reduction();
	
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
		__syncthreads();
		
		if (s_bid >= num_subtask || s_bid == -1) /* If all subtasks have been executed */
			return;
			
		const unsigned int tid = threadIdx.x;
//		unsigned long int i = (s_bid * iter_per_subtask + iter * blockDim.x)*(blockDim.x*2) + tid;
		unsigned long int i = (unsigned long int) (s_bid * blockDim.x * 2) + tid;
		const unsigned long int gridSize = blockDim.x*2*num_subtask;
		const unsigned int blockSize = blockDim.x;

		
		for (int iter=0; iter<iter_per_subtask; iter++) {
			sdata[tid] = 0.0f;

			// Reduce multiple elements per thread
			while (i < n)
			{
				sdata[tid] += g_idata[i] + g_idata[i+blockSize];
				i += gridSize;
			}
			__syncthreads();

			// Reduce the contents of shared memory
			// NB: This is an unrolled loop, and assumes warp-syncrhonous
			// execution.
			if (blockSize >= 512)
			{
				if (tid < 256)
				{
					sdata[tid] += sdata[tid + 256];
				}
				__syncthreads();
			}
			if (blockSize >= 256)
			{
				if (tid < 128)
				{
					sdata[tid] += sdata[tid + 128];
				}
				__syncthreads();
			}
			if (blockSize >= 128)
			{
				if (tid < 64)  { sdata[tid] += sdata[tid + 64]; }  __syncthreads();
			}
			if (tid < warpSize)
			{
				// NB2: This section would also need __sync calls if warp
				// synchronous execution were not assumed
				if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
				if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
				if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
				if (blockSize >= 8)  sdata[tid] += sdata[tid + 4];
				if (blockSize >= 4)  sdata[tid] += sdata[tid + 2];
				if (blockSize >= 2)  sdata[tid] += sdata[tid + 1];
			}

			// Write result for this block to global memory
			if (tid == 0)
			{
				g_odata[s_bid] = sdata[0];
			}
		}
	}
}



#endif // REDUCTION_KERNEL_H_
