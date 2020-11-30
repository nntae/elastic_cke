#include "cudacommon.h"
#include <cassert>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include "SPMV.h"
#include <semaphore.h>
#include "../elastic_kernel.h"

#include "../memaddrcnt.cuh"

texture<float, 1> vecTex;  // vector textures
texture<int2, 1>  vecTexD;


// Texture Readers (used so kernels can be templated)
struct texReaderSP {
   __device__ __forceinline__ float operator()(const int idx) const
   {
       return tex1Dfetch(vecTex, idx); 
   }
};

extern t_tqueue *tqueues;


__device__ uint get_smid_SPMV(void) {

     uint ret;

     asm("mov.u32 %0, %smid;" : "=r"(ret) );

     return ret;

}


// ****************************************************************************
// Function: spmv_csr_scalar_kernel
//
// Purpose:
//   Computes sparse matrix - vector multiplication on the GPU using
//   the CSR data storage format, using a thread per row of the sparse
//   matrix; based on Bell (SC09) and Baskaran (IBM Tech Report)
//
// Arguments:
//   val: array holding the non-zero values for the matrix
//   cols: array of column indices for each element of the sparse matrix
//   rowDelimiters: array of size dim+1 holding indices to rows of the matrix
//                  last element is the index one past the last
//                  element of the matrix
//   dim: number of rows in the matrix
//   out: output - result from the spmv calculation
//
// Returns:  nothing
//           out indirectly through a pointer
//
// Programmer: Lukasz Wesolowski
// Creation: June 28, 2010
//
// Modifications:
//
// ****************************************************************************
__global__ void
original_spmv_csr_scalar_kernel(const float * __restrict__ val,
                       const int    * __restrict__ cols,
                       const int    * __restrict__ rowDelimiters,
                       const int dim, float * __restrict__ out)
{
    int myRow = blockIdx.x * blockDim.x + threadIdx.x;
    texReaderSP vecTexReader;
	
	while (myRow < dim)
    //if (myRow < dim)
    {
        float t = 0.0f;
        int start = rowDelimiters[myRow];
        int end = rowDelimiters[myRow+1];
        for (int j = start; j < end; j++)
        {
            int col = cols[j];
            t += val[j] * vecTexReader(col);
        }
        out[myRow] = t;
		myRow += gridDim.x * blockDim.x;
    }
}

__global__ void
slicing_spmv_csr_scalar_kernel(const float * __restrict__ val,
                       const int    * __restrict__ cols,
                       const int    * __restrict__ rowDelimiters,
                       const int dim, float * __restrict__ out, int gridDimX, int init_blkIdx, int *zc_slc)
{
    int myRow = (blockIdx.x + init_blkIdx) * blockDim.x + threadIdx.x;
    texReaderSP vecTexReader;
	
	if (threadIdx.x == 0 && threadIdx.y == 0) atomicAdd(zc_slc, 1);

	while (myRow < dim)
    //if (myRow < dim)
    {
        float t = 0.0f;
        int start = rowDelimiters[myRow];
        int end = rowDelimiters[myRow+1];
        for (int j = start; j < end; j++)
        {
            int col = cols[j];
            t += val[j] * vecTexReader(col);
        }
        out[myRow] = t;
		myRow +=  gridDimX * blockDim.x;
    }
}

__global__ void
preemp_SMK_spmv_csr_scalar_kernel(const float * __restrict__ val,
                       const int    * __restrict__ cols,
                       const int    * __restrict__ rowDelimiters,
                       const int dim, float * __restrict__ out,
					   
					   int max_blocks_per_SM, 
						int num_subtask,
						int iter_per_subtask,
						int *cont_SM,
						int *cont_subtask,
						State *status
					   )
{
	int myRow;
	
	__shared__ int s_bid, s_index;
	
	unsigned int SM_id = get_smid_SPMV();
	
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
	
		myRow = s_bid * blockDim.x + threadIdx.x;
		texReaderSP vecTexReader;

		while (myRow < dim)
		{
			float t = 0.0f;
			int start = rowDelimiters[myRow];
			int end = rowDelimiters[myRow+1];
			for (int j = start; j < end; j++) {
				int col = cols[j];
				t += val[j] * vecTexReader(col);
			}
			out[myRow] = t;
			myRow += num_subtask * blockDim.x;
		}
	}
}


__global__ void
profiling_csr_scalar_kernel(const float * __restrict__ val,
						const int    * __restrict__ cols,
						const int    * __restrict__ rowDelimiters,
						const int dim, float * __restrict__ out,
						
						int num_subtask,
						int iter_per_subtask,
						int *cont_SM,
						int *cont_subtask,
						State *status
						)
{
	int myRow;
	__shared__ int s_bid, CTA_cont;
	
	unsigned int SM_id = get_smid_SPMV();
	
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
		//delay();
		return;
	}
	
	//if (threadIdx.x == 0)
	//	printf ("SM=%d CTA = %d\n", SM_id, CTA_cont);

	int cont_task = 0;
	/*if (threadIdx.x == 0 && blockIdx.x % 40 == 0)
		printf("Bloque=%d SM_id=%d\n", blockIdx.x, SM_id);  */
		
	while (1){
		
		/********** Task Id calculation *************/
		if (threadIdx.x == 0) {
			if (*status == TOEVICT) {
				//printf("Sennal eviction %d %d %d %d\n", blockIdx.x, *cont_subtask, iter_per_subtask, num_subtask);
				s_bid = -1;
			}
			else
				s_bid = atomicAdd((int *)cont_subtask, 1);
		}
		
		__syncthreads();
		
		if (s_bid >= num_subtask || s_bid == -1){ /* If all subtasks have been executed */
			//if (blockIdx.x ==0 && threadIdx.x == 0) printf("Blk=%d num_tasks=%d Saliendo por %d\n", blockIdx.x, num_subtask, s_bid );
			if (threadIdx.x == 0)
				printf ("SM=%d CTA=%d Executed_tasks= %d \n", SM_id, CTA_cont, cont_task);
			return;
		}
		
		if (threadIdx.x == 0) // Acumula numeor de tareas ejecutadas
			cont_task++;
		
		//for (int iter=0; iter<iter_per_subtask; iter++) {
	
			myRow = s_bid * blockDim.x + threadIdx.x;
			//int myRow = s_bid * blockDim.x * iter_per_subtask + iter * blockDim.x + threadIdx.x;
			texReaderSP vecTexReader;

			//if (blockIdx.x==0 && threadIdx.x==0)
				//printf("bid=%d Row=%d, start=%d, end=%d ", s_bid, myRow, rowDelimiters[myRow], rowDelimiters[myRow+1]);
			if (myRow < dim)
			{
				float t = 0.0f;
				int start = rowDelimiters[myRow];
				int end = rowDelimiters[myRow+1];
				for (int j = start; j < end; j++) {
					int col = cols[j];
					t += val[j] * vecTexReader(col);
				}
				out[myRow] = t;
				//if (blockIdx.x==0 && threadIdx.x==0)
				//	printf("Result=%f\n", t);
			}
		//}
	}
}


__global__ void
preemp_SMT_spmv_csr_scalar_kernel(const float * __restrict__ val,
                       const int    * __restrict__ cols,
                       const int    * __restrict__ rowDelimiters,
                       const int dim, float * __restrict__ out,

						int SIMD_min,
						int SIMD_max,
						int num_subtask,
						int iter_per_subtask,
						int *cont_subtask,
						State *status
					   )
{
	int myRow;
	__shared__ int s_bid;
	
	unsigned int SM_id = get_smid_SPMV();
	
	if (SM_id <SIMD_min || SM_id > SIMD_max) /* Only blocks executing within SIMD_min and SIMD_max can progress */ 
			return;
			
	/*if (threadIdx.x == 0 && blockIdx.x % 40 == 0)
		printf("Bloque=%d SM_id=%d\n", blockIdx.x, SM_id);  */
		
	while (1){
		
		/********** Task Id calculation *************/
		if (threadIdx.x == 0) {
			if (*status == TOEVICT) {
				//printf("Sennal eviction %d %d %d %d\n", blockIdx.x, *cont_subtask, iter_per_subtask, num_subtask);
				s_bid = -1;
			}
			else
				s_bid = atomicAdd((int *)cont_subtask, 1);
		}
		
		__syncthreads();
		
		if (s_bid >= num_subtask || s_bid == -1){ /* If all subtasks have been executed */
			//if (blockIdx.x ==0 && threadIdx.x == 0) printf("Blk=%d num_tasks=%d Saliendo por %d\n", blockIdx.x, num_subtask, s_bid );
			return;
		}
		
		//for (int iter=0; iter<iter_per_subtask; iter++) {
	
			myRow = s_bid * blockDim.x + threadIdx.x;
			//int myRow = s_bid * blockDim.x * iter_per_subtask + iter * blockDim.x + threadIdx.x;
			texReaderSP vecTexReader;

			//if (blockIdx.x==0 && threadIdx.x==0)
				//printf("bid=%d Row=%d, start=%d, end=%d ", s_bid, myRow, rowDelimiters[myRow], rowDelimiters[myRow+1]);
			if (myRow < dim)
			{
				float t = 0.0f;
				int start = rowDelimiters[myRow];
				int end = rowDelimiters[myRow+1];
				for (int j = start; j < end; j++) {
					int col = cols[j];
					t += val[j] * vecTexReader(col);
				}
				out[myRow] = t;
				//if (blockIdx.x==0 && threadIdx.x==0)
				//	printf("Result=%f\n", t);
			}
		//}
	}
}

__global__ void
memaddr_preemp_SMT_spmv_csr_scalar_kernel(const float * __restrict__ val,
                       const int    * __restrict__ cols,
                       const int    * __restrict__ rowDelimiters,
                       const int dim, float * __restrict__ out,

						int *numUniqueAddr, int SIMD_min,
						int SIMD_max,
						int num_subtask,
						int iter_per_subtask,
						int *cont_subtask,
						State *status
					   )
{
	int myRow;
	__shared__ int s_bid;
	
	unsigned int SM_id = get_smid_SPMV();
	
	if (SM_id <SIMD_min || SM_id > SIMD_max) /* Only blocks executing within SIMD_min and SIMD_max can progress */ 
			return;
			
	/*if (threadIdx.x == 0 && blockIdx.x % 40 == 0)
		printf("Bloque=%d SM_id=%d\n", blockIdx.x, SM_id);  */
		
	while (1){
		
		/********** Task Id calculation *************/
		if (threadIdx.x == 0) {
			if (*status == TOEVICT) {
				//printf("Sennal eviction %d %d %d %d\n", blockIdx.x, *cont_subtask, iter_per_subtask, num_subtask);
				s_bid = -1;
			}
			else
				s_bid = atomicAdd((int *)cont_subtask, 1);
		}
		
		__syncthreads();
		
		if (s_bid >= num_subtask || s_bid == -1){ /* If all subtasks have been executed */
			//if (blockIdx.x ==0 && threadIdx.x == 0) printf("Blk=%d num_tasks=%d Saliendo por %d\n", blockIdx.x, num_subtask, s_bid );
			return;
		}
		
		//for (int iter=0; iter<iter_per_subtask; iter++) {
	
			myRow = s_bid * blockDim.x + threadIdx.x;
			//int myRow = s_bid * blockDim.x * iter_per_subtask + iter * blockDim.x + threadIdx.x;
			texReaderSP vecTexReader;

			//if (blockIdx.x==0 && threadIdx.x==0)
				//printf("bid=%d Row=%d, start=%d, end=%d ", s_bid, myRow, rowDelimiters[myRow], rowDelimiters[myRow+1]);
			if (myRow < dim)
			{
				float t = 0.0f;
#if defined(COUNT_ALL_TASKS)
				if ( s_bid == 0 )
#endif
				{
					get_unique_lines((intptr_t) &rowDelimiters[myRow], numUniqueAddr);
				}
				int start = rowDelimiters[myRow];
#if defined(COUNT_ALL_TASKS)
				if ( s_bid == 0 )
#endif
				{
					get_unique_lines((intptr_t) &rowDelimiters[myRow+1], numUniqueAddr);
				}
				int end = rowDelimiters[myRow+1];
				for (int j = start; j < end; j++) {
#if defined(COUNT_ALL_TASKS)
				if ( s_bid == 0 )
#endif
					{
						get_unique_lines((intptr_t) &cols[j], numUniqueAddr);
					}
					int col = cols[j];
#if defined(COUNT_ALL_TASKS)
				if ( s_bid == 0 )
#endif
					{
						get_unique_lines((intptr_t) &val[j], numUniqueAddr);
					}
					t += val[j] * vecTexReader(col);
				}
#if defined(COUNT_ALL_TASKS)
				if ( s_bid == 0 )
#endif
				{
					get_unique_lines((intptr_t) &out[myRow], numUniqueAddr);
				}
				out[myRow] = t;
				//if (blockIdx.x==0 && threadIdx.x==0)
				//	printf("Result=%f\n", t);
			}
		//}
	}
}

// ****************************************************************************
// Function: spmv_csr_vector_kernel
//
// Purpose:
//   Computes sparse matrix - vector multiplication on the GPU using
//   the CSR data storage format, using a warp per row of the sparse
//   matrix; based on Bell (SC09) and Baskaran (IBM Tech Report)
//
// Arguments:
//   val: array holding the non-zero values for the matrix
//   cols: array of column indices for each element of the sparse matrix
//   rowDelimiters: array of size dim+1 holding indices to rows of the matrix
//                  last element is the index one past the last
//                  element of the matrix
//   dim: number of rows in the matrix
//   out: output - result from the spmv calculation
//
// Returns:  nothing
//           out indirectly through a pointer
//
// Programmer: Lukasz Wesolowski
// Creation: June 28, 2010
//
// Modifications:
//
// ****************************************************************************
template <int BLOCK_SIZE>
__global__ void
original_spmv_csr_vector_kernel(const float * __restrict__ val,
                       const int    * __restrict__ cols,
                       const int    * __restrict__ rowDelimiters,
                       const int dim, float * __restrict__ out)
{
    // Thread ID in block
    int t = threadIdx.x;
    // Thread ID within warp
    int id = t & (warpSize-1);
    int warpsPerBlock = blockDim.x / warpSize;
    // One row per warp
    int myRow = (blockIdx.x * warpsPerBlock) + (t / warpSize);
    // Texture reader for the dense vector
    texReaderSP vecTexReader;

    __shared__ volatile float partialSums[BLOCK_SIZE];

    if (myRow < dim)
    {
        int warpStart = rowDelimiters[myRow];
        int warpEnd = rowDelimiters[myRow+1];
        float mySum = 0;
        for (int j = warpStart + id; j < warpEnd; j += warpSize)
        {
            int col = cols[j];
            mySum += val[j] * vecTexReader(col);
        }
        partialSums[t] = mySum;

        // Reduce partial sums
        if (id < 16) partialSums[t] += partialSums[t+16];
        if (id <  8) partialSums[t] += partialSums[t+ 8];
        if (id <  4) partialSums[t] += partialSums[t+ 4];
        if (id <  2) partialSums[t] += partialSums[t+ 2];
        if (id <  1) partialSums[t] += partialSums[t+ 1];

        // Write result
        if (id == 0)
        {
            out[myRow] = partialSums[t];
        }
    }
}

template <int BLOCK_SIZE>
__global__ void
preemp_SMK_spmv_csr_vector_kernel(const float * __restrict__ val,
                       const int    * __restrict__ cols,
                       const int    * __restrict__ rowDelimiters,
                       const int dim, float * __restrict__ out,
						
						int max_blocks_per_SM, 
						int num_subtask,
						int iter_per_subtask,
						int *cont_SM,
						int *cont_subtask,
						State *status					   
					   )
{
	
	__shared__ int s_bid, s_index;
	__shared__ volatile float partialSums[BLOCK_SIZE];
	
	unsigned int SM_id = get_smid_SPMV();
	
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
		
		for (int iter=0; iter<iter_per_subtask; iter++) {
	
			// Thread ID in block
			int t = threadIdx.x;
			// Thread ID within warp
			int id = t & (warpSize-1);
			int warpsPerBlock = blockDim.x / warpSize;
			// One row per warp
			int myRow = (s_bid * warpsPerBlock * iter_per_subtask + iter * warpsPerBlock) + (t / warpSize);
			// Texture reader for the dense vector
			texReaderSP vecTexReader;

			if (myRow < dim)
			{
				int warpStart = rowDelimiters[myRow];
				int warpEnd = rowDelimiters[myRow+1];
				float mySum = 0;
				for (int j = warpStart + id; j < warpEnd; j += warpSize) {
					int col = cols[j];
					mySum += val[j] * vecTexReader(col);
				}
				partialSums[t] = mySum;

				// Reduce partial sums
				if (id < 16) partialSums[t] += partialSums[t+16];
				if (id <  8) partialSums[t] += partialSums[t+ 8];
				if (id <  4) partialSums[t] += partialSums[t+ 4];
				if (id <  2) partialSums[t] += partialSums[t+ 2];
				if (id <  1) partialSums[t] += partialSums[t+ 1];

				// Write result
				if (id == 0) {
					out[myRow] = partialSums[t];
				}
			}
		}
	}
		
}

template <int BLOCK_SIZE>
__global__ void
preemp_SMT_spmv_csr_vector_kernel(const float * __restrict__ val,
                       const int    * __restrict__ cols,
                       const int    * __restrict__ rowDelimiters,
                       const int dim, float * __restrict__ out,
						
						int SIMD_min,
						int SIMD_max,
						int num_subtask,
						int iter_per_subtask,
						int *cont_subtask,
						State *status				   
					   )
{
	
	__shared__ int s_bid;
	__shared__ volatile float partialSums[BLOCK_SIZE];
	
	unsigned int SM_id = get_smid_SPMV();
	
	if (SM_id <SIMD_min || SM_id > SIMD_max) /* Only blocks executing within SIMD_min and SIMD_max can progress */ 
			return;
		
	while (1){
		
		/********** Task Id calculation *************/
		if (threadIdx.x == 0) {
			if (*status == TOEVICT){
				s_bid = -1;
			}
			else {
				s_bid = atomicAdd(cont_subtask, 1);
			}
		}
		
		__syncthreads();
		
		if (s_bid >= num_subtask || s_bid == -1){ /* If all subtasks have been executed */
			return;
		}
		
		for (int iter=0; iter<iter_per_subtask; iter++) {
	
			// Thread ID in block
			int t = threadIdx.x;
			// Thread ID within warp
			int id = t & (warpSize-1);
			int warpsPerBlock = blockDim.x / warpSize;
			// One row per warp
			int myRow = (s_bid * warpsPerBlock * iter_per_subtask + iter * warpsPerBlock) + (t / warpSize);
			// Texture reader for the dense vector
			texReaderSP vecTexReader;

			if (myRow < dim)
			{
				int warpStart = rowDelimiters[myRow];
				int warpEnd = rowDelimiters[myRow+1];
				float mySum = 0;
				for (int j = warpStart + id; j < warpEnd; j += warpSize) {
					int col = cols[j];
					mySum += val[j] * vecTexReader(col);
				}
				partialSums[t] = mySum;

				// Reduce partial sums
				if (id < 16) partialSums[t] += partialSums[t+16];
				if (id <  8) partialSums[t] += partialSums[t+ 8];
				if (id <  4) partialSums[t] += partialSums[t+ 4];
				if (id <  2) partialSums[t] += partialSums[t+ 2];
				if (id <  1) partialSums[t] += partialSums[t+ 1];

				// Write result
				if (id == 0) {
					out[myRow] = partialSums[t];
				}
			}
		}
	}
		
}

template <int BLOCK_SIZE>
__global__ void
memaddr_preemp_SMT_spmv_csr_vector_kernel(const float * __restrict__ val,
                       const int    * __restrict__ cols,
                       const int    * __restrict__ rowDelimiters,
                       const int dim, float * __restrict__ out,
						
						int *numUniqueAddr,
						int SIMD_min,
						int SIMD_max,
						int num_subtask,
						int iter_per_subtask,
						int *cont_subtask,
						State *status				   
					   )
{
	
	__shared__ int s_bid;
	__shared__ volatile float partialSums[BLOCK_SIZE];
	
	unsigned int SM_id = get_smid_SPMV();
	
	if (SM_id <SIMD_min || SM_id > SIMD_max) /* Only blocks executing within SIMD_min and SIMD_max can progress */ 
			return;
		
	while (1){
		
		/********** Task Id calculation *************/
		if (threadIdx.x == 0) {
			if (*status == TOEVICT){
				s_bid = -1;
			}
			else {
				s_bid = atomicAdd(cont_subtask, 1);
			}
		}
		
		__syncthreads();
		
		if (s_bid >= num_subtask || s_bid == -1){ /* If all subtasks have been executed */
			return;
		}
		
		for (int iter=0; iter<iter_per_subtask; iter++) {
	
			// Thread ID in block
			int t = threadIdx.x;
			// Thread ID within warp
			int id = t & (warpSize-1);
			int warpsPerBlock = blockDim.x / warpSize;
			// One row per warp
			int myRow = (s_bid * warpsPerBlock * iter_per_subtask + iter * warpsPerBlock) + (t / warpSize);
			// Texture reader for the dense vector
			texReaderSP vecTexReader;

			if (myRow < dim)
			{
#if defined(COUNT_ALL_TASKS)
				if ( s_bid == 0 )
#endif
				{
					get_unique_lines((intptr_t) &rowDelimiters[myRow], numUniqueAddr);
				}
				int warpStart = rowDelimiters[myRow];
#if defined(COUNT_ALL_TASKS)
				if ( s_bid == 0 )
#endif
				{
					get_unique_lines((intptr_t) &rowDelimiters[myRow+1], numUniqueAddr);
				}
				int warpEnd = rowDelimiters[myRow+1];
				float mySum = 0;
				for (int j = warpStart + id; j < warpEnd; j += warpSize) {
#if defined(COUNT_ALL_TASKS)
				if ( s_bid == 0 )
#endif
					{
						get_unique_lines((intptr_t) &cols[j], numUniqueAddr);
					}
					int col = cols[j];
#if defined(COUNT_ALL_TASKS)
				if ( s_bid == 0 )
#endif
					{
						get_unique_lines((intptr_t) &val[j], numUniqueAddr);
					}
					mySum += val[j] * vecTexReader(col);
				}
#if defined(COUNT_ALL_TASKS)
				if ( s_bid == 0 )
#endif
					get_conflicting_banks( (intptr_t) &partialSums[t], &numUniqueAddr[1] );					
				partialSums[t] = mySum;

				// Reduce partial sums
				if (id < 16)
				{
#if defined(COUNT_ALL_TASKS)
				if ( s_bid == 0 )
#endif
					{
						get_conflicting_banks( (intptr_t) &partialSums[t], &numUniqueAddr[1] );					
						get_conflicting_banks( (intptr_t) &partialSums[t+16], &numUniqueAddr[1] );					
					}
					partialSums[t] += partialSums[t+16];
				}
				if (id <  8)
				{
#if defined(COUNT_ALL_TASKS)
				if ( s_bid == 0 )
#endif
					{
						get_conflicting_banks( (intptr_t) &partialSums[t], &numUniqueAddr[1] );					
						get_conflicting_banks( (intptr_t) &partialSums[t+8], &numUniqueAddr[1] );					
					}
					partialSums[t] += partialSums[t+ 8];
				}
				if (id <  4)
				{
#if defined(COUNT_ALL_TASKS)
				if ( s_bid == 0 )
#endif
					{
						get_conflicting_banks( (intptr_t) &partialSums[t], &numUniqueAddr[1] );					
						get_conflicting_banks( (intptr_t) &partialSums[t+4], &numUniqueAddr[1] );					
					}
					partialSums[t] += partialSums[t+ 4];
				}
				if (id <  2)
				{
#if defined(COUNT_ALL_TASKS)
				if ( s_bid == 0 )
#endif
					{
						get_conflicting_banks( (intptr_t) &partialSums[t], &numUniqueAddr[1] );					
						get_conflicting_banks( (intptr_t) &partialSums[t+2], &numUniqueAddr[1] );					
					}
					partialSums[t] += partialSums[t+ 2];
				}
				if (id <  1)
				{
#if defined(COUNT_ALL_TASKS)
				if ( s_bid == 0 )
#endif
					{
						get_conflicting_banks( (intptr_t) &partialSums[t], &numUniqueAddr[1] );					
						get_conflicting_banks( (intptr_t) &partialSums[t+1], &numUniqueAddr[1] );					
					}
					partialSums[t] += partialSums[t+ 1];
				}

				// Write result
				if (id == 0) {
#if defined(COUNT_ALL_TASKS)
				if ( s_bid == 0 )
#endif
					{
						get_unique_lines((intptr_t) &out[myRow], numUniqueAddr);
						get_conflicting_banks( (intptr_t) &partialSums[t], &numUniqueAddr[1] );					
					}
					out[myRow] = partialSums[t];
				}
			}
		}
	}
		
}

// ****************************************************************************
// Function: spmv_ellpackr_kernel
//
// Purpose:
//   Computes sparse matrix - vector multiplication on the GPU using
//   the ELLPACK-R data storage format; based on Vazquez et al (Univ. of
//   Almeria Tech Report 2009)
//
// Arguments:
//   val: array holding the non-zero values for the matrix in column
//   major format and padded with zeros up to the length of longest row
//   cols: array of column indices for each element of the sparse matrix
//   rowLengths: array storing the length of each row of the sparse matrix
//   dim: number of rows in the matrix
//   out: output - result from the spmv calculation
//
// Returns:  nothing directly
//           out indirectly through a pointer
//
// Programmer: Lukasz Wesolowski
// Creation: June 29, 2010
//
// Modifications:
//
// ****************************************************************************
__global__ void
original_spmv_ellpackr_kernel(const float * __restrict__ val,
                     const int    * __restrict__ cols,
                     const int    * __restrict__ rowLengths,
                     const int dim, float * __restrict__ out)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    texReaderSP vecTexReader;

    if (t < dim)
    {
        float result = 0.0f;
        int max = rowLengths[t];
        for (int i = 0; i < max; i++)
        {
            int ind = i*dim+t;
            result += val[ind] * vecTexReader(cols[ind]);
        }
        out[t] = result;
    }
}


__global__ void
preemp_SMK_spmv_ellpackr_kernel(const float * __restrict__ val,
                     const int    * __restrict__ cols,
                     const int    * __restrict__ rowLengths,
                     const int dim, float * __restrict__ out,
					 						
						int max_blocks_per_SM, 
						int num_subtask,
						int iter_per_subtask,
						int *cont_SM,
						int *cont_subtask,
						State *status					   
					 )
{
	__shared__ int s_bid, s_index;
	
	unsigned int SM_id = get_smid_SPMV();
	
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
		
		for (int iter=0; iter<iter_per_subtask; iter++) {
	
			int t = blockIdx.x * blockDim.x * iter_per_subtask + iter * blockDim.x + threadIdx.x;
			texReaderSP vecTexReader;

			if (t < dim)
			{
				float result = 0.0f;
				int max = rowLengths[t];
				for (int i = 0; i < max; i++) {
					int ind = i*dim+t;
					result += val[ind] * vecTexReader(cols[ind]);
				}
				out[t] = result;
			}
		}
    }
}

__global__ void
preemp_SMT_spmv_ellpackr_kernel(const float * __restrict__ val,
                     const int    * __restrict__ cols,
                     const int    * __restrict__ rowLengths,
                     const int dim, float * __restrict__ out,
					 						
						int SIMD_min,
						int SIMD_max,
						int num_subtask,
						int iter_per_subtask,
						int *cont_subtask,
						State *status			   
					 )
{
	__shared__ int s_bid;
	
	unsigned int SM_id = get_smid_SPMV();
	
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
		
		for (int iter=0; iter<iter_per_subtask; iter++) {
	
			int t = blockIdx.x * blockDim.x * iter_per_subtask + iter * blockDim.x + threadIdx.x;
			texReaderSP vecTexReader;

			if (t < dim)
			{
				float result = 0.0f;
				int max = rowLengths[t];
				for (int i = 0; i < max; i++) {
					int ind = i*dim+t;
					result += val[ind] * vecTexReader(cols[ind]);
				}
				out[t] = result;
			}
		}
    }
}

__global__ void
memaddr_preemp_SMT_spmv_ellpackr_kernel(const float * __restrict__ val,
                     const int    * __restrict__ cols,
                     const int    * __restrict__ rowLengths,
                     const int dim, float * __restrict__ out,
					 						
						int *numUniqueAddr,
						int SIMD_min,
						int SIMD_max,
						int num_subtask,
						int iter_per_subtask,
						int *cont_subtask,
						State *status			   
					 )
{
	__shared__ int s_bid;
	
	unsigned int SM_id = get_smid_SPMV();
	
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
		
		for (int iter=0; iter<iter_per_subtask; iter++) {
	
			int t = blockIdx.x * blockDim.x * iter_per_subtask + iter * blockDim.x + threadIdx.x;
			texReaderSP vecTexReader;

			if (t < dim)
			{
				float result = 0.0f;
#if defined(COUNT_ALL_TASKS)
				if ( s_bid == 0 )
#endif
				{
					get_unique_lines((intptr_t) &rowLengths[t], numUniqueAddr);
				}
				int max = rowLengths[t];
				for (int i = 0; i < max; i++) {
					int ind = i*dim+t;
#if defined(COUNT_ALL_TASKS)
				if ( s_bid == 0 )
#endif
					{
						get_unique_lines((intptr_t) &val[ind], numUniqueAddr);
					}
					result += val[ind] * vecTexReader(cols[ind]);
				}
#if defined(COUNT_ALL_TASKS)
				if ( s_bid == 0 )
#endif
				{
					get_unique_lines((intptr_t) &out[t], numUniqueAddr);
				}				
				out[t] = result;
			}
		}
    }
}


//*************************************************/
// Memory Allocation
//*************************************************/


// float *h_val, *h_vec, *refOut, *h_out;
// int *h_cols, *h_rowDelimiters;

// float *d_val, *d_out, *d_vec;
// int *d_cols, *d_rowDelimiters;

// int numNonZeroes, numRows;
	 
int SPMVcsr_start_kernel(void *arg)
 {
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	t_SPMV_params * params = (t_SPMV_params *)kstub->params;
	
	// numRows = params->numRows;
    // int nItems = params->nItems;
	//numRows = kstub->kconf.gridsize.x * kstub->kconf.blocksize.x * kstub->kconf.coarsening;

	//Data set 1
	//nItems = numRows * numRows * 0.000005; // 5% of entries will be non-zero

	//Data set 2
	//nItems = numRows * numRows / 14;

	float maxval = 50.0;

	// Allocate and set up host data (only for scalar csr)
	CUDA_SAFE_CALL(cudaMallocHost(&params->h_val, params->nItems * sizeof(float)));
	CUDA_SAFE_CALL(cudaMallocHost(&params->h_vec, params->numRows * sizeof(float)));
	CUDA_SAFE_CALL(cudaMallocHost(&params->h_cols, params->nItems * sizeof(int)));
	CUDA_SAFE_CALL(cudaMallocHost(&params->h_rowDelimiters, (params->numRows + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMallocHost(&params->h_out,  params->numRows * sizeof(float)));

	fill(params->h_val, params->nItems, maxval);
	initRandomMatrix_ver3(params->h_cols, params->h_rowDelimiters, params->nItems, params->numRows);
	fill(params->h_vec, params->numRows, maxval);

	// Allocate device memory
	//numNonZeroes = nItems;
	CUDA_SAFE_CALL(cudaMalloc(&params->d_val,  params->numNonZeroes * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc(&params->d_cols, params->numNonZeroes * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc(&params->d_vec,  params->numRows * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc(&params->d_out,  params->numRows * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc(&params->d_rowDelimiters, (params->numRows+1) * sizeof(int)));

	// Bind texture for position
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	CUDA_SAFE_CALL(cudaBindTexture(0, vecTex, params->d_vec, channelDesc, params->numRows * sizeof(float)));

	CUDA_SAFE_CALL(cudaMemcpy(params->d_val, params->h_val,   params->numNonZeroes * sizeof(float),
		cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(params->d_vec, params->h_vec,   params->numRows* sizeof(float),
		cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(params->d_cols, params->h_cols, params->numNonZeroes * sizeof(int),
        cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(params->d_rowDelimiters, params->h_rowDelimiters,
        (params->numRows+1) * sizeof(int), cudaMemcpyHostToDevice));
		
	return 0;
 }

int SPMVcsr_start_mallocs(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	t_SPMV_params * params = (t_SPMV_params *)kstub->params;	

	// globalmemory position for launched ctas counter
	cudaMalloc((void **)&params->zc_slc, sizeof(int));
	
	// numRows = params->numRows;
    // int nItems = params->nItems;

	//numRows = kstub->kconf.gridsize.x * kstub->kconf.blocksize.x * kstub->kconf.coarsening;
	//nItems = (int)((double)numRows * (double)(numRows) * 0.000001); // 5% of entries will be non-zero
	//Data set 1
	//nItems = numRows * numRows * 0.000005; // 5% of entries will be non-zero

	//Data set 2
	//nItems = numRows * numRows / 14;

	float maxval = 50.0;

	//printf("Items per row =%d\n", (int)((double)nItems/(double)numRows));

	// Allocate and set up host data (only for scalar csr)
	CUDA_SAFE_CALL(cudaMallocHost(&params->h_val, params->nItems * sizeof(float)));
	CUDA_SAFE_CALL(cudaMallocHost(&params->h_vec, params->numRows * sizeof(float)));
	CUDA_SAFE_CALL(cudaMallocHost(&params->h_cols, params->nItems * sizeof(int)));
	CUDA_SAFE_CALL(cudaMallocHost(&params->h_rowDelimiters, (params->numRows + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMallocHost(&params->h_out,  params->numRows * sizeof(float)));

	fill(params->h_val, params->nItems, maxval);
	initRandomMatrix_ver3(params->h_cols, params->h_rowDelimiters, params->nItems, params->numRows);
	fill(params->h_vec, params->numRows, maxval);
	
	spmvCpu(params->h_val, params->h_cols, params->h_rowDelimiters,
	     params->h_vec, params->numRows, params->h_out);

	// Allocate device memory
	// numNonZeroes = nItems;
	CUDA_SAFE_CALL(cudaMalloc(&params->d_val,  params->numNonZeroes * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc(&params->d_cols, params->numNonZeroes * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc(&params->d_vec,  params->numRows * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc(&params->d_out,  params->numRows * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc(&params->d_rowDelimiters, (params->numRows+1) * sizeof(int)));

	// Bind texture for position
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	CUDA_SAFE_CALL(cudaBindTexture(0, vecTex, params->d_vec, channelDesc, params->numRows * sizeof(float)));
	
	return 0;
}

//*************************************************/
// HtD Transfers
//*************************************************/

int SPMVcsr_start_transfers(void *arg){
	
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	t_SPMV_params * params = (t_SPMV_params *)kstub->params;

	
#if defined(MEMCPY_ASYNC)

	//enqueue_tcomamnd(tqueues, d_val, h_val, numNonZeroes * sizeof(float), cudaMemcpyHostToDevice, 
	//					kstub->transfer_s[0], NONBLOCKING, DATA, MEDIUM, kstub);
	cudaMemcpyAsync(params->d_val, params->h_val, params->numNonZeroes * sizeof(float), cudaMemcpyHostToDevice, kstub->transfer_s[0]);
						
	//enqueue_tcomamnd(tqueues, d_vec, h_vec, numRows * sizeof(float), cudaMemcpyHostToDevice, 
	//					kstub->transfer_s[0], NONBLOCKING, DATA, MEDIUM, kstub);
	cudaMemcpyAsync(params->d_vec, params->h_vec, params->numRows * sizeof(float), cudaMemcpyHostToDevice, kstub->transfer_s[0]);
	
	//enqueue_tcomamnd(tqueues, d_cols, h_cols, numNonZeroes * sizeof(float), cudaMemcpyHostToDevice, 
	//					kstub->transfer_s[0], NONBLOCKING, DATA, MEDIUM, kstub);
	cudaMemcpyAsync(params->d_cols, params->h_cols, params->numNonZeroes * sizeof(float), cudaMemcpyHostToDevice, kstub->transfer_s[0]);

	//enqueue_tcomamnd(tqueues, d_rowDelimiters, h_rowDelimiters, (numRows+1) * sizeof(int), cudaMemcpyHostToDevice, 
	//					kstub->transfer_s[0], NONBLOCKING, LAST_TRANSFER, MEDIUM, kstub);
	cudaMemcpyAsync(params->d_rowDelimiters, params->h_rowDelimiters, (params->numRows+1) * sizeof(int), cudaMemcpyHostToDevice, kstub->transfer_s[0]);


							
#else

	CUDA_SAFE_CALL(cudaMemcpy(params->d_val, params->h_val,   params->numNonZeroes * sizeof(float),
              cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(params->d_vec, params->h_vec,   params->numRows* sizeof(float),
              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(params->d_cols, params->h_cols, params->numNonZeroes * sizeof(int),
              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(params->d_rowDelimiters, params->h_rowDelimiters,
              (numRows+1) * sizeof(int), cudaMemcpyHostToDevice));
#endif

	//kstub->HtD_tranfers_finished = 1;
      
	return 0;
}

//*************************************************/
// DtH transfers and deallocation
//*************************************************/

int SPMVcsr_end_kernel(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	t_SPMV_params * params = (t_SPMV_params *)kstub->params;

#if defined(MEMCPY_ASYNC)
	//cudaEventSynchronize(kstub->end_Exec);

		printf("-->Comienzo de DtH para tarea %d\n", kstub->id);

	//enqueue_tcomamnd(tqueues, h_out, d_out, numRows * sizeof(float), cudaMemcpyDeviceToHost, kstub->transfer_s[1] , NONBLOCKING, LAST_TRANSFER, MEDIUM, kstub);
	cudaMemcpyAsync(params->h_out, params->d_out, params->numRows * sizeof(float), cudaMemcpyDeviceToHost, kstub->transfer_s[1]);
	//cudaEventRecord(kstub->end_DtH, kstub->transfer_s[1]);
	
	//kstub->DtH_tranfers_finished = 1;
	
	//	printf("-->Fin de DtH para tarea %d\n", kstub->id);



#else
	
	cudaEventSynchronize(kstub->end_Exec);
	
	CUDA_SAFE_CALL(cudaMemcpy(params->h_out, params->d_out, params->numRows * sizeof(float),
                  cudaMemcpyDeviceToHost));
				 		 
	CUDA_SAFE_CALL(cudaFree(params->d_val));
	CUDA_SAFE_CALL(cudaFree(params->d_vec));

	CUDA_SAFE_CALL(cudaFree(params->d_cols));
	CUDA_SAFE_CALL(cudaFree(params->d_rowDelimiters));
	CUDA_SAFE_CALL(cudaFree(params->d_out));
#endif
	
	// Compute results on CPU
	params->refOut = new float[params->numRows];
    spmvCpu(params->h_val, params->h_cols, params->h_rowDelimiters, params->h_vec, params->numRows, params->refOut);
	
	if (verifyResults(params->refOut, params->h_out, params->numRows) == false)
		printf("!!!! Error verifying SPMV csr\n");
	
	free(params->refOut);
	cudaFree(params->h_vec);
	cudaFree(params->h_val);
	cudaFree(params->h_cols);
	cudaFree(params->h_out);
	cudaFree(params->h_rowDelimiters);
	
	return 0;
}
	

int launch_orig_SPMVcsr(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	t_SPMV_params * params = (t_SPMV_params *)kstub->params;
	
	original_spmv_csr_scalar_kernel<<<kstub->kconf.gridsize.x, kstub->kconf.blocksize.x>>>
			(params->d_val, params->d_cols, params->d_rowDelimiters, params->numRows, params->d_out);
	
	return 0;
}

int launch_slc_SPMVcsr(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	t_SPMV_params * params = (t_SPMV_params *)kstub->params;
	
	slicing_spmv_csr_scalar_kernel<<<kstub->total_tasks, kstub->kconf.blocksize.x, 0, *(kstub->execution_s)>>>
			(params->d_val, params->d_cols, params->d_rowDelimiters, params->numRows, params->d_out, params->gridDimX, kstub->kconf.initial_blockID, params->zc_slc);
	
	return 0;
}

int prof_SPMV(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	t_SPMV_params * params = (t_SPMV_params *)kstub->params;
	
	profiling_csr_scalar_kernel<<<kstub->kconf.numSMs * kstub->kconf.max_persistent_blocks, kstub->kconf.blocksize.x, 0, *(kstub->execution_s)>>>
			(params->d_val, params->d_cols, params->d_rowDelimiters, params->numRows, params->d_out,
			
			kstub->total_tasks,
			kstub->kconf.coarsening,
			kstub->d_SMs_cont,
			kstub->d_executed_tasks,
			&kstub->gm_state[kstub->stream_index]);
			
	return 0;
}

int launch_preemp_SPMVcsr(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	t_SPMV_params * params = (t_SPMV_params *)kstub->params;
	
	#ifdef SMT
	if ( !(kstub->memaddr_profile) )	
		preemp_SMT_spmv_csr_scalar_kernel<<<kstub->kconf.numSMs * kstub->kconf.max_persistent_blocks, kstub->kconf.blocksize.x, 0, *(kstub->execution_s)>>>
			(params->d_val, params->d_cols, params->d_rowDelimiters, params->numRows, params->d_out,
			kstub->idSMs[0],
			kstub->idSMs[1],
			kstub->total_tasks, 
			kstub->kconf.coarsening,
			kstub->d_executed_tasks,
			kstub->gm_state);
	else	
		memaddr_preemp_SMT_spmv_csr_scalar_kernel<<<kstub->kconf.numSMs * kstub->kconf.max_persistent_blocks, kstub->kconf.blocksize.x, 0, *(kstub->execution_s)>>>
			(params->d_val, params->d_cols, params->d_rowDelimiters, params->numRows, params->d_out,
			kstub->d_numUniqueAddr,				
			kstub->idSMs[0],
			kstub->idSMs[1],
			kstub->total_tasks, 
			kstub->kconf.coarsening,
			kstub->d_executed_tasks,
			kstub->gm_state);
	#else
		
	preemp_SMK_spmv_csr_scalar_kernel<<<kstub->kconf.numSMs * kstub->kconf.max_persistent_blocks, kstub->kconf.blocksize.x, 0, *(kstub->execution_s)>>>
			(params->d_val, params->d_cols, params->d_rowDelimiters, params->numRows, params->d_out, 
			kstub->num_blocks_per_SM,
			kstub->total_tasks,
			kstub->kconf.coarsening,
			kstub->d_SMs_cont,
			kstub->d_executed_tasks,
			kstub->gm_state
	);	
		
	#endif
	
	return 0;
}
	 
    
			
	