#ifndef DUMMY_KERNEL_H_
#define DUMMY_KERNEL_H_

#include <cuda.h>

// A dummy kernel: original version
__global__ void
dummy_kernel(float* idata, float* odata, const unsigned long int n, const int iter_per_subtask)
{

	unsigned long int bid = blockIdx.x + blockIdx.y * gridDim.x; // Block ID
    unsigned long int tid = threadIdx.y * blockDim.x + threadIdx.x; // Thread ID in block
	unsigned long int gtid = bid * (blockDim.x * blockDim.y) + tid; // Thread ID in grid

	if ( gtid < n )
	{
		float factor = 1;
		for ( int i = 0; i < iter_per_subtask; i++ )
			factor *= idata[gtid];
		odata[gtid] = factor;
	}

}

// Auxiliary function to get SM id

__device__ uint get_smid_dummy(void) {

     uint ret;

     asm("mov.u32 %0, %smid;" : "=r"(ret) );

     return ret;

}

// A dummy kernel: SMT version
__global__ void
preemp_SMT_dummy_kernel(const float *idata, float *odata, const unsigned long int n, 
		int SIMD_min, // Min and max id of SM that execute tasks
		int SIMD_max,
		unsigned long int num_subtask, // Total number of subtasks
		const int iter_per_subtask, // A coarsening factor
		int *cont_subtask, // Current number of executed subtasks
		State *status // Status of kernel: RUNNING or TOEVICT
		)
{
	__shared__ int s_bid;  // Shared counter of executed subtasks

	const unsigned int SM_id = get_smid_dummy();
	const unsigned int tid = threadIdx.y*blockDim.x + threadIdx.x;
	
	if (SM_id <SIMD_min || SM_id > SIMD_max) /* Only blocks executing within SIMD_min and SIMD_max can progress */ 
			return;
		
	while (1)
	{
		/********** Task Id calculation *************/
		if (tid == 0) {
			if (*status == TOEVICT)
				s_bid = -1;
			else
				s_bid = atomicAdd(cont_subtask, 1);
		}
		
		__syncthreads();
		
		if (s_bid >= num_subtask || s_bid == -1){ /* If all subtasks have been executed */
			return;
		}
		
		const unsigned int gtid = s_bid * (blockDim.x * blockDim.y) + tid; // Thread ID in grid
		if ( gtid < n )
		{
			odata[gtid] = 1;
			for (int iter=0; iter<iter_per_subtask; iter++)
				odata[gtid] *= idata[gtid];
		}
	}
}

// A dummy kernel: SMK version
__global__ void
preemp_SMK_dummy_kernel(const float *idata, float *odata, const unsigned long int n, 
		int max_blocks_per_SM, // Maximum number of allowed blocks in a SM
		unsigned long int num_subtask, // Total number of subtasks
		const int iter_per_subtask, // A coarsening factor
		int *cont_SM,
		int *cont_subtask, // Current number of executed subtasks
		State *status // Status of kernel: RUNNING or TOEVICT
		)
{
	__shared__ int s_bid, s_index;  // Shared counter of executed subtasks and number of blocks per SM

	const unsigned int SM_id = get_smid_dummy();
	const unsigned int tid = threadIdx.y*blockDim.x + threadIdx.x;
	
	if ( tid == 0)  
		s_index = atomicAdd(&cont_SM[SM_id],1);

	__syncthreads();

	if (s_index > max_blocks_per_SM)
		return;

	while (1)
	{
		/********** Task Id calculation *************/
		if (tid == 0) {
			if (*status == TOEVICT)
				s_bid = -1;
			else
				s_bid = atomicAdd(cont_subtask, 1);
		}
		
		__syncthreads();
		
		if (s_bid >= num_subtask || s_bid == -1){ /* If all subtasks have been executed */
			return;
		}
		
		const unsigned int gtid = s_bid * (blockDim.x * blockDim.y) + tid; // Thread ID in grid
		if ( gtid < n )
		{
			odata[gtid] = 1;
			for (int iter=0; iter<iter_per_subtask; iter++)
				odata[gtid] *= idata[gtid];
		}
	}
}

#endif // DUMMY_KERNEL_H_
