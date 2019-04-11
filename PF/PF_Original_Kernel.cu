// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>

// Files
#include <sstream>
#include <string>
#include <fstream>
#include <iostream>

// Vectors
#include <vector>

using namespace std;

#include "../elastic_kernel.h"
#include "PF.h"

#define HALO 1 // halo width along one direction when advancing to the next iteration

int rows, cols;
int* data;
int** wall;
int* result;
#define M_SEED 9
int pyramid_height;

int final_ret;
int borderCols;
int smallBlockCol;
int blockCols;
int *gpuWall, *gpuResult[2];
#define BLOCK_SIZE 256

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )

extern t_tqueue *tqueues;

 __device__ uint get_smid_PF(void) {
	uint ret;

	asm("mov.u32 %0, %smid;" : "=r"(ret) );

	return ret;
}


/**
 * Path Finder (CUDA Kernel)
 */
__global__ void
original_pathFinderCUDA(int pyramid_heightPF, int *gpuWall, int *gpuSrc, int *gpuResults, int cols,  int rows, int startStep, int border)
{
    for(startStep = 0; startStep < rows - 1; startStep += pyramid_heightPF){
	
		int iteration = MIN(pyramid_heightPF, rows-startStep-1);
		
		__shared__ int prev[BLOCK_SIZE];
		__shared__ int result[BLOCK_SIZE];

		int bx = blockIdx.x;
		int tx = threadIdx.x;

		// each block finally computes result for a small block
		// after N iterations. 
		// it is the non-overlapping small blocks that cover 
		// all the input data

		// calculate the small block size
		int small_block_cols = BLOCK_SIZE-iteration*HALO*2;

		// calculate the boundary for the block according to 
		// the boundary of its small block
		int blkX = small_block_cols*bx-border;
		int blkXmax = blkX+BLOCK_SIZE-1;

		// calculate the global thread coordination
		int xidx = blkX+tx;

		// effective range within this block that falls within 
		// the valid range of the input data
		// used to rule out computation outside the boundary.
		int validXmin = (blkX < 0) ? -blkX : 0;
		int validXmax = (blkXmax > cols-1) ? BLOCK_SIZE-1-(blkXmax-cols+1) : BLOCK_SIZE-1;

		int W = tx-1;
		int E = tx+1;

		W = (W < validXmin) ? validXmin : W;
		E = (E > validXmax) ? validXmax : E;

		bool isValid = IN_RANGE(tx, validXmin, validXmax);

		if(IN_RANGE(xidx, 0, cols-1)){
			prev[tx] = gpuSrc[xidx];
		}
		__syncthreads(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
		bool computed;
		for (int i=0; i<iteration ; i++){ 
			computed = false;
			if( IN_RANGE(tx, i+1, BLOCK_SIZE-i-2) &&  \
			isValid){
				computed = true;
				int left = prev[W];
				int up = prev[tx];
				int right = prev[E];
				int shortest = MIN(left, up);
				shortest = MIN(shortest, right);
				int index = cols*(startStep+i)+xidx;
				result[tx] = shortest + gpuWall[index];
			}
			__syncthreads();
			if(i==iteration-1)
				break;
			if(computed)   //Assign the computation range
				prev[tx]= result[tx];
			__syncthreads(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
		}

		// update the global memory
		// after the last iteration, only threads coordinated within the 
		// small block perform the calculation and switch on ``computed''
		if (computed){
			gpuResults[xidx]=result[tx];    
		}
	}
}

__global__ void
SMT_pathFinderCUDA(int pyramid_heightPF, int *gpuWall, int *gpuSrc, int *gpuResults, int cols,  int rows, int startStep, int border,
					int SIMD_min, int SIMD_max,
					int num_subtask, int iter_per_subtask, int *cont_subtask, State *status)
{
	
	__shared__ int s_bid;
	
	unsigned int SM_id = get_smid_PF();
	
	if (SM_id <SIMD_min || SM_id > SIMD_max) /* Only blocks executing within SIMD_min and SIMD_max can progress */ 
			return;
	
	while (1){
		
		/********** Task Id calculation *************/
		
		if (threadIdx.x == 0) { 
			if (*status == TOEVICT)
				s_bid = -1;
			else {
				s_bid = atomicAdd(cont_subtask, 1);				//subtask_id
				//printf("Blq=%d cont=%d\n", blockIdx.x, s_bid);
			}
		}
		
		__syncthreads();
		
		//if (s_bid[warpid] >= num_subtask || s_bid[warpid] == -1)
		if (s_bid >=num_subtask || s_bid ==-1) /* If all subtasks have been executed */{
			//if (threadIdx.x == 0)  printf("El bloque %d se sale con %d\n", blockIdx.x, s_bid); 
			return;
		}

		for(startStep = 0; startStep < rows - 1; startStep += pyramid_heightPF){	
			int iteration = MIN(pyramid_heightPF, rows-startStep-1);
			
			__shared__ int prev[BLOCK_SIZE];
			__shared__ int result[BLOCK_SIZE];

			//int bx = blockIdx.x;
			int bx = s_bid;
			int tx = threadIdx.x;

			// each block finally computes result for a small block
			// after N iterations. 
			// it is the non-overlapping small blocks that cover 
			// all the input data

			// calculate the small block size
			int small_block_cols = BLOCK_SIZE-iteration*HALO*2;

			// calculate the boundary for the block according to 
			// the boundary of its small block
			int blkX = small_block_cols*bx-border;
			int blkXmax = blkX+BLOCK_SIZE-1;

			// calculate the global thread coordination
			int xidx = blkX+tx;

			// effective range within this block that falls within 
			// the valid range of the input data
			// used to rule out computation outside the boundary.
			int validXmin = (blkX < 0) ? -blkX : 0;
			int validXmax = (blkXmax > cols-1) ? BLOCK_SIZE-1-(blkXmax-cols+1) : BLOCK_SIZE-1;

			int W = tx-1;
			int E = tx+1;

			W = (W < validXmin) ? validXmin : W;
			E = (E > validXmax) ? validXmax : E;

			bool isValid = IN_RANGE(tx, validXmin, validXmax);

			if(IN_RANGE(xidx, 0, cols-1)){
				prev[tx] = gpuSrc[xidx];
			}
			__syncthreads(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
			bool computed;
			for (int i=0; i<iteration ; i++){ 
				computed = false;
				if( IN_RANGE(tx, i+1, BLOCK_SIZE-i-2) &&  \
				isValid){
					computed = true;
					int left = prev[W];
					int up = prev[tx];
					int right = prev[E];
					int shortest = MIN(left, up);
					shortest = MIN(shortest, right);
					int index = cols*(startStep+i)+xidx;
					result[tx] = shortest + gpuWall[index];
				}
				__syncthreads();
				if(i==iteration-1)
					break;
				if(computed)   //Assign the computation range
					prev[tx]= result[tx];
				__syncthreads(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
			}

			// update the global memory
			// after the last iteration, only threads coordinated within the 
			// small block perform the calculation and switch on ``computed''
			if (computed){
				gpuResults[xidx]=result[tx];    
			}
		}
	}
}

__global__ void
SMK_pathFinderCUDA(int pyramid_heightPF, int *gpuWall, int *gpuSrc, int *gpuResults, int cols,  int rows, int startStep, int border,
					int max_blocks_per_SM,
					int num_subtask,
					int iter_per_subtask,
					int *cont_SM,
					int *cont_subtask,
					State *status
)
{
	__shared__ int s_bid, s_index;
	
	unsigned int SM_id = get_smid_PF();
	
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
		
		for(startStep = 0; startStep < rows - 1; startStep += pyramid_heightPF){	
			int iteration = MIN(pyramid_heightPF, rows-startStep-1);
			
			__shared__ int prev[BLOCK_SIZE];
			__shared__ int result[BLOCK_SIZE];

			int bx = s_bid;
			int tx = threadIdx.x;

			// each block finally computes result for a small block
			// after N iterations. 
			// it is the non-overlapping small blocks that cover 
			// all the input data

			// calculate the small block size
			int small_block_cols = BLOCK_SIZE-iteration*HALO*2;

			// calculate the boundary for the block according to 
			// the boundary of its small block
			int blkX = small_block_cols*bx-border;
			int blkXmax = blkX+BLOCK_SIZE-1;

			// calculate the global thread coordination
			int xidx = blkX+tx;

			// effective range within this block that falls within 
			// the valid range of the input data
			// used to rule out computation outside the boundary.
			int validXmin = (blkX < 0) ? -blkX : 0;
			int validXmax = (blkXmax > cols-1) ? BLOCK_SIZE-1-(blkXmax-cols+1) : BLOCK_SIZE-1;

			int W = tx-1;
			int E = tx+1;

			W = (W < validXmin) ? validXmin : W;
			E = (E > validXmax) ? validXmax : E;

			bool isValid = IN_RANGE(tx, validXmin, validXmax);

			if(IN_RANGE(xidx, 0, cols-1)){
				prev[tx] = gpuSrc[xidx];
			}
			__syncthreads(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
			bool computed;
			for (int i=0; i<iteration ; i++){ 
				computed = false;
				if( IN_RANGE(tx, i+1, BLOCK_SIZE-i-2) &&  \
				isValid){
					computed = true;
					int left = prev[W];
					int up = prev[tx];
					int right = prev[E];
					int shortest = MIN(left, up);
					shortest = MIN(shortest, right);
					int index = cols*(startStep+i)+xidx;
					result[tx] = shortest + gpuWall[index];
				}
				__syncthreads();
				if(i==iteration-1)
					break;
				if(computed)   //Assign the computation range
					prev[tx]= result[tx];
				__syncthreads(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
			}

			// update the global memory
			// after the last iteration, only threads coordinated within the 
			// small block perform the calculation and switch on ``computed''
			if (computed){
				gpuResults[xidx]=result[tx];    
			}
		}
	}
}

void
init()
{
	data = new int[rows*cols];
	wall = new int*[rows];
	for(int n=0; n<rows; n++)
		wall[n]=data+cols*n;
	result = new int[cols];
	
	int seed = M_SEED;
	srand(seed);

	for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            wall[i][j] = rand() % 10;
        }
    }

	// printf("WALL\n");
    // for (int i = 0; i < rows; i++)
    // {
        // for (int j = 0; j < cols; j++)
        // {
            // printf("%d ",wall[i][j]) ;
        // }
        // printf("\n") ;
    // }

}

int PF_start_kernel(void *arg) 
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	int blocksize = kstub->kconf.blocksize.x;
	
	t_PF_params * params = (t_PF_params *)kstub->params;
	
	cols = params->nCols;
	rows = params->nRows;
	pyramid_height = params->param_pyramid_height;
	
	data = new int[rows*cols];
	wall = new int*[rows];
	result = new int[cols];
	
	init();

	/* --------------- pyramid parameters --------------- */
    borderCols = (pyramid_height)*HALO;
    smallBlockCol = blocksize-(pyramid_height)*HALO*2;
    blockCols = cols/smallBlockCol+((cols%smallBlockCol==0)?0:1);

    // printf("pyramidHeight: %d\ngridSize: [%d]\nborder:[%d]\nblockSize: %d\nblockGrid:[%d]\ntargetBlock:[%d]\n",\
	// pyramid_height, cols, borderCols, blocksize, blockCols, smallBlockCol);
	
    int size = rows*cols;

    cudaMalloc((void**)&gpuResult[0], sizeof(int)*cols);
    cudaMalloc((void**)&gpuResult[1], sizeof(int)*cols);
    cudaMemcpy(gpuResult[0], data, sizeof(int)*cols, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&gpuWall, sizeof(int)*(size-cols));
    cudaMemcpy(gpuWall, data+cols, sizeof(int)*(size-cols), cudaMemcpyHostToDevice);
	
	return 0;
}

int PF_start_mallocs(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	int blocksize = kstub->kconf.blocksize.x;
	
	t_PF_params * params = (t_PF_params *)kstub->params;
	
	cols = params->nCols;
	rows = params->nRows;
	pyramid_height = params->param_pyramid_height;

	/* --------------- pyramid parameters --------------- */
    borderCols = (pyramid_height)*HALO;
    smallBlockCol = blocksize-(pyramid_height)*HALO*2;
    blockCols = cols/smallBlockCol+((cols%smallBlockCol==0)?0:1);

    // printf("pyramidHeight: %d\ngridSize: [%d]\nborder:[%d]\nblockSize: %d\nblockGrid:[%d]\ntargetBlock:[%d]\n",\
	// pyramid_height, cols, borderCols, blocksize, blockCols, smallBlockCol);
	
    int size = rows*cols;
	
#if defined(MEMCPY_SYNC) || defined(MEMCPY_ASYNC)
	cudaMallocHost(&data, sizeof(int)*(rows*cols));
	cudaMallocHost(&wall, sizeof(int)*rows);
	cudaMallocHost(&result, sizeof(int)*cols);

	cudaMalloc((void**)&gpuResult[0], sizeof(int)*cols);
    cudaMalloc((void**)&gpuResult[1], sizeof(int)*cols);
	cudaMalloc((void**)&gpuWall, sizeof(int)*(size-cols));
#else
	#ifdef MANAGED_MEM

	cudaMallocManaged(&data, sizeof(int)*(rows, cols));
	cudaMallocManaged(&wall, sizeof(int)*rows);
	cudaMallocManaged(&result, sizeof(int)*cols);
	
	gpuResult[0] = data;
	gpuWall = data+cols;
	#else
		printf("No transfer model: Exiting ...\n");
		exit(-1);
	#endif
#endif

	// Verify that allocations succeeded
    if (data == NULL || wall == NULL || result == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    init();

	return 0;
}

int PF_start_transfers(void *arg)
{
	cudaError_t err = cudaSuccess;
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	
	int size = rows*cols;
	
#ifdef MEMCPY_SYNC
	enqueue_tcomamnd(tqueues, gpuResult[0], data, sizeof(int)*cols, cudaMemcpyHostToDevice, 0, BLOCKING, DATA, LOW, kstub);

	enqueue_tcomamnd(tqueues, gpuWall, data+cols, sizeof(int)*(size-cols), cudaMemcpyHostToDevice, 0, BLOCKING, DATA, LOW, kstub);
	
	kstub->HtD_tranfers_finished = 1;

	
#else
	
	#ifdef MEMCPY_ASYNC
	
	//enqueue_tcomamnd(tqueues, gpuResult[0], data, sizeof(int)*cols, cudaMemcpyHostToDevice, 0, NONBLOCKING, DATA, MEDIUM, kstub);
	//enqueue_tcomamnd(tqueues, gpuWall, data+cols, sizeof(int)*(size-cols), cudaMemcpyHostToDevice, 0, NONBLOCKING, LAST_TRANSFER, MEDIUM, kstub);

	 err = cudaMemcpyAsync(gpuResult[0], data, sizeof(int)*cols, cudaMemcpyHostToDevice, kstub->transfer_s[0]);
	 err = cudaMemcpyAsync(gpuWall, data+cols, sizeof(int)*(size-cols), cudaMemcpyHostToDevice, kstub->transfer_s[0]);
	
	#else
	#ifdef MANAGED_MEM

	cudaDeviceProp p;
    cudaGetDeviceProperties(&p, kstub->deviceId);
	
	if (p.concurrentManagedAccess)
	{
		err = cudaMemPrefetchAsync(data, sizeof(int)*(rows, cols), kstub->deviceId);
		if ( err != cudaSuccess) {
			printf("Error in vAdd:cudaMemPrefetchAsync\n");
			exit(EXIT_FAILURE);
		}
		err = cudaMemPrefetchAsync(wall, sizeof(int)*rows, kstub->deviceId);
		if ( err != cudaSuccess) {
			printf("Error in vAdd:cudaMemPrefetchAsync\n");
			exit(EXIT_FAILURE);
		}
		err = cudaMemPrefetchAsync(result, sizeof(int)*cols, kstub->deviceId);
		if ( err != cudaSuccess) {
			printf("Error in vAdd:cudaMemPrefetchAsync\n");
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

// int PF_end_kernel_dummy(void *arg)
// {
	// cudaMemcpy(result, gpuResult[final_ret], sizeof(int)*cols, cudaMemcpyDeviceToHost);
	
	
	// cudaFree(gpuWall);
    // cudaFree(gpuResult[0]);
    // cudaFree(gpuResult[1]);

    // cudaFreeHost(data);
	// cudaFreeHost(wall);
	// cudaFreeHost(result);

    // return 0;
// }

int PF_end_kernel(void *arg)
{
	
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	
#ifdef MEMCPY_SYNC

	cudaEventSynchronize(kstub->end_Exec);

	enqueue_tcomamnd(tqueues, result, gpuResult[final_ret], sizeof(int)*cols, cudaMemcpyDeviceToHost, 0, BLOCKING, DATA, LOW, kstub);
	 
#else
	#ifdef MEMCPY_ASYNC
	printf("-->Comienzo de DtH para tarea %d\n", kstub->id);

	//enqueue_tcomamnd(tqueues, result, gpuResult[final_ret], sizeof(int)*cols, cudaMemcpyDeviceToHost, kstub->transfer_s[1] , NONBLOCKING, LAST_TRANSFER, MEDIUM, kstub);
	
	cudaMemcpyAsync(result, gpuResult[final_ret], sizeof(int)*cols, cudaMemcpyDeviceToHost, kstub->transfer_s[1]);
	#else
		#ifdef MANAGED_MEM
			cudaStreamSynchronize(*(kstub->execution_s)); // To be sure kernel execution has finished before processing output data
		#endif
	#endif
#endif

	return 0;
}	
 
int launch_orig_PF(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	
	// Setup execution parameters
    //dim3 threads(kstub->kconf.blocksize.x, kstub->kconf.blocksize.y);
	//dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(blockCols);
	
	int src = 1, dst = 0, t = 0;
	
	int temp = src;
	src = dst;
	dst = temp;
	
	original_pathFinderCUDA<<<kstub->kconf.gridsize.x, kstub->kconf.blocksize.x>>>(
		pyramid_height, 
		gpuWall, gpuResult[src], gpuResult[dst],
		cols,rows, t, borderCols);
		
	// for the measurement fairness
	//cudaDeviceSynchronize();
	
	final_ret = dst;

	return 0;
}

int launch_preemp_PF(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	
	// Setup execution parameters
    //dim3 threads(kstub->kconf.blocksize.x, kstub->kconf.blocksize.y);
	//dim3 dimBlock(BLOCK_SIZE);
	//dim3 dimGrid(blockCols);
	
    int src = 1, dst = 0, t = 0;
	
	int temp = src;
	src = dst;
	dst = temp;
	
	#ifdef SMT
		SMT_pathFinderCUDA<<< kstub->kconf.numSMs * kstub->kconf.max_persistent_blocks, kstub->kconf.blocksize.x, 0, *(kstub->execution_s) >>>(
			pyramid_height, 
			gpuWall, gpuResult[src], gpuResult[dst],
			cols,rows, t, borderCols,
			
			kstub->idSMs[0],
			kstub->idSMs[1],
			kstub->total_tasks,
			kstub->kconf.coarsening,
			kstub->d_executed_tasks,
			&(kstub->gm_state[kstub->stream_index]));
	#else
		SMK_pathFinderCUDA<<< kstub->kconf.numSMs * kstub->kconf.max_persistent_blocks, kstub->kconf.blocksize.x, 0, *(kstub->execution_s) >>>(
			pyramid_height, 
			gpuWall, gpuResult[src], gpuResult[dst],
			cols,rows, t, borderCols,
			
			kstub->num_blocks_per_SM,
			kstub->total_tasks,
			kstub->kconf.coarsening,
			kstub->d_SMs_cont,
			kstub->d_executed_tasks,
			&(kstub->gm_state[kstub->stream_index]));
	#endif

	// for the measurement fairness
	//cudaDeviceSynchronize();
	
	final_ret = dst;
	
	return 0;
}
 



 