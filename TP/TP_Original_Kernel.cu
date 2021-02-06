////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

// ----------------------------------------------------------------------------------------
// Transpose
//
// This file contains both device and host code for transposing a floating-point
// matrix.  It performs several transpose kernels, which incrementally improve performance
// through coalescing, removing shared memory bank conflicts, and eliminating partition
// camping.  Several of the kernels perform a copy, used to represent the best case
// performance that a transpose can achieve.
//
// Please see the whitepaper in the docs folder of the transpose project for a detailed
// description of this performance study.
// ----------------------------------------------------------------------------------------
#include "TP.h"
#include "../elastic_kernel.h"

// Each block transposes/copies a tile of TILE_DIM x TILE_DIM elements
// using TILE_DIM x BLOCK_ROWS threads, so that each thread transposes
// TILE_DIM/BLOCK_ROWS elements.  TILE_DIM must be an integral multiple of BLOCK_ROWS

// This sample assumes that MATRIX_SIZE_X = MATRIX_SIZE_Y

// Number of repetitions used for timing.  Two sets of repetitions are performed:
// 1) over kernel launches and 2) inside the kernel over just the loads and stores

// Coalesced transpose with no bank conflicts

template <int TILE_DIM, int BLOCK_ROWS>
__global__ void original_transposeNoBankConflicts(float *odata, float *idata, int width, int height, int gridDimX)
{
    int bx = blockIdx.x % gridDimX; // This is necessary because of the grid linearization
    int by = blockIdx.x / gridDimX;

    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    __shared__ float tile[TILE_DIM][TILE_DIM+1];

    int xIndex = bx * TILE_DIM + threadIdx.x;
    int yIndex = by * TILE_DIM + threadIdx.y;
    int index_in = xIndex + (yIndex)*width;

    xIndex = by * TILE_DIM + threadIdx.x;
    yIndex = bx * TILE_DIM + threadIdx.y;
    int index_out = xIndex + (yIndex)*height;

    for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
    {
        tile[threadIdx.y+i][threadIdx.x] = idata[index_in+i*width];
    }

    cg::sync(cta);

    for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
    {
        odata[index_out+i*height] = tile[threadIdx.x][threadIdx.y+i];
    }
}

// -------------------------------------------------------
// Transposes
// width and height must be integral multiples of TILE_DIM
// -------------------------------------------------------

// Coalesced transpose with no bank conflicts

template <int TILE_DIM, int BLOCK_ROWS>
__global__ void slicing_transposeNoBankConflicts(float *odata, float *idata, int width, int height, int gridDimX, int init_BlkIdx, int *zc_slc)
{
	if (threadIdx.x == 0 && threadIdx.y == 0) atomicAdd(zc_slc, 1);
    int blk_index = blockIdx.x + init_BlkIdx;
    int bx = blk_index % gridDimX; // This is necessary because of the grid linearization
    int by = blk_index / gridDimX;

    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    __shared__ float tile[TILE_DIM][TILE_DIM+1];

    int xIndex = bx * TILE_DIM + threadIdx.x;
    int yIndex = by * TILE_DIM + threadIdx.y;
    int index_in = xIndex + (yIndex)*width;

    xIndex = by * TILE_DIM + threadIdx.x;
    yIndex = bx * TILE_DIM + threadIdx.y;
    int index_out = xIndex + (yIndex)*height;

    for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
    {
        tile[threadIdx.y+i][threadIdx.x] = idata[index_in+i*width];
    }

    cg::sync(cta);

    for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
    {
        odata[index_out+i*height] = tile[threadIdx.x][threadIdx.y+i];
    }
}

// ---------------------
// host utility routines
// ---------------------

void computeTransposeGold(float *gold, float *idata,
                          const  int size_x, const  int size_y)
{
    for (int y = 0; y < size_y; ++y)
    {
        for (int x = 0; x < size_x; ++x)
        {
            gold[(x * size_y) + y] = idata[(y * size_x) + x];
        }
    }
}


void fixSize(int &size_x, int &size_y, int max_tile_dim)
{
    size_x = max_tile_dim;
    size_x = FLOOR(size_x, 512);
    size_y = max_tile_dim;
    size_y = FLOOR(size_y, 512);
}

int TP_start_mallocs(void *arg) {
    t_kernel_stub *kstub = (t_kernel_stub *)arg;
	t_TP_params *params = (t_TP_params *)kstub->params;

    // Start logs
    printf("%s Starting...\n\n", params->sSDKsample);

    cudaDeviceProp deviceProp;

    // get number of SMs on this GPU
    checkCudaErrors(cudaGetDevice(&kstub->deviceId));
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, kstub->deviceId));

    // compute the scaling factor (for GPUs with fewer MPs)
    float total_tiles;

    printf("> Device %d: \"%s\"\n", kstub->deviceId, deviceProp.name);
    printf("> SM Capability %d.%d detected:\n", deviceProp.major, deviceProp.minor);

    // Calculate number of tiles we will run for the Matrix Transpose performance tests
    int max_matrix_dim, matrix_size_test;

    matrix_size_test = 512;  // we round down max_matrix_dim for this perf test
    total_tiles = (float)params->max_tiles;

    max_matrix_dim = FLOOR((int)(floor(sqrt(total_tiles))* params->tile_dim), matrix_size_test);

    // This is the minimum size allowed
    if (max_matrix_dim == 0)
    {
        max_matrix_dim = matrix_size_test;
    }

    printf("> [%s] has %d MP(s) x %d (Cores/MP) = %d (Cores)\n",
           deviceProp.name, deviceProp.multiProcessorCount,
           _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
           _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);

    // Fijar tamaños X e Y redondeando a múltiplos de 512
    fixSize(params->size_x, params->size_y, max_matrix_dim);

    if (params->size_x != params->size_y)
    {
        printf("\n[%s] does not support non-square matrices (row_dim_size(%d) != col_dim_size(%d))\nExiting...\n\n", params->sSDKsample, params->size_x, params->size_y);
        exit(EXIT_FAILURE);
    }

    if (params->size_x%params->tile_dim != 0 || params->size_y%params->tile_dim != 0)
    {
        printf("[%s] Matrix size must be integral multiple of tile size\nExiting...\n\n", params->sSDKsample);
        exit(EXIT_FAILURE);
    }

    // size of memory required to store the matrix
    params->mem_size = static_cast<size_t>(sizeof(float) * params->size_x*params->size_y);

    if (2*params->mem_size > deviceProp.totalGlobalMem)
    {
        printf("Input matrix size is larger than the available device memory!\n");
        printf("Please choose a smaller size matrix\n");
        exit(EXIT_FAILURE);
    }

    // grid size check
    if (params->size_x/params->tile_dim < 1 || params->size_y/params->tile_dim < 1)
    {
        printf("[%s] grid size computation incorrect in test \nExiting...\n\n", params->sSDKsample);
        exit(EXIT_FAILURE);
    }

    // allocate host memory
    params->h_idata = (float *) malloc(params->mem_size);
    params->h_odata = (float *) malloc(params->mem_size);
    params->transposeGold = (float *) malloc(params->mem_size);

    // allocate device memory
    checkCudaErrors(cudaMalloc((void **) &params->d_idata, params->mem_size));
    checkCudaErrors(cudaMalloc((void **) &params->d_odata, params->mem_size));
	// commom cta counter
	cudaMalloc((void **)&params->zc_slc, sizeof(int));

    // initialize host data
    for (int i = 0; i < (params->size_x*params->size_y); ++i)
    {
        params->h_idata[i] = (float) i;
    }

    // Compute reference transpose solution
    computeTransposeGold(params->transposeGold, params->h_idata, params->size_x, params->size_y);
    params->gold = params->transposeGold;

    // print out common data for all kernels
    printf("\nMatrix size: %dx%d (%dx%d tiles), tile size: %dx%d, block size: %dx%d\n\n",
           params->size_x, params->size_y, params->size_x/params->tile_dim, params->size_y/params->tile_dim, params->tile_dim, params->tile_dim, params->tile_dim, params->block_rows);
    
    params->success = true;

    return 0;
}

int TP_start_transfers(void *arg) {
    t_kernel_stub *kstub = (t_kernel_stub *)arg;
	t_TP_params *params = (t_TP_params *)kstub->params;

    // copy host data to device
    checkCudaErrors(cudaMemcpy(params->d_idata, params->h_idata, params->mem_size, cudaMemcpyHostToDevice));
    return 0;
}

int TP_end_kernel(void *arg) {
    t_kernel_stub *kstub = (t_kernel_stub *)arg;
	t_TP_params *params = (t_TP_params *)kstub->params;

    checkCudaErrors(cudaMemcpy(params->h_odata, params->d_odata, params->mem_size, cudaMemcpyDeviceToHost));
    bool res = compareData(params->gold, params->h_odata, params->size_x*params->size_y, 0.01f, 0.0f);

    if (res == false)
    {
        printf("*** kernel FAILED ***\n");
        params->success = false;
    }

    // take measurements for loop inside kernel
    checkCudaErrors(cudaMemcpy(params->h_odata, params->d_odata, params->mem_size, cudaMemcpyDeviceToHost));
    res = compareData(params->gold, params->h_odata, params->size_x*params->size_y, 0.01f, 0.0f);

    if (res == false)
    {
        printf("*** kernel FAILED ***\n");
        params->success = false;
    }

    // cleanup
    free(params->h_idata);
    free(params->h_odata);
    free(params->transposeGold);
    cudaFree(params->d_idata);
    cudaFree(params->d_odata);

    if (!params->success)
    {
        printf("Test failed!\n");
    }

    printf("Test passed\n");
    return 0;
}

// ----
// main
// ----
int
launch_orig_TP(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	t_TP_params * params = (t_TP_params *)kstub->params;

    const int tile_dim = params->tile_dim;
    const int block_rows = params->block_rows;

    kstub->kconf.gridsize.x = params->size_x/params->tile_dim; 
    kstub->kconf.gridsize.y = params->size_y/params->tile_dim; 
    kstub->total_tasks = kstub->kconf.gridsize.x*kstub->kconf.gridsize.y; 
    dim3 threads(params->tile_dim,params->block_rows);

    // Clear error status
    checkCudaErrors(cudaGetLastError());

    // warmup to avoid timing startup
    // original_transposeNoBankConflicts<<<grid, threads>>>(params->d_odata, params->d_idata, params->size_x, params->size_y);

    if (params->tile_dim == 16 && params->block_rows == 16) {
        original_transposeNoBankConflicts<16,16><<<kstub->total_tasks, threads>>>(params->d_odata, params->d_idata, params->size_x, params->size_y, kstub->kconf.gridsize.x);
    }
    else {
        printf("TILE_DIM or BLOCK_ROWS not properly defined");
        exit(EXIT_FAILURE);
    }

    // Ensure no launch failure
    checkCudaErrors(cudaGetLastError());
    exit(EXIT_SUCCESS);
}

int
launch_slc_TP(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	t_TP_params * params = (t_TP_params *)kstub->params;

    kstub->kconf.gridsize.x = params->size_x/params->tile_dim; 
    kstub->kconf.gridsize.y = params->size_y/params->tile_dim; 
    kstub->total_tasks = kstub->kconf.gridsize.x*kstub->kconf.gridsize.y; 
    dim3 threads(params->tile_dim,params->block_rows);

    // Clear error status
    checkCudaErrors(cudaGetLastError());

    // warmup to avoid timing startup
    // original_transposeNoBankConflicts<<<grid, threads>>>(params->d_odata, params->d_idata, params->size_x, params->size_y);

    if (params->tile_dim == 16 && params->block_rows == 16) {
        slicing_transposeNoBankConflicts<16,16><<<kstub->total_tasks, threads>>>(params->d_odata, params->d_idata, params->size_x, params->size_y, kstub->kconf.gridsize.x, kstub->kconf.initial_blockID, params->zc_slc);
    }
    else {
        printf("TILE_DIM or BLOCK_ROWS not properly defined");
        exit(EXIT_FAILURE);
    }

    // Ensure no launch failure
    checkCudaErrors(cudaGetLastError());
    exit(EXIT_SUCCESS);
}