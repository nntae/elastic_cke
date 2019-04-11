/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include "../cudacommon.h"
#include "FDTD3dGPU.h"

#include <iostream>
#include <algorithm>
#include <helper_functions.h>
#include <helper_cuda.h>
#include "../elastic_kernel.h"

#include "FDTD3dGPUKernel.cuh"
extern t_tqueue *tqueues;

bool getTargetDeviceGlobalMemSize(memsize_t *result, const int argc, const char **argv)
{
    int               deviceCount  = 0;
    int               targetDevice = 0;
    size_t            memsize      = 0;

    // Get the number of CUDA enabled GPU devices
    printf(" cudaGetDeviceCount\n");
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));

    // Select target device (device 0 by default)
    targetDevice = findCudaDevice(argc, (const char **)argv);

    // Query target device for maximum memory allocation
    printf(" cudaGetDeviceProperties\n");
    struct cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, targetDevice));

    memsize = deviceProp.totalGlobalMem;

    // Save the result
    *result = (memsize_t)memsize;
    return true;
}

// Original routine
bool fdtdGPU(float *output, const float *input, const float *coeff, const int dimx, const int dimy, const int dimz, const int radius, const int timesteps, const int argc, const char **argv)
{
    const int         outerDimx  = dimx + 2 * radius;
    const int         outerDimy  = dimy + 2 * radius;
    const int         outerDimz  = dimz + 2 * radius;
    const size_t      volumeSize = outerDimx * outerDimy * outerDimz;
    int               deviceCount  = 0;
    int               targetDevice = 0;
    float            *bufferOut    = 0;
    float            *bufferIn     = 0;
    dim3              dimBlock;
    dim3              dimGrid;

    // Ensure that the inner data starts on a 128B boundary
    const int padding = (128 / sizeof(float)) - radius;
    const size_t paddedVolumeSize = volumeSize + padding;

#ifdef GPU_PROFILING
    cudaEvent_t profileStart = 0;
    cudaEvent_t profileEnd   = 0;
    const int profileTimesteps = timesteps - 1;

    if (profileTimesteps < 1)
    {
        printf(" cannot profile with fewer than two timesteps (timesteps=%d), profiling is disabled.\n", timesteps);
    }

#endif

    // Check the radius is valid
    if (radius != RADIUS)
    {
        printf("radius is invalid, must be %d - see kernel for details.\n", RADIUS);
        exit(EXIT_FAILURE);
    }

    // Get the number of CUDA enabled GPU devices
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));

    // Select target device (device 0 by default)
    targetDevice = findCudaDevice(argc, (const char **)argv);

    checkCudaErrors(cudaSetDevice(targetDevice));

    // Allocate memory buffers
    checkCudaErrors(cudaMalloc((void **)&bufferOut, paddedVolumeSize * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&bufferIn, paddedVolumeSize * sizeof(float)));

    // Check for a command-line specified block size
    int userBlockSize;

    if (checkCmdLineFlag(argc, (const char **)argv, "block-size"))
    {
        userBlockSize = getCmdLineArgumentInt(argc, argv, "block-size");
        // Constrain to a multiple of k_blockDimX
        userBlockSize = (userBlockSize / k_blockDimX * k_blockDimX);

        // Constrain within allowed bounds
        userBlockSize = MIN(MAX(userBlockSize, k_blockSizeMin), k_blockSizeMax);
    }
    else
    {
        userBlockSize = k_blockSizeMax;
    }

    // Check the device limit on the number of threads
    struct cudaFuncAttributes funcAttrib;
    checkCudaErrors(cudaFuncGetAttributes(&funcAttrib, FiniteDifferencesKernel));

    userBlockSize = MIN(userBlockSize, funcAttrib.maxThreadsPerBlock);

    // Set the block size
    dimBlock.x = k_blockDimX;
    // Visual Studio 2005 does not like std::min
    //    dimBlock.y = std::min<size_t>(userBlockSize / k_blockDimX, (size_t)k_blockDimMaxY);
    dimBlock.y = ((userBlockSize / k_blockDimX) < (size_t)k_blockDimMaxY) ? (userBlockSize / k_blockDimX) : (size_t)k_blockDimMaxY;
    dimGrid.x  = (unsigned int)ceil((float)dimx / dimBlock.x);
    dimGrid.y  = (unsigned int)ceil((float)dimy / dimBlock.y);
    printf(" set block size to %dx%d\n", dimBlock.x, dimBlock.y);
    printf(" set grid size to %dx%d\n", dimGrid.x, dimGrid.y);

    // Check the block size is valid
    if (dimBlock.x < RADIUS || dimBlock.y < RADIUS)
    {
        printf("invalid block size, x (%d) and y (%d) must be >= radius (%d).\n", dimBlock.x, dimBlock.y, RADIUS);
        exit(EXIT_FAILURE);
    }

    // Copy the input to the device input buffer
    checkCudaErrors(cudaMemcpy(bufferIn + padding, input, volumeSize * sizeof(float), cudaMemcpyHostToDevice));

    // Copy the input to the device output buffer (actually only need the halo)
    checkCudaErrors(cudaMemcpy(bufferOut + padding, input, volumeSize * sizeof(float), cudaMemcpyHostToDevice));

    // Copy the coefficients to the device coefficient buffer
    checkCudaErrors(cudaMemcpyToSymbol(stencil, (void *)coeff, (radius + 1) * sizeof(float)));


#ifdef GPU_PROFILING

    // Create the events
    checkCudaErrors(cudaEventCreate(&profileStart));
    checkCudaErrors(cudaEventCreate(&profileEnd));

#endif

    // Execute the FDTD
    float *bufferSrc = bufferIn + padding;
    float *bufferDst = bufferOut + padding;
    printf(" GPU FDTD loop\n");


#ifdef GPU_PROFILING
    // Enqueue start event
    checkCudaErrors(cudaEventRecord(profileStart, 0));
#endif

    for (int it = 0 ; it < timesteps ; it++)
    {
        printf("\tt = %d ", it);

        // Launch the kernel
        printf("launch kernel\n");
        FiniteDifferencesKernel<<<dimGrid, dimBlock>>>(bufferDst, bufferSrc, dimx, dimy, dimz);

        // Toggle the buffers
        // Visual Studio 2005 does not like std::swap
        //    std::swap<float *>(bufferSrc, bufferDst);
        float *tmp = bufferDst;
        bufferDst = bufferSrc;
        bufferSrc = tmp;
    }

    printf("\n");

#ifdef GPU_PROFILING
    // Enqueue end event
    checkCudaErrors(cudaEventRecord(profileEnd, 0));
#endif

    // Wait for the kernel to complete
    checkCudaErrors(cudaDeviceSynchronize());

    // Read the result back, result is in bufferSrc (after final toggle)
    checkCudaErrors(cudaMemcpy(output, bufferSrc, volumeSize * sizeof(float), cudaMemcpyDeviceToHost));

    // Report time
#ifdef GPU_PROFILING
    float elapsedTimeMS = 0;

    if (profileTimesteps > 0)
    {
        checkCudaErrors(cudaEventElapsedTime(&elapsedTimeMS, profileStart, profileEnd));
    }

    if (profileTimesteps > 0)
    {
        // Convert milliseconds to seconds
        double elapsedTime    = elapsedTimeMS * 1.0e-3;
        double avgElapsedTime = elapsedTime / (double)profileTimesteps;
        // Determine number of computations per timestep
        size_t pointsComputed = dimx * dimy * dimz;
        // Determine throughput
        double throughputM    = 1.0e-6 * (double)pointsComputed / avgElapsedTime;
        printf("FDTD3d, Throughput = %.4f MPoints/s, Time = %.5f s, Size = %u Points, NumDevsUsed = %u, Blocksize = %u\n",
               throughputM, avgElapsedTime, pointsComputed, 1, dimBlock.x * dimBlock.y);
    }

#endif

    // Cleanup
    if (bufferIn)
    {
        checkCudaErrors(cudaFree(bufferIn));
    }

    if (bufferOut)
    {
        checkCudaErrors(cudaFree(bufferOut));
    }

#ifdef GPU_PROFILING

    if (profileStart)
    {
        checkCudaErrors(cudaEventDestroy(profileStart));
    }

    if (profileEnd)
    {
        checkCudaErrors(cudaEventDestroy(profileEnd));
    }

#endif
    return true;
}

// Routines from FDTD3dReference

bool fdtdReference(float *output, const float *input, const float *coeff, const int dimx, const int dimy, const int dimz, const int radius, const int timesteps)
{
    const int     outerDimx    = dimx + 2 * radius;
    const int     outerDimy    = dimy + 2 * radius;
    const int     outerDimz    = dimz + 2 * radius;
    const size_t  volumeSize   = outerDimx * outerDimy * outerDimz;
    const int     stride_y     = outerDimx;
    const int     stride_z     = stride_y * outerDimy;
    float        *intermediate = 0;
    const float  *bufsrc       = 0;
    float        *bufdst       = 0;
    float        *bufdstnext   = 0;

    // Allocate temporary buffer
//    printf(" calloc intermediate\n");
    intermediate = (float *)calloc(volumeSize, sizeof(float));

    // Decide which buffer to use first (result should end up in output)
    if ((timesteps % 2) == 0)
    {
        bufsrc     = input;
        bufdst     = intermediate;
        bufdstnext = output;
    }
    else
    {
        bufsrc     = input;
        bufdst     = output;
        bufdstnext = intermediate;
    }

    // Run the FDTD (naive method)
//    printf(" Host FDTD loop\n");

    for (int it = 0 ; it < timesteps ; it++)
    {
//        printf("\tt = %d\n", it);
        const float *src = bufsrc;
        float *dst       = bufdst;

        for (int iz = -radius ; iz < dimz + radius ; iz++)
        {
            for (int iy = -radius ; iy < dimy + radius ; iy++)
            {
                for (int ix = -radius ; ix < dimx + radius ; ix++)
                {
                    if (ix >= 0 && ix < dimx && iy >= 0 && iy < dimy && iz >= 0 && iz < dimz)
                    {
                        float value = (*src) * coeff[0];

                        for (int ir = 1 ; ir <= radius ; ir++)
                        {
                            value += coeff[ir] * (*(src + ir) + *(src - ir));                       // horizontal
                            value += coeff[ir] * (*(src + ir * stride_y) + *(src - ir * stride_y)); // vertical
                            value += coeff[ir] * (*(src + ir * stride_z) + *(src - ir * stride_z)); // in front & behind
                        }

                        *dst = value;
                    }
                    else
                    {
                        *dst = *src;
                    }

                    ++dst;
                    ++src;
                }
            }
        }

        // Rotate buffers
        float *tmp = bufdst;
        bufdst     = bufdstnext;
        bufdstnext = tmp;
        bufsrc = (const float *)tmp;
    }

//    printf("\n");

    if (intermediate)
        free(intermediate);

    return true;
}

bool compareData(const float *output, const float *reference, const int dimx, const int dimy, const int dimz, const int radius, const float tolerance)
{
    for (int iz = -radius ; iz < dimz + radius ; iz++)
    {
        for (int iy = -radius ; iy < dimy + radius ; iy++)
        {
            for (int ix = -radius ; ix < dimx + radius ; ix++)
            {
                if (ix >= 0 && ix < dimx && iy >= 0 && iy < dimy && iz >= 0 && iz < dimz)
                {
                    // Determine the absolute difference
                    float difference = fabs(*reference - *output);
                    float error;

                    // Determine the relative error
                    if (*reference != 0)
                        error = difference / *reference;
                    else
                        error = difference;

                    // Check the error is within the tolerance
                    if (error > tolerance)
                    {
                        printf("Data error at point (%d,%d,%d)\t%f instead of %f\n", ix, iy, iz, *output, *reference);
                        return false;
                    }
                }

                ++output;
                ++reference;
            }
        }
    }

    return true;
}

//
// Preemption variables and routines
//

float			*hOutput	= 0;
float			*hInput		= 0;
float			*hCoeff		= 0;
float			*dbufferOut	= 0;
float			*dbufferIn	= 0;
int				processedsteps = 0;
const int		dimx 		= 376;
const int		dimy 		= 376;
const int		dimz 		= 376;
const int		radius 		= 4;
const int		timesteps	= 5;
const int	 	padding		= (128 / sizeof(float)) - radius;
const int		outerDimx	= dimx + 2 * radius;
const int		outerDimy	= dimy + 2 * radius;
const int		outerDimz	= dimz + 2 * radius;
const size_t	volumeSize	= outerDimx * outerDimy * outerDimz;
dim3			dimBlock;
dim3			dimGrid;

// A testing routine
int FDTD3d_test(void *arg, float *tHtD, float *tK, float *tDtH)
{
	
	int nsteps = 2;
	float t1 = 0, t2 = 0, t3 = 0;
	int devId = 5;
	
	cudaSetDevice(devId);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, devId);
	printf("Device=%s\n", deviceProp.name);

	cudaEvent_t profileStart1, profileEnd1, profileStart2, profileEnd2, profileStart3, profileEnd3;
	CUDA_SAFE_CALL(cudaEventCreate(&profileStart1));
    CUDA_SAFE_CALL(cudaEventCreate(&profileEnd1));
	CUDA_SAFE_CALL(cudaEventCreate(&profileStart2));
    CUDA_SAFE_CALL(cudaEventCreate(&profileEnd2));
	CUDA_SAFE_CALL(cudaEventCreate(&profileStart3));
    CUDA_SAFE_CALL(cudaEventCreate(&profileEnd3));

	tHtD[0] = 0;
	tK[0] = 0;
	tDtH[0] = 0;

    // Enqueue start event
	for ( int i = 0; i < nsteps; i++ )
	{
		FDTD3d_start_mallocs(arg);

		cudaDeviceSynchronize();
		checkCudaErrors(cudaEventRecord(profileStart1, 0));
		// Transfer data from host to device
		// Copy the coefficients to the device coefficient buffer (already defined in constant memory space)
		CUDA_SAFE_CALL(cudaMemcpyToSymbol(stencil, (void *)hCoeff, (radius + 1) * sizeof(float)));
		// Copy the input to the device input buffer
		CUDA_SAFE_CALL(cudaMemcpy(dbufferIn + padding, hInput, volumeSize * sizeof(float), cudaMemcpyHostToDevice));
		// Copy the input to the device output buffer (actually only need the halo)
		CUDA_SAFE_CALL(cudaMemcpy(dbufferOut + padding, hInput, volumeSize * sizeof(float), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaEventRecord(profileEnd1, 0));

	    // Wait for the kernel to complete
		cudaDeviceSynchronize();
        checkCudaErrors(cudaEventElapsedTime(&t1, profileStart1, profileEnd1));
		tHtD[0] += t1;
		
		checkCudaErrors(cudaEventRecord(profileStart2, 0));
		launch_orig_FDTD3d(arg);
		checkCudaErrors(cudaEventRecord(profileEnd2, 0));

		cudaDeviceSynchronize();
        checkCudaErrors(cudaEventElapsedTime(&t2, profileStart2, profileEnd2));
		tK[0] += t2;

		checkCudaErrors(cudaEventRecord(profileStart3, 0));
		float *bufferDst = dbufferOut + padding;
		CUDA_SAFE_CALL(cudaMemcpy(hOutput, bufferDst, volumeSize * sizeof(float), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaFree(dbufferIn));
		CUDA_SAFE_CALL(cudaFree(dbufferOut));		
		checkCudaErrors(cudaEventRecord(profileEnd3, 0));

		cudaDeviceSynchronize();
        checkCudaErrors(cudaEventElapsedTime(&t3, profileStart3, profileEnd3));
		tDtH[0] += t3;
		
		// Compute results on CPU
	
		float *refOutput = (float *)calloc(volumeSize, sizeof(float));
		float tolerance = 0.0001f;
		printf("\nCompareData (tolerance %f)...\n", tolerance);
		fdtdReference(refOutput, hInput, hCoeff, dimx, dimy, dimz, radius, timesteps);
		bool compare = compareData(hOutput, refOutput, dimx, dimy, dimz, radius, tolerance);
		if ( compare == true )
			printf("Test passed\n");
		free(refOutput);
	}

	tHtD[0] /= nsteps;
	tK[0] /= nsteps;
	tDtH[0] /= nsteps;

	return 0;
}

// A generic routine that is not used anymore?
int FDTD3d_start_kernel(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;

	FDTD3d_start_mallocs(arg);
	
	// Transfer data from host to device
    // Copy the coefficients to the device coefficient buffer (already defined in constant memory space)
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(stencil, (void *)hCoeff, (radius + 1) * sizeof(float)));
    // Copy the input to the device input buffer
    CUDA_SAFE_CALL(cudaMemcpy(dbufferIn + padding, hInput, volumeSize * sizeof(float), cudaMemcpyHostToDevice));
    // Copy the input to the device output buffer (actually only need the halo)
    CUDA_SAFE_CALL(cudaMemcpy(dbufferOut + padding, hInput, volumeSize * sizeof(float), cudaMemcpyHostToDevice));

  
	return 0;
 }
 
// Memory allocation
int FDTD3d_start_mallocs(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;

	dimBlock.x = kstub->kconf.blocksize.x;
	dimBlock.y = kstub->kconf.blocksize.y;
	dimGrid.x = kstub->kconf.gridsize.x;
	dimGrid.y = kstub->kconf.gridsize.y;
	
	printf("Working on a %d array\n", volumeSize);
	// Allocate and set up host data 
	CUDA_SAFE_CALL(cudaMallocHost((void**)&hOutput, volumeSize * sizeof(float)));
    CUDA_SAFE_CALL(cudaMallocHost((void**)&hInput,  volumeSize * sizeof(float)));
    CUDA_SAFE_CALL(cudaMallocHost((void**)&hCoeff,  (radius + 1) * sizeof(float)));

    // Create coefficients
    for (int i = 0 ; i <= radius ; i++)
    {
        hCoeff[i] = 0.1f;
    }	

	// Generate random data

    const float lowerBound = 0.0f;
    const float upperBound = 1.0f;
	srand(0);
	float *data = hInput;
    for (int iz = 0 ; iz < outerDimz ; iz++)
    {
        for (int iy = 0 ; iy < outerDimy ; iy++)
        {
            for (int ix = 0 ; ix < outerDimx ; ix++)
            {
                *data = (float)(lowerBound + ((float)rand() / (float)RAND_MAX) * (upperBound - lowerBound));
                ++data;
            }
        }
    }
	
	// Zero output data
	memset( hOutput, 0, volumeSize * sizeof(float) );
	
    // Allocate memory buffers in device
    // Ensure that the inner data starts on a 128B boundary
    const size_t paddedVolumeSize = volumeSize + padding;

    CUDA_SAFE_CALL(cudaMalloc((void **)&dbufferOut, paddedVolumeSize * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&dbufferIn,  paddedVolumeSize * sizeof(float)));
	
	return 0;
}

// Start transfers
int FDTD3d_start_transfers(void *arg){
	
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
	printf("Sending two %d array\n", volumeSize);

    // Copy the coefficients to the device coefficient buffer (already defined in constant memory space)
//    CUDA_SAFE_CALL(cudaMemcpyToSymbolAsync(stencil, (void *)hCoeff, (radius + 1) * sizeof(float)));

#if defined(MEMCPY_ASYNC)

    // Copy the input to the device input buffer
	enqueue_tcomamnd(tqueues, dbufferIn + padding, hInput, volumeSize * sizeof(float), cudaMemcpyHostToDevice, 
						kstub->transfer_s[0], NONBLOCKING, DATA, MEDIUM, kstub);

    // Copy the input to the device output buffer (actually only need the halo)
	enqueue_tcomamnd(tqueues, dbufferOut + padding, hInput, volumeSize * sizeof(float), cudaMemcpyHostToDevice, 
						kstub->transfer_s[0], NONBLOCKING, LAST_TRANSFER, MEDIUM, kstub);
	
#else

    // Copy the input to the device input buffer
    CUDA_SAFE_CALL(cudaMemcpy(dbufferIn + padding, hInput, volumeSize * sizeof(float), cudaMemcpyHostToDevice));

    // Copy the input to the device output buffer (actually only need the halo)
    CUDA_SAFE_CALL(cudaMemcpy(dbufferOut + padding, hInput, volumeSize * sizeof(float), cudaMemcpyHostToDevice));

#endif

	kstub->HtD_tranfers_finished = 1;
      
	return 0;
}

// DtH transfers and deallocation

int FDTD3d_end_kernel(void *arg)
{

	t_kernel_stub *kstub = (t_kernel_stub *)arg;

#if defined(MEMCPY_ASYNC)
	cudaEventSynchronize(kstub->end_Exec);

    // Read the result back, result is in bufferSrc (after final toggle)
	float *bufferDst = dbufferOut + padding;
	enqueue_tcomamnd(tqueues, hOutput, bufferDst, volumeSize * sizeof(float), cudaMemcpyDeviceToHost, kstub->transfer_s[1] , NONBLOCKING, LAST_TRANSFER, MEDIUM, kstub);

#else
	
	cudaEventSynchronize(kstub->end_Exec);
	
    // Read the result back, result is in bufferSrc (after final toggle)
	float *bufferDst = dbufferOut + padding;
	CUDA_SAFE_CALL(cudaMemcpy(hOutput, bufferDst, volumeSize * sizeof(float),
                  cudaMemcpyDeviceToHost));
				 		 
	CUDA_SAFE_CALL(cudaFree(dbufferIn));
	CUDA_SAFE_CALL(cudaFree(dbufferOut));

#endif

	// Compute results on CPU
	
	// float *refOutput = (float *)calloc(volumeSize, sizeof(float));
    // float tolerance = 0.0001f;
    // printf("\nCompareData (tolerance %f)...\n", tolerance);

    // fdtdReference(refOutput, hInput, hCoeff, dimx, dimy, dimz, radius, timesteps);

    // bool compare = compareData(hOutput, refOutput, dimx, dimy, dimz, radius, tolerance);
	// if ( compare == true )
		// printf("Test passed\n");

	// free(refOutput);
	cudaFree(hOutput);
	cudaFree(hInput);
	
	return 0;
}

// Launch original kernel
int launch_orig_FDTD3d(void *arg)
{
	t_kernel_stub *kstub = (t_kernel_stub *)arg;
    // Execute the FDTD
    float *bufferSrc = dbufferIn + padding;
    float *bufferDst = dbufferOut + padding;

//	printf("Computation with %dx%d blocks of size %dx%d\n", dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y);
    for (int it = 0 ; it < timesteps ; it++)
    {
//       printf("\tt = %d ", it);

        // Launch the kernel
        FiniteDifferencesKernel<<<dimGrid, dimBlock>>>(bufferDst, bufferSrc, dimx, dimy, dimz);

        // Toggle the buffers
        float *tmp = bufferDst;
        bufferDst = bufferSrc;
        bufferSrc = tmp;
    }
	
	return 0;
}

int launch_preemp_FDTD3d(void *arg)
{
	
	t_kernel_stub *kstub = (t_kernel_stub *)arg;

    // Execute the FDTD
    float *bufferSrc = dbufferIn + padding;
    float *bufferDst = dbufferOut + padding;

//    for (int it = 0 ; it < timesteps ; it++)
//    {
//        printf("\tt = %d ", it);

        // Launch the kernel
//        FiniteDifferencesKernel<<<dimGrid, dimBlock>>>(bufferDst, bufferSrc, dimx, dimy, dimz);

        // Toggle the buffers
//        float *tmp = bufferDst;
//        bufferDst = bufferSrc;
//        bufferSrc = tmp;
//    }
	
	#ifdef SMT

	printf("Launching %d blocks of %d x %d threads between SM %d and %d for computing %d tasks\n", kstub->kconf.numSMs * kstub->kconf.max_persistent_blocks, kstub->kconf.blocksize.x, kstub->kconf.blocksize.y, kstub->idSMs[0], kstub->idSMs[1], kstub->total_tasks);
	preemp_SMT_FiniteDifferencesKernel<<<kstub->kconf.numSMs * kstub->kconf.max_persistent_blocks, kstub->kconf.blocksize.x, 0, *(kstub->execution_s)>>>
			(bufferDst, bufferSrc, dimx, dimy, dimz,
			kstub->idSMs[0],
			kstub->idSMs[1],
			kstub->total_tasks,
			kstub->kconf.coarsening,
			kstub->d_executed_tasks,
			kstub->gm_state);


	#else

	preemp_SMT_FiniteDifferencesKernel<<<kstub->kconf.numSMs * kstub->kconf.max_persistent_blocks, kstub->kconf.blocksize.x, 0, *(kstub->execution_s)>>>
			(bufferDst, bufferSrc, dimx, dimy, dimz,
			kstub->idSMs[0],
			kstub->idSMs[1],
			kstub->total_tasks,
			kstub->kconf.coarsening,
			kstub->d_executed_tasks,
			kstub->gm_state);
			// kstub->num_blocks_per_SM,
			// kstub->total_tasks,
			// kstub->kconf.coarsening,
			// kstub->d_SMs_cont,
			// kstub->d_executed_tasks,
			// kstub->gm_state
	);	
		
	#endif
	
	CUDA_SAFE_CALL(cudaMemcpy(kstub->h_executed_tasks, kstub->d_executed_tasks, sizeof(int), cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();
	printf("Executed tasks %d\n", kstub->h_executed_tasks);
	return 0;
}
