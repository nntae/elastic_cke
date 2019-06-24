#include <unistd.h>
#include <string.h>
#include <helper_functions.h>   // helper functions for string parsing
#include <helper_cuda.h>   
#include "elastic_kernel.h"
#include "BS/BS.h"
#include "VA/VA.h"
#include "MM/MM.h"
#include "RSC/RSC.h"
#include "SPMV/SPMV.h"
#include "PF/PF.h"
#include "Reduction/reduction.h"
#include "FDTD3d/FDTD3dGPU.h"
#include "Dummy/Dummy.h"
#include "CONV/CONV.h"
#include "CEDD/CEDD.h"
#include "HST/HST256.h"

int create_stubinfo(t_kernel_stub **stub, int deviceId, t_Kernel id, cudaStream_t *transfer_s, cudaStream_t *preemp_s)
{
	
	cudaError_t err;

	t_kernel_stub *k_stub = (t_kernel_stub *)calloc(1, sizeof(t_kernel_stub)); // Create kernel stub
	k_stub->deviceId = deviceId;
	k_stub->id = id;
	k_stub->kernel_finished = 0;
	k_stub->HtD_tranfers_finished = 0;
	k_stub->DtH_tranfers_finished = 0;
	
	// Streams
	cudaStream_t *kernel_s, *m_transfer_s;
	kernel_s = (cudaStream_t *)malloc(sizeof(cudaStream_t));
	err = cudaStreamCreate(kernel_s);
	checkCudaErrors(err);
	
	m_transfer_s = (cudaStream_t *)malloc(2*sizeof(cudaStream_t));
	err = cudaStreamCreate(&m_transfer_s[0]);
	err = cudaStreamCreate(&m_transfer_s[1]);
	checkCudaErrors(err);
	k_stub->execution_s = kernel_s;
	k_stub->transfer_s = m_transfer_s;
	k_stub->preemp_s = preemp_s;
	
	/** Get device name*/
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, deviceId);
	char *device_name = deviceProp.name;
	
	// Updating kernel info
	
	switch (id) {

		case BS:
			k_stub->launchCKEkernel = launch_preemp_BS;
			k_stub->launchORIkernel = launch_orig_BS;
			k_stub->startKernel = BS_start_kernel_dummy;
			k_stub->startMallocs = BS_start_mallocs;
			k_stub->startTransfers = BS_start_transfers;
			k_stub->endKernel = BS_end_kernel_dummy;
			
			if (strcmp(device_name, "Tesla K20c") == 0) {
				k_stub->kconf.numSMs = 13;
				k_stub->kconf.max_persistent_blocks = 16;
				k_stub->kconf.blocksize.x = 128,1,1;
				k_stub->kconf.gridsize.x  = 50 * k_stub->kconf.numSMs * k_stub->kconf.max_persistent_blocks;
				k_stub->total_tasks = k_stub->kconf.gridsize.x;
				k_stub->kconf.coarsening = 40;
			}
			else {
				if (strcmp(device_name, "GeForce GTX 980") == 0) {
					k_stub->kconf.numSMs = 16;
					k_stub->kconf.max_persistent_blocks = 16;
					k_stub->kconf.blocksize.x = 256;
					k_stub->kconf.gridsize.x = 25 * k_stub->kconf.numSMs * k_stub->kconf.max_persistent_blocks;
					k_stub->total_tasks = k_stub->kconf.gridsize.x;
					k_stub->kconf.coarsening = 40;
				}
				else{
					
					if (strcmp(device_name, "TITAN X (Pascal)") == 0) {
						k_stub->kconf.numSMs = 28;
						k_stub->kconf.max_persistent_blocks = 8;
						k_stub->kconf.blocksize.x = 256;
						//Data set 1 
						//k_stub->kconf.gridsize.x = 25 * k_stub->kconf.numSMs * k_stub->kconf.max_persistent_blocks;
						//Data set 2
						k_stub->kconf.gridsize.x = 50 * k_stub->kconf.numSMs * k_stub->kconf.max_persistent_blocks;
						k_stub->total_tasks = k_stub->kconf.gridsize.x;
						k_stub->kconf.coarsening = 40;
					}
					else{
						printf("Error: Unknown device\n");
						return -1;
					}
				}
			}
		
			break;
			
		case VA: 
			k_stub->launchCKEkernel = launch_preemp_VA;
			k_stub->launchORIkernel = launch_orig_VA;
			k_stub->startKernel = VA_start_kernel_dummy;
			k_stub->startMallocs = VA_start_mallocs;
			k_stub->startTransfers = VA_start_transfers;
			k_stub->endKernel = VA_end_kernel_dummy;
			
			if (strcmp(device_name, "Tesla K20c") == 0) {
				k_stub->kconf.numSMs = 13;
				k_stub->kconf.max_persistent_blocks = 16;
				k_stub->kconf.blocksize.x = 128;
				k_stub->kconf.gridsize.x = 50 * k_stub->kconf.numSMs * k_stub->kconf.max_persistent_blocks;
				k_stub->total_tasks = k_stub->kconf.gridsize.x;
				k_stub->kconf.coarsening = 40;
			}
			else {
				if (strcmp(device_name, "GeForce GTX 980") == 0) {
					k_stub->kconf.numSMs = 16;
					k_stub->kconf.max_persistent_blocks = 16;
					k_stub->kconf.blocksize.x = 256;
					k_stub->kconf.gridsize.x = 50 * k_stub->kconf.numSMs * k_stub->kconf.max_persistent_blocks;
					k_stub->total_tasks = k_stub->kconf.gridsize.x;
					k_stub->kconf.coarsening = 40;
				}
				else{
					if (strcmp(device_name, "TITAN X (Pascal)") == 0) {
						k_stub->kconf.numSMs = 28;
						k_stub->kconf.max_persistent_blocks = 8;
						k_stub->kconf.blocksize.x = 256;
						// Data set 1
						//k_stub->kconf.gridsize.x = 50 * k_stub->kconf.numSMs * k_stub->kconf.max_persistent_blocks;
						// Data set 2
						k_stub->kconf.gridsize.x = 100 * k_stub->kconf.numSMs * k_stub->kconf.max_persistent_blocks;
						k_stub->total_tasks = k_stub->kconf.gridsize.x;
						k_stub->kconf.coarsening = 40;
					}
					else{
						printf("Error: Unknown device\n");
						return -1;
					}
				}
			}
		
			break;
			
		case MM: 
		
			t_MM_params *MM_params;
			
			MM_params = (t_MM_params *)calloc(1, sizeof(t_MM_params));
			
			/*//Dataset 1
			MM_params->Asize.x=4096;MM_params->Asize.y=4096;
			MM_params->Bsize.x=4096;MM_params->Bsize.y=4096;*/
			
			//Dataset 2
			MM_params->Asize.x=2048;MM_params->Asize.y=2048;
			MM_params->Bsize.x=2048;MM_params->Bsize.y=2048;
			
			k_stub->params = (void *)MM_params;
			
			k_stub->launchCKEkernel = launch_preemp_MM;
			k_stub->launchORIkernel = launch_orig_MM;
			k_stub->startKernel = MM_start_kernel;
			k_stub->startMallocs = MM_start_mallocs;
			k_stub->startTransfers = MM_start_transfers;
			k_stub->endKernel = MM_end_kernel;
			
			if (strcmp(device_name, "Tesla K20c") == 0) {
				k_stub->kconf.numSMs = 13;
				k_stub->kconf.max_persistent_blocks = 8;
				k_stub->kconf.blocksize.x = 16;
				k_stub->kconf.blocksize.y = 16;
				MM_params->gridDimX = MM_params->Bsize.x/k_stub->kconf.blocksize.x; // Add information loss during linearization
				k_stub->kconf.gridsize.x = MM_params->Bsize.x/k_stub->kconf.blocksize.x * MM_params->Asize.y/k_stub->kconf.blocksize.y;
				k_stub->kconf.gridsize.y = 1; //Grid Linearization
				k_stub->total_tasks = k_stub->kconf.gridsize.x;
				k_stub->kconf.coarsening = 1;
			}
			else {
				if (strcmp(device_name, "GeForce GTX 980") == 0) {
					k_stub->kconf.numSMs = 16;
					k_stub->kconf.max_persistent_blocks = 8;
					k_stub->kconf.blocksize.x = 16;
					k_stub->kconf.blocksize.y = 16;
					MM_params->gridDimX = MM_params->Bsize.x/k_stub->kconf.blocksize.x; // Add information loss during linearization
					k_stub->kconf.gridsize.x = MM_params->Bsize.x/k_stub->kconf.blocksize.x * MM_params->Asize.y/k_stub->kconf.blocksize.y;
					k_stub->kconf.gridsize.y = 1; //Grid Linearization
					k_stub->total_tasks = k_stub->kconf.gridsize.x;
					k_stub->kconf.coarsening = 1;
				}
				else{
					if (strcmp(device_name, "TITAN X (Pascal)") == 0) {
						k_stub->kconf.numSMs = 28;
						k_stub->kconf.max_persistent_blocks = 8;
						k_stub->kconf.blocksize.x = 16;
						k_stub->kconf.blocksize.y = 16;
						MM_params->gridDimX = MM_params->Bsize.x/k_stub->kconf.blocksize.x; // Add information loss during linearization
						k_stub->kconf.gridsize.x = MM_params->Bsize.x/k_stub->kconf.blocksize.x * MM_params->Asize.y/k_stub->kconf.blocksize.y;
						k_stub->kconf.gridsize.y = 1; //Grid Linearization
						k_stub->total_tasks = k_stub->kconf.gridsize.x;
						k_stub->kconf.coarsening = 1;
					}
					else{
						printf("Error: Unknown device\n");
						return -1;
					}
				}
			}

			break;
			
		case RSC_MODEL:
			t_RSC_params *RSC_params;
	 
			RSC_params = (t_RSC_params *)calloc(1, sizeof(t_RSC_params));
			k_stub->params = (void *)RSC_params;
			
			k_stub->launchCKEkernel = launch_preemp_RSC_model;
			k_stub->launchORIkernel = launch_orig_RSC_model;
			k_stub->startKernel = RSC_model_start_kernel;
			k_stub->endKernel = RSC_model_end_kernel;
			
			if (strcmp(device_name, "Tesla K20c") == 0) {
				k_stub->kconf.numSMs = 13;
				k_stub->kconf.max_persistent_blocks = 25; //6;
				k_stub->kconf.blocksize.x = 64 ;//256;
				k_stub->kconf.blocksize.y = 1;
				k_stub->kconf.gridsize.x = 104;
				k_stub->kconf.gridsize.y = 1;
				k_stub->total_tasks = k_stub->kconf.gridsize.x;
				k_stub->kconf.coarsening = 1;
			}
			else {
				if (strcmp(device_name, "Gefore GTX 980") == 0) {
					k_stub->kconf.numSMs = 16;
					k_stub->kconf.max_persistent_blocks = 6;
					k_stub->kconf.blocksize.x = 256;
					k_stub->kconf.blocksize.y = 1;
					k_stub->kconf.gridsize.x = 104; //6 bloques permanentes * 13 SMs
					k_stub->kconf.gridsize.y = 1; //Grid Linearization
					k_stub->total_tasks = k_stub->kconf.gridsize.x;
					k_stub->kconf.coarsening = 1;
				}
				else{
					printf("Error: Unknown device\n");
					return -1;
				}
			}

			break;
			
		case RSC_EVALUATE:
			t_RSC_params *RSCE_params;
	 
			//RSCE_params = (t_RSC_params *)calloc(1, sizeof(t_RSC_params));
			//k_stub->params = (void *)RSCE_params;
			
			k_stub->launchCKEkernel = launch_preemp_RSC_evaluate;
			k_stub->launchORIkernel = launch_orig_RSC_evaluate;
			k_stub->startKernel = RSC_evaluate_start_kernel;
			k_stub->endKernel = RSC_evaluate_end_kernel;
			
			if (strcmp(device_name, "Tesla K20c") == 0) {
				k_stub->kconf.numSMs = 13;
				k_stub->kconf.max_persistent_blocks = 8;
				k_stub->kconf.blocksize.x = 256;
				k_stub->kconf.blocksize.y = 1;
				k_stub->kconf.gridsize.x = 104;
				k_stub->kconf.gridsize.y = 1;
				k_stub->total_tasks = k_stub->kconf.gridsize.x;
				k_stub->kconf.coarsening = 1;
			}
			else {
				if (strcmp(device_name, "Gefore GTX 980") == 0) {
					k_stub->kconf.numSMs = 16;
					k_stub->kconf.max_persistent_blocks = 8;
					k_stub->kconf.blocksize.x = 256;
					k_stub->kconf.blocksize.y = 1;
					k_stub->kconf.gridsize.x = 104;
					k_stub->kconf.gridsize.y = 1; //Grid Linearization
					k_stub->total_tasks = k_stub->kconf.gridsize.x;
					k_stub->kconf.coarsening = 1;
				}
				else{
					printf("Error: Unknown device\n");
					return -1;
				}
			}

			break;
			
		case SPMV_CSRscalar:
			
			k_stub->launchCKEkernel = launch_preemp_SPMVcsr;
			k_stub->launchORIkernel = launch_orig_SPMVcsr;
			k_stub->startKernel = SPMVcsr_start_kernel;
			k_stub->startMallocs = SPMVcsr_start_mallocs;
			k_stub->startTransfers = SPMVcsr_start_transfers;
			k_stub->endKernel = SPMVcsr_end_kernel;
		
			if (strcmp(device_name, "TITAN X (Pascal)") == 0) {
					k_stub->kconf.numSMs = 28;
					k_stub->kconf.max_persistent_blocks = 16;
					k_stub->kconf.blocksize.x = 128;
					k_stub->kconf.blocksize.y = 1;
					k_stub->kconf.gridsize.x = k_stub->kconf.numSMs * k_stub->kconf.max_persistent_blocks * k_stub->kconf.blocksize.x ;//At least one row per thread when all thread
					k_stub->kconf.gridsize.y = 1; //Grid Linearization
					k_stub->total_tasks = k_stub->kconf.gridsize.x;
					k_stub->kconf.coarsening = 1;
			}
			else{
				printf("Error: Unknown device\n");
				return -1;
			}
			
			break;

		case Reduction:
			
			k_stub->launchCKEkernel = launch_preemp_reduce;
			k_stub->launchORIkernel = launch_orig_reduce;
			k_stub->startKernel = reduce_start_kernel;
			k_stub->startMallocs = reduce_start_mallocs;
			k_stub->startTransfers = reduce_start_transfers;
			k_stub->endKernel = reduce_end_kernel;
		
			if (strcmp(device_name, "TITAN X (Pascal)") == 0) {
					k_stub->kconf.numSMs = 28;
					k_stub->kconf.max_persistent_blocks = 8;
					k_stub->kconf.blocksize.x = 256;
					k_stub->kconf.blocksize.y = 1;
					//Data set 1
					//k_stub->kconf.gridsize.x =  64*28*8;
					// Data set 2
					k_stub->kconf.gridsize.x =  640*28*8; // 64 * number_of_permanent_blocks, a ver que tal
					k_stub->kconf.gridsize.y = 1; //Grid Linearization
					k_stub->total_tasks = k_stub->kconf.gridsize.x;
					k_stub->kconf.coarsening = 20;
			}
			else{
				printf("Error: Unknown device\n");
				return -1;
			}

			break;
			
		/*case FDTD3d:
			
			k_stub->launchCKEkernel = launch_preemp_FDTD3d;
			k_stub->launchORIkernel = launch_orig_FDTD3d;
			k_stub->startKernel = FDTD3d_start_kernel;
			k_stub->startMallocs = FDTD3d_start_mallocs;
			k_stub->startTransfers = FDTD3d_start_transfers;
			k_stub->endKernel = FDTD3d_end_kernel;

			if (strcmp(device_name, "TITAN X (Pascal)") == 0) {
					k_stub->kconf.numSMs = 28;
					k_stub->kconf.max_persistent_blocks = 1; // Needs many registers by thread
					k_stub->kconf.blocksize.x = 32;
					k_stub->kconf.blocksize.y = 16;
					k_stub->kconf.gridsize.x =  12; // Original is 12
					k_stub->kconf.gridsize.y = 	24;  // Original is 24
					k_stub->total_tasks = k_stub->kconf.gridsize.x*k_stub->kconf.gridsize.y;
					k_stub->kconf.coarsening = 1;
			}
			else{
				printf("Error: Unknown device\n");
				return -1;
			}

			break;
		*/
		case PF:
			t_PF_params *PF_params;
			
			PF_params = (t_PF_params *)calloc(1, sizeof(t_PF_params));
			
			// Data set 1
			//PF_params->nRows = 500;
			//PF_params->nCols = 6000;
			//PF_params->param_pyramid_height = 126;
			
			
			//Data set 2
			PF_params->nRows = 5000;
			PF_params->nCols = 6000;
			PF_params->param_pyramid_height = 126;
			
			k_stub->params = (void *)PF_params;
		
			k_stub->launchCKEkernel = launch_preemp_PF;
			k_stub->launchORIkernel = launch_orig_PF;
			k_stub->startKernel = PF_start_kernel;
			k_stub->startMallocs = PF_start_mallocs;
			k_stub->startTransfers = PF_start_transfers;
			k_stub->endKernel = PF_end_kernel;
			
			if (strcmp(device_name, "Tesla K20c") == 0) {
				k_stub->kconf.numSMs = 13;
				k_stub->kconf.max_persistent_blocks = 8;
				k_stub->kconf.blocksize.x = 256;
				int smallBlockCol = k_stub->kconf.blocksize.x-(PF_params->param_pyramid_height)*2;
				k_stub->kconf.gridsize.x  = PF_params->nCols/smallBlockCol+((PF_params->nCols%smallBlockCol==0)?0:1);
				k_stub->total_tasks = k_stub->kconf.gridsize.x;
				k_stub->kconf.coarsening = 1;
			}
			else {
				if (strcmp(device_name, "GeForce GTX 980") == 0) {
					k_stub->kconf.numSMs = 16;
					k_stub->kconf.max_persistent_blocks = 8;
					k_stub->kconf.blocksize.x = 256;
					int smallBlockCol = k_stub->kconf.blocksize.x-(PF_params->param_pyramid_height)*2;
					k_stub->kconf.gridsize.x = PF_params->nCols/smallBlockCol+((PF_params->nCols%smallBlockCol==0)?0:1);
					k_stub->total_tasks = k_stub->kconf.gridsize.x;
					k_stub->kconf.coarsening = 1;
				}
				else{
					
					if (strcmp(device_name, "TITAN X (Pascal)") == 0) {
						k_stub->kconf.numSMs = 28;
						k_stub->kconf.max_persistent_blocks = 8;
						k_stub->kconf.blocksize.x = 256;
						int smallBlockCol = k_stub->kconf.blocksize.x-(PF_params->param_pyramid_height)*2;
						k_stub->kconf.gridsize.x = PF_params->nCols/smallBlockCol+((PF_params->nCols%smallBlockCol==0)?0:1);
						k_stub->total_tasks = k_stub->kconf.gridsize.x;
						k_stub->kconf.coarsening = 1;
					}
					else{
						printf("Error: Unknown device\n");
						return -1;
					}
				}
			}
		
			break;
			
		case RCONV:
			t_CONV_params *RCONV_params;
			
			RCONV_params = (t_CONV_params *)calloc(1, sizeof(t_CONV_params));
			
			RCONV_params->conv_rows=6144;
			RCONV_params->conv_cols=6144;
			RCONV_params->gridDimY = 768 * 2;
			k_stub->params = (void *)RCONV_params;
		
			k_stub->launchCKEkernel = launch_preemp_RCONV;
			k_stub->launchORIkernel = launch_orig_RCONV;
			k_stub->startKernel = RCONV_start_kernel;
			k_stub->startMallocs = RCONV_start_mallocs;
			k_stub->startTransfers = RCONV_start_transfers;
			k_stub->endKernel = RCONV_end_kernel;
			
			if (strcmp(device_name, "Tesla K20c") == 0) {		
				//RCONV
				k_stub->kconf.numSMs = 13;
				k_stub->kconf.max_persistent_blocks = 16;
				k_stub->kconf.blocksize.x = 16;
				k_stub->kconf.blocksize.y = 4;
				k_stub->kconf.gridsize.x = 24 * 768 * 4;
				k_stub->kconf.gridsize.y = 1; //Grid Linearization
				k_stub->total_tasks = k_stub->kconf.gridsize.x;
				k_stub->kconf.coarsening = 1;
			}
			else {
				if (strcmp(device_name, "GeForce GTX 980") == 0) {
					//RCONV
					k_stub->kconf.numSMs = 16;
					k_stub->kconf.max_persistent_blocks = 32;
					k_stub->kconf.blocksize.x = 16;
					k_stub->kconf.blocksize.y = 4;
					k_stub->kconf.gridsize.x = 24 * 768 * 4;
					k_stub->kconf.gridsize.y = 1; //Grid Linearization
					k_stub->total_tasks = k_stub->kconf.gridsize.x;
					k_stub->kconf.coarsening = 1;
				}
				else{
					if (strcmp(device_name, "TITAN X (Pascal)") == 0) {
						//RCONV
						k_stub->kconf.numSMs = 28;
						k_stub->kconf.max_persistent_blocks = 23;
						k_stub->kconf.blocksize.x = 16;
						k_stub->kconf.blocksize.y = 4;
						k_stub->kconf.gridsize.x = 24 * 768 * 4;
						k_stub->kconf.gridsize.y = 1; //Grid Linearization
						k_stub->total_tasks = k_stub->kconf.gridsize.x;
						k_stub->kconf.coarsening = 1;
					}
					else{
						printf("Error: Unknown device\n");
						return -1;
					}
				}
			}
		
			break;
			
		case CCONV:
			t_CONV_params *CCONV_params;
			
			CCONV_params = (t_CONV_params *)calloc(1, sizeof(t_CONV_params));
			
			CCONV_params->conv_rows=6144;
			CCONV_params->conv_cols=6144;
			CCONV_params->gridDimY = 48 * 2;
			k_stub->params = (void *)CCONV_params;
		
			k_stub->launchCKEkernel = launch_preemp_CCONV;
			k_stub->launchORIkernel = launch_orig_CCONV;
			
			if (strcmp(device_name, "Tesla K20c") == 0) {		
				k_stub->kconf.numSMs = 13;
				k_stub->kconf.max_persistent_blocks = 9;
				k_stub->kconf.blocksize.x = 16;
				k_stub->kconf.blocksize.y = 8;
				k_stub->kconf.gridsize.x = 192 * 48 * 4;
				k_stub->kconf.gridsize.y = 1; //Grid Linearization
				k_stub->total_tasks = k_stub->kconf.gridsize.x;
				k_stub->kconf.coarsening = 1;
			}
			else {
				if (strcmp(device_name, "GeForce GTX 980") == 0) {
					k_stub->kconf.numSMs = 16;
					k_stub->kconf.max_persistent_blocks = 16;
					k_stub->kconf.blocksize.x = 16;
					k_stub->kconf.blocksize.y = 8;
					k_stub->kconf.gridsize.x = 192 * 48 * 4;
					k_stub->kconf.gridsize.y = 1; //Grid Linearization
					k_stub->total_tasks = k_stub->kconf.gridsize.x;
					k_stub->kconf.coarsening = 1;
				}
				else{
					if (strcmp(device_name, "TITAN X (Pascal)") == 0) {
						k_stub->kconf.numSMs = 28; 
						k_stub->kconf.blocksize.x = 16;
						k_stub->kconf.blocksize.y = 8;
						k_stub->kconf.gridsize.x = 192 * 48 * 4;
						k_stub->kconf.gridsize.y = 1; //Grid Linearization
						k_stub->total_tasks = k_stub->kconf.gridsize.x;
						k_stub->kconf.coarsening = 1;
					}
					else{
						printf("Error: Unknown device\n");
						return -1;
					}
				}
			}
		
			break;

		/*case Dummy:
			
			k_stub->launchCKEkernel = launch_preemp_dummy;
			k_stub->launchORIkernel = launch_orig_dummy;
			k_stub->startKernel = dummy_start_kernel;
			k_stub->startMallocs = dummy_start_mallocs;
			k_stub->startTransfers = dummy_start_transfers;
			k_stub->endKernel = dummy_end_kernel;
		
			if (strcmp(device_name, "TITAN X (Pascal)") == 0) {
					k_stub->kconf.numSMs = 28;
					k_stub->kconf.max_persistent_blocks = 8;
					k_stub->kconf.blocksize.x = 16;
					k_stub->kconf.blocksize.y = 16;
					k_stub->kconf.gridsize.x =  64;
					k_stub->kconf.gridsize.y = 64;
					k_stub->total_tasks = k_stub->kconf.gridsize.x*k_stub->kconf.gridsize.y;
					k_stub->kconf.coarsening = 5000;
			}
			else{
				printf("Error: Unknown device\n");
				return -1;
			}

			break;
			*/			
		case GCEDD:
			t_CEDD_params *GCEDD_params;
			
			GCEDD_params = (t_CEDD_params *)calloc(1, sizeof(t_CEDD_params));
			
			GCEDD_params->nRows=3072 * 2;
			GCEDD_params->nCols=4608 * 2;
			k_stub->params = (void *)GCEDD_params;
		
			k_stub->launchCKEkernel = launch_preemp_GCEDD;
			k_stub->launchORIkernel = launch_orig_GCEDD;
			k_stub->startKernel = GCEDD_start_kernel;
			k_stub->endKernel = GCEDD_end_kernel;

			k_stub->startMallocs = GCEDD_start_mallocs;
			k_stub->startTransfers = GCEDD_start_transfers;
			
			if (strcmp(device_name, "Tesla K20c") == 0) {			
				k_stub->kconf.numSMs = 13;
				k_stub->kconf.max_persistent_blocks = 8;
				k_stub->kconf.blocksize.x = 16;
				k_stub->kconf.blocksize.y = 16;
				GCEDD_params->gridDimX = (GCEDD_params->nCols - 2)/k_stub->kconf.blocksize.x; // Add information loss during linearization
				GCEDD_params->gridDimY = (GCEDD_params->nRows - 2)/k_stub->kconf.blocksize.y;
				k_stub->kconf.gridsize.x = GCEDD_params->gridDimX * GCEDD_params->gridDimY;
				k_stub->kconf.gridsize.y = 1; //Grid Linearization
				k_stub->total_tasks = k_stub->kconf.gridsize.x;
				k_stub->kconf.coarsening = 1;
			}
			else {
				if (strcmp(device_name, "GeForce GTX 980") == 0) {
					k_stub->kconf.numSMs = 16;
					k_stub->kconf.max_persistent_blocks = 8;
					k_stub->kconf.blocksize.x = 16;
					k_stub->kconf.blocksize.y = 16;
					GCEDD_params->gridDimX = (GCEDD_params->nCols - 2)/k_stub->kconf.blocksize.x; // Add information loss during linearization
					GCEDD_params->gridDimY = (GCEDD_params->nRows - 2)/k_stub->kconf.blocksize.y;
					k_stub->kconf.gridsize.x = GCEDD_params->gridDimX * GCEDD_params->gridDimY;
					k_stub->kconf.gridsize.y = 1; //Grid Linearization
					k_stub->total_tasks = k_stub->kconf.gridsize.x;
					k_stub->kconf.coarsening = 1;
				}
				else{
					if (strcmp(device_name, "TITAN X (Pascal)") == 0) {
						k_stub->kconf.numSMs = 28;
						k_stub->kconf.max_persistent_blocks = 8;
						k_stub->kconf.blocksize.x = 16;
						k_stub->kconf.blocksize.y = 16;
						GCEDD_params->gridDimX = (GCEDD_params->nCols - 2)/k_stub->kconf.blocksize.x; // Add information loss during linearization
						GCEDD_params->gridDimY = (GCEDD_params->nRows - 2)/k_stub->kconf.blocksize.y;
						k_stub->kconf.gridsize.x = GCEDD_params->gridDimX * GCEDD_params->gridDimY;
						k_stub->kconf.gridsize.y = 1; //Grid Linearization
						k_stub->total_tasks = k_stub->kconf.gridsize.x;
						k_stub->kconf.coarsening = 1;
					}
					else{
						printf("Error: Unknown device\n");
						return -1;
					}
				}
			}
		
			break;
			
		case SCEDD:
			t_CEDD_params *SCEDD_params;
			
			SCEDD_params = (t_CEDD_params *)calloc(1, sizeof(t_CEDD_params));
			
			SCEDD_params->nRows=3072 * 2;
			SCEDD_params->nCols=4608 * 2;
			k_stub->params = (void *)SCEDD_params;
		
			k_stub->launchCKEkernel = launch_preemp_SCEDD;
			k_stub->launchORIkernel = launch_orig_SCEDD;
			
			if (strcmp(device_name, "Tesla K20c") == 0) {			
				k_stub->kconf.numSMs = 13;
				k_stub->kconf.max_persistent_blocks = 8;
				k_stub->kconf.blocksize.x = 16;
				k_stub->kconf.blocksize.y = 16;
				SCEDD_params->gridDimX = (SCEDD_params->nCols - 2)/k_stub->kconf.blocksize.x; // Add information loss during linearization
				SCEDD_params->gridDimY = (SCEDD_params->nRows - 2)/k_stub->kconf.blocksize.y;
				k_stub->kconf.gridsize.x = (SCEDD_params->gridDimX * SCEDD_params->gridDimY) / 1;
				k_stub->kconf.gridsize.y = 1; //Grid Linearization
				k_stub->total_tasks = k_stub->kconf.gridsize.x;
				k_stub->kconf.coarsening = 1;
			}
			else {
				if (strcmp(device_name, "GeForce GTX 980") == 0) {
					k_stub->kconf.numSMs = 16;
					k_stub->kconf.max_persistent_blocks = 8;
					k_stub->kconf.blocksize.x = 16;
					k_stub->kconf.blocksize.y = 16;
					SCEDD_params->gridDimX = (SCEDD_params->nCols - 2)/k_stub->kconf.blocksize.x; // Add information loss during linearization
					SCEDD_params->gridDimY = (SCEDD_params->nRows - 2)/k_stub->kconf.blocksize.y;
					k_stub->kconf.gridsize.x = SCEDD_params->gridDimX * SCEDD_params->gridDimY;
					k_stub->kconf.gridsize.y = 1; //Grid Linearization
					k_stub->total_tasks = k_stub->kconf.gridsize.x;
					k_stub->kconf.coarsening = 1;
				}
				else{
					if (strcmp(device_name, "TITAN X (Pascal)") == 0) {
						k_stub->kconf.numSMs = 28;
						k_stub->kconf.max_persistent_blocks = 8;
						k_stub->kconf.blocksize.x = 16;
						k_stub->kconf.blocksize.y = 16;
						SCEDD_params->gridDimX = (SCEDD_params->nCols - 2)/k_stub->kconf.blocksize.x; // Add information loss during linearization
						SCEDD_params->gridDimY = (SCEDD_params->nRows - 2)/k_stub->kconf.blocksize.y;
						k_stub->kconf.gridsize.x = SCEDD_params->gridDimX * SCEDD_params->gridDimY;
						k_stub->kconf.gridsize.y = 1; //Grid Linearization
						k_stub->total_tasks = k_stub->kconf.gridsize.x;
						k_stub->kconf.coarsening = 1;
					}
					else{
						printf("Error: Unknown device\n");
						return -1;
					}
				}
			}
		
			break;
			
		case NCEDD:
			t_CEDD_params *NCEDD_params;
			
			NCEDD_params = (t_CEDD_params *)calloc(1, sizeof(t_CEDD_params));
			
			NCEDD_params->nRows=3072 * 2;
			NCEDD_params->nCols=4608 * 2;
			k_stub->params = (void *)NCEDD_params;
		
			k_stub->launchCKEkernel = launch_preemp_NCEDD;
			k_stub->launchORIkernel = launch_orig_NCEDD;
			
			if (strcmp(device_name, "Tesla K20c") == 0) {			
				k_stub->kconf.numSMs = 13;
				k_stub->kconf.max_persistent_blocks = 8;
				k_stub->kconf.blocksize.x = 16;
				k_stub->kconf.blocksize.y = 16;
				NCEDD_params->gridDimX = (NCEDD_params->nCols - 2)/k_stub->kconf.blocksize.x; // Add information loss during linearization
				NCEDD_params->gridDimY = (NCEDD_params->nRows - 2)/k_stub->kconf.blocksize.y;
				k_stub->kconf.gridsize.x = NCEDD_params->gridDimX * NCEDD_params->gridDimY;
				k_stub->kconf.gridsize.y = 1; //Grid Linearization
				k_stub->total_tasks = k_stub->kconf.gridsize.x;
				k_stub->kconf.coarsening = 1;
			}
			else {
				if (strcmp(device_name, "GeForce GTX 980") == 0) {
					k_stub->kconf.numSMs = 16;
					k_stub->kconf.max_persistent_blocks = 8;
					k_stub->kconf.blocksize.x = 16;
					k_stub->kconf.blocksize.y = 16;
					NCEDD_params->gridDimX = (NCEDD_params->nCols - 2)/k_stub->kconf.blocksize.x; // Add information loss during linearization
					NCEDD_params->gridDimY = (NCEDD_params->nRows - 2)/k_stub->kconf.blocksize.y;
					k_stub->kconf.gridsize.x = NCEDD_params->gridDimX * NCEDD_params->gridDimY;
					k_stub->kconf.gridsize.y = 1; //Grid Linearization
					k_stub->total_tasks = k_stub->kconf.gridsize.x;
					k_stub->kconf.coarsening = 1;
				}
				else{
					if (strcmp(device_name, "TITAN X (Pascal)") == 0) {
						k_stub->kconf.numSMs = 28;
						k_stub->kconf.max_persistent_blocks = 8;
						k_stub->kconf.blocksize.x = 16;
						k_stub->kconf.blocksize.y = 16;
						NCEDD_params->gridDimX = (NCEDD_params->nCols - 2)/k_stub->kconf.blocksize.x; // Add information loss during linearization
						NCEDD_params->gridDimY = (NCEDD_params->nRows - 2)/k_stub->kconf.blocksize.y;
						k_stub->kconf.gridsize.x = NCEDD_params->gridDimX * NCEDD_params->gridDimY;
						k_stub->kconf.gridsize.y = 1; //Grid Linearization
						k_stub->total_tasks = k_stub->kconf.gridsize.x;
						k_stub->kconf.coarsening = 1;
					}
					else{
						printf("Error: Unknown device\n");
						return -1;
					}
				}
			}
		
			break;
			
		case HCEDD:
			t_CEDD_params *HCEDD_params;
			
			HCEDD_params = (t_CEDD_params *)calloc(1, sizeof(t_CEDD_params));
			
			HCEDD_params->nRows=3072 * 2;
			HCEDD_params->nCols=4608 * 2;
			k_stub->params = (void *)HCEDD_params;
		
			k_stub->launchCKEkernel = launch_preemp_HCEDD;
			k_stub->launchORIkernel = launch_orig_HCEDD;
			
			if (strcmp(device_name, "Tesla K20c") == 0) {			
				k_stub->kconf.numSMs = 13;
				k_stub->kconf.max_persistent_blocks = 8;
				k_stub->kconf.blocksize.x = 16;
				k_stub->kconf.blocksize.y = 16;
				HCEDD_params->gridDimX = (HCEDD_params->nCols - 2)/k_stub->kconf.blocksize.x; // Add information loss during linearization
				HCEDD_params->gridDimY = (HCEDD_params->nRows - 2)/k_stub->kconf.blocksize.y;
				k_stub->kconf.gridsize.x = HCEDD_params->gridDimX * HCEDD_params->gridDimY;
				k_stub->kconf.gridsize.y = 1; //Grid Linearization
				k_stub->total_tasks = k_stub->kconf.gridsize.x;
				k_stub->kconf.coarsening = 1;
			}
			else {
				if (strcmp(device_name, "GeForce GTX 980") == 0) {
					k_stub->kconf.numSMs = 16;
					k_stub->kconf.max_persistent_blocks = 8;
					k_stub->kconf.blocksize.x = 16;
					k_stub->kconf.blocksize.y = 16;
					HCEDD_params->gridDimX = (HCEDD_params->nCols - 2)/k_stub->kconf.blocksize.x; // Add information loss during linearization
					HCEDD_params->gridDimY = (HCEDD_params->nRows - 2)/k_stub->kconf.blocksize.y;
					k_stub->kconf.gridsize.x = HCEDD_params->gridDimX * HCEDD_params->gridDimY;
					k_stub->kconf.gridsize.y = 1; //Grid Linearization
					k_stub->total_tasks = k_stub->kconf.gridsize.x;
					k_stub->kconf.coarsening = 1;
				}
				else{
					if (strcmp(device_name, "TITAN X (Pascal)") == 0) {
						k_stub->kconf.numSMs = 28;
						k_stub->kconf.max_persistent_blocks = 8;
						k_stub->kconf.blocksize.x = 16;
						k_stub->kconf.blocksize.y = 16;
						HCEDD_params->gridDimX = (HCEDD_params->nCols - 2)/k_stub->kconf.blocksize.x; // Add information loss during linearization
						HCEDD_params->gridDimY = (HCEDD_params->nRows - 2)/k_stub->kconf.blocksize.y;
						k_stub->kconf.gridsize.x = HCEDD_params->gridDimX * HCEDD_params->gridDimY;
						k_stub->kconf.gridsize.y = 1; //Grid Linearization
						k_stub->total_tasks = k_stub->kconf.gridsize.x;
						k_stub->kconf.coarsening = 1;
					}
					else{
						printf("Error: Unknown device\n");
						return -1;
					}
				}
			}
		
			break;
			
		case HST256:
			k_stub->launchCKEkernel = launch_preemp_HST256;
			k_stub->launchORIkernel = launch_orig_HST256;
			k_stub->startKernel = HST256_start_kernel;
			k_stub->endKernel = HST256_end_kernel;
			
			k_stub->startMallocs = HST256_start_mallocs;
			k_stub->startTransfers = HST256_start_transfers;
			
			if (strcmp(device_name, "Tesla K20c") == 0) {
				k_stub->kconf.numSMs = 13;
				k_stub->kconf.max_persistent_blocks = 8;
				k_stub->kconf.blocksize.x = 192;
				k_stub->kconf.gridsize.x  = 240;
				k_stub->total_tasks = k_stub->kconf.gridsize.x * 24;
				//k_stub->total_tasks = (64 * 1048576)/k_stub->kconf.blocksize.x + (((64 * 1048576)%k_stub->kconf.blocksize.x==0)?0:1);
				k_stub->kconf.coarsening = 1;
			}
			else {
				if (strcmp(device_name, "GeForce GTX 980") == 0) {
					k_stub->kconf.numSMs = 16;
					k_stub->kconf.max_persistent_blocks = 10;
					k_stub->kconf.blocksize.x = 192;
					k_stub->kconf.gridsize.x  = 240;
					k_stub->total_tasks = k_stub->kconf.gridsize.x * 24;
					//k_stub->total_tasks = (64 * 1048576)/k_stub->kconf.blocksize.x + (((64 * 1048576)%k_stub->kconf.blocksize.x==0)?0:1);
					k_stub->kconf.coarsening = 1;
				}
				else{
					if (strcmp(device_name, "TITAN X (Pascal)") == 0) {
						k_stub->kconf.numSMs = 28;
						k_stub->kconf.max_persistent_blocks = 10;
						k_stub->kconf.blocksize.x = 192;
						k_stub->kconf.gridsize.x  = 240;
						k_stub->total_tasks = k_stub->kconf.gridsize.x * 24;
						//k_stub->total_tasks = (64 * 1048576)/k_stub->kconf.blocksize.x + (((64 * 1048576)%k_stub->kconf.blocksize.x==0)?0:1);
						k_stub->kconf.coarsening = 1;
					}
					else{
						printf("Error: Unknown device\n");
						return -1;
					}
				}
			}
		
			break;
		
		default:
			printf("Unknown kernel\n");
			return -1;
	}
	 
	// Allocate task support on CPU memory (pinned memory)
	checkCudaErrors(cudaHostAlloc((void **)&(k_stub->h_state), sizeof(State) * MAX_STREAMS_PER_KERNEL, cudaHostAllocDefault)); // In Pinned memory
	for (int i=0; i<MAX_STREAMS_PER_KERNEL; i++)
		k_stub->h_state[i] = PREP;
	
	checkCudaErrors(cudaHostAlloc((void **)&(k_stub->h_executed_tasks), sizeof(int), cudaHostAllocDefault)); // In Pinned memory
	checkCudaErrors(cudaHostAlloc((void **)&(k_stub->h_SMs_cont), sizeof(int)*k_stub->kconf.numSMs, cudaHostAllocDefault)); // In Pinned memory
	
	// Proxy support for zero-copy
	#ifdef ZEROCOPY

	// Zero-copy eviction state (host process indicates eviction to proxy)
	
	checkCudaErrors(cudaHostAlloc((void **)&(k_stub->h_proxy_eviction), sizeof(int), cudaHostAllocMapped));
	checkCudaErrors(cudaHostGetDevicePointer((void **)&(k_stub->d_proxy_eviction),  (void *)(k_stub->h_proxy_eviction) , 0));
	
	// Zero-copy: proxy send to host, when kernels finishes, the number of kernel excecuted tasks  
	checkCudaErrors(cudaHostAlloc((void **)&(k_stub->h_exec_tasks_proxy), sizeof(int), cudaHostAllocMapped));
	checkCudaErrors(cudaHostGetDevicePointer((void **)&(k_stub->d_exec_tasks_proxy),  (void *)(k_stub->h_exec_tasks_proxy) , 0));
	
	// Stream to launch proxy
	k_stub->proxy_s = (cudaStream_t *)malloc(sizeof(cudaStream_t));
	err = cudaStreamCreate(k_stub->proxy_s);
	checkCudaErrors(err);
	
	#endif

	// Allocate and initialize task support in device memory
	checkCudaErrors(cudaMalloc((void **)&k_stub->d_executed_tasks,  sizeof(int))); // Subtask counter: kernel use it to obtain id substak
	cudaMemset(k_stub->d_executed_tasks, 0, sizeof(int));
	
	checkCudaErrors(cudaMalloc((void **)&k_stub->gm_state,  sizeof(State) * MAX_STREAMS_PER_KERNEL)); //get the pointer to global memory position to communicate from CPU evicted state
	cudaMemcpy(k_stub->gm_state,  k_stub->h_state, sizeof(State) * MAX_STREAMS_PER_KERNEL, cudaMemcpyHostToDevice);
	
	checkCudaErrors(cudaMalloc((void **)&k_stub->d_SMs_cont, sizeof(int)*k_stub->kconf.numSMs)); // create an array (one position per SM) for SMK specific support
	cudaMemset(k_stub->d_SMs_cont, 0, sizeof(int)*k_stub->kconf.numSMs);
	
	*stub = k_stub;
	
	return 0;
}




