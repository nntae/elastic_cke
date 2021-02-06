#include <unistd.h>
#include <string.h>
#include <helper_functions.h>   // helper functions for string parsing
#include <helper_cuda.h>   
#include "elastic_kernel.h"
#include "BS/BS.h"
#include "VA/VA.h"
#include "MM/MM.h"
//#include "RSC/RSC.h"
#include "SPMV/SPMV.h"
#include "PF/PF.h"
#include "Reduction/reduction.h"
#include "FDTD3d/FDTD3dGPU.h"
//#include "Dummy/Dummy.h"
#include "CONV/CONV.h"
#include "CEDD/CEDD.h"
#include "TP/TP.h"
#include "DXTC/DXTC.h"
#include "HST/HST256.h"

//#define DATA_SET_1

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
			t_BS_params *BS_params;
			BS_params = (t_BS_params *)calloc(1, sizeof(t_BS_params));
			k_stub->params = (void *)BS_params;
			
			k_stub->launchCKEkernel = launch_preemp_BS;
			k_stub->launchORIkernel = launch_orig_BS;
			k_stub->launchSLCkernel = launch_slc_BS;
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
						#ifdef DATA_SET_1
						k_stub->kconf.gridsize.x = 25 * k_stub->kconf.numSMs * k_stub->kconf.max_persistent_blocks;
						#else
						k_stub->kconf.gridsize.x = 50 * k_stub->kconf.numSMs * k_stub->kconf.max_persistent_blocks;
						#endif
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
			t_VA_params *VA_params;
			VA_params = (t_VA_params *)calloc(1, sizeof(t_VA_params));
			k_stub->params = (void *)VA_params;
		
			k_stub->launchCKEkernel = launch_preemp_VA;
			k_stub->launchORIkernel = launch_orig_VA;
			k_stub->launchSLCkernel = launch_slc_VA;
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
						k_stub->kconf.coarsening = 160; // Antes de slicing: 40
						#ifdef DATA_SET_1
						k_stub->kconf.gridsize.x = 12 * k_stub->kconf.numSMs * k_stub->kconf.max_persistent_blocks; // Antes de slicing 50 *
						#else
						k_stub->kconf.gridsize.x = 15 * k_stub->kconf.numSMs * k_stub->kconf.max_persistent_blocks; // Antes de slicing 60 *
						#endif
						k_stub->total_tasks = k_stub->kconf.gridsize.x;
						VA_params->gridDimX = k_stub->kconf.gridsize.x;
						
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
			
			#ifdef DATA_SET_1
			MM_params->Asize.x=4096;MM_params->Asize.y=4096;
			MM_params->Bsize.x=4096;MM_params->Bsize.y=4096;
			#else
			MM_params->Asize.x=2048;MM_params->Asize.y=2048;
			MM_params->Bsize.x=2048;MM_params->Bsize.y=2048;
			#endif
			
			k_stub->params = (void *)MM_params;
			
			k_stub->launchCKEkernel = launch_preemp_MM;
			k_stub->launchORIkernel = launch_orig_MM; 
			k_stub->launchSLCkernel = launch_slc_MM;
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
		
/*		
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
*/
			
		case SPMV_CSRscalar:
			t_SPMV_params *SPMV_params;
			SPMV_params = (t_SPMV_params *)calloc(1, sizeof(t_SPMV_params));
			k_stub->params = (void *)SPMV_params;
			
			k_stub->launchCKEkernel = launch_preemp_SPMVcsr;
			k_stub->launchORIkernel = launch_orig_SPMVcsr;
			k_stub->launchSLCkernel = launch_slc_SPMVcsr;
			k_stub->startKernel = SPMVcsr_start_kernel;
			k_stub->startMallocs = SPMVcsr_start_mallocs;
			k_stub->startTransfers = SPMVcsr_start_transfers;
			k_stub->endKernel = SPMVcsr_end_kernel;
		
			if (strcmp(device_name, "TITAN X (Pascal)") == 0) {
					k_stub->kconf.numSMs = 28;
					k_stub->kconf.max_persistent_blocks = 16;
					//k_stub->kconf.blocksize.x = 32;
					k_stub->kconf.blocksize.x = 128;
					k_stub->kconf.blocksize.y = 1;
					//->esto estaba antes k_stub->kconf.gridsize.x =  k_stub->kconf.numSMs * k_stub->kconf.max_persistent_blocks * k_stub->kconf.blocksize.x / 2;
					k_stub->kconf.gridsize.x = k_stub->kconf.numSMs * k_stub->kconf.max_persistent_blocks * k_stub->kconf.blocksize.x ;//One row per thread when all thread in original version
					k_stub->kconf.gridsize.y = 1; //Grid Linearization
					k_stub->kconf.coarsening = 10;
					k_stub->total_tasks = k_stub->kconf.gridsize.x  ; //* k_stub->kconf.coarsening;
					SPMV_params->gridDimX = k_stub->kconf.gridsize.x;
					
			}
			else{
				printf("Error: Unknown device\n");
				return -1;
			}
			
			// Esto estaba antes --> SPMV_params->numRows = k_stub->kconf.gridsize.x * k_stub->kconf.blocksize.x * k_stub->kconf.coarsening;
			SPMV_params->numRows = k_stub->total_tasks * k_stub->kconf.coarsening;
			
			#ifdef DATA_SET_1
			SPMV_params->nItems = (long int)SPMV_params->numRows * (long int)SPMV_params->numRows * 0.05; // 5% of entries will be non-zero
			#else
			SPMV_params->nItems = (long int)SPMV_params->numRows * (long int)SPMV_params->numRows * 0.00017; //	
		
			//--> ESto habia antes SPMV_params->nItems = SPMV_params->numRows * SPMV_params->numRows / 20;
			#endif
			
			SPMV_params->numNonZeroes = SPMV_params->nItems;
			
			break;

		case Reduction:
			t_reduction_params *reduction_params;
			reduction_params = (t_reduction_params *)calloc(1, sizeof(t_reduction_params));
			k_stub->params = (void *)reduction_params;
			
			k_stub->launchCKEkernel = launch_preemp_reduce;
			k_stub->launchORIkernel = launch_orig_reduce;
			k_stub->launchSLCkernel = launch_slc_reduce;
			k_stub->startKernel = reduce_start_kernel;
			k_stub->startMallocs = reduce_start_mallocs;
			k_stub->startTransfers = reduce_start_transfers;
			k_stub->endKernel = reduce_end_kernel;
			
			// reduction_params->size = 1<<24;
			// reduction_params->size *= 50;
			reduction_params->size =  802816000 / 2;
		
			if (strcmp(device_name, "TITAN X (Pascal)") == 0) {
					k_stub->kconf.numSMs = 28;
					k_stub->kconf.max_persistent_blocks = 8;
					k_stub->kconf.blocksize.x = 256;
					k_stub->kconf.blocksize.y = 1;
					
					reduction_params->gridDimX = k_stub->kconf.gridsize.x;
					//#ifdef DATA_SET_1
					//k_stub->kconf.gridsize.x =  64*28*8;
					//#else
					//k_stub->kconf.gridsize.x =  640*28*8; // 64 * number_of_permanent_blocks, a ver que tal
					//k_stub->kconf.gridsize.x =  64 * 7;
					//#endif

					k_stub->kconf.coarsening = 64;					
					k_stub->kconf.gridsize.x = reduction_params->size / (k_stub->kconf.blocksize.x * 2 * k_stub->kconf.coarsening);
					k_stub->kconf.gridsize.y = 1; //Grid Linearization
					reduction_params->gridDimX = k_stub->kconf.gridsize.x;
					k_stub->total_tasks =  k_stub->kconf.gridsize.x;
				
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
			
			#ifdef DATA_SET_1
			PF_params->nRows = 500;
			PF_params->nCols = 6000;
			#else
			PF_params->nRows = 500; 
			PF_params->nCols = 30000;
			#endif
			
			PF_params->param_pyramid_height = 126;
			
			k_stub->params = (void *)PF_params;
		
			k_stub->launchCKEkernel = launch_preemp_PF;
			k_stub->launchORIkernel = launch_orig_PF;
			k_stub->launchSLCkernel = launch_slc_PF;
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
			t_CONV_params *CONV_params;
			
			CONV_params = (t_CONV_params *)calloc(1, sizeof(t_CONV_params));

			k_stub->kconf.coarsening = 16;
			
			#ifdef DATA_SET_1
			CONV_params->conv_rows=6144;
			CONV_params->conv_cols=6144  * k_stub->kconf.coarsening; // 4 is the coarsening factor
 			#else 
			CONV_params->conv_rows=8192;//16384;
			CONV_params->conv_cols=2048 * k_stub->kconf.coarsening; //16384;
			#endif
			
			k_stub->params = (void *)CONV_params;
		
			k_stub->launchCKEkernel = launch_preemp_RCONV;
			k_stub->launchORIkernel = launch_orig_RCONV;
			k_stub->launchSLCkernel = launch_slc_RCONV;
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
				k_stub->kconf.gridsize.x = (CONV_params->conv_rows / (8 * 16)) * (CONV_params->conv_cols / 4);
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
					k_stub->kconf.gridsize.x = (CONV_params->conv_rows / (8 * 16)) * (CONV_params->conv_cols / 4);
					k_stub->kconf.gridsize.y = 1; //Grid Linearization
					k_stub->total_tasks = k_stub->kconf.gridsize.x;
					k_stub->kconf.coarsening = 1;
				}
				else{
					if (strcmp(device_name, "TITAN X (Pascal)") == 0) {
						//RCONV
						k_stub->kconf.numSMs = 28;
						k_stub->kconf.max_persistent_blocks = 32;
						k_stub->kconf.blocksize.x = 16;
						k_stub->kconf.blocksize.y = 4;
						/*
						CONV_params->gridDimX[0] = CONV_params->conv_cols / k_stub->kconf.blocksize.x;
						CONV_params->gridDimY[0] = CONV_params->conv_cols / k_stub->kconf.blocksize.y;
						k_stub->kconf.gridsize.x = (CONV_params->conv_rows / (8 * 16)) * (CONV_params->conv_cols / 4);
						k_stub->kconf.gridsize.y = 1; //Grid Linearization*/
						k_stub->kconf.gridsize.x = CONV_params->conv_cols / (8 * k_stub->kconf.blocksize.x * k_stub->kconf.coarsening );
						k_stub->kconf.gridsize.y = CONV_params->conv_rows / k_stub->kconf.blocksize.y;
						k_stub->total_tasks = (k_stub->kconf.gridsize.x * k_stub->kconf.gridsize.y)/k_stub->kconf.coarsening;
						CONV_params->gridDimX[0] = k_stub->kconf.gridsize.x;
						CONV_params->gridDimY[0] = k_stub->kconf.gridsize.y;
					}
					else{
						printf("Error: Unknown device\n");
						return -1;
					}
				}
			}
		
			break;
			
		case CCONV:
			/*t_CONV_params *CONV_params;
			
			CONV_params = (t_CONV_params *)calloc(1, sizeof(t_CONV_params));
			
			#ifdef DATA_SEnT_1
			CONV_params->conv_rows=6144;
			CONV_params->conv_cols=6144;
			#else
			CONV_params->conv_rows=18048;
			CONV_params->conv_cols=18048;
		    #endif*/
			
			k_stub->params = (void *)CONV_params;
		
			k_stub->launchCKEkernel = launch_preemp_CCONV;
			k_stub->launchORIkernel = launch_orig_CCONV;
			k_stub->launchSLCkernel = launch_slc_RCONV;
			k_stub->startKernel = CCONV_start_kernel;
			k_stub->startMallocs = CCONV_start_mallocs;
			k_stub->startTransfers = CCONV_start_transfers;
			k_stub->endKernel = CCONV_end_kernel;
			
			if (strcmp(device_name, "Tesla K20c") == 0) {		
				k_stub->kconf.numSMs = 13;
				k_stub->kconf.max_persistent_blocks = 9;
				k_stub->kconf.blocksize.x = 16;
				k_stub->kconf.blocksize.y = 8;
				k_stub->kconf.gridsize.x = (CONV_params->conv_rows / 16) * (CONV_params->conv_cols / (8 * 8));
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
					k_stub->kconf.gridsize.x = (CONV_params->conv_rows / 16) * (CONV_params->conv_cols / (8 * 8));
					k_stub->kconf.gridsize.y = 1; //Grid Linearization
					k_stub->total_tasks = k_stub->kconf.gridsize.x;
					k_stub->kconf.coarsening = 1;
				}
				else{
					if (strcmp(device_name, "TITAN X (Pascal)") == 0) {
						k_stub->kconf.numSMs = 28; 
						k_stub->kconf.max_persistent_blocks = 16;
						k_stub->kconf.blocksize.x = 16;
						k_stub->kconf.blocksize.y = 8;
						k_stub->kconf.coarsening = 8; 
						k_stub->kconf.gridsize.x = CONV_params->conv_cols / (8 * k_stub->kconf.blocksize.x * k_stub->kconf.coarsening);
						k_stub->kconf.gridsize.y = CONV_params->conv_rows / k_stub->kconf.blocksize.y;
						k_stub->total_tasks = k_stub->kconf.gridsize.x * k_stub->kconf.gridsize.y;
						CONV_params->gridDimX[1] = k_stub->kconf.gridsize.x;
						CONV_params->gridDimY[1] = k_stub->kconf.gridsize.y ;
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
			t_CEDD_params *CEDD_params;
			
			CEDD_params = (t_CEDD_params *)calloc(1, sizeof(t_CEDD_params));
			
			#ifdef DATA_SET_1
			CEDD_params->nRows=4608;
			CEDD_params->nCols=1200 * k_stub->kconf.coarsening;
			#else
			CEDD_params->nRows=4096;
			//CEDD_params->nCols=1200 * k_stub->kconf.coarsening;
			CEDD_params->nCols=4096;
			#endif
			
			k_stub->params = (void *)CEDD_params;
		
			k_stub->launchCKEkernel = launch_preemp_GCEDD;
			k_stub->launchORIkernel = launch_orig_GCEDD;
			k_stub->launchSLCkernel = launch_slc_GCEDD;
			k_stub->startKernel = GCEDD_start_kernel;
			k_stub->endKernel = GCEDD_end_kernel;

			k_stub->startMallocs = GCEDD_start_mallocs;
			k_stub->startTransfers = GCEDD_start_transfers;
			
			if (strcmp(device_name, "Tesla K20c") == 0) {			
				k_stub->kconf.numSMs = 13;
				k_stub->kconf.max_persistent_blocks = 8;
				k_stub->kconf.blocksize.x = 16;
				k_stub->kconf.blocksize.y = 16;
				k_stub->kconf.gridsize.x = CEDD_params->gridDimX * CEDD_params->gridDimY;
				k_stub->kconf.gridsize.y = 1; //Grid Linearization
				k_stub->total_tasks = k_stub->kconf.gridsize.x;
			}
			else {
				if (strcmp(device_name, "GeForce GTX 980") == 0) {
					k_stub->kconf.numSMs = 16;
					k_stub->kconf.max_persistent_blocks = 8;
					k_stub->kconf.blocksize.x = 16;
					k_stub->kconf.blocksize.y = 16;
					CEDD_params->gridDimX = (CEDD_params->nCols - 2)/k_stub->kconf.blocksize.x; // Add information loss during linearization
					CEDD_params->gridDimY = (CEDD_params->nRows - 2)/k_stub->kconf.blocksize.y;
					k_stub->kconf.gridsize.x = CEDD_params->gridDimX * CEDD_params->gridDimY;
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
						k_stub->kconf.coarsening = 16;
						//CEDD_params->gridDimX = (CEDD_params->nCols - 2)/(k_stub->kconf.blocksize.x *  k_stub->kconf.coarsening); // Add information loss during linearization
						CEDD_params->gridDimX = (CEDD_params->nCols-2)/(k_stub->kconf.blocksize.x * k_stub->kconf.coarsening);
						CEDD_params->gridDimY = (CEDD_params->nRows-2)/k_stub->kconf.blocksize.y;
						k_stub->kconf.gridsize.x = CEDD_params->gridDimX * CEDD_params->gridDimY;
						k_stub->kconf.gridsize.y = 1; //Grid Linearization
						//k_stub->total_tasks = k_stub->kconf.gridsize.x;
						k_stub->total_tasks = k_stub->kconf.gridsize.x;
					}
					else{
						printf("Error: Unknown device\n");
						return -1;
					}
				}
			}
		
			break;
			
		case SCEDD:
			// t_CEDD_params *SCEDD_params;
			
			// SCEDD_params = (t_CEDD_params *)calloc(1, sizeof(t_CEDD_params));
			
			// #ifdef DATA_SET_1
			// SCEDD_params->nRows=3072 * 2;
			// SCEDD_params->nCols=4608 * 2;
			// #else
			// SCEDD_params->nRows=4608 * 2.6;
			// SCEDD_params->nCols=4608 * 2.6;
			// #endif
			
			// *SCEDD_params->h_in_out = *GCEDD_params->h_in_out;
			// SCEDD_params->data_CEDD = GCEDD_params->data_CEDD;
			// SCEDD_params->out_CEDD = GCEDD_params->out_CEDD;
			// SCEDD_params->theta_CEDD = GCEDD_params->theta_CEDD;
			
			k_stub->params = (void *)CEDD_params;
		
			k_stub->launchCKEkernel = launch_preemp_SCEDD;
			k_stub->launchORIkernel = launch_orig_SCEDD;
			k_stub->launchORIkernel = launch_slc_SCEDD;
			k_stub->startKernel = SCEDD_start_kernel;
			k_stub->endKernel = SCEDD_end_kernel;

			k_stub->startMallocs = SCEDD_start_mallocs;
			k_stub->startTransfers = SCEDD_start_transfers;
			
			if (strcmp(device_name, "Tesla K20c") == 0) {			
				k_stub->kconf.numSMs = 13;
				k_stub->kconf.max_persistent_blocks = 8;
				k_stub->kconf.blocksize.x = 16;
				k_stub->kconf.blocksize.y = 16;
				CEDD_params->gridDimX = (CEDD_params->nCols - 2)/k_stub->kconf.blocksize.x; // Add information loss during linearization
				CEDD_params->gridDimY = (CEDD_params->nRows - 2)/k_stub->kconf.blocksize.y;
				k_stub->kconf.gridsize.x = (CEDD_params->gridDimX * CEDD_params->gridDimY) / 1;
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
					CEDD_params->gridDimX = (CEDD_params->nCols - 2)/k_stub->kconf.blocksize.x; // Add information loss during linearization
					CEDD_params->gridDimY = (CEDD_params->nRows - 2)/k_stub->kconf.blocksize.y;
					k_stub->kconf.gridsize.x = CEDD_params->gridDimX * CEDD_params->gridDimY;
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
						CEDD_params->gridDimX = (CEDD_params->nCols - 2)/k_stub->kconf.blocksize.x; // Add information loss during linearization
						CEDD_params->gridDimY = (CEDD_params->nRows - 2)/k_stub->kconf.blocksize.y;
						k_stub->kconf.gridsize.x = CEDD_params->gridDimX * CEDD_params->gridDimY;
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
			// t_CEDD_params *NCEDD_params;
			
			// NCEDD_params = (t_CEDD_params *)calloc(1, sizeof(t_CEDD_params));
			
			// #ifdef DATA_SET_1
			// NCEDD_params->nRows=3072 * 2;
			// NCEDD_params->nCols=4608 * 2;
			// #else
			// NCEDD_params->nRows=4608 * 2.6;
			// NCEDD_params->nCols=4608 * 2.6;
			// #endif
			
			// *NCEDD_params->h_in_out = *GCEDD_params->h_in_out;
			// NCEDD_params->data_CEDD = GCEDD_params->data_CEDD;
			// NCEDD_params->out_CEDD = GCEDD_params->out_CEDD;
			// NCEDD_params->theta_CEDD = GCEDD_params->theta_CEDD;
			
			k_stub->params = (void *)CEDD_params;
		
			k_stub->launchCKEkernel = launch_preemp_NCEDD;
			k_stub->launchORIkernel = launch_orig_NCEDD;
			k_stub->launchSLCkernel = launch_slc_NCEDD;
			k_stub->startKernel = NCEDD_start_kernel;
			k_stub->endKernel = NCEDD_end_kernel;

			k_stub->startMallocs = NCEDD_start_mallocs;
			k_stub->startTransfers = NCEDD_start_transfers;
			
			if (strcmp(device_name, "Tesla K20c") == 0) {			
				k_stub->kconf.numSMs = 13;
				k_stub->kconf.max_persistent_blocks = 8;
				k_stub->kconf.blocksize.x = 16;
				k_stub->kconf.blocksize.y = 16;
				CEDD_params->gridDimX = (CEDD_params->nCols - 2)/k_stub->kconf.blocksize.x; // Add information loss during linearization
				CEDD_params->gridDimY = (CEDD_params->nRows - 2)/k_stub->kconf.blocksize.y;
				k_stub->kconf.gridsize.x = CEDD_params->gridDimX * CEDD_params->gridDimY;
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
					CEDD_params->gridDimX = (CEDD_params->nCols - 2)/k_stub->kconf.blocksize.x; // Add information loss during linearization
					CEDD_params->gridDimY = (CEDD_params->nRows - 2)/k_stub->kconf.blocksize.y;
					k_stub->kconf.gridsize.x = CEDD_params->gridDimX * CEDD_params->gridDimY;
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
						CEDD_params->gridDimX = (CEDD_params->nCols - 2)/k_stub->kconf.blocksize.x; // Add information loss during linearization
						CEDD_params->gridDimY = (CEDD_params->nRows - 2)/k_stub->kconf.blocksize.y;
						k_stub->kconf.gridsize.x = CEDD_params->gridDimX * CEDD_params->gridDimY;
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
			// t_CEDD_params *HCEDD_params;
			
			// HCEDD_params = (t_CEDD_params *)calloc(1, sizeof(t_CEDD_params));
			
			// #ifdef DATA_SET_1
			// HCEDD_params->nRows=3072 * 2;
			// HCEDD_params->nCols=4608 * 2;
			// #else
			// HCEDD_params->nRows=4608 * 2.6;
			// HCEDD_params->nCols=4608 * 2.6;
			// #endif
			
			// *HCEDD_params->h_in_out = *GCEDD_params->h_in_out;
			// HCEDD_params->data_CEDD = GCEDD_params->data_CEDD;
			// HCEDD_params->out_CEDD = GCEDD_params->out_CEDD;
			// HCEDD_params->theta_CEDD = GCEDD_params->theta_CEDD;
			
			k_stub->params = (void *)CEDD_params;
		
			k_stub->launchCKEkernel = launch_preemp_HCEDD;
			k_stub->launchORIkernel = launch_orig_HCEDD;
			k_stub->launchSLCkernel = launch_slc_HCEDD;
			k_stub->startKernel = HCEDD_start_kernel;
			k_stub->endKernel = HCEDD_end_kernel;

			k_stub->startMallocs = HCEDD_start_mallocs;
			k_stub->startTransfers = HCEDD_start_transfers;
			
			if (strcmp(device_name, "Tesla K20c") == 0) {			
				k_stub->kconf.numSMs = 13;
				k_stub->kconf.max_persistent_blocks = 8;
				k_stub->kconf.blocksize.x = 16;
				k_stub->kconf.blocksize.y = 16;
				CEDD_params->gridDimX = (CEDD_params->nCols - 2)/k_stub->kconf.blocksize.x; // Add information loss during linearization
				CEDD_params->gridDimY = (CEDD_params->nRows - 2)/k_stub->kconf.blocksize.y;
				k_stub->kconf.gridsize.x = CEDD_params->gridDimX * CEDD_params->gridDimY;
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
					CEDD_params->gridDimX = (CEDD_params->nCols - 2)/k_stub->kconf.blocksize.x; // Add information loss during linearization
					CEDD_params->gridDimY = (CEDD_params->nRows - 2)/k_stub->kconf.blocksize.y;
					k_stub->kconf.gridsize.x = CEDD_params->gridDimX * CEDD_params->gridDimY;
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
						CEDD_params->gridDimX = (CEDD_params->nCols - 2)/k_stub->kconf.blocksize.x; // Add information loss during linearization
						CEDD_params->gridDimY = (CEDD_params->nRows - 2)/k_stub->kconf.blocksize.y;
						k_stub->kconf.gridsize.x = CEDD_params->gridDimX * CEDD_params->gridDimY;
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
			t_HST256_params *HST256_params;
			HST256_params = (t_HST256_params *)calloc(1, sizeof(t_HST256_params));
			k_stub->params = (void *)HST256_params;
	
			k_stub->launchCKEkernel = launch_preemp_HST256;
			k_stub->launchORIkernel = launch_orig_HST256;
			k_stub->launchSLCkernel = launch_slc_HST256;
			k_stub->startKernel = HST256_start_kernel;
			k_stub->endKernel = HST256_end_kernel;
			
			k_stub->startMallocs = HST256_start_mallocs;
			k_stub->startTransfers = HST256_start_transfers;
			
			if (strcmp(device_name, "Tesla K20c") == 0) {
				HST256_params->warp_count = 6;
				HST256_params->histogram256_threadblock_size = HST256_params->warp_count * WARP_SIZE;
				HST256_params->histogram256_threadblock_memory = HST256_params->warp_count * HISTOGRAM256_BIN_COUNT;
				
				#ifdef DATA_SET_1
				HST256_params->byteCount256 = 64 * 1048576 * HST256_params->warp_count;
				#else
				HST256_params->byteCount256 = 64 * 1048576 * HST256_params->warp_count;
				#endif
				
				k_stub->kconf.numSMs = 13;
				k_stub->kconf.max_persistent_blocks = 8;
				k_stub->kconf.blocksize.x = 192;
				k_stub->kconf.gridsize.x  = 240;
				k_stub->total_tasks = k_stub->kconf.gridsize.x;
				//k_stub->total_tasks = (64 * 1048576)/k_stub->kconf.blocksize.x + (((64 * 1048576)%k_stub->kconf.blocksize.x==0)?0:1);
				k_stub->kconf.coarsening = 1;
			}
			else {
				if (strcmp(device_name, "GeForce GTX 980") == 0) {
					HST256_params->warp_count = 6;
					HST256_params->histogram256_threadblock_size = HST256_params->warp_count * WARP_SIZE;
					HST256_params->histogram256_threadblock_memory = HST256_params->warp_count * HISTOGRAM256_BIN_COUNT;
					
					#ifdef DATA_SET_1
					HST256_params->byteCount256 = 64 * 1048576 * HST256_params->warp_count;
					#else
					HST256_params->byteCount256 = 64 * 1048576 * HST256_params->warp_count;
					#endif
				
					k_stub->kconf.numSMs = 16;
					k_stub->kconf.max_persistent_blocks = 10;
					k_stub->kconf.blocksize.x = 192;
					k_stub->kconf.gridsize.x  = 240;
					k_stub->total_tasks = k_stub->kconf.gridsize.x;
					//k_stub->total_tasks = (64 * 1048576)/k_stub->kconf.blocksize.x + (((64 * 1048576)%k_stub->kconf.blocksize.x==0)?0:1);
					k_stub->kconf.coarsening = 1;
				}
				else{
					if (strcmp(device_name, "TITAN X (Pascal)") == 0) {
						HST256_params->warp_count = 8;
						HST256_params->histogram256_threadblock_size = HST256_params->warp_count * WARP_SIZE;
						HST256_params->histogram256_threadblock_memory = HST256_params->warp_count * HISTOGRAM256_BIN_COUNT;
						
						#ifdef DATA_SET_1
						HST256_params->byteCount256 = 64 * 1089536 * 8;
						#else
						HST256_params->byteCount256 = 64 * 1089536; //* 8 *2;
						#endif
						
						k_stub->kconf.numSMs = 28;
						k_stub->kconf.max_persistent_blocks = 8;
						k_stub->kconf.blocksize.x = 256;
						k_stub->kconf.coarsening = 128; // Ojo coarsening tienen que ser 1 para prubas con slicing (cCUDA)
						//k_stub->kconf.gridsize.x  = HST256_params->byteCount256 / (sizeof(uint) * k_stub->kconf.coarsening * k_stub->kconf.blocksize.x);
						k_stub->kconf.gridsize.x  = k_stub->kconf.numSMs * k_stub->kconf.max_persistent_blocks;
						HST256_params->gridDimX = k_stub->kconf.gridsize.x;
						//k_stub->total_tasks = k_stub->kconf.gridsize.x;
						k_stub->total_tasks = HST256_params->byteCount256 / (sizeof(uint) * k_stub->kconf.blocksize.x * k_stub->kconf.coarsening);
						// k_stub->total_tasks = (k_stub->kconf.gridsize.x * ((HST256_params->byteCount256 / sizeof(uint)) / (k_stub->kconf.blocksize.x * k_stub->kconf.gridsize.x))) / k_stub->kconf.coarsening;
						//k_stub->total_tasks = (64 * 1048576)/k_stub->kconf.blocksize.x + (((64 * 1048576)%k_stub->kconf.blocksize.x==0)?0:1);
					}
					else{
						printf("Error: Unknown device\n");
						return -1;
					}
				}
			}
			break;
		
		case TP:
			t_TP_params *TP_params;
			TP_params = (t_TP_params *)calloc(1, sizeof(t_TP_params));
			TP_params->sSDKsample = "Transpose";
			TP_params->matrix_size_x = 6144; // Multiplo de 512
			TP_params->matrix_size_y = 6144; // Multiplo de 512
			TP_params->tile_dim = 16;
			TP_params->block_rows = 16;
			TP_params->mul_factor = TP_params->tile_dim;
			TP_params->max_tiles = (FLOOR(TP_params->matrix_size_x,512) * FLOOR(TP_params->matrix_size_y,512)) / (TP_params->tile_dim * TP_params->tile_dim);
			k_stub->params = (void *)TP_params;
			// TP_params->num_reps = 100;
	
			k_stub->launchORIkernel = launch_orig_TP;
			k_stub->launchSLCkernel = launch_slc_TP;
			//k_stub->startKernel = TP_start_kernel;
			k_stub->endKernel = TP_end_kernel;
			k_stub->startMallocs = TP_start_mallocs;
			k_stub->startTransfers = TP_start_transfers;
			
			if (strcmp(device_name, "Tesla K20c") == 0) {
				k_stub->kconf.numSMs = 15;
				k_stub->kconf.max_persistent_blocks = 16;
				// k_stub->kconf.blocksize.x = TP_params->tile_dim; // Calculados en launch
				// k_stub->kconf.blocksize.y = TP_params->block_rows; // Calculados en launch
				//k_stub->kconf.gridsize.x // Calculados en launch
				//k_stub->kconf.gridsize.y // Calculados en launch
				// k_stub->total_tasks = k_stub->kconf.gridsize.x;
				// k_stub->kconf.coarsening = _;
			}
			else{
				printf("Error: Unknown device\n");
				return -1;
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

// Create stub info but alspo pass a t_params sturncture pointer from a previous kstub: It is used for kstubs fron an applications (several kernels) 
int create_stubinfo_with_params(t_kernel_stub **stub, int deviceId, t_Kernel id, cudaStream_t *transfer_s, cudaStream_t *preemp_s, void *params)
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
		
		case CCONV:
		{
			t_CONV_params *CONV_params = (t_CONV_params *)params;
			
			CONV_params->gridDimY[1] = CONV_params->conv_cols /  (8 * 8);
			k_stub->params = (void *)CONV_params;
		
			k_stub->launchCKEkernel = launch_preemp_CCONV;
			k_stub->launchORIkernel = launch_orig_CCONV;
			k_stub->launchSLCkernel = launch_slc_CCONV;
			k_stub->startKernel = CCONV_start_kernel;
			k_stub->startMallocs = CCONV_start_mallocs;
			k_stub->startTransfers = CCONV_start_transfers;
			k_stub->endKernel = CCONV_end_kernel;
			
			if (strcmp(device_name, "Tesla K20c") == 0) {		
				k_stub->kconf.numSMs = 12;
				k_stub->kconf.max_persistent_blocks = 9;
				k_stub->kconf.blocksize.x = 16;
				k_stub->kconf.blocksize.y = 8;
				k_stub->kconf.gridsize.x = (CONV_params->conv_rows / 16) * (CONV_params->conv_cols / (8 * 8));
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
					k_stub->kconf.gridsize.x = (CONV_params->conv_rows / 16) * (CONV_params->conv_cols / (8 * 8));
					k_stub->kconf.gridsize.y = 1; //Grid Linearization
					k_stub->total_tasks = k_stub->kconf.gridsize.x;
					k_stub->kconf.coarsening = 1;
				}
				else{
					if (strcmp(device_name, "TITAN X (Pascal)") == 0) {
						k_stub->kconf.numSMs = 28; 
						k_stub->kconf.max_persistent_blocks = 16;
						k_stub->kconf.blocksize.x = 16;
						k_stub->kconf.blocksize.y = 8;
						k_stub->kconf.coarsening = 4; 
						k_stub->kconf.gridsize.x = CONV_params->conv_cols / (k_stub->kconf.blocksize.x);
						k_stub->kconf.gridsize.y = CONV_params->conv_rows / ( 8 * k_stub->kconf.blocksize.y) ;
						k_stub->total_tasks = k_stub->kconf.gridsize.x * k_stub->kconf.gridsize.y / k_stub->kconf.coarsening;
						CONV_params->gridDimX[1] = k_stub->kconf.gridsize.x;
						CONV_params->gridDimY[1] = k_stub->kconf.gridsize.y ;
					}
					else{
						printf("Error: Unknown device\n");
						return -1;
					}
				}
			}
		}
		break;
		
		case SCEDD:
			// t_CEDD_params *SCEDD_params;
			
			// SCEDD_params = (t_CEDD_params *)calloc(1, sizeof(t_CEDD_params));
			
			// #ifdef DATA_SET_1
			// SCEDD_params->nRows=3072 * 2;
			// SCEDD_params->nCols=4608 * 2;
			// #else
			// SCEDD_params->nRows=4608 * 2.6;
			// SCEDD_params->nCols=4608 * 2.6;
			// #endif
			
			// *SCEDD_params->h_in_out = *GCEDD_params->h_in_out;
			// SCEDD_params->data_CEDD = GCEDD_params->data_CEDD;
			// SCEDD_params->out_CEDD = GCEDD_params->out_CEDD;
			// SCEDD_params->theta_CEDD = GCEDD_params->theta_CEDD;
			{
			
			t_CEDD_params *CEDD_params = (t_CEDD_params *)params;
			k_stub->params = params;
		
			k_stub->launchCKEkernel = launch_preemp_SCEDD;
			k_stub->launchORIkernel = launch_orig_SCEDD;
			k_stub->launchSLCkernel = launch_slc_SCEDD;
			k_stub->startKernel = SCEDD_start_kernel;
			k_stub->endKernel = SCEDD_end_kernel;

			k_stub->startMallocs = SCEDD_start_mallocs;
			k_stub->startTransfers = SCEDD_start_transfers;
			
			if (strcmp(device_name, "Tesla K20c") == 0) {			
				k_stub->kconf.numSMs = 13;
				k_stub->kconf.max_persistent_blocks = 8;
				k_stub->kconf.blocksize.x = 16;
				k_stub->kconf.blocksize.y = 16;
				CEDD_params->gridDimX = (CEDD_params->nCols - 2)/k_stub->kconf.blocksize.x; // Add information loss during linearization
				CEDD_params->gridDimY = (CEDD_params->nRows - 2)/k_stub->kconf.blocksize.y;
				k_stub->kconf.gridsize.x = (CEDD_params->gridDimX * CEDD_params->gridDimY) / 1;
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
					CEDD_params->gridDimX = (CEDD_params->nCols - 2)/k_stub->kconf.blocksize.x; // Add information loss during linearization
					CEDD_params->gridDimY = (CEDD_params->nRows - 2)/k_stub->kconf.blocksize.y;
					k_stub->kconf.gridsize.x = CEDD_params->gridDimX * CEDD_params->gridDimY;
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
						k_stub->kconf.coarsening = 16;
						//CEDD_params->gridDimX = (CEDD_params->nCols - 2)/(k_stub->kconf.blocksize.x *  k_stub->kconf.coarsening); // Add information loss during linearization
						CEDD_params->gridDimX = (CEDD_params->nCols-2)/(k_stub->kconf.blocksize.x * k_stub->kconf.coarsening);
						CEDD_params->gridDimY = (CEDD_params->nRows-2)/k_stub->kconf.blocksize.y;
						k_stub->kconf.gridsize.x = CEDD_params->gridDimX * CEDD_params->gridDimY;
						k_stub->kconf.gridsize.y = 1; //Grid Linearization
						//k_stub->total_tasks = k_stub->kconf.gridsize.x;
						k_stub->total_tasks = k_stub->kconf.gridsize.x;
					}
					else{
						printf("Error: Unknown device\n");
						return -1;
					}
				}
			}
			}
		
			break;
			
		case NCEDD:
			// t_CEDD_params *NCEDD_params;
			
			// NCEDD_params = (t_CEDD_params *)calloc(1, sizeof(t_CEDD_params));
			
			// #ifdef DATA_SET_1
			// NCEDD_params->nRows=3072 * 2;
			// NCEDD_params->nCols=4608 * 2;
			// #else
			// NCEDD_params->nRows=4608 * 2.6;
			// NCEDD_params->nCols=4608 * 2.6;
			// #endif
			
			// *NCEDD_params->h_in_out = *GCEDD_params->h_in_out;
			// NCEDD_params->data_CEDD = GCEDD_params->data_CEDD;
			// NCEDD_params->out_CEDD = GCEDD_params->out_CEDD;
			// NCEDD_params->theta_CEDD = GCEDD_params->theta_CEDD;
			{
			t_CEDD_params *CEDD_params = (t_CEDD_params *)params;

			k_stub->params = params;
		
			k_stub->launchCKEkernel = launch_preemp_NCEDD;
			k_stub->launchORIkernel = launch_orig_NCEDD;
			k_stub->launchSLCkernel = launch_slc_NCEDD;
			k_stub->startKernel = NCEDD_start_kernel;
			k_stub->endKernel = NCEDD_end_kernel;

			k_stub->startMallocs = NCEDD_start_mallocs;
			k_stub->startTransfers = NCEDD_start_transfers;
			
			if (strcmp(device_name, "Tesla K20c") == 0) {			
				k_stub->kconf.numSMs = 13;
				k_stub->kconf.max_persistent_blocks = 8;
				k_stub->kconf.blocksize.x = 16;
				k_stub->kconf.blocksize.y = 16;
				CEDD_params->gridDimX = (CEDD_params->nCols - 2)/k_stub->kconf.blocksize.x; // Add information loss during linearization
				CEDD_params->gridDimY = (CEDD_params->nRows - 2)/k_stub->kconf.blocksize.y;
				k_stub->kconf.gridsize.x = CEDD_params->gridDimX * CEDD_params->gridDimY;
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
					CEDD_params->gridDimX = (CEDD_params->nCols - 2)/k_stub->kconf.blocksize.x; // Add information loss during linearization
					CEDD_params->gridDimY = (CEDD_params->nRows - 2)/k_stub->kconf.blocksize.y;
					k_stub->kconf.gridsize.x = CEDD_params->gridDimX * CEDD_params->gridDimY;
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
						k_stub->kconf.coarsening = 16;
						//CEDD_params->gridDimX = (CEDD_params->nCols - 2)/(k_stub->kconf.blocksize.x *  k_stub->kconf.coarsening);
						CEDD_params->gridDimX = (CEDD_params->nCols - 2)/(k_stub->kconf.blocksize.x * k_stub->kconf.coarsening);
						CEDD_params->gridDimY = (CEDD_params->nRows - 2)/k_stub->kconf.blocksize.y;
						k_stub->kconf.gridsize.x = CEDD_params->gridDimX * CEDD_params->gridDimY;
						k_stub->kconf.gridsize.y = 1; //Grid Linearization
						k_stub->total_tasks = k_stub->kconf.gridsize.x;
					}
					else{
						printf("Error: Unknown device\n");
						return -1;
					}
				}
			}
			}
		
			break;
			
		case HCEDD:
			// t_CEDD_params *HCEDD_params;
			
			// HCEDD_params = (t_CEDD_params *)calloc(1, sizeof(t_CEDD_params));
			
			// #ifdef DATA_SET_1
			// HCEDD_params->nRows=3072 * 2;
			// HCEDD_params->nCols=4608 * 2;
			// #else
			// HCEDD_params->nRows=4608 * 2.6;
			// HCEDD_params->nCols=4608 * 2.6;
			// #endif
			
			// *HCEDD_params->h_in_out = *GCEDD_params->h_in_out;
			// HCEDD_params->data_CEDD = GCEDD_params->data_CEDD;
			// HCEDD_params->out_CEDD = GCEDD_params->out_CEDD;
			// HCEDD_params->theta_CEDD = GCEDD_params->theta_CEDD;
			
			{
			t_CEDD_params *CEDD_params = (t_CEDD_params *)params;
			k_stub->params = params;
		
			k_stub->launchCKEkernel = launch_preemp_HCEDD;
			k_stub->launchORIkernel = launch_orig_HCEDD;
			k_stub->launchSLCkernel = launch_slc_HCEDD;
			k_stub->startKernel = HCEDD_start_kernel;
			k_stub->endKernel = HCEDD_end_kernel;

			k_stub->startMallocs = HCEDD_start_mallocs;
			k_stub->startTransfers = HCEDD_start_transfers;
			
			if (strcmp(device_name, "Tesla K20c") == 0) {			
				k_stub->kconf.numSMs = 13;
				k_stub->kconf.max_persistent_blocks = 8;
				k_stub->kconf.blocksize.x = 16;
				k_stub->kconf.blocksize.y = 16;
				CEDD_params->gridDimX = (CEDD_params->nCols - 2)/k_stub->kconf.blocksize.x; // Add information loss during linearization
				CEDD_params->gridDimY = (CEDD_params->nRows - 2)/k_stub->kconf.blocksize.y;
				k_stub->kconf.gridsize.x = CEDD_params->gridDimX * CEDD_params->gridDimY;
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
					CEDD_params->gridDimX = (CEDD_params->nCols - 2)/k_stub->kconf.blocksize.x; // Add information loss during linearization
					CEDD_params->gridDimY = (CEDD_params->nRows - 2)/k_stub->kconf.blocksize.y;
					k_stub->kconf.gridsize.x = CEDD_params->gridDimX * CEDD_params->gridDimY;
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
						k_stub->kconf.coarsening = 16;
						//CEDD_params->gridDimX = (CEDD_params->nCols - 2)/(k_stub->kconf.blocksize.x *  k_stub->kconf.coarsening);
						CEDD_params->gridDimX = (CEDD_params->nCols - 2)/(k_stub->kconf.blocksize.x * k_stub->kconf.coarsening);
						CEDD_params->gridDimY = (CEDD_params->nRows - 2)/k_stub->kconf.blocksize.y;
						k_stub->kconf.gridsize.x = CEDD_params->gridDimX * CEDD_params->gridDimY;
						k_stub->kconf.gridsize.y = 1; //Grid Linearization
						k_stub->total_tasks = k_stub->kconf.gridsize.x;
					}
					else{
						printf("Error: Unknown device\n");
						return -1;
					}
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
	
	 // Allocate and initialize memory address calculation support in CP     U memory
    k_stub->num_addr_counters = 2;
	checkCudaErrors(cudaHostAlloc((void **)&(k_stub->h_numUniqueAddr),      k_stub->num_addr_counters * sizeof(int), cudaHostAllocDefault)); // In Pinn     ed memory

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
