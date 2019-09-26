#include <unistd.h>
#include <string.h>
#include <helper_functions.h>   // helper functions for string parsing
#include <helper_cuda.h>   
#include "elastic_kernel.h"

// Initialize params of the architecture 6.0
int init_cc60(t_cc *cc)
{	
cc->TpW = 32; //Threads per Warp
cc->Max_WpSM = 64; //Max Warps per Multiprocessor
cc->Max_BpSM = 32; //Max Thread Blocks per Multiprocessor
cc->Max_TpSM = 2048; // Max Threads per Multiprocessor
cc->Max_TpB =	1024; //Maximum Thread Block Size
cc->RpSM = 65536; //Registers per Multiprocessor
cc->Max_RpB = 65536 ; //Max Registers per Thread Block
cc->Max_RpT = 255; //Max Registers per Thread
cc->Max_SmpSM = 65536; //Shared Memory per Multiprocessor (bytes)
cc->Max_SmpB = 49152; // Max Shared Memory per Block
cc->RallSize = 256; //Register allocation unit (warp) size
cc->SmAllsize =	256; //Shared Memory allocation unit size
cc->WAllG= 2; //Warp allocation granularity

return 0;
}

int init_cc61(t_cc *cc)
{	
cc->TpW = 32; //Threads per Warp
cc->Max_WpSM = 64; //Max Warps per Multiprocessor
cc->Max_BpSM = 32; //Max Thread Blocks per Multiprocessor
cc->Max_TpSM = 2048; // Max Threads per Multiprocessor
cc->Max_TpB =	1024; //Maximum Thread Block Size
cc->RpSM = 65536; //Registers per Multiprocessor
cc->Max_RpB = 65536 ; //Max Registers per Thread Block
cc->Max_RpT = 255; //Max Registers per Thread
cc->Max_SmpSM = 98304; //Shared Memory per Multiprocessor (bytes)
cc->Max_SmpB = 49152; // Max Shared Memory per Block
cc->RallSize = 256; //Register allocation unit (warp) size
cc->SmAllsize =	256; //Shared Memory allocation unit size
cc->WAllG= 4; //Warp allocation granularity

return 0;
}

int get_resources(int req_BpSM, int TpB, int RpT, int SmpB, t_used_res *ures, t_cc cc60)
{
	if (TpB > cc60.Max_TpB){
		printf ("Error: TpB too big\n");
		return -1;
	}
	
	if (RpT > cc60.Max_RpT){
		printf("Error: RpT too big\n");
		return -1;
	}
	
	if (SmpB > cc60.Max_SmpB){
		printf("Error: MaxSmpB too big\n");
		return -1;
	}
	
	if (TpB % cc60.TpW !=0){
		printf("Error: TpB must be a multiple of TpW\n");
		return -1;
	}
	
	// Calculate number of blocks based on TpB
	
	int BpSM_1 = (cc60.Max_TpSM-ures->used_TpSM) / TpB; 
	
	// Calculate number of blocks based on number of registers 
	
	int RpW_2 =  ceil ((float)(RpT * cc60.TpW)/(float)cc60.RallSize) * cc60.RallSize;
	int numwarps = TpB / cc60.TpW;
	int BpSM_2 = (cc60.RpSM - ures->used_RpSM)/(RpW_2*numwarps);
	
	// Caclulate number of blocks based on Shared Memory
	
	int BpSM_3;
	if (SmpB == 0)
		BpSM_3 = BpSM_2 + 1; // Thus, BpSM_3 is not the minimum
	else
		BpSM_3 = (cc60.Max_SmpSM - ures->used_SmpSM)/SmpB;
	
	// Minimum BpSM
	
	int minBpSM;
	
	if (BpSM_1 < BpSM_2)
		minBpSM = BpSM_1;
	else
		minBpSM = BpSM_1;
	
	if (BpSM_3 < minBpSM)
		minBpSM = BpSM_3;
	
	
	// Check max number of blocks per SM
	if ( minBpSM + ures->used_BpSM > cc60.Max_BpSM)
		minBpSM = cc60.Max_BpSM - ures->used_BpSM ;
	
	// Check max number of warps per SM
	if ( minBpSM * numwarps + ures->used_WpSM > cc60.Max_WpSM)
		minBpSM = (cc60.Max_WpSM - ures->used_WpSM)/numwarps;
	
	if (req_BpSM > minBpSM) {
		printf("Error:vToo many blocks per SM requested\n");
		return -1;
	}
	
	// Add used resources
	
	ures->used_TpSM += req_BpSM * TpB;
	ures->used_RpSM += req_BpSM * RpW_2 * numwarps;
	ures->used_SmpSM += req_BpSM * SmpB;
	ures->used_BpSM += req_BpSM;
	ures->used_WpSM += req_BpSM * numwarps;
	
	return 0 ;
}
	
// Assuming a number of resources are being used by a kernel in a SM, this function calculates the maximum
// number of active CTAs that can allocate in that SM.
int get_max_resources(int TpB, int RpT, int SmpB, t_used_res *ures, t_cc cc60, int *BpSM)
{
	if (TpB > cc60.Max_TpB){
		printf ("Error: TpB too big\n");
		return -1;
	}
	
	if (RpT > cc60.Max_RpT){
		printf("Error: RpT too big\n");
		return -1;
	}
	
	if (SmpB > cc60.Max_SmpB){
		printf("Error: MaxSmpB too big\n");
		return -1;
	}
	
	if (TpB % cc60.TpW !=0){
		printf("Error: TpB must be a multiple of TpW\n");
		return -1;
	}
	
	// Calculate number of blocks based on TpB
	
	int BpSM_1 = (cc60.Max_TpSM-ures->used_TpSM) / TpB; 
	
	// Calculate number of blocks based on number of registers 
	
	int RpW_2 =  ceil ((float)(RpT * cc60.TpW)/(float)cc60.RallSize) * cc60.RallSize;
	int numwarps = TpB / cc60.TpW;
	int BpSM_2 = (cc60.RpSM - ures->used_RpSM)/(RpW_2*numwarps);
	
	// Caclulate number of blocks based on Shared Memory
	
	int BpSM_3;
	if (SmpB == 0)
		BpSM_3 = BpSM_2 + 1; // Thus, BpSM_3 is not the minimum
	else
		BpSM_3 = (cc60.Max_SmpSM - ures->used_SmpSM)/SmpB;
	
	// Minimum BpSM
	
	int minBpSM;
	
	if (BpSM_1 < BpSM_2)
		minBpSM = BpSM_1;
	else
		minBpSM = BpSM_1;
	
	if (BpSM_3 < minBpSM)
		minBpSM = BpSM_3;
	
	
	// Check max number of blocks per SM
	if ( minBpSM + ures->used_BpSM > cc60.Max_BpSM)
		minBpSM = cc60.Max_BpSM - ures->used_BpSM ;
	
	// Check max number of warps per SM
	if ( minBpSM * numwarps + ures->used_WpSM > cc60.Max_WpSM)
		minBpSM = (cc60.Max_WpSM - ures->used_WpSM)/numwarps;
	
	// Add used resources
	
	ures->used_TpSM += minBpSM * TpB;
	ures->used_RpSM += minBpSM * RpW_2 * numwarps;
	ures->used_SmpSM += minBpSM * SmpB;
	ures->used_BpSM += minBpSM;
	ures->used_WpSM += minBpSM * numwarps;
	
	*BpSM = minBpSM;
	
	return 0;
}

// Use resources: res[0] -> threads per block, res[1]->registers_per_thread, res[2]->bytes of SM per block
int get_kernel_use(t_Kernel kid, int *res)
{
	
	// Kernels are compiled for cc 6.1
	switch (kid) {
		case SPMV_CSRscalar: 
			res[0] = 128; res[1]=32; res[2]= 4;
			break;
		case VA: 
			res[0] = 256; res[1]=21; res[2]= 4;
			break;
		case BS: 
			res[0] = 256; res[1]=26; res[2]= 4;
			break;
		case MM: 
			res[0] = 256; res[1]=31; res[2]= 2052;
			break;
		case HST256:
			res[0] = 192; res[1]=29; res[2]= 6004;
			break;
		case Reduction: 
			res[0] = 256; res[1]=17; res[2]= 1028;
			break;
		case PF:
			res[0] = 256; res[1]=21; res[2]= 2052;
			break;
		case GCEDD:
			res[0] = 256; res[1]=25; res[2]= 1296;
			break;
		case RCONV:
			res[0] = 64; res[1]=31; res[2]= 2564;
			break;
		case CCONV:
			res[0] = 128; res[1]=32; res[2]=5188;
			break;
		case SCEDD:
			res[0] = 256; res[1]=23; res[2]=1296;
			break;
		case HCEDD:
			res[0] = 256; res[1]=19; res[2]=16;
			break;
		case NCEDD:
			res[0] = 256; res[1]=24; res[2]=1296;
			break;
		default:
			printf("Error: Unknown kernel type\n");
			return -1;
	}
	
	return 0;
}


// Calculate all possibe combinations of CTAs for concurrent execution of two kernels 
int twokernels_occypancy(t_Kernel kid1, t_Kernel kid2)
{
	t_cc cc61;
	t_used_res *u;
	int BpSM;
    
	// Create and initialize structure 
	u = (t_used_res *)calloc(1, sizeof(t_used_res));
	init_cc61(&cc61);
	
	// Get maximum number of BpSM of first kernel
	char name1[50], name2[50];
	int res1[3], max_BpSM1;
	get_kernel_use(kid1, res1);
	get_max_resources(res1[0], res1[1], res1[2], u, cc61, &max_BpSM1);
	kid_from_index(kid1, name1);
	printf("%s Max_BpSM=%d\n",  name1, max_BpSM1);
	
	// Get use of second kernel
	int res2[3];
	get_kernel_use(kid2, res2);
	
	// For each possible numbe of blocks
	kid_from_index(kid1, name1);
	kid_from_index(kid2, name2);
	printf("%s\t\t%s\n", name1, name2);
	for (int i = 1; i<max_BpSM1; i++) {	
		
		memset(u, 0, sizeof(t_used_res));
		
		if (get_resources(i, res1[0], res1[1], res1[2], u, cc61) < 0 )
		return -1;

		get_max_resources(res2[0], res2[1], res2[2], u, cc61, &BpSM);
		
		printf("%d\t/\t%d \n", i, BpSM);
	}
	
	return 0;
}

int main (int argc, char **argv)
{
	t_Kernel kid[2];
	
	kid[0]=Reduction; 
	kid[1]=HST256; 
	twokernels_occypancy(kid[0], kid[1]);
	 
	return 0;
	
}


