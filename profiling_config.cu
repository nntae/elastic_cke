#include <unistd.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <helper_functions.h>   // helper functions for string parsing
#include <helper_cuda.h>   
#include "elastic_kernel.h"

t_coBlocks info_coBlocks[Number_of_Kernels-1][Number_of_Kernels-1];

t_solo info_solo[Number_of_Kernels-1];

t_coBlocks *fill_head(t_Kernel k1, t_Kernel k2, int num_configs)
{
	t_coBlocks *myinfo;
	
	myinfo = &info_coBlocks[k1][k2];
	myinfo->kid[0] = k1; myinfo->kid[1] = k2; 
	myinfo->num_configs = num_configs;
	
	myinfo->pairs = (int **) calloc(myinfo->num_configs, sizeof(int *));
	myinfo->tpms = (double **) calloc(myinfo->num_configs, sizeof(double *));
	for (int i=0; i < myinfo->num_configs; i++) {
		myinfo->pairs[i] = (int *)calloc(2, sizeof(int));
		myinfo->tpms[i] = (double *)calloc(2, sizeof(double));
	}
	
	return myinfo;
}

int reverse_values(t_coBlocks *info)
{
	t_coBlocks *new_info = fill_head(info->kid[1], info->kid[0], info->num_configs);
	
	for (int i=0; i < info->num_configs; i++) {
		new_info->pairs[i][0] = info->pairs[i][1];
		new_info->pairs[i][1] = info->pairs[i][0];
	}
		
	return 0;
}

int fill_coBlocks()
{
	
	memset (info_coBlocks, 0, (Number_of_Kernels-1) * (Number_of_Kernels-1) * sizeof(t_coBlocks));
	t_coBlocks *myinfo, *save_info;
	
	//MM-BS
	/*myinfo = &info_coBlocks[MM][BS];
	myinfo->kid[0] = MM; myinfo->kid[1] = BS; 
	myinfo->num_configs = 7;
	
	myinfo->pairs = (int **) calloc(myinfo->num_configs, sizeof(int *));
	for (int i=0; i < myinfo->num_configs; i++)
		myinfo->pairs[i] = (int *)calloc(2, sizeof(int));
	*/
	
	myinfo = fill_head(MM, BS, 7);

	myinfo->pairs[0][0] = 1; myinfo->pairs[0][1] = 7;
	myinfo->pairs[1][0] = 2; myinfo->pairs[1][1] = 6;
	myinfo->pairs[2][0] = 3; myinfo->pairs[2][1] = 5;
	myinfo->pairs[3][0] = 4; myinfo->pairs[3][1] = 4;
	myinfo->pairs[4][0] = 5; myinfo->pairs[4][1] = 3;
	myinfo->pairs[5][0] = 6; myinfo->pairs[5][1] = 2;
	myinfo->pairs[6][0] = 7; myinfo->pairs[6][1] = 1;
	
	reverse_values(myinfo);

	save_info = myinfo;
	
	//MM-VA
	
	myinfo = fill_head(MM, VA, 7);
	
	for (int i=0; i<7;i++)
		memcpy(myinfo->pairs[i], save_info->pairs[i], 2 * sizeof(int)); 
	
	reverse_values(myinfo);
	
	//MM-Reduction
	
	myinfo = fill_head(MM, Reduction, 7);
	
	for (int i=0; i<7;i++)
		memcpy(myinfo->pairs[i], save_info->pairs[i], 2 * sizeof(int)); 
	
	reverse_values(myinfo);

	//MM-PF
	
	myinfo = fill_head(MM, PF, 7);
	
	for (int i=0; i<7;i++)
		memcpy(myinfo->pairs[i], save_info->pairs[i], 2 * sizeof(int)); 
	reverse_values(myinfo);
	
	//MM-GCEDD
	
	myinfo = fill_head(MM, GCEDD, 7);
	
	for (int i=0; i<7;i++)
		memcpy(myinfo->pairs[i], save_info->pairs[i], 2 * sizeof(int)); 
	reverse_values(myinfo);
	
	//MM-SPMV_CSRscalar
	
	myinfo = fill_head(MM, SPMV_CSRscalar, 7);
	
	myinfo->pairs[0][0] = 1; myinfo->pairs[0][1] = 14;
	myinfo->pairs[1][0] = 2; myinfo->pairs[1][1] = 12;
	myinfo->pairs[2][0] = 3; myinfo->pairs[2][1] = 10;
	myinfo->pairs[3][0] = 4; myinfo->pairs[3][1] = 8;
	myinfo->pairs[4][0] = 5; myinfo->pairs[4][1] = 6;
	myinfo->pairs[5][0] = 6; myinfo->pairs[5][1] = 4;
	myinfo->pairs[6][0] = 7; myinfo->pairs[6][1] = 2;
	reverse_values(myinfo);
	
	//MM-HST256
	
	myinfo = fill_head(MM, HST256, 7);
	
	myinfo->pairs[0][0] = 1; myinfo->pairs[0][1] = 9;
	myinfo->pairs[1][0] = 2; myinfo->pairs[1][1] = 8;
	myinfo->pairs[2][0] = 3; myinfo->pairs[2][1] = 6;
	myinfo->pairs[3][0] = 4; myinfo->pairs[3][1] = 5;
	myinfo->pairs[4][0] = 5; myinfo->pairs[4][1] = 4;
	myinfo->pairs[5][0] = 6; myinfo->pairs[5][1] = 2;
	myinfo->pairs[6][0] = 7; myinfo->pairs[6][1] = 1;
	reverse_values(myinfo);
	
	//MM-RCONV
	
	myinfo = fill_head(MM, RCONV, 7);
	
	myinfo->pairs[0][0] = 1; myinfo->pairs[0][1] = 25;
	myinfo->pairs[1][0] = 2; myinfo->pairs[1][1] = 24;
	myinfo->pairs[2][0] = 3; myinfo->pairs[2][1] = 20;
	myinfo->pairs[3][0] = 4; myinfo->pairs[3][1] = 16;
	myinfo->pairs[4][0] = 5; myinfo->pairs[4][1] = 12;
	myinfo->pairs[5][0] = 6; myinfo->pairs[5][1] = 8;
	myinfo->pairs[6][0] = 7; myinfo->pairs[6][1] = 4;
	reverse_values(myinfo);
	
	/////////////////////////////////////////////////////////////////
	
	//BS-VA
	
	myinfo = fill_head(BS, VA, 7);
	
	for (int i=0; i<7;i++)
		memcpy(myinfo->pairs[i], save_info->pairs[i], 2 * sizeof(int)); 
	
	reverse_values(myinfo);
	
	//BS-Reduction
	
	myinfo = fill_head(BS, Reduction, 7);
	
	for (int i=0; i<7;i++)
		memcpy(myinfo->pairs[i], save_info->pairs[i], 2 * sizeof(int)); 
	reverse_values(myinfo);
	
	//BS-PF
	
	myinfo = fill_head(BS, PF, 7);
	
	for (int i=0; i<7;i++)
		memcpy(myinfo->pairs[i], save_info->pairs[i], 2 * sizeof(int)); 
	reverse_values(myinfo);
	
	//BS-GCEDD
	
	myinfo = fill_head(BS, GCEDD, 7);
	
	for (int i=0; i<7;i++)
		memcpy(myinfo->pairs[i], save_info->pairs[i], 2 * sizeof(int)); 
	reverse_values(myinfo);
	
	//BS-SPMV_CSRscalar
	
	myinfo = fill_head(BS, SPMV_CSRscalar, 7);
	
	myinfo->pairs[0][0] = 1; myinfo->pairs[0][1] = 14;
	myinfo->pairs[1][0] = 2; myinfo->pairs[1][1] = 12;
	myinfo->pairs[2][0] = 3; myinfo->pairs[2][1] = 10;
	myinfo->pairs[3][0] = 4; myinfo->pairs[3][1] = 8;
	myinfo->pairs[4][0] = 5; myinfo->pairs[4][1] = 6;
	myinfo->pairs[5][0] = 6; myinfo->pairs[5][1] = 4;
	myinfo->pairs[6][0] = 7; myinfo->pairs[6][1] = 2;
	reverse_values(myinfo);
	
	//BS-HST256
	
	myinfo = fill_head(BS, HST256, 7);
	
	myinfo->pairs[0][0] = 1; myinfo->pairs[0][1] = 9;
	myinfo->pairs[1][0] = 2; myinfo->pairs[1][1] = 8;
	myinfo->pairs[2][0] = 3; myinfo->pairs[2][1] = 6;
	myinfo->pairs[3][0] = 4; myinfo->pairs[3][1] = 5;
	myinfo->pairs[4][0] = 5; myinfo->pairs[4][1] = 4;
	myinfo->pairs[5][0] = 6; myinfo->pairs[5][1] = 2;
	myinfo->pairs[6][0] = 7; myinfo->pairs[6][1] = 1;
	reverse_values(myinfo);
	
	//BS-RCONV
	
	myinfo = fill_head(BS, RCONV, 7);
	
	myinfo->pairs[0][0] = 1; myinfo->pairs[0][1] = 26;
	myinfo->pairs[1][0] = 2; myinfo->pairs[1][1] = 24;
	myinfo->pairs[2][0] = 3; myinfo->pairs[2][1] = 20;
	myinfo->pairs[3][0] = 4; myinfo->pairs[3][1] = 16;
	myinfo->pairs[4][0] = 5; myinfo->pairs[4][1] = 12;
	myinfo->pairs[5][0] = 6; myinfo->pairs[5][1] = 8;
	myinfo->pairs[6][0] = 7; myinfo->pairs[6][1] = 4;
	reverse_values(myinfo);
	
	////////////////////////////////////////////////////////////////////////////////////////
	
	//VA-Reduction
	
	myinfo = fill_head(VA, Reduction, 7);
	
	for (int i=0; i<7;i++)
		memcpy(myinfo->pairs[i], save_info->pairs[i], 2 * sizeof(int)); 
	reverse_values(myinfo);
	
	//VA-PF
	
	myinfo = fill_head(VA, PF, 7);
	
	for (int i=0; i<7;i++)
		memcpy(myinfo->pairs[i], save_info->pairs[i], 2 * sizeof(int)); 
	reverse_values(myinfo);
	
	//VA-GCEDD
	
	myinfo = fill_head(VA, GCEDD, 7);
	
	for (int i=0; i<7;i++)
		memcpy(myinfo->pairs[i], save_info->pairs[i], 2 * sizeof(int)); 
	reverse_values(myinfo);
	
	//VA-SPMV_CSRscalar
	
	myinfo = fill_head(VA, SPMV_CSRscalar, 7);
	
	myinfo->pairs[0][0] = 1; myinfo->pairs[0][1] = 14;
	myinfo->pairs[1][0] = 2; myinfo->pairs[1][1] = 12;
	myinfo->pairs[2][0] = 3; myinfo->pairs[2][1] = 10;
	myinfo->pairs[3][0] = 4; myinfo->pairs[3][1] = 8;
	myinfo->pairs[4][0] = 5; myinfo->pairs[4][1] = 6;
	myinfo->pairs[5][0] = 6; myinfo->pairs[5][1] = 4;
	myinfo->pairs[6][0] = 7; myinfo->pairs[6][1] = 2;
	reverse_values(myinfo);
	
	//VA-HST256
	
	myinfo = fill_head(VA, HST256, 7);
	
	myinfo->pairs[0][0] = 1; myinfo->pairs[0][1] = 9;
	myinfo->pairs[1][0] = 2; myinfo->pairs[1][1] = 8;
	myinfo->pairs[2][0] = 3; myinfo->pairs[2][1] = 6;
	myinfo->pairs[3][0] = 4; myinfo->pairs[3][1] = 5;
	myinfo->pairs[4][0] = 5; myinfo->pairs[4][1] = 4;
	myinfo->pairs[5][0] = 6; myinfo->pairs[5][1] = 2;
	myinfo->pairs[6][0] = 7; myinfo->pairs[6][1] = 1;
	reverse_values(myinfo);
	
	//VA-RCONV
	
	myinfo = fill_head(VA, RCONV, 7);
	
	myinfo->pairs[0][0] = 1; myinfo->pairs[0][1] = 26;
	myinfo->pairs[1][0] = 2; myinfo->pairs[1][1] = 24;
	myinfo->pairs[2][0] = 3; myinfo->pairs[2][1] = 20;
	myinfo->pairs[3][0] = 4; myinfo->pairs[3][1] = 16;
	myinfo->pairs[4][0] = 5; myinfo->pairs[4][1] = 12;
	myinfo->pairs[5][0] = 6; myinfo->pairs[5][1] = 8;
	myinfo->pairs[6][0] = 7; myinfo->pairs[6][1] = 4;
	reverse_values(myinfo);
	
	//////////////////////////////////////////////////////////////
	
	//SPMV_CSRscalar -Reduction
	
	myinfo = fill_head(SPMV_CSRscalar, Reduction, 7);
	
	myinfo->pairs[0][0] = 2; myinfo->pairs[0][1] = 7;
	myinfo->pairs[1][0] = 4; myinfo->pairs[1][1] = 6;
	myinfo->pairs[2][0] = 6; myinfo->pairs[2][1] = 5;
	myinfo->pairs[3][0] = 8; myinfo->pairs[3][1] = 4;
	myinfo->pairs[4][0] = 10; myinfo->pairs[4][1] = 3;
	myinfo->pairs[5][0] = 12; myinfo->pairs[5][1] = 2;
	myinfo->pairs[6][0] = 14; myinfo->pairs[6][1] = 1;
	reverse_values(myinfo);
	
	save_info = myinfo; 
	
	//SPMV_CSRscalar - PF
	
	myinfo = fill_head(SPMV_CSRscalar, PF, 7);
	
	for (int i=0; i<7;i++)
		memcpy(myinfo->pairs[i], save_info->pairs[i], 2 * sizeof(int)); 
	reverse_values(myinfo);
	
	//SPMV_CSRscalar - GCEDD
	
	myinfo = fill_head(SPMV_CSRscalar, GCEDD, 7);
	
	for (int i=0; i<7;i++)
		memcpy(myinfo->pairs[i], save_info->pairs[i], 2 * sizeof(int)); 
	reverse_values(myinfo);
	
	//SPMV_CSRscalar - RCONV
	
	myinfo = fill_head(SPMV_CSRscalar, RCONV, 13);
	
	
	myinfo->pairs[0][0] = 3; myinfo->pairs[0][1] = 26;
	myinfo->pairs[1][0] = 4; myinfo->pairs[1][1] = 24;
	myinfo->pairs[2][0] = 5; myinfo->pairs[2][1] = 22;
	myinfo->pairs[3][0] = 6; myinfo->pairs[3][1] = 20;
	myinfo->pairs[4][0] = 7; myinfo->pairs[4][1] = 18;
	myinfo->pairs[5][0] = 8; myinfo->pairs[5][1] = 16;
	myinfo->pairs[6][0] = 9; myinfo->pairs[6][1] = 14;
	myinfo->pairs[7][0] = 10; myinfo->pairs[7][1] = 12;
	myinfo->pairs[8][0] = 11; myinfo->pairs[8][1] = 10;
	myinfo->pairs[9][0] = 12; myinfo->pairs[9][1] = 8;
	myinfo->pairs[10][0] = 13; myinfo->pairs[10][1] = 6;
	myinfo->pairs[11][0] = 14; myinfo->pairs[11][1] = 4;
	myinfo->pairs[12][0] = 15; myinfo->pairs[12][1] = 2;
	reverse_values(myinfo);
	
	//SPMV_CSRscalar - HST256
	
	myinfo = fill_head(SPMV_CSRscalar, HST256, 10);
	
	myinfo->pairs[0][0] = 1; myinfo->pairs[0][1] = 10;
	myinfo->pairs[1][0] = 2; myinfo->pairs[1][1] = 9;
	myinfo->pairs[2][0] = 4; myinfo->pairs[2][1] = 8;
	myinfo->pairs[3][0] = 5; myinfo->pairs[3][1] = 7;
	myinfo->pairs[4][0] = 7; myinfo->pairs[4][1] = 6;
	myinfo->pairs[5][0] = 8; myinfo->pairs[5][1] = 5;
	myinfo->pairs[6][0] = 10; myinfo->pairs[6][1] = 4;
	myinfo->pairs[7][0] = 11; myinfo->pairs[7][1] = 3;
	myinfo->pairs[8][0] = 12; myinfo->pairs[8][1] = 2;
	myinfo->pairs[9][0] = 14; myinfo->pairs[9][1] = 1;
	reverse_values(myinfo);
	
	////////////////////////////////////////////////////////////////
	
	//Reduction - PF
	
	myinfo = fill_head(Reduction, PF, 7);
	
	myinfo->pairs[0][0] = 1; myinfo->pairs[0][1] = 7;
	myinfo->pairs[1][0] = 2; myinfo->pairs[1][1] = 6;
	myinfo->pairs[2][0] = 3; myinfo->pairs[2][1] = 5;
	myinfo->pairs[3][0] = 4; myinfo->pairs[3][1] = 4;
	myinfo->pairs[4][0] = 5; myinfo->pairs[4][1] = 3;
	myinfo->pairs[5][0] = 6; myinfo->pairs[5][1] = 2;
	myinfo->pairs[6][0] = 7; myinfo->pairs[6][1] = 1;
	save_info = myinfo;
	reverse_values(myinfo);

	//Reduction - GCEDD
	
	myinfo = fill_head(Reduction, GCEDD, 7);
	
	for (int i=0; i<7;i++)
		memcpy(myinfo->pairs[i], save_info->pairs[i], 2 * sizeof(int)); 
	reverse_values(myinfo);

	// Reduction - RCONV 
	
	myinfo = fill_head(Reduction, RCONV, 7);
	
	myinfo->pairs[0][0] = 1; myinfo->pairs[0][1] = 26;
	myinfo->pairs[1][0] = 2; myinfo->pairs[1][1] = 24;
	myinfo->pairs[2][0] = 3; myinfo->pairs[2][1] = 20;
	myinfo->pairs[3][0] = 4; myinfo->pairs[3][1] = 16;
	myinfo->pairs[4][0] = 5; myinfo->pairs[4][1] = 12;
	myinfo->pairs[5][0] = 6; myinfo->pairs[5][1] = 8;
	myinfo->pairs[6][0] = 7; myinfo->pairs[6][1] = 4;
	reverse_values(myinfo);
	
	// Reduction -  HST256
	
	myinfo = fill_head(Reduction, HST256, 7);
	
	myinfo->pairs[0][0] = 1; myinfo->pairs[0][1] = 9;
	myinfo->pairs[1][0] = 2; myinfo->pairs[1][1] = 8;
	myinfo->pairs[2][0] = 3; myinfo->pairs[2][1] = 6;
	myinfo->pairs[3][0] = 4; myinfo->pairs[3][1] = 5;
	myinfo->pairs[4][0] = 5; myinfo->pairs[4][1] = 4;
	myinfo->pairs[5][0] = 6; myinfo->pairs[5][1] = 2;
	myinfo->pairs[6][0] = 7; myinfo->pairs[6][1] = 1;
	reverse_values(myinfo);
	
	////////////////////////////////////////////////////////////////////////////////
	
	// PF - GCEDD
	
	myinfo = fill_head(PF, GCEDD, 7);
	
	myinfo->pairs[0][0] = 1; myinfo->pairs[0][1] = 7;
	myinfo->pairs[1][0] = 2; myinfo->pairs[1][1] = 6;
	myinfo->pairs[2][0] = 3; myinfo->pairs[2][1] = 5;
	myinfo->pairs[3][0] = 4; myinfo->pairs[3][1] = 4;
	myinfo->pairs[4][0] = 5; myinfo->pairs[4][1] = 3;
	myinfo->pairs[5][0] = 6; myinfo->pairs[5][1] = 2;
	myinfo->pairs[6][0] = 7; myinfo->pairs[6][1] = 1;
	reverse_values(myinfo);
	
	// PF - RCONV
	
	myinfo = fill_head(PF, RCONV, 7);
	
	myinfo->pairs[0][0] = 1; myinfo->pairs[0][1] = 26;
	myinfo->pairs[1][0] = 2; myinfo->pairs[1][1] = 24;
	myinfo->pairs[2][0] = 3; myinfo->pairs[2][1] = 20;
	myinfo->pairs[3][0] = 4; myinfo->pairs[3][1] = 16;
	myinfo->pairs[4][0] = 5; myinfo->pairs[4][1] = 12;
	myinfo->pairs[5][0] = 6; myinfo->pairs[5][1] = 8;
	myinfo->pairs[6][0] = 7; myinfo->pairs[6][1] = 4;
	reverse_values(myinfo);
	
	// PF - HST256
	
	myinfo = fill_head(PF, HST256, 7);
	
	myinfo->pairs[0][0] = 1; myinfo->pairs[0][1] = 9;
	myinfo->pairs[1][0] = 2; myinfo->pairs[1][1] = 8;
	myinfo->pairs[2][0] = 3; myinfo->pairs[2][1] = 6;
	myinfo->pairs[3][0] = 4; myinfo->pairs[3][1] = 5;
	myinfo->pairs[4][0] = 5; myinfo->pairs[4][1] = 4;
	myinfo->pairs[5][0] = 6; myinfo->pairs[5][1] = 2;
	myinfo->pairs[6][0] = 7; myinfo->pairs[6][1] = 1;
	reverse_values(myinfo);
	
	//////////////////////////////////////////////////////////
	
	// RCONV-GCEDD
	
	myinfo = fill_head(RCONV, GCEDD, 7);
	
	myinfo->pairs[0][0] = 4; myinfo->pairs[0][1] = 7;
	myinfo->pairs[1][0] = 8; myinfo->pairs[1][1] = 6;
	myinfo->pairs[2][0] = 12; myinfo->pairs[2][1] = 5;
	myinfo->pairs[3][0] = 16; myinfo->pairs[3][1] = 4;
	myinfo->pairs[4][0] = 20; myinfo->pairs[4][1] = 3;
	myinfo->pairs[5][0] = 24; myinfo->pairs[5][1] = 2;
	myinfo->pairs[6][0] = 25; myinfo->pairs[6][1] = 1;
	reverse_values(myinfo);
	
	// RCONV - HST256
	
	myinfo = fill_head(RCONV, HST256, 10);
	
	myinfo->pairs[0][0] = 2; myinfo->pairs[0][1] = 10;
	myinfo->pairs[1][0] = 4; myinfo->pairs[1][1] = 9;
	myinfo->pairs[2][0] = 6; myinfo->pairs[2][1] = 8;
	myinfo->pairs[3][0] = 9; myinfo->pairs[3][1] = 7;
	myinfo->pairs[4][0] = 11; myinfo->pairs[4][1] = 6;
	myinfo->pairs[5][0] = 14; myinfo->pairs[5][1] = 5;
	myinfo->pairs[6][0] = 16; myinfo->pairs[6][1] = 4;
	myinfo->pairs[7][0] = 18; myinfo->pairs[7][1] = 3;
	myinfo->pairs[8][0] = 21; myinfo->pairs[8][1] = 2;
	myinfo->pairs[9][0] = 23; myinfo->pairs[9][1] = 1;
	
	reverse_values(myinfo);

	////////////////////////////////////////////////////////////
		
	// GCEDD - HST256
	
	myinfo = fill_head(GCEDD, HST256, 7);

	myinfo->pairs[0][0] = 1; myinfo->pairs[0][1] = 9;
	myinfo->pairs[1][0] = 2; myinfo->pairs[1][1] = 8;
	myinfo->pairs[2][0] = 3; myinfo->pairs[2][1] = 6;
	myinfo->pairs[3][0] = 4; myinfo->pairs[3][1] = 5;
	myinfo->pairs[4][0] = 5; myinfo->pairs[4][1] = 4;
	myinfo->pairs[5][0] = 6; myinfo->pairs[5][1] = 2;
	myinfo->pairs[6][0] = 7; myinfo->pairs[6][1] = 1;
	reverse_values(myinfo);
	
	return 0;
}

int fill_solo()
{
	
	info_solo[MM].num_configs=8; //max_nm_blocks_per_SM
	info_solo[MM].tpms = (double *)calloc(info_solo[MM].num_configs, sizeof(double));
	
	info_solo[BS].num_configs=8; //max_nm_blocks_per_SM
	info_solo[BS].tpms = (double *)calloc(info_solo[BS].num_configs, sizeof(double));
	
	info_solo[VA].num_configs=8; //max_nm_blocks_per_SM
	info_solo[VA].tpms = (double *)calloc(info_solo[VA].num_configs, sizeof(double));
	
	info_solo[SPMV_CSRscalar].num_configs=16; //max_nm_blocks_per_SM
	info_solo[SPMV_CSRscalar].tpms = (double *)calloc(info_solo[SPMV_CSRscalar].num_configs, sizeof(double));
	
	info_solo[Reduction].num_configs=8; //max_nm_blocks_per_SM
	info_solo[Reduction].tpms = (double *)calloc(info_solo[Reduction].num_configs, sizeof(double));
	
	info_solo[PF].num_configs=8; //max_nm_blocks_per_SM
	info_solo[PF].tpms = (double *)calloc(info_solo[PF].num_configs, sizeof(double));
	
	info_solo[RCONV].num_configs=25; //max_nm_blocks_per_SM
	info_solo[RCONV].tpms = (double *)calloc(info_solo[RCONV].num_configs, sizeof(double));
	
	info_solo[GCEDD].num_configs=8; //max_nm_blocks_per_SM
	info_solo[GCEDD].tpms = (double *)calloc(info_solo[GCEDD].num_configs, sizeof(double));
	
	info_solo[HST256].num_configs=10; //max_nm_blocks_per_SM
	info_solo[HST256].tpms = (double *)calloc(info_solo[HST256].num_configs, sizeof(double));
	
	return 0;
}
