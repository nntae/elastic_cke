#include <stdio.h>
#include "elastic_kernel.h"

t_solo_performance *sperf;

/*int get_index(t_Kernel kid, int *index)
{
	switch (kid)
	{
		case MM:
			*index = 0;
			break;
			
		case BS:
			*index = 1;
			break;
			
		case VA:
			*index = 2;
			break;
			
		case Reduction:
			*index = 3;
			break;
			
		case PF:
			*index = 4;
			break;
			
		case GCEDD:
			*index = 5;
			break;
			
		case SPMV_CSRscalar:
			*index = 6;
			break;
			
		case HST256:
			*index = 7;
			break;
			
		case RCONV:
			*index = 8;
			break;
			
		case CCONV:
			*index = 9;
			break;
			
		case SCEDD:
			*index = 10;
			break;
			
		case NCEDD:
			*index = 11;
			break;
			
		case HCEDD:
			*index = 12;
			break;
			
		default:
			printf("kid not supported\n");
			return -1;
	}
	
	return 0;
}*/

int kid_from_index(int index, char *skid)
{
	switch (index)
	{
		case 0:
			strcpy(skid, "MM");
			break;
			
		case 1:
			strcpy(skid, "BS");
			break;
			
		case 2:
			strcpy(skid, "VA");
			break;
			
		case 6:
			strcpy(skid, "Reduction");
			break;
			
		case 7:
			strcpy(skid, "PF");
			break;
			
		case 12:
			strcpy(skid, "GCEDD");
			break;
			
		case 5:
			strcpy(skid, "SPMV_CSRscalar");
			break;
			
		case 16:
			strcpy(skid, "HST256");
			break;
			
		case 8:
			strcpy(skid, "RCONV");
			break;
			
		case 9:
			strcpy(skid, "CCONV");
			break;
			
		case 13:
			strcpy(skid, "SCEDD");
			break;
			
		case 14:
			strcpy(skid, "NCEDD");
			break;
			
		case 15:
			strcpy(skid, "HCEDD");
			break;

		case 17:
			strcpy(skid, "TP");
			break;
			
		case 18:
			strcpy(skid, "DXTC");
			break;

		case 19:
			strcpy(skid, "EMPTY");
			break;
			
		default:
			strcpy(skid, "UNKNOWN");
			printf("kid not supported\n");
			return -1;
	}
	
	return 0;
}

t_Kernel kid_from_name(char *name)
{
	int kid = -1;
	
	if (strcmp(name, "MM") == 0){
		kid = MM;
	}
	
	if (strcmp(name, "BS") == 0){
		kid = BS;
	}
		
	if (strcmp(name, "VA") == 0){
		kid = VA;
	}
	
	if (strcmp(name, "PF") == 0){
		kid = PF;
	}
	
	if (strcmp(name, "SPMV_CSRscalar") == 0){
		kid = SPMV_CSRscalar;
	}
	
	if (strcmp(name, "RCONV") == 0){
		kid = RCONV;
	}
	
	if (strcmp(name, "CCONV") == 0){
		kid = CCONV;
	}
	
	if (strcmp(name, "HST256") == 0){
		kid = HST256;
	}
	
	if (strcmp(name, "GCEDD") == 0){
		kid = GCEDD;
	}
	
	if (strcmp(name, "SCEDD") == 0){
		kid = SCEDD;
	}
	
	if (strcmp(name, "NCEDD") == 0){
		kid = NCEDD;
	}
	
	if (strcmp(name, "HCEDD") == 0){
		kid = HCEDD;
	}
	
	if (strcmp(name, "Reduction") == 0){
		kid = Reduction;
	}
	
	if (strcmp(name, "TP") == 0){
		kid = TP;
	}
	
	if (strcmp(name, "DXTC") == 0){
		kid = DXTC;
	}
	
	if (kid == -1){
		printf("Unknown kernel\n");
		return EMPTY;
	}
	else
		return (t_Kernel)kid;
}
	

/*int initialize_performance()
{
	
	int num_kernels = 9;
	
	max_num_block=(int *)calloc(num_kernels, sizeof(int));
	max_num_block[0] = 8; //MM
	max_num_block[1] = 8; //BS
	max_num_block[2] = 8; //VA
	max_num_block[3] = 8; //Reduction
	max_num_block[4] = 8; //PF
	max_num_block[5] = 8; //GCEDD
	max_num_block[6] = 16; //SPMV_CSRscalar
	max_num_block[7] = 10; //HST
	max_num_block[8] = 25; //RCONV
	
	perf = (t_cke_performance **)calloc(num_kernels, sizeof(t_cke_performance *));
	for (int i=0; i<num_kernels; i++)
		perf[i] = (t_cke_performance *)calloc(num_kernels, sizeof(t_cke_performance));
		
	// MM 
	
	perf[0][0].id[0]=MM;	perf[0][0].id[1]=MM;			perf[0][0].blocks[0]=4;	perf[0][0].blocks[1]=4; perf[0][0].speedup=0.90;
	perf[0][1].id[0]=MM;	perf[0][1].id[1]=BS;			perf[0][1].blocks[0]=5;	perf[0][1].blocks[1]=3; perf[0][1].speedup=1.16;
	perf[0][2].id[0]=MM;	perf[0][2].id[1]=VA;			perf[0][2].blocks[0]=6;	perf[0][2].blocks[1]=2; perf[0][2].speedup=1.21;
	perf[0][3].id[0]=MM;	perf[0][3].id[1]=Reduction;		perf[0][3].blocks[0]=3;	perf[0][3].blocks[1]=5; perf[0][3].speedup=1.13;
	perf[0][4].id[0]=MM;	perf[0][4].id[1]=PF;			perf[0][4].blocks[0]=5;	perf[0][4].blocks[1]=3; perf[0][4].speedup=1.06;
	perf[0][5].id[0]=MM;	perf[0][5].id[1]=GCEDD;			perf[0][5].blocks[0]=7;	perf[0][5].blocks[1]=1; perf[0][5].speedup=0.98;
	perf[0][6].id[0]=MM;	perf[0][6].id[1]=SPMV_CSRscalar;perf[0][6].blocks[0]=6;	perf[0][6].blocks[1]=4; perf[0][6].speedup=1.29;
	perf[0][7].id[0]=MM;	perf[0][7].id[1]=HST256;		perf[0][7].blocks[0]=4;	perf[0][7].blocks[1]=5; perf[0][7].speedup=1.03;
	perf[0][8].id[0]=MM;	perf[0][8].id[1]=RCONV;			perf[0][8].blocks[0]=6;	perf[0][8].blocks[1]=8; perf[0][8].speedup=1.23;
	
	//	BS 
	perf[1][0].id[0]=BS;	perf[1][0].id[1]=MM;			perf[1][0].blocks[0]=3;	perf[1][0].blocks[1]=5; perf[1][0].speedup=1.16;
	perf[1][1].id[0]=BS;	perf[1][1].id[1]=BS;			perf[1][1].blocks[0]=4;	perf[1][1].blocks[1]=4; perf[1][1].speedup=0.90;
	perf[1][2].id[0]=BS;	perf[1][2].id[1]=VA;			perf[1][2].blocks[0]=5;	perf[1][2].blocks[1]=3; perf[1][2].speedup=1.04;
	perf[1][3].id[0]=BS;	perf[1][3].id[1]=Reduction;		perf[1][3].blocks[0]=4;	perf[1][3].blocks[1]=4; perf[1][3].speedup=1.26;
	perf[1][4].id[0]=BS;	perf[1][4].id[1]=PF;			perf[1][4].blocks[0]=3;	perf[1][4].blocks[1]=5; perf[1][4].speedup=1.13;
	perf[1][5].id[0]=BS;	perf[1][5].id[1]=GCEDD;			perf[1][5].blocks[0]=1;	perf[1][5].blocks[1]=7; perf[1][5].speedup=0.95;
	perf[1][6].id[0]=BS;	perf[1][6].id[1]=SPMV_CSRscalar;perf[1][6].blocks[0]=4;	perf[1][6].blocks[1]=8; perf[1][6].speedup=1.06;
	perf[1][7].id[0]=BS;	perf[1][7].id[1]=HST256;		perf[1][7].blocks[0]=2;	perf[1][7].blocks[1]=8; perf[1][7].speedup=0.97;
	perf[1][8].id[0]=BS;	perf[1][8].id[1]=RCONV;			perf[1][8].blocks[0]=7;	perf[1][8].blocks[1]=4; perf[1][8].speedup=1.09;
	
	// VA 
	perf[2][0].id[0]=VA;	perf[2][0].id[1]=MM;			perf[2][0].blocks[0]=2;	perf[2][0].blocks[1]=6; 	perf[2][0].speedup=1.21;
	perf[2][1].id[0]=VA;	perf[2][1].id[1]=BS;			perf[2][1].blocks[0]=3;	perf[2][1].blocks[1]=5; 	perf[2][1].speedup=1.04;
	perf[2][2].id[0]=VA;	perf[2][2].id[1]=VA;			perf[2][2].blocks[0]=4;	perf[2][2].blocks[1]=4; 	perf[2][2].speedup=0.9;
	perf[2][3].id[0]=VA;	perf[2][3].id[1]=Reduction;		perf[2][3].blocks[0]=3;	perf[2][3].blocks[1]=5; 	perf[2][3].speedup=1.59;
	perf[2][4].id[0]=VA;	perf[2][4].id[1]=PF;			perf[2][4].blocks[0]=2;	perf[2][4].blocks[1]=6; 	perf[2][4].speedup=1.24;
	perf[2][5].id[0]=VA;	perf[2][5].id[1]=GCEDD;			perf[2][5].blocks[0]=1;	perf[2][5].blocks[1]=7; 	perf[2][5].speedup=0.99;
	perf[2][6].id[0]=VA;	perf[2][6].id[1]=SPMV_CSRscalar;perf[2][6].blocks[0]=2;	perf[2][6].blocks[1]=12; 	perf[2][6].speedup=1.00;
	perf[2][7].id[0]=VA;	perf[2][7].id[1]=HST256;		perf[2][7].blocks[0]=4;	perf[2][7].blocks[1]=5; 	perf[2][7].speedup=1.00;
	perf[2][8].id[0]=VA;	perf[2][8].id[1]=RCONV;			perf[2][8].blocks[0]=7;	perf[2][8].blocks[1]=4; 	perf[2][8].speedup=0.97;
	
	// RED	
	perf[3][0].id[0]=Reduction;		perf[3][0].id[1]=MM;			perf[3][0].blocks[0]=5;	perf[3][0].blocks[1]=3; 	perf[3][0].speedup=1.13;
	perf[3][1].id[0]=Reduction;		perf[3][1].id[1]=BS;			perf[3][1].blocks[0]=4;	perf[3][1].blocks[1]=4; 	perf[3][1].speedup=1.26;
	perf[3][2].id[0]=Reduction;		perf[3][2].id[1]=VA;			perf[3][2].blocks[0]=5;	perf[3][2].blocks[1]=3; 	perf[3][2].speedup=1.59;
	perf[3][3].id[0]=Reduction;		perf[3][3].id[1]=Reduction;		perf[3][3].blocks[0]=4;	perf[3][3].blocks[1]=4; 	perf[3][3].speedup=0.9;
	perf[3][4].id[0]=Reduction;		perf[3][4].id[1]=PF;			perf[3][4].blocks[0]=6;	perf[3][4].blocks[1]=2; 	perf[3][4].speedup=0.99;
	perf[3][5].id[0]=Reduction;		perf[3][5].id[1]=GCEDD;			perf[3][5].blocks[0]=3;	perf[3][5].blocks[1]=5; 	perf[3][5].speedup=1.10;
	perf[3][6].id[0]=Reduction;		perf[3][6].id[1]=SPMV_CSRscalar;perf[3][6].blocks[0]=4;	perf[3][6].blocks[1]=6; 	perf[3][6].speedup=1.70;
	perf[3][7].id[0]=Reduction;		perf[3][7].id[1]=HST256;		perf[3][7].blocks[0]=5;	perf[3][7].blocks[1]=4; 	perf[3][7].speedup=1.13;
	perf[3][8].id[0]=Reduction;		perf[3][8].id[1]=RCONV;			perf[3][8].blocks[0]=6;	perf[3][8].blocks[1]=8; 	perf[3][8].speedup=1.36;
	
	// PF 
	perf[4][0].id[0]=PF;	perf[4][0].id[1]=MM;			perf[4][0].blocks[0]=5;	perf[4][0].blocks[1]=3; 	perf[4][0].speedup=1.05;
	perf[4][1].id[0]=PF;	perf[4][1].id[1]=BS;			perf[4][1].blocks[0]=5;	perf[4][1].blocks[1]=3; 	perf[4][1].speedup=1.13;
	perf[4][2].id[0]=PF;	perf[4][2].id[1]=VA;			perf[4][2].blocks[0]=6;	perf[4][2].blocks[1]=2; 	perf[4][2].speedup=1.24;
	perf[4][3].id[0]=PF;	perf[4][3].id[1]=Reduction;		perf[4][3].blocks[0]=6;	perf[4][3].blocks[1]=2; 	perf[4][3].speedup=0.99;
	perf[4][4].id[0]=PF;	perf[4][4].id[1]=PF;			perf[4][4].blocks[0]=4;	perf[4][4].blocks[1]=4; 	perf[4][4].speedup=0.90;
	perf[4][5].id[0]=PF;	perf[4][5].id[1]=GCEDD;			perf[4][5].blocks[0]=2;	perf[4][5].blocks[1]=6; 	perf[4][5].speedup=0.97;
	perf[4][6].id[0]=PF;	perf[4][6].id[1]=SPMV_CSRscalar;perf[4][6].blocks[0]=4;	perf[4][6].blocks[1]=6; 	perf[4][6].speedup=1.39;
	perf[4][7].id[0]=PF;	perf[4][7].id[1]=HST256;		perf[4][7].blocks[0]=5;	perf[4][7].blocks[1]=4; 	perf[4][7].speedup=1.13;
	perf[4][8].id[0]=PF;	perf[4][8].id[1]=RCONV;			perf[4][8].blocks[0]=6;	perf[4][8].blocks[1]=8; 	perf[4][8].speedup=1.30;
	
	// GCEDD 
	perf[5][0].id[0]=GCEDD;	perf[5][0].id[1]=MM;			perf[5][0].blocks[0]=1;	perf[5][0].blocks[1]=7; 	perf[5][0].speedup=0.98;
	perf[5][1].id[0]=GCEDD;	perf[5][1].id[1]=BS;			perf[5][1].blocks[0]=7;	perf[5][1].blocks[1]=1; 	perf[5][1].speedup=0.95;
	perf[5][2].id[0]=GCEDD;	perf[5][2].id[1]=VA;			perf[5][2].blocks[0]=7;	perf[5][2].blocks[1]=1; 	perf[5][2].speedup=0.99;
	perf[5][3].id[0]=GCEDD;	perf[5][3].id[1]=Reduction;		perf[5][3].blocks[0]=3;	perf[5][3].blocks[1]=5; 	perf[5][3].speedup=1.10;
	perf[5][4].id[0]=GCEDD;	perf[5][4].id[1]=PF;			perf[5][4].blocks[0]=6;	perf[5][4].blocks[1]=2; 	perf[5][4].speedup=0.97;
	perf[5][5].id[0]=GCEDD;	perf[5][5].id[1]=GCEDD;			perf[5][5].blocks[0]=4;	perf[5][5].blocks[1]=4;  	perf[5][5].speedup=0.90;
	perf[5][7].id[0]=GCEDD;	perf[5][6].id[1]=SPMV_CSRscalar;perf[5][6].blocks[0]=2;	perf[5][6].blocks[1]=7; 	perf[5][6].speedup=1.13;
	perf[5][7].id[0]=GCEDD;	perf[5][7].id[1]=HST256;		perf[5][7].blocks[0]=7;	perf[5][7].blocks[1]=1; 	perf[5][7].speedup=0.91;
	perf[5][8].id[0]=GCEDD;	perf[5][8].id[1]=RCONV;			perf[5][8].blocks[0]=25;perf[5][8].blocks[1]=1; 	perf[5][8].speedup=0.97;
	
	//SPMV 
	perf[6][0].id[0]=SPMV_CSRscalar;		perf[6][0].id[1]=MM;			perf[6][0].blocks[0]=4;	perf[6][0].blocks[1]=6; 	perf[6][0].speedup=1.29;
	perf[6][1].id[0]=SPMV_CSRscalar;		perf[6][1].id[1]=BS;			perf[6][1].blocks[0]=8;	perf[6][1].blocks[1]=4; 	perf[6][1].speedup=1.05;
	perf[6][2].id[0]=SPMV_CSRscalar;		perf[6][2].id[1]=VA;			perf[6][2].blocks[0]=12;perf[6][2].blocks[1]=2; 	perf[6][2].speedup=1.00;
	perf[6][3].id[0]=SPMV_CSRscalar;		perf[6][3].id[1]=Reduction;		perf[6][3].blocks[0]=4;	perf[6][3].blocks[1]=6; 	perf[6][3].speedup=1.70;
	perf[6][4].id[0]=SPMV_CSRscalar;		perf[6][4].id[1]=PF;			perf[6][4].blocks[0]=6;	perf[6][4].blocks[1]=4; 	perf[6][4].speedup=1.39;
	perf[6][5].id[0]=SPMV_CSRscalar;		perf[6][5].id[1]=GCEDD;			perf[6][5].blocks[0]=7;	perf[6][5].blocks[1]=2; 	perf[6][5].speedup=1.13;
	perf[6][6].id[0]=SPMV_CSRscalar;		perf[6][6].id[1]=SPMV_CSRscalar;perf[6][6].blocks[0]=8;	perf[6][6].blocks[1]=8; 	perf[6][6].speedup=0.9;
	perf[6][7].id[0]=SPMV_CSRscalar;		perf[6][7].id[1]=HST256;		perf[6][7].blocks[0]=8;	perf[6][7].blocks[1]=5; 	perf[6][7].speedup=1.08;
	perf[6][8].id[0]=SPMV_CSRscalar;		perf[6][8].id[1]=RCONV;			perf[6][8].blocks[0]=15;perf[6][8].blocks[1]=2; 	perf[6][8].speedup=0.97;
	
	//HST256 
	perf[7][0].id[0]=HST256;		perf[7][0].id[1]=MM;			perf[7][0].blocks[0]=5;	perf[7][0].blocks[1]=4; 	perf[7][0].speedup=1.02;
	perf[7][1].id[0]=HST256;		perf[7][1].id[1]=BS;			perf[7][1].blocks[0]=8;	perf[7][1].blocks[1]=2; 	perf[7][1].speedup=0.97;
	perf[7][2].id[0]=HST256;		perf[7][2].id[1]=VA;			perf[7][2].blocks[0]=5;	perf[7][2].blocks[1]=4; 	perf[7][2].speedup=1.00;
	perf[7][3].id[0]=HST256;		perf[7][3].id[1]=Reduction;		perf[7][3].blocks[0]=5;	perf[7][3].blocks[1]=4; 	perf[7][3].speedup=1.13;
	perf[7][4].id[0]=HST256;		perf[7][4].id[1]=PF;			perf[7][4].blocks[0]=4;	perf[7][4].blocks[1]=5; 	perf[7][4].speedup=0.99;
	perf[7][5].id[0]=HST256;		perf[7][5].id[1]=GCEDD;			perf[7][5].blocks[0]=1;	perf[7][5].blocks[1]=7; 	perf[7][5].speedup=0.91;
	perf[7][6].id[0]=HST256;		perf[7][6].id[1]=SPMV_CSRscalar;perf[7][6].blocks[0]=5;	perf[7][6].blocks[1]=8; 	perf[7][6].speedup=1.08;
	perf[7][7].id[0]=HST256;		perf[7][7].id[1]=HST256;		perf[7][7].blocks[0]=5;	perf[7][7].blocks[1]=5; 	perf[7][7].speedup=0.9;
	perf[7][8].id[0]=HST256;		perf[7][8].id[1]=RCONV;			perf[7][8].blocks[0]=9;	perf[7][8].blocks[1]=7; 	perf[7][8].speedup=1.06;
	
	// RCONV 
	perf[8][0].id[0]=RCONV;		perf[8][0].id[1]=MM;			perf[8][0].blocks[0]=8;	perf[8][0].blocks[1]=6; 	perf[8][0].speedup=1.22;
	perf[8][1].id[0]=RCONV;		perf[8][1].id[1]=BS;			perf[8][1].blocks[0]=4;	perf[8][1].blocks[1]=7; 	perf[8][1].speedup=1.08;
	perf[8][2].id[0]=RCONV;		perf[8][2].id[1]=VA;			perf[8][2].blocks[0]=4;	perf[8][2].blocks[1]=7; 	perf[8][2].speedup=0.97;
	perf[8][3].id[0]=RCONV;		perf[8][3].id[1]=Reduction;		perf[8][3].blocks[0]=7;	perf[8][3].blocks[1]=4; 	perf[8][3].speedup=1.36;
	perf[8][4].id[0]=RCONV;		perf[8][4].id[1]=PF;			perf[8][4].blocks[0]=8;	perf[8][4].blocks[1]=6; 	perf[8][4].speedup=0.30;
	perf[8][5].id[0]=RCONV;		perf[8][5].id[1]=GCEDD;			perf[8][5].blocks[0]=1;	perf[8][5].blocks[1]=25; 	perf[8][5].speedup=0.97;
	perf[8][6].id[0]=RCONV;		perf[8][6].id[1]=SPMV_CSRscalar;perf[8][6].blocks[0]=2;	perf[8][6].blocks[1]=25; 	perf[8][6].speedup=0.97;
	perf[8][7].id[0]=RCONV;		perf[8][7].id[1]=HST256;		perf[8][7].blocks[0]=7;	perf[8][7].blocks[1]=9; 	perf[8][7].speedup=1.06;
	perf[8][8].id[0]=RCONV;		perf[8][8].id[1]=RCONV;			perf[8][8].blocks[0]=12;perf[8][8].blocks[1]=13; 	perf[8][8].speedup=0.9;

	
	return 0;
}
*/
/*
int initialize_theoretical_performance()
{
	
	int num_kernels = 9;
	
	th_perf = (t_cke_performance **)calloc(num_kernels, sizeof(t_cke_performance *));
	for (int i=0; i<num_kernels; i++)
		th_perf[i] = (t_cke_performance *)calloc(num_kernels, sizeof(t_cke_performance));
		
	// MM 
	
	th_perf[0][0].id[0]=MM;	th_perf[0][0].id[1]=MM;			th_perf[0][0].blocks[0]=4;	th_perf[0][0].blocks[1]=4; th_perf[0][0].speedup=1.0;
	th_perf[0][1].id[0]=MM;	th_perf[0][1].id[1]=BS;			th_perf[0][1].blocks[0]=4;	th_perf[0][1].blocks[1]=4; th_perf[0][1].speedup=1.60;
	th_perf[0][2].id[0]=MM;	th_perf[0][2].id[1]=VA;			th_perf[0][2].blocks[0]=5;	th_perf[0][2].blocks[1]=3; th_perf[0][2].speedup=1.82;
	th_perf[0][3].id[0]=MM;	th_perf[0][3].id[1]=Reduction;		th_perf[0][3].blocks[0]=3;	th_perf[0][3].blocks[1]=5; th_perf[0][3].speedup=1.48;
	th_perf[0][4].id[0]=MM;	th_perf[0][4].id[1]=PF;			th_perf[0][4].blocks[0]=4;	th_perf[0][4].blocks[1]=4; th_perf[0][4].speedup=1.30;
	th_perf[0][5].id[0]=MM;	th_perf[0][5].id[1]=GCEDD;			th_perf[0][5].blocks[0]=3;	th_perf[0][5].blocks[1]=5; th_perf[0][5].speedup=1.41;
	th_perf[0][6].id[0]=MM;	th_perf[0][6].id[1]=SPMV_CSRscalar;th_perf[0][6].blocks[0]=6;	th_perf[0][6].blocks[1]=4; th_perf[0][6].speedup=1.97;
	th_perf[0][7].id[0]=MM;	th_perf[0][7].id[1]=HST256;		th_perf[0][7].blocks[0]=6;	th_perf[0][7].blocks[1]=2; th_perf[0][7].speedup=1.32;
	th_perf[0][8].id[0]=MM;	th_perf[0][8].id[1]=RCONV;			th_perf[0][8].blocks[0]=6;	th_perf[0][8].blocks[1]=8; th_perf[0][8].speedup=1.94;
	
	// BS 
	th_perf[1][0].id[0]=BS;	th_perf[1][0].id[1]=MM;			th_perf[1][0].blocks[0]=4;	th_perf[1][0].blocks[1]=4; th_perf[1][0].speedup=1.60;
	th_perf[1][1].id[0]=BS;	th_perf[1][1].id[1]=BS;			th_perf[1][1].blocks[0]=4;	th_perf[1][1].blocks[1]=4; th_perf[1][1].speedup=1.00;
	th_perf[1][2].id[0]=BS;	th_perf[1][2].id[1]=VA;			th_perf[1][2].blocks[0]=5;	th_perf[1][2].blocks[1]=3; th_perf[1][2].speedup=1.87;
	th_perf[1][3].id[0]=BS;	th_perf[1][3].id[1]=Reduction;		th_perf[1][3].blocks[0]=3;	th_perf[1][3].blocks[1]=5; th_perf[1][3].speedup=1.55;
	th_perf[1][4].id[0]=BS;	th_perf[1][4].id[1]=PF;			th_perf[1][4].blocks[0]=3;	th_perf[1][4].blocks[1]=5; th_perf[1][4].speedup=1.37;
	th_perf[1][5].id[0]=BS;	th_perf[1][5].id[1]=GCEDD;			th_perf[1][5].blocks[0]=3;	th_perf[1][5].blocks[1]=5; th_perf[1][5].speedup=1.49;
	th_perf[1][6].id[0]=BS;	th_perf[1][6].id[1]=SPMV_CSRscalar;th_perf[1][6].blocks[0]=6;	th_perf[1][6].blocks[1]=4; th_perf[1][6].speedup=2.00;
	th_perf[1][7].id[0]=BS;	th_perf[1][7].id[1]=HST256;		th_perf[1][7].blocks[0]=6;	th_perf[1][7].blocks[1]=2; th_perf[1][7].speedup=1.34;
	th_perf[1][8].id[0]=BS;	th_perf[1][8].id[1]=RCONV;			th_perf[1][8].blocks[0]=6;	th_perf[1][8].blocks[1]=8; th_perf[1][8].speedup=1.97;
	
	// VA 
	th_perf[2][0].id[0]=VA;	th_perf[2][0].id[1]=MM;			th_perf[2][0].blocks[0]=3;	th_perf[2][0].blocks[1]=5; 	th_perf[2][0].speedup=1.82;
	th_perf[2][1].id[0]=VA;	th_perf[2][1].id[1]=BS;			th_perf[2][1].blocks[0]=5;	th_perf[2][1].blocks[1]=3; 	th_perf[2][1].speedup=1.87;
	th_perf[2][2].id[0]=VA;	th_perf[2][2].id[1]=VA;			th_perf[2][2].blocks[0]=4;	th_perf[2][2].blocks[1]=4; 	th_perf[2][2].speedup=1.00;
	th_perf[2][3].id[0]=VA;	th_perf[2][3].id[1]=Reduction;		th_perf[2][3].blocks[0]=3;	th_perf[2][3].blocks[1]=5; 	th_perf[2][2].speedup=1.80;
	th_perf[2][4].id[0]=VA;	th_perf[2][4].id[1]=PF;			th_perf[2][4].blocks[0]=3;	th_perf[2][4].blocks[1]=5; 	th_perf[2][3].speedup=1.61;
	th_perf[2][5].id[0]=VA;	th_perf[2][5].id[1]=GCEDD;			th_perf[2][5].blocks[0]=2;	th_perf[2][5].blocks[1]=6; 	th_perf[2][4].speedup=1.75;
	th_perf[2][6].id[0]=VA;	th_perf[2][6].id[1]=SPMV_CSRscalar;th_perf[2][6].blocks[0]=3;	th_perf[2][6].blocks[1]=10; 	th_perf[2][5].speedup=2.07;
	th_perf[2][7].id[0]=VA;	th_perf[2][7].id[1]=HST256;		th_perf[2][7].blocks[0]=5;	th_perf[2][7].blocks[1]=4; 	th_perf[2][6].speedup=1.66;
	th_perf[2][8].id[0]=VA;	th_perf[2][8].id[1]=RCONV;			th_perf[2][8].blocks[0]=5;	th_perf[2][8].blocks[1]=12; 	th_perf[2][7].speedup=2.00;
	
	// RED	
	th_perf[3][0].id[0]=Reduction;		th_perf[3][0].id[1]=MM;			th_perf[3][0].blocks[0]=5;	th_perf[3][0].blocks[1]=3; 	th_perf[3][0].speedup=1.48;
	th_perf[3][1].id[0]=Reduction;		th_perf[3][1].id[1]=BS;			th_perf[3][1].blocks[0]=5;	th_perf[3][1].blocks[1]=3; 	th_perf[3][1].speedup=1.55;
	th_perf[3][2].id[0]=Reduction;		th_perf[3][2].id[1]=VA;			th_perf[3][2].blocks[0]=5;	th_perf[3][2].blocks[1]=3; 	th_perf[3][2].speedup=1.80;
	th_perf[3][3].id[0]=Reduction;		th_perf[3][3].id[1]=Reduction;		th_perf[3][3].blocks[0]=4;	th_perf[3][3].blocks[1]=4; 	th_perf[3][3].speedup=1.00;
	th_perf[3][4].id[0]=Reduction;		th_perf[3][4].id[1]=PF;			th_perf[3][4].blocks[0]=4;	th_perf[3][4].blocks[1]=4; 	th_perf[3][4].speedup=1.24;
	th_perf[3][5].id[0]=Reduction;		th_perf[3][5].id[1]=GCEDD;			th_perf[3][5].blocks[0]=4;	th_perf[3][5].blocks[1]=4; 	th_perf[3][5].speedup=1.34;
	th_perf[3][6].id[0]=Reduction;		th_perf[3][6].id[1]=SPMV_CSRscalar;th_perf[3][6].blocks[0]=4;	th_perf[3][6].blocks[1]=6; 	th_perf[3][6].speedup=1.95;
	th_perf[3][7].id[0]=Reduction;		th_perf[3][7].id[1]=HST256;		th_perf[3][7].blocks[0]=5;	th_perf[3][7].blocks[1]=4; 	th_perf[3][7].speedup=1.48;
	th_perf[3][8].id[0]=Reduction;		th_perf[3][8].id[1]=RCONV;			th_perf[3][8].blocks[0]=6;	th_perf[3][8].blocks[1]=8; 	th_perf[3][8].speedup=1.91;
	
	// PF 
	th_perf[4][0].id[0]=PF;	th_perf[4][0].id[1]=MM;			th_perf[4][0].blocks[0]=4;	th_perf[4][0].blocks[1]=4; 	th_perf[4][0].speedup=1.30;
	th_perf[4][1].id[0]=PF;	th_perf[4][1].id[1]=BS;			th_perf[4][1].blocks[0]=5;	th_perf[4][1].blocks[1]=3; 	th_perf[4][1].speedup=1.37;
	th_perf[4][2].id[0]=PF;	th_perf[4][2].id[1]=VA;			th_perf[4][2].blocks[0]=5;	th_perf[4][2].blocks[1]=3; 	th_perf[4][2].speedup=1.61;
	th_perf[4][3].id[0]=PF;	th_perf[4][3].id[1]=Reduction;		th_perf[4][3].blocks[0]=4;	th_perf[4][3].blocks[1]=4; 	th_perf[4][3].speedup=1.24;
	th_perf[4][4].id[0]=PF;	th_perf[4][4].id[1]=PF;			th_perf[4][4].blocks[0]=4;	th_perf[4][4].blocks[1]=4; 	th_perf[4][4].speedup=1.00;
	th_perf[4][5].id[0]=PF;	th_perf[4][5].id[1]=GCEDD;			th_perf[4][5].blocks[0]=4;	th_perf[4][5].blocks[1]=4; 	th_perf[4][5].speedup=1.18;
	th_perf[4][6].id[0]=PF;	th_perf[4][6].id[1]=SPMV_CSRscalar;th_perf[4][6].blocks[0]=6;	th_perf[4][6].blocks[1]=4; 	th_perf[4][6].speedup=1.80;
	th_perf[4][7].id[0]=PF;	th_perf[4][7].id[1]=HST256;		th_perf[4][7].blocks[0]=5;	th_perf[4][7].blocks[1]=4; 	th_perf[4][7].speedup=1.30;
	th_perf[4][8].id[0]=PF;	th_perf[4][8].id[1]=RCONV;			th_perf[4][8].blocks[0]=6;	th_perf[4][8].blocks[1]=8; 	th_perf[4][8].speedup=1.76;
	
	// GCEDD 
	th_perf[5][0].id[0]=GCEDD;	th_perf[5][0].id[1]=MM;			th_perf[5][0].blocks[0]=5;	th_perf[5][0].blocks[1]=3; 	th_perf[5][0].speedup=1.41;
	th_perf[5][1].id[0]=GCEDD;	th_perf[5][1].id[1]=BS;			th_perf[5][1].blocks[0]=5;	th_perf[5][1].blocks[1]=3; 	th_perf[5][1].speedup=1.49;
	th_perf[5][2].id[0]=GCEDD;	th_perf[5][2].id[1]=VA;			th_perf[5][2].blocks[0]=6;	th_perf[5][2].blocks[1]=2; 	th_perf[5][2].speedup=1.75;
	th_perf[5][3].id[0]=GCEDD;	th_perf[5][3].id[1]=Reduction;		th_perf[5][3].blocks[0]=4;	th_perf[5][3].blocks[1]=4; 	th_perf[5][3].speedup=1.34;
	th_perf[5][4].id[0]=GCEDD;	th_perf[5][4].id[1]=PF;			th_perf[5][4].blocks[0]=4;	th_perf[5][4].blocks[1]=4; 	th_perf[5][4].speedup=1.18;
	th_perf[5][5].id[0]=GCEDD;	th_perf[5][5].id[1]=GCEDD;			th_perf[5][5].blocks[0]=4;	th_perf[5][5].blocks[1]=4;  	th_perf[5][5].speedup=1.00;
	th_perf[5][7].id[0]=GCEDD;	th_perf[5][6].id[1]=SPMV_CSRscalar;th_perf[5][6].blocks[0]=6;	th_perf[5][6].blocks[1]=4; 	th_perf[5][6].speedup=1.89;
	th_perf[5][7].id[0]=GCEDD;	th_perf[5][7].id[1]=HST256;		th_perf[5][7].blocks[0]=5;	th_perf[5][7].blocks[1]=4;		th_perf[5][7].speedup=1.42;
	th_perf[5][8].id[0]=GCEDD;	th_perf[5][8].id[1]=RCONV;			th_perf[5][8].blocks[0]=6; th_perf[5][8].blocks[1]=8; 	th_perf[5][8].speedup=1.86;
	
	// SPMV 
	th_perf[6][0].id[0]=SPMV_CSRscalar;		th_perf[6][0].id[1]=MM;			th_perf[6][0].blocks[0]=4;	th_perf[6][0].blocks[1]=6; 	th_perf[6][0].speedup=1.97;
	th_perf[6][1].id[0]=SPMV_CSRscalar;		th_perf[6][1].id[1]=BS;			th_perf[6][1].blocks[0]=4;	th_perf[6][1].blocks[1]=6; 	th_perf[6][1].speedup=2.00;
	th_perf[6][2].id[0]=SPMV_CSRscalar;		th_perf[6][2].id[1]=VA;			th_perf[6][2].blocks[0]=10;th_perf[6][2].blocks[1]=3; 	th_perf[6][2].speedup=2.07;
	th_perf[6][3].id[0]=SPMV_CSRscalar;		th_perf[6][3].id[1]=Reduction;		th_perf[6][3].blocks[0]=6;	th_perf[6][3].blocks[1]=4; 	th_perf[6][3].speedup=1.95;
	th_perf[6][4].id[0]=SPMV_CSRscalar;		th_perf[6][4].id[1]=PF;			th_perf[6][4].blocks[0]=4;	th_perf[6][4].blocks[1]=6; 	th_perf[6][4].speedup=1.80;
	th_perf[6][5].id[0]=SPMV_CSRscalar;		th_perf[6][5].id[1]=GCEDD;			th_perf[6][5].blocks[0]=4;	th_perf[6][5].blocks[1]=6; 	th_perf[6][5].speedup=1.89;
	th_perf[6][6].id[0]=SPMV_CSRscalar;		th_perf[6][6].id[1]=SPMV_CSRscalar;th_perf[6][6].blocks[0]=8;	th_perf[6][6].blocks[1]=8; 	th_perf[6][6].speedup=1.00;
	th_perf[6][7].id[0]=SPMV_CSRscalar;		th_perf[6][7].id[1]=HST256;		th_perf[6][7].blocks[0]=5;	th_perf[6][7].blocks[1]=7; 	th_perf[6][7].speedup=1.97;
	th_perf[6][8].id[0]=SPMV_CSRscalar;		th_perf[6][8].id[1]=RCONV;			th_perf[6][8].blocks[0]=7; th_perf[6][8].blocks[1]=18; 	th_perf[6][8].speedup=2.00;
	
	// HST256
	th_perf[7][0].id[0]=HST256;		th_perf[7][0].id[1]=MM;			th_perf[7][0].blocks[0]=2;	th_perf[7][0].blocks[1]=6; 	th_perf[7][0].speedup=1.32;
	th_perf[7][1].id[0]=HST256;		th_perf[7][1].id[1]=BS;			th_perf[7][1].blocks[0]=2;	th_perf[7][1].blocks[1]=6; 	th_perf[7][1].speedup=1.34;
	th_perf[7][2].id[0]=HST256;		th_perf[7][2].id[1]=VA;			th_perf[7][2].blocks[0]=5;	th_perf[7][2].blocks[1]=4; 	th_perf[7][2].speedup=1.00;
	th_perf[7][3].id[0]=HST256;		th_perf[7][3].id[1]=Reduction;		th_perf[7][3].blocks[0]=4;	th_perf[7][3].blocks[1]=5; 	th_perf[7][3].speedup=1.48;
	th_perf[7][4].id[0]=HST256;		th_perf[7][4].id[1]=PF;			th_perf[7][4].blocks[0]=5;	th_perf[7][4].blocks[1]=4; 	th_perf[7][4].speedup=1.30;
	th_perf[7][5].id[0]=HST256;		th_perf[7][5].id[1]=GCEDD;			th_perf[7][5].blocks[0]=4;	th_perf[7][5].blocks[1]=5; 	th_perf[7][5].speedup=1.42;
	th_perf[7][6].id[0]=HST256;		th_perf[7][6].id[1]=SPMV_CSRscalar;th_perf[7][6].blocks[0]=7;	th_perf[7][6].blocks[1]=5; 	th_perf[7][6].speedup=1.97;
	th_perf[7][7].id[0]=HST256;		th_perf[7][7].id[1]=HST256;		th_perf[7][7].blocks[0]=5;	th_perf[7][7].blocks[1]=5; 	th_perf[7][7].speedup=1.00;
	th_perf[7][8].id[0]=HST256;		th_perf[7][8].id[1]=RCONV;			th_perf[7][8].blocks[0]=8;	th_perf[7][8].blocks[1]=6; 	th_perf[7][8].speedup=1.91;
	
	// RCONV 
	th_perf[8][0].id[0]=RCONV;		th_perf[8][0].id[1]=MM;			th_perf[8][0].blocks[0]=8;	th_perf[8][0].blocks[1]=6; 	th_perf[8][0].speedup=1.94;
	th_perf[8][1].id[0]=RCONV;		th_perf[8][1].id[1]=BS;			th_perf[8][1].blocks[0]=8;	th_perf[8][1].blocks[1]=6; 	th_perf[8][1].speedup=1.94;
	th_perf[8][2].id[0]=RCONV;		th_perf[8][2].id[1]=VA;			th_perf[8][2].blocks[0]=12;th_perf[8][2].blocks[1]=5; 	th_perf[8][2].speedup=2.00;
	th_perf[8][3].id[0]=RCONV;		th_perf[8][3].id[1]=Reduction;		th_perf[8][3].blocks[0]=8;	th_perf[8][3].blocks[1]=6; 	th_perf[8][3].speedup=1.91;
	th_perf[8][4].id[0]=RCONV;		th_perf[8][4].id[1]=PF;			th_perf[8][4].blocks[0]=8;	th_perf[8][4].blocks[1]=6; 	th_perf[8][4].speedup=0.30;
	th_perf[8][5].id[0]=RCONV;		th_perf[8][5].id[1]=GCEDD;			th_perf[8][5].blocks[0]=8;	th_perf[8][5].blocks[1]=6; 	th_perf[8][5].speedup=1.76;
	th_perf[8][6].id[0]=RCONV;		th_perf[8][6].id[1]=SPMV_CSRscalar;th_perf[8][6].blocks[0]=18;th_perf[8][6].blocks[1]=7; 	th_perf[8][6].speedup=2.10;
	th_perf[8][7].id[0]=RCONV;		th_perf[8][7].id[1]=HST256;		th_perf[8][7].blocks[0]=6;	th_perf[8][7].blocks[1]=8; 	th_perf[8][7].speedup=1.91;
	th_perf[8][8].id[0]=RCONV;		th_perf[8][8].id[1]=RCONV;			th_perf[8][8].blocks[0]=12;th_perf[8][8].blocks[1]=13; 	th_perf[8][8].speedup=1.00;

	
	return 0;
}
*/
/*
int get_best_partner(t_Kernel *kid, int *k_done, int num_kernels, t_Kernel curr_kid, t_Kernel *next_kid, int *index_next, int *b0, int *b1)
{
	int index1, index2, save_index2;
	
	get_index(curr_kid, &index1); // Get the index of the running kernel
	
	double best_perf = -1.0;
	int best_index;
	t_Kernel best_kid;
	
	for (int i=0; i<num_kernels; i++){ // For the remainning kernels 
		if (k_done[i] == 0){ // If kernel has not been executed yet	
			get_index(kid[i], &index2); // Get the index of the new kernel
			if (perf[index1][index2].speedup > best_perf) { // Check performance and annotate if better
				best_perf = perf[index1][index2].speedup;
				best_index = i;
				best_kid = kid[i];
				save_index2 = index2;
			}
		}
	}
			
	if (best_perf >= MIN_SPEEDUP) {
		*next_kid = best_kid;
		*index_next = best_index;
		*b0 = perf[index1][save_index2].blocks[0];
		*b1 = perf[index1][save_index2].blocks[1];
	}else{
		*next_kid = EMPTY;
		*b0 = max_num_block[index1]; // If performace is low the running kernel is executed with all the blocks
	}
	
	return 0;
	
}*/

/*
int get_best_partner_theoretical(t_Kernel curr_kid, t_Kernel *kid, int *k_done, float **bad_partner, int num_kernels, t_Kernel *select_kid, int *select_index, int *b0, int *b1)
{
	int index1, index2, save_index2;
	
	get_index(curr_kid, &index1); // Get the index of the running kernel for perfomance tables
	
	double best_perf = -1.0;
	int best_index;
	t_Kernel best_kid;
	
	for (int i=0; i<num_kernels; i++){ // For the remainning kernels 
		if (k_done[i] == 0 && bad_partner[curr_kid][kid[i]] == 0){ // If kernel has not been executed yet	and it is not a bad partner
			get_index(kid[i], &index2); // Get the index of the new kernel
			if (perf[index1][index2].speedup > best_perf) { // Check performance and annotate if better
				best_perf = perf[index1][index2].speedup;
				best_index = i;
				best_kid = kid[i];
				save_index2 = index2;
			}
		}
	}
			
	if (best_perf >=MIN_SPEEDUP) {
		*select_kid = best_kid;
		*select_index = best_index;
		*b0 = perf[index1][save_index2].blocks[0];
		*b1 = perf[index1][save_index2].blocks[1];
	}else{
		*select_kid = EMPTY;
		*b0 = max_num_block[index1]; // If performace is low the running kernel is executed with all the blocks
	}
	
	return 0;
}*/

/*
int get_last_kernel (t_Kernel kid, int *num_blocks)
{
	int index2;
	
	get_index(kid, &index2);
	*num_blocks = max_num_block[index2];
	
	return 0;
}
	*/
int initialize_solo_performance()
{
	int num_kernels = 9;
	
	sperf = (t_solo_performance *)calloc(num_kernels, sizeof(t_solo_performance));
	
	sperf[0].id = MM; 				sperf[0].max_tpms=735;
	sperf[1].id = BS; 				sperf[1].max_tpms=2431;
	sperf[2].id = VA; 				sperf[2].max_tpms=2914;
	sperf[3].id = Reduction; 		sperf[3].max_tpms=17231;
	sperf[4].id = PF; 				sperf[4].max_tpms=1221;
	sperf[5].id = GCEDD; 			sperf[5].max_tpms=86893;
	sperf[6].id = SPMV_CSRscalar; 	sperf[6].max_tpms=34309;
	sperf[7].id = HST256; 			sperf[7].max_tpms=4632;
	sperf[8].id = RCONV; 			sperf[8].max_tpms=66377;
	
	return 0;
}


double get_solo_perf(t_Kernel id)
{
	
	int num_kernels = 9;
	
	for (int i=0; i<num_kernels; i++)
		if (id == sperf[i].id)
			return sperf[i].max_tpms;
		
	printf("Error: id not found\n");
	
	return 0;
}


/*
int get_max_blocks(t_Kernel kid)
{
	int index;
	get_index(kid, &index);
	return max_num_block[index];
}*/