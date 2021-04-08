#include <unistd.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <helper_functions.h>   // helper functions for string parsing
#include <helper_cuda.h>   
#include <math.h>
#include "elastic_kernel.h"
#include "cupti_profiler.h"
#include "VA/VA.h"
#include "PF/PF.h"
#include "BS/BS.h"
#include "MM/MM.h"
#include "HST/HST256.h"
#include "Reduction/reduction.h"
#include "CEDD/CEDD.h"
#include "SPMV/SPMV.h"
#include "CONV/CONV.h"
#include "TP/TP.h"
#include "DXTC/DXTC.h"


int *get_cta_counter_position(t_kernel_stub *k)
{

    if (k->id == VA){
        t_VA_params *VAparams = (t_VA_params *)k->params;
        return VAparams->zc_slc;
    }
    if (k->id == BS){
        t_BS_params *BSparams = (t_BS_params *)k->params;
        return BSparams->zc_slc;
    }
    if (k->id == PF){
        t_PF_params *PFparams = (t_PF_params *)k->params;
        return PFparams->zc_slc;
    }
    if (k->id == MM){
        t_MM_params *MMparams = (t_MM_params *)k->params;
        return MMparams->zc_slc;
    }
    if (k->id == HST256){
        t_HST256_params *HST256params = (t_HST256_params *)k->params;
        return HST256params->zc_slc;
    }
    if (k->id == Reduction){
        t_reduction_params *REDparams = (t_reduction_params *)k->params;
        return REDparams->zc_slc;
    }
    if (k->id == GCEDD  || k->id == SCEDD || k->id == NCEDD || k->id == HCEDD){
        t_CEDD_params *CEDDparams = (t_CEDD_params *)k->params;
        return CEDDparams->zc_slc;
    }
    if (k->id == SPMV_CSRscalar){
        t_SPMV_params *SPMVparams = (t_SPMV_params *)k->params;
        return SPMVparams->zc_slc;
    }
    if (k->id == RCONV  || k->id == CCONV){
        t_CONV_params *CONVparams = (t_CONV_params *)k->params;
        return CONVparams->zc_slc;
    }
    if (k->id == TP) {
        t_TP_params *TPparams = (t_TP_params *)k->params;
        return TPparams->zc_slc;
    }
    if (k->id == DXTC) {
        t_DXTC_params *DXTCparams = (t_DXTC_params *)k->params;
        return DXTCparams->zc_slc;
    }
    
    return NULL;
}
    

// Calculate the number of blocks per SM to cover all cores of a SM
int calculate_pi (int block_size, int cores_per_SM)
{
    if (block_size >= cores_per_SM)
        return 1;
    else 
        return (int)ceil((double)block_size/cores_per_SM);
}

// Calculate execution time of pi blocks per SM
double calculate_pi_time (t_kernel_stub *kernel, int pi)
{
    struct timespec now1, now2;
    double init_time, curr_time;
    
    kernel->startKernel((void*)kernel);
    kernel->total_tasks = pi * kernel->kconf.numSMs; // Change number of SMs
    kernel->kconf.initial_blockID = 0;
    clock_gettime(CLOCK_REALTIME, &now1);
    kernel->launchSLCkernel((void *)kernel);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_REALTIME, &now2);
    init_time = (double)now1.tv_sec+(double)now1.tv_nsec*1e-9;
 	curr_time = (double)now2.tv_sec+(double)now2.tv_nsec*1e-9;

    //kernel->endKernel((void *)kernel);
    
    return (curr_time-init_time); 
}

int coexecution(t_kernel_stub *k0, int pi0, t_kernel_stub *k1, int pi1, int slowest_slice_idx)
{
    struct timespec now1, now2;
    double init_time, curr_time;

   /* k0->startKernel((void*)k0);
    k1->startKernel((void*)k1);
*/
    k0->total_tasks = pi0 * k0->kconf.numSMs; // Numer of block to be execute in a slice
    k0->kconf.initial_blockID = 0; // Initial block index
    k1->total_tasks = pi1 * k1->kconf.numSMs;
    k1->kconf.initial_blockID = 0;

    if (k0->kconf.gridsize.y == 0) k0->kconf.gridsize.y = 1;
    if (k1->kconf.gridsize.y == 0) k1->kconf.gridsize.y = 1;

    int remaining_blocks0 =  k0->kconf.gridsize.x * k0->kconf.gridsize.y; // Total block of the original kernel
    int remaining_blocks1 =  k1->kconf.gridsize.x * k1->kconf.gridsize.y;

    clock_gettime(CLOCK_REALTIME, &now1);

    int cont =0;
    while ( remaining_blocks0 >= pi0 * k0->kconf.numSMs && remaining_blocks1 >= pi1 * k1->kconf.numSMs) {
    
        if (slowest_slice_idx == 0) { 
            k0->launchSLCkernel((void *)k0); // Launch kernels. The one with longest slicing is launched first to garanty cke
            k1->launchSLCkernel((void *)k1);
        } 
        else
        {
            k1->launchSLCkernel((void *)k1); 
            k0->launchSLCkernel((void *)k0); 
        }
        cont++;

        k0->kconf.initial_blockID += pi0 * k0->kconf.numSMs;
        k1->kconf.initial_blockID += pi1 * k1->kconf.numSMs;

        remaining_blocks0 -= pi0 * k0->kconf.numSMs;
        remaining_blocks1 -= pi1 * k1->kconf.numSMs;

        cudaDeviceSynchronize();

    }


    clock_gettime(CLOCK_REALTIME, &now2);
    init_time = (double)now1.tv_sec+(double)now1.tv_nsec*1e-9;
    curr_time = (double)now2.tv_sec+(double)now2.tv_nsec*1e-9;
    double coexec_time =  curr_time - init_time;
   // printf("Concurrent executed-> executed tasks (%d, %d) time=%f cont=%d\n",
    //k0->kconf.gridsize.x - remaining_blocks0,  k1->kconf.gridsize.x - remaining_blocks1, coexec_time, cont);

    // * Sequential execution//

    k0->total_tasks = k0->kconf.gridsize.x * k0->kconf.gridsize.y - remaining_blocks0;
    k1->total_tasks = k1->kconf.gridsize.x * k1->kconf.gridsize.y - remaining_blocks1;
    k0->kconf.initial_blockID = 0;
    k1->kconf.initial_blockID = 0;

    clock_gettime(CLOCK_REALTIME, &now1);
    k0->launchSLCkernel((void *)k0);
    k1->launchSLCkernel((void *)k1);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_REALTIME, &now2);
    init_time = (double)now1.tv_sec+(double)now1.tv_nsec*1e-9;
    curr_time = (double)now2.tv_sec+(double)now2.tv_nsec*1e-9;
    double seq_time =  curr_time - init_time;
    char name[2][20];
    kid_from_index(k0->id, name[0]);
    kid_from_index(k1->id, name[1]);
    int ctas0, ctas1;
    if (slowest_slice_idx == 0){
        ctas0 = pi0;
        ctas1 = pi1;
    }else
    {
        ctas0 = pi1;
        ctas1 = pi0;
    }
    //printf("%s/%s, %d, %d, %.3f, %.2f, %.3f, %d\n", name[slowest_slice_idx], name[(slowest_slice_idx + 1) % 2], ctas0, ctas1, seq_time * 1000, seq_time/coexec_time, coexec_time * 1000/(double)cont, cont);

    return 0;
}

// Like the cCida code in github
int launch_like_original_cCuda(t_kernel_stub **kstub, double ktime_s[2])
{

    t_kernel_stub *k0, *k1;
    double rate;
    int slowest_kernel_slice;
    struct timespec now1, now2;
    double init_time, curr_time;

    k0 = kstub[0];
    k1 = kstub[1];

    if (ktime_s[0] > ktime_s[1]){
        slowest_kernel_slice = 0;
        rate = ktime_s[0]/ktime_s[1];
    }
    else {
        slowest_kernel_slice = 1;
        rate = round(ktime_s[1]/ktime_s[0]);
    }

    k0->total_tasks = k0->kconf.numSMs; // Numer of block to be execute in a slice
    k0->kconf.initial_blockID = 0; // Initial block index
    k1->total_tasks = k1->kconf.numSMs;
    k1->kconf.initial_blockID = 0;

    if (k0->kconf.gridsize.y == 0) k0->kconf.gridsize.y = 1;
    if (k1->kconf.gridsize.y == 0) k1->kconf.gridsize.y = 1;

    int remaining_blocks0 =  k0->kconf.gridsize.x * k0->kconf.gridsize.y; // Total block of the original kernel
    int remaining_blocks1 =  k1->kconf.gridsize.x * k1->kconf.gridsize.y;

    int iteration = 0;
    clock_gettime(CLOCK_REALTIME, &now1);

    while ( remaining_blocks0 >= k0->kconf.numSMs && remaining_blocks1 >= k1->kconf.numSMs) {

        if (slowest_kernel_slice == 0) { 
            k0->launchSLCkernel((void *)k0); // Launch kernels. The one with longest slicing is launched first to garanty cke
            remaining_blocks0 -= k0->kconf.numSMs;
            
            int start  = (int) round((double)iteration * rate);
            int end = (int) round ((double) (iteration +1) * rate); 

            for (int i=start; i<end ; i++) {
                k1->launchSLCkernel((void *)k1); // Iterlave slice of k1;
                remaining_blocks1 -= k1->kconf.numSMs;
                if (remaining_blocks1 <= 0)
                    break;
            }
            iteration++;
        } 
        else
        {
            k1->launchSLCkernel((void *)k1); // Launch kernels. The one with longest slicing is launched first to garanty cke
            remaining_blocks1 -= k1->kconf.numSMs;

            int start  = (int) round((double)iteration * rate);
            int end = (int) round ((double) (iteration +1) * rate); 

            for (int i=start; i< end; i++) {
                k0->launchSLCkernel((void *)k0); // Iterlave slice of k1;
                remaining_blocks0 -= k1->kconf.numSMs;
                if (remaining_blocks0 <= 0)
                    break;
            }
            iteration++;
        }
    }

    // Suncronize with the stream of the finished kernel
    if (remaining_blocks0 <  k0->kconf.numSMs )
        cudaStreamSynchronize(*(k0->execution_s));
    else    
        cudaStreamSynchronize(*(k1->execution_s));
    

    clock_gettime(CLOCK_REALTIME, &now2);
    init_time = (double)now1.tv_sec+(double)now1.tv_nsec*1e-9;
    curr_time = (double)now2.tv_sec+(double)now2.tv_nsec*1e-9;
    double coexec_time =  curr_time - init_time;
    cudaDeviceSynchronize();

   // printf("Concurrent executed-> executed tasks (%d, %d) time=%f cont=%d\n",
    //k0->kconf.gridsize.x - remaining_blocks0,  k1->kconf.gridsize.x - remaining_blocks1, coexec_time, cont);

    // * Sequential execution//

    k0->total_tasks = k0->kconf.gridsize.x * k0->kconf.gridsize.y - remaining_blocks0;
    k1->total_tasks = k1->kconf.gridsize.x * k1->kconf.gridsize.y - remaining_blocks1;
    k0->kconf.initial_blockID = 0;
    k1->kconf.initial_blockID = 0;

    clock_gettime(CLOCK_REALTIME, &now1);
    k0->launchSLCkernel((void *)k0);
    k1->launchSLCkernel((void *)k1);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_REALTIME, &now2);
    init_time = (double)now1.tv_sec+(double)now1.tv_nsec*1e-9;
    curr_time = (double)now2.tv_sec+(double)now2.tv_nsec*1e-9;
    double seq_time =  curr_time - init_time;
    char name[2][20];
    kid_from_index(k0->id, name[0]);
    kid_from_index(k1->id, name[1]);
    //printf("%s/%s, %.3f, %.3f, %.2f\n", name[slowest_kernel_slice], name[(slowest_kernel_slice+ 1) % 2], coexec_time * 1000, seq_time * 1000, seq_time/coexec_time);
    exit(0);
    return 0;

}

int launch_one_kernel(t_kernel_stub *kstub)
{
    int *zc_slc;

    if ((zc_slc= get_cta_counter_position(kstub)) == NULL){
        printf("Error: zc_slc0\n");
        return -1;
    }

    cudaMemset(zc_slc, 0, sizeof(int));

    int saved_total_tasks = kstub->total_tasks;

    kstub->total_tasks = 56;

    while (saved_total_tasks> kstub->total_tasks)
    {
         kstub->launchSLCkernel((void *)kstub);
         saved_total_tasks -= kstub->total_tasks;
    }

    cudaDeviceSynchronize();

    return 0;
}

int launch_two_kernels(t_kernel_stub **kstub)
{
    t_kernel_stub *k0, *k1;
    int *zc_slc0, *zc_slc1;

    k0 = kstub[0];
    k1 = kstub[1];

    if ((zc_slc0= get_cta_counter_position(k0)) == NULL){
        printf("Error: zc_slc0\n");
        return -1;
    }

    if ((zc_slc1= get_cta_counter_position(k1)) == NULL){
        printf("Error: zc_slc0\n");
        return -1;
    }

    cudaMemset(zc_slc0, 0, sizeof(int));
    cudaMemset(zc_slc1, 0, sizeof(int));

    int saved_total_tasks0 = k0->total_tasks;
    int saved_total_tasks1 = k1->total_tasks;

    k0->total_tasks = 56;
    k1->total_tasks = 224 - 56;

    while (saved_total_tasks0 > k0->total_tasks && saved_total_tasks1 > k1->total_tasks)
    {
         k0->launchSLCkernel((void *)k0);
         saved_total_tasks0 -= k0->total_tasks;

         k1->launchSLCkernel((void *)k1);
         saved_total_tasks1 -= k1->total_tasks;
    }

    cudaDeviceSynchronize();

    return 0;
}


#define NUM_RUNS 1

// Like  cCuda code but better: two streams, no sychronization but allowing slices with more that one cta per SM 
int launch_improved_cCuda(t_kernel_stub **kstub, int max_ctas[2])
{

    t_kernel_stub *k0, *k1;
    double rate;
    struct timespec now1, now2;
    double init_time, curr_time, coexec_time;

    double speedup[32][32];
    
    k0 = kstub[0];
    k1 = kstub[1];

    int *zc_slc0, *zc_slc1;
    if ((zc_slc0 = get_cta_counter_position(kstub[0])) == NULL){
        printf("Error: zc_slc0\n");
        return -1;
    }
    if ((zc_slc1 = get_cta_counter_position(kstub[1])) == NULL){
        printf("Error: zc_slc1\n");
        return -1;
    }

    if (k0->kconf.gridsize.y == 0) k0->kconf.gridsize.y = 1;
    if (k1->kconf.gridsize.y == 0) k1->kconf.gridsize.y = 1;

    int f0, f1;
    if (max_ctas[0] >= max_ctas[1]){
        f0 = round(max_ctas[0]/max_ctas[1]);
        f1 = 1;
    }
    else {
        f1 = round(max_ctas[1]/max_ctas[0]);
        f0 = 1;
    }
    
    int pi0, pi1, shortest_kernel=-1;
    int launched_ctas0, launched_ctas1;

    int saved_total_tasks0 = k0->total_tasks;
    if (k0->id == HST256)
        saved_total_tasks0 = k0->kconf.gridsize.x; // Mayoria de kernels gridsize coinicide con total tasks pero no en HST 
    int saved_total_tasks1 = k1->total_tasks;
    if (k1->id == HST256)
        saved_total_tasks1 = k1->kconf.gridsize.x;

    for (int iter =0; iter <NUM_RUNS; iter++) // Como salen valores vaiables en cada iteracion vamos a hacer la media
    {
    for (pi0=f0; pi0 < max_ctas[0]; pi0+=f0)
    {
        pi1 = max_ctas[1] - pi0/f0 * f1;

        int remaining_blocks0 =  saved_total_tasks0 ; // Total block of the original kernel
        int remaining_blocks1 =  saved_total_tasks1;

        k0->total_tasks = pi0 * k0->kconf.numSMs; // Numer of blocks to be execute in a slice
        k0->kconf.initial_blockID = 0; // Initial block index
        cudaMemset(zc_slc0, 0, sizeof(int)); //launched ctas counter
        k1->total_tasks = pi1 * k1->kconf.numSMs;
        k1->kconf.initial_blockID = 0;
        cudaMemset(zc_slc1, 0, sizeof(int));

        //cudaMemcpy(&launched_ctas0, zc_slc0, sizeof(int), cudaMemcpyDeviceToHost);
        //cudaMemcpy(&launched_ctas1, zc_slc1, sizeof(int), cudaMemcpyDeviceToHost);

        clock_gettime(CLOCK_REALTIME, &now1);
        
        while (1){

            if (remaining_blocks0 >= pi0 * k0->kconf.numSMs) {
                k0->launchSLCkernel((void *)k0); // Launch kernels. The one with longest slicing is launched first to garanty cke
                remaining_blocks0 -= k0->total_tasks;
                k0->kconf.initial_blockID +=  k0->total_tasks;
            }
         
            
            if (remaining_blocks1 >= pi1 * k1->kconf.numSMs) {
                k1->launchSLCkernel((void *)k1); // Launch kernels. The one with longest slicing is launched first to garanty cke
                remaining_blocks1 -= k1->total_tasks;
                k1->kconf.initial_blockID +=  k1->total_tasks;
            }

            if (cudaStreamQuery (*(k0->execution_s)) == cudaSuccess && remaining_blocks0 < k0->total_tasks )
            {
                shortest_kernel = 0;
                cudaMemcpyAsync(&launched_ctas1, zc_slc1, sizeof(int), cudaMemcpyDeviceToHost, *(k0->preemp_s));
                cudaMemcpyAsync(&launched_ctas0, zc_slc0, sizeof(int), cudaMemcpyDeviceToHost, *(k0->preemp_s));
                cudaStreamSynchronize(*k0->preemp_s);
                
                clock_gettime(CLOCK_REALTIME, &now2);
                init_time = (double)now1.tv_sec+(double)now1.tv_nsec*1e-9;
                curr_time = (double)now2.tv_sec+(double)now2.tv_nsec*1e-9;
                coexec_time =  curr_time - init_time;
                break;
            }

            if (cudaStreamQuery (*(k1->execution_s)) == cudaSuccess && remaining_blocks1 < k1->total_tasks  )
            {
                shortest_kernel = 1;
                cudaMemcpyAsync(&launched_ctas0, zc_slc0, sizeof(int), cudaMemcpyDeviceToHost, *(k0->preemp_s));
                cudaMemcpyAsync(&launched_ctas1, zc_slc1, sizeof(int), cudaMemcpyDeviceToHost, *(k0->preemp_s));
                cudaStreamSynchronize(*k0->preemp_s);

                clock_gettime(CLOCK_REALTIME, &now2);
                init_time = (double)now1.tv_sec+(double)now1.tv_nsec*1e-9;
                curr_time = (double)now2.tv_sec+(double)now2.tv_nsec*1e-9;
                coexec_time =  curr_time - init_time;
                break;
            }
        }

        cudaDeviceSynchronize();
        //printf("sk=%d %d %d %d %d %f %f \n", shortest_kernel, remaining_blocks0, remaining_blocks1, launched_ctas0, launched_ctas1,
        //        (double)launched_ctas0/(double)(k0->kconf.gridsize.x * k0->kconf.gridsize.y), (double)launched_ctas1/(double)(k1->kconf.gridsize.x * k1->kconf.gridsize.y));

        // Sequential execution
        k0->total_tasks = launched_ctas0;
        k1->total_tasks = launched_ctas1;
        k0->kconf.initial_blockID = 0;
        k1->kconf.initial_blockID = 0;
            
        clock_gettime(CLOCK_REALTIME, &now1);
        k0->launchSLCkernel((void *)k0);
        cudaDeviceSynchronize();
        k1->launchSLCkernel((void *)k1);
        cudaDeviceSynchronize();
        clock_gettime(CLOCK_REALTIME, &now2);
        init_time = (double)now1.tv_sec+(double)now1.tv_nsec*1e-9;
        curr_time = (double)now2.tv_sec+(double)now2.tv_nsec*1e-9;
        double seq_time =  curr_time - init_time;
        char name[2][20];
        kid_from_index(k0->id, name[0]);
        kid_from_index(k1->id, name[1]);
        //printf("%s/%s, %d/%d, %.3f, %.3f, %.2f\n", name[shortest_kernel], name[(shortest_kernel+ 1) % 2], pi0, pi1, coexec_time * 1000, seq_time * 1000, seq_time/coexec_time);
        speedup[iter][pi0] = seq_time/coexec_time;
    }
    }

    double avrg_speedup[32];
    for (pi0=f0; pi0 < max_ctas[0]; pi0+=f0) {
        avrg_speedup[pi0]=0;
        for (int iter =0; iter < NUM_RUNS; iter++)  // Como salen valores vaiables en cada iteracion vamos a hacer la media
            avrg_speedup[pi0] += speedup[iter][pi0];
        avrg_speedup[pi0] /= NUM_RUNS;
    }

    char name[2][20];
    kid_from_index(k0->id, name[0]);
    kid_from_index(k1->id, name[1]);

    double max_speedup=0, min_speedup=100000;
    for (pi0=f0; pi0 < max_ctas[0]; pi0+=f0){
        pi1 = max_ctas[1] - pi0/f0 * f1;
        printf("%s/%s, %d/%d, %.2f\n", name[shortest_kernel], name[(shortest_kernel+ 1) % 2], pi0, pi1, avrg_speedup[pi0]);
        if (min_speedup > avrg_speedup[pi0])
            min_speedup = avrg_speedup[pi0];
        if (max_speedup < avrg_speedup[pi0])
            max_speedup = avrg_speedup[pi0];
    }
    //printf("%s, %s, %f, %f\n", name[0], name[1], max_speedup, min_speedup);

    return 0;
}

// Get the pi value that obtain the most similar execution time among kernels
// By brute force: increase the number of block of the fastest kernel block until  kernel slice becomes the slowest (and then discount one)
int calculate_fitting_pi(t_kernel_stub **kstub, int ctas, int kernel_idx, int *val0, int *val1)
{
    int pi0, pi1;
    t_kernel_stub *k0, *k1;
    k0 = kstub[0];
    k1 = kstub[1];
    
    if (kernel_idx == 0){
        pi0 = ctas;
        pi1 = ctas;
    }
    else{
        pi0 = ctas;
        pi1 = ctas;
    }

    // Find best pi value of the fastest 
    int end=0;
    int k0_inc=0, k1_inc = 0;
    if (k0->kconf.gridsize.y == 0) k0->kconf.gridsize.y = 1;
    if (k1->kconf.gridsize.y == 0) k1->kconf.gridsize.y = 1;
    int total_blocks0 =  k0->kconf.gridsize.x * k0->kconf.gridsize.y; // Total block of the original kernel
    int total_blocks1 =  k1->kconf.gridsize.x * k1->kconf.gridsize.y;

    while (end != 1) {

        k0->total_tasks = pi0 * k0->kconf.numSMs; // Numer of block to be execute in a slice
        k0->kconf.initial_blockID = 0; // Initial block index
        k1->total_tasks = pi1 * k1->kconf.numSMs;
        k1->kconf.initial_blockID = 0;

        //printf("Values pi0=%d, pi1=%d\n", pi0, pi1);

        if (kernel_idx == 0) {
            k0->launchSLCkernel((void *)k0); // Launch kernels. The one with longest slicing is launched first to guaranty cke
            k1->launchSLCkernel((void *)k1);
        } 
        else
        {
            k1->launchSLCkernel((void *)k1); 
            k0->launchSLCkernel((void *)k0); 
        }

        while (1) { // Wait until fastest kernel slice finishes 
			
            if (cudaStreamQuery(*k0->execution_s) == cudaSuccess) {
                //mprintf("0 - %d %d\n", pi0, pi1);
               if (kernel_idx == 1){ // If pi1 has the longest slice
                    pi0++; // increase number of blocks in kernel0 slice because k0 has finished earlier
                    if (pi0 > total_blocks0){
                        printf("Error: pi0 too high\n");
                        return -1;
                    }
                    k0_inc = 1;
                }
                else{
                   /* if (first_execution == 1) {
                        printf("Error: Kernel 0 is the fastest (it shouldn´t be)\n"); //If pi0 has de longest slice ...
                        return -1;
                    }*/
                    pi1--; // KErnel1 slice is now the slowest: discount un kernel block so it is a bit faster that kernel0 slice
                    if (pi1 == 0) {
                        printf("Error: pi1 value\n");
                        return -1;
                    }
                    if (k1_inc)
                        end = 1;
                }
                cudaDeviceSynchronize();
                break;
            }
            
            if (cudaStreamQuery(*k1->execution_s) == cudaSuccess) {
                //printf("1 - %d %d\n", pi0, pi1);
                if (kernel_idx == 0) {
                    pi1++;
                    if (pi1 > total_blocks1){
                        printf("Error: pi1 too high\n");
                        return -1;
                    }
                    k1_inc = 1;
                }
                else {
                    /*if (first_execution == 1) {
                        printf("Error: Kernel 1 is the fastest (it shouldn´t be)\n"); //If pi0 has de longest slice ...
                        return -1;
                    }*/
                    pi0--;
                    if (pi0 == 0) {
                        printf("Error: pi0 value\n");
                        return -1;
                    }
                    if (k0_inc)
                        end = 1;
                }
                cudaDeviceSynchronize();
                break;
            }
        }
        //printf("pi0=%d pi1=%d\n", pi0, pi1);
    }

    *val0 = pi0;
    *val1 = pi1;

    return 0;
        
}

int calculate_fitting_pi_ver2(t_kernel_stub **kstub, int ctas, int kernel_idx, int *val0, int *val1)
{
    
    cudaEvent_t start0, stop0, start1, stop1;
    
    int pi0, pi1;
    t_kernel_stub *k0, *k1;
    k0 = kstub[0];
    k1 = kstub[1];
    
    if (kernel_idx == 0){
        pi0 = ctas;
        pi1 = ctas;
    }
    else{
        pi0 = ctas;
        pi1 = ctas;
    }

    cudaEventCreate(&start0); cudaEventCreate(&start1);
    cudaEventCreate(&stop0);  cudaEventCreate(&stop1);

    // Find best pi value of the fastest 
    int end=0;
    while (end != 1) {

        k0->total_tasks = pi0 * k0->kconf.numSMs; // Numer of block to be execute in a slice
        k0->kconf.initial_blockID = 0; // Initial block index
        k1->total_tasks = pi1 * k1->kconf.numSMs;
        k1->kconf.initial_blockID = 0;
        
        //printf("Values pi0=%d, pi1=%d\n", pi0, pi1);

        if (kernel_idx == 0) {
            cudaEventRecord(start0, *(k0->execution_s));
            k0->launchSLCkernel((void *)k0); // Launch kernels. The one with longest slicing is launched first to guaranty cke
            cudaEventRecord(stop0, *(k0->execution_s));
            
            cudaEventRecord(start1, *(k1->execution_s));
            k1->launchSLCkernel((void *)k1);
            cudaEventRecord(stop1, *(k1->execution_s));
        } 
        else
        {
            cudaEventRecord(start1, *(k1->execution_s));
            k1->launchSLCkernel((void *)k1);    // Launch kernels. The one with longest slicing is launched first to guaranty cke
            cudaEventRecord(stop1, *(k1->execution_s));

            cudaEventRecord(start0, *(k0->execution_s));
            k0->launchSLCkernel((void *)k0); 
            cudaEventRecord(stop0, *(k0->execution_s));
            
        }

        float ms0, ms1;
        int str0 = 0, str1 = 0;

        while (str0 != 1 || str1 !=1) { // Wait until fastest kernel slice finishes 
			
            if (cudaStreamQuery(*k0->execution_s) == cudaSuccess) {
                cudaEventSynchronize(stop0);
                cudaEventElapsedTime(&ms0, start0, stop0);
                str0 = 1;
            }

            if (cudaStreamQuery(*k1->execution_s) == cudaSuccess) {
                cudaEventSynchronize(stop1);
                cudaEventElapsedTime(&ms1, start1, stop1);
                str1 = 1;
            }

        }

        //printf("ms0=%f ms1=%f\n", ms0, ms1);

        if (kernel_idx == 0) { 
            if (ms0 < ms1){
                pi1--;
                end = 1;
            }
            else {
                pi1++;
            }
        }
        else {
            if (ms1 < ms0)
            { 
                pi0--;
                end =1;
            }
            else {
                pi0++;
            }
        }

        cudaDeviceSynchronize();

    }

    *val0 = pi0;
    *val1 = pi1;

    return 0;
        
}

int get_slowest_kernel_ver2(t_kernel_stub *kstub[2], int max_ctas[2], double time[2])
{
    cudaEvent_t start,stop;
    t_kernel_stub *k0, *k1;
    k0 = kstub[0];
    k1 = kstub[1];

    // Detect the slowestt slice  running the same number of blocks per kernel
    k0->total_tasks = k0->kconf.numSMs; //k0->kconf.numSMs; // Numer of blocks to be execute in a slice
    k0->kconf.initial_blockID = 0; // Initial block index
    k1->total_tasks = k1->kconf.numSMs; //k1->kconf.numSMs;
    k1->kconf.initial_blockID = 0;

    // Warming up: to redice the laluching command execution time
    for (int i=0; i<5;i++) {
        k0->launchSLCkernel((void *)k0);
        k1->launchSLCkernel((void *)k1); // Launch both kernels to detect the slowest one
        cudaDeviceSynchronize();
    }

    // Measure execution time of a slice of kernels, containing num_max_ctas_per_SM * num_SMs

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, *(k0->execution_s));

    k0->launchSLCkernel((void *)k0);
    
    cudaEventRecord(stop, *(k0->execution_s));
    cudaEventSynchronize(stop);
    float ms0;
    cudaEventElapsedTime(&ms0, start, stop);

    cudaEventRecord(start, *(k1->execution_s));

    k1->launchSLCkernel((void *)k1);
    
    cudaEventRecord(stop, *(k1->execution_s));
    cudaEventSynchronize(stop);
    float ms1;
    cudaEventElapsedTime(&ms1, start, stop);
    time[0] = ms0;
    time[1] = ms1;

    //printf("%f, %f\n", ms0, ms1);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (ms0 > ms1)
        return 0;
    else
        return 1;

}

int get_slowest_kernel(t_kernel_stub *kstub[2])
{
    int slowest_index;
    t_kernel_stub *k0, *k1;
    k0 = kstub[0];
    k1 = kstub[1];

    // Detect the slowestt slice  running the same number of blocks per kernel
    int pi0 = 1;
    int pi1 = 1;
    k0->total_tasks = pi0 * k0->kconf.numSMs; // Numer of blocks to be execute in a slice
    k0->kconf.initial_blockID = 0; // Initial block index
    k1->total_tasks = pi1 * k1->kconf.numSMs;

    // Warming up: to reduce launching command execution time
    /*for (int i=0; i<5;i++) {
        k0->launchSLCkernel((void *)k0);
        k1->launchSLCkernel((void *)k1);
        cudaDeviceSynchronize();
    }*/
    
    k0->launchSLCkernel((void *)k0);
    k1->launchSLCkernel((void *)k1); // Launch both kernels to detect the slowest one

    while (1) { // Wait until fastest kernel slice finishes 
			
        if (cudaStreamQuery(*k0->execution_s) == cudaSuccess) {
            slowest_index = 1;
            break;
        }
            
        if (cudaStreamQuery(*k1->execution_s) == cudaSuccess) {
            slowest_index = 0;
            break;
        } 
    }

    cudaDeviceSynchronize();
    exit(0);
    return slowest_index;
}

int get_max_ctas(t_kernel_stub * kstub, char * device_name)
{
    if (kstub->kconf.max_persistent_blocks != 0) {
        return kstub->kconf.max_persistent_blocks;
    }

    switch(kstub->id)
    {
        case VA:
        case BS:
            if (strcmp(device_name, "Tesla K40c") == 0) {
                return 16;
            }
            else {
                return 8;
            }
        case MM: 
        case PF:
        case HST256:
        case Reduction:
        case GCEDD:
        case SCEDD:
        case NCEDD:
        case HCEDD:
        case TP:
        case DXTC:
            return 8;

        case SPMV_CSRscalar:
        case CCONV:
            return 16;
        
        case RCONV:
            return 32;

        default:
            printf("Error: Kernel not known\n");
            return -1;  
    }
}



int main (int argc, char *argv[])
{
    cudaError_t err;

    t_Kernel kid[2];
    int max_ctas[2]; // MaX ctas per SM
    int deviceId = atoi(argv[1]);
    kid[0] = kid_from_name(argv[2]);
    kid[1] = kid_from_name(argv[3]);
    
    if (argc <4) {
        printf("Error in arguments: cCuda device_id kernel_name1 kernel_name2 \n");
        return -1;
    }

    // Select device
	cudaSetDevice(deviceId);
	cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);	
    //printf("Device=%s\n", deviceProp.name);
    char * device_name = deviceProp.name;
    int cores_per_SM = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);

	/** Create commom streams for all kernels: two for asynchronous transfers, one for preemption commands*/
	cudaStream_t *transfers_s;
	transfers_s = (cudaStream_t *)calloc(2, sizeof(cudaStream_t));
	
	for (int i=0;i<2;i++){
		err = cudaStreamCreate(&transfers_s[i]);
		checkCudaErrors(err);
    } 
    
    cudaStream_t preemp_s;
    checkCudaErrors(cudaStreamCreateWithFlags(&preemp_s, cudaStreamNonBlocking)); 
    
     // Create kstubs. Check if dependent kernels must be launched in advance
    t_kernel_stub *kstub[2];
    kstub[0] = NULL; kstub[1]=NULL;

    for (int i=0; i< 2; i++) {
        switch (kid[i]){
            
            case CCONV:
                t_kernel_stub *RCONV_stub;
                create_stubinfo(&RCONV_stub, deviceId, RCONV, transfers_s, &preemp_s);
                RCONV_stub->startMallocs((void*)RCONV_stub);
                RCONV_stub->startTransfers((void*)RCONV_stub);
                cudaDeviceSynchronize();
                RCONV_stub->total_tasks =  RCONV_stub->kconf.gridsize.x;
                RCONV_stub->kconf.initial_blockID = 0;
                //RCONV_stub->launchSLCkernel((void*)RCONV_stub);
                //cudaDeviceSynchronize();

                create_stubinfo_with_params(&kstub[i], deviceId, kid[i], transfers_s, &preemp_s, (void *)RCONV_stub->params);
                break;

            case SCEDD:
                t_kernel_stub *GCEDD_stub;
                create_stubinfo(&GCEDD_stub, deviceId, GCEDD, transfers_s, &preemp_s);
                GCEDD_stub->startMallocs((void*)GCEDD_stub);
                GCEDD_stub->startTransfers((void*)GCEDD_stub);
                GCEDD_stub->launchORIkernel((void*)GCEDD_stub); // Launch original as concurrency is no goint to be check on this kernel
                cudaDeviceSynchronize();
                create_stubinfo_with_params(&kstub[i], deviceId, kid[i], transfers_s, &preemp_s, (void *)GCEDD_stub->params);
                break;

            case NCEDD:
                create_stubinfo(&GCEDD_stub, deviceId, GCEDD, transfers_s, &preemp_s);
                GCEDD_stub->startMallocs((void*)GCEDD_stub);
                GCEDD_stub->startTransfers((void*)GCEDD_stub);
                GCEDD_stub->launchORIkernel((void*)GCEDD_stub);
                cudaDeviceSynchronize();

                t_kernel_stub *SCEDD_stub;
                create_stubinfo_with_params(&SCEDD_stub, deviceId, SCEDD, transfers_s, &preemp_s, (void *)GCEDD_stub->params);
                SCEDD_stub->startMallocs((void*)SCEDD_stub);
                SCEDD_stub->startTransfers((void*)SCEDD_stub);
                SCEDD_stub->launchORIkernel((void*)SCEDD_stub);
                cudaDeviceSynchronize();

                create_stubinfo_with_params(&kstub[i], deviceId, kid[i], transfers_s, &preemp_s, (void *)GCEDD_stub->params);
                break;

            case HCEDD:
                create_stubinfo(&GCEDD_stub, deviceId, GCEDD, transfers_s, &preemp_s);
                GCEDD_stub->startMallocs((void*)GCEDD_stub);
                GCEDD_stub->startTransfers((void*)GCEDD_stub);
                GCEDD_stub->launchORIkernel((void*)GCEDD_stub);
                cudaDeviceSynchronize();

                create_stubinfo_with_params(&SCEDD_stub, deviceId, SCEDD, transfers_s, &preemp_s, (void *)GCEDD_stub->params);
                SCEDD_stub->startMallocs((void*)SCEDD_stub);
                SCEDD_stub->startTransfers((void*)SCEDD_stub);
                SCEDD_stub->launchORIkernel((void*)SCEDD_stub);
                cudaDeviceSynchronize();

                t_kernel_stub *NCEDD_stub;
                create_stubinfo_with_params(&NCEDD_stub, deviceId, NCEDD, transfers_s, &preemp_s, (void *)GCEDD_stub->params);
                NCEDD_stub->startMallocs((void*)NCEDD_stub);
                NCEDD_stub->startTransfers((void*)NCEDD_stub);
                NCEDD_stub->launchORIkernel((void*)NCEDD_stub);
                cudaDeviceSynchronize();
                
                create_stubinfo_with_params(&kstub[i], deviceId, kid[i], transfers_s, &preemp_s, (void *)GCEDD_stub->params);
                break;

            default:
                create_stubinfo(&kstub[i], deviceId, kid[i], transfers_s, &preemp_s);
        }
    }

    if ((max_ctas[0] = get_max_ctas(kstub[0], device_name))<0)
        return -1;
    if ((max_ctas[1] = get_max_ctas(kstub[1], device_name))<0)
        return -1;

    // Memeory allocation and HtD trasnfers
    
    kstub[0]->startMallocs((void*)kstub[0]);
    kstub[0]->startTransfers((void*)kstub[0]);
    kstub[1]->startMallocs((void*)kstub[1]);
    kstub[1]->startTransfers((void*)kstub[1]);
    cudaDeviceSynchronize();
    //coexecution(kstub[0],  kstub[0]->kconf.gridsize.x/kstub[0]->kconf.numSMs, kstub[1], 4, 1);
    //return 0;

   /* launch_two_kernels(kstub);
    return 0;

    launch_one_kernel(kstub[0]);
    return 0;
*/
    launch_improved_cCuda(kstub, max_ctas);
    return 0;

    //* Calclulate slowest slice (each slice has the same number of ctas) 
    double ktime_slice[2];
    int kernel_idx = get_slowest_kernel_ver2(kstub, max_ctas, ktime_slice);
    
    // launch_like_original_cCuda(kstub, ktime_slice); Implementing cCuda as indicates in github distribution
    
    for (int num_ctas=1; num_ctas<max_ctas[kernel_idx]; num_ctas++) {
        int pi0, pi1;
        if (calculate_fitting_pi_ver2(kstub, num_ctas, kernel_idx, &pi0, &pi1) < 0)
            continue;
        if (pi0 < 0 || pi1 < 0){
            printf("Error: num_ctas cannot be negative\n");
            return -1;
        }
        if (kstub[0]->kconf.gridsize.y == 0) kstub[0]->kconf.gridsize.y = 1;
        if (kstub[1]->kconf.gridsize.y == 0) kstub[1]->kconf.gridsize.y = 1;
        if (pi0 * kstub[0]->kconf.numSMs > kstub[0]->kconf.gridsize.x * kstub[0]->kconf.gridsize.y || pi1 * kstub[1]->kconf.numSMs> kstub[1]->kconf.gridsize.x * kstub[1]->kconf.gridsize.y ) {
            printf("Error: Too much ctas\n");
        }
        else
            // coexecution //
            coexecution(kstub[0], pi0, kstub[1], pi1, kernel_idx);
    }

    return 0;
}




    


