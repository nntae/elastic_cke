#include <stdio.h>          /* printf()                 */
#include <stdlib.h>         /* exit(), malloc(), free() */
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>      /* key_t, sem_t, pid_t      */
#include <sys/shm.h>        /* shmat(), IPC_RMID        */
#include <errno.h>          /* errno, ECHILD            */
#include <semaphore.h>      /* sem_open(), sem_destroy(), sem_wait().. */
#include <fcntl.h>          /* O_CREAT, O_EXEC          */
#include <pthread.h>
#include <time.h>

#include <cuda_profiler_api.h>
#include <helper_functions.h>   // helper functions for string parsing
#include <helper_cuda.h>   
#include "elastic_kernel.h"

int run_original(t_kernel_stub *kstub, double *exectime_s)
{
	//cudaEvent_t start, stop;
	//float elapsedTime;
	
	//cudaEventCreate(&start);
	//cudaEventRecord(start, 0);
	
	kstub->launchORIkernel(kstub);
	cudaDeviceSynchronize();
	
	//cudaEventCreate(&stop);
	//cudaEventRecord(stop,0);
	//cudaEventSynchronize(stop);

	//cudaEventElapsedTime(&elapsedTime, start, stop);
	
	//*exectime_s = (double)elapsedTime/1000;
	
	return 0;
}


typedef struct{
	t_kernel_stub **kstubs;
	int index; // Index in kstubs array
}t_args;

void *launch_app(void *arg)
{
	t_args *args;
	
	args = (t_args *)arg;
	int index = args->index;
	t_kernel_stub *kstub = args->kstubs[index];
	
	printf("Launching kid=%d\n", kstub->id);
	
	int deviceId = 2;
	cudaSetDevice(deviceId);
	
	double exec_time;
	run_original(kstub, &exec_time);
	
	 if (kstub->id == RCONV) {
		kstub = args->kstubs[index + 1];
		run_original(kstub, &exec_time);
	 }
	 
	 if (kstub->id == GCEDD) {
		kstub = args->kstubs[index + 1];
		run_original(kstub, &exec_time);
		
		kstub = args->kstubs[index + 2];
		run_original(kstub, &exec_time);
		
		kstub = args->kstubs[index + 3];
		run_original(kstub, &exec_time);
	 }
	 
	 pthread_exit(NULL);
}

int hyperQ_threads()
{
	
	t_Kernel kid[9];
	kid[0]=MM;
	kid[1]=VA;
	kid[2]=BS;
	kid[3]=Reduction;
	kid[4]=PF;
	kid[5]=GCEDD; // Ojo: en profiling.cu se procesan tambien los tres kernels restantes de la aplicacion
	kid[6]=SPMV_CSRscalar;
	kid[7]=RCONV; // Ojo: en profiling se procesa tambien CCONV
	kid[8]=HST256;
	
	int num_kernels = 2;
	
	// context and streams
	
	cudaError_t err;

	// Select device
	int deviceId = 2;
	cudaSetDevice(deviceId);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, deviceId);	
	printf("Device=%s\n", deviceProp.name);
	
	/** Create commom streams for all kernels: two for asynchronous transfers, one for preemption commands*/
	cudaStream_t *transfers_s;
	transfers_s = (cudaStream_t *)calloc(2, sizeof(cudaStream_t));
	
	for (int i=0;i<2;i++){
		err = cudaStreamCreate(&transfers_s[i]);
		checkCudaErrors(err);
	}
	
	cudaStream_t preemp_s;
	checkCudaErrors(cudaStreamCreateWithFlags(&preemp_s, cudaStreamNonBlocking)); 
		
	// Create kstbus
	int cont = 0;			
	t_kernel_stub **kstubs = (t_kernel_stub **)calloc(13, sizeof(t_kernel_stub*)); // 13 is the max number of kernels for all app
	
	int index[9];
	for (int i=0; i< num_kernels; i++) {
		
		index[i] = cont;
		create_stubinfo(&kstubs[cont], deviceId, kid[i], transfers_s, &preemp_s);
		cont++;
		
		if (kid[i] == RCONV){ // RCONV params struct must be passed to CCONV 
			create_stubinfo_with_params(&kstubs[cont], deviceId, CCONV, transfers_s, &preemp_s, (void *)kstubs[cont-1]->params);
			cont++;
		}
		
		if (kid[i] == GCEDD){
			create_stubinfo_with_params(&kstubs[cont], deviceId, SCEDD, transfers_s, &preemp_s, (void *)kstubs[cont-1]->params);
			cont++;
			
			create_stubinfo_with_params(&kstubs[cont], deviceId, NCEDD, transfers_s, &preemp_s, (void *)kstubs[cont-2]->params);
			cont++;
			
			create_stubinfo_with_params(&kstubs[cont], deviceId, HCEDD, transfers_s, &preemp_s, (void *)kstubs[cont-3]->params);
			cont++;
		}
	}
		
	// make HtD transfers of all kernels
	make_transfers(kstubs, cont);
	
	// Create threads to lauch app
	
	t_args args[9];
	for (int i=0; i<9; i++)
		args[i].kstubs = kstubs;
	pthread_t *thid = (pthread_t *) calloc(num_kernels, sizeof(pthread_t));
	for (int i=0; i<num_kernels; i++) {
		args[i].index = index[i];	
		pthread_create(&thid[i], NULL, launch_app, &args[i]);
	}
	
	for (int i=0; i<num_kernels; i++)
		pthread_join(thid[i], NULL);
	
	cudaDeviceSynchronize();
	
	return 0;
}
		

int main (int argc, char **argv)
{
	
	int it;                        /*      loop variables          */
    key_t shmkey;                 /*      shared memory key       */
    int shmid;                    /*      shared memory id        */
    sem_t *sem;                   /*      synch semaphore         *//*shared */
    pid_t pid;                    /*      fork pid                */
    int *p;                       /*      shared variable         *//*shared */
    unsigned int value;           /*      semaphore value         */
	
//	hyperQ_threads();
//	return 0;


    /* initialize a shared variable in shared memory */
    shmkey = ftok ("/dev/null", 5);       /* valid directory name and a number */
    printf ("shmkey for p = %d\n", shmkey);
    shmid = shmget (shmkey, sizeof (int), 0644 | IPC_CREAT);
    if (shmid < 0){                           /* shared memory error check */
        perror ("shmget\n");
        exit (1);
    }

    p = (int *) shmat (shmid, NULL, 0);   /* attach p to shared memory */
    *p = 0;
    printf ("p=%d is allocated in shared memory.\n\n", *p);

    /********************************************************/

    /* initialize semaphores for shared processes */
    sem = sem_open ("pSem", O_CREAT | O_EXCL, 0644, 1); // Binary semaphore 
    /* name of semaphore is "pSem", semaphore is reached using this name */

    printf ("semaphores initialized.\n\n");
	
	/*// kstubs
	
	cudaError_t err;

	// Select device
	int deviceId = 2;
	cudaSetDevice(deviceId);
	cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);	
	printf("Device=%s\n", deviceProp.name);
	
	// Create commom streams for all kernels: two for asynchronous transfers, one for preemption commands
	cudaStream_t *transfers_s;
	transfers_s = (cudaStream_t *)calloc(2, sizeof(cudaStream_t));
	
	for (int i=0;i<2;i++){
		err = cudaStreamCreate(&transfers_s[i]);
		checkCudaErrors(err);
	} 
	
	cudaStream_t preemp_s;
	checkCudaErrors(cudaStreamCreateWithFlags(&preemp_s, cudaStreamNonBlocking)); */

	t_Kernel kid[9];
	int index[9];
	kid[0]=MM;
	kid[1]=VA;
	kid[2]=BS;
	kid[3]=Reduction;
	kid[4]=PF;
	kid[5]=GCEDD; // Ojo: en profiling.cu se procesan tambien los tres kernels restantes de la aplicacion
	kid[6]=SPMV_CSRscalar;
	kid[7]=RCONV; // Ojo: en profiling se procesa tambien CCONV
	kid[8]=HST256;
	
	int num_kernels = 2;
	/*for (int i=0; i<num_kernels; i++){
		total_num_kernels++;
		if (kid[i] == RCONV) total_num_kernels++;
		if (kid[i] == GCEDD) total_num_kernels += 3;
	}*/
	
	/** Create stubs ***/
	// Ojo la lista de kernels sólo debe ponerse el primero de una aplicacion. Los demás
	// son creados por el siguiente código
	/*t_kernel_stub **kstubs = (t_kernel_stub **)calloc(total_num_kernels, sizeof(t_kernel_stub*));
	for (int i=0, cont=0; i<num_kernels; i++) {	
		create_stubinfo(&kstubs[cont], deviceId, kid[i], transfers_s, &preemp_s);
		index[i] = cont;
		cont++;
		if (kid[i] == RCONV){ // RCONV params struct must be passed to CCONV 
			create_stubinfo_with_params(&kstubs[cont], deviceId, CCONV, transfers_s, &preemp_s, (void *)kstubs[cont-1]->params);
			cont++;
		}
		
		if (kid[i] == GCEDD){
			create_stubinfo_with_params(&kstubs[cont], deviceId, SCEDD, transfers_s, &preemp_s, (void *)kstubs[cont-1]->params);
			cont++;
			
			create_stubinfo_with_params(&kstubs[cont], deviceId, NCEDD, transfers_s, &preemp_s, (void *)kstubs[cont-2]->params);
			cont++;
			
			create_stubinfo_with_params(&kstubs[cont], deviceId, HCEDD, transfers_s, &preemp_s, (void *)kstubs[cont-3]->params);
			cont++;
		}
	}

	// make HtD transfers of all kernels
	make_transfers(kstubs, total_num_kernels);*/
	
    /* fork child processes */
    for (it = 0; it < num_kernels; it++){
        pid = fork ();
        if (pid < 0) {
        /* check for error      */
            sem_unlink ("pSem");   
            sem_close(sem);  
            /* unlink prevents the semaphore existing forever */
            /* if a crash occurs during the execution         */
            printf ("Fork error.\n");
        }
        else if (pid == 0)
            break;                  /* child processes */
    }


    /******************************************************/
    /******************   PARENT PROCESS   ****************/
    /******************************************************/
    if (pid != 0){
        /* wait for all children to exit */
        while (pid = waitpid (-1, NULL, 0)){
            if (errno == ECHILD)
                break;
        }

        printf ("\nParent: All children have exited.\n");

        /* shared memory detach */
        shmdt (p);
        shmctl (shmid, IPC_RMID, 0);

        /* cleanup semaphores */
        sem_unlink ("pSem");   
        sem_close(sem);  
        /* unlink prevents the semaphore existing forever */
        /* if a crash occurs during the execution         */
        exit (0);
    }

    /******************************************************/
    /******************   CHILD PROCESS   *****************/
    /******************************************************/
    else{
		
		// context and streams
	
		cudaError_t err;

		// Select device
		int deviceId = 2;
		cudaSetDevice(deviceId);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, deviceId);	
		printf("Device=%s\n", deviceProp.name);
	
		/** Create commom streams for all kernels: two for asynchronous transfers, one for preemption commands*/
		cudaStream_t *transfers_s;
		transfers_s = (cudaStream_t *)calloc(2, sizeof(cudaStream_t));
	
		for (int i=0;i<2;i++){
			err = cudaStreamCreate(&transfers_s[i]);
			checkCudaErrors(err);
		}
	
		cudaStream_t preemp_s;
		checkCudaErrors(cudaStreamCreateWithFlags(&preemp_s, cudaStreamNonBlocking)); 
		
		printf("Child=%d Creating kstubs\n", it);
	
		// Create kstbus
		int cont = 0;
		t_kernel_stub **kstubs = (t_kernel_stub **)calloc(4, sizeof(t_kernel_stub*)); // Four is the man number of kernels of a app
		create_stubinfo(&kstubs[cont], deviceId, kid[it], transfers_s, &preemp_s);
		cont++;
		
		if (kid[it] == RCONV){ // RCONV params struct must be passed to CCONV 
			create_stubinfo_with_params(&kstubs[cont], deviceId, CCONV, transfers_s, &preemp_s, (void *)kstubs[cont-1]->params);
			cont++;
		}
		
		if (kid[it] == GCEDD){
			create_stubinfo_with_params(&kstubs[cont], deviceId, SCEDD, transfers_s, &preemp_s, (void *)kstubs[cont-1]->params);
			cont++;
			
			create_stubinfo_with_params(&kstubs[cont], deviceId, NCEDD, transfers_s, &preemp_s, (void *)kstubs[cont-2]->params);
			cont++;
			
			create_stubinfo_with_params(&kstubs[cont], deviceId, HCEDD, transfers_s, &preemp_s, (void *)kstubs[cont-3]->params);
			cont++;
		}
		
		// make HtD transfers of all kernels
		make_transfers(kstubs, cont);
		
		printf("Child=%d Transferencia terminada\n", it);

		
		// Barrier
   /*     sem_wait (sem);           // P operation 
        printf ("  Child(%d) is in critical section.\n", it);
        //sleep (1);
        *p += 1 ;              //increment *p by 0, 1 or 2 based on i 
        printf ("  Child(%d) new value of *p=%d.\n", it, *p);
        sem_post (sem);           /// V operation 
     */   
		*p += 1;
		while (*p < num_kernels); // Spin lock

		
		// Solo original profiling
	
		printf("kid=%d\n", kstubs[0]->id);
		double exectime_s[4];
		struct timespec now;
		clock_gettime(CLOCK_REALTIME, &now);
		double time1 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;		
		for (int i=0; i < cont; i++) {	
			run_original(kstubs[i], &exectime_s[i]);
			clock_gettime(CLOCK_REALTIME, &now);
			double time2 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;	
			printf("Child %d lanzando kernel %d: start=%f end=%f exectime=%f\n", it,  kstubs[i]->id, time1, time2, time2-time1);
		}
		
		/*
		if (kid[it] == GCEDD) {
			double exectime_s[4];
			for (int i=0; i < 4; i++) 
				run_original(kstubs[i], &exectime_s[i]);
		}
		else if (kid[index[it]] == RCONV) {
			double exectime_s[2];
			for (int i=0; i < 2; i++) 
				run_original(kstubs[i], &exectime_s[i]);
		}
		else {
			double exectime_s;
			run_original(kstubs[0], &exectime_s);
			printf("Child %d lanzando kernel %d. Tiempo=%f\n", it,  kid[it], exectime_s);
		}
		*/
		cudaDeviceSynchronize();
		exit(0);
    }
}