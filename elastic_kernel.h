#include <semaphore.h>
#include <time.h>

#define SMT // If not defined -> SMK
#define SPATIAL

//#define MEMCPY_SYNC
#define MEMCPY_ASYNC
//#define MANAGED_MEM 

#define MAX_STREAMS_PER_KERNEL 16
#define MAX_NUM_COEXEC_KERNELS 2
#define MIN_SPEEDUP 1.0

#define ZEROCOPY
 
#define C_S 1000000000 // Chunksize: 18Mbytes/s // Changed to 2MBytes to reduce preemption delay

typedef enum {MM=0, BS, VA, RSC_MODEL, RSC_EVALUATE, SPMV_CSRscalar, Reduction, PF, RCONV, CCONV, FDTD3d, Dummy, GCEDD, SCEDD, NCEDD, HCEDD, HST256, EMPTY, Number_of_Kernels} t_Kernel;
typedef enum {NONBLOCKING=0, BLOCKING, STREAM_SYNCHRO, NOACTION} t_tsynchro;
typedef enum {DATA=0, LAUNCH, NOLAUNCH, EVICT, PENDING, SYNCHRO, LAST_TRANSFER} t_type;  

/* 	DATA for data tranfers (kernel input and output data. Synchro may be BLOCKING (MemCpy) and NONBLOCKING (MenCpyAsync)
	
	EVICT for scheduler commands that performs a HtD asynchronous transfer (MemCpyAsync), NONBLOCKING, to change the kernel 
	state to EVICT. 

	PENDING for scheduler commands that reads (from device memory) the number of pending tasks for the just evicted kernel. It 
	performs a DtH asynchronous transfer (MemCpyAsync). Synchro is always NONBLOCKING. 
	
	LAUNCH for scheduler commands that carries out an HtD asynchronous transfer HtD to change the State of the kernel to 
	RUNNING (synchro is always NONBLOCKING)
	
	Synchronization of PENDING and LAUNCH scheduler transfer commnads requires a explicit synchronization. This is carried out using 
	a STREAM_SYNCHRO synchronization command. Thus, no real transfer are performed for this comamnd. The transfer manager just executes
	a cudaStreamSynchro when process the command.

	NOACTION does not generates a real transfer. Is is used in combination with NOLAUNCH to generate a virtual launch command
	when no more kernels are ready to be launched by the scheduler

*/	
	
#define MAX_NUM_PRIORITIES 3
typedef enum {HIGH=0, MEDIUM, LOW} t_tpriority;

typedef enum state {
	NONE, 
	PREP, //No memory allocated, No HtD transfers
	READY, // Ready to run (allocation and transfers have been previous done
	TORUN, 
	RUNNING, // Kernel is executing
	TOEVICT, // Blocks will be evicted using task granularity
	PROXY_EVICT,
	EVICTED, // Kernel has been completely evicted
	DONE // Kernel finished, pending memory free and DtH data transfers
	} 
State;

typedef struct {
	int numSMs;
	int max_persistent_blocks;
	dim3 blocksize;
	dim3 gridsize;
	int coarsening;
}t_Kconfig;

typedef struct {
	
	int deviceId; // Device identifier where kernel runs
	t_Kernel id;
	
	t_Kconfig kconf;
	
	int stream_index; // index of stream to launch kernel
	
	int (*launchCKEkernel)(void *); //Pointer to the function that executes the kernel
	int (*launchORIkernel)(void *);
	int (*startKernel)(void *);
	int (*startMallocs)(void *);
	int (*startTransfers)(void *);
	int (*endKernel)(void *);
	
	State *h_state; // Kernel state on CPU: it should be reserved in pinned memory
	State *gm_state; // Kernel state on GPU: Pointer to state in device memory (use Memcpy to cpu from/to CPU and GPU) 
	
	// Only for zero-copy
	#ifdef ZEROCOPY
	int *h_exec_tasks_proxy;
	int *d_exec_tasks_proxy;
	
	int *h_proxy_eviction;
	int *d_proxy_eviction;
	
	cudaStream_t *proxy_s;
	#endif
	
	// SMT & SMK support for application kernels
	int *idSMs; // SMs to be used by the permanent blocks
	int total_tasks; // Number of tasks to be executed by the kernel, typically gridDim_original (before transformation to resident blocks)
	int *h_executed_tasks; // Number of tasks executed by the kernel (pinned memory): its value must be copied from device memory.  
	int *d_executed_tasks;
	State *d_other_gm_state; //Support from preemption. Pointer to gm_state of the other cocurrent kernel
	
	// SMK 
	int num_blocks_per_SM; //maximum number of blocks per SM 
	int *h_SMs_cont;
	int *d_SMs_cont; // Counterr array, one per SM. Employed to count the number of blocks to be executed in a SM
	
	cudaStream_t *transfer_s; // Two streams: first associated to H->D, second associated to D->H
	cudaStream_t *execution_s;
	cudaStream_t *preemp_s; // For preemption (using MemcpyCpyAsync to stop kernel execution)
	
	cudaEvent_t end_HtD, end_Exec, end_DtH; // Events for synchro
	int kernel_finished;
	int HtD_tranfers_finished;
	int DtH_tranfers_finished;
	
	int priority;
	
	sem_t ready; // Kernel and scheduler synchronization from PREP to READY 
	sem_t evicted; // Kernel and scheduler synchronization from EVICTED to RUNNING
	
	// Specific params
	
	void *params;
	
	// Annotate execution time
	struct timespec starttime_HtD; 
	struct timespec endtime_HtD; 
	struct timespec starttime_kernel; 
	struct timespec endtime_kernel; 
	struct timespec starttime_DtH; 
	struct timespec endtime_DtH;
	float exec_time; // Acumulatte execution time on GPU
	float waittime; // Time elpased from the task was inserted in queue (ready)
	double enqueued_time; // Time task was inserted in scheduling queue
	
	// Support for fair schedulin based on slowdown
	double time_per_task;
	char noyetrun;
	int remaining_tasks;
	double slowdown;	
}t_kernel_stub;

typedef struct {
	cudaStream_t *proxy_s;
	int num_conc_kernels;
	int num_streams;
	State *kernel_evict_zc;
	State *kernel_evict;
	int *cont_tasks_zc;
	int *gm_cont_tasks;
}t_sched;

typedef struct{
	t_kernel_stub *rconv;
	t_kernel_stub *cconv;
}t_kernels_conv;

typedef struct{
	t_kernel_stub *gcedd;
	t_kernel_stub *scedd;
	t_kernel_stub *ncedd;
	t_kernel_stub *hcedd;
}t_kernels_cedd;

struct entry {
	t_kernel_stub *k_stub;
	struct entry *p;
};

struct t_CONV {
	t_kernel_stub *k_stubRows;
	t_kernel_stub *k_stubCols;
};

// Types for transfers queues

typedef entry t_entry; 

struct transfer_command{
	void *dest;
	void *source;
	int  bytesize;
	cudaMemcpyKind kind;
	cudaStream_t s;
	int priority;
	t_type type;
	cudaEvent_t end_transfers;
	t_tsynchro synchro;
	t_kernel_stub *kstub;
	pthread_mutex_t lock; 
	pthread_cond_t cond;
	struct transfer_command *next;
};

typedef struct transfer_command t_tcommand;

typedef struct{
	t_tcommand *head;
	int len;
}t_tqueue;

typedef struct {
int TpW;  //Threads per Warp
int Max_WpSM ; //Max Warps per Multiprocessor
int Max_BpSM ; //Max Thread Blocks per Multiprocessor
int Max_TpSM ; // Max Threads per Multiprocessor
int Max_TpB ; //Maximum Thread Block Size
int RpSM ; //Registers per Multiprocessor
int Max_RpB ; //Max Registers per Thread Block
int Max_RpT ; //Max Registers per Thread
int Max_SmpSM ; //Shared Memory per Multiprocessor (bytes)
int Max_SmpB ; // Max Shared Memory per Block
int RallSize ; //Register allocation unit (warp) size
int SmAllsize; //Shared Memory allocation unit size
int WAllG; //Warp allocation granularity
}
t_cc60;

typedef struct{
	int used_TpSM;
	int used_BpSM;
	int used_RpSM;
	int used_SmpSM;
	int used_WpSM;
}t_used_res;

typedef struct {
	t_Kernel id[2];
	int blocks[2];
	float speedup;
}t_cke_performance;

typedef struct {
	t_Kernel id;
	double max_tpms;
}t_solo_performance;

__device__ uint get_smid(void);

typedef struct{
	t_kernel_stub *kstub;
	int save_cont_task; // Storage value of yye task counter (0 at the beginning and X after a complete kernel eviction
	int num_streams; // Numner of executing streams of the kernel
	int save_cont_tasks; // Here we save the number if executed tasks of the kernel when it is evicted (used to start with this value kernel is restarted)
	cudaStream_t str[MAX_STREAMS_PER_KERNEL]; // Streams id
}t_kstreams;

typedef struct{
	int num_kernels;
	t_kstreams **kstr;
	int *num_streams; // Number of streams to be lauched for each kernel 
	int *queue_index; // Index to the queue of ready kernel. It inidcates the position of the kernel in that queue
}t_kcoexec;

typedef struct{ // Configuration of number og blocks per kernel in coexection and task per ms achieved
	t_Kernel kid[2];
	int num_configs;
	int **pairs;
	double **tpms;
}
t_smk_coBlocks;

typedef struct{ // Configuration of number og blocks per kernel in coexection and task per ms achieved
	t_Kernel kid[2];
	int num_configs;
	int **pairs;
	double **tpms;
}
t_smt_coBlocks;

typedef struct{
	int num_configs;
	double *tpms;
} t_smk_solo;

typedef struct{
	int num_configs;
	double *tpms;
} t_smt_solo;

typedef struct {
	int pairs[2];
	int speedup;
}t_co_speedup;


int create_stubinfo(t_kernel_stub **stub, int deviceId, t_Kernel id, cudaStream_t *transfer_s, cudaStream_t *preemp_s);
int start_linux_scheduler(t_Kernel *kernel_list, int list_size);

// Function for transfers queues

int create_tqueues(t_tqueue **head_tqueues, int num_priorities);
t_tcommand dequeue_tcommand(t_tqueue *head_queues, int num_priorities);
t_tcommand *enqueue_tcomamnd(t_tqueue *head_queues, void *dest, void *source, int  bytesize, cudaMemcpyKind kind, cudaStream_t s, t_tsynchro synchro, 
t_type type, int priority, t_kernel_stub *kstub);
void *tranfers_manager(void *arg);

int launch_generic_proxy(void *arg);
int conc_scheduler();
int one_kernel_benchmark(int deviceId, t_Kernel kid, double *max_tpms);
int multiple_kernel_benchmark(int deviceId, int num_kernels, t_Kernel *kid, int *BpSM, double *max_tpms);
int multiple_kernel_benchmark_ver2(int deviceId, int num_kernels, t_Kernel *kid, int *BpSM, double *max_tpms);
int all_multiple_kernel_benchmark(int deviceId, int num_kernels, t_Kernel *kid, int num_benchs, int **BpSM, double *max_tpms);

int get_max_resources(int TpB, int RpT, int SmpB, t_used_res *ures, t_cc60 cc60, int *BpSM);
int init_cc60(t_cc60 *cc);
int get_resources(int req_BpSM, int TpB, int RpT, int SmpB, t_used_res *ures, t_cc60 cc60);
int BpSM_benchmark(int deviceId, t_Kernel kid, int numBpSM);
int two_kernel_bench_spatial(int deviceId, t_Kernel *kid, double *max_tpms);


int initialize_performance();
int initialize_solo_performance();
int initialize_theoretical_performance();
int get_best_partner(t_Kernel *kid, int *k_done, int num_kernels, t_Kernel curr_kid, t_Kernel *next_kid, int *index_next, int *b0, int *b1);
int get_best_partner_theoretical(t_Kernel curr_kid, t_Kernel *kid, int *k_done, float **bad_partner, int num_kernels, t_Kernel *select_kid, int *select_index, int *b0, int *b1);
int get_last_kernel (t_Kernel kid, int *num_blocks);
int all_nocke_execution(t_kernel_stub **kstubs, int num_kernels);
int kid_from_index(int index, char *skid);
double get_solo_perf(t_Kernel id);
int get_max_blocks(t_Kernel kid);

int make_transfers(t_kernel_stub **kstubs, int num_kernels);
int all_profiling(t_Kernel *kid, int num_kernels, int deviceId);
int smk_fill_coBlocks();
int smk_fill_solo();

int create_sched(t_sched *sched);
int create_kstreams(t_kernel_stub *kstub, t_kstreams *kstr);
int create_coexec(t_kcoexec *coexec, int num_kernels);
int kernel_in_coexec(t_kcoexec *coexec, t_kstreams *kstr, int *pos);
int add_kernel_for_coexecution(t_kcoexec *coexec, t_sched * sched, t_kstreams *kstr, int num_streams, int pos);
int rem_kernel_from_coexecution(t_kcoexec *coexec, t_sched *sched, t_kstreams *kstr);
int evict_streams(t_kstreams *kstr, int num_streams);
int add_streams_to_kernel(t_kcoexec *coexec, t_sched *sched, t_kstreams *kstr, int num_streams);
int launch_coexec(t_kcoexec *coexec);
int wait_for_kernel_termination_with_proxy(t_sched *sched, t_kcoexec *info, int *kernelid, double *speedup);
int add_streams_to_kernel(t_kcoexec *coexec, t_sched *sched, t_kstreams *kstr, int num_streams);
int greedy_coexecution(int deviceId);

#ifdef ZEROCOPY
int launch_proxy(void *arg);
#endif