#include "./elastic_kernel.h"
// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h> 
#include <unistd.h>

pthread_mutex_t tq_mutex = PTHREAD_MUTEX_INITIALIZER;

extern int pending_tasks;
extern t_tqueue *tqueues;

int create_tqueues(t_tqueue **head_tqueues, int num_priorities)
{
	
	*head_tqueues= (t_tqueue *) calloc(1, num_priorities * sizeof(t_tqueue)); // Create head
	
	return 0;
}

t_tcommand *dequeue_tcommand(t_tqueue *head_queues, t_tpriority priority, int *bytesize, int *last_subtransfer)
{	

	
	pthread_mutex_lock(&tq_mutex);
	//Check queues in prioritu order 0->higher periority
	int i;
	for (i=0;i<=priority; i++) { //Polling queues until priority
		if (head_queues[i].len!=0)
			break;
	}
	
	if (i > priority){		/* If no pending commands, return */
		pthread_mutex_unlock(&tq_mutex);
		return NULL;
	}
	
	
	/** Search for a EVICT as it indicates switching to high priority queue. It must be done asap **/
	
	t_tcommand *p = head_queues[i].head;
	
	int cont = 0;
	t_tcommand *p_prev = NULL;
	while (p != NULL) {
		if (p->type == EVICT){
			if (cont == 0)
				head_queues[i].head = p->next;
			else
				p_prev->next = p->next;
			
			head_queues[i].len--;
			*bytesize = p->bytesize;
			*last_subtransfer = 1;
			pthread_mutex_unlock(&tq_mutex);

			return p;
		}
		cont++;
		p_prev = p;
		p = p->next;
	}
	
	/* If no EVICT has been found, it continues */	
	/* Sería bueno desencolar los comandos en orden FIFO ya que esto beneficia que se termine
		la transferencia completa de un comando sin que se entrelace con la de nuevos comandos **/
	
	/* FIFO */
	 p = head_queues[i].head;

	switch (p->synchro) {
		
		case STREAM_SYNCHRO:
		case NOACTION:
			head_queues[i].head = p->next;
			head_queues[i].len--;
			break;
			
		case BLOCKING:
		case NONBLOCKING:
			if (p->bytesize <= C_S) { /** The entry is deleted from the queue*/
			head_queues[i].head = p->next;
			head_queues[i].len--;
			*bytesize = p->bytesize;
			*last_subtransfer = 1;
			}
			else {
				*bytesize = C_S;
				*last_subtransfer = 0;
			}
		break;
		
		default: 
			printf("Uknown transfer command\n");
	}

	pthread_mutex_unlock(&tq_mutex);

	return p;
}

t_tcommand *enqueue_tcomamnd(t_tqueue *head_queues, void *dest, void *source, int  bytesize, cudaMemcpyKind kind, cudaStream_t s, t_tsynchro synchro, t_type type, 
	int priority, t_kernel_stub *kstub)
{
	t_tcommand *entry ;
	
	//printf("Encolando datos=%d %d\n", bytesize, kind);
	//if (kind ==  cudaMemcpyDeviceToHost) printf ("Encolando =%d datos\n", bytesize);
	
	pthread_mutex_lock(&tq_mutex);

	if (head_queues == NULL){
		//printf("Error: transfer command queues head has not been created\n");
		pthread_mutex_unlock(&tq_mutex);
		return NULL;
	}
	
	t_tcommand *head = head_queues[priority].head; 
	
	/* FIFO */
	
	if (head == NULL){
		entry = (t_tcommand *) calloc(1, sizeof(t_tcommand)); // First element
		
		head_queues[priority].head = entry;
		head_queues[priority].len = 1;
		
		if (synchro == STREAM_SYNCHRO || synchro == NOACTION) {
			entry->kstub = kstub;
			entry->synchro = synchro;
			entry->s = s;
			entry->kind = kind;
			entry->priority = priority;
			entry->type = type;
			pthread_mutex_init(&entry->lock, NULL);
			pthread_cond_init (&entry->cond, NULL);
		}
		else {			
			//if (synchro == NONBLOCKING) printf("Encolando NONBLOCKING kinf=%d pr=%d\n", kind, priority);
			entry->kstub = kstub;
			entry->dest = dest;
			entry->source = source;
			entry->bytesize = bytesize;
			entry->kind = kind;
			entry->s = s;
			entry->synchro = synchro;
			entry->type = type;
			pthread_mutex_init(&entry->lock, NULL);
			pthread_cond_init (&entry->cond, NULL);
			cudaEventCreate(&entry->end_transfers);
			entry->priority = priority;
		} 
	}
	else{
		
		t_tcommand *p = head;
		
		while (p->next != NULL)
			p = p->next;
		
		entry = (t_tcommand *) calloc(1, sizeof(t_tcommand)); // First element
		
		head_queues[priority].len++;
		
		if (synchro == STREAM_SYNCHRO || synchro == NOACTION) {
			entry->kstub = kstub;
			entry->synchro = synchro;
			entry->s = s;
			entry->kind = kind;
			entry->priority = priority;
			entry->type = type;
			pthread_mutex_init(&entry->lock, NULL);
			pthread_cond_init (&entry->cond, NULL);
			p->next = entry;
		}
		else {
			entry->kstub = kstub;
			entry->dest = dest;
			entry->source = source;
			entry->bytesize = bytesize;
			entry->kind = kind;
			entry->priority = priority;
			entry->type = type;
			entry->s = s;
			entry->synchro = synchro;
			pthread_mutex_init(&entry->lock, NULL);
			pthread_cond_init(&entry->cond, NULL);
			cudaEventCreate(&entry->end_transfers);
			p->next = entry;
		}
	}
	
	pthread_mutex_unlock(&tq_mutex);

	// Block calling thread when executing synchro transfer or cudaStreamSynchronize commands
	if (synchro == BLOCKING || synchro == STREAM_SYNCHRO) {
		pthread_mutex_lock(&entry->lock);
		pthread_cond_wait( &entry->cond, &entry->lock); 
		pthread_mutex_unlock(&entry->lock);
	}
	
	return entry;
}

int HtD_data_transfer(void *d, void *s, int bytesize, int chunk_size, cudaStream_t str,  t_tsynchro synchro)
{
	int i;
	char *dc= (char *)d;
	char *sc= (char *)s;
	
	int iter = (bytesize+chunk_size-1)/chunk_size;
	
	for (i=0; i<iter-1; i++)
		if (synchro == BLOCKING)
			checkCudaErrors(cudaMemcpy((void *)(dc + i * chunk_size), (void *)(sc + i * chunk_size), chunk_size, cudaMemcpyHostToDevice));
		else
			checkCudaErrors(cudaMemcpyAsync((void *)(dc + i * chunk_size), (void *)(sc + i * chunk_size), chunk_size, cudaMemcpyHostToDevice, str));
		
	if (bytesize - chunk_size * (iter - 1)>0)
		if (synchro == BLOCKING)
			checkCudaErrors(cudaMemcpy((void *)(dc + i * chunk_size), (void *)(sc + i * chunk_size), bytesize - chunk_size * (iter - 1), cudaMemcpyHostToDevice));
		else
			checkCudaErrors(cudaMemcpyAsync((void *)(dc + i * chunk_size), (void *)(sc + i * chunk_size), bytesize - chunk_size * (iter - 1), cudaMemcpyHostToDevice, str));

	return 0;
}

int DtH_data_transfer(void *d, void *s, int bytesize, int chunk_size, cudaStream_t str,  t_tsynchro synchro)
{
	int i;
	char *dc= (char *)d;
	char *sc= (char *)s;

	int iter = (bytesize+chunk_size-1)/chunk_size;
	
	for (i=0; i<iter-1; i++)
		if (synchro == BLOCKING)
			checkCudaErrors(cudaMemcpy((void *)(dc + i * chunk_size), (void *)(sc + i * chunk_size), chunk_size, cudaMemcpyDeviceToHost));
		else
			checkCudaErrors(cudaMemcpyAsync((void *)(dc + i * chunk_size), (void *)(sc + i * chunk_size), chunk_size, cudaMemcpyDeviceToHost, str));

	if (bytesize - chunk_size * (iter - 1)>0)
		if (synchro == BLOCKING)
			checkCudaErrors(cudaMemcpy((void *)(dc + i * chunk_size), (void *)(sc + i * chunk_size), bytesize - chunk_size* (iter - 1), cudaMemcpyDeviceToHost));
		else
			checkCudaErrors(cudaMemcpyAsync((void *)(dc + i * chunk_size), (void *)(sc + i * chunk_size), bytesize - chunk_size* (iter - 1), cudaMemcpyDeviceToHost, str));

	return 0;
}

extern int pending_synchro, launch_synchro;

void *tranfers_manager(void *arg)
{
	int *deviceId = (int *)arg;
	
	cudaSetDevice(*deviceId);
	
	int bytesize, last_subtransfer;
	
	t_tpriority min_priority = LOW;
	
	while (pending_tasks >0) {
		
		// Get next transfer commands
		t_tcommand *com;
		//printf("chequeando\n");
		// dequeue command in size no bigger than C_S
		com = dequeue_tcommand(tqueues, min_priority, &bytesize, &last_subtransfer);
		if (com == NULL)
			continue;
		
	/*	if (com->synchro == BLOCKING || com->synchro == NONBLOCKING) 
			printf ("Desencolando datos =%d kind=%d  priority=%d\n", bytesize, com->kind, com->priority);
		else
			printf("Desencolando StreamSynchronization, kind=%d priority=%d\n", com->kind, com->priority);
	*/	
		// Do segmented transfer
		switch (com->synchro) {
		
			case BLOCKING:
				checkCudaErrors(cudaMemcpy(com->dest, com->source, bytesize, com->kind));
				com->bytesize -= C_S;
				com->dest = (void *)((char *)com->dest + C_S);
				com->source = (void *)((char *)com->source + C_S);
				break;
			
			case NONBLOCKING:
				//if (com->kind ==  cudaMemcpyDeviceToHost) printf ("DTH->byte size =%d\n", com->bytesize);
				checkCudaErrors(cudaMemcpyAsync(com->dest, com->source, bytesize, com->kind, com->s));
				com->bytesize -= C_S;
				com->dest = (void *)((char *)com->dest + C_S);
				com->source = (void *)((char *)com->source + C_S); 
				
				// Wait until each segment is transfered. Otherwise all segment transfer commands would be emited in row and
				// high prority transferences could suffer a long delay
				cudaEventRecord(com->end_transfers, com->s);
				cudaEventSynchronize(com->end_transfers);
				
				
				if (com->bytesize <= 0 && com->type == LAST_TRANSFER){ // End of last subtransfer, record an event to be captured by the kernel CPU thread
					if (com->kind ==  cudaMemcpyHostToDevice){
						com->kstub->HtD_tranfers_finished = 1;
						//printf("***Indicado HtD_transfe_finished\n");
					}
					else{
						if (com->kind ==  cudaMemcpyDeviceToHost);
						{
							com->kstub->DtH_tranfers_finished = 1;
							//printf("DTH done for task %d\n", com->kstub->id);
						}
					}
				}
				
				if (com->type == PENDING) {// End of last subtransfer, record an event to be captured by the kernel CPU thread
						cudaEventRecord(com->end_transfers, com->s);
						cudaEventSynchronize(com->end_transfers);
						pending_synchro = 1;
				}
				
				if (com->type == LAUNCH)
						launch_synchro = 1;
					
		
				break;
			
			case STREAM_SYNCHRO:
				cudaStreamSynchronize(com->s);
				if (com->kind == cudaMemcpyDeviceToHost)
					printf("xx->Sincronization for task\n", com->kind);  
				break;
				
			case NOACTION:
				break;
			
			default:
				printf("Transfer synchro not valid\n");
		}
		
		/*
			if (com->synchro == NONBLOCKING)
				checkCudaErrors(cudaMemcpyAsync(com->dest, com->source, bytesize, com->kind, com->s));
			else
				cudaStreamSynchronize(com->s);
			
	
		com->bytesize -= C_S;
		com->dest = (void *)((char *)com->dest + C_S);
		com->source = (void *)((char *)com->source + C_S);*/

		// If transfer is blocking, unblock CPU thread which is running kernel
		// CudaStreamSynchro command, unblock CPU thread
		
		/*if (com->synchro == BLOCKING && last_subtransfer == 1) {
			//printf("Unlocking\n");
			pthread_mutex_lock(&com->lock);
			pthread_cond_signal(&com->cond); 
			pthread_mutex_unlock(&com->lock);
		}*/
		
		if (com->synchro == STREAM_SYNCHRO){
			//printf("Unlocking\n");
			pthread_mutex_lock(&com->lock);
			pthread_cond_signal(&com->cond); 
			pthread_mutex_unlock(&com->lock);
		}
		
		// Check if dequeued transfer type is EVICT
		
		if (com->type == EVICT){
			min_priority = HIGH; // Only poll high priority transfer queue
			//printf("Accediendo solo a HIGH\n");
		}
		
		if (com->type == LAUNCH || com->type == NOLAUNCH) {
			min_priority = LOW; // Poll all transfer queues
			//printf("Accediendo a TODOS\n");
		}
	}
	
	printf("Transfer manager thread finished\n");
	
	return NULL;
		
}
	
	
