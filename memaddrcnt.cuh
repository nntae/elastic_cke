// Funciones para contar los accesos a memoria. Instrucciones de uso:
//
// 1. Añadir #include "../memaddrcnt.cuh" al fichero con el código de CUDA
// 2. Pasar a la función de CUDA el parámetro kstub->d_numUniqueAddr
// 3. Incluir en la función el parámetro int *numUniqueAddr
// 4. En cada acceso poner (por ejemplo acceso a gpuSrc(xidx]
//				if ( s_bid == 0 )
//					get_unique_lines((intptr_t) &gpuSrc[xidx], numUniqueAddr);

#ifndef MEMORY_ADDRESS_COUNTER
#define MEMORY_ADDRESS_COUNTER

// Comment to count only for first task

#define COUNT_ALL_TASKS

#pragma once

#include "cuda_runtime.h"

#define OFFSET_BITS 5

// Global Memory Addresses Counts for each kernel. The variable is indexed by kernel id
// BS, SPMV, NECDD values vary from one execution to another
// A value of -1 means there is no value for that kernel
// int glmemaddrcnt[] = { 8224, 2560, 3840, -1, -1, 24884, 129, 5751, 136, 272, -1, -11, 86, 110, 168, 48, 288};

/////////////////////////////////////////////////////////////////////////////
//
//  Get a thread's lane ID.
//
/////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__ unsigned int get_laneid(void) {
  unsigned int laneid;
  asm volatile ("mov.u32 %0, %laneid;" : "=r"(laneid));
  return laneid;
}

/////////////////////////////////////////////////////////////////////////////
//
//  A semi-generic warp broadcast function.
//
/////////////////////////////////////////////////////////////////////////////
template <typename T>
__device__ T __broadcast(T t, int fromWhom)
{
  union {
    int32_t shflVals[sizeof(T)];
    T t;
  } p;

  p.t = t;
  #pragma unroll
  for (int i = 0; i < sizeof(T); i++) {
    int32_t shfl = (int32_t)p.shflVals[i];
    p.shflVals[i] = __shfl(shfl, fromWhom);
  }
  return p.t;
}

__device__ __inline__ int get_unique_lines( intptr_t addrAsInt, int *numUniqueAddr )
{
	int unique = 0;
	// Shift off the offset bits into the cache line.
	intptr_t lineAddr = addrAsInt >> OFFSET_BITS;

    int workset = __ballot(1);
	int firstActive = __ffs(workset) - 1;
	int numActive = __popc(workset);
	while (workset) 
	{
        // Elect a leader, get its line, see who all matches it.
        int leader = __ffs(workset) - 1;
        intptr_t leadersAddr = __broadcast<intptr_t>(lineAddr, leader);
        int notMatchesLeader = __ballot(leadersAddr != lineAddr);

        // We have accounted for all values that match the leader's.
        // Let's remove them all from the workset.
        workset = workset & notMatchesLeader;
        unique++;
//		assert(unique <= 32);
	}

//      assert(unique > 0 && unique <= 32);
	// Each thread independently computed 'numActive' and 'unique'.
	// Let's let the first active thread actually tally the result.
	int threadsLaneId = get_laneid();
	if (threadsLaneId == firstActive)
//		numUniqueAddr[0] += unique;
		atomicAdd(numUniqueAddr, unique);
//        atomicAdd(&(sassi_counters[numActive][unique]), 1LL);	}
	
	return unique;	
}

__device__ __inline__ int get_conflicting_banks( intptr_t addrAsInt, int *numUniqueAddr )
{

	int conflicts = 0;
	intptr_t bankAddr = (addrAsInt >> 2) & 0x0FF;
    int workset = __ballot(1);
	int firstActive = __ffs(workset) - 1;

	int numActive = __popc(workset);
	while (workset) 
	{
        // Elect a leader, get its bank, see who all matches it.
        int leader = __ffs(workset) - 1;
        intptr_t leadersAddr = __broadcast<intptr_t>(bankAddr, leader);
        int notMatchesLeader = __ballot(leadersAddr != bankAddr);
		//int numConflicts = __popc(MatchesLeader);
        conflicts++;// =(numConflicts-1);

        // We have accounted for all values that match the leader's.
        // Let's remove them all from the workset.
        workset = workset & notMatchesLeader;
//		assert(unique <= 32);
	}

//      assert(unique > 0 && unique <= 32);
	// Each thread independently computed 'numActive' and 'unique'.
	// Let's let the first active thread actually tally the result.
	int threadsLaneId = get_laneid();
	conflicts = 1;
	if (threadsLaneId == firstActive)
//		numUniqueAddr[0] += unique;
		atomicAdd(numUniqueAddr, conflicts);
//        atomicAdd(&(sassi_counters[numActive][unique]), 1LL);	}
	
	return conflicts;	
}

#endif