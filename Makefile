################################################################################
#
# Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO USER:   
#
# This source code is subject to NVIDIA ownership rights under U.S. and 
# international Copyright laws.  
#
# NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
# CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
# IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
# REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
# MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
# IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
# OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
# OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
# OR PERFORMANCE OF THIS SOURCE CODE.  
#
# U.S. Government End Users.  This source code is a "commercial item" as 
# that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
# "commercial computer software" and "commercial computer software 
# documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
# and is provided to the U.S. Government only as a commercial end item.  
# Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
# 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
# source code with only those rights set forth herein.
#
################################################################################
#
# Makefile project only supported on Mac OS X and Linux Platforms)
#
################################################################################

#include ./findcudalib.mk
include ./findcudalib.mk

# Location of the CUDA Toolkit
CUDA_PATH ?= "/usr/local/cuda"

# internal flags
NVCCFLAGS   := -g -m${OS_SIZE} --ptxas-options=-v  
CCFLAGS     := 
NVCCLDFLAGS :=
LDFLAGS     :=

# Extra user flags
EXTRA_NVCCFLAGS   ?= 
EXTRA_NVCCLDFLAGS ?=
EXTRA_LDFLAGS     ?= -rpath /usr/local/cuda/extras/CUPTI/lib64
EXTRA_CCFLAGS     ?= 

# OS-specific build flags
ifneq ($(DARWIN),) 
  LDFLAGS += -rpath $(CUDA_PATH)/lib
  CCFLAGS += -arch $(OS_ARCH) $(STDLIB)  
else
  ifeq ($(OS_ARCH),armv7l)
    ifeq ($(abi),gnueabi)
      CCFLAGS += -mfloat-abi=softfp
    else
      # default to gnueabihf
      override abi := gnueabihf
      LDFLAGS += --dynamic-linker=/lib/ld-linux-armhf.so.3
      CCFLAGS += -mfloat-abi=hard
    endif
  endif
endif

ifeq ($(ARMv7),1)
NVCCFLAGS += -target-cpu-arch ARM
ifneq ($(TARGET_FS),) 
CCFLAGS += --sysroot=$(TARGET_FS)
LDFLAGS += --sysroot=$(TARGET_FS)
LDFLAGS += -rpath-link=$(TARGET_FS)/lib
LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib
LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib/arm-linux-$(abi)
endif
endif

# Debug build flags
ifeq ($(dbg),1)
      NVCCFLAGS += -g -G
      TARGET := debug
else
      TARGET := release
endif

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(NVCCLDFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(EXTRA_NVCCLDFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))

# Common includes and paths for CUDA
export LD_LIBRARY_PATH := $(LD_LIBRARY_PATH):/usr/local/cuda/extras/CUPTI/lib:/usr/local/cuda/extras/CUPTI/lib64

INCLUDES  := -I/usr/local/cuda/samples/common/inc -I/usr/local/cuda/extras/CUPTI/include
LIBRARIES := -lpthread -lcuda -L /usr/local/cuda/extras/CUPTI/lib -L /usr/local/cuda/extras/CUPTI/lib64 -lcupti

################################################################################

# CUDA code generation flags
#ifneq ($(OS_ARCH),armv7l)
#GENCODE_SM10    := -gencode arch=compute_10,code=sm_10
#endif
GENCODE_SM35    := -gencode arch=compute_35,code=\"sm_35,compute_35\"
GENCODE_SM52    := -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=\"sm_52,compute_52\"
GENCODE_SM61	:= -gencode arch=compute_61,code=sm_61 -gencode arch=compute_61,code=\"sm_61,compute_61\"
GENCODE_FLAGS   := $(GENCODE_SM61)

################################################################################

# Target rules
all: build

#build: basic solo_exec prof_conc
build: basic prof_conc_sincupti solo_exec occ_calc mps_test cCuda prof_conc cCuda_classifier

reduction_original.o: Reduction/reduction_original.cu elastic_kernel.h
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

BS_Original_Kernel.o: BS/BS_Original_Kernel.cu elastic_kernel.h
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
	
vAdd_Original_Kernel.o: VA/vAdd_Original_Kernel.cu elastic_kernel.h
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
	
MM_Original_Kernel.o: MM/MM_Original_Kernel.cu elastic_kernel.h
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
	
#RSC_Original_Kernel.o: RSC/RSC_Original_kernel.cu elastic_kernel.h
#	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
	
SPMV_common.o: SPMV/SPMV_common.cu
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
	
SPMV_Original_Kernel.o: SPMV/SPMV_Original_Kernel.cu elastic_kernel.h
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
	
PF_Original_Kernel.o: PF/PF_Original_Kernel.cu elastic_kernel.h
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
	
CONV_Original_Kernel.o: CONV/CONV_Original_Kernel.cu elastic_kernel.h
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
	
CEDD_Original_Kernel.o: CEDD/CEDD_Original_Kernel.cu elastic_kernel.h
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
	
HST256_Original_Kernel.o: HST/HST256_Original_Kernel.cu elastic_kernel.h
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
	
TP_Original_Kernel.o: TP/TP_Original_Kernel.cu elastic_kernel.h
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

DXTC_Original_Kernel.o: DXTC/DXTC_Original_Kernel.cu elastic_kernel.h
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

proxy.o: proxy/proxy.cu elastic_kernel.h
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
	
kstub.o: kstub.cu elastic_kernel.h BS/BS.h MM/MM.h
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
	
basic.o: basic.cu elastic_kernel.h
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
	
profiling.o: profiling.cu elastic_kernel.h
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
	
profiling_config.o: profiling_config.cu elastic_kernel.h
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
	
utils.o: utils.cu elastic_kernel.h
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
	
schedulers.o: schedulers.cu elastic_kernel.h
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
	
solo_exec.o: solo_exec.cu elastic_kernel.h
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

fast_profiling.o: fast_profiling.cu elastic_kernel.h
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

profiler_overhead.o: profiler_overhead.cu elastic_kernel.h
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

cCuda.o: cCuda.cu elastic_kernel.h
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
	
cCuda_classifier.o: cCuda_classifier.cu
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

prof_conc.o: prof_conc.cu elastic_kernel.h
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
	
cupti_profiler.o: cupti_profiler.cu cupti_profiler.h
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

prof_conc_sincupti.o: prof_conc_sincupti.cu elastic_kernel.h
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
	
occupancy_calculator.o: occupancy_calculator.cu elastic_kernel.h
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
	
mps_test.o: mps_test.cu elastic_kernel.h
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<  
	
basic: profiler_overhead.o schedulers.o fast_profiling.o utils.o kstub.o basic.o proxy.o profiling.o profiling_config.o BS_Original_Kernel.o vAdd_Original_Kernel.o MM_Original_Kernel.o SPMV_Original_Kernel.o SPMV_common.o reduction_original.o PF_Original_Kernel.o CONV_Original_Kernel.o CEDD_Original_Kernel.o HST256_Original_Kernel.o
	$(NVCC) $(ALL_LDFLAGS) -o $@ $+ $(LIBRARIES)
	
solo_exec: solo_exec.o utils.o kstub.o proxy.o profiling.o profiling_config.o BS_Original_Kernel.o vAdd_Original_Kernel.o MM_Original_Kernel.o SPMV_Original_Kernel.o SPMV_common.o reduction_original.o PF_Original_Kernel.o CONV_Original_Kernel.o CEDD_Original_Kernel.o HST256_Original_Kernel.o
	$(NVCC) $(ALL_LDFLAGS) -o $@ $+ $(LIBRARIES)

prof_conc: prof_conc.o utils.o kstub.o proxy.o profiling.o profiling_config.o BS_Original_Kernel.o vAdd_Original_Kernel.o MM_Original_Kernel.o SPMV_Original_Kernel.o SPMV_common.o reduction_original.o PF_Original_Kernel.o CONV_Original_Kernel.o CEDD_Original_Kernel.o HST256_Original_Kernel.o cupti_profiler.o
	$(NVCC) $(ALL_LDFLAGS) -o $@ $+ $(LIBRARIES)
	
prof_conc_sincupti: prof_conc_sincupti.o utils.o kstub.o proxy.o profiling.o profiling_config.o BS_Original_Kernel.o vAdd_Original_Kernel.o MM_Original_Kernel.o SPMV_Original_Kernel.o SPMV_common.o reduction_original.o PF_Original_Kernel.o CONV_Original_Kernel.o CEDD_Original_Kernel.o HST256_Original_Kernel.o 
	$(NVCC) $(ALL_LDFLAGS) -o $@ $+ $(LIBRARIES)
	
occ_calc: occupancy_calculator.o utils.o kstub.o proxy.o profiling.o profiling_config.o BS_Original_Kernel.o vAdd_Original_Kernel.o MM_Original_Kernel.o SPMV_Original_Kernel.o SPMV_common.o reduction_original.o PF_Original_Kernel.o CONV_Original_Kernel.o CEDD_Original_Kernel.o HST256_Original_Kernel.o 
	$(NVCC) $(ALL_LDFLAGS) -o $@ $+ $(LIBRARIES)
	
mps_test: mps_test.o utils.o kstub.o proxy.o profiling.o profiling_config.o BS_Original_Kernel.o vAdd_Original_Kernel.o MM_Original_Kernel.o SPMV_Original_Kernel.o SPMV_common.o reduction_original.o PF_Original_Kernel.o CONV_Original_Kernel.o CEDD_Original_Kernel.o HST256_Original_Kernel.o
	$(NVCC) $(ALL_LDFLAGS) -o $@ $+ $(LIBRARIES)

cCuda: cCuda.o utils.o kstub.o proxy.o profiling.o profiling_config.o BS_Original_Kernel.o vAdd_Original_Kernel.o MM_Original_Kernel.o SPMV_Original_Kernel.o SPMV_common.o reduction_original.o PF_Original_Kernel.o CONV_Original_Kernel.o CEDD_Original_Kernel.o HST256_Original_Kernel.o TP_Original_Kernel.o DXTC_Original_Kernel.o cupti_profiler.o
	$(NVCC) $(ALL_LDFLAGS) -o $@ $+ $(LIBRARIES)

cCuda_classifier: cCuda_classifier.o utils.o kstub.o proxy.o profiling.o profiling_config.o BS_Original_Kernel.o vAdd_Original_Kernel.o MM_Original_Kernel.o SPMV_Original_Kernel.o SPMV_common.o reduction_original.o PF_Original_Kernel.o CONV_Original_Kernel.o CEDD_Original_Kernel.o HST256_Original_Kernel.o 
	$(NVCC) $(ALL_LDFLAGS) -o $@ $+ $(LIBRARIES)
	
run: build
	./basic

clean:
	rm -f profiler_overhead basic solo_exec prof_conc_sincupti occ_calc *.o

clobber: clean
