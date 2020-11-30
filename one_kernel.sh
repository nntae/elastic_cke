#!/bin/bash 

	# No profiling: accurate execution time
#	COUNTER=1
#         while [  $COUNTER -lt $2 ]; do
#            ./solo_exec 2 $1 $COUNTER$ 3
#             #./one_kernel 2 VA $COUNTER$
#             let COUNTER=COUNTER+1
#         done


# Incrment the number of BpSM
#         COUNTER=1
#         while [  $COUNTER -lt $2 ]; do
#             nvprof --print-gpu-trace --csv  --profile-from-start off --concurrent-kernels on --metrics  dram_read_throughput,dram_write_throughput,tex_utilization,l2_utilization,shared_utilization,ldst_fu_utilization,cf_fu_utilization,special_fu_utilization,tex_fu_utilization,single_precision_fu_utilization,double_precision_fu_utilization,dram_utilization ./solo_exec 2 $1 $COUNTER 1
#             let COUNTER=COUNTER+1 
#         done
		 
# For a specific BpSM
nvprof --print-gpu-trace --csv  --profile-from-start off --concurrent-kernels on --metrics  dram_read_throughput,dram_write_throughput,tex_utilization,l2_utilization,shared_utilization,ldst_fu_utilization,cf_fu_utilization,special_fu_utilization,tex_fu_utilization,single_precision_fu_utilization,double_precision_fu_utilization,dram_utilization ./solo_exec 2 $1 $2 1
             #nvprof --print-gpu-trace --csv  --profile-from-start off --concurrent-kernels on --metrics flop_count_sp,flop_sp_efficiency,inst_integer,gld_transactions,gst_transactions,gld_efficiency,gst_efficiency,gld_requested_throughput,dram_utilization,dram_read_throughput,dram_write_throughput ./solo_exec 2 $1 $2 1


