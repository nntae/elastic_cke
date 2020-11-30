#!/bin/bash
INITIAL_KERNEL="GCEDD"

COKERNELS=(SPMV_CSRscalar RCONV CCONV SCEDD NCEDD HCEDD)

for j in ${COKERNELS[@]}
do
    ./cCuda 2 $INITIAL_KERNEL $j
    echo ""
done 
