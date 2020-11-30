#!/bin/bash
INITIAL_KERNEL="CCONV"

COKERNELS=(SCEDD NCEDD HCEDD)

for j in ${COKERNELS[@]}
do
    ./cCuda 2 $INITIAL_KERNEL $j
    echo ""
done 
