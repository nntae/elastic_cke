#!/bin/bash
INITIAL_KERNEL="SCEDD"

COKERNELS=(NCEDD HCEDD)

for j in ${COKERNELS[@]}
do
    ./cCuda 2 $INITIAL_KERNEL $j
    echo ""
done 
