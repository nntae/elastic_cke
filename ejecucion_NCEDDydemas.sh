#!/bin/bash
INITIAL_KERNEL="NCEDD"

COKERNELS=(HCEDD)

for j in ${COKERNELS[@]}
do
    ./cCuda 2 $INITIAL_KERNEL $j
    echo ""
done 
