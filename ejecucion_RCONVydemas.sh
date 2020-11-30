#!/bin/bash
INITIAL_KERNEL="RCONV"

COKERNELS=(CCONV SCEDD NCEDD HCEDD)

for j in ${COKERNELS[@]}
do
    ./cCuda 2 $INITIAL_KERNEL $j
    echo ""
done 
