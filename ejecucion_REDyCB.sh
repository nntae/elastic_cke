#!/bin/bash
INITIAL_KERNEL="Reduction"

COKERNELS=(GCEDD SCEDD NCEDD HCEDD MM PF)

for j in ${COKERNELS[@]}
do
    ./cCuda 2 $INITIAL_KERNEL $j
done 
