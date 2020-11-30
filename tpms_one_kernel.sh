#!/bin/bash 

	# No profiling: accurate execution time
	COUNTER=1
         while [  $COUNTER -lt $2 ]; do
            ./solo_exec 2 $1 $COUNTER$ 3
             let COUNTER=COUNTER+1
         done

