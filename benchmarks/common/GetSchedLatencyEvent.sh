#!/bin/bash

echo "WAIT"
for event in "$2"	
do
	cd ~/home/maria/gem5-gpu/benchmarks/edge/$event/
 	for type in preemption
 	do
 		for bg_task in "$1"
 		do
			echo "$event&$type" 			
			for prio in 1
 			do	
                		for f in prio_1_all_opt_"$type"_bg_task_"$bg_task"_no_overlap_*rate.log
                		do
                    			if [ "$event" == "memc_conv" ]
                    			then
					    ../../common/ExtractEventWait.sh "$f" "GetKernel"			
                    			elif [ "$event" == "des_encryption" ]
                    			then
                    			    ../../common/ExtractEventWait.sh "$f" "des_encrypt"
                    			else
					    ../../common/ExtractEventWait.sh "$f" "$event"			
                    			fi
                		done
 			done
 		done
 	done
done

echo "INSTANT"
for event in "$2"	
do
	cd ~/home/maria/gem5-gpu/benchmarks/edge/$event/
 	for type in preemption
 	do
 		for bg_task in "$1"
 		do
			for prio in 1
 			do	
                		for f in prio_1_all_opt_"$type"_bg_task_"$bg_task"_no_overlap_*rate.log
                		do
                    			if [ "$event" == "memc_conv" ]
                    			then
					    ../../common/ExtractEventInstantPreempt.sh "$f" "GetKernel"			
                    			elif [ "$event" == "des_encryption" ]
                    			then
                    			    ../../common/ExtractEventInstantPreempt.sh "$f" "des_encrypt"
                    			else
					    ../../common/ExtractEventInstantPreempt.sh "$f" "$event"			
                    			fi
                		done
 			done
 		done
 	done
done
