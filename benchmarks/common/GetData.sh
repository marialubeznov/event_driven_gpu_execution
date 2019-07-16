#!/bin/bash
echo "EVENT"
for event in ipv4_fwd ipv6_fwd memc_conv des_encryption	
do
	cd ~/home/maria/gem5-gpu/benchmarks/edge/$event/
 	for type in draining 8sm_reserve preemption
 	do
 		for bg_task in "$1"
 		do
			echo "$event $type" 			
			for prio in 1
 			do	
				if test -f prio_1_all_opt_"$type"_bg_task_"$bg_task"_"$2".log; then
					./ExtractEvent.sh prio_1_all_opt_"$type"_bg_task_"$bg_task"_"$2".log			
				fi
 			done
 		done
 	done
done

echo "BACKGROUND TASK"
for event in ipv4_fwd ipv6_fwd memc_conv des_encryption
do	
	cd ~/home/maria/gem5-gpu/benchmarks/edge/$event/
 	for type in draining 8sm_reserve preemption
 	do
 		for bg_task in "$1"
 		do
			echo "$event $type" 			
			for prio in 1
 			do	
				if test -f prio_1_all_opt_"$type"_bg_task_"$bg_task"_"$2".log; then	
					if [ $bg_task -eq 0 ] 
            				then
                				../../common/ExtractBG.sh filterActs prio_1_all_opt_"$type"_bg_task_"$bg_task"_"$2".log 
            				elif [ $bg_task -eq 1 ] 
            				then
                				../../common//ExtractBG.sh matrixMul prio_1_all_opt_"$type"_bg_task_"$bg_task"_"$2".log
            				elif [ $bg_task -eq 2 ]
            				then
                				../../common/ExtractBackprop.sh prio_1_all_opt_"$type"_bg_task_"$bg_task"_"$2".log
            				elif [ $bg_task -eq 3 ]
            				then
                				../../common/ExtractBFS.sh prio_1_all_opt_"$type"_bg_task_"$bg_task"_"$2".log
            				else
                				echo "unexpected bg_task"
            				fi
				fi			
 			done
 		done
 	done
done

echo "WAIT"
for event in ipv4_fwd ipv6_fwd memc_conv des_encryption	
do
	cd ~/home/maria/gem5-gpu/benchmarks/edge/$event/
 	for type in draining preemption
 	do
 		for bg_task in "$1"
 		do
			echo "$event $type" 			
			for prio in 1
 			do	
                for f in prio_1_all_opt_"$type"_bg_task_"$bg_task"_"$2"*rate.log
                do 
					./ExtractEventWait.sh "$f"			
                done
 			done
 		done
 	done
done
