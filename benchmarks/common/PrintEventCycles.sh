#!/bin/bash
rm -f temp.log 
 for type in preemption draining 8sm_reserve
 do
 	echo "$type" >> temp.log
 	for bg_task in 0 1
 	do
 		echo "bg_task $bg_task" >> temp.log
 		for prio in 1 2 3
 		do
            echo "prio $prio" >> temp.log
            ./ExtractEvent.sh $1/prio_"$prio"_all_opt_"$type"_bg_task_"$bg_task".log >> temp.log 
 		done
 	done
 done
cat temp.log