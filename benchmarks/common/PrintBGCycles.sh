#!/bin/bash
rm -f temp.log 
 for type in 8sm_reserve
 do
    echo "$type" >> temp.log
 
 	for bg_task in 0 1 2 3
 	do
 		echo "bg_task $bg_task" >> temp.log
 
        for prio in 1
 		do
            echo "prio $prio" >> temp.log
            if [ $bg_task -eq 0 ] 
            then
            	../../common/ExtractBG.sh filterActs $1/prio_"$prio"_all_opt_"$type"_bg_task_"$bg_task"_"$2".log >> temp.log 
            elif [ $bg_task -eq 1 ] 
            then
            	../../common//ExtractBG.sh matrixMul $1/prio_"$prio"_all_opt_"$type"_bg_task_"$bg_task"_"$2".log >> temp.log
            elif [ $bg_task -eq 2 ]
            then
                ../../common/ExtractBackprop.sh $1/prio_"$prio"_all_opt_"$type"_bg_task_"$bg_task"_"$2".log >> temp.log
            elif [ $bg_task -eq 3 ]
            then
                ../../common/ExtractBFS.sh $1/prio_"$prio"_all_opt_"$type"_bg_task_"$bg_task"_"$2".log >> temp.log
            else
            	echo "unexpected bg_task"
            fi
 		done
 	done
 done
cat temp.log