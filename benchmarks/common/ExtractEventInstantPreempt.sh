#!/bin/bash
FILES="$1"
#echo "min max avg p95";
rm -f run_cycles.log total_cycles.log wait_cycles.log wait_cycles_no_min.log wait_cycles_no_min2.log;
for file in $FILES
do
  	rm -f tranc_file; 
  	line=$(grep -n "L2 cache stats" ${file} | tail -1 | cut -d: -f1);
  	tail -n +$line ${file} > tranc_file;
  	grep "Kernel_name.*$2" -A8 tranc_file | grep kernel_n_cycles | grep -o '[[:digit:]]\+' >> run_cycles.log
  	grep "Kernel_name.*$2" -A8 tranc_file | grep event_kernel_cycles_since_interrupt | grep -o '[[:digit:]]\+' >> total_cycles.log
  	paste ./total_cycles.log ./run_cycles.log | awk '{print $1 - $2}' >> wait_cycles.log
    	min=`cat wait_cycles.log | sort -n | head -1`
	grep -v "^2$" wait_cycles.log >> wait_cycles_no_min.log
    	preemption=`grep fastpath "$1" | cut -d" " -f21`
    	if [ $preemption -eq 1 ]
    	then
        	grep -v "^3$" wait_cycles_no_min.log >> wait_cycles_no_min2.log
        	cp wait_cycles_no_min2.log wait_cycles_no_min.log
    	fi
    	not_instant=`cat wait_cycles_no_min.log | wc -l`
	total=`cat wait_cycles.log | wc -l`
	instant=`expr $total - $not_instant`
	echo "$instant $total"
done

