#!/bin/bash
FILES="$1"
echo "min max avg p95";
rm -f run_cycles.log total_cycles.log wait_cycles.log;
for file in $FILES
do
  	rm -f tranc_file; 
  	line=$(grep -n "L2 cache stats" ${file} | tail -1 | cut -d: -f1);
  	tail -n +$line ${file} > tranc_file;
  	grep "Kernel_name.*ipv6_fwd_kernel" -A8 tranc_file | grep kernel_n_cycles | grep -o '[[:digit:]]\+' >> run_cycles.log
  	grep "Kernel_name.*ipv6_fwd_kernel" -A8 tranc_file | grep event_kernel_cycles_since_interrupt | grep -o '[[:digit:]]\+' >> total_cycles.log
  	paste ./total_cycles.log ./run_cycles.log | awk '{print $1 - $2}' >> wait_cycles.log
done
../../common/calc.py wait_cycles.log
