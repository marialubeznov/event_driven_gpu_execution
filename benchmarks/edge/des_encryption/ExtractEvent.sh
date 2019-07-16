#!/bin/bash
FILES="$1"
echo "min max avg p95";
rm -f results_temp.log;
for file in $FILES
do
  	rm -f tranc_file;
  	rm -f results_temp.log;
  	line=$(grep -n "L2 cache stats" ${file} | tail -1 | cut -d: -f1);
  	tail -n +$line ${file} > tranc_file;
  	grep "Kernel_name = _Z26matrixMultiplicationKernelPfS_S_i" -A8 tranc_file | grep event_kernel_cycles_since_interrupt | grep -o '[[:digit:]]\+' >> results_temp.log
done
./calc.py