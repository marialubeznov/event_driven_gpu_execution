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
  	grep "Kernel_name = _Z3ISRv" -A3 tranc_file | grep kernel_n_cycles | grep -o '[[:digit:]]\+' | tail -n +17 >> results_temp.log
done
./calc.py