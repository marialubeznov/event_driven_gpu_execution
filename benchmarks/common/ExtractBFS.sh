#!/bin/bash
FILES="$1"
rm -f results_temp.log;
for file in $FILES
do
  	rm -f tranc_file;
  	rm -f results_temp.log;
  	line=$(grep -n "L2 cache stats" ${file} | tail -1 | cut -d: -f1);
  	tail -n +$line ${file} > tranc_file;
  	grep KernelP4NodePiPbS -A5 tranc_file | grep kernel_n_cycles | grep -o '[[:digit:]]\+' | uniq >> results_temp.log
  	grep Kernel2PbS -A5 tranc_file | grep kernel_n_cycles | grep -o '[[:digit:]]\+'  | uniq >> results_temp.log
done
paste -s -d+ results_temp.log | bc
