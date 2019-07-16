#!/bin/bash
FILES="$1"
for file in $FILES
do
	endt="$(grep "completed for kernel" ${file} | tail -1 | cut -d" " -f2 | cut -d":" -f1)"
	startt="$(grep "Interrupt received on core 0 on cycle" no_preemption_rate_200_32warps.log | head -1 | cut -d" " -f9)"
	((diff=endt-startt))
  	echo $diff
done