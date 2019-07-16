#!/bin/bash
FILES="$*"
for file in $FILES
do
  	echo "${file}"
  	grep "num_memc_kernels = " -A9 "${file}" | tail -n10 | grep avg_event_kernel_cycles_wait_in_preemption_queue
  	grep "num_memc_kernels = " -A9 "${file}" | tail -n10 | grep avg_event_kernel_preemption_len_in_cycles
done