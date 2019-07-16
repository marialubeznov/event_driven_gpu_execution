#!/bin/bash

mkdir results_1G_Apr08
 for type in preemption draining
 do
 	for bg_task in 0 1 2 3
 	do
 		for prio in 1
 		do
 			cd ~/home/maria/gem5-gpu/gem5-gpu/configs/gpu_config
 			cp gpgpusim.fermi.config.template."$type" gpgpusim.fermi.config.template
 			echo >> gpgpusim.fermi.config.template
 			echo "-edge_max_timer_events 4096" >> gpgpusim.fermi.config.template
 			echo "-gpgpu_max_concurrent_kernel 65" >> gpgpusim.fermi.config.template
 			echo "-edge_event_priority $prio" >> gpgpusim.fermi.config.template
 			echo "-edge_limit_concurrent_events 0" >> gpgpusim.fermi.config.template
 			echo "-edge_gen_requests_pattern_by_delay_and_limit 0" >> gpgpusim.fermi.config.template 			
 			cd ~/home/maria/gem5-gpu/benchmarks/edge/memc_conv
 			#run test
			qsub -v PRIO="$prio",TYPE="$type",BG_TASK="$bg_task" run_1G_memc.pbs
			sleep 5
 			#move to results dir
 			#mv prio_"$prio"_all_opt_"$type"_bg_task_"$bg_task".log results_1G_Apr04
 		done
 	done
 done
