#!/bin/bash

mkdir results_10G_Apr04
 for type in preemption draining 8sm_reserve
 do
 	for bg_task in 0 1
 	do
 		for prio in 1
 		do
 			cd ~/maria_home/perforce/home/maria/gem5-gpu/gem5-gpu/configs/gpu_config
 			cp gpgpusim.fermi.config.template."$type" gpgpusim.fermi.config.template
 			echo >> gpgpusim.fermi.config.template
 			echo "-edge_max_timer_events 4096" >> gpgpusim.fermi.config.template
 			echo "-gpgpu_max_concurrent_kernel 65" >> gpgpusim.fermi.config.template
 			echo "-edge_event_priority $prio" >> gpgpusim.fermi.config.template
 			echo "-edge_limit_concurrent_events 0" >> gpgpusim.fermi.config.template
 			echo "-edge_gen_requests_pattern_by_delay_and_limit 0" >> gpgpusim.fermi.config.template 			
 			cd ~/maria_home/perforce/home/maria/gem5-gpu/benchmarks/edge/memc_conv
 			#run test
 			../../../gem5/build/X86_VI_hammer_GPU/gem5.opt ../../../gem5-gpu/configs/se_fusion.py -c /home/maria/maria_home/perforce/home/maria/gem5-gpu/benchmarks/edge/memc_conv/gem5_fusion_memc_conv -o "-t 5 -p 2300 -n 1024 -g $bg_task" > prio_"$prio"_all_opt_"$type"_bg_task_"$bg_task".log
 			#move to results dir
 			mv prio_"$prio"_all_opt_"$type"_bg_task_"$bg_task".log results_10G_Apr04
 		done
 	done
 done
