#!/bin/bash

 for type in draining preemption 8sm_reserve
 do
 	for bg_task in 0 1 2 3
 	do
 		for prio in 1
 		do
 			cd ${LOCAL_GEM5_PATH}/gem5-gpu/configs/gpu_config
 			cp gpgpusim.fermi.config.template."$type" gpgpusim.fermi.config.template
 			echo >> gpgpusim.fermi.config.template
 			echo "-edge_max_timer_events 4096" >> gpgpusim.fermi.config.template
 			echo "-gpgpu_max_concurrent_kernel 65" >> gpgpusim.fermi.config.template
 			echo "-edge_event_priority $prio" >> gpgpusim.fermi.config.template
 			echo "-edge_limit_concurrent_events 32" >> gpgpusim.fermi.config.template
 			echo "-edge_gen_requests_pattern_by_delay_and_limit 0" >> gpgpusim.fermi.config.template 
			export config_path="config_high_util_prio_"$prio"_type_"$type"_bg_"$bg_task""
            		cp gpgpusim.fermi.config.template ${LOCAL_GEM5_PATH}/benchmarks/edge/ipv6_fwd/"$config_path"				
 			cd ${LOCAL_GEM5_PATH}/benchmarks/edge/ipv6_fwd
 			#run test
			qsub -v PRIO="$prio",TYPE="$type",BG_TASK="$bg_task",CONFIG_PATH="$config_path" run_ipv6_high_util.pbs
            sleep 5
 		done
 	done
 done
