#!/bin/bash

 for type in preemption draining
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
 			echo "-edge_limit_concurrent_events 1" >> gpgpusim.fermi.config.template
 			echo "-edge_gen_requests_pattern_by_delay_and_limit 0" >> gpgpusim.fermi.config.template
            echo "-edge_dont_launch_event_kernel 1" >> gpgpusim.fermi.config.template
 			cd ${LOCAL_GEM5_PATH}/benchmarks/edge/ipv4_fwd
 			for n in 0 1 2
			do
 				rate=`shuf -i 2000-5000 -n 1`
				qsub -v PRIO="$prio",TYPE="$type",BG_TASK="$bg_task",RATE="$rate" run_ipv4_no_overlap.pbs
				sleep 10
			done
 		done
 	done
 done
