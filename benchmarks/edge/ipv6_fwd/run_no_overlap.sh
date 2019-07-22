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
 			cd ${LOCAL_GEM5_PATH}/benchmarks/edge/ipv6_fwd
 			for n in 0 1 2
			do
 				rate=`shuf -i 2000-5000 -n 1`
				../../../gem5/build/X86_VI_hammer_GPU/gem5.opt ../../../gem5-gpu/configs/se_fusion.py -c ${LOCAL_GEM5_PATH}/benchmarks/edge/ipv6_fwd/gem5_fusion_ip_forward_conv -o "-t 3 -p ${rate} -n 64 -g ${bg_task}" > prio_"${prio}"_all_opt_"${type}"_bg_task_"${bg_task}"_no_overlap_"${rate}"_rate.log
                sleep 10
			done
 		done
 	done
 done
