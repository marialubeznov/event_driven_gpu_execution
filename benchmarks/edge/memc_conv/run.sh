#!/bin/bash
#rm -rf results_256
#mkdir results_256
#mkdir results_burst_opt

#save_regs=1
#for type in sched_prio sched_and_fetch_prio sched_fetch_victim_prio sched_fetch_victim_prio_flush sched_fetch_victim_prio_flush_replay sched_fetch_victim_prio_flush_replay_p2 sched_fetch_victim_prio_flush_replay_p2_stop_warps_sm
#do
#	#update config according to batch size
# 	cd ~/maria_home/perforce/home/maria/gem5-gpu/gem5-gpu/configs/gpu_config
# 	cp gpgpusim.fermi.config.template."$type" gpgpusim.fermi.config.template
# 	echo >> gpgpusim.fermi.config.template
# 	echo "-edge_run_small_event_as_fastpath 1" >> gpgpusim.fermi.config.template
# 	cd ~/maria_home/perforce/home/maria/gem5-gpu/benchmarks/edge/ip_forward
# 	#run test
# 	../../../gem5/build/X86_VI_hammer_GPU/gem5.opt ../../../gem5-gpu/configs/se_fusion.py -c /home/maria/maria_home/perforce/home/maria/gem5-gpu/benchmarks/edge/ip_forward/gem5_fusion_ip_forward_conv -o "-t 4 -p 50000 -n 1024 -r $save_regs" > burst_preemption_"$type"_matrixmul.log
# 	#move to results dir
# 	mv burst_preemption_"$type"_matrixmul.log results_burst_opt
# done

#save_regs=0
#for type in sched_prio sched_and_fetch_prio sched_fetch_victim_prio sched_fetch_victim_prio_flush sched_fetch_victim_prio_flush_replay sched_fetch_victim_prio_flush_replay_p2 sched_fetch_victim_prio_flush_replay_p2_stop_warps_sm
#do
#	#update config according to batch size
# 	cd ~/maria_home/perforce/home/maria/gem5-gpu/gem5-gpu/configs/gpu_config
# 	cp gpgpusim.fermi.config.template."$type" gpgpusim.fermi.config.template
# 	echo >> gpgpusim.fermi.config.template
# 	echo "-edge_run_small_event_as_fastpath 0" >> gpgpusim.fermi.config.template
# 	cd ~/maria_home/perforce/home/maria/gem5-gpu/benchmarks/edge/ip_forward
# 	#run test
# 	../../../gem5/build/X86_VI_hammer_GPU/gem5.opt ../../../gem5-gpu/configs/se_fusion.py -c /home/maria/maria_home/perforce/home/maria/gem5-gpu/benchmarks/edge/ip_forward/gem5_fusion_ip_forward_conv -o "-t 4 -p 50000 -n 1024 -r $save_regs" > burst_draining_"$type"_matrixmul.log
 #	#move to results dir
 #	mv burst_draining_"$type"_matrixmul.log results_burst_opt
 #done

mkdir results_burst_March18
 for type in preemption draining 8sm_reserve
 do
 	for bg_task in 0 1
 	do
 		for prio in 1 2 3
 		do
 			cd ~/maria_home/perforce/home/maria/gem5-gpu/gem5-gpu/configs/gpu_config
 			cp gpgpusim.fermi.config.template."$type" gpgpusim.fermi.config.template
 			echo >> gpgpusim.fermi.config.template
 			echo "-edge_max_timer_events 256" >> gpgpusim.fermi.config.template
 			echo "-gpgpu_max_concurrent_kernel 257" >> gpgpusim.fermi.config.template
 			echo "-edge_event_priority $prio" >> gpgpusim.fermi.config.template
 			cd ~/maria_home/perforce/home/maria/gem5-gpu/benchmarks/edge/memc_conv
 			#run test
 			../../../gem5/build/X86_VI_hammer_GPU/gem5.opt ../../../gem5-gpu/configs/se_fusion.py -c /home/maria/maria_home/perforce/home/maria/gem5-gpu/benchmarks/edge/memc_conv/gem5_fusion_memc_conv -o '-t 6 -p 100000 -n 256 -i 1000' > prio_"$prio"_all_opt_"$type"_"$bg_task".log
 			#move to results dir
 			mv prio_"$prio"_all_opt_"$type"_"$bg_task".log results_burst_March18
 		done
 	done
 done



# save_regs=1
# for type in best_p1_renaming best_p2_renaming best_p1 best_p2
# do
# 	for batch in 1024
# 	do	
# 		for rate in 1000 2000 4000 8000 16000
# 		do
# 			#update config according to batch size
# 			cd ~/maria_home/perforce/home/maria/gem5-gpu/gem5-gpu/configs/gpu_config
# 			cp gpgpusim.fermi.config.template."$type" gpgpusim.fermi.config.template
# 			echo >> gpgpusim.fermi.config.template
# 			echo "-edge_max_timer_events $batch" >> gpgpusim.fermi.config.template 
# 			echo "-gpgpu_max_concurrent_kernel $batch" >> gpgpusim.fermi.config.template 
# 			cd ~/maria_home/perforce/home/maria/gem5-gpu/benchmarks/edge/ip_forward
# 			#run test
# 			../../../gem5/build/X86_VI_hammer_GPU/gem5.opt ../../../gem5-gpu/configs/se_fusion.py -c /home/maria/maria_home/perforce/home/maria/gem5-gpu/benchmarks/edge/ip_forward/gem5_fusion_ip_forward_conv -o "-t 3 -p $rate -n $batch -r $save_regs" > warps_"$type"_rate_"$rate".log
# 			#move to results dir
# 			mv warps_"$type"_rate_"$rate".log results_Mar01
# 		done
# 	done
# done

# save_regs=0
# for type in draining_p1 reservation_1sm_p1 reservation_1cta_p1 draining_p2 reservation_1sm_p2 reservation_1cta_p2 
# do
# 	for batch in 1024
# 	do	
# 		for rate in 1000 2000 4000 8000 16000
# 		do
# 			#update config according to batch size
# 			cd ~/maria_home/perforce/home/maria/gem5-gpu/gem5-gpu/configs/gpu_config
# 			cp gpgpusim.fermi.config.template."$type" gpgpusim.fermi.config.template
# 			echo >> gpgpusim.fermi.config.template
# 			echo "-edge_max_timer_events $batch" >> gpgpusim.fermi.config.template 
# 			echo "-gpgpu_max_concurrent_kernel $batch" >> gpgpusim.fermi.config.template 
# 			cd ~/maria_home/perforce/home/maria/gem5-gpu/benchmarks/edge/ip_forward
# 			#run test
# 			../../../gem5/build/X86_VI_hammer_GPU/gem5.opt ../../../gem5-gpu/configs/se_fusion.py -c /home/maria/maria_home/perforce/home/maria/gem5-gpu/benchmarks/edge/ip_forward/gem5_fusion_ip_forward_conv -o "-t 3 -p $rate -n $batch -r $save_regs" > warps_"$type"_rate_"$rate".log
# 			#move to results dir
# 			mv warps_"$type"_rate_"$rate".log results_Mar01
# 		done
# 	done
# done

# for type in best_p2_renaming best_p1_renaming best_p1 best_p2
# do
# 	for batch in 1024
# 	do	
# 		for rate in 1000 2000 4000 8000 16000
# 		do
# 			./ExtractEvent.sh warps_"$type"_rate_"$rate".log
# 		done
# 	done
# done
