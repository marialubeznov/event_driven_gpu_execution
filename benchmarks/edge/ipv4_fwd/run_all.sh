#!/bin/bash
#rm -rf results_256
#mkdir results_256
for type in best base no_preemption 
do
	for rate in 200 500 1000 2000 4000 8000 12000 24000
	do
	
		cd ~/maria_home/perforce/home/maria/gem5-gpu/gem5-gpu/configs/gpu_config
		cp gpgpusim.fermi.config.template."$type" gpgpusim.fermi.config.template
		cd ~/maria_home/perforce/home/maria/gem5-gpu/benchmarks/edge/ip_forward
		../../../gem5/build/X86_VI_hammer_GPU/gem5.opt --debug-flags=GPUSyscalls,CudaGPU ../../../gem5-gpu/configs/se_fusion.py -c /home/maria/maria_home/perforce/home/maria/gem5-gpu/benchmarks/edge/ip_forward/gem5_fusion_ip_forward_conv -o "-t 3 -p $rate -b 16" > "$type"_rate_"${rate}".log
	done
done
