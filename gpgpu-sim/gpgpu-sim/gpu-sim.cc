// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung, George L. Yuan,
// Ali Bakhoda, Andrew Turner, Ivan Sham
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
// Neither the name of The University of British Columbia nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


#include "gpu/gpgpu-sim/cuda_gpu.hh"

#include "gpu-sim.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "zlib.h"


#include "shader.h"
#include "dram.h"
#include "mem_fetch.h"

#include <time.h>
#include "gpu-cache.h"
#include "gpu-misc.h"
#include "delayqueue.h"
#include "shader.h"
#include "icnt_wrapper.h"
#include "dram.h"
#include "addrdec.h"
#include "stat-tool.h"
#include "l2cache.h"

#include "../cuda-sim/ptx-stats.h"
#include "../statwrapper.h"
#include "../abstract_hardware_model.h"
#include "../debug.h"
#include "../gpgpusim_entrypoint.h"
#include "../cuda-sim/cuda-sim.h"
#include "../trace.h"
#include "mem_latency_stat.h"
#include "power_stat.h"
#include "visualizer.h"
#include "stats.h"

#include "../edge.h"
#include "../edge_helper.h"

#ifdef GPGPUSIM_POWER_MODEL
#include "power_interface.h"
#else
class  gpgpu_sim_wrapper {};
#endif

#include <stdio.h>
#include <string.h>
#include <iostream>
#include <sstream>
#include <string>

#define MAX(a,b) (((a)>(b))?(a):(b))


bool g_interactive_debugger_enabled=false;

unsigned long long  gpu_sim_cycle = 0;
unsigned long long  gpu_tot_sim_cycle = 0;


// performance counter for stalls due to congestion.
unsigned int gpu_stall_dramfull = 0; 
unsigned int gpu_stall_icnt2sh = 0;

/* Clock Domains */

#define  CORE  0x01
#define  L2    0x02
#define  DRAM  0x04
#define  ICNT  0x08  


#define MEM_LATENCY_STAT_IMPL




#include "mem_latency_stat.h"

void power_config::reg_options(class OptionParser * opp)
{


	  option_parser_register(opp, "-gpuwattch_xml_file", OPT_CSTR,
			  	  	  	  	 &g_power_config_name,"GPUWattch XML file",
	                   "gpuwattch.xml");

	   option_parser_register(opp, "-power_simulation_enabled", OPT_BOOL,
	                          &g_power_simulation_enabled, "Turn on power simulator (1=On, 0=Off)",
	                          "0");

	   option_parser_register(opp, "-power_per_cycle_dump", OPT_BOOL,
	                          &g_power_per_cycle_dump, "Dump detailed power output each cycle",
	                          "0");

	   // Output Data Formats
	   option_parser_register(opp, "-power_trace_enabled", OPT_BOOL,
	                          &g_power_trace_enabled, "produce a file for the power trace (1=On, 0=Off)",
	                          "0");

	   option_parser_register(opp, "-power_trace_zlevel", OPT_INT32,
	                          &g_power_trace_zlevel, "Compression level of the power trace output log (0=no comp, 9=highest)",
	                          "6");

	   option_parser_register(opp, "-steady_power_levels_enabled", OPT_BOOL,
	                          &g_steady_power_levels_enabled, "produce a file for the steady power levels (1=On, 0=Off)",
	                          "0");

	   option_parser_register(opp, "-steady_state_definition", OPT_CSTR,
			   	  &gpu_steady_state_definition, "allowed deviation:number of samples",
	                 	  "8:4");

}

void memory_config::reg_options(class OptionParser * opp)
{
    option_parser_register(opp, "-gpgpu_dram_scheduler", OPT_INT32, &scheduler_type, 
                                "0 = fifo, 1 = FR-FCFS (defaul)", "1");
    option_parser_register(opp, "-gpgpu_dram_partition_queues", OPT_CSTR, &gpgpu_L2_queue_config, 
                           "i2$:$2d:d2$:$2i",
                           "8:8:8:8");

    option_parser_register(opp, "-l2_ideal", OPT_BOOL, &l2_ideal, 
                           "Use a ideal L2 cache that always hit",
                           "0");
    option_parser_register(opp, "-gpgpu_cache:dl2", OPT_CSTR, &m_L2_config.m_config_string, 
                   "unified banked L2 data cache config "
                   " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq>}",
                   "64:128:8,L:B:m:N,A:16:4,4");
    option_parser_register(opp, "-gpgpu_cache:dl2_texture_only", OPT_BOOL, &m_L2_texure_only, 
                           "L2 cache used for texture only",
                           "1");
    option_parser_register(opp, "-gpgpu_n_mem", OPT_UINT32, &m_n_mem, 
                 "number of memory modules (e.g. memory controllers) in gpu",
                 "8");
    option_parser_register(opp, "-gpgpu_n_sub_partition_per_mchannel", OPT_UINT32, &m_n_sub_partition_per_memory_channel, 
                 "number of memory subpartition in each memory module",
                 "1");
    option_parser_register(opp, "-gpgpu_n_mem_per_ctrlr", OPT_UINT32, &gpu_n_mem_per_ctrlr, 
                 "number of memory chips per memory controller",
                 "1");
    option_parser_register(opp, "-gpgpu_memlatency_stat", OPT_INT32, &gpgpu_memlatency_stat, 
                "track and display latency statistics 0x2 enables MC, 0x4 enables queue logs",
                "0");
    option_parser_register(opp, "-gpgpu_frfcfs_dram_sched_queue_size", OPT_INT32, &gpgpu_frfcfs_dram_sched_queue_size, 
                "0 = unlimited (default); # entries per chip",
                "0");
    option_parser_register(opp, "-gpgpu_dram_return_queue_size", OPT_INT32, &gpgpu_dram_return_queue_size, 
                "0 = unlimited (default); # entries per chip",
                "0");
    option_parser_register(opp, "-gpgpu_dram_buswidth", OPT_UINT32, &busW, 
                 "default = 4 bytes (8 bytes per cycle at DDR)",
                 "4");
    option_parser_register(opp, "-gpgpu_dram_burst_length", OPT_UINT32, &BL, 
                 "Burst length of each DRAM request (default = 4 data bus cycle)",
                 "4");
    option_parser_register(opp, "-dram_data_command_freq_ratio", OPT_UINT32, &data_command_freq_ratio, 
                 "Frequency ratio between DRAM data bus and command bus (default = 2 times, i.e. DDR)",
                 "2");
    option_parser_register(opp, "-gpgpu_dram_timing_opt", OPT_CSTR, &gpgpu_dram_timing_opt, 
                "DRAM timing parameters = {nbk:tCCD:tRRD:tRCD:tRAS:tRP:tRC:CL:WL:tCDLR:tWR:nbkgrp:tCCDL:tRTPL}",
                "4:2:8:12:21:13:34:9:4:5:13:1:0:0");
    option_parser_register(opp, "-rop_latency", OPT_UINT32, &rop_latency,
                     "ROP queue latency (default 85)",
                     "85");
    option_parser_register(opp, "-dram_latency", OPT_UINT32, &dram_latency,
                     "DRAM latency (default 30)",
                     "30");

    m_address_mapping.addrdec_setoption(opp);
}

void shader_core_config::reg_options(class OptionParser * opp)
{
    option_parser_register(opp, "-gpgpu_simd_model", OPT_INT32, &model, 
                   "1 = post-dominator", "1");
    option_parser_register(opp, "-gpgpu_shader_core_pipeline", OPT_CSTR, &gpgpu_shader_core_pipeline_opt, 
                   "shader core pipeline config, i.e., {<nthread>:<warpsize>}",
                   "1024:32");
    option_parser_register(opp, "-gpgpu_tex_cache:l1", OPT_CSTR, &m_L1T_config.m_config_string, 
                   "per-shader L1 texture cache  (READ-ONLY) config "
                   " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq>:<rf>}",
                   "8:128:5,L:R:m:N,F:128:4,128:2");
    option_parser_register(opp, "-gpgpu_const_cache:l1", OPT_CSTR, &m_L1C_config.m_config_string, 
                   "per-shader L1 constant memory cache  (READ-ONLY) config "
                   " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq>} ",
                   "64:64:2,L:R:f:N,A:2:32,4" );
    option_parser_register(opp, "-gpgpu_cache:il1", OPT_CSTR, &m_L1I_config.m_config_string, 
                   "shader L1 instruction cache config "
                   " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq>} ",
                   "4:256:4,L:R:f:N,A:2:32,4" );
    option_parser_register(opp, "-gpgpu_cache:dl1", OPT_CSTR, &m_L1D_config.m_config_string,
                   "per-shader L1 data cache config "
                   " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq> | none}",
                   "none" );
    option_parser_register(opp, "-gpgpu_cache:dl1PrefL1", OPT_CSTR, &m_L1D_config.m_config_stringPrefL1,
                   "per-shader L1 data cache config "
                   " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq> | none}",
                   "none" );
    option_parser_register(opp, "-gpgpu_cache:dl1PreShared", OPT_CSTR, &m_L1D_config.m_config_stringPrefShared,
                   "per-shader L1 data cache config "
                   " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq> | none}",
                   "none" );
    option_parser_register(opp, "-gmem_skip_L1D", OPT_BOOL, &gmem_skip_L1D, 
                   "global memory access skip L1D cache (implements -Xptxas -dlcm=cg, default=no skip)",
                   "0");

    option_parser_register(opp, "-gpgpu_perfect_mem", OPT_BOOL, &gpgpu_perfect_mem, 
                 "enable perfect memory mode (no cache miss)",
                 "0");
    option_parser_register(opp, "-n_regfile_gating_group", OPT_UINT32, &n_regfile_gating_group,
                 "group of lanes that should be read/written together)",
                 "4");
    option_parser_register(opp, "-gpgpu_clock_gated_reg_file", OPT_BOOL, &gpgpu_clock_gated_reg_file,
                 "enable clock gated reg file for power calculations",
                 "0");
    option_parser_register(opp, "-gpgpu_clock_gated_lanes", OPT_BOOL, &gpgpu_clock_gated_lanes,
                 "enable clock gated lanes for power calculations",
                 "0");
    option_parser_register(opp, "-gpgpu_shader_registers", OPT_UINT32, &gpgpu_shader_registers, 
                 "Number of registers per shader core. Limits number of concurrent CTAs. (default 8192)",
                 "8192");
    option_parser_register(opp, "-gpgpu_shader_cta", OPT_UINT32, &max_cta_per_core, 
                 "Maximum number of concurrent CTAs in shader (default 8)",
                 "8");
    option_parser_register(opp, "-gpgpu_n_clusters", OPT_UINT32, &n_simt_clusters, 
                 "number of processing clusters",
                 "10");
    option_parser_register(opp, "-gpgpu_n_cores_per_cluster", OPT_UINT32, &n_simt_cores_per_cluster, 
                 "number of simd cores per cluster",
                 "3");
    option_parser_register(opp, "-gpgpu_n_cluster_ejection_buffer_size", OPT_UINT32, &n_simt_ejection_buffer_size, 
                 "number of packets in ejection buffer",
                 "8");
    option_parser_register(opp, "-gpgpu_n_ldst_response_buffer_size", OPT_UINT32, &ldst_unit_response_queue_size, 
                 "number of response packets in ld/st unit ejection buffer",
                 "2");
    option_parser_register(opp, "-gpgpu_shmem_size", OPT_UINT32, &gpgpu_shmem_size,
                 "Size of shared memory per shader core (default 48kB)",
                 "49152");
    option_parser_register(opp, "-gpgpu_shmem_size", OPT_UINT32, &gpgpu_shmem_sizeDefault,
                 "Default Size of shared memory per shader core (default 16kB)",
                 "16384");
    option_parser_register(opp, "-gpgpu_shmem_size_PrefL1", OPT_UINT32, &gpgpu_shmem_sizePrefL1,
                 "Prefered L1 Size of shared memory per shader core (default 16kB)",
                 "16384");
    option_parser_register(opp, "-gpgpu_shmem_size_PrefShared", OPT_UINT32, &gpgpu_shmem_sizePrefShared,
                 "Prefered shared Size of shared memory per shader core (default 16kB)",
                 "49152");
    option_parser_register(opp, "-gpgpu_shmem_access_latency", OPT_UINT32, &gpgpu_shmem_access_latency,
                 "Shared load buffer depth (default 13: Fermi, Maxwell = 21)",
                 "13");
    option_parser_register(opp, "-gpgpu_shmem_num_banks", OPT_UINT32, &num_shmem_bank, 
                 "Number of banks in the shared memory in each shader core (default 16)",
                 "16");
    option_parser_register(opp, "-gpgpu_shmem_limited_broadcast", OPT_BOOL, &shmem_limited_broadcast, 
                 "Limit shared memory to do one broadcast per cycle (default on)",
                 "1");
    option_parser_register(opp, "-gpgpu_shmem_warp_parts", OPT_INT32, &mem_warp_parts,  
                 "Number of portions a warp is divided into for shared memory bank conflict check ",
                 "2");
    option_parser_register(opp, "-gpgpu_warpdistro_shader", OPT_INT32, &gpgpu_warpdistro_shader, 
                "Specify which shader core to collect the warp size distribution from", 
                "-1");
    option_parser_register(opp, "-gpgpu_warp_issue_shader", OPT_INT32, &gpgpu_warp_issue_shader, 
                "Specify which shader core to collect the warp issue distribution from", 
                "0");
    option_parser_register(opp, "-gpgpu_local_mem_map", OPT_BOOL, &gpgpu_local_mem_map, 
                "Mapping from local memory space address to simulated GPU physical address space (default = enabled)", 
                "1");
    option_parser_register(opp, "-gpgpu_num_reg_banks", OPT_INT32, &gpgpu_num_reg_banks, 
                "Number of register banks (default = 8)", 
                "8");
    option_parser_register(opp, "-gpgpu_reg_bank_use_warp_id", OPT_BOOL, &gpgpu_reg_bank_use_warp_id,
             "Use warp ID in mapping registers to banks (default = off)",
             "0");
    option_parser_register(opp, "-gpgpu_operand_collector_num_units_sp", OPT_INT32, &gpgpu_operand_collector_num_units_sp,
                "number of collector units (default = 4)", 
                "4");
    option_parser_register(opp, "-gpgpu_operand_collector_num_units_sfu", OPT_INT32, &gpgpu_operand_collector_num_units_sfu,
                "number of collector units (default = 4)", 
                "4");
    option_parser_register(opp, "-gpgpu_operand_collector_num_units_mem", OPT_INT32, &gpgpu_operand_collector_num_units_mem,
                "number of collector units (default = 2)", 
                "2");
    option_parser_register(opp, "-gpgpu_operand_collector_num_units_gen", OPT_INT32, &gpgpu_operand_collector_num_units_gen,
                "number of collector units (default = 0)", 
                "0");
    option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_sp", OPT_INT32, &gpgpu_operand_collector_num_in_ports_sp,
                           "number of collector unit in ports (default = 1)", 
                           "1");
    option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_sfu", OPT_INT32, &gpgpu_operand_collector_num_in_ports_sfu,
                           "number of collector unit in ports (default = 1)", 
                           "1");
    option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_mem", OPT_INT32, &gpgpu_operand_collector_num_in_ports_mem,
                           "number of collector unit in ports (default = 1)", 
                           "1");
    option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_gen", OPT_INT32, &gpgpu_operand_collector_num_in_ports_gen,
                           "number of collector unit in ports (default = 0)", 
                           "0");
    option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_sp", OPT_INT32, &gpgpu_operand_collector_num_out_ports_sp,
                           "number of collector unit in ports (default = 1)", 
                           "1");
    option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_sfu", OPT_INT32, &gpgpu_operand_collector_num_out_ports_sfu,
                           "number of collector unit in ports (default = 1)", 
                           "1");
    option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_mem", OPT_INT32, &gpgpu_operand_collector_num_out_ports_mem,
                           "number of collector unit in ports (default = 1)", 
                           "1");
    option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_gen", OPT_INT32, &gpgpu_operand_collector_num_out_ports_gen,
                           "number of collector unit in ports (default = 0)", 
                           "0");
    option_parser_register(opp, "-gpgpu_coalesce_arch", OPT_INT32, &gpgpu_coalesce_arch, 
                            "Coalescing arch (default = 13, anything else is off for now)", 
                            "13");
    option_parser_register(opp, "-gpgpu_cycle_sched_prio", OPT_BOOL, &gpgpu_cycle_sched_prio,
                            "Whether to cycle the priority of warp schedulers (default=false)",
                            "0");
    option_parser_register(opp, "-gpgpu_num_sched_per_core", OPT_INT32, &gpgpu_num_sched_per_core, 
                            "Number of warp schedulers per core", 
                            "1");
    option_parser_register(opp, "-gpgpu_max_insn_issue_per_warp", OPT_INT32, &gpgpu_max_insn_issue_per_warp,
                            "Max number of instructions that can be issued per warp in one cycle by scheduler",
                            "2");
    option_parser_register(opp, "-gpgpu_simt_core_sim_order", OPT_INT32, &simt_core_sim_order,
                            "Select the simulation order of cores in a cluster (0=Fix, 1=Round-Robin)",
                            "1");
    option_parser_register(opp, "-gpgpu_pipeline_widths", OPT_CSTR, &pipeline_widths_string,
                            "Pipeline widths "
                            "ID_OC_SP,ID_OC_SFU,ID_OC_MEM,OC_EX_SP,OC_EX_SFU,OC_EX_MEM,EX_WB",
                            "1,1,1,1,1,1,1" );
    option_parser_register(opp, "-gpgpu_num_sp_units", OPT_INT32, &gpgpu_num_sp_units,
                            "Number of SP units (default=1)",
                            "1");
    option_parser_register(opp, "-gpgpu_num_sfu_units", OPT_INT32, &gpgpu_num_sfu_units,
                            "Number of SF units (default=1)",
                            "1");
    option_parser_register(opp, "-gpgpu_num_mem_units", OPT_INT32, &gpgpu_num_mem_units,
                            "Number if ldst units (default=1) WARNING: not hooked up to anything",
                             "1");
    option_parser_register(opp, "-gpgpu_scheduler", OPT_CSTR, &gpgpu_scheduler_string,
                                "Scheduler configuration: < lrr | gto | two_level_active > "
                                "If two_level_active:<num_active_warps>:<inner_prioritization>:<outer_prioritization>"
                                "For complete list of prioritization values see shader.h enum scheduler_prioritization_type"
                                "Default: gto",
                                 "gto");
    option_parser_register(opp, "-gpgpu_fetch_decode_width", OPT_INT32, &gpgpu_fetch_decode_width,
                            "Number of instructions to fetch per cycle (default=2)",
                            "2");

    option_parser_register(opp, "-edge_memc_conv", OPT_BOOL, &_isMemcConv, 
                 "Is this a Memcached + Convolution benchmark (default = No)",
                 "0");

    // EDGE
    option_parser_register(opp, "-edge_print_stat", OPT_BOOL, &_edgePrintStat, 
                 "print runtime statistics for EDGE",
                 "1");

    option_parser_register(opp, "-edge_int_mode", OPT_BOOL, &_intMode, 
                 "enable interrupt mode (single interrupt handler warp per SM (default = disabled))",
                 "0");

    option_parser_register(opp, "-edge_int_config", OPT_CSTR, &_edgeIntConfig,
                                "EDGE Interrupt configuration: < # CTAs : # warps > "
                                "Default: 0:0",
                                 "0:0");

    option_parser_register(opp, "-edge_warp_selection", OPT_CSTR, &_edgeWarpSelectionStr,
    "EDGE Interrupt warp selection configuration: < dedicated | oldest | newest | best | psuedo_dedicated > "
                                "Default: dedicated",
                                 "dedicated");

    option_parser_register(opp, "-edge_internal_int_mode", OPT_BOOL, &_edgeInternalInt, 
                 "enable internal interrupt generation (default = disabled)",
                 "0");

    option_parser_register(opp, "-edge_internal_int_period", OPT_UINT32, &_edgeInternalIntPeriod, 
                 "Number of cycles between internally generated interrupts (default 5000)",
                 "5000");

    option_parser_register(opp, "-edge_internal_int_delay", OPT_UINT32, &_edgeInternalIntDelay, 
              "Number of dummy operations to perform in the interrupt to change ISR time (default 0)",           
              "0");

    option_parser_register(opp, "-edge_internal_event_id", OPT_UINT32, &_edgeInternalEventId, 
                 "Event ID to launch on internally generated interrupt (default = 0 [NULL EVENT])",
                 "0");

    // CDP - Concurrent kernel SM
    option_parser_register(opp, "-gpgpu_concurrent_kernel_sm", OPT_BOOL, &gpgpu_concurrent_kernel_sm, 
                 "Support concurrent kernels on a SM (default = disabled)", 
                 "0");

    option_parser_register(opp, "-edge_victim_warp_high_priority", OPT_BOOL, &_edgeVictimWarpHighPriority, 
                 "Set the priority of a victim interrupt warp to the highest when flushing (default = false))",
                 "0");

    option_parser_register(opp, "-edge_int_schedule_high_priority", OPT_BOOL, &_edgeIntSchedulePriority, 
                 "Interrupt warp high SCHEDULE priority (default = false))",
                 "0");
    
    option_parser_register(opp, "-edge_int_fetch_high_priority", OPT_BOOL, &_edgeIntFetchPriority, 
                 "Interrupt warp high FETCH priority (default = false))",
                 "0");

    option_parser_register(opp, "-edge_event_kernel_launch_latency", OPT_UINT32, &_edgeEventKernelLaunchLatency, 
                 "Event kernel launch latency in GPU cycles (default = 0 [no latency])",
                 "0");


    option_parser_register(opp, "-edge_flush_ibuffer", OPT_BOOL, &_edgeFlushIBuffer, 
                 "Interrupt warp flush iBuffer on pipeline flush (default = no)",
                 "0");

    option_parser_register(opp, "-edge_replay_loads", OPT_BOOL, &_edgeReplayLoads, 
                 "Drop loads and replay loads on interrupt (default = no)",
                 "0");

    option_parser_register(opp, "-edge_event_priority", OPT_UINT32, &_edgeEventPriority, 
                 "Does an event kernel get priority for scheudling? (0=No, block) (1=yes, start selecting when possible)"
                 " (2=stop all other CTAs on any SM running an event kernel to prioritize that kernel)",
                 " (3=stop fetching and scheduling warps if event kernel is running on a GPU"
                 "0");

   option_parser_register(opp, "-edge_warmup_int", OPT_BOOL, &_edgeWarmupInt, 
                 "Interrupt warp high FETCH priority (default = false))",
                 "0");

   option_parser_register(opp, "-edge_int_reserve_icache", OPT_BOOL, &_edgeIntReserveIcache, 
                 "Interrupt warp high FETCH priority (default = false))",
                 "0");

    option_parser_register(opp, "-edge_event_kernel_reserve_cta", OPT_UINT32, &_edgeEventReserveCta, 
                 "Number of CTAs to reserve per SM for each event kernel (default = 0)",
                 "0");

    option_parser_register(opp, "-edge_event_kernel_reserve_sm", OPT_UINT32, &_edgeEventReserveSm, 
                 "Number of SMs for event kernels (default = 0)",
                 "0");

    option_parser_register(opp, "-edge_max_timer_events", OPT_UINT32, &_edgeMaxTimerEvents, 
                 "Max number of timer events to launch on the GPU (default = 65536)",
                 "65536");

   option_parser_register(opp, "-edge_timer_event_only", OPT_BOOL, &_edgeTimerEventOnly, 
                 "If a timer event kernel will be running by itself (default = false))",
                 "0");

   option_parser_register(opp, "-edge_skip_barrier", OPT_BOOL, &_edgeSkipBarrier, 
                 "Can we select a victim warp to interrupt waiting at a barrier (default = false)",
                 "0");

   option_parser_register(opp, "-edge_single_sm_for_isr", OPT_BOOL, &_edgeSingleSmForIsr, 
                 "Run ISR on a dedicated SM (analogues to command processor) (default = false)",
                 "0");

   option_parser_register(opp, "-edge_single_sm_for_isr_idx", OPT_UINT32, &_edgeSingleSmForIsrIdx, 
                 "Valid only when _edgeSingleSmForIsr=1. The idx of the SM to run the ISR (default = 15)",
                 "15");

   option_parser_register(opp, "-edge_use_all_sms_for_event_kernels", OPT_BOOL, &_edgeUseAllSmsForEventKernels, 
                 "Run event kernels even on not reserved SMs (default = 0)",
                 "0");

   option_parser_register(opp, "-edge_event_kernel_preempt_sm", OPT_UINT32, &_edgeEventPreemptSm, 
                 "Number of SMs that are being preempted for event kernels (default = 0)",
                 "0");   
   option_parser_register(opp, "-edge_run_isr", OPT_BOOL, &_edgeRunISR, 
                 "Run EDGE in old mode, with ISR launching the kernel (default = 0)",
                 "0");  
   option_parser_register(opp, "-edge_run_small_event_as_fastpath", OPT_BOOL, &_edgeRunSmallEventAsFastPath, 
                 "Run warp sized events as fast path (preempting a warp) (default = 1)",
                 "1"); 
   option_parser_register(opp, "-edge_use_issue_block_2core_when_free_warp", OPT_BOOL, &_edgeUseIssueBlock2CoreWhenFreeWarp, 
                 "Run warp sized events through issue_block2core when CTA is free(default = 0)",
                 "0"); 
   option_parser_register(opp, "-edge_use_int_coreid", OPT_BOOL, &_edgeUseIntCoreId, 
                 "Schedule the event on core id provided in interrupt request (default = 0)",
                 "0"); 
   option_parser_register(opp, "-edge_stop_other_warps_in_preempted_cta", OPT_BOOL, &_edgeStopOtherWarpsInPreemptedCTA, 
                 "Stop other warps in the preempted CTA while event is running in an iwarp (default = 1)",
                 "1"); 
   option_parser_register(opp, "-edge_stop_other_warps_in_preempted_sm", OPT_BOOL, &_edgeStopOtherWarpsInPreemptedSm, 
                 "Stop other warps in the preempted SM while event is running (default = 0)",
                 "0"); 
   option_parser_register(opp, "-edge_event_ctas_per_core", OPT_UINT32, &_edgeEventCtasPerCore, 
                 "Number of additional CTAs for event kernels per core (default 32)",
                 "32");
   option_parser_register(opp, "-edge_event_start_cycle", OPT_UINT32, &_edgeEventStartCycle, 
                 "GPU cycle of the first timer event (default 200)",
                 "200");
   option_parser_register(opp, "-edge_enable_register_renaming_instead_backup", OPT_BOOL, &_edgeEnableRegisterRenamingInsteadBackup, 
                 "When register file utilization allows, no need to save&restore victim warp's regs. Can use available ones instead. (default = 1)",
                 "1"); 
   option_parser_register(opp, "-edge_limit_concurrent_events", OPT_UINT32, &_edgeLimitConcurrentEvents, 
                 "Generate events with preconfigured rate, untill reached maximum of running events (default = 0)",
                 "0"); 
   option_parser_register(opp, "-edge_gen_requests_pattern_by_delay_and_limit", OPT_BOOL, &_edgeGenRequestsPatternByDelayAndLimit, 
                 "Generate events with preconfigured rate, untill reached maximum of running events. For batches, the delay is defined as number of cycles since last batch completes (default = 0)",
                 "0"); 
   option_parser_register(opp, "-edge_dont_launch_event_kernel", OPT_BOOL, &_edgeDontLaunchEventKernel,
                 "Just perform the preemption, don't launch the kernel (default = 0)",
                 "0");    
}

void gpgpu_sim_config::reg_options(option_parser_t opp)
{
    gpgpu_functional_sim_config::reg_options(opp);
    m_shader_config.reg_options(opp);
    m_memory_config.reg_options(opp);
    power_config::reg_options(opp);
   option_parser_register(opp, "-gpgpu_max_cycle", OPT_INT32, &gpu_max_cycle_opt, 
               "terminates gpu simulation early (0 = no limit)",
               "0");
   option_parser_register(opp, "-gpgpu_max_insn", OPT_INT32, &gpu_max_insn_opt, 
               "terminates gpu simulation early (0 = no limit)",
               "0");
   option_parser_register(opp, "-gpgpu_max_cta", OPT_INT32, &gpu_max_cta_opt, 
               "terminates gpu simulation early (0 = no limit)",
               "0");
   option_parser_register(opp, "-gpgpu_runtime_stat", OPT_CSTR, &gpgpu_runtime_stat, 
                  "display runtime statistics such as dram utilization {<freq>:<flag>}",
                  "10000:0");
   option_parser_register(opp, "-liveness_message_freq", OPT_INT64, &liveness_message_freq, 
               "Minimum number of seconds between simulation liveness messages (0 = always print)",
               "1");
   option_parser_register(opp, "-gpgpu_flush_l1_cache", OPT_BOOL, &gpgpu_flush_l1_cache,
                "Flush L1 cache at the end of each kernel call",
                "0");
   option_parser_register(opp, "-gpgpu_flush_l2_cache", OPT_BOOL, &gpgpu_flush_l2_cache,
                   "Flush L2 cache at the end of each kernel call",
                   "0");

   option_parser_register(opp, "-gpgpu_deadlock_detect", OPT_BOOL, &gpu_deadlock_detect, 
                "Stop the simulation at deadlock (1=on (default), 0=off)", 
                "1");
   option_parser_register(opp, "-gpgpu_ptx_instruction_classification", OPT_INT32, 
               &gpgpu_ptx_instruction_classification, 
               "if enabled will classify ptx instruction types per kernel (Max 255 kernels now)", 
               "0");
   option_parser_register(opp, "-gpgpu_ptx_sim_mode", OPT_INT32, &g_ptx_sim_mode, 
               "Select between Performance (default) or Functional simulation (1)", 
               "0");
   option_parser_register(opp, "-gpgpu_clock_domains", OPT_CSTR, &gpgpu_clock_domains, 
                  "Clock Domain Frequencies in MhZ {<Core Clock>:<ICNT Clock>:<L2 Clock>:<DRAM Clock>}",
                  "500.0:2000.0:2000.0:2000.0");
   option_parser_register(opp, "-gpgpu_max_concurrent_kernel", OPT_INT32, &max_concurrent_kernel,
                          "maximum kernels that can run concurrently on GPU", "32" /* "8" */ );
   option_parser_register(opp, "-gpgpu_cflog_interval", OPT_INT32, &gpgpu_cflog_interval, 
               "Interval between each snapshot in control flow logger", 
               "0");
   option_parser_register(opp, "-visualizer_enabled", OPT_BOOL,
                          &g_visualizer_enabled, "Turn on visualizer output (1=On, 0=Off)",
                          "1");
   option_parser_register(opp, "-visualizer_outputfile", OPT_CSTR, 
                          &g_visualizer_filename, "Specifies the output log file for visualizer",
                          NULL);
   option_parser_register(opp, "-visualizer_zlevel", OPT_INT32,
                          &g_visualizer_zlevel, "Compression level of the visualizer output log (0=no comp, 9=highest)",
                          "6");
    option_parser_register(opp, "-trace_enabled", OPT_BOOL, 
                          &Trace_gpgpu::enabled, "Turn on traces",
                          "0");
    option_parser_register(opp, "-trace_components", OPT_CSTR, 
                          &Trace_gpgpu::config_str, "comma seperated list of traces to enable. "
                          "Complete list found in trace_streams.tup. "
                          "Default none",
                          "none");
    option_parser_register(opp, "-trace_sampling_core", OPT_INT32, 
                          &Trace_gpgpu::sampling_core, "The core which is printed using CORE_DPRINTF. Default 0",
                          "0");
    option_parser_register(opp, "-trace_sampling_memory_partition", OPT_INT32, 
                          &Trace_gpgpu::sampling_memory_partition, "The memory partition which is printed using MEMPART_DPRINTF. Default -1 (i.e. all)",
                          "-1");
   ptx_file_line_stats_options(opp);
}

/////////////////////////////////////////////////////////////////////////////

void increment_x_then_y_then_z( dim3 &i, const dim3 &bound)
{
   i.x++;
   if ( i.x >= bound.x ) {
      i.x = 0;
      i.y++;
      if ( i.y >= bound.y ) {
         i.y = 0;
         if( i.z < bound.z ) 
            i.z++;
      }
   }
}

void gpgpu_sim::QueueEventKernel(kernel_info_t* kernel) 
{
    // Construct a TMD entry for the event kernel queue, which is a pair of launch latency delay and kernel_info_t pointer
    TMDEntry tmde = std::make_pair(m_shader_config->_edgeEventKernelLaunchLatency, kernel);
    _eventKernelQueue.push_back(tmde); 
}

void gpgpu_sim::LaunchEventKernel()
{
    // Check conditions to schedule event kernel:
    //      1: A free kernel spot is available
    //      2: There is a pending event kernel to run
    //      3: The launch latency for the pending event kernel is complete
    if( kernelSpotAvailable() && !_eventKernelQueue.empty()  ) {
        TMDEntry& tmde = _eventKernelQueue.front();
        if( tmde.first == 0 ) {         // Launch latency is done
            launch( tmde.second );      // Launch the event kernel
            _eventKernelQueue.erase(_eventKernelQueue.begin()); // Remove from the queue
        }
    }

    // For all entries in the event kernel queue, decrement their launch latency
    for( TMDQueue::iterator it = _eventKernelQueue.begin(); it != _eventKernelQueue.end(); ++it ) {
        if( it->first > 0 )
            it->first--;        
    }
}

void gpgpu_sim::launch( kernel_info_t *kinfo )
{
   unsigned cta_size = kinfo->threads_per_cta();
   if ( cta_size > m_shader_config->n_thread_per_shader ) {
      printf("Execution error: Shader kernel CTA (block) size is too large for microarch config.\n");
      printf("                 CTA size (x*y*z) = %u, max supported = %u\n", cta_size, 
             m_shader_config->n_thread_per_shader );
      printf("                 => either change -gpgpu_shader argument in gpgpusim.config file or\n");
      printf("                 modify the CUDA source to decrease the kernel block size.\n");
      abort();
   }

    // EDGE HACK:
    //if( kinfo->isEventKernel() == 2 ) {
    //    QueueEventKernel(kinfo);
    //    return;
    //}

   unsigned n=0;
   for(n=0; n < m_running_kernels.size(); n++ ) {
       if( (NULL==m_running_kernels[n]) || m_running_kernels[n]->done() ) {
           m_running_kernels[n] = kinfo;
           break;
       }
   }
   assert(n < m_running_kernels.size());
}

bool gpgpu_sim::kernelSpotAvailable()
{
    for(unsigned n=0; n < m_running_kernels.size(); n++ ) {
        if( (NULL==m_running_kernels[n]) || m_running_kernels[n]->done() ) 
            return true;
    }
    return false;
}

bool gpgpu_sim::can_start_kernel()
{
   // EDGE: FIXME: Pending event kernels get priority
   if( !_eventKernelQueue.empty() ) 
       return false;
    
   return kernelSpotAvailable();
}

bool gpgpu_sim::hit_max_cta_count() const 
{
   if( m_config.gpu_max_cta_opt != 0 ) {
      if( (gpu_tot_issued_cta + m_total_cta_launched) >= m_config.gpu_max_cta_opt )
          return true;
   }
   return false;
}

bool gpgpu_sim::kernel_more_cta_left(kernel_info_t *kernel) const 
{
    if( hit_max_cta_count() )
       return false;

    if( kernel && !kernel->no_more_ctas_to_run() )
        return true;

    return false;
}


bool gpgpu_sim::get_more_cta_left() const
{ 
   if (m_config.gpu_max_cta_opt != 0) {
      if( m_total_cta_launched >= m_config.gpu_max_cta_opt )
          return false;
   }
   
   // EDGE
   if( !_eventKernelQueue.empty() )
       return true;

   for(unsigned n=0; n < m_running_kernels.size(); n++ ) {
       if( m_running_kernels[n] && !m_running_kernels[n]->no_more_ctas_to_run() ) 
           return true;
   }
   return false;
}

kernel_info_t *gpgpu_sim::select_kernel(unsigned coreId)
{

    bool eventRunning = false;
    if( m_shader_config->_edgeEventPriority > 0 ) {

        // First check if any Events are running
        for( unsigned n=0; n<m_running_kernels.size(); ++n ) {
            if( m_running_kernels[n] != NULL && m_running_kernels[n]->isEventKernel() && m_running_kernels[n]->running())  // && !m_running_kernels[n]->GetPreemption() ) 
                eventRunning = true;
        }
        
        // If we're not reserving an SM, always check for the event kernel first. If we're reserving SMs for the event, 
        // make sure that this is one of the reserved SMs if we're going to look at selecting an event kernel
        if( ( m_shader_config->_edgeEventReserveSm == 0 && m_shader_config->_edgeEventPreemptSm == 0 ) || (coreId < m_shader_config->_edgeEventReserveSm) || 
          (coreId < m_shader_config->_edgeEventPreemptSm) || m_shader_config->_edgeUseAllSmsForEventKernels ) { 

            // EDGE: First give priority to Event Kernels. Search for event kernel with lowest start_cycle. 
            unsigned long long oldestEvent = 0xFFFFFFFFFFFFFFFF;
            int oldestEventIdx = -1;
            for( unsigned n=0; n<m_running_kernels.size(); ++n ) {
                if( m_running_kernels[n] != NULL && m_running_kernels[n]->isEventKernel() && !m_running_kernels[n]->no_more_ctas_to_run() && !m_running_kernels[n]->GetPreemption()) {
                    unsigned launch_uid = m_running_kernels[n]->get_uid();   

                    if( m_running_kernels[n]->start_cycle > 0 ) { // If this event kernel has been launched already
                        assert(std::find(m_executed_kernel_uids.begin(), m_executed_kernel_uids.end(), launch_uid) != m_executed_kernel_uids.end());
                        if( m_running_kernels[n]->start_cycle < oldestEvent ) { // If the start cycle is the smallest (oldest)
                            oldestEvent = m_running_kernels[n]->start_cycle; // Select this event kernel to run
                            oldestEventIdx = n;
                        }
                    } else {
                        // Otherwise, we should not have launched the event kernel yet. So select it if none have been selected yet. 
                        assert(std::find(m_executed_kernel_uids.begin(), m_executed_kernel_uids.end(), launch_uid) == m_executed_kernel_uids.end());
                        if( oldestEventIdx == -1 )
                            oldestEventIdx = n;
                    }
                }
            }

            if( oldestEventIdx != -1 ) {
                // We found an event kernel to launch (note, oldest logic should always choose the same one until complete).
                unsigned launch_uid = m_running_kernels[oldestEventIdx]->get_uid(); 
                if( std::find(m_executed_kernel_uids.begin(), m_executed_kernel_uids.end(), launch_uid) == m_executed_kernel_uids.end() ) {
                    m_running_kernels[oldestEventIdx]->start_cycle = gpu_sim_cycle + gpu_tot_sim_cycle;
                    m_executed_kernel_uids.push_back(launch_uid);
                    m_executed_kernel_names.push_back(m_running_kernels[oldestEventIdx]->name());
                }

                return m_running_kernels[oldestEventIdx];
            }
        }
    }

    if( m_shader_config->_edgeEventPriority <= 1 || !eventRunning ) {
        // CDP - Concurrent kernel SM
        // Always select the last issued kernel first 
        if( m_running_kernels[m_last_issued_kernel] && !m_running_kernels[m_last_issued_kernel]->no_more_ctas_to_run() && !m_running_kernels[m_last_issued_kernel]->GetPreemption()) {
            unsigned launch_uid = m_running_kernels[m_last_issued_kernel]->get_uid(); 

            if( std::find(m_executed_kernel_uids.begin(), m_executed_kernel_uids.end(), launch_uid) == m_executed_kernel_uids.end() ) {
                m_running_kernels[m_last_issued_kernel]->start_cycle = gpu_sim_cycle + gpu_tot_sim_cycle;
                m_executed_kernel_uids.push_back(launch_uid); 
                m_executed_kernel_names.push_back(m_running_kernels[m_last_issued_kernel]->name()); 
            }

            return m_running_kernels[m_last_issued_kernel];
        }    


        for(unsigned n=0; n < m_running_kernels.size(); n++ ) {
            unsigned idx = (n+m_last_issued_kernel+1)%m_config.max_concurrent_kernel;

            if( kernel_more_cta_left(m_running_kernels[idx])) {
                if( (m_shader_config->_edgeEventPriority == 0 && !m_shader_config->_edgeRunSmallEventAsFastPath) || !m_running_kernels[idx]->isEventKernel() ) {
                    m_last_issued_kernel=idx;
                    m_running_kernels[m_last_issued_kernel]->start_cycle = gpu_sim_cycle + gpu_tot_sim_cycle;
                    // record this kernel for stat print if it is the first time this kernel is selected for execution  
                    unsigned launch_uid = m_running_kernels[idx]->get_uid(); 
                    assert(std::find(m_executed_kernel_uids.begin(), m_executed_kernel_uids.end(), launch_uid) == m_executed_kernel_uids.end());
                    
                    m_executed_kernel_uids.push_back(launch_uid); 
                    m_executed_kernel_names.push_back(m_running_kernels[idx]->name()); 
                    
                    return m_running_kernels[idx];
                }
            }
        }
    }
    return NULL;
}



unsigned gpgpu_sim::finished_kernel()
{
    // This should never be called now
    assert(0);
    if( m_finished_kernel.empty() ) 
        return 0;
    unsigned result = m_finished_kernel.front();
    m_finished_kernel.pop_front();
    return result;
}

void gpgpu_sim::set_kernel_done( kernel_info_t *kernel ) 
{ 
    unsigned uid = kernel->get_uid();
    //m_finished_kernel.push_back(uid);
    printf("Calling set_kernel_done with kernel %d %s \n", kernel, kernel->entry()->get_name().c_str());
    if( kernel->isEventKernel() != 1 ) {
        gem5CudaGPU->MarkKernelParamMem(kernel->getParamMem()); // FIXME: Mark kernel parameter memory for cleanup
        gem5CudaGPU->finishKernel(uid);
    } else {
        // EDGE: TODO: Need to signal the device waiting for the response of the event
        // that the event is complete. Could do this with an interrupt, callback, or 
        // write to memory to signify that the result buffer is now valid. 
        kernel->getEvent()->completeEvent();
        gem5CudaGPU->finishEvent();
    }

    std::vector<kernel_info_t*>::iterator k;
    for( k=m_running_kernels.begin(); k!=m_running_kernels.end(); k++ ) {
        if( *k == kernel ) {
            kernel->end_cycle = gpu_sim_cycle + gpu_tot_sim_cycle;
            *k = NULL;
            if( kernel->isEventKernel() == 1 ) // EDGE: If this is an event kernel, we should free up the memory.
                delete kernel;
            break;
        }
    }
    assert( k != m_running_kernels.end() ); 
}

void set_ptx_warp_size(const struct core_config * warp_size);

gpgpu_sim::gpgpu_sim( const gpgpu_sim_config &config, CudaGPU *cuda_gpu )
    : gpgpu_t(config, cuda_gpu), m_config(config)
{ 
    m_shader_config = &m_config.m_shader_config;
    m_memory_config = &m_config.m_memory_config;
    set_ptx_warp_size(m_shader_config);
    ptx_file_line_stats_create_exposed_latency_tracker(m_config.num_shader());

#ifdef GPGPUSIM_POWER_MODEL
        m_gpgpusim_wrapper = new gpgpu_sim_wrapper(config.g_power_simulation_enabled,config.g_power_config_name);
#endif

    m_shader_stats = new shader_core_stats(m_shader_config);
    m_memory_stats = new memory_stats_t(m_config.num_shader(),m_shader_config,m_memory_config);
    average_pipeline_duty_cycle = (float *)malloc(sizeof(float));
    active_sms=(float *)malloc(sizeof(float));
    m_power_stats = new power_stat_t(m_shader_config,average_pipeline_duty_cycle,active_sms,m_shader_stats,m_memory_config,m_memory_stats);

    gpu_sim_insn = 0;
    gpu_tot_sim_insn = 0;
    gpu_tot_issued_cta = 0;
    gpu_deadlock = false;

    // EDGE
    _isEDGEInit = false;
    _intWarpDelay = 0;
    _edgeNumTimerEventsLaunched = 0;
    _edgeHasFreeIntWarpPerCycle = 0;

    m_cluster = new simt_core_cluster*[m_shader_config->n_simt_clusters];
    for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++) 
        m_cluster[i] = new simt_core_cluster(this,i,m_shader_config,m_memory_config,m_shader_stats,m_memory_stats);

    m_memory_partition_unit = new memory_partition_unit*[m_memory_config->m_n_mem];
    m_memory_sub_partition = new memory_sub_partition*[m_memory_config->m_n_mem_sub_partition];
    for (unsigned i=0;i<m_memory_config->m_n_mem;i++) {
        m_memory_partition_unit[i] = new memory_partition_unit(i, m_memory_config, m_memory_stats);
        for (unsigned p = 0; p < m_memory_config->m_n_sub_partition_per_memory_channel; p++) {
            unsigned submpid = i * m_memory_config->m_n_sub_partition_per_memory_channel + p; 
            m_memory_sub_partition[submpid] = m_memory_partition_unit[i]->get_sub_partition(p); 
        }
    }

    icnt_wrapper_init();
    icnt_create(m_shader_config->n_simt_clusters,m_memory_config->m_n_mem_sub_partition);

    time_vector_create(NUM_MEM_REQ_STAT);
    fprintf(stdout, "GPGPU-Sim uArch: performance model initialization complete.\n");

    m_running_kernels.resize( config.max_concurrent_kernel, NULL );
    m_last_issued_kernel = 0;
    m_last_cluster_issue = 0;
    *average_pipeline_duty_cycle=0;
    *active_sms=0;

    last_liveness_message_time = 0;
}

int gpgpu_sim::shared_mem_size() const
{
   return m_shader_config->gpgpu_shmem_size;
}

int gpgpu_sim::num_registers_per_core() const
{
   return m_shader_config->gpgpu_shader_registers;
}

int gpgpu_sim::wrp_size() const
{
   return m_shader_config->warp_size;
}

int gpgpu_sim::shader_clock() const
{
   return m_config.core_freq/1000;
}

void gpgpu_sim::set_prop( cudaDeviceProp *prop )
{
   m_cuda_properties = prop;
}

const struct cudaDeviceProp *gpgpu_sim::get_prop() const
{
   return m_cuda_properties;
}

enum divergence_support_t gpgpu_sim::simd_model() const
{
   return m_shader_config->model;
}

void gpgpu_sim_config::init_clock_domains(void ) 
{
   sscanf(gpgpu_clock_domains,"%lf:%lf:%lf:%lf", 
          &core_freq, &icnt_freq, &l2_freq, &dram_freq);
   core_freq = core_freq MhZ;
   icnt_freq = icnt_freq MhZ;
   l2_freq = l2_freq MhZ;
   dram_freq = dram_freq MhZ;        
   core_period = 1/core_freq;
   icnt_period = 1/icnt_freq;
   dram_period = 1/dram_freq;
   l2_period = 1/l2_freq;
   printf("GPGPU-Sim uArch: clock freqs: %lf:%lf:%lf:%lf\n",core_freq,icnt_freq,l2_freq,dram_freq);
   printf("GPGPU-Sim uArch: clock periods: %.20lf:%.20lf:%.20lf:%.20lf\n",core_period,icnt_period,l2_period,dram_period);
}

void gpgpu_sim::reinit_clock_domains(void)
{
   core_time = 0;
   dram_time = 0;
   icnt_time = 0;
   l2_time = 0;
}

bool gpgpu_sim::active()
{
    if (m_config.gpu_max_cycle_opt && (gpu_tot_sim_cycle + gpu_sim_cycle) >= m_config.gpu_max_cycle_opt) 
       return false;
        if (m_config.gpu_max_insn_opt && (gpu_tot_sim_insn + gpu_sim_insn) >= m_config.gpu_max_insn_opt)
           return false;
        if (m_config.gpu_max_cta_opt && (gpu_tot_issued_cta >= m_config.gpu_max_cta_opt) )
            return false;
        if (m_config.gpu_deadlock_detect && gpu_deadlock) {
           deadlock_check();
           return false;
        }
        for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++) 
           if( m_cluster[i]->get_not_completed()>0 ) 
               return true;
        for (unsigned i=0;i<m_memory_config->m_n_mem;i++) 
           if( m_memory_partition_unit[i]->busy()>0 )
               return true;
        if( icnt_busy() )
            return true;
        if( get_more_cta_left() )
            return true;
        
        if( _edgeManager->pendingEvents() )
            return true;

        if( !_timerEventMap.empty() && m_shader_config->_edgeTimerEventOnly )
            return true;

        // EDGE
        // The CPU can request to sleep for N GPU cycles. 
        if( gem5CudaGPU->cpuThreadSleeping() )
            return true;
        
        for( unsigned i=0; i<m_shader_config->n_simt_clusters; ++i ) {
            if (get_shader(i)->kernelCompletePending()) {
                //printf("MARIA DEBUG active returns tru pending complete kernels \n");
                return true;
            }
        }
    

    /*
    for( unsigned i=0; i<m_shader_config->n_simt_clusters; ++i ) {
        if( m_cluster[i]->isIWarpRunning() )
            return true;
    }
    */

    return false;
}

void gpgpu_sim::init()
{
    // run a CUDA grid on the GPU microarchitecture simulator
    gpu_sim_cycle = 0;
    gpu_sim_insn = 0;
    last_gpu_sim_insn = 0;
    m_total_cta_launched=0;

    reinit_clock_domains();
    set_param_gpgpu_num_shaders(m_config.num_shader());
    for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++) 
       m_cluster[i]->reinit();
    m_shader_stats->new_grid();
    // initialize the control-flow, memory access, memory latency logger
    if (m_config.g_visualizer_enabled) {
        create_thread_CFlogger( m_config.num_shader(), m_shader_config->n_thread_per_shader, 0, m_config.gpgpu_cflog_interval );
    }
    shader_CTA_count_create( m_config.num_shader(), m_config.gpgpu_cflog_interval);
    if (m_config.gpgpu_cflog_interval != 0) {
       insn_warp_occ_create( m_config.num_shader(), m_shader_config->warp_size );
       shader_warp_occ_create( m_config.num_shader(), m_shader_config->warp_size, m_config.gpgpu_cflog_interval);
       shader_mem_acc_create( m_config.num_shader(), m_memory_config->m_n_mem, 4, m_config.gpgpu_cflog_interval);
       shader_mem_lat_create( m_config.num_shader(), m_config.gpgpu_cflog_interval);
       shader_cache_access_create( m_config.num_shader(), 3, m_config.gpgpu_cflog_interval);
       set_spill_interval (m_config.gpgpu_cflog_interval * 40);
    }

    if (g_network_mode)
       icnt_init();

    // McPAT initialization function. Called on first launch of GPU
#ifdef GPGPUSIM_POWER_MODEL
    if(m_config.g_power_simulation_enabled){
        init_mcpat(m_config, m_gpgpusim_wrapper, m_config.gpu_stat_sample_freq,  gpu_tot_sim_insn, gpu_sim_insn);
    }
#endif
}

void gpgpu_sim::update_stats() {
    m_memory_stats->memlatstat_lat_pw();
    gpu_tot_sim_cycle += gpu_sim_cycle;
    gpu_tot_sim_insn += gpu_sim_insn;
}

void gpgpu_sim::print_stats()
{
    ptx_file_line_stats_write_file();
    gpu_print_stat();

    if (g_network_mode) {
        printf("----------------------------Interconnect-DETAILS--------------------------------\n" );
        icnt_display_stats();
        icnt_display_overall_stats();
        printf("----------------------------END-of-Interconnect-DETAILS-------------------------\n" );
    }
}

void gpgpu_sim::deadlock_check()
{
   if (m_config.gpu_deadlock_detect && gpu_deadlock) {
      fflush(stdout);
      printf("\n\nGPGPU-Sim uArch: ERROR ** deadlock detected: last writeback core %u @ gpu_sim_cycle %u (+ gpu_tot_sim_cycle %u) (%u cycles ago)\n", 
             gpu_sim_insn_last_update_sid,
             (unsigned) gpu_sim_insn_last_update, (unsigned) (gpu_tot_sim_cycle-gpu_sim_cycle),
             (unsigned) (gpu_sim_cycle - gpu_sim_insn_last_update )); 
      unsigned num_cores=0;
      for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++) {
         unsigned not_completed = m_cluster[i]->get_not_completed();
         if( not_completed ) {
             if ( !num_cores )  {
                 printf("GPGPU-Sim uArch: DEADLOCK  shader cores no longer committing instructions [core(# threads)]:\n" );
                 printf("GPGPU-Sim uArch: DEADLOCK  ");
                 m_cluster[i]->print_not_completed(stdout);
             } else if (num_cores < 8 ) {
                 m_cluster[i]->print_not_completed(stdout);
             } else if (num_cores >= 8 ) {
                 printf(" + others ... ");
             }
             num_cores+=m_shader_config->n_simt_cores_per_cluster;
         }
      }
      printf("\n");
      for (unsigned i=0;i<m_memory_config->m_n_mem;i++) {
         bool busy = m_memory_partition_unit[i]->busy();
         if( busy ) 
             printf("GPGPU-Sim uArch DEADLOCK:  memory partition %u busy\n", i );
      }
      if( icnt_busy() ) {
         printf("GPGPU-Sim uArch DEADLOCK:  iterconnect contains traffic\n");
         icnt_display_state( stdout );
      }
      printf("\nRe-run the simulator in gdb and use debug routines in .gdbinit to debug this\n");
      fflush(stdout);
      abort();
   }
}

/// printing the names and uids of a set of executed kernels (usually there is only one)
std::string gpgpu_sim::executed_kernel_info_string() 
{
   std::stringstream statout; 

   statout << "kernel_name = "; 
   for (unsigned int k = 0; k < m_executed_kernel_names.size(); k++) {
      statout << m_executed_kernel_names[k] << " "; 
   }
   statout << std::endl; 
   statout << "kernel_launch_uid = ";
   for (unsigned int k = 0; k < m_executed_kernel_uids.size(); k++) {
      statout << m_executed_kernel_uids[k] << " "; 
   }
   statout << std::endl; 

   return statout.str(); 
}
void gpgpu_sim::set_cache_config(std::string kernel_name,  FuncCache cacheConfig )
{
	m_special_cache_config[kernel_name]=cacheConfig ;
}

FuncCache gpgpu_sim::get_cache_config(std::string kernel_name)
{
	for (	std::map<std::string, FuncCache>::iterator iter = m_special_cache_config.begin(); iter != m_special_cache_config.end(); iter++){
		    std::string kernel= iter->first;
			if (kernel_name.compare(kernel) == 0){
				return iter->second;
			}
	}
	return (FuncCache)0;
}

bool gpgpu_sim::has_special_cache_config(std::string kernel_name)
{
	for (	std::map<std::string, FuncCache>::iterator iter = m_special_cache_config.begin(); iter != m_special_cache_config.end(); iter++){
	    	std::string kernel= iter->first;
			if (kernel_name.compare(kernel) == 0){
				return true;
			}
	}
	return false;
}


void gpgpu_sim::set_cache_config(std::string kernel_name)
{
	if(has_special_cache_config(kernel_name)){
		change_cache_config(get_cache_config(kernel_name));
	}else{
		change_cache_config(FuncCachePreferNone);
	}
}


void gpgpu_sim::change_cache_config(FuncCache cache_config)
{
	if(cache_config != m_shader_config->m_L1D_config.get_cache_status()){
		printf("FLUSH L1 Cache at configuration change between kernels\n");
		for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++) {
			m_cluster[i]->cache_flush();
	    }
	}

	switch(cache_config){
	case FuncCachePreferNone:
		m_shader_config->m_L1D_config.init(m_shader_config->m_L1D_config.m_config_string, FuncCachePreferNone);
		m_shader_config->gpgpu_shmem_size=m_shader_config->gpgpu_shmem_sizeDefault;
		break;
	case FuncCachePreferL1:
		if((m_shader_config->m_L1D_config.m_config_stringPrefL1 == NULL) || (m_shader_config->gpgpu_shmem_sizePrefL1 == (unsigned)-1))
		{
			printf("WARNING: missing Preferred L1 configuration\n");
			m_shader_config->m_L1D_config.init(m_shader_config->m_L1D_config.m_config_string, FuncCachePreferNone);
			m_shader_config->gpgpu_shmem_size=m_shader_config->gpgpu_shmem_sizeDefault;

		}else{
			m_shader_config->m_L1D_config.init(m_shader_config->m_L1D_config.m_config_stringPrefL1, FuncCachePreferL1);
			m_shader_config->gpgpu_shmem_size=m_shader_config->gpgpu_shmem_sizePrefL1;
		}
		break;
	case FuncCachePreferShared:
		if((m_shader_config->m_L1D_config.m_config_stringPrefShared == NULL) || (m_shader_config->gpgpu_shmem_sizePrefShared == (unsigned)-1))
		{
			printf("WARNING: missing Preferred L1 configuration\n");
			m_shader_config->m_L1D_config.init(m_shader_config->m_L1D_config.m_config_string, FuncCachePreferNone);
			m_shader_config->gpgpu_shmem_size=m_shader_config->gpgpu_shmem_sizeDefault;
		}else{
			m_shader_config->m_L1D_config.init(m_shader_config->m_L1D_config.m_config_stringPrefShared, FuncCachePreferShared);
			m_shader_config->gpgpu_shmem_size=m_shader_config->gpgpu_shmem_sizePrefShared;
		}
		break;
	default:
		break;
	}
}

void gpgpu_sim::UpdateAvgCTATime(unsigned cta_time) {
    gpu_tot_completed_cta++;
    gpu_avg_cta_time = ( gpu_avg_cta_time * (gpu_tot_completed_cta-1) + cta_time ) / gpu_tot_completed_cta;
    if (gpu_tot_completed_cta>1) {
        //printf("MARIA DEBUG STATS: gpu_cta_completion_delay = %lld\n", gpu_sim_cycle - gpu_cta_time.back());
    } else {
        //printf("MARIA DEBUG STATS: gpu_cta_completion_delay = %lld\n", cta_time);
    }

    gpu_cta_time.push_back(gpu_sim_cycle);
}

void gpgpu_sim::clear_executed_kernel_info()
{
   m_executed_kernel_names.clear();
   m_executed_kernel_uids.clear();
}

void gpgpu_sim::gpu_print_stat() 
{  
   FILE *statfout = stdout; 

   std::string kernel_info_str = executed_kernel_info_string(); 
   fprintf(statfout, "%s", kernel_info_str.c_str()); 

   unsigned long long gpu_avg_event_completion_rate = 0;
   unsigned long long prev_complete_cycle = _completedKernelStats[0].second._edgeCompleted;
   std::string memcName("ipv4_fwd_kernel");
   for( gpgpu_sim::KernelStatVector::iterator it = _completedKernelStats.begin(); 
           it != _completedKernelStats.end(); ++it ) {
      if (memcName.compare(it->first) || it==_completedKernelStats.begin()) {
        continue;
      }
      gpu_avg_event_completion_rate += it->second._edgeCompleted - prev_complete_cycle;
      prev_complete_cycle = it->second._edgeCompleted;
   }
   gpu_avg_event_completion_rate = gpu_avg_event_completion_rate / _completedKernelStats.size();

   //printf("MARIA DEBUG gpu_avg_events_running = %d nEventRunning=%d _completedKernelStats.size() = %d \n", 
   //       gpu_avg_events_running, nEventRunning, _completedKernelStats.size());
   //gpu_avg_events_running = ( gpu_avg_events_running * (_completedKernelStats.size()-3) + nEventRunning ) / (_completedKernelStats.size()-2);

   printf("gpu_sim_cycle = %lld\n", gpu_sim_cycle);
   printf("gpu_sim_insn = %lld\n", gpu_sim_insn);
   printf("gpu_ipc = %12.4f\n", (float)gpu_sim_insn / gpu_sim_cycle);
   printf("gpu_tot_sim_cycle = %lld\n", gpu_tot_sim_cycle+gpu_sim_cycle);
   printf("gpu_tot_sim_insn = %lld\n", gpu_tot_sim_insn+gpu_sim_insn);
   printf("gpu_tot_ipc = %12.4f\n", (float)(gpu_tot_sim_insn+gpu_sim_insn) / (gpu_tot_sim_cycle+gpu_sim_cycle));
   printf("gpu_tot_issued_cta = %lld\n", gpu_tot_issued_cta);
   //printf("gpu_avg_events_running = %lld\n", gpu_avg_events_running);
   printf("gpu_avg_cta_time = %lld\n", gpu_avg_cta_time);
   printf("gpu_avg_event_completion_rate = %lld\n", gpu_avg_event_completion_rate);
   printf("gpu_idle_cycles_per_sm = ");
   for( unsigned i=0; i< m_shader_config->n_simt_clusters; ++i) {
      printf("%lld ", get_shader(i)->getIdleCyclesNum());
   }
   printf("\n");


   // performance counter for stalls due to congestion.
   printf("gpu_stall_dramfull = %d\n", gpu_stall_dramfull);
   printf("gpu_stall_icnt2sh    = %d\n", gpu_stall_icnt2sh );

   time_t curr_time;
   time(&curr_time);
   unsigned long long elapsed_time = MAX( curr_time - g_simulation_starttime, 1 );
   printf( "gpu_total_sim_rate=%u\n", (unsigned)( ( gpu_tot_sim_insn + gpu_sim_insn ) / elapsed_time ) );

   //shader_print_l1_miss_stat( stdout );
   shader_print_cache_stats(stdout);

   cache_stats core_cache_stats;
   core_cache_stats.clear();
   for(unsigned i=0; i<m_config.num_cluster(); i++){
       m_cluster[i]->get_cache_stats(core_cache_stats);
   }
   printf("\nTotal_core_cache_stats:\n");
   core_cache_stats.print_stats(stdout, "Total_core_cache_stats_breakdown");
   shader_print_scheduler_stat( stdout, false );

   m_shader_stats->print(stdout);
#ifdef GPGPUSIM_POWER_MODEL
   if(m_config.g_power_simulation_enabled){
	   m_gpgpusim_wrapper->print_power_kernel_stats(gpu_sim_cycle, gpu_tot_sim_cycle, gpu_tot_sim_insn + gpu_sim_insn, kernel_info_str, true );
	   mcpat_reset_perf_count(m_gpgpusim_wrapper);
   }
#endif

   // performance counter that are not local to one shader
   m_memory_stats->memlatstat_print(m_memory_config->m_n_mem,m_memory_config->nbk);
   for (unsigned i=0;i<m_memory_config->m_n_mem;i++)
      m_memory_partition_unit[i]->print(stdout);

   // L2 cache stats
   if(!m_memory_config->m_L2_config.disabled()){
       cache_stats l2_stats;
       struct cache_sub_stats l2_css;
       struct cache_sub_stats total_l2_css;
       l2_stats.clear();
       l2_css.clear();
       total_l2_css.clear();

       printf("\n========= L2 cache stats =========\n");
       for (unsigned i=0;i<m_memory_config->m_n_mem_sub_partition;i++){
           m_memory_sub_partition[i]->accumulate_L2cache_stats(l2_stats);
           m_memory_sub_partition[i]->get_L2cache_sub_stats(l2_css);

           fprintf( stdout, "L2_cache_bank[%d]: Access = %u, Miss = %u, Miss_rate = %.3lf, Pending_hits = %u, Reservation_fails = %u\n",
                    i, l2_css.accesses, l2_css.misses, (double)l2_css.misses / (double)l2_css.accesses, l2_css.pending_hits, l2_css.res_fails);

           total_l2_css += l2_css;
       }
       if (!m_memory_config->m_L2_config.disabled() && m_memory_config->m_L2_config.get_num_lines()) {
          //L2c_print_cache_stat();
          printf("L2_total_cache_accesses = %u\n", total_l2_css.accesses);
          printf("L2_total_cache_misses = %u\n", total_l2_css.misses);
          if(total_l2_css.accesses > 0)
              printf("L2_total_cache_miss_rate = %.4lf\n", (double)total_l2_css.misses/(double)total_l2_css.accesses);
          printf("L2_total_cache_pending_hits = %u\n", total_l2_css.pending_hits);
          printf("L2_total_cache_reservation_fails = %u\n", total_l2_css.res_fails);
          printf("L2_total_cache_breakdown:\n");
          l2_stats.print_stats(stdout, "L2_cache_stats_breakdown");
          total_l2_css.print_port_stats(stdout, "L2_cache");
       }
   }

   if (m_config.gpgpu_cflog_interval != 0) {
      spill_log_to_file (stdout, 1, gpu_sim_cycle);
      insn_warp_occ_print(stdout);
   }
   if ( gpgpu_ptx_instruction_classification ) {
      StatDisp( g_inst_classification_stat[g_ptx_kernel_count]);
      StatDisp( g_inst_op_classification_stat[g_ptx_kernel_count]);
   }

#ifdef GPGPUSIM_POWER_MODEL
   if(m_config.g_power_simulation_enabled){
       m_gpgpusim_wrapper->detect_print_steady_state(1,gpu_tot_sim_insn+gpu_sim_insn);
   }
#endif


   // Interconnect power stat print
   long total_simt_to_mem=0;
   long total_mem_to_simt=0;
   long temp_stm=0;
   long temp_mts = 0;
   for(unsigned i=0; i<m_config.num_cluster(); i++){
	   m_cluster[i]->get_icnt_stats(temp_stm, temp_mts);
	   total_simt_to_mem += temp_stm;
	   total_mem_to_simt += temp_mts;
   }
   printf("\nicnt_total_pkts_mem_to_simt=%ld\n", total_mem_to_simt);
   printf("icnt_total_pkts_simt_to_mem=%ld\n", total_simt_to_mem);

   time_vector_print();
   fflush(stdout);

   // EDGE: Removing this... Other kernels may be mid run. Don't just erase everything. 
   //clear_executed_kernel_info();

   unsigned isrCount = 0;
   KernelStats totalIsrStats;
   double totalIPC = 0.0;
   double totalAvgInstStallCycles = 0.0;


    unsigned long long totalInsn = 0;
    unsigned long long totalCycles = 0;

    unsigned totalWithInstStalls = 0;

   for( gpgpu_sim::KernelStatVector::iterator it = _completedKernelStats.begin(); 
           it != _completedKernelStats.end(); ++it ) {    
        printf("=====================\n");
        printf("Kernel_name = %s\n", it->first.c_str());
        it->second.print(stdout);
        printf("=====================\n");

        // Treat ISR and Regular kernels differently for stats
        if( it->first.find("ISR") != std::string::npos ) {
            // ISR kernel
            isrCount++;

            if( !m_shader_config->_edgeWarmupInt || isrCount > m_config.num_cluster() ) {
                totalIsrStats += it->second;
                totalIPC += (double)it->second._nInsn / (double)it->second._nCycles;
                if( it->second._nWarpInstStalls > 0 ) {
                    totalWithInstStalls++;
                    totalAvgInstStallCycles += (double)it->second._nInstStallCycles / (double)it->second._nWarpInstStalls;
                }
            }


        } else {
            // Not an ISR kernel
            totalInsn += it->second._nInsn;
            totalCycles += it->second._nCycles;
        }
   }

    if( m_shader_config->_edgeWarmupInt )
        isrCount -= m_config.num_cluster();

    printf("=====================\n");
    printf("Average ISR stats (%d)\n", isrCount); 
    printf("avg_isr_n_insn = %.4lf\n", (double)totalIsrStats._nInsn / (double)isrCount);
    printf("avg_isr_n_cycles = %.4lf\n", (double)totalIsrStats._nCycles / (double)isrCount);
    printf("avg_isr_IPC = %.4lf\n", totalIPC / (double)isrCount);
    printf("total_isr_with_inst_stalls = %u\n", totalWithInstStalls);
    printf("total_isr_without_stalls = %u\n", isrCount - totalWithInstStalls);
    printf("total_isr_avg_n_warp_inst_stalls = %.4lf\n", (double)totalIsrStats._nWarpInstStalls / (double)totalWithInstStalls);
    printf("total_isr_avg_inst_stall_cycles_per_warp = %.4lf\n", totalAvgInstStallCycles / (double)totalWithInstStalls );

    printf("=====================\n");

    printf("Average Kernel stats (%d)\n", _completedKernelStats.size() - isrCount);
    printf("total_kernel_n_insn = %d\n", totalInsn);
    printf("total_kernel_n_cycles = %d\n", totalCycles);
    printf("total_kernel_IPC = %.4lf\n", (double)totalInsn / (double)totalCycles);
    printf("=====================\n");

    for( unsigned i=0; i<m_config.num_cluster(); ++i ) {
        printf("core_%d_n_ints = %lld\n", i, get_shader(i)->getNumInts());
    }

    if( m_shader_config->_isMemcConv ) {
        unsigned totalConvKernels = 0;
        unsigned totalMemcKernels = 0;
        KernelStats totalConvStats;
        KernelStats totalMemcStats;
        std::string convName("_Z20filterActs_YxX_colorILi4ELi32ELi1ELi4ELi1ELb0ELb1EEvPfS0_S0_iiiiiiiiiiffi");
        std::string memcName("ipv4_fwd_kernel");//

       for( gpgpu_sim::KernelStatVector::iterator it = _completedKernelStats.begin(); 
               it != _completedKernelStats.end(); ++it ) {    

            // If a Convolution kernel           
            if( !convName.compare(it->first) ) {
                totalConvKernels++;
                totalConvStats += it->second;
            }

            // If a Memcached kernel
            if( !memcName.compare(it->first) ) {
                totalMemcKernels++;
                totalMemcStats += it->second;
            }
        }

        printf("=====================\n");
        printf("num_conv_kernels = %u\n", totalConvKernels);
        totalConvStats.print(stdout, convName);
        printf("=====================\n");
        printf("num_memc_kernels = %u\n", totalMemcKernels);
        totalMemcStats.print(stdout, memcName, totalMemcKernels);
        printf("=====================\n");
    }
    unsigned totalMemcAtomics = 0;
    for( unsigned i=0; i< m_shader_config->_edgeEventReserveSm; ++i) {
      totalMemcAtomics += get_shader(i)->get_total_n_atomic();       
    };
    printf("num_memc_atomics = %u\n", totalMemcAtomics);
    printf("=====================\n");


    unsigned long long totalInts = 0;
    unsigned long long totalFreeInts = 0;
    unsigned long long totalVictimInts = 0;
    unsigned long long totalNoFlushVictimInts = 0;
    unsigned long long totalExitInts = 0;

    unsigned long long totalIntWarpSchedStalls = 0;
    unsigned long long minIntWarpSchedStalls = (unsigned long long)-1;
    unsigned long long maxIntWarpSchedStalls = 0;

    EdgeIntStallStats edgeIntStallStats;

    unsigned long long totalWarpOccupancyPerCycle = 0;
    unsigned long long totalRegisterUtilization = 0;

    unsigned long long totalLoadReplays = 0;
    unsigned long long totalBadLoadReplays = 0;

    unsigned long long totalEdgeBarriers = 0;
    unsigned long long totalEdgeReleaseBarriers = 0;

    unsigned long long totalIntSchedCycles = 0;
    unsigned long long totalIntRunCycles = 0;

    unsigned long long totalBarriersSkipped = 0;
    unsigned long long totalBarriersRestored = 0;

    for( unsigned i=0; i<m_config.num_cluster(); ++i ) { 
        printf("core_%d: \n", i);
        printf("\tnum_ints = %lld\n", get_shader(i)->getNumInts());
        printf("\tnum_free_warp_ints = %lld\n", get_shader(i)->getNumIntFreeWarps());
        printf("\tnum_victim_warp_ints = %lld\n", get_shader(i)->getNumIntVictimWarps());
        totalInts += get_shader(i)->getNumInts();
        totalFreeInts += get_shader(i)->getNumIntFreeWarps();
        totalVictimInts += get_shader(i)->getNumIntVictimWarps();
        totalNoFlushVictimInts += get_shader(i)->getNumNoFlushVictimWarps();
        totalExitInts += get_shader(i)->getNumIntExitWarps();
        totalIntWarpSchedStalls += get_shader(i)->getTotalIntSchedStalls();

        if( get_shader(i)->getMinIntSchedStalls() < minIntWarpSchedStalls )
            minIntWarpSchedStalls = get_shader(i)->getMinIntSchedStalls();

        if( get_shader(i)->getMaxIntSchedStalls() > maxIntWarpSchedStalls ) 
            maxIntWarpSchedStalls = get_shader(i)->getMaxIntSchedStalls(); 

        edgeIntStallStats += get_shader(i)->getEdgeIntStallStats();

        totalWarpOccupancyPerCycle += get_shader(i)->getWarpOccupancyPerCycle();
        totalRegisterUtilization += get_shader(i)->getRegisterUtilization();

        totalLoadReplays += get_shader(i)->getNumReplayLoads();
        totalBadLoadReplays += get_shader(i)->getNumBadReplayLoads();

        totalEdgeBarriers += get_shader(i)->getNumEdgeBarriers();
        totalEdgeReleaseBarriers += get_shader(i)->getNumEdgeReleaseBarriers();

        totalIntSchedCycles += get_shader(i)->getTotalIntSchedCycles();
        totalIntRunCycles += get_shader(i)->getTotalIntRunCycles();

        totalBarriersSkipped += get_shader(i)->getTotalBarriersSkipped();
        totalBarriersRestored += get_shader(i)->getTotalBarriersRestored();
    }

    printf("Total: \n");
    printf("\tnum_ints = %lld\n", totalInts);
    printf("\tnum_free_warp_ints = %lld\n", totalFreeInts);
    printf("\tnum_victim_warp_ints = %lld\n", totalVictimInts);
    printf("\tnum_no_flush_victim_ints = %lld\n", totalNoFlushVictimInts);
    printf("\tnum_exit_warp_ints = %lld\n", totalExitInts);
    printf("\tfraction_free_warp_ints = %.4lf\n\n", (double)totalFreeInts / (double)totalInts);

    printf("\ttotal_avg_int_warp_sched_stalls = %.4lf\n", (double)totalIntWarpSchedStalls / (double)totalVictimInts);
    printf("\ttotal_min_int_warp_sched_stalls = %lld\n", minIntWarpSchedStalls);
    printf("\ttotal_max_int_warp_sched_stalls = %lld\n", maxIntWarpSchedStalls);
   
    printf("\ttotal_avg_int_sched_cycles = %.4lf\n", (double)totalIntSchedCycles / (double)totalInts);
    printf("\ttotal_avg_int_run_cycles = %.4lf\n", (double)totalIntRunCycles / (double)totalInts);

    printf("\ttotal_load_replays = %lld\n", totalLoadReplays);
    printf("\ttotal_bad_load_replays = %lld\n", totalBadLoadReplays);
    
    printf("\ttotal_edge_barriers = %lld\n", totalEdgeBarriers);
    printf("\ttotal_edge_release_barriers = %lld\n", totalEdgeReleaseBarriers);
   
    printf("\tnum_event_kernels_launched = %u\n", _edgeNumTimerEventsLaunched);

    double occ = (double)totalWarpOccupancyPerCycle /
        (double)(m_shader_config->n_thread_per_shader*m_config.num_cluster()*gpu_sim_cycle);

    printf("\ttotal_warp_occupancy_per_cycle = %.4lf\n", occ);

    occ = (double)totalRegisterUtilization /
        (double)(m_shader_config->gpgpu_shader_registers*m_config.num_cluster()*gpu_sim_cycle);

    printf("\ttotal_register_utilization_per_cycle = %.4lf\n", occ);

    printf("\ttotal_cycles_free_int_warp_available = %.4lf\n", 
            (double)_edgeHasFreeIntWarpPerCycle / (double)gpu_sim_cycle);

    printf("\ttotal_barriers_skipped = %lld\n", totalBarriersSkipped);
    printf("\ttotal_barriers_restored = %lld\n", totalBarriersRestored);


    //edgeIntStallStats.printBreakdown(totalIntWarpSchedStalls);
    edgeIntStallStats.printBreakdown(totalVictimInts);


    for( unsigned i=0; i<m_config.num_cluster(); ++i ) {
        get_shader(i)->printStats(stdout);
    }
}


// performance counter that are not local to one shader
unsigned gpgpu_sim::threads_per_core() const 
{ 
   return m_shader_config->n_thread_per_shader; 
}

void shader_core_ctx::mem_instruction_stats(const warp_inst_t &inst)
{
    unsigned active_count = inst.active_count(); 
    //this breaks some encapsulation: the is_[space] functions, if you change those, change this.
    switch (inst.space.get_type()) {
    case undefined_space:
    case reg_space:
        break;
    case shared_space:
        m_stats->gpgpu_n_shmem_insn += active_count; 
        break;
    case const_space:
        m_stats->gpgpu_n_const_insn += active_count;
        break;
    case param_space_kernel:
    case param_space_local:
        m_stats->gpgpu_n_param_insn += active_count;
        break;
    case tex_space:
        m_stats->gpgpu_n_tex_insn += active_count;
        break;
    case global_space:
    case local_space:
        if( inst.is_store() )
            m_stats->gpgpu_n_store_insn += active_count;
        else 
            m_stats->gpgpu_n_load_insn += active_count;
        break;
    default:
        abort();
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////
// CDP code for concurrent kernels on same SM
////////////////////////////////////////////////////////////////////////////////////////////////
kernel_info_t* shader_core_ctx::get_hwtid_kernel(int tid)
{
    if( !m_threadState[tid].m_active )
        return NULL;
    else
        return &m_thread[tid]->get_kernel();
}

bool shader_core_ctx::can_issue_1block(kernel_info_t& kernel) 
{
    //Jin: concurrent kernels on one SM
    if( m_config->gpgpu_concurrent_kernel_sm ) {    
        if( m_config->max_cta(kernel) < 1 )
            return false;

        return occupy_shader_resource_1block(kernel, false);
    } else {
        return (get_n_active_cta() < m_config->max_cta(kernel));
    } 
}

// Checks for cta_size contiguous hardware thread contexts on this SM
int shader_core_ctx::find_available_hwtid(unsigned int cta_size, bool occupy) 
{
    int nThreads = m_config->n_thread_per_shader;

    // If we're running in dedicated interrupt mode, we can't use the interrupt
    // thread contexts (last _nIntTheads). 
    if( m_config->isIntDedicated() ) {
        nThreads -= m_config->_nIntThreads;
    }

    unsigned int step;
    for( step = 0; step < nThreads; step += cta_size ) {
         unsigned int hw_tid;
         for( hw_tid = step; hw_tid < step + cta_size; hw_tid++ ) {
             if( m_occupied_hwtid.test(hw_tid) ) // if any thread in this range is set, skip and move to the next range
                 break;
         }
         if( hw_tid == step + cta_size ) //consecutive non-active, so we can select this range
             break;
    }

    if( step >= nThreads ) { //didn't find
        return -1;
    } else {
        if( occupy ) { // If occupy, then actually occupy these threads
            for( unsigned hw_tid = step; hw_tid < step + cta_size; hw_tid++ )
                m_occupied_hwtid.set(hw_tid);
        }
        return step;
    }
}

// Checks for cta_size contiguous hardware thread contexts on this SM
int shader_core_ctx::edgeFindAvailableHwtid(unsigned int cta_size, bool occupy) 
{
    int nThreads = m_config->n_thread_per_shader;

    // If we're running in dedicated interrupt mode, we can't use the interrupt
    // thread contexts (last _nIntTheads). 
    if( m_config->isIntDedicated() ) {
        nThreads -= m_config->_nIntThreads;
    }

    unsigned int step;
    for( step=0; step<nThreads; ++step ) {
        if( (step+cta_size) > nThreads ) { // Exit if we don't have enough thread resources
            step = nThreads;
            break;
        }

        unsigned hwTid = 0;
        for( hwTid = step; hwTid < (step+cta_size); ++hwTid ) {
            if( m_occupied_hwtid.test(hwTid) )
                break;
        }

        if( hwTid == (step+cta_size) ) {
            break;
        } 
    }


#if 0
    unsigned int step;
    for( step = 0; step < nThreads; step += cta_size ) {
         unsigned int hw_tid;
         for( hw_tid = step; hw_tid < step + cta_size; hw_tid++ ) {
             if( m_occupied_hwtid.test(hw_tid) ) // if any thread in this range is set, skip and move to the next range
                 break;
         }
         if( hw_tid == step + cta_size ) //consecutive non-active, so we can select this range
             break;
    }
#endif

    if( step >= nThreads ) { //didn't find
        return -1;
    } else {
        if( occupy ) { // If occupy, then actually occupy these threads
            for( unsigned hw_tid = step; hw_tid < step + cta_size; hw_tid++ )
                m_occupied_hwtid.set(hw_tid);
        }
        return step;
    }
}

// EDGE FIXME: Will need to modify some of these values to include the iWarp, iCTA, and iKernel
// Checks if we can occupy a single CTA on this shader for kernel, k. If occupy, then actually occupy it
bool shader_core_ctx::occupy_shader_resource_1block(kernel_info_t& k, bool occupy, bool interrupt) 
{
    unsigned threads_per_cta  = k.threads_per_cta();
    const class function_info *kernel = k.entry();
    unsigned int padded_cta_size = threads_per_cta;
    unsigned int warp_size = m_config->warp_size;

    // Pad CTA to warp size
    if( padded_cta_size%warp_size ) 
        padded_cta_size = ((padded_cta_size/warp_size)+1)*(warp_size);

    // Can't if we're beyond the maximum number of threads for this shader
    if( (m_occupied_n_threads + padded_cta_size) > m_config->n_thread_per_shader ) {
        return false;
    }
          
    // Can't if there are not enough contiguous thread contexts for this CTA on this SM
    // EDGE: This function will not select the interrupt thread contexts
    //if( find_available_hwtid(padded_cta_size, false) == -1 )
    if( edgeFindAvailableHwtid(padded_cta_size, false) == -1 ) { 
        return false;
    }
    // Get the ptx_sim_kernel_info structure for this kernel
    const struct gpgpu_ptx_sim_kernel_info* kernel_info = ptx_sim_kernel_info(kernel);

    // Can't if the shared memory requirement is greater than the current shared memory 
    if( (m_occupied_shmem + kernel_info->smem) > m_config->gpgpu_shmem_size ) {
        return false;
      }

    // Can't if the number of registers used is greater than the total registers for this SM
    unsigned int used_regs = padded_cta_size * ((kernel_info->regs+3)&~3);
    if( (m_occupied_regs + used_regs) > m_config->gpgpu_shader_registers ) {
        return false;
      }

    // Can't if we've exceeded the maximum number of CTAs for this SM 
    // EDGE: Decrease maximum number of CTAs for interrupt CTAs, if they exist
    unsigned maxCtaPerCore = m_config->max_cta_per_core;
    if( m_config->_intMode && !interrupt && !m_config->_edgeRunISR)
        maxCtaPerCore -= m_config->_edgeEventCtasPerCore;

    if( (m_occupied_ctas + 1) > maxCtaPerCore ) {
        return false;
      }

    // If occupy set, then actually occupy the SM with a CTA from this kernel
    if( occupy ) {
        m_occupied_n_threads += padded_cta_size;
        m_occupied_shmem += kernel_info->smem;
        m_occupied_regs += (padded_cta_size * ((kernel_info->regs+3)&~3));
        m_occupied_ctas++;

        printf("GPGPU-Sim uArch: Shader %d occupied %d threads, %d shared mem, %d registers, %d ctas\n",
                m_sid, m_occupied_n_threads, m_occupied_shmem, m_occupied_regs, m_occupied_ctas);  
    }

    return true;
}

void shader_core_ctx::release_shader_resource_1block(unsigned hw_ctaid, kernel_info_t& k) {
    if( m_config->gpgpu_concurrent_kernel_sm ) {
        unsigned threads_per_cta  = k.threads_per_cta();
        const class function_info* kernel = k.entry();
        unsigned int padded_cta_size = threads_per_cta;
        unsigned int warp_size = m_config->warp_size; 
        
        if( padded_cta_size%warp_size ) 
            padded_cta_size = ((padded_cta_size/warp_size)+1)*(warp_size);
    
        if( !k.isEventKernel() || !m_config->_edgeRunSmallEventAsFastPath || k.EdgeOccupiedExtraThreads() ) {
            //printf("MARIA DEBUG %s decrementing occupied threads on shader %d by %d (%d) \n", 
            //  k.entry()->get_name().c_str(), m_sid, padded_cta_size, m_occupied_n_threads);
            assert(m_occupied_n_threads >= padded_cta_size);
            m_occupied_n_threads -= padded_cta_size;
        }

        int start_thread = m_occupied_cta_to_hwtid[hw_ctaid];

        if( !k.isEventKernel() || !m_config->_edgeRunSmallEventAsFastPath || k.EdgeOccupiedExtraThreads() ) {
            for( unsigned hwtid = start_thread; hwtid < start_thread + padded_cta_size; hwtid++ )
                m_occupied_hwtid.reset(hwtid);
        }
        
        m_occupied_cta_to_hwtid.erase(hw_ctaid);

        const struct gpgpu_ptx_sim_kernel_info *kernel_info = ptx_sim_kernel_info(kernel);
    
        if( k.isISRKernel() )
            assert( (unsigned)kernel_info->smem == 0 );

        assert(m_occupied_shmem >= (unsigned int)kernel_info->smem);
        m_occupied_shmem -= kernel_info->smem;

        
        if( !k.isEventKernel() || !m_config->_edgeRunSmallEventAsFastPath || k.EdgeOccupiedExtraRegs() ) {
            unsigned int used_regs = padded_cta_size * ((kernel_info->regs+3)&~3);
            //printf("MARIA DEBUG %s decrementing occupied regs on shader %d by %d (%d) \n", 
            //  k.entry()->get_name().c_str(), m_sid, used_regs, m_occupied_regs);
            assert(m_occupied_regs >= used_regs);
            m_occupied_regs -= used_regs;
        }

        assert(m_occupied_ctas >= 1);
        m_occupied_ctas--;
    }
}



////////////////////////////////////////////////////////////////////////////////////////////////
// EDGE: Launches an event CTA to an SM
void shader_core_ctx::issueEventBlock2Core(kernel_info_t& kernel, unsigned idx)
{
    // If we're here, we KNOW we have the rseouces
    
    // Now the running count signifies the total number of CTAs running for a kernel,
    // not the number of cores a kernel is running on.
    kernel.inc_running();

    // find a free CTA context 
    unsigned free_cta_hw_id = idx;

    assert( free_cta_hw_id!=(unsigned)-1 );

    // determine hardware threads and warps that will be used for this CTA
    int cta_size = kernel.threads_per_cta();

    // hw warp id = hw thread id mod warp size, so we need to find a range 
    // of hardware thread ids corresponding to an integral number of hardware
    // thread ids
    int padded_cta_size = cta_size; 
    if (cta_size%m_config->warp_size)
        padded_cta_size = ((cta_size/m_config->warp_size)+1)*(m_config->warp_size);

    unsigned int start_thread = free_cta_hw_id * padded_cta_size;
    unsigned int end_thread = start_thread + cta_size;

    assert( m_occupied_cta_to_hwtid[free_cta_hw_id] == start_thread );


    // reset the microarchitecture state of the selected hardware thread and warp contexts
    reinit(start_thread, end_thread,false);
     
    // initalize scalar threads and determine which hardware warps they are allocated to
    // bind functional simulation state of threads to hardware resources (simulation) 
    warp_set_t warps;
    unsigned nthreads_in_block= 0;
    for (unsigned i = start_thread; i<end_thread; i++) {
        m_threadState[i].m_cta_id = free_cta_hw_id;
        unsigned warp_id = i/m_config->warp_size;
        nthreads_in_block += ptx_sim_init_thread(kernel,
                                            &m_thread[i],
                                            m_sid,
                                            i,
                                            cta_size-(i-start_thread),
                                            m_config->n_thread_per_shader,
                                            this,
                                            free_cta_hw_id,
                                            warp_id,
                                            m_cluster->get_gpu());
        m_threadState[i].m_active = true; 
        warps.set( warp_id );
        m_warp[warp_id].kernel = &kernel;
        assert( m_warp[warp_id].isReserved() ); // Make sure this warp is reserved for us
        
    }
    assert( nthreads_in_block > 0 && nthreads_in_block <= m_config->n_thread_per_shader); // should be at least one, but less than max
    m_cta_status[free_cta_hw_id]=nthreads_in_block;

    // now that we know which warps are used in this CTA, we can allocate
    // resources for use in CTA-wide barrier operations
    m_barriers.allocate_barrier(free_cta_hw_id,warps);

    // initialize the SIMT stacks and fetch hardware
    init_warps( free_cta_hw_id, start_thread, end_thread);
   
    int startWid = start_thread / m_config->warp_size;
    int endWid = end_thread / m_config->warp_size;

    // Handle the case where threads not a multiple of warp size
    if( end_thread % m_config->warp_size )
        endWid++;

    for( unsigned i=startWid; i<endWid; ++i ) {
        _warpIntStatsVector[i].push_back( new WarpIntStats() );
        _warpIntStatsVector[i].back()->_startCycle = gpu_sim_cycle;
    }
     
    // EDGE
    _CTAKernelMap[free_cta_hw_id] = &kernel;
    
	
            if(kernel.isEventKernel() && m_config->_edgeDontLaunchEventKernel )
		{
                for( unsigned i=start_thread; i<end_thread; ++i ) {
                    m_thread[i]->set_done();
                    m_thread[i]->registerExit();
                    m_thread[i]->exitCore();

                 }
		for(unsigned j=startWid; j< endWid; ++j)
		{
                for(int i=0;i<m_config->warp_size;i++)
                {
                    m_warp[j].set_completed(i);
                 }
		}
             }

    m_n_active_cta++;

    shader_CTA_count_log(m_sid, 1);
    printf("GPGPU-Sim uArch Reserved Event: core:%3d, cta:%2u, start_tid: %4u, end_tid: %4u, initialized @(%lld,%lld)\n", 
                m_sid, free_cta_hw_id, start_thread, end_thread, gpu_sim_cycle, gpu_tot_sim_cycle );

    m_gpu->gem5CudaGPU->getCudaCore(m_sid)->record_block_issue(free_cta_hw_id);


}



/**
 * Launches a cooperative thread array (CTA). 
 *  
 * @param kernel 
 *    object that tells us which kernel to ask for a CTA from 
 */

// CDP for concurrent kernel
void shader_core_ctx::issue_block2core( kernel_info_t &kernel ) 
{
    std::string kernel_name = kernel.entry()->get_name().c_str();
    //printf("MARIA DEBUG TB scheduled on core %d for kernel %s\n", get_sid(), kernel.entry()->get_name().c_str());
    if (kernel_name.find("filterActs") != std::string::npos) {
        edge_inc_launched_background_task_tbs_num();
    }
    if (kernel.isEventKernel()) {
        edge_inc_launched_event_kernel_tbs_num();
    }
    //printf("MARIA DEBUG LaunchedConvTbsNum on core %d = %d\n", get_sid(), launched_background_task_tbs_num());
    if( !m_config->gpgpu_concurrent_kernel_sm ) {
        set_max_cta(kernel);
    } else {
        assert(occupy_shader_resource_1block(kernel, true));
    }

    // Now the running count signifies the total number of CTAs running for a kernel,
    // not the number of cores a kernel is running on.
    kernel.inc_running();

    // find a free CTA context 
    unsigned free_cta_hw_id=(unsigned)-1;

    unsigned max_cta_per_core;
    if( !m_config->gpgpu_concurrent_kernel_sm )
        max_cta_per_core = kernel_max_cta_per_shader;
    else
        max_cta_per_core = m_config->max_cta_per_core;

    // EDGE: Don't select an Interrupt CTA, if it exists
    //if( m_config->isIntDedicated() )
    if( m_config->_intMode )
        max_cta_per_core -= m_config->_edgeEventCtasPerCore;

    unsigned ctaStartIdx = 0;
    if( m_config->_edgeEventReserveCta > 0 )
        ctaStartIdx = m_config->_edgeEventReserveCta;

    for (unsigned i=ctaStartIdx; i < max_cta_per_core; i++ ) {
        if( m_cta_status[i]==0 ) {
            free_cta_hw_id=i;
            break;
        }
    }
    assert( free_cta_hw_id!=(unsigned)-1 );

    // determine hardware threads and warps that will be used for this CTA
    int cta_size = kernel.threads_per_cta();

    // hw warp id = hw thread id mod warp size, so we need to find a range 
    // of hardware thread ids corresponding to an integral number of hardware
    // thread ids
    int padded_cta_size = cta_size; 
    if (cta_size%m_config->warp_size)
        padded_cta_size = ((cta_size/m_config->warp_size)+1)*(m_config->warp_size);

    unsigned int start_thread;
    unsigned int end_thread;

    if( !m_config->gpgpu_concurrent_kernel_sm ) {
        abort(); // EDGE FIXME: This won't work in the current system with EDGE
        start_thread = free_cta_hw_id * padded_cta_size;
        end_thread  = start_thread +  cta_size;
    } else {
        start_thread = edgeFindAvailableHwtid(padded_cta_size, true);
        assert((int)start_thread != -1);
        end_thread = start_thread + cta_size;
        assert(m_occupied_cta_to_hwtid.find(free_cta_hw_id) == m_occupied_cta_to_hwtid.end());
        m_occupied_cta_to_hwtid[free_cta_hw_id] = start_thread;
    }

    // reset the microarchitecture state of the selected hardware thread and warp contexts
    reinit(start_thread, end_thread,false);
     
    // initalize scalar threads and determine which hardware warps they are allocated to
    // bind functional simulation state of threads to hardware resources (simulation) 
    warp_set_t warps;
    unsigned nthreads_in_block= 0;
    unsigned warp_id;
    for (unsigned i = start_thread; i<end_thread; i++) {
        m_threadState[i].m_cta_id = free_cta_hw_id;
        warp_id = i/m_config->warp_size;
        nthreads_in_block += ptx_sim_init_thread(kernel,
                                            &m_thread[i],
                                            m_sid,
                                            i,
                                            cta_size-(i-start_thread),
                                            m_config->n_thread_per_shader,
                                            this,
                                            free_cta_hw_id,
                                            warp_id,
                                            m_cluster->get_gpu());
        m_threadState[i].m_active = true; 
        warps.set( warp_id );
        m_warp[warp_id].kernel = &kernel;
    }
    if (kernel.isEventKernel() && kernel.hasSingleWarp()) {
        //assert(!m_config->_edgeRunSmallEventAsFastPath);
        _edgeEventWarpIds.push_back(warp_id);
        kernel.SetEdgeOccupiedExtraThreads();
        m_gpu->IncEventRunning();
        EDGE_DPRINT(EdgeDebug, "Pushing slow path warp %d to _edgeEventWarpIds on SM %d \n", warp_id, m_sid);
    }

    assert( nthreads_in_block > 0 && nthreads_in_block <= m_config->n_thread_per_shader); // should be at least one, but less than max
    m_cta_status[free_cta_hw_id]=nthreads_in_block;
    m_cta_start_time[free_cta_hw_id]=gpu_sim_cycle;

    // now that we know which warps are used in this CTA, we can allocate
    // resources for use in CTA-wide barrier operations
    m_barriers.allocate_barrier(free_cta_hw_id,warps);

    // initialize the SIMT stacks and fetch hardware
    init_warps( free_cta_hw_id, start_thread, end_thread);
   
    int startWid = start_thread / m_config->warp_size;
    int endWid = end_thread / m_config->warp_size;

    // Handle the case where threads not a multiple of warp size
    if( end_thread % m_config->warp_size )
        endWid++;

    for( unsigned i=startWid; i<endWid; ++i ) {
        _warpIntStatsVector[i].push_back( new WarpIntStats() );
        _warpIntStatsVector[i].back()->_startCycle = gpu_sim_cycle;
    }
     
    // EDGE
    _CTAKernelMap[free_cta_hw_id] = &kernel;
	// Skippng launch of event kernel if fon't launch event kernel is set
     if(kernel.isEventKernel() && m_config->_edgeDontLaunchEventKernel )
    {
        for( unsigned i=start_thread; i<end_thread; ++i ) {
          m_thread[i]->set_done();
          m_thread[i]->registerExit();
          m_thread[i]->exitCore();
          }
      for(unsigned j=startWid; j< endWid; ++j)
      {
        for(int i=0;i<m_config->warp_size;i++)
        {
          m_warp[j].set_completed(i);
        }
      }
    }
    m_n_active_cta++;

    shader_CTA_count_log(m_sid, 1);
    printf("GPGPU-Sim uArch: issue_block2core for %s KernelId=%d on core:%3d, cta:%2u, start_tid: %4u, end_tid: %4u, initialized @(%lld,%lld)\n", 
                kernel.entry()->get_name().c_str(), kernel.get_uid(), m_sid, free_cta_hw_id, start_thread, end_thread, gpu_sim_cycle, gpu_tot_sim_cycle);

    m_gpu->gem5CudaGPU->getCudaCore(m_sid)->record_block_issue(free_cta_hw_id);

}

///////////////////////////////////////////////////////////////////////////////////////////

void dram_t::dram_log( int task ) 
{
   if (task == SAMPLELOG) {
      StatAddSample(mrqq_Dist, que_length());   
   } else if (task == DUMPLOG) {
      printf ("Queue Length DRAM[%d] ",id);StatDisp(mrqq_Dist);
   }
}

bool gpgpu_sim::EventIsRunning() {
    for (unsigned i=0; i<m_shader_config->n_simt_clusters; i++) { 
        if (get_shader(i)->EventIsRunning() || get_shader(i)->PreemptionInProgress()) {
          return true;
        }
    }
    return false;
}

//If there are fast path events in the event queue, schedule them first
//Fast path event kernel is a kernel which has 32 threads + it meets the 
//resources requirements (number of RFs etc)
//First check if any SM has an available warp (32 threads). If yes, done.
//If all warps in all SMs are busy, preempt it using some modification of 
//edgeIntCycle (will take more than one cycle obviously)
bool gpgpu_sim::ScheduleFastPathEventKernels(unsigned *smIdx) {
    kernel_info_t* eventKernel = NULL;
    bool found = false;
    for(int n=0; n < m_running_kernels.size(); n++ ) {
       if( m_running_kernels[n]!=NULL && m_running_kernels[n]->isEventKernel() && 
          !m_running_kernels[n]->running() && !m_running_kernels[n]->done() && 
          !m_running_kernels[n]->no_more_ctas_to_run() && m_running_kernels[n]->hasSingleWarp() &&
          !m_running_kernels[n]->GetPreemption() && m_shader_config->_edgeRunSmallEventAsFastPath) {
              eventKernel = m_running_kernels[n];
              found = true;
              break;
       }
    }
    if (!found) { //either no event kernel or all are big and will be scheduled as defined in old EDGE
        return false;
    }
    assert(eventKernel);

    int idx = ChooseSMWithShortestPreemptionQueueAndOldestRunningEvent(GetFreeSMs(eventKernel), eventKernel); //look for SM wo event kernels + empty preemption queue
    
    if (idx<0) {
        idx = ChooseSMWithShortestPreemptionQueueAndOldestRunningEvent(GetAllSMs(eventKernel), eventKernel); //look for all SMs with no preemption in progress + lowest cost
    }

    if (idx<0) {
        //EDGE_DPRINT(EdgeDebug, "%lld: All SMs preemption engines are busy, can't assign a new event: ", gpu_sim_cycle);
        //for (unsigned i=0; i<m_shader_config->n_simt_clusters; i++) {
        //    printf( "SM%d: NoCTA=%d, IntState=%d FreeWarp: %d canIntWarp: %d ",
        //            i, get_shader(i)->_edgeCtas.empty(), get_shader(i)->getEdgeState(), 
        //            get_shader(i)->occupy_shader_resource_1block(*(eventKernel->GetEdgeSwapEventKernel()), false, false), 
        //            get_shader(i)->ChooseVictimWarp(eventKernel->GetEdgeSwapEventKernel())!=NULL );
        //}
        printf("\n");
        return false;
    }

    eventKernel->SetPreemption();
    if (eventKernel->GetEdgeSwapEventKernel() != NULL) {
        eventKernel->GetEdgeSwapEventKernel()->SetPreemption();
    }

    if (m_shader_config->_edgeUseIntCoreId) {
        idx = eventKernel->GetInterruptCoreId();
    }

    int ctaId, warpId;
    bool IsFreeWarp = get_shader(idx)->selectIntCtaWarpCtx(ctaId, warpId, eventKernel, false, false);
    EDGE_DPRINT(EdgeDebug, "%lld: Chosen SM %d for fast path event %s %p. Free warp: %d Warp Id: %d \n", 
                gpu_sim_cycle, idx, eventKernel->entry()->get_name().c_str(), eventKernel, IsFreeWarp, warpId);

    //preempt a warp (if needed) + launch using same mechanism as in old EDGE    
    get_shader(idx)->ScheduleFastPathEvent(eventKernel);
    m_last_cluster_issue = idx;
    *smIdx = idx;
    return true;
}

std::vector<unsigned> gpgpu_sim::GetFreeSMs(kernel_info_t* eventKernel) {
std::vector<unsigned> result;
for (unsigned i=0; i<m_shader_config->n_simt_clusters; i++) {
    if (get_shader(i)->CanRunEdgeEvent(eventKernel) && get_shader(i)->getEdgeState() == IDLE && !get_shader(i)->EventIsRunning() && get_shader(i)->EventQueueSize()==0) {
        result.push_back(i); 
    }
}
return result;
}

std::vector<unsigned> gpgpu_sim::GetAllSMs(kernel_info_t* eventKernel) {
std::vector<unsigned> result;
for (unsigned i=0; i<m_shader_config->n_simt_clusters; i++) {
  if (get_shader(i)->CanRunEdgeEvent(eventKernel))
    result.push_back(i); 
}
return result;
}

unsigned gpgpu_sim::ChooseSMWithFreeWarp(std::vector<unsigned> sms_set, kernel_info_t* eventKernel) {
assert(!sms_set.empty());
for (unsigned i=0; i<sms_set.size(); i++) {
    int ctaId, warpId;
    bool isFreeWarp = get_shader(sms_set[i])->selectIntCtaWarpCtx(ctaId, warpId, eventKernel, false, false);
    if (isFreeWarp) {
        //EDGE_DPRINT(EdgeDebug, "%lld: Found free warp %d in a free SM %d (wo running event kernel) for fast path event %s %p \n", 
                    //gpu_sim_cycle, warpId, sms_set[i], eventKernel->entry()->get_name().c_str(), eventKernel);
        return sms_set[i];
    }
}
return sms_set[0];
}

int gpgpu_sim::ChooseSMWithShortestPreemptionQueueAndOldestRunningEvent(std::vector<unsigned> sms_set, kernel_info_t* eventKernel) {
if (sms_set.empty()) {
    return -1;
}
unsigned min_core_schedule_cost = get_shader(sms_set[0])->EdgeCoreScheduleCost(eventKernel);
unsigned min_queue_idx = sms_set[0];
EDGE_DPRINT(EdgeDebug, "%lld: SMs costs: ", gpu_sim_cycle); 
for (unsigned i=0; i<sms_set.size(); i++) {
  unsigned core_schedule_cost = get_shader(sms_set[i])->EdgeCoreScheduleCost(eventKernel);
  printf(" %d ", core_schedule_cost); 
    if (core_schedule_cost < min_core_schedule_cost) {
        min_queue_idx = sms_set[i];
        min_core_schedule_cost = core_schedule_cost;
    }
}
printf("\n");
return min_queue_idx;
}

// int ctaId, warpId, idx;
// bool isFreeSM, isFreeWarp;
// for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++) {
//     idx = (i + m_last_cluster_issue + 1) % m_shader_config->n_simt_clusters;
//     isFreeSM = get_shader(idx)->getEdgeState() == IDLE && !get_shader(idx)->EventIsRunning();
//     isFreeWarp = get_shader(idx)->selectIntCtaWarpCtx(ctaId, warpId, eventKernel, false);
//     if (isFreeSM) {
//         EDGE_DPRINT(EdgeDebug, "%lld: Found SM %d wo running event kernel for fast path event %s %p. Warp free: %d \n", 
//           gpu_sim_cycle, idx, eventKernel->entry()->get_name().c_str(), eventKernel, isFreeWarp);
//         break;
//     }
// }
// if (~isFreeSM) { //choose SM with oldest event kernel
//     idx = ChooseSMWithShortestPreemptionQueueAndOldestRunningEvent();
//     EDGE_DPRINT(EdgeDebug, "%lld: No free warp found for fast path event %s %p, preempting on SM %d. %d in line for preemption. \n", gpu_sim_cycle, eventKernel->entry()->get_name().c_str(), eventKernel, idx, get_shader(idx)->EventQueueSize()+1);
// } else if (~isFreeWarp) { //if multiple SMs are "free" of events, choose one with free warp or one with shortest premption queue
//     int new_idx = ChooseSMWithFreeWarp();
//     if (idx == new_idx) { //need to preempt
//         idx = ChooseSM2Preempt();
//     }
// }

// unsigned gpgpu_sim::ChooseSM2Preempt() {
//     unsigned idx = (m_last_cluster_issue + 1) % m_shader_config->n_simt_clusters;
//     unsigned min_queue_len = get_shader(idx)->EventQueueSize();
//     unsigned min_queue_idx = idx;

//     if (get_shader(idx)->getEdgeState() != IDLE) {
//         for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++) {
//             idx = (i + m_last_cluster_issue + 1) % m_shader_config->n_simt_clusters;
//             if (get_shader(idx)->getEdgeState() == IDLE) {
//                 min_queue_idx = idx;
//                 break;
//             }
//             if (get_shader(idx)->EventQueueSize() < min_queue_len) {
//                 min_queue_idx = idx;
//                 min_queue_len = get_shader(idx)->EventQueueSize(); 
//             }
//         }  
//     }
//     m_last_cluster_issue=min_queue_idx;
//     return min_queue_idx;  
// }

void gpgpu_sim::issue_block2core()
{
unsigned skipSmIdx;
bool skip = ScheduleFastPathEventKernels(&skipSmIdx); 

unsigned last_issued = m_last_cluster_issue;     
for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++) {
    unsigned idx = (i + last_issued + 1) % m_shader_config->n_simt_clusters;
    if (skip && (idx==skipSmIdx)) {
      continue;
    }
    unsigned num = m_cluster[idx]->issue_block2core();
    if( num ) {
        m_last_cluster_issue=idx;
        m_total_cta_launched += num;
    }
}
}

unsigned long long g_single_step=0; // set this in gdb to single step the pipeline

void
gpgpu_sim::core_cycle_start()
{
/////////////////////////////////////////////////////
/////////////////////// EDGE ////////////////////////
/////////////////////////////////////////////////////
// Try and launch an event kernel if any are pending
LaunchEventKernel();

// Decrement the CPU sleep counter if sleeping
gem5CudaGPU->decrementThreadSleepCycles();

/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////

// L1 cache + shader core pipeline stages
m_power_stats->pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].clear();
for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++) {
   if (m_cluster[i]->get_not_completed() || get_more_cta_left()  || m_cluster[i]->intInProgress() ) {
         m_cluster[i]->core_cycle();
         *active_sms+=m_cluster[i]->get_n_active_sms();
   }
   // Update core icnt/cache stats for GPUWattch
   m_cluster[i]->get_icnt_stats(m_power_stats->pwr_mem_stat->n_simt_to_mem[CURRENT_STAT_IDX][i], m_power_stats->pwr_mem_stat->n_mem_to_simt[CURRENT_STAT_IDX][i]);
   m_cluster[i]->get_cache_stats(m_power_stats->pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX]);
}
float temp=0;
for (unsigned i=0;i<m_shader_config->num_shader();i++){
  temp+=m_shader_stats->m_pipeline_duty_cycle[i];
}
temp=temp/m_shader_config->num_shader();
*average_pipeline_duty_cycle=((*average_pipeline_duty_cycle)+temp);

if( g_single_step && ((gpu_sim_cycle+gpu_tot_sim_cycle) >= g_single_step) ) {
    asm("int $03");
}

gpu_sim_cycle++;
if( g_interactive_debugger_enabled ) 
   gpgpu_debug();


/////////////////////////////////////////////////////
/////////////////////// EDGE ////////////////////////
/////////////////////////////////////////////////////

// Increment per kernel cycle statistics
for( unsigned i=0; i< m_running_kernels.size(); ++i ) {
    if( m_running_kernels[i] != NULL && m_running_kernels[i]->hasStarted() ) {
        m_running_kernels[i]->stats()._nCycles++;
    }
}

// Debugging
if( _intWarpDelay > 0 )
    _intWarpDelay--;

// Cycle the interrupt controller
_edgeManager->cycle();

// Check if we should trigger an internal interrupt (timer or debug)
generateInternalInt(); 

// Check to see if we COULD launch an interrupt warp on ANY SM without
// requiring a context switch this cycle. 
if (m_shader_config->_edgeRunISR) {
    for( unsigned i=0; i< m_shader_config->n_simt_clusters; ++i) {
        if( get_shader(i)->isFreeIntWarpAvailable() ) {
            _edgeHasFreeIntWarpPerCycle++;
            break;
        }
    }
}

//count idle cycles per sm
for( unsigned i=0; i< m_shader_config->n_simt_clusters; ++i) {
    if(!get_shader(i)->isactive() ) {
            get_shader(i)->incIdleCyclesNum();
        }
    }
    
    /////////////////////////////////////////////////////
    /////////////////////////////////////////////////////
    /////////////////////////////////////////////////////

    // McPAT main cycle (interface with McPAT)
#ifdef GPGPUSIM_POWER_MODEL
    if(m_config.g_power_simulation_enabled){
        mcpat_cycle(m_config, getShaderCoreConfig(), m_gpgpusim_wrapper, m_power_stats, m_config.gpu_stat_sample_freq, gpu_tot_sim_cycle, gpu_sim_cycle, gpu_tot_sim_insn, gpu_sim_insn);
    }
#endif

    issue_block2core();
     
    // Depending on configuration, flush the caches once all of threads are completed.
    int all_threads_complete = 1;
    if (m_config.gpgpu_flush_l1_cache) {
       for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++) {
          if (m_cluster[i]->get_not_completed() == 0)
              m_cluster[i]->cache_flush();
          else
             all_threads_complete = 0 ;
       }
    }

    if(m_config.gpgpu_flush_l2_cache){
        if(!m_config.gpgpu_flush_l1_cache){
            for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++) {
                if (m_cluster[i]->get_not_completed() != 0){
                    all_threads_complete = 0 ;
                    break;
                }
            }
        }

       if (all_threads_complete && !m_memory_config->m_L2_config.disabled() ) {
          printf("Flushed L2 caches...\n");
          if (m_memory_config->m_L2_config.get_num_lines()) {
             int dlc = 0;
             for (unsigned i=0;i<m_memory_config->m_n_mem;i++) {
                dlc = m_memory_sub_partition[i]->flushL2();
                assert (dlc == 0); // need to model actual writes to DRAM here
                printf("Dirty lines flushed from L2 %d is %d\n", i, dlc  );
             }
          }
       }
    }

    if (!(gpu_sim_cycle % m_config.gpu_stat_sample_freq)) {
       time_t days, hrs, minutes, sec;
       time_t curr_time;
       time(&curr_time);
       unsigned long long  elapsed_time = MAX(curr_time - g_simulation_starttime, 1);
       if ( (elapsed_time - last_liveness_message_time) >= m_config.liveness_message_freq ) {
          days    = elapsed_time/(3600*24);
          hrs     = elapsed_time/3600 - 24*days;
          minutes = elapsed_time/60 - 60*(hrs + 24*days);
          sec = elapsed_time - 60*(minutes + 60*(hrs + 24*days));
          printf("GPGPU-Sim uArch: cycles simulated: %lld  inst.: %lld (ipc=%4.1f) sim_rate=%u (inst/sec) elapsed = %u:%u:%02u:%02u / %s", 
                 gpu_tot_sim_cycle + gpu_sim_cycle, gpu_tot_sim_insn + gpu_sim_insn, 
                 (double)gpu_sim_insn/(double)gpu_sim_cycle,
                 (unsigned)((gpu_tot_sim_insn+gpu_sim_insn) / elapsed_time),
                 (unsigned)days,(unsigned)hrs,(unsigned)minutes,(unsigned)sec,
                 ctime(&curr_time));
          fflush(stdout);
          last_liveness_message_time = elapsed_time; 
       }
       visualizer_printstat();
       m_memory_stats->memlatstat_lat_pw();
       if (m_config.gpgpu_runtime_stat && (m_config.gpu_runtime_stat_flag != 0) ) {
          if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_BW_STAT) {
             for (unsigned i=0;i<m_memory_config->m_n_mem;i++) 
                m_memory_partition_unit[i]->print_stat(stdout);
             printf("maxmrqlatency = %d \n", m_memory_stats->max_mrq_latency);
             printf("maxmflatency = %d \n", m_memory_stats->max_mf_latency);
          }
          if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_SHD_INFO) 
             shader_print_runtime_stat( stdout );
          if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_L1MISS) 
             shader_print_l1_miss_stat( stdout );
          if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_SCHED) 
             shader_print_scheduler_stat( stdout, false );
       }
       if (m_shader_config->_edgePrintStat) {
        printf("EDGE STAT LaunchedConvTbsNum on core 0..15: ");
        for( unsigned i=0; i< m_shader_config->n_simt_clusters; ++i) {
          printf("%d ", get_shader(i)->edge_launched_background_task_tbs_num());
        };
        printf("\n");
        printf("EDGE STAT Number of launched event kernels tbs on core 0..15: ");
        for( unsigned i=0; i< m_shader_config->n_simt_clusters; ++i) {
             printf("%d ", get_shader(i)->edge_launched_event_kernel_tbs_num());      
        };
        printf("\n");
        printf("EDGE STAT Number of atomics on core 0..15: ");
        for( unsigned i=0; i< m_shader_config->n_simt_clusters; ++i) {
          printf("%d ", get_shader(i)->get_total_n_atomic());       
        };
        printf("\n");
      }
    }

    if (!(gpu_sim_cycle % 2000000)) {
       // deadlock detection 
       if (m_config.gpu_deadlock_detect && gpu_sim_insn == last_gpu_sim_insn) {
          gpu_deadlock = true;
       } else {
          last_gpu_sim_insn = gpu_sim_insn;
       }
    }
    try_snap_shot(gpu_sim_cycle);
    spill_log_to_file (stdout, 0, gpu_sim_cycle);
}

void
gpgpu_sim::core_cycle_end()
{
    // shader core loading (pop from ICNT into core) follows CORE clock
    for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++)
        m_cluster[i]->icnt_cycle();
}

void
gpgpu_sim::icnt_cycle_start()
{
    // pop from memory controller to interconnect
    for (unsigned i=0;i<m_memory_config->m_n_mem_sub_partition;i++) {
        mem_fetch* mf = m_memory_sub_partition[i]->top();
        if (mf) {
            unsigned response_size = mf->get_is_write()?mf->get_ctrl_size():mf->size();
            if ( ::icnt_has_buffer( m_shader_config->mem2device(i), response_size ) ) {
                if (!mf->get_is_write())
                   mf->set_return_timestamp(gpu_sim_cycle+gpu_tot_sim_cycle);
                mf->set_status(IN_ICNT_TO_SHADER,gpu_sim_cycle+gpu_tot_sim_cycle);
                ::icnt_push( m_shader_config->mem2device(i), mf->get_tpc(), mf, response_size );
                m_memory_sub_partition[i]->pop();
            } else {
                gpu_stall_icnt2sh++;
            }
        } else {
           m_memory_sub_partition[i]->pop();
        }
    }
}

void
gpgpu_sim::icnt_cycle_end()
{
    icnt_transfer();
}

void
gpgpu_sim::dram_cycle()
{
    for (unsigned i=0;i<m_memory_config->m_n_mem;i++){
       m_memory_partition_unit[i]->dram_cycle(); // Issue the dram command (scheduler + delay model)
       // Update performance counters for DRAM
       m_memory_partition_unit[i]->set_dram_power_stats(m_power_stats->pwr_mem_stat->n_cmd[CURRENT_STAT_IDX][i], m_power_stats->pwr_mem_stat->n_activity[CURRENT_STAT_IDX][i],
                      m_power_stats->pwr_mem_stat->n_nop[CURRENT_STAT_IDX][i], m_power_stats->pwr_mem_stat->n_act[CURRENT_STAT_IDX][i], m_power_stats->pwr_mem_stat->n_pre[CURRENT_STAT_IDX][i],
                      m_power_stats->pwr_mem_stat->n_rd[CURRENT_STAT_IDX][i], m_power_stats->pwr_mem_stat->n_wr[CURRENT_STAT_IDX][i], m_power_stats->pwr_mem_stat->n_req[CURRENT_STAT_IDX][i]);
    }
}

void
gpgpu_sim::l2_cycle()
{
    m_power_stats->pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX].clear();
    for (unsigned i=0;i<m_memory_config->m_n_mem_sub_partition;i++) {
        //move memory request from interconnect into memory partition (if not backed up)
        //Note:This needs to be called in DRAM clock domain if there is no L2 cache in the system
        if ( m_memory_sub_partition[i]->full() ) {
            gpu_stall_dramfull++;
        } else {
            mem_fetch* mf = (mem_fetch*) icnt_pop( m_shader_config->mem2device(i) );
            // NOTE: gem5-gpu still uses this path for parameter memory access
            m_memory_sub_partition[i]->push( mf, gpu_sim_cycle + gpu_tot_sim_cycle );
        }
        m_memory_sub_partition[i]->cache_cycle(gpu_sim_cycle+gpu_tot_sim_cycle);
        m_memory_sub_partition[i]->accumulate_L2cache_stats(m_power_stats->pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX]);
    }
}

//void gpgpu_sim::cycle()
//{
//   int clock_mask = next_clock_domain();
//
//   if (clock_mask & CORE ) {
//       // shader core loading (pop from ICNT into core) follows CORE clock
//      for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++)
//         m_cluster[i]->icnt_cycle();
//   }
//    if (clock_mask & ICNT) {
//        // pop from memory controller to interconnect
//        for (unsigned i=0;i<m_memory_config->m_n_mem_sub_partition;i++) {
//            mem_fetch* mf = m_memory_sub_partition[i]->top();
//            if (mf) {
//                unsigned response_size = mf->get_is_write()?mf->get_ctrl_size():mf->size();
//                if ( ::icnt_has_buffer( m_shader_config->mem2device(i), response_size ) ) {
//                    if (!mf->get_is_write())
//                       mf->set_return_timestamp(gpu_sim_cycle+gpu_tot_sim_cycle);
//                    mf->set_status(IN_ICNT_TO_SHADER,gpu_sim_cycle+gpu_tot_sim_cycle);
//                    ::icnt_push( m_shader_config->mem2device(i), mf->get_tpc(), mf, response_size );
//                    m_memory_sub_partition[i]->pop();
//                } else {
//                    gpu_stall_icnt2sh++;
//                }
//            } else {
//               m_memory_sub_partition[i]->pop();
//            }
//        }
//    }
//
//   if (clock_mask & DRAM) {
//      for (unsigned i=0;i<m_memory_config->m_n_mem;i++){
//         m_memory_partition_unit[i]->dram_cycle(); // /Issue the dram command (scheduler + delay model)
//         // Update performance counters for DRAM
//         m_memory_partition_unit[i]->set_dram_power_stats(m_power_stats->pwr_mem_stat->n_cmd[CURRENT_STAT_IDX][i], m_power_stats->pwr_mem_stat->n_activity[CURRENT_STAT_IDX][i],
//                        m_power_stats->pwr_mem_stat->n_nop[CURRENT_STAT_IDX][i], m_power_stats->pwr_mem_stat->n_act[CURRENT_STAT_IDX][i], m_power_stats->pwr_mem_stat->n_pre[CURRENT_STAT_IDX][i],
//                        m_power_stats->pwr_mem_stat->n_rd[CURRENT_STAT_IDX][i], m_power_stats->pwr_mem_stat->n_wr[CURRENT_STAT_IDX][i], m_power_stats->pwr_mem_stat->n_req[CURRENT_STAT_IDX][i]);
//      }
//   }
//
//   // L2 operations follow L2 clock domain
//   if (clock_mask & L2) {
//       m_power_stats->pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX].clear();
//      for (unsigned i=0;i<m_memory_config->m_n_mem_sub_partition;i++) {
//          //move memory request from interconnect into memory partition (if not backed up)
//          //Note:This needs to be called in DRAM clock domain if there is no L2 cache in the system
//          if ( m_memory_sub_partition[i]->full() ) {
//             gpu_stall_dramfull++;
//          } else {
//              mem_fetch* mf = (mem_fetch*) icnt_pop( m_shader_config->mem2device(i) );
//              m_memory_sub_partition[i]->push( mf, gpu_sim_cycle + gpu_tot_sim_cycle );
//          }
//          m_memory_sub_partition[i]->cache_cycle(gpu_sim_cycle+gpu_tot_sim_cycle);
//          m_memory_sub_partition[i]->accumulate_L2cache_stats(m_power_stats->pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX]);
//       }
//   }
//
//   if (clock_mask & ICNT) {
//      icnt_transfer();
//   }
//
//   if (clock_mask & CORE) {
//      // L1 cache + shader core pipeline stages
//      m_power_stats->pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].clear();
//      for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++) {
//         if (m_cluster[i]->get_not_completed() || get_more_cta_left() ) {
//               m_cluster[i]->core_cycle();
//               *active_sms+=m_cluster[i]->get_n_active_sms();
//         }
//         // Update core icnt/cache stats for GPUWattch
//         m_cluster[i]->get_icnt_stats(m_power_stats->pwr_mem_stat->n_simt_to_mem[CURRENT_STAT_IDX][i], m_power_stats->pwr_mem_stat->n_mem_to_simt[CURRENT_STAT_IDX][i]);
//         m_cluster[i]->get_cache_stats(m_power_stats->pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX]);
//      }
//      float temp=0;
//      for (unsigned i=0;i<m_shader_config->num_shader();i++){
//        temp+=m_shader_stats->m_pipeline_duty_cycle[i];
//      }
//      temp=temp/m_shader_config->num_shader();
//      *average_pipeline_duty_cycle=((*average_pipeline_duty_cycle)+temp);
//        //cout<<"Average pipeline duty cycle: "<<*average_pipeline_duty_cycle<<endl;
//
//
//      if( g_single_step && ((gpu_sim_cycle+gpu_tot_sim_cycle) >= g_single_step) ) {
//          asm("int $03");
//      }
//      gpu_sim_cycle++;
//      if( g_interactive_debugger_enabled )
//         gpgpu_debug();
//
//      // McPAT main cycle (interface with McPAT)
//#ifdef GPGPUSIM_POWER_MODEL
//      if(m_config.g_power_simulation_enabled){
//          mcpat_cycle(m_config, getShaderCoreConfig(), m_gpgpusim_wrapper, m_power_stats, m_config.gpu_stat_sample_freq, gpu_tot_sim_cycle, gpu_sim_cycle, gpu_tot_sim_insn, gpu_sim_insn);
//      }
//#endif
//
//      issue_block2core();
//
//      // Depending on configuration, flush the caches once all of threads are completed.
//      int all_threads_complete = 1;
//      if (m_config.gpgpu_flush_l1_cache) {
//         for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++) {
//            if (m_cluster[i]->get_not_completed() == 0)
//                m_cluster[i]->cache_flush();
//            else
//               all_threads_complete = 0 ;
//         }
//      }
//
//      if(m_config.gpgpu_flush_l2_cache){
//          if(!m_config.gpgpu_flush_l1_cache){
//              for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++) {
//                  if (m_cluster[i]->get_not_completed() != 0){
//                      all_threads_complete = 0 ;
//                      break;
//                  }
//              }
//          }
//
//         if (all_threads_complete && !m_memory_config->m_L2_config.disabled() ) {
//            printf("Flushed L2 caches...\n");
//            if (m_memory_config->m_L2_config.get_num_lines()) {
//               int dlc = 0;
//               for (unsigned i=0;i<m_memory_config->m_n_mem;i++) {
//                  dlc = m_memory_sub_partition[i]->flushL2();
//                  assert (dlc == 0); // need to model actual writes to DRAM here
//                  printf("Dirty lines flushed from L2 %d is %d\n", i, dlc  );
//               }
//            }
//         }
//      }
//
//      if (!(gpu_sim_cycle % m_config.gpu_stat_sample_freq)) {
//         time_t days, hrs, minutes, sec;
//         time_t curr_time;
//         time(&curr_time);
//         unsigned long long  elapsed_time = MAX(curr_time - g_simulation_starttime, 1);
//         if ( (elapsed_time - last_liveness_message_time) >= m_config.liveness_message_freq ) {
//            days    = elapsed_time/(3600*24);
//            hrs     = elapsed_time/3600 - 24*days;
//            minutes = elapsed_time/60 - 60*(hrs + 24*days);
//            sec = elapsed_time - 60*(minutes + 60*(hrs + 24*days));
//            printf("GPGPU-Sim uArch: cycles simulated: %lld  inst.: %lld (ipc=%4.1f) sim_rate=%u (inst/sec) elapsed = %u:%u:%02u:%02u / %s",
//                   gpu_tot_sim_cycle + gpu_sim_cycle, gpu_tot_sim_insn + gpu_sim_insn,
//                   (double)gpu_sim_insn/(double)gpu_sim_cycle,
//                   (unsigned)((gpu_tot_sim_insn+gpu_sim_insn) / elapsed_time),
//                   (unsigned)days,(unsigned)hrs,(unsigned)minutes,(unsigned)sec,
//                   ctime(&curr_time));
//            fflush(stdout);
//            last_liveness_message_time = elapsed_time;
//         }
//         visualizer_printstat();
//         m_memory_stats->memlatstat_lat_pw();
//         if (m_config.gpgpu_runtime_stat && (m_config.gpu_runtime_stat_flag != 0) ) {
//            if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_BW_STAT) {
//               for (unsigned i=0;i<m_memory_config->m_n_mem;i++)
//                  m_memory_partition_unit[i]->print_stat(stdout);
//               printf("maxmrqlatency = %d \n", m_memory_stats->max_mrq_latency);
//               printf("maxmflatency = %d \n", m_memory_stats->max_mf_latency);
//            }
//            if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_SHD_INFO)
//               shader_print_runtime_stat( stdout );
//            if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_L1MISS)
//               shader_print_l1_miss_stat( stdout );
//            if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_SCHED)
//               shader_print_scheduler_stat( stdout, false );
//         }
//      }
//
//      if (!(gpu_sim_cycle % 20000)) {
//         // deadlock detection
//         if (m_config.gpu_deadlock_detect && gpu_sim_insn == last_gpu_sim_insn) {
//            gpu_deadlock = true;
//         } else {
//            last_gpu_sim_insn = gpu_sim_insn;
//         }
//      }
//      try_snap_shot(gpu_sim_cycle);
//      spill_log_to_file (stdout, 0, gpu_sim_cycle);
//   }
//}


void shader_core_ctx::dump_warp_state( FILE *fout ) const
{
   fprintf(fout, "\n");
   fprintf(fout, "per warp functional simulation status:\n");
   for (unsigned w=0; w < m_config->max_warps_per_shader; w++ ) 
       m_warp[w].print(fout);
}

void gpgpu_sim::dump_pipeline( int mask, int s, int m ) const
{
/*
   You may want to use this function while running GPGPU-Sim in gdb.
   One way to do that is add the following to your .gdbinit file:
 
      define dp
         call g_the_gpu.dump_pipeline((0x40|0x4|0x1),$arg0,0)
      end
 
   Then, typing "dp 3" will show the contents of the pipeline for shader core 3.
*/

   printf("Dumping pipeline state...\n");
   if(!mask) mask = 0xFFFFFFFF;
   for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++) {
      if(s != -1) {
         i = s;
      }
      if(mask&1) m_cluster[m_shader_config->sid_to_cluster(i)]->display_pipeline(i,stdout,1,mask & 0x2E);
      if(s != -1) {
         break;
      }
   }
   if(mask&0x10000) {
      for (unsigned i=0;i<m_memory_config->m_n_mem;i++) {
         if(m != -1) {
            i=m;
         }
         printf("DRAM / memory controller %u:\n", i);
         if(mask&0x100000) m_memory_partition_unit[i]->print_stat(stdout);
         if(mask&0x1000000)   m_memory_partition_unit[i]->visualize();
         if(mask&0x10000000)   m_memory_partition_unit[i]->print(stdout);
         if(m != -1) {
            break;
         }
      }
   }
   fflush(stdout);
}

const struct shader_core_config * gpgpu_sim::getShaderCoreConfig()
{
   return m_shader_config;
}

const struct memory_config * gpgpu_sim::getMemoryConfig()
{
   return m_memory_config;
}

simt_core_cluster * gpgpu_sim::getSIMTCluster()
{
   return *m_cluster;
}

shader_core_ctx* gpgpu_sim::get_shader(int id)
{
//    int clusters = m_config.m_shader_config.n_simt_clusters;
    int shaders_per_cluster = m_config.m_shader_config.n_simt_cores_per_cluster;
    int cluster = id/shaders_per_cluster;
    int shader_in_cluster = id%shaders_per_cluster;
    assert(shader_in_cluster < shaders_per_cluster);
    assert(cluster < m_config.m_shader_config.n_simt_clusters);

    return m_cluster[cluster]->get_core(shader_in_cluster);
}


// EDGE
void gpgpu_sim::initEDGE(KernelVector& k)
{
    assert( k.size() == m_shader_config->n_simt_clusters );

    _edgeManager = new GPUEventManager(this, m_shader_config->n_simt_clusters);

    for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++) 
       m_cluster[i]->initEDGE(k[i]);

    if( m_shader_config->_edgeWarmupInt )
        _edgeManager->warmup();

    _isEDGEInit = true;
}

//MARIA EDGE 2. coreID is basicly unused. Per GPU unit that gets the interrupt, reads everything from the EKT and adds
//the event kernel to the event kernel queue
void gpgpu_sim::setIntSignal(unsigned coreID) 
{
    assert(_isEDGEInit);
    if (get_shader(coreID)->get_config()->_edgeRunISR) {
        get_shader(coreID)->setIntSignal();
    } else if(!get_shader(coreID)->get_config()->_intMode ) {
        EDGE_DPRINT(EdgeErr, "Trying to set interrupt when interrupt mode not enabled.\n");
        abort();
    } else {
        EDGE_DPRINT(EdgeDebug, "Interrupt received on core %d on cycle %lld \n", coreID, gpu_sim_cycle + gpu_tot_sim_cycle);
        assert(! (m_shader_config->_edgeEventPriority==3 && m_shader_config->_edgeRunSmallEventAsFastPath==0 && 
                  m_shader_config->_edgeEventReserveSm==0 ) ); //draining + CTABlock can't work together
        _edgeManager->beginInt(coreID);        
        GPUEvent* event = _edgeManager->eventInProgress(coreID);
        assert(event->getType() == EDGE_USER_EVENT); //no warmup, no barrier

        if( !event->beginEvent() ) { 
            EDGE_DPRINT(EdgeErr, "Exceeded maximum # of concurrent events...\n");
            abort();
        }

        kernel_info_t *eventKernel = new kernel_info_t( event->getKernel() );     // Create a clone of the event kernel
        eventKernel->setParamMem( event->getNextParamAddr() );  // Update the parameter memory
        eventKernel->setEvent( event );                         // Backpointer to the event for when the kernel completes;
        eventKernel->SetInterruptCoreId(coreID);
        eventKernel->setEventKernel(1);
        long long int_start_cycle = get_shader(coreID)->edge_get_int_start_cycle();
        //printf("MARIA setting int start cycles of event kernel to %lld\n", int_start_cycle);
        eventKernel->edge_set_int_start_cycle(int_start_cycle);

        kernel_info_t *eventKernel2 = new kernel_info_t( event->getKernel2() );     // Create a clone of the event kernel
        if (eventKernel2 != NULL) {
          eventKernel2->setParamMem( eventKernel->getParamMem() );  // Update the parameter memory
          eventKernel2->setEvent( event ); 
          eventKernel2->setEventKernel(1);                        // Backpointer to the event for when the kernel completes;
          eventKernel2->SetInterruptCoreId(coreID);
          eventKernel2->edge_set_int_start_cycle(int_start_cycle);
          eventKernel->SetEdgeSwapEventKernel(eventKernel2);
        }

        EDGE_DPRINT(EdgeDebug, "Config event kernel KernelId=%d %s, param mem %p\n", eventKernel->get_uid(), eventKernel->name().c_str(), (void*)eventKernel->getParamMem());
        _edgeManager->completeInt(coreID);

        QueueEventKernel(eventKernel);

    }
}

void gpgpu_sim::printICache()
{
    for( unsigned i=0; i<m_config.num_cluster(); ++i ) {
        get_shader(i)->printICacheUtilization();
    }
}

// EDGE DEBUG
void gpgpu_sim::flushMemcKernels()
{
    assert( m_shader_config->_isMemcConv );
    if( !_eventKernelQueue.empty() ) {
        printf("EDGE: Convolution kernel completing, flushing %d pending Memcached kernels\n", _eventKernelQueue.size());
        // TODO: Clear up kernels... but this is at the end of simulation, so no real need. 
        _eventKernelQueue.clear();
    }

    int runningCount = 0;
    int notStartedCount = 0;
    for( unsigned n=0; n < m_running_kernels.size(); ++n ) {
        kernel_info_t* k = m_running_kernels[n];
        if( k != NULL && !k->done() && k->isEventKernel() ) {
            unsigned launch_uid = k->get_uid();   
            if( std::find(m_executed_kernel_uids.begin(), m_executed_kernel_uids.end(), launch_uid) == m_executed_kernel_uids.end() ) {
                notStartedCount++;
                delete k;
                m_running_kernels[n] = NULL;       
            } else {
                runningCount++;
            }
        }
    }

    printf("EDGE: Found %d total Memcached kernels. %d running, %d haven't started yet. Flushed %d pending kernels\n", 
            (notStartedCount+runningCount), runningCount, notStartedCount, notStartedCount);

    printf("EDGE Memcached Convolution completion cycle = %lld\n", gpu_sim_cycle);
}

void gpgpu_sim::reserveEventResources(kernel_info_t* k)
{
    if( m_shader_config->_edgeEventReserveCta > 0 ) {
        for( unsigned i=0; i<m_config.num_cluster(); ++i ) {
            get_shader(i)->edgeReserveEventResources(k, m_shader_config->_edgeEventReserveCta);
        }
    } else if ( m_shader_config->_edgeEventReserveSm > 0 ) {

    }
}

unsigned gpgpu_sim::RunningEventKernelsCount() {
    unsigned result = 0;
    for ( unsigned i=0; i<m_running_kernels.size(); ++i ) {
        if ( m_running_kernels[i]!=NULL && m_running_kernels[i]->isEventKernel() ) {
            result++;
        }
    }
    return result;
}

bool gpgpu_sim::isNonEventKernelRunning()
{
    if( m_shader_config->_edgeTimerEventOnly ) 
        return true;
    
    for( unsigned i=0; i<m_running_kernels.size(); ++i ) {
        kernel_info_t* k = m_running_kernels[i];
        if( k != NULL && !k->isEventKernel() && !k->done() )
            return true;
    }
    return false;
}

void gpgpu_sim::SwapRunningKernel(kernel_info_t* orig, kernel_info_t* new_one) {
    unsigned n;
    for(n=0; n < m_running_kernels.size(); n++ ) {
        if( m_running_kernels[n] == orig) {
            m_running_kernels[n] = new_one;
            break;
        }
    }
    assert(n < m_running_kernels.size());
}

void gpgpu_sim::generateInternalIntRequestsPatternByDelayAndLimit() {
    for( TimerEventMap::iterator it = _timerEventMap.begin(); it != _timerEventMap.end(); ++it ) {
        unsigned eventId = it->first;
        unsigned long long eventPeriod = it->second;
        unsigned start_cycle = m_shader_config->_edgeEventStartCycle;
        bool schedule_new_batch = ( ( gpu_sim_cycle > start_cycle && gpu_sim_cycle < eventPeriod && !start_cycle_used ) || 
                                    ((gpu_sim_cycle - _edgePrevBatchEndCycle) >= eventPeriod) && (gpu_sim_cycle > 0) && !_edgeBatchIsRunning &&
                                    (_edgeNumTimerEventsLaunched < m_shader_config->_edgeMaxTimerEvents) && isNonEventKernelRunning() );
        if (schedule_new_batch) {
            start_cycle_used = true;
            _edgeScheduleNewBatch = true;
            _edgeBatchIsRunning = true;
        }
    }
    
    for( TimerEventMap::iterator it = _timerBatchEventMap.begin(); it != _timerBatchEventMap.end(); ++it ) {
        unsigned eventId = it->first;
        unsigned long long eventPeriod = it->second;
        bool schedule_new_event = ( !(gpu_sim_cycle % eventPeriod) && (gpu_sim_cycle > 0) && 
                                    (_edgeNumTimerEventsLaunchedBatch < _edgeTimerBatchSize) && isNonEventKernelRunning() && 
                                    (m_shader_config->_edgeLimitConcurrentEvents==0 || RunningEventKernelsCount() < m_shader_config->_edgeLimitConcurrentEvents) );
        //printf("generateInternalIntRequestsPatternByDelayAndLimit called cycle=%lld _edgeScheduleNewBatch=%d schedule_new_event=%d eventPeriod=%d _edgeNumTimerEventsLaunchedBatch=%d _edgeTimerBatchSize=%d _edgeLimitConcurrentEvents=%d RunningEventKernelsCount=%d\n", 
        //        gpu_sim_cycle, _edgeScheduleNewBatch, schedule_new_event, eventPeriod, _edgeNumTimerEventsLaunchedBatch, _edgeTimerBatchSize, m_shader_config->_edgeLimitConcurrentEvents, RunningEventKernelsCount());
        if(schedule_new_event && _edgeScheduleNewBatch){            
            _edgeNumTimerEventsLaunchedBatch++;
            if( !_edgeManager->scheduleGPUEvent(eventId) ) {
                printf("EDGE Event Launch error. Couldn't schedule the event: %u\n", eventId);
            }
        }
    }

    if( _edgeNumTimerEventsLaunchedBatch >= _edgeTimerBatchSize ) {
        _edgeNumTimerEventsLaunched+=_edgeTimerBatchSize;
        _edgeScheduleNewBatch = false;
        _edgeNumTimerEventsLaunchedBatch = 0;
        //printf("MARIA DEBUG done scheduling batch \n");
    }

    if (_edgeBatchIsRunning && !_edgeScheduleNewBatch && !EventIsRunning()) { //if event kernel in a batch are still running, wait. once all complete, start counting cycles till the next batch
        _edgePrevBatchEndCycle = gpu_sim_cycle;
        _edgeBatchIsRunning = false;
    }

    if( _edgeNumTimerEventsLaunched >= m_shader_config->_edgeMaxTimerEvents )
        _timerEventMap.clear();
}

void gpgpu_sim::generateInternalInt()
{
    if (m_shader_config->_edgeGenRequestsPatternByDelayAndLimit) {
        generateInternalIntRequestsPatternByDelayAndLimit();
        return;
    }

    // First check if we're generating an internal timer interrupt for the null event
    if( m_shader_config->_edgeInternalInt ) {
        bool isRunning = false;
        for( unsigned i=0; i<m_running_kernels.size(); ++i ) {
            if( m_running_kernels[i] != NULL && !m_running_kernels[i]->done() ) {
                isRunning = true;
                break;
            }
        }

        if( isRunning ) {
            if( (gpu_sim_cycle > 0) && !(gpu_sim_cycle % m_shader_config->_edgeInternalIntPeriod) &&                
                     !_edgeManager->nextCorePendingEvent() ) {                                                              
                assert( _edgeManager->scheduleGPUEvent(m_shader_config->_edgeInternalEventId) );
            }
        } 
    }

    // Then see if there are any timer events to schedule. Ideally no need to have both methods, 
    // the first is really just for debugging purposes. 
    for( TimerEventMap::iterator it = _timerEventMap.begin(); it != _timerEventMap.end(); ++it ) {
        unsigned eventId = it->first;
        unsigned long long eventPeriod = it->second;
        unsigned start_cycle = m_shader_config->_edgeEventStartCycle;
        bool schedule_new_event = ( ( gpu_sim_cycle > start_cycle && gpu_sim_cycle < eventPeriod && !start_cycle_used ) || 
                                    (!( (gpu_sim_cycle-start_cycle) % eventPeriod) && (gpu_sim_cycle > 0))) && 
                                    (_edgeNumTimerEventsLaunched < m_shader_config->_edgeMaxTimerEvents) && isNonEventKernelRunning() &&
                                    (m_shader_config->_edgeLimitConcurrentEvents==0 || RunningEventKernelsCount() < m_shader_config->_edgeLimitConcurrentEvents);
        if (schedule_new_event) {
            start_cycle_used = true;
        }
        if(schedule_new_event && !_edgeUsingTimerBatch){            
            _edgeNumTimerEventsLaunched++;
            if( !_edgeManager->scheduleGPUEvent(eventId) ) {
                printf("EDGE Event Launch error. Couldn't schedule the event: %u\n", eventId);
            }
        } else if (schedule_new_event) { //batch
            _edgeScheduleNewBatch = true;
        }
    }

    for( TimerEventMap::iterator it = _timerBatchEventMap.begin(); it != _timerBatchEventMap.end(); ++it ) {
        unsigned eventId = it->first;
        unsigned long long eventPeriod = it->second;
        bool schedule_new_event = !(gpu_sim_cycle % eventPeriod) && (gpu_sim_cycle > 0) && (_edgeNumTimerEventsLaunchedBatch < _edgeTimerBatchSize) && isNonEventKernelRunning();
        if(schedule_new_event && _edgeScheduleNewBatch){            
            _edgeNumTimerEventsLaunchedBatch++;
            if( !_edgeManager->scheduleGPUEvent(eventId) ) {
                printf("EDGE Event Launch error. Couldn't schedule the event: %u\n", eventId);
            }
        }
    }

    if( _edgeNumTimerEventsLaunchedBatch >= _edgeTimerBatchSize ) {
        _edgeNumTimerEventsLaunched+=_edgeTimerBatchSize;
        _edgeScheduleNewBatch = false;
        _edgeNumTimerEventsLaunchedBatch = 0;
    }

    if( _edgeNumTimerEventsLaunched >= m_shader_config->_edgeMaxTimerEvents )
        _timerEventMap.clear();
}

bool gpgpu_sim::scheduleTimerEvent(unsigned eventId, unsigned long long N)
{
    assert(!_edgeUsingTimerBatch);
    TimerEventMap::iterator it = _timerEventMap.find(eventId);
    if( it == _timerEventMap.end() ) {
        _timerEventMap[eventId] = N;
        return true;
    }
    return false;
}

bool gpgpu_sim::scheduleTimerBatchEvent(int eventId, unsigned long long Nouter, unsigned long long batch, unsigned long long Ninner) {
    if (m_shader_config->_edgeGenRequestsPatternByDelayAndLimit)
        EDGE_DPRINT(EdgeDebug, "Scheduling timer batch event. Delay between batches is %lld cycles, while delay between requests in a batch is %lld. Maximum %d overlapping resusts \n", 
                            Nouter, Ninner, m_shader_config->_edgeLimitConcurrentEvents);    

      printf("MARIA DEBUG scheduleTimerBatchEvent was called \n");
    TimerEventMap::iterator it = _timerEventMap.find(eventId);
    if( it == _timerEventMap.end() ) {
        _timerEventMap[eventId] = Nouter; 
        _edgeUsingTimerBatch = true;
        TimerEventMap::iterator it = _timerBatchEventMap.find(eventId);
        if( it == _timerBatchEventMap.end() ) {
            _timerBatchEventMap[eventId] = Ninner;
            _edgeTimerBatchSize = batch;
            return true;
        }
    }
    return false;
}

void gpgpu_sim::pushKernelStats(kernel_info_t* k)
{
    _completedKernelStats.push_back(std::make_pair(k->entry()->get_name(), k->stats()));
}

bool gpgpu_sim::delayIntWarp() const 
{
    return (_intWarpDelay > 0);
}

// Perform an EDGE operation if address range is correct 
bool gpgpu_sim::EDGEOp(int sid, const warp_inst_t& inst)
{
    static std::map<int, kernel_info_t*> pendingEvents;

    if( !inst.active(0) ) // Thread 0 should always be active in an EDGE warp op
        return false;

    new_addr_type addr = inst.get_addr(0);
    
    if( !EDGE_OP(addr) ) // Check if this is a valid EDGE op or regular memory op
        return false;

    //EDGE_DPRINT(EdgeDebug, "Performing EDGE Request: %p\n", addr);
    GPUEvent* event = NULL;
    kernel_info_t* eventKernel = NULL;

    //gem5CudaGPU->executeEDGEop(inst); // Can interact with Gem5 through here
    EDGE_DPRINT(EdgeDebug, "EDGEOP is called with addr=%d\n", addr);
    switch( addr ) {
        case EDGE_BEGIN_INT:
            // Clear the event from the event manager for SM, sid. Deasserts the interrupt signal
            // if no other pending events. 
            assert( inst.is_store() );
            _edgeManager->beginInt(sid);      // Accept the event on core sid
            break;

        case EDGE_COMPLETE_INT:
            assert( inst.is_store() );
            _edgeManager->completeInt(sid);
            break;

        case EDGE_READ_EVENT_TYPE:
            assert( inst.is_load() ); 
            event = _edgeManager->eventInProgress(sid);
            _edgeManager->pushWBQueue(sid, inst, event->getType());
            break;

        case EDGE_READ_EVENT_ID:
            assert( inst.is_load() );
            _edgeManager->pushWBQueue(sid, inst, event->getEventId());        
            break;
   
        case EDGE_CONFIG_EVENT:
            assert( inst.is_store() );
            assert( pendingEvents.find(sid) == pendingEvents.end() );
            event = _edgeManager->eventInProgress(sid);
           
            if( event->getType() != EDGE_WARMUP ) {
                if( !event->beginEvent() ) { 
                    EDGE_DPRINT(EdgeErr, "Exceeded maximum # of concurrent events...\n");
                    abort();
                }

                eventKernel = new kernel_info_t( event->getKernel() );     // Create a clone of the event kernel
                eventKernel->setParamMem( event->getNextParamAddr() );  // Update the parameter memory
                eventKernel->setEvent( event );                         // Backpointer to the event for when the kernel completes;
                //MARIA
                long long int_start_cycle = get_shader(sid)->edge_get_int_start_cycle();
                printf("MARIA setting int start cycles of event kernel to %lld on core %d\n", int_start_cycle, sid);
                eventKernel->edge_set_int_start_cycle(int_start_cycle);

                EDGE_DPRINT(EdgeDebug, "Config event kernel %s, param mem %p\n", 
                        eventKernel->name().c_str(), (void*)eventKernel->getParamMem());
                pendingEvents[sid] = eventKernel;
            } else {
                EDGE_DPRINT(EdgeDebug, "EDGE_WARMUP Event kernel ignored\n");
            }
            break;
        
        case EDGE_GET_PARAM_BASE_ADDR:
            assert( inst.is_load() );
            assert( pendingEvents.find(sid) != pendingEvents.end() );
            eventKernel = pendingEvents[sid];
            _edgeManager->pushWBQueue(sid, inst, eventKernel->getParamMem());
            break;
        
        case EDGE_SCHEDULE_EVENT:
            event = _edgeManager->eventInProgress(sid);
            if( event->getType() != EDGE_WARMUP ) {
                assert( inst.is_store() );
                assert( pendingEvents.find(sid) != pendingEvents.end() );
                eventKernel = pendingEvents[sid];
                pendingEvents.erase(sid);

                QueueEventKernel(eventKernel); // Queue the event kernel to be scheduled later (highest priority)
            }
            break; 

        ////////////// DEBUG ///////////////
        case EDGE_READ_SM_ID:
            assert( inst.is_load() );
            _edgeManager->pushWBQueue(sid, inst, sid);
            break;

        case EDGE_DELAY_INT:
            assert( inst.is_store() );
            _intWarpDelay = 1000; // EDGE: FIXME
            break;

        case EDGE_READ_NUM_DELAY_OPS:
            assert( inst.is_load() );
            _edgeManager->pushWBQueue(sid, inst, m_shader_config->_edgeInternalIntDelay);
            break;
        
        default:
            EDGE_DPRINT(EdgeErr, "Unknown EDGE Operation type\n");
            break;
    }

    return true;
}
