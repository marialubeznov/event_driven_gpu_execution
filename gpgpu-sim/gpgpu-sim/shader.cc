// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung, Ali Bakhoda,
// George L. Yuan, Andrew Turner, Inderpreet Singh 
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

#include <float.h>
#include "shader.h"
#include "gpu-sim.h"
#include "addrdec.h"
#include "dram.h"
#include "stat-tool.h"
#include "gpu-misc.h"
#include "gpu-cache_gem5.h"
#include "../cuda-sim/ptx_sim.h"
#include "../cuda-sim/ptx-stats.h"
#include "../cuda-sim/cuda-sim.h"
#include "gpu-sim.h"
#include "mem_fetch.h"
#include "mem_latency_stat.h"
#include "visualizer.h"
#include "../statwrapper.h"
#include "icnt_wrapper.h"
#include <string.h>
#include <limits.h>
#include "traffic_breakdown.h"
#include "shader_trace.h"

#include "../edge.h"

#define PRIORITIZE_MSHR_OVER_WB 1
#define MAX(a,b) (((a)>(b))?(a):(b))
#define MIN(a,b) (((a)<(b))?(a):(b))
    

extern gpgpu_sim *g_the_gpu;

/////////////////////////////////////////////////////////////////////////////

std::list<unsigned> shader_core_ctx::get_regs_written( const inst_t &fvt ) const
{
   std::list<unsigned> result;
   for( unsigned op=0; op < MAX_REG_OPERANDS; op++ ) {
      int reg_num = fvt.arch_reg.dst[op]; // this math needs to match that used in function_info::ptx_decode_inst
      if( reg_num >= 0 ) // valid register
         result.push_back(reg_num);
   }
   return result;
}

void shader_core_ctx::printICacheUtilization() 
{
    printf("Shader: %d\n", m_sid);
    if( m_L1I ) {
        l1icache_gem5* c = dynamic_cast<l1icache_gem5*>(m_L1I);
        c->printEdgeUtilization();
    }
}



// EDGE
shader_core_ctx::shader_core_ctx( class gpgpu_sim *gpu, 
                                  class simt_core_cluster *cluster,
                                  unsigned shader_id,
                                  unsigned tpc_id,
                                  const struct shader_core_config *config,
                                  const struct memory_config *mem_config,
                                  shader_core_stats *stats )
  : core_t( gpu, NULL, config->warp_size, config->n_thread_per_shader, config->_intMode ),
    m_barriers( config->max_warps_per_shader, config->max_cta_per_core ),
//    : core_t( gpu, NULL, config->warp_size, 
//             config->n_thread_per_shader+config->_nIntThreads, 
//             config->_intMode ),
//     m_barriers( config->max_warps_per_shader+config->_nIntWarps, 
//                 config->max_cta_per_core+config->nIntCTAs ),
     m_dynamic_warp_id(0), _iWarpRunning(false), _intSignal(false), _edgeTotalIntSchedCycles(0), _edgeTotalIntRunCycles(0), _edgeNumInts(0), 
     _edgeNumFreeWarpInts(0), _edgeNumVictimWarpInts(0), _edgeNumNoFlushVictimWarpInts(0), _edgeNumExitWarpInts(0), _edgeCurrentIntSchedStallCycles(0),
     _edgeTotalIntSchedStall(0), _edgeMaxIntSchedStall(0), _edgeMinIntSchedStall((unsigned long long)-1), _nReplayLoads(0), _nBadReplayLoads(0), 
     _edgeNumEdgeBarriers(0), _edgeNumEdgeReleaseBarriers(0), _edgeSkippedBarriers(0), _edgeBarriersRestored(0)
{
    _edgeIntState = IDLE;
    //if (m_config->_edgeRunISR) { //commenting since causes seg fault for some reason
    _edgeSaveState = new EdgeSaveState(this, config->warp_size);
    
    //}

    // EDGE
    _warpIntStatsVector.resize(config->max_warps_per_shader);

    _warpOccupancyPerCycle = 0;
    _registerUtilizationPerCycle = 0;

    _kernelFinishing.clear();
    //m_kernel_finishing = false;
    m_cluster = cluster;
    m_config = config;
    m_memory_config = mem_config;
    m_stats = stats;
    unsigned warp_size=config->warp_size;
    
    m_sid = shader_id;
    m_tpc = tpc_id;

    //edge
    for (int i=0; i<m_config->_edgeEventCtasPerCore; i++) 
        _edgeCtas.push_back(m_config->max_cta_per_core-1-i);
    
    for (int i=0; i<m_config->max_cta_per_core; i++) 
        _CTAKernelMap.push_back(NULL);

    
    // Setup pipeline
    m_pipeline_reg.reserve(N_PIPELINE_STAGES);
    for (int j = 0; j<N_PIPELINE_STAGES; j++) {
        m_pipeline_reg.push_back(register_set(m_config->pipe_widths[j],pipeline_stage_name_decode[j]));
    }
   
    // Initialize thread contexts
    m_threadState = (thread_ctx_t*) calloc(sizeof(thread_ctx_t), config->n_thread_per_shader);
    
    // Initialize warps
    m_not_completed = 0;
    m_active_threads.reset();
    m_n_active_cta = 0;
    
    for ( unsigned i = 0; i<MAX_CTA_PER_SHADER; i++ ) 
        m_cta_status[i]=0;

    for (unsigned i = 0; i<config->n_thread_per_shader; i++) {
        m_thread[i]= NULL;
        m_threadState[i].m_cta_id = -1;
        m_threadState[i].m_active = false;
    }
    
    if ( m_config->gpgpu_perfect_mem ) {
        m_icnt = new perfect_memory_interface(this,cluster);
    } else {
        m_icnt = new shader_memory_interface(this,cluster);
    }
    m_mem_fetch_allocator = new shader_core_mem_fetch_allocator(shader_id,tpc_id,mem_config);
    
    // fetch
    m_last_warp_fetched = 0;
    
    #define STRSIZE 1024
    char name[STRSIZE];
    snprintf(name, STRSIZE, "L1I_%03d", m_sid);
    m_L1I = new l1icache_gem5(m_gpu, name, m_config->m_L1I_config, m_sid, get_shader_instruction_cache_id(), m_icnt, IN_L1I_MISS_QUEUE);
    
    m_warp.resize(m_config->max_warps_per_shader, shd_warp_t(this, warp_size));

    m_scoreboard = new Scoreboard(m_sid, m_config->max_warps_per_shader);

    //scedulers
    //must currently occur after all inputs have been initialized.
    std::string sched_config = m_config->gpgpu_scheduler_string;
    const concrete_scheduler scheduler = sched_config.find("lrr") != std::string::npos ?
                                         CONCRETE_SCHEDULER_LRR :
                                         sched_config.find("two_level_active") != std::string::npos ?
                                         CONCRETE_SCHEDULER_TWO_LEVEL_ACTIVE :
                                         sched_config.find("gto") != std::string::npos ?
                                         CONCRETE_SCHEDULER_GTO :
                                         sched_config.find("warp_limiting") != std::string::npos ?
                                         CONCRETE_SCHEDULER_WARP_LIMITING :
                                         sched_config.find("edge") != std::string::npos ? 
                                         CONCRETE_SCHEDULER_EDGE :
                                         NUM_CONCRETE_SCHEDULERS;
    assert ( scheduler != NUM_CONCRETE_SCHEDULERS );

    m_scheduler_prio = 0;
    for (int i = 0; i < m_config->gpgpu_num_sched_per_core; i++) {
        switch( scheduler )
        {
            case CONCRETE_SCHEDULER_LRR:
                schedulers.push_back(
                    new lrr_scheduler( m_stats,
                                       this,
                                       m_scoreboard,
                                       m_simt_stack,
                                       &m_warp,
                                       &m_pipeline_reg[ID_OC_SP],
                                       &m_pipeline_reg[ID_OC_SFU],
                                       &m_pipeline_reg[ID_OC_MEM],
                                       i
                                     )
                );
                break;
            case CONCRETE_SCHEDULER_TWO_LEVEL_ACTIVE:
                schedulers.push_back(
                    new two_level_active_scheduler( m_stats,
                                                    this,
                                                    m_scoreboard,
                                                    m_simt_stack,
                                                    &m_warp,
                                                    &m_pipeline_reg[ID_OC_SP],
                                                    &m_pipeline_reg[ID_OC_SFU],
                                                    &m_pipeline_reg[ID_OC_MEM],
                                                    i,
                                                    config->gpgpu_scheduler_string
                                                  )
                );
                break;
            case CONCRETE_SCHEDULER_GTO:
                schedulers.push_back(
                    new gto_scheduler( m_stats,
                                       this,
                                       m_scoreboard,
                                       m_simt_stack,
                                       &m_warp,
                                       &m_pipeline_reg[ID_OC_SP],
                                       &m_pipeline_reg[ID_OC_SFU],
                                       &m_pipeline_reg[ID_OC_MEM],
                                       i
                                     )
                );
                break;
            case CONCRETE_SCHEDULER_WARP_LIMITING:
                schedulers.push_back(
                    new swl_scheduler( m_stats,
                                       this,
                                       m_scoreboard,
                                       m_simt_stack,
                                       &m_warp,
                                       &m_pipeline_reg[ID_OC_SP],
                                       &m_pipeline_reg[ID_OC_SFU],
                                       &m_pipeline_reg[ID_OC_MEM],
                                       i,
                                       config->gpgpu_scheduler_string
                                     )
                );
                break;
            case CONCRETE_SCHEDULER_EDGE:
                // Should be in EDGE interrupt mode if using the EDGE scheduler
                  assert( m_config->_intMode ); 
                schedulers.push_back(
                    new EdgeScheduler( m_stats,
                                       this,
                                       m_scoreboard,
                                       m_simt_stack,
                                       &m_warp,
                                       &m_pipeline_reg[ID_OC_SP],
                                       &m_pipeline_reg[ID_OC_SFU],
                                       &m_pipeline_reg[ID_OC_MEM],
                                       i
                                     ));
                break;
            default:
                abort();
        };
    }
    
    // EDGE: FIXME:
    //for (unsigned i = 0; i < m_warp.size()-m_config->_nIntWarps; i++) {
    for (unsigned i = 0; i < m_warp.size(); i++) {
        //distribute i's evenly though schedulers;
        schedulers[i%m_config->gpgpu_num_sched_per_core]->add_supervised_warp_id(i);
    }
    for ( int i = 0; i < m_config->gpgpu_num_sched_per_core; ++i ) {
        schedulers[i]->done_adding_supervised_warps();
    }
    
    //op collector configuration
    enum { SP_CUS, SFU_CUS, MEM_CUS, GEN_CUS };
    m_operand_collector.add_cu_set(SP_CUS, m_config->gpgpu_operand_collector_num_units_sp, m_config->gpgpu_operand_collector_num_out_ports_sp);
    m_operand_collector.add_cu_set(SFU_CUS, m_config->gpgpu_operand_collector_num_units_sfu, m_config->gpgpu_operand_collector_num_out_ports_sfu);
    m_operand_collector.add_cu_set(MEM_CUS, m_config->gpgpu_operand_collector_num_units_mem, m_config->gpgpu_operand_collector_num_out_ports_mem);
    m_operand_collector.add_cu_set(GEN_CUS, m_config->gpgpu_operand_collector_num_units_gen, m_config->gpgpu_operand_collector_num_out_ports_gen);
    
    opndcoll_rfu_t::port_vector_t in_ports;
    opndcoll_rfu_t::port_vector_t out_ports;
    opndcoll_rfu_t::uint_vector_t cu_sets;
    for (unsigned i = 0; i < m_config->gpgpu_operand_collector_num_in_ports_sp; i++) {
        in_ports.push_back(&m_pipeline_reg[ID_OC_SP]);
        out_ports.push_back(&m_pipeline_reg[OC_EX_SP]);
        cu_sets.push_back((unsigned)SP_CUS);
        cu_sets.push_back((unsigned)GEN_CUS);
        m_operand_collector.add_port(in_ports,out_ports,cu_sets);
        in_ports.clear(),out_ports.clear(),cu_sets.clear();
    }
    
    for (unsigned i = 0; i < m_config->gpgpu_operand_collector_num_in_ports_sfu; i++) {
        in_ports.push_back(&m_pipeline_reg[ID_OC_SFU]);
        out_ports.push_back(&m_pipeline_reg[OC_EX_SFU]);
        cu_sets.push_back((unsigned)SFU_CUS);
        cu_sets.push_back((unsigned)GEN_CUS);
        m_operand_collector.add_port(in_ports,out_ports,cu_sets);
        in_ports.clear(),out_ports.clear(),cu_sets.clear();
    }
    
    for (unsigned i = 0; i < m_config->gpgpu_operand_collector_num_in_ports_mem; i++) {
        in_ports.push_back(&m_pipeline_reg[ID_OC_MEM]);
        out_ports.push_back(&m_pipeline_reg[OC_EX_MEM]);
        cu_sets.push_back((unsigned)MEM_CUS);
        cu_sets.push_back((unsigned)GEN_CUS);                       
        m_operand_collector.add_port(in_ports,out_ports,cu_sets);
        in_ports.clear(),out_ports.clear(),cu_sets.clear();
    }   
    
    
    for (unsigned i = 0; i < m_config->gpgpu_operand_collector_num_in_ports_gen; i++) {
        in_ports.push_back(&m_pipeline_reg[ID_OC_SP]);
        in_ports.push_back(&m_pipeline_reg[ID_OC_SFU]);
        in_ports.push_back(&m_pipeline_reg[ID_OC_MEM]);
        out_ports.push_back(&m_pipeline_reg[OC_EX_SP]);
        out_ports.push_back(&m_pipeline_reg[OC_EX_SFU]);
        out_ports.push_back(&m_pipeline_reg[OC_EX_MEM]);
        cu_sets.push_back((unsigned)GEN_CUS);   
        m_operand_collector.add_port(in_ports,out_ports,cu_sets);
        in_ports.clear(),out_ports.clear(),cu_sets.clear();
    }
    
    m_operand_collector.init( m_config->gpgpu_num_reg_banks, this );
    
    // execute
    m_num_function_units = m_config->gpgpu_num_sp_units + m_config->gpgpu_num_sfu_units + 1; // sp_unit, sfu, ldst_unit
    //m_dispatch_port = new enum pipeline_stage_name_t[ m_num_function_units ];
    //m_issue_port = new enum pipeline_stage_name_t[ m_num_function_units ];
    
    //m_fu = new simd_function_unit*[m_num_function_units];
    
    for (int k = 0; k < m_config->gpgpu_num_sp_units; k++) {
        m_fu.push_back(new sp_unit( &m_pipeline_reg[EX_WB], m_config, this ));
        m_dispatch_port.push_back(ID_OC_SP);
        m_issue_port.push_back(OC_EX_SP);
    }
    
    for (int k = 0; k < m_config->gpgpu_num_sfu_units; k++) {
        m_fu.push_back(new sfu( &m_pipeline_reg[EX_WB], m_config, this ));
        m_dispatch_port.push_back(ID_OC_SFU);
        m_issue_port.push_back(OC_EX_SFU);
    }
    
    m_ldst_unit = new ldst_unit( m_icnt, m_mem_fetch_allocator, this, &m_operand_collector, m_scoreboard, config, mem_config, stats, shader_id, tpc_id );
    m_fu.push_back(m_ldst_unit);
    m_dispatch_port.push_back(ID_OC_MEM);
    m_issue_port.push_back(OC_EX_MEM);
    
    assert(m_num_function_units == m_fu.size() and m_fu.size() == m_dispatch_port.size() and m_fu.size() == m_issue_port.size());
    
    //there are as many result buses as the width of the EX_WB stage
    num_result_bus = config->pipe_widths[EX_WB];
    for(unsigned i=0; i<num_result_bus; i++){
        this->m_result_bus.push_back(new std::bitset<MAX_ALU_LATENCY>());
    }
    
    m_last_inst_gpu_sim_cycle = 0;
    m_last_inst_gpu_tot_sim_cycle = 0;

    resetOccupiedResources();
}

void shader_core_ctx::resetOccupiedResources()
{
    // CDP - Concurrent kernels
    m_occupied_n_threads = 0;
    m_occupied_shmem     = 0;
    m_occupied_regs      = 0;
    m_occupied_ctas      = 0;
    m_occupied_hwtid.reset();
    m_occupied_cta_to_hwtid.clear();


    // EDGE - Need to pre-occupy any dedicated interrupt warp resources
    if( m_config->isIntDedicated() == 2 && _iKernel != NULL ) {
        int warpId = m_config->max_warps_per_shader-1;    // Set the interrupt warp to the last warp
        
        // Threads            
        m_occupied_n_threads += m_config->_nIntThreads;

        // Registers
        m_occupied_regs += (m_config->_nIntThreads * ((ptx_sim_kernel_info(_iKernel->entry())->regs+3)&~3));

        // No Shared Memory                
        // No additional CTAs

        int startThread = warpId * m_config->warp_size;
        int endThread = startThread + m_config->warp_size;

        for( unsigned i=startThread; i<endThread; ++i ) {
            assert( !m_occupied_hwtid.test(i) );
            m_occupied_hwtid.set(i);
        }   
    }
}


void shader_core_ctx::reinit(unsigned start_thread, unsigned end_thread, bool reset_not_completed ) 
{
   if( reset_not_completed ) {
       m_not_completed = 0;
       m_active_threads.reset();
       resetOccupiedResources();
   }
   for (unsigned i = start_thread; i<end_thread; i++) {
      m_threadState[i].reset();
   }
   for (unsigned i = start_thread / m_config->warp_size; i < end_thread / m_config->warp_size; ++i) {
      m_warp[i].reset();
      m_simt_stack[i]->reset();
   }
}

void shader_core_ctx::init_warps( unsigned cta_id, unsigned start_thread, unsigned end_thread )
{
    address_type start_pc = next_pc(start_thread);
    if (m_config->model == POST_DOMINATOR) {
        unsigned start_warp = start_thread / m_config->warp_size;
        unsigned end_warp = end_thread / m_config->warp_size + ((end_thread % m_config->warp_size)? 1 : 0);
        for (unsigned i = start_warp; i < end_warp; ++i) {
            unsigned n_active=0;
            simt_mask_t active_threads;
            for (unsigned t = 0; t < m_config->warp_size; t++) {
                unsigned hwtid = i * m_config->warp_size + t;
                if ( hwtid < end_thread ) {
                    n_active++;
                    //if (m_active_threads.test(hwtid)) {
                    //    printf("MARIA DEBUG hwtid %d is set in m_active_threads\n", hwtid);
                    //}
                    assert( !m_active_threads.test(hwtid) );
                    //printf("MARIA DEBUG hwtid %d setting in m_active_threads core %d \n", hwtid, m_sid);
                    m_active_threads.set( hwtid );
                    active_threads.set(t);
                }
            }
            m_simt_stack[i]->launch(start_pc,active_threads);
            m_warp[i].init(start_pc,cta_id,i,active_threads, m_dynamic_warp_id);
            ++m_dynamic_warp_id;
    
            m_not_completed += n_active;
      }
   }
}

/**
 * Return a free hardware CTA ID or -1 if none are free. 
 */
int shader_core_ctx::findFreeCta()
{
    unsigned freeCtaHwId = (unsigned)-1;
    unsigned maxCtaPerCore;

    if( !m_config->gpgpu_concurrent_kernel_sm )
        maxCtaPerCore = kernel_max_cta_per_shader;
    else
        maxCtaPerCore = m_config->max_cta_per_core;

    for( unsigned i=0; i<maxCtaPerCore; ++i ) {
        if( m_cta_status[i] == 0 ) {
            freeCtaHwId = i;
            break;
        }
    }
    
    return freeCtaHwId;
}

bool shader_core_ctx::AreThereFreeRegsOnShader(kernel_info_t* k) {
    if (!m_config->_edgeEnableRegisterRenamingInsteadBackup) {
        return false;
    }
   
    const struct gpgpu_ptx_sim_kernel_info* kernel_info = ptx_sim_kernel_info(k->entry());
    unsigned threads_per_cta  = k->threads_per_cta();
    unsigned int padded_cta_size = threads_per_cta;
    unsigned int warp_size = m_config->warp_size; 
        
    if( padded_cta_size%warp_size ) 
        padded_cta_size = ((padded_cta_size/warp_size)+1)*(warp_size);

    unsigned int used_regs = padded_cta_size * ((kernel_info->regs+3)&~3);
    bool result = ( (m_occupied_regs + used_regs) <= m_config->gpgpu_shader_registers );
    
    if (result) {
        EDGE_DPRINT(EdgeDebug, "%lld: Event kernel preempted a warp %d on shader %d. Reg file has %d/%d registers available, so no need to save&restore the registers of the victim warp\n", 
                    gpu_sim_cycle, _edgeWid, m_sid, (m_config->gpgpu_shader_registers-m_occupied_regs+used_regs), 
                    m_config->gpgpu_shader_registers );  
        EDGE_DPRINT(EdgeDebug, "%lld:  Incrementing occupied regs on shader %d by %d \n", gpu_sim_cycle, m_sid, used_regs);
        m_occupied_regs += used_regs;
        k->SetEdgeOccupiedExtraRegs();
    }

    return result;
}

bool shader_core_ctx::edgeReserveEventResources(kernel_info_t* k, unsigned nCtas)
{    
    assert( m_config->_edgeEventReserveCta > 0 );

    assert( m_config->isIntDedicated() != 2 );
    assert( m_occupied_n_threads == 0 );
    assert( m_occupied_shmem == 0 );
    assert( m_occupied_regs == 0 );
    assert( m_occupied_ctas == 0 );
    assert( m_occupied_hwtid.none() );

    for( unsigned i=0; i<nCtas; ++i ) {
        assert( m_cta_status[i] == 0 );
        assert( m_occupied_cta_to_hwtid.find(i) == m_occupied_cta_to_hwtid.end() );
    }

    unsigned threads_per_cta  = k->threads_per_cta();
    const class function_info *kernel = k->entry();
    unsigned int padded_cta_size = threads_per_cta;
    unsigned int warp_size = m_config->warp_size;

    // Pad CTA to warp size
    if( padded_cta_size%warp_size ) 
        padded_cta_size = ((padded_cta_size/warp_size)+1)*(warp_size);

    padded_cta_size *= nCtas; // Check for number of requested CTAs. 

    // Can't if we're beyond the maximum number of threads for this shader
    unsigned nThreads = m_config->n_thread_per_shader;
    if( m_config->isIntDedicated() )
        nThreads -= m_config->_nIntThreads;

    assert( (m_occupied_n_threads + padded_cta_size) <= m_config->n_thread_per_shader );
          
    // Can't if there are not enough contiguous thread contexts for this CTA on this SM
    // EDGE: This function will not select the interrupt thread contexts
    //if( find_available_hwtid(padded_cta_size, false) == -1 )
    assert( edgeFindAvailableHwtid(padded_cta_size, false) != -1 );

    // Get the ptx_sim_kernel_info structure for this kernel
    // Can't if the shared memory requirement is greater than the current shared memory 
    const struct gpgpu_ptx_sim_kernel_info* kernel_info = ptx_sim_kernel_info(kernel);
    assert( (m_occupied_shmem + kernel_info->smem*nCtas ) <= m_config->gpgpu_shmem_size );

    // Can't if the number of registers used is greater than the total registers for this SM
    unsigned int used_regs = padded_cta_size * ((kernel_info->regs+3)&~3);
    assert( (m_occupied_regs + used_regs) <= m_config->gpgpu_shader_registers );

    // Can't if we've exceeded the maximum number of CTAs for this SM 
    // EDGE: Decrease maximum number of CTAs for interrupt CTAs, if they exist
    unsigned maxCtaPerCore = m_config->max_cta_per_core;
    if( m_config->_intMode )
        maxCtaPerCore -= m_config->_edgeEventCtasPerCore;

    assert( (m_occupied_ctas + nCtas) <= maxCtaPerCore );

    // Okay, so now we know the resources are good to go, let's reserve what we need!

    // If occupy set, then actually occupy the SM with a CTA from this kernel
    m_occupied_n_threads += padded_cta_size;
    m_occupied_shmem += kernel_info->smem*nCtas;
    m_occupied_regs += (padded_cta_size * ((kernel_info->regs+3)&~3));
    m_occupied_ctas += nCtas;

    for( unsigned i=0; i<padded_cta_size; ++i ) {
        assert( !m_occupied_hwtid.test(i) );
        m_occupied_hwtid.set(i);
    }

    _edgeActiveReservedCtas.resize(nCtas);
    int startThread = 0;
    int singleCtaSize = threads_per_cta;
    if( singleCtaSize%warp_size ) 
        singleCtaSize = ((singleCtaSize/warp_size)+1)*(warp_size);

    for( unsigned i=0; i<nCtas; ++i ) {
        _edgeActiveReservedCtas[i] = false;
        m_occupied_cta_to_hwtid[i] = startThread + i*singleCtaSize;
    }

    unsigned nWarps = padded_cta_size / m_config->warp_size;
    for( unsigned i=0; i<nWarps; ++i ) {
        m_warp[i].setReserved();
    }

    printf("EDGE Reserved resources for Event kernel %s. Shader %d: %d threads, %d shared mem, %d registers, %d ctas\n",
                kernel->get_name().c_str(), m_sid, m_occupied_n_threads, m_occupied_shmem, m_occupied_regs, m_occupied_ctas);  

    return true;
}



bool shader_core_ctx::isFreeIntWarpAvailable()
{
    return occupy_shader_resource_1block(*_iKernel, false, true);
}

/**
 * Select a CTA and Warp to handle the interrupt. 
 *
 * Based on the EDGE configuration, this may be a dedicated CTA/Warp context,
 * or may re-use an existing context. 
 */
bool 
shader_core_ctx::selectIntCtaWarpCtx(int& ctaId, int& warpId, kernel_info_t* EventKernel, bool interrupt, bool occupy)
{
    assert( EventKernel );
    assert( EventKernel->threads_per_cta() == m_config->warp_size );

    bool isFree = false;                         
    int ctaSize = EventKernel->threads_per_cta();

    std::string edgeWarpConfig = m_config->_edgeWarpSelectionStr;
    const GPUIntWarpSelection warpSelection = edgeWarpConfig.find("pseudo_dedicated") != std::string::npos ?
                                                EDGE_PSEUDO_DEDICATED :
                                                edgeWarpConfig.find("dedicated") != std::string::npos ?
                                                EDGE_DEDICATED :
                                                edgeWarpConfig.find("oldest") != std::string::npos ?
                                                EDGE_OLDEST :
                                                edgeWarpConfig.find("newest") != std::string::npos ?
                                                EDGE_NEWEST :
                                                edgeWarpConfig.find("best") != std::string::npos ?
                                                EDGE_BEST :
                                                
                                                EDGE_NUM_WARP_SELECTION;

    assert ( warpSelection != EDGE_NUM_WARP_SELECTION );

    unsigned minPendingLoads = (unsigned)-1;
    int warpIdx = -1;    

    // EDGE NEW FIXME: We always have a valid free CTA context to use. We may just steal a warp/thread
    // context from another CTA. TODO: Can't have barrier_set with this. Needs to be -1 and have max_cta_per_core be +1
    if (occupy) {
        assert(!_edgeCtas.empty());
        ctaId = _edgeCtas.front(); //m_config->max_cta_per_core-1; // Always use the last CTA 
        _edgeCtas.pop_front();
        //printf("MARIA DEBUG occupying cta %d on sm %d _edgeCtas.size()=%d \n", ctaId, m_sid, _edgeCtas.size());
    };

    if( warpSelection == EDGE_DEDICATED ) {
        // Dedicated hardware for the interrupt CTA/warp. Just return these indices. No need to save context. 
        isFree = true;
        //ctaId = m_config->max_cta_per_core-1;         // Set the interrupt CTA to the last CTA
        warpId = m_config->max_warps_per_shader-1;    // Set the interrupt warp to the last warp

    } else if( warpSelection == EDGE_PSEUDO_DEDICATED ) {
        // This is similar to EDGE_DEDICATED, however, instead of adding resources to accomodate the interrupt warp, 
        // we just reserve one warp context for the interrupt warp always. 
        isFree = true;
        warpId = m_config->max_warps_per_shader-1;
                    
    } else {
        // Use an existing CTA/Warp context
        isFree = occupy_shader_resource_1block(*EventKernel, false, interrupt);
       
        // EDGE FIXME: No longer attempting to use an existing CTA context. Use a dedicated
        // CTA context for the interrupt. TODO: Verify hardware overheads.  
        // Get a CTA ID
        //ctaId = findFreeCta();

        if( isFree ) { // There is a free CTA/Warp context available, use them!
            assert( ctaId != (unsigned)-1 );

            // Get the warp ID
            unsigned startThread = find_available_hwtid(ctaSize, false); 
            assert( startThread != (unsigned)-1 );
            warpId = startThread / m_config->warp_size; 
               
        } else { // Need to re-use an existing context

            shd_warp_t* victimWarp = ChooseVictimWarp(EventKernel);            
            
            // Set the cta and warp IDs
            if( victimWarp == NULL) {
                EDGE_DPRINT(EdgeErr, "Could not find a valid warp context to evict on SM %d, something went wrong!\n", m_sid);
                abort();
            } else {
                assert( victimWarp );
                warpId = victimWarp->get_warp_id();
                // EDGE FIXME: Now using a dedicated CTA id
                //ctaId = m_threadState[warpId*m_config->warp_size].m_cta_id;
            }
        }
    }
    fflush(stdout);
    return isFree;
}

shd_warp_t* shader_core_ctx::ChooseVictimWarp(kernel_info_t* EventKernel) {
    std::string edgeWarpConfig = m_config->_edgeWarpSelectionStr;
    const GPUIntWarpSelection warpSelection = edgeWarpConfig.find("pseudo_dedicated") != std::string::npos ?
                                                EDGE_PSEUDO_DEDICATED :
                                                edgeWarpConfig.find("dedicated") != std::string::npos ?
                                                EDGE_DEDICATED :
                                                edgeWarpConfig.find("oldest") != std::string::npos ?
                                                EDGE_OLDEST :
                                                edgeWarpConfig.find("newest") != std::string::npos ?
                                                EDGE_NEWEST :
                                                edgeWarpConfig.find("best") != std::string::npos ?
                                                EDGE_BEST :
                                                
                                                EDGE_NUM_WARP_SELECTION;

    assert ( warpSelection != EDGE_NUM_WARP_SELECTION );
    int warpIdx;
    unsigned minPendingLoads = (unsigned)-1;
    // Get all supervised warps into a single vector
            WarpVector temp;
            for( int i=0; i<m_config->gpgpu_num_sched_per_core; ++i) {
                ((EdgeScheduler*)schedulers[i])->getSupervisedWarps(temp);
            }           

            // Sort based on oldest dynamic ID
            std::sort( temp.begin(), temp.end(), scheduler_unit::sort_warps_by_oldest_dynamic_id );
            
            // Select a victim warp
            shd_warp_t* victimWarp = NULL;
            switch( warpSelection ) {
                case EDGE_OLDEST:
                    // Search from oldest to newest
                    //printf("CanInterruptWarp results for SM %d: ", m_sid);
                    for( WarpVector::iterator it = temp.begin(); it < temp.end(); ++it ) {                   
                        if( canInterruptWarp(*it, EventKernel) == 0 ) {
                            victimWarp = *it;
                            break;
                        //} else {
                            //printf("%d ", canInterruptWarp(*it, EventKernel));
                        }
                    }
                    //printf("\n");
                    break;

                case EDGE_NEWEST:
                    // Search from newest to oldest
                    for( WarpVector::iterator it = temp.end()-1; it >= temp.begin(); --it ) {
                        if( canInterruptWarp(*it, EventKernel) == 0 ) {
                            victimWarp = *it;
                            break;
                        }
                    }

                    break;
                
                case EDGE_RANDOM:
                    EDGE_DPRINT(EdgeErr, "Interrupt warp selection not implemented: %d\n", warpSelection);
                    break;

                case EDGE_BEST:
                    // Search for any warps which don't require a flush
                    for( WarpVector::iterator it = temp.begin(); it < temp.end(); ++it ) {
                        if( canInterruptWarp(*it, EventKernel) == 0 ) {
                            if( !warpNeedsFlush( (*it)->get_warp_id() ) ) {
                                // Great! We now have a warp that doesn't require a flush
                                victimWarp = *it;
                                _edgeNumNoFlushVictimWarpInts++;
                                break;
                            }

                            // Otherwise, try and find the warp with the lowest number of pending loads
                            if( warpPendingLoads( (*it)->get_warp_id() ) < minPendingLoads ) {
                                warpIdx = (*it)->get_warp_id();
                                minPendingLoads = warpPendingLoads( warpIdx );
                            }

                        }
                    }

                    // Sadly every warp requires a flush, so choose the ne with the lowest
                    if( !victimWarp ) {
                        assert( warpIdx != -1 );
                        victimWarp = &m_warp[warpIdx];
                    }

                   break;
            


                default:
                    EDGE_DPRINT(EdgeErr, "Undefined interrupt warp selection identifier: %d\n", warpSelection);
                    abort();
            }
            return victimWarp;
}

int shader_core_ctx::canInterruptWarp(const shd_warp_t* victimWarp, kernel_info_t* EventKernel) const
{
    //victimWarp->print(stdout);

    // Warp conditions for not selecting this warp to interrupt:
    if( (victimWarp->get_warp_id() == (unsigned)-1) ||      // Invalid warp
        (victimWarp->done_exit()) ||                        // Warp is in the process of completing 
        (victimWarp->functional_done()) ||                  // Warp is funcitonally complete
        (victimWarp->isReserved()) )                        // Warp is reserved
    {
        return 1;
    }


    // Now, can we fit in this warp context
    const function_info* intFuncInfo = EventKernel->entry();
    const struct gpgpu_ptx_sim_kernel_info* intSimKernelInfo = ptx_sim_kernel_info(intFuncInfo);

    int tid = victimWarp->get_warp_id() * m_config->warp_size;
    kernel_info_t* victimKernelInfo = &m_thread[tid]->get_kernel();

    if (victimKernelInfo->isEventKernel()) {
        return 2;
    }

    const function_info* victimFuncInfo = victimKernelInfo->entry();
    const struct gpgpu_ptx_sim_kernel_info* victimSimKernelInfo = ptx_sim_kernel_info(victimFuncInfo);

    // Assert the warp is the same
    for( unsigned i=1; i<m_config->warp_size; ++i ) {
        if( m_threadState[tid+i].m_active ) {
            assert( victimKernelInfo == &m_thread[tid+i]->get_kernel() );
        }
    }
    
    // Check if we have enough registers
    unsigned ikernel_regs = ptx_sim_kernel_info(intFuncInfo)->regs;
    if( victimSimKernelInfo->regs <  ikernel_regs ) 
        return 3;

    //printf("MARIA can interrupt warp returns true!!! victimSimKernelInfo->regs=%d ikernel_regs=%d \n", ((victimSimKernelInfo->regs+3)&~3), ((ikernel_regs+3)&~3));
    /*
    printf( "Kernel info -- Threads per cta: %d, block dim (%d,%d,%d), regs: %d\n",
            k->threads_per_cta(), k->get_cta_dim().x, k->get_cta_dim().y, k->get_cta_dim().z,
            32*((ki->regs+3)&~3) );
    */
    
    return 0;
}



// EDGE: Anything that should be initialized once and only once. Reset above can be called
//       at the end of each iWarp execution.
void
shader_core_ctx::initEDGE(kernel_info_t* k)
{
    if( m_config->_intMode ) {
        _intSignal = false;
        _iWarpRunning = false;
        _edgeWid = -1;
        _edgeCid = -1;
        _edgeDoFlush = false;
        _edgeIsFree = false;
        _edgeDropLoads = false;
        
        _iKernel = k;                                   // Save the kernel pointer for the interrupt kernel
        resetOccupiedResources();

        // Set the instruction cache address range
        if( m_config->_edgeIntReserveIcache ) {
            l1icache_gem5* icache = dynamic_cast<l1icache_gem5*>(m_L1I);
            const function_info* fi = k->entry();

            new_addr_type start = fi->get_start_PC() + k->get_inst_base_vaddr();
            icache->setEdgeInstRange(start, fi->get_instr_mem_size());
        }

    } else {
        EDGE_DPRINT(EdgeErr, "Trying to call InitEDGE() when not in Interrupt mode\n");
    }
}


// EDGE TODO: This resets the state of interrupt CTA and warp. This function assumes 
// that all of the state for these structures has been correctly saved and reset prior to 
// calling this function
void 
shader_core_ctx::configureIntCtx(int cid, int wid, bool isFree)
{
    assert( m_config->_intMode );
    assert( _iKernel );

    warp_set_t warps;
    unsigned nThreadsInBlock = 0;

    unsigned startThread = wid*m_config->warp_size;
    unsigned endThread = startThread + m_config->_nIntThreads;
    
    //printf("MARIA DEBUG adding ikernel to CTAKernelMap cid=%d wid=%d core %d\n", cid, wid, m_sid);
    _CTAKernelMap[cid] = _iKernel;

    //m_warp[wid].setIntWarp();
    assert( m_warp[wid].isIntWarp() );
    
    assert(m_occupied_cta_to_hwtid.find(cid) == m_occupied_cta_to_hwtid.end());
    m_occupied_cta_to_hwtid[cid] = startThread;

    EDGE_DPRINT(EdgeDebug, "Initializing thread context before running the interrupt on core %d. Threads %d-%d.\n", m_sid, startThread, endThread); 
    for( unsigned i=startThread; i<endThread; ++i ) {

        m_active_threads.reset(i);
        m_threadState[i].m_cta_id = cid;

        // -- Now we're fully initializing the thread contexts
        nThreadsInBlock += ptx_sim_init_thread( *_iKernel, 
                                                &m_thread[i],
                                                m_sid, 
                                                i,
                                                m_config->_nIntThreads-(i-startThread),
                                                m_config->n_thread_per_shader,
                                                this,
                                                cid,
                                                wid,
                                                m_cluster->get_gpu(),
                                                true,
                                                isFree);
        
         
        m_threadState[i].m_active = true;
        warps.set(wid);
        
    }   
    m_warp[wid].kernel = _iKernel;
    _warpIntStatsVector[wid].push_back( new WarpIntStats() );
    _warpIntStatsVector[wid].back()->_startCycle = gpu_sim_cycle;

    // Should only have as many threads as interrupt threads
    assert(nThreadsInBlock == m_config->_nIntThreads);
    
    m_cta_status[cid] += nThreadsInBlock;
                                                             
    // Initialize barrier for iWarp
    m_barriers.allocate_barrier(cid, warps);
                                                               
    // Initialize SIMT stack and fetch hardware
    init_warps(cid, startThread, endThread);

    //reset the sb for data hazards
    schedulers[wid % m_config->gpgpu_num_sched_per_core]->ClearScoreboard(wid);
}

void shader_core_ctx::setIntSignal() 
{
    if( m_config->_intMode ) {
        if( _edgeIntState == IDLE ) {
            assert( !_intSignal );
            _intSignal = true;
            _edgeIntState = SELECT_WARP;
            _intSigCount += 1;
            if (!m_config->_edgeWarmupInt || _intSigCount > m_config->num_cluster()) {
                printf("MARIA setIntSignal for core %d on cycle %lld \n", get_sid(), gpu_sim_cycle + gpu_tot_sim_cycle);
                //_edgeInterruptAssertCycle.push_back(gpu_sim_cycle + gpu_tot_sim_cycle);
            } else {
                printf("MARIA setIntSignal for core %d on cycle %lld - warmup, ignoring \n", get_sid(), gpu_sim_cycle + gpu_tot_sim_cycle);
            }
        }
    } else {
        EDGE_DPRINT(EdgeErr, "Trying to set interrupt when interrupt mode not enabled on core %d.\n", get_sid());
        abort();
    }
}


void shader_core_ctx::clearIntSignal() 
{
    if( m_config->_intMode ) {
        EDGE_DPRINT(EdgeDebug, "Clearing interrupt signal for shader <%d>\n", m_sid);    
        assert( _intSignal );
        assert( _edgeIntState == IWARP_RUNNING );
        _edgeIntState = IWARP_COMPLETING;
    
    } else {
        EDGE_DPRINT(EdgeErr, "Trying to clear interrupt when interrupt mode not enabled..\n");
        abort();
    }
}


EdgeSaveState::EdgeSaveState(class shader_core_ctx *shader, unsigned warpSize) : _valid(false), _warpSize(warpSize)
{
    _warp = new shd_warp_t(shader, warpSize);
    _threadInfo = new ptx_thread_info*[warpSize];
    _threadState = new thread_ctx_t[warpSize];
    
    _activeThreads.reset();
    _occupiedHwThreads.reset();

    _isActive = false;
    _isAtBarrier = false;
}


EdgeSaveState::~EdgeSaveState()
{
    if( _warp )
        delete _warp;
    
    if( _simtStack )
        delete _simtStack;

    if( _threadInfo )
        delete[] _threadInfo;

    if( _threadState ) 
        delete[] _threadState;
}


bool EdgeSaveState::saveState(unsigned wid, unsigned cid, unsigned ctaStatus, shd_warp_t* warp,
                                simt_stack* simtStack, ptx_thread_info** threadInfo, thread_ctx_t* threadState,
                                std::bitset<MAX_THREAD_PER_SM>& activeThreads, kernel_info_t* kernel,
                                std::bitset<MAX_THREAD_PER_SM>& occupiedHwThreads,
                                bool isActive, bool isAtBarrier)
{
    if( _valid ) 
        return false;

    unsigned startThread = wid*_warpSize;

    _wid = wid;
    _cid = cid;
    _ctaStatus = ctaStatus;
    *_warp = *warp;
    _simtStack = simtStack;
    _kernel = kernel;

    for( unsigned i=0; i<_warpSize; ++i ) {
        _threadInfo[i] = threadInfo[i];
        _threadState[i] = threadState[i];
        _activeThreads[i] = activeThreads[i+startThread];
        _occupiedHwThreads[i] = occupiedHwThreads[i+startThread];
    }

    _isActive = isActive;
    _isAtBarrier = isAtBarrier;

    _valid = true;
    return true;
}


bool EdgeSaveState::restoreState(shd_warp_t* warp, simt_stack** simtStack, ptx_thread_info** threadInfo,
        thread_ctx_t* threadState, std::bitset<MAX_THREAD_PER_SM>& activeThreads, 
        std::bitset<MAX_THREAD_PER_SM>& occupiedHwThreads, bool& isActive, bool& isAtBarrier)
{
    if( !_valid )
        return false;

    *warp = *_warp;
    assert( *simtStack != NULL );
    delete *simtStack;
    *simtStack = _simtStack;

    unsigned startThread = _wid*_warpSize;
    for( unsigned i=0; i<_warpSize; ++i ) {
        if( threadInfo[i] ) {
            delete threadInfo[i];
            threadInfo[i] = NULL;
        }
        threadInfo[i] = _threadInfo[i];
        threadState[i] = _threadState[i];
        activeThreads[i+startThread] = _activeThreads[i]; 
        occupiedHwThreads[i+startThread] = _occupiedHwThreads[i];
    }

    isActive = _isActive;
    isAtBarrier = _isAtBarrier;

    _valid = false;
    return true;
}


void shader_core_ctx::edgeResetIntState(unsigned CurrEdgeWid, unsigned CurrEdgeCid, bool freeThreads)
{
    assert( CurrEdgeWid != -1 && CurrEdgeCid != -1 );
    // Reset and configure warp state
    m_warp[CurrEdgeWid].reset();
    m_simt_stack[CurrEdgeWid]->reset();
    //m_warp[CurrEdgeWid].unset_done_exit();

    int startThread = CurrEdgeWid*m_config->warp_size;
    for( int i=startThread; i<startThread+m_config->warp_size; ++i ) {
        // Only free threads if they're interrupt warp threads. Otherwise, they've been saved and should
        // not be freed. 
        if( freeThreads && m_thread[i] ) {
            assert( m_thread[i]->is_done() );
            m_thread[i]->m_cta_info->register_deleted_thread(m_thread[i]);
            delete m_thread[i];      
        }

        m_thread[i] = NULL;
        m_threadState[i].reset();
        m_active_threads.reset(i);
    }
}


void shader_core_ctx::edgeSaveResetHwState(EdgeSaveState* _edgeSaveStateInput)
{
    assert( (_edgeWid != -1) && (_edgeCid != -1) && !_edgeIsFree );

    int startThread = _edgeWid * m_config->warp_size;

    // Check if we're waiting at a barrier
    bool atBarrier = false;
    if( m_config->_edgeSkipBarrier ) 
        atBarrier = m_barriers.edgeSetVictimBarrier( m_warp[_edgeWid].get_cta_id(), _edgeWid );

    if( _edgeSaveStateInput->saveState(_edgeWid, _edgeCid, m_cta_status[_edgeCid], &m_warp[_edgeWid],
                m_simt_stack[_edgeWid], &m_thread[startThread], &m_threadState[startThread], 
                m_active_threads, m_warp[_edgeWid].kernel, m_occupied_hwtid, 
                m_barriers.warp_active(_edgeWid), atBarrier) ) {
        EDGE_DPRINT(EdgeDebug, "Save state for CTA %d, Warp %d, successful\n", _edgeCid, _edgeWid);    

        if( atBarrier )
            _edgeSkippedBarriers++;

        m_simt_stack[_edgeWid] = new simt_stack(_edgeWid, m_config->warp_size);
       
        // Increment the interrupt counter for this warp
        _warpIntStatsVector[_edgeWid].back()->_nInterrupts++;

        edgeResetIntState(_edgeWid, _edgeCid, false);

        m_warp[_edgeWid].setIntWarp();

    } else {
        EDGE_DPRINT(EdgeErr, "Save state for CTA %d, Warp %d, failed\n", _edgeCid, _edgeWid);    
    }

}

void shader_core_ctx::edgeRestoreHwState(EdgeSaveState* _edgeSaveStateInput)
{
    int startThread = _edgeSaveStateInput->wid() * m_config->warp_size;
    bool isActive = false;
    bool isAtBarrier = false;
    if( _edgeSaveStateInput->restoreState(&m_warp[_edgeSaveStateInput->wid()], &m_simt_stack[_edgeSaveStateInput->wid()], &m_thread[startThread], 
                &m_threadState[startThread], m_active_threads, m_occupied_hwtid, isActive, isAtBarrier) ) {
        EDGE_DPRINT(EdgeDebug, "Restore state for CTA %d, Warp %d, successful. isAtBarrier: %d, isActive: %d done_exit: %d \n", 
            _edgeSaveStateInput->cid(), _edgeSaveStateInput->wid(), isAtBarrier, isActive, m_warp[_edgeSaveStateInput->wid()].done_exit());    

        if( isAtBarrier ) {
            //assert( 0 );
            //m_barriers.set_at_barrier(_edgeWid);
            if( !m_barriers.edgeIsVictimBarrierDone(_edgeSaveStateInput->wid()) ) {
                m_barriers.edgeRestoreVictimBarrier(_edgeSaveStateInput->wid());
                _edgeBarriersRestored++;
            }
        }

        if( isActive )
            m_barriers.set_active(_edgeSaveStateInput->wid());


        if( m_config->_edgeReplayLoads ) {
            assert( m_config->_edgeFlushIBuffer ); // Need both
            m_warp[_edgeSaveStateInput->wid()].replayLoads(); 
        }
        assert(_edgeSaveStateInput->kernel());
        m_warp[_edgeSaveStateInput->wid()].kernel = _edgeSaveStateInput->kernel();
    } else {
        EDGE_DPRINT(EdgeErr, "Restore state for CTA %d, Warp %d, failed\n", _edgeSaveStateInput->cid(), _edgeSaveStateInput->wid());    
    }
}


bool ldst_unit::releaseRegisters(const warp_inst_t* inst)
{
    m_scoreboard->releaseRegisters(inst);

    // for each output register (up to 4 for vectors)
    for( unsigned r=0; r < 4; r++ ) {
        if( inst->out[r] > 0 ) {
            assert( inst->space.get_type() != shared_space );

            int warp_id = inst->warp_id();
            assert( m_pending_writes[warp_id][inst->out[r]] > 0 );
            unsigned still_pending = --m_pending_writes[warp_id][inst->out[r]];
            
            if( !still_pending ) {
                m_pending_writes[warp_id].erase(inst->out[r]);
            }
        }
    }

    return true;
}

bool shader_core_ctx::edgeWarpPendingRegWrites(int warpId)
{
    return m_ldst_unit->warpPendingRegWrites(warpId);
}

bool ldst_unit::warpPendingRegWrites(int warpId)
{
    if( m_pending_writes[warpId].size() > 0 )
        return true;
    else
        return false;
}

bool shader_core_ctx::delayIntWarp() const
{
    return m_gpu->delayIntWarp();
}

void shader_core_ctx::printStats(FILE* f)
{
    fprintf(f, "=========================\n");
    fprintf(f, "Core %d: \n", m_sid);

    fprintf(f, "\tAvg_int_warp_sched_stalls = %.4lf\n", (double)_edgeTotalIntSchedStall / (double)_edgeNumVictimWarpInts);
    fprintf(f, "\tMin_int_warp_sched_stalls = %lld\n", _edgeMinIntSchedStall);
    fprintf(f, "\tMax_int_warp_sched_stalls = %lld\n\n", _edgeMaxIntSchedStall);

#if 0
    fprintf(f, "\twarp_id, start_cycle, end_cycle, n_interrupts | ..., star_cycle, end_cycle, n_interrupts\n");
    for( unsigned i=0; i<_warpIntStatsVector.size(); ++i ) {
        if( _warpIntStatsVector[i].size() > 0 ) {
            fprintf(f, "\t%d: ", i);
            for( unsigned j=0; j<_warpIntStatsVector[i].size(); ++j ) {
                WarpIntStats* s = _warpIntStatsVector[i].at(j);
                fprintf(f, "%lld, %lld, %lld | ", s->_startCycle, s->_endCycle, s->_nInterrupts);
            }
            fprintf(f, "\n");
        }
    }
    for( unsigned i=0; i<_warpIntStatsVector.size(); ++i ) {
        if( _warpIntStatsVector[i].size() > 0 ) {
            fprintf(f, "\t%d: ", i);
            for( unsigned j=0; j<_warpIntStatsVector[i].size(); ++j ) {
                WarpIntStats* s = _warpIntStatsVector[i].at(j);
                fprintf(f, "%lld, ", s->_endCycle - s->_startCycle);
            }
            fprintf(f, "\n");
        }
    }
#endif
    fprintf(f, "=========================\n");
}

void shader_core_ctx::edgeOccupyShaderResourceIntWhenWarpIsFree(kernel_info_t* k)
{
    // EDGE FIXME: Occupy a single CTA for this interrupt. Don't need to call 
    // occupy_shader_resource_1block(), since we already know at this point that 
    // we have an available context to run on. 

    // Pseudo dedicated configuraiton ALWAYS has the resources consumed. Don't duplicate. 
    if( m_config->isIntDedicated() != 2 ) {
        // Threads            
        m_occupied_n_threads += m_config->_nIntThreads;

        // Registers - need to update only when selected warp is free. 
        // if warp is preempted and there are other registers available in reg file --> renaming, set in AreThereFreeRegsOnShader
        // if no regs available they are saved and restored inside an event kernel --> no extra resources consumed
        const struct gpgpu_ptx_sim_kernel_info* kernel_info = ptx_sim_kernel_info(k->entry());
        unsigned threads_per_cta  = k->threads_per_cta();
        unsigned int padded_cta_size = threads_per_cta;
        unsigned int warp_size = m_config->warp_size; 
        
        if( padded_cta_size%warp_size ) 
            padded_cta_size = ((padded_cta_size/warp_size)+1)*(warp_size);

        unsigned int used_regs = padded_cta_size * ((kernel_info->regs+3)&~3);
        m_occupied_regs += used_regs;
        k->SetEdgeOccupiedExtraRegs();

        // No Shared Memory                
        // No additional CTAs

        int startThread = _edgeWid * m_config->warp_size;
        int endThread = startThread + m_config->warp_size;

        for( unsigned i=startThread; i<endThread; ++i ) {
            assert( !m_occupied_hwtid.test(i) );
            m_occupied_hwtid.set(i);
        }  
        k->SetEdgeOccupiedExtraThreads(); 
    }
}

void shader_core_ctx::edgeIntCycleReset()
{
    _edgeWid = -1;
    _edgeCid = -1;
    _edgeDoFlush = false;
    _edgeIsFree = false;
    _edgeDropLoads = false;
    _edgeIntWarpSelectCycle = 0;
    _edgeIntLaunchCycle = 0;
}

void shader_core_ctx::edgeRecordIntStall()
{
    // Verify overlapping conditions
    if( !m_warp[_edgeWid].ibuffer_empty() )     assert( m_warp[_edgeWid].inst_in_pipeline() );
    if( edgeWarpPendingRegWrites(_edgeWid) )    assert( m_scoreboard->pendingWrites(_edgeWid) );
    
    // Record stall conditions
    if( m_warp[_edgeWid].inst_in_pipeline() )   _edgeIntStallStats._instInPipeline++;
    if( m_warp[_edgeWid].imiss_pending() )      _edgeIntStallStats._iMissPending++;
    if( warp_waiting_at_barrier(_edgeWid) )     _edgeIntStallStats._atBarrier++;
    if( warp_waiting_at_mem_barrier(_edgeWid) ) _edgeIntStallStats._atMemBarrier++;
    if( m_warp[_edgeWid].get_n_atomic() > 0 )   _edgeIntStallStats._atomics++; 
    if( m_scoreboard->pendingWrites(_edgeWid) ) _edgeIntStallStats._scoreboardRegPending++;
    if( warpPendingLoads(_edgeWid) )            _edgeIntStallStats._pendingLoads++;
}

unsigned shader_core_ctx::allWarpsPendingLoads() {
    unsigned result = 0;
    WarpVector temp;
    for( int i=0; i<m_config->gpgpu_num_sched_per_core; ++i) {
        ((EdgeScheduler*)schedulers[i])->getSupervisedWarps(temp);
    }
    for( WarpVector::iterator it = temp.begin(); it < temp.end(); ++it ) {                   
        result += warpPendingLoads((*it)->get_warp_id());
    }   
    return result;                                                                                                            
                                                

}

bool shader_core_ctx::warpNeedsFlush(int wid)
{
            // EDGE FIXME: Need to verify all of the flushing conditions here. 
            // Signal to the shader to stop issuing instructions for _edgeWid
            /* !warpLoadPending(_edgeWid) && */// New condition to avoid loads returning after we've interrupted the warp
#if 0            
            if( !m_warp[_edgeWid].inst_in_pipeline() && !m_warp[_edgeWid].imiss_pending() && 
                    m_warp[_edgeWid].ibuffer_empty() && !edgeWarpPendingRegWrites(_edgeWid) &&
                    !warp_waiting_at_barrier(_edgeWid) &&
                    !warp_waiting_at_mem_barrier(_edgeWid) &&
                    m_warp[_edgeWid].get_n_atomic() == 0 &&
                    !m_scoreboard->pendingWrites(_edgeWid) ) {  
#else

            // TODO: Store ACk? Why is this missing. Do I never hit this condition? The ACK should cause a failure as is...
            if( m_warp[wid].inst_in_pipeline() )
                return true;

            if( m_warp[wid].imiss_pending() )
                return true;

            if( warp_waiting_at_mem_barrier(wid) )
                return true;

            if( warp_waiting_at_barrier(wid) && !m_config->_edgeSkipBarrier )
                return true;

            if( m_warp[wid].get_n_atomic() > 0 )
                return true;
            
            if( m_scoreboard->pendingWrites(wid) )
                return true;

            return false;    
#endif
}

bool shd_warp_t::loadNextWb()
{
    for( std::list<PendingLoadInst>::iterator it = _inFlightLoadQueue.begin(); 
            it != _inFlightLoadQueue.end(); ++it ) {
        if( m_shader->loadNextWb( &(*it)._inst ) )
            return true;
    }
    return false;
}

// void shader_core_ctx::NewEdgeSaveStateAndIssueBlock2Core() {
//     //save state
//     edgeSaveResetHwState(GetEdgeState(_iKernel->GetEdgeSaveStateId()));
//     //launch kernel
//     _iKernel->edge_set_int_start_cycle(edge_get_int_start_cycle_for_isr());
//     m_occupied_n_threads -= m_config->warp_size;
//     unsigned num = m_cluster->issue_block2core();
//     assert(num);
//     //m_last_cluster_issue=idx;
//     //m_total_cta_launched += num;
//     //done
//     EDGE_DPRINT(EdgeDebug, "%lld: Event kernel can run, preemption done. Moving to IDLE\n", gpu_sim_cycle);
//     StopPreemption();
// }

void shader_core_ctx::StopPreemption(bool restoreState, EdgeIntStates currState) {
    if (restoreState) {
        edgeRestoreHwState(GetEdgeState(_iKernel->GetEdgeSaveStateId()));
    }
    _iKernel->UnsetEdgeSaveStateValid();
    _intSignal = false;
    _iWarpRunning = false;
    //for( int i=0; i<m_config->gpgpu_num_sched_per_core; ++i) {
    //    EdgeScheduler* es = (EdgeScheduler*)schedulers[i];    
    //    es->clearIntSignal(); 
    //}
    edgeResetIntState(_edgeWid, _edgeCid, false); // Moving here
    edgeIntCycleReset();
    _iKernel->UnsetISRKernel();
    _edgeIntState = IDLE;
    EDGE_DPRINT(EdgeDebug, "%lld: While preempting a warp on Shader %d (state=%d), kernel has already been scheduled on another SM \n", gpu_sim_cycle, m_sid, currState);  
}

void shader_core_ctx::StartNewWarpPreemption(kernel_info_t* kernel) {
    setiKernel(kernel);
    setPrivateIntSignal();
    setEdgeState(SELECT_WARP);
    incIntSigCount();

    kernel->stats()._edgePreemptionQueueWait = gpu_sim_cycle - kernel->stats()._edgePreemptionQueueWait;
    kernel->stats()._edgePreemptionLen = gpu_sim_cycle;
    //printf("MARIA DEBUG assigning _edgePreemptionQueueWait to %lld and _edgePreemptionLen to %lld\n", 
    //    kernel->stats()._edgePreemptionQueueWait, kernel->stats()._edgePreemptionLen);
}

void shader_core_ctx::ScheduleFastPathEvent(kernel_info_t* kernel) {
    kernel->stats()._edgePreemptionQueueWait = gpu_sim_cycle;
    //printf("MARIA DEBUG assigning _edgePreemptionQueueWait to %lld\n", 
    //    kernel->stats()._edgePreemptionQueueWait);
    if (_edgeIntState == IDLE) {
        StartNewWarpPreemption(kernel);
    } else { //queue
        _eventKernelQueue.push_back(kernel); 
    }
}

void shader_core_ctx::EnableRegSavingInKernel() {
    if (AreThereFreeRegsOnShader(_iKernel)) {
        return;
    }
    //replace the _ikernel with its version withot reg saving code. 
    //also replace in all related structs in gpgpusim (like m_running_kernels)    
    m_gpu->SwapRunningKernel(_iKernel, _iKernel->GetEdgeSwapEventKernel());
    EDGE_DPRINT(EdgeDebug, "%lld: Event kernel requires register backup %d on shader %d, replacing the kernel with its version w register save&restore (%d --> %d) \n", 
                gpu_sim_cycle, _edgeWid, m_sid, _iKernel, _iKernel->GetEdgeSwapEventKernel());  

    _iKernel = _iKernel->GetEdgeSwapEventKernel();  
    m_warp[_edgeWid].kernel = _iKernel;     
}

void shader_core_ctx::edgeIntCycle()
{
    unsigned startThread,endThread;
    switch( _edgeIntState ) {
        case IDLE:
            // No interrupt, just return
            edgeIntCycleReset();
            _edgeCurrentIntSchedStallCycles = 0;
            if (!_eventKernelQueue.empty()) {
                StartNewWarpPreemption(_eventKernelQueue.front());
                _eventKernelQueue.erase(_eventKernelQueue.begin());
            }
            break;

        case SELECT_WARP:
            EDGE_DPRINT(EdgeDebug, "%lld: Starting preemption on SM %d. EdgeState = SELECT_WARP \n", gpu_sim_cycle, m_sid);
            //EDGE_DPRINT(EdgeDebug, "%lld: Selecting an interrupt warp on Shader %d: ", gpu_sim_cycle, m_sid);    
            assert(!_iKernel->no_more_ctas_to_run());
            m_gpu->IncEventRunning();
            if (!m_config->_edgeRunISR) {
                _iKernel->SetISRKernel();
            }

            _edgeNumInts++;

            // Interrupt was just received, select a warp context to handle the interrupt
            assert( _intSignal );
            
            // EDGE FIXME: This should reserve the context so no other warps can take it...
            _edgeIsFree = selectIntCtaWarpCtx(_edgeCid, _edgeWid, _iKernel, m_config->_edgeRunISR, true);

            assert( !m_warp[_edgeWid].isReserved() );

            _edgeIntWarpSelectCycle = gpu_sim_cycle; // Record the cycle when the int warp was selected

            // Dedicated should ALWAYS be free
            if( m_config->isIntDedicated() )
                assert( _edgeIsFree );

            // Need to save state and interrupt a warp if none are frqee
            if( !_edgeIsFree ) {
                _edgeDoFlush = true;
                _edgeIntState = FLUSH_PIPELINE;
                _edgeNumVictimWarpInts++;

                // Signal to the warp scheduler's that they should prioritize the victim warp
                if( m_config->_edgeVictimWarpHighPriority ) {
                    for( int i=0; i<m_config->gpgpu_num_sched_per_core; ++i) {
                        ((EdgeScheduler*)schedulers[i])->prioritizeVictimWarp(_edgeWid);
                    }
                }

                // Flush any pending intructions from the iBuffer
                if( m_config->_edgeFlushIBuffer )
                    m_warp[_edgeWid].ibuffer_flush();


                // Replay loads
                if( m_config->_edgeReplayLoads ) {
                    assert( m_config->_edgeFlushIBuffer ); // Need both
                    if( m_warp[_edgeWid].pendingLoads() ) {
                        _edgeDropLoads = true;
                    }
                    //m_warp[_edgeWid].dropLoads();
                }




                // TODO: Need to adjust the amount of resources used. But we don't want another block being able to be launched
                // now because we have fewer resources utilized. Likely don't need to do anything, since the resources are
                // already held by the interrupted warp. 
            } else {
                _edgeIntState = LAUNCH_IWARP;
                _edgeNumFreeWarpInts++;

                edgeOccupyShaderResourceIntWhenWarpIsFree(_iKernel);
                m_warp[_edgeWid].setIntWarp();
            }

            //EDGE_DPRINT(EdgeDebug, "(free? %s, cta_id: %d, warp_id: %d preempted pc: %d )\n", _edgeIsFree ? "yes" : "no", _edgeCid, _edgeWid, m_warp[_edgeWid].get_pc());

            break;

        case FLUSH_PIPELINE:
            assert(!_iKernel->no_more_ctas_to_run());
            // Wait for all of the pending instructions to finish for the iWarp, if any, before interrupting
            assert( !_edgeIsFree && _edgeCid != -1 && _edgeWid != -1 );

            //EDGE_DPRINT(EdgeDebug, "%lld: Preemption on SM %d, warp %d. EdgeState = FLUSH_PIPELINE. _edgeCurrentIntSchedStallCycles = %d _edgeEventWarpIds size = %d \n", 
                //gpu_sim_cycle, m_sid, _edgeWid, _edgeCurrentIntSchedStallCycles, _edgeEventWarpIds.size());

            // Record reason for the interrupt stall
            edgeRecordIntStall();

            _edgeCurrentIntSchedStallCycles++;

            if( _edgeDropLoads == true ) {
                if( m_warp[_edgeWid].inst_in_pipeline() || m_warp[_edgeWid].loadNextWb() ) {
                    _edgeIntState = FLUSH_PIPELINE;
                    break;
                } else {
                    _edgeDropLoads = false;
                    m_warp[_edgeWid].dropLoads();
                }
            }

            if( !warpNeedsFlush(_edgeWid) ) {

                // Signal to the warp scheduler's that they should de-prioritize the victim warp
                if( m_config->_edgeVictimWarpHighPriority ) {
                    for( int i=0; i<m_config->gpgpu_num_sched_per_core; ++i) {
                        ((EdgeScheduler*)schedulers[i])->restoreVictimWarp(_edgeWid);
                    }
                }
               
                // Interesting corner case where a warp is selected to interrupt, then completes its instructions while flushing, 
                // which never gets to exit because we're stalling the warp from ever reaching the exit code. Basically, 
                // a warp which needed to be interrupted no longer needs to be interrupted!
                if( m_warp[_edgeWid].done_exit() ) {
                    _edgeNumExitWarpInts++;
                    

                    // EDGE FIXME NEW TEST: If we've happend to select a warp to interrupt, which exits while we're
                    // flushing the pipeline, just re-select a new warp. Taking this warp will stall an entire CTA
                    // launch, since this CTA will have to block until the interrupy warp completes, or will have to 
                    // completely separate the warp from the CTA
                    #define EDGE_RETRY_INT_ON_EXIT
                    #ifdef EDGE_RETRY_INT_ON_EXIT
                        EDGE_DPRINT(EdgeDebug, "%lld: Warp %d exited, retrying interrupt with a new warp selection on Shader %d: ", gpu_sim_cycle, _edgeWid, m_sid);    
                        edgeIntCycleReset();
                        _edgeIntState = SELECT_WARP;
                    #else
                        _edgeIsFree = true;
                        _edgeDoFlush = false;

                        // Remove this warp from it's previous CTA, if necessary
                        m_barriers.removeWarpFromCta( _edgeWid, m_warp[_edgeWid].get_cta_id() ); 

                        // Now need to occupy the resources for the interrupt warp, since it's free 
                        // Moved from below. Need to occupy the resources. 
                        edgeOccupyShaderResourceIntWhenWarpIsFree(_iKernel);

                        _edgeIntState = LAUNCH_IWARP;
                    #endif

                } else {
                    //EDGE_DPRINT(EdgeDebug, "%lld: Flush is done for warp %d on SM %d. \n", gpu_sim_cycle, _edgeWid, m_sid);
                    assert(!m_warp[_edgeWid].inst_in_pipeline());
                    _edgeIntState = SAVE_HW_CTX;    // Once pipeline is flushed, save the hardware state for this warp
                }
                

                // Stall cycle stats
                _edgeTotalIntSchedStall += _edgeCurrentIntSchedStallCycles;

                if( _edgeCurrentIntSchedStallCycles > _edgeMaxIntSchedStall ) 
                    _edgeMaxIntSchedStall = _edgeCurrentIntSchedStallCycles;

                if( _edgeCurrentIntSchedStallCycles < _edgeMinIntSchedStall )
                    _edgeMinIntSchedStall = _edgeCurrentIntSchedStallCycles;
        
                _edgeCurrentIntSchedStallCycles = 0;

            } else {
                _edgeIntState = FLUSH_PIPELINE; // Otherwise, continue waiting until flush is complete
            
                // If we're at a barrier, record and release it
//                if( m_config->_edgeSkipBarrier ) {
//                    m_barriers.edgeSetVictimBarrier( m_warp[_edgeWid].get_cta_id(), _edgeWid );
//                }
            }

            break;

        case SAVE_HW_CTX:
            assert(!m_warp[_edgeWid].inst_in_pipeline());
            assert(!_iKernel->no_more_ctas_to_run());
            //EDGE_DPRINT(EdgeDebug, "%lld: Saving context of warp %d with pc=%d on Shader %d\n", 
            //    gpu_sim_cycle, _edgeWid, m_warp[_edgeWid].get_pc(), m_sid);
                
            // HACK: FIXME: If in between the flushing cycle and this cycle the warp ends up completing (because the flush now allows
            // a warp to complete before testing the flush stall condition), just reselect a new warp
            if( m_warp[_edgeWid].done_exit() ) {
                _edgeNumExitWarpInts++;
                edgeIntCycleReset();
                _edgeIntState = SELECT_WARP;
            } else {
                EnableRegSavingInKernel();
                if (!m_config->_edgeRunISR) {
                    _iKernel->SetEdgeSaveStateValid();
                }
                _edgeSaveState = new EdgeSaveState(this, m_config->warp_size);
                _iKernel->SetEdgeSaveStateId(gpu_sim_cycle);
                AddNewEdgeState(_iKernel->GetEdgeSaveStateId(), _edgeSaveState);
                if (m_config->_edgeRunISR) {
                    edgeSaveResetHwState(_edgeSaveState);
                } else {
                    edgeSaveResetHwState(GetEdgeState(_iKernel->GetEdgeSaveStateId()));
                }  
                _edgeDoFlush = false;
                _edgeIntState = LAUNCH_IWARP;
            }
            break;

        case LAUNCH_IWARP:
            _edgeEventWarpIds.push_back(_edgeWid);
            assert(!_iKernel->no_more_ctas_to_run());
            
            // Record how long it took us to get from warp selection to warp launching (both free and victim warps). 
            _edgeTotalIntSchedCycles += (gpu_sim_cycle - _edgeIntWarpSelectCycle);

            _edgeIntLaunchCycle = gpu_sim_cycle;

            // Initialize the interrupt CTA and Warp context
            configureIntCtx(_edgeCid, _edgeWid, _edgeIsFree);

	    startThread = _edgeWid*m_config->warp_size;
            endThread = startThread + m_config->_nIntThreads;

            if (m_config->_edgeDontLaunchEventKernel) {
                for( unsigned i=startThread; i<endThread; ++i ) {
                    m_thread[i]->set_done();
                    m_thread[i]->registerExit();
                    m_thread[i]->exitCore();

                 }
                for(int i=0;i<m_config->warp_size;i++)
                {
                    m_warp[_edgeWid].set_completed(i);
                 }
             }


            shader_CTA_count_log(m_sid, 1);
            m_n_active_cta++;
            m_occupied_ctas++;
                
            // Interaction with GEM5 - Issue the CTA if it hasn't already been issued 
            m_gpu->gem5CudaGPU->getCudaCore(m_sid)->record_block_issue(_edgeCid);

            // EDGE FIXME: Used to reserve resources for the int warp here, a subsequent cycle from "SELECT_WARP".
            // There is a potential race condition here for a free interrupt warp context in which 
            // we select a warp, and then occupy it here. Moving this code up to the selection process.
            
            // Signal to the warp scheduler's that they should prioritize the iWarp
            if (m_config->_edgeRunISR) {
                for( int i=0; i<m_config->gpgpu_num_sched_per_core; ++i) {
                    ((EdgeScheduler*)schedulers[i])->setIntSignal();
                }
            }

            // CDP - Concurrent kernels on SM
            assert( !_iKernel->running() );
            _iKernel->inc_running();
            _iWarpRunning = true;
            _iKernel->start();

            //long long int_start_cycle_isr = edge_get_int_start_cycle_for_isr();
            //printf("MARIA setting int start cycles of ISR kernel to %lld on core %d\n", int_start_cycle_isr, m_sid);
            if (!_iKernel->isEventKernel()) {
                _iKernel->edge_set_int_start_cycle(edge_get_int_start_cycle_for_isr());
            }
            
            EDGE_DPRINT(EdgeDebug, "%lld: Interrupt warp on Shader %d occupied %d threads, %d shared mem, %d registers, %d ctas for kernel %s %p \n",
                    gpu_sim_cycle, m_sid, m_occupied_n_threads, m_occupied_shmem, m_occupied_regs, m_occupied_ctas, _iKernel->entry()->get_name().c_str(), _iKernel);

            if (m_config->_edgeRunISR) {
                _edgeIntState = IWARP_RUNNING;
            } else { //done here!
                _edgeTotalIntRunCycles += (gpu_sim_cycle - _edgeIntLaunchCycle);
                _intSignal = false;
                _iWarpRunning = false;
                _iKernel->stats()._edgePreemptionLen = gpu_sim_cycle - _iKernel->stats()._edgePreemptionLen;
                //printf("MARIA DEBUG assigning _edgePreemptionLen to %lld\n", _iKernel->stats()._edgePreemptionLen);
                
                //edgeResetIntState(_edgeWid, _edgeCid, false);     
                EDGE_DPRINT(EdgeDebug, "%lld: Event kernel KernelId=%d has been launched on SM %d warp %d CTA %d, preemption done. Get back to IDLE\n", 
                            gpu_sim_cycle, _iKernel->get_uid(), m_sid, _edgeWid, _edgeCid);    
                _iKernel = NULL;   
                edgeIntCycleReset();                
                _edgeIntState = IDLE;
            }

            break;

        case IWARP_RUNNING:
            assert(m_config->_edgeRunISR);
            // else - Nothing to do while iWarp running            
            break;
        
        case IWARP_COMPLETING:
            assert(m_config->_edgeRunISR);
            EDGE_DPRINT(EdgeDebug, "%lld: ISR Kernel finishing!!\n", gpu_sim_cycle);

            _edgeTotalIntRunCycles += (gpu_sim_cycle - _edgeIntLaunchCycle);

            _intSignal = false;
            _iWarpRunning = false;
            for( int i=0; i<m_config->gpgpu_num_sched_per_core; ++i) {
                EdgeScheduler* es = (EdgeScheduler*)schedulers[i];    
                es->clearIntSignal(); 
            }

            assert( !_iKernel->running() );
            _iKernel->resetKernel();
            

            if( _edgeIsFree ) {
                edgeResetIntState(_edgeWid, _edgeCid, true); // Moving here
                edgeIntCycleReset();
                _edgeIntState = IDLE;
            } else {
                _edgeIntState = RESTORE_HW_CTX;
            }

            break;

        case RESTORE_HW_CTX:
            assert(m_config->_edgeRunISR);
            // EDGE: Moving this here to avoid another race condition between cycles where something
            // is freed and then the interrupt warp/restore warp state is in some transient state. 
            edgeResetIntState(_edgeWid, _edgeCid, true);
            edgeRestoreHwState(_edgeSaveState);
            edgeIntCycleReset();
            EDGE_DPRINT(EdgeDebug, "%lld: Restoring hw state and get back to IDLE\n", gpu_sim_cycle);
            _edgeIntState = IDLE;
            break;

        default: 
            abort();
    }

    return;

}

unsigned shader_core_ctx::TotalInstInPipeline() {
    unsigned result = 0;
    for( unsigned i=0; i < m_config->max_warps_per_shader; i++ ) {
        result += m_warp[i].num_inst_in_pipeline();
    }
}

//minimum number of event kernels running + shorttest preemption queue
unsigned shader_core_ctx::EdgeCoreScheduleCost(kernel_info_t* eventKernel) {
    unsigned result = _edgeSaveStateList.size() + _eventKernelQueue.size(); // + allWarpsPendingLoads(); // + TotalInstInPipeline();
    if (_edgeIntState != IDLE) {
        result++;
    }
    int ctaId, warpId;
    if (!selectIntCtaWarpCtx(ctaId, warpId, eventKernel->GetEdgeSwapEventKernel(), false, false)) {
        result++;
    }
    return result;
}

bool shader_core_ctx::CanRunEdgeEvent(kernel_info_t* eventKernel) {
    if (_edgeCtas.empty()) {
        return false;
    }
    if (_edgeIntState != IDLE) {
        return false;
    }
    //checking resources with bigger kernel
    bool isFree = occupy_shader_resource_1block(*(eventKernel->GetEdgeSwapEventKernel()), false, false);
    bool canIntWarp = ChooseVictimWarp(eventKernel->GetEdgeSwapEventKernel())!=NULL;
    if (!isFree && !canIntWarp) {
        return false;
    }
    return true;
}

// return the next pc of a thread 
address_type shader_core_ctx::next_pc( int tid ) const
{
    if( tid == -1 ) 
        return -1;
    ptx_thread_info *the_thread = m_thread[tid];
    if ( the_thread == NULL )
        return -1;
    return the_thread->get_pc(); // PC should already be updatd to next PC at this point (was set in shader_decode() last time thread ran)
}

void gpgpu_sim::get_pdom_stack_top_info( unsigned sid, unsigned tid, unsigned *pc, unsigned *rpc )
{
    unsigned cluster_id = m_shader_config->sid_to_cluster(sid);
    m_cluster[cluster_id]->get_pdom_stack_top_info(sid,tid,pc,rpc);
}

void shader_core_ctx::get_pdom_stack_top_info( unsigned tid, unsigned *pc, unsigned *rpc ) const
{
    unsigned warp_id = tid/m_config->warp_size;
    m_simt_stack[warp_id]->get_pdom_stack_top_info(pc,rpc);
}

void shader_core_stats::print( FILE* fout ) const
{
	unsigned long long  thread_icount_uarch=0;
	unsigned long long  warp_icount_uarch=0;

    for(unsigned i=0; i < m_config->num_shader(); i++) {
        thread_icount_uarch += m_num_sim_insn[i];
        warp_icount_uarch += m_num_sim_winsn[i];
    }
    fprintf(fout,"gpgpu_n_tot_thrd_icount = %lld\n", thread_icount_uarch);
    fprintf(fout,"gpgpu_n_tot_w_icount = %lld\n", warp_icount_uarch);

    fprintf(fout,"gpgpu_n_stall_shd_mem = %d\n", gpgpu_n_stall_shd_mem );
    fprintf(fout,"gpgpu_n_mem_read_local = %d\n", gpgpu_n_mem_read_local);
    fprintf(fout,"gpgpu_n_mem_write_local = %d\n", gpgpu_n_mem_write_local);
    fprintf(fout,"gpgpu_n_mem_read_global = %d\n", gpgpu_n_mem_read_global);
    fprintf(fout,"gpgpu_n_mem_write_global = %d\n", gpgpu_n_mem_write_global);
    fprintf(fout,"gpgpu_n_mem_texture = %d\n", gpgpu_n_mem_texture);
    fprintf(fout,"gpgpu_n_mem_const = %d\n", gpgpu_n_mem_const);

   fprintf(fout, "gpgpu_n_load_insn  = %d\n", gpgpu_n_load_insn);
   fprintf(fout, "gpgpu_n_store_insn = %d\n", gpgpu_n_store_insn);
   fprintf(fout, "gpgpu_n_shmem_insn = %d\n", gpgpu_n_shmem_insn);
   fprintf(fout, "gpgpu_n_tex_insn = %d\n", gpgpu_n_tex_insn);
   fprintf(fout, "gpgpu_n_const_mem_insn = %d\n", gpgpu_n_const_insn);
   fprintf(fout, "gpgpu_n_param_mem_insn = %d\n", gpgpu_n_param_insn);

   fprintf(fout, "gpgpu_n_shmem_bkconflict = %d\n", gpgpu_n_shmem_bkconflict);
   fprintf(fout, "gpgpu_n_cache_bkconflict = %d\n", gpgpu_n_cache_bkconflict);   

   fprintf(fout, "gpgpu_n_intrawarp_mshr_merge = %d\n", gpgpu_n_intrawarp_mshr_merge);
   fprintf(fout, "gpgpu_n_cmem_portconflict = %d\n", gpgpu_n_cmem_portconflict);

   fprintf(fout, "gpgpu_stall_shd_mem[c_mem][bk_conf] = %d\n", gpu_stall_shd_mem_breakdown[C_MEM][BK_CONF]);
   fprintf(fout, "gpgpu_stall_shd_mem[c_mem][mshr_rc] = %d\n", gpu_stall_shd_mem_breakdown[C_MEM][MSHR_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[c_mem][icnt_rc] = %d\n", gpu_stall_shd_mem_breakdown[C_MEM][ICNT_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[c_mem][data_port_stall] = %d\n", gpu_stall_shd_mem_breakdown[C_MEM][DATA_PORT_STALL]);
   fprintf(fout, "gpgpu_stall_shd_mem[t_mem][mshr_rc] = %d\n", gpu_stall_shd_mem_breakdown[T_MEM][MSHR_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[t_mem][icnt_rc] = %d\n", gpu_stall_shd_mem_breakdown[T_MEM][ICNT_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[t_mem][data_port_stall] = %d\n", gpu_stall_shd_mem_breakdown[T_MEM][DATA_PORT_STALL]);
   fprintf(fout, "gpgpu_stall_shd_mem[s_mem][bk_conf] = %d\n", gpu_stall_shd_mem_breakdown[S_MEM][BK_CONF]);
   fprintf(fout, "gpgpu_stall_shd_mem[gl_mem][bk_conf] = %d\n", 
           gpu_stall_shd_mem_breakdown[G_MEM_LD][BK_CONF] + 
           gpu_stall_shd_mem_breakdown[G_MEM_ST][BK_CONF] + 
           gpu_stall_shd_mem_breakdown[L_MEM_LD][BK_CONF] + 
           gpu_stall_shd_mem_breakdown[L_MEM_ST][BK_CONF]   
           ); // coalescing stall at data cache 
   fprintf(fout, "gpgpu_stall_shd_mem[gl_mem][coal_stall] = %d\n", 
           gpu_stall_shd_mem_breakdown[G_MEM_LD][COAL_STALL] + 
           gpu_stall_shd_mem_breakdown[G_MEM_ST][COAL_STALL] + 
           gpu_stall_shd_mem_breakdown[L_MEM_LD][COAL_STALL] + 
           gpu_stall_shd_mem_breakdown[L_MEM_ST][COAL_STALL]    
           ); // coalescing stall + bank conflict at data cache 
   fprintf(fout, "gpgpu_stall_shd_mem[gl_mem][data_port_stall] = %d\n", 
           gpu_stall_shd_mem_breakdown[G_MEM_LD][DATA_PORT_STALL] + 
           gpu_stall_shd_mem_breakdown[G_MEM_ST][DATA_PORT_STALL] + 
           gpu_stall_shd_mem_breakdown[L_MEM_LD][DATA_PORT_STALL] + 
           gpu_stall_shd_mem_breakdown[L_MEM_ST][DATA_PORT_STALL]    
           ); // data port stall at data cache 
   fprintf(fout, "gpgpu_stall_shd_mem[g_mem_ld][mshr_rc] = %d\n", gpu_stall_shd_mem_breakdown[G_MEM_LD][MSHR_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[g_mem_ld][icnt_rc] = %d\n", gpu_stall_shd_mem_breakdown[G_MEM_LD][ICNT_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[g_mem_ld][wb_icnt_rc] = %d\n", gpu_stall_shd_mem_breakdown[G_MEM_LD][WB_ICNT_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[g_mem_ld][wb_rsrv_fail] = %d\n", gpu_stall_shd_mem_breakdown[G_MEM_LD][WB_CACHE_RSRV_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[g_mem_st][mshr_rc] = %d\n", gpu_stall_shd_mem_breakdown[G_MEM_ST][MSHR_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[g_mem_st][icnt_rc] = %d\n", gpu_stall_shd_mem_breakdown[G_MEM_ST][ICNT_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[g_mem_st][wb_icnt_rc] = %d\n", gpu_stall_shd_mem_breakdown[G_MEM_ST][WB_ICNT_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[g_mem_st][wb_rsrv_fail] = %d\n", gpu_stall_shd_mem_breakdown[G_MEM_ST][WB_CACHE_RSRV_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[l_mem_ld][mshr_rc] = %d\n", gpu_stall_shd_mem_breakdown[L_MEM_LD][MSHR_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[l_mem_ld][icnt_rc] = %d\n", gpu_stall_shd_mem_breakdown[L_MEM_LD][ICNT_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[l_mem_ld][wb_icnt_rc] = %d\n", gpu_stall_shd_mem_breakdown[L_MEM_LD][WB_ICNT_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[l_mem_ld][wb_rsrv_fail] = %d\n", gpu_stall_shd_mem_breakdown[L_MEM_LD][WB_CACHE_RSRV_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[l_mem_st][mshr_rc] = %d\n", gpu_stall_shd_mem_breakdown[L_MEM_ST][MSHR_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[l_mem_st][icnt_rc] = %d\n", gpu_stall_shd_mem_breakdown[L_MEM_ST][ICNT_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[l_mem_ld][wb_icnt_rc] = %d\n", gpu_stall_shd_mem_breakdown[L_MEM_ST][WB_ICNT_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[l_mem_ld][wb_rsrv_fail] = %d\n", gpu_stall_shd_mem_breakdown[L_MEM_ST][WB_CACHE_RSRV_FAIL]);

   fprintf(fout, "gpu_reg_bank_conflict_stalls = %d\n", gpu_reg_bank_conflict_stalls);

   fprintf(fout, "Warp Occupancy Distribution:\n");
   fprintf(fout, "Stall:%d\t", shader_cycle_distro[2]);
   fprintf(fout, "W0_Idle:%d\t", shader_cycle_distro[0]);
   fprintf(fout, "W0_Scoreboard:%d", shader_cycle_distro[1]);
   for (unsigned i = 3; i < m_config->warp_size + 3; i++) 
      fprintf(fout, "\tW%d:%d", i-2, shader_cycle_distro[i]);
   fprintf(fout, "\n");

   m_outgoing_traffic_stats->print(fout); 
   m_incoming_traffic_stats->print(fout); 
}

void shader_core_stats::event_warp_issued( unsigned s_id, unsigned warp_id, unsigned num_issued, unsigned dynamic_warp_id ) {
    assert( warp_id <= m_config->max_warps_per_shader );
    for ( unsigned i = 0; i < num_issued; ++i ) {
        if ( m_shader_dynamic_warp_issue_distro[ s_id ].size() <= dynamic_warp_id ) {
            m_shader_dynamic_warp_issue_distro[ s_id ].resize(dynamic_warp_id + 1);
        }
        ++m_shader_dynamic_warp_issue_distro[ s_id ][ dynamic_warp_id ];
        if ( m_shader_warp_slot_issue_distro[ s_id ].size() <= warp_id ) {
            m_shader_warp_slot_issue_distro[ s_id ].resize(warp_id + 1);
        }
        ++m_shader_warp_slot_issue_distro[ s_id ][ warp_id ];
    }
}

void shader_core_stats::visualizer_print( gzFile visualizer_file )
{
    // warp divergence breakdown
    gzprintf(visualizer_file, "WarpDivergenceBreakdown:");
    unsigned int total=0;
    unsigned int cf = (m_config->gpgpu_warpdistro_shader==-1)?m_config->num_shader():1;
    gzprintf(visualizer_file, " %d", (shader_cycle_distro[0] - last_shader_cycle_distro[0]) / cf );
    gzprintf(visualizer_file, " %d", (shader_cycle_distro[1] - last_shader_cycle_distro[1]) / cf );
    gzprintf(visualizer_file, " %d", (shader_cycle_distro[2] - last_shader_cycle_distro[2]) / cf );
    for (unsigned i=0; i<m_config->warp_size+3; i++) {
       if ( i>=3 ) {
          total += (shader_cycle_distro[i] - last_shader_cycle_distro[i]);
          if ( ((i-3) % (m_config->warp_size/8)) == ((m_config->warp_size/8)-1) ) {
             gzprintf(visualizer_file, " %d", total / cf );
             total=0;
          }
       }
       last_shader_cycle_distro[i] = shader_cycle_distro[i];
    }
    gzprintf(visualizer_file,"\n");

    // warp issue breakdown
    unsigned sid = m_config->gpgpu_warp_issue_shader;
    unsigned count = 0;
    unsigned warp_id_issued_sum = 0;
    gzprintf(visualizer_file, "WarpIssueSlotBreakdown:");
    if(m_shader_warp_slot_issue_distro[sid].size() > 0){
        for ( std::vector<unsigned>::const_iterator iter = m_shader_warp_slot_issue_distro[ sid ].begin();
              iter != m_shader_warp_slot_issue_distro[ sid ].end(); iter++, count++ ) {
            unsigned diff = count < m_last_shader_warp_slot_issue_distro.size() ?
                            *iter - m_last_shader_warp_slot_issue_distro[ count ] :
                            *iter;
            gzprintf( visualizer_file, " %d", diff );
            warp_id_issued_sum += diff;
        }
        m_last_shader_warp_slot_issue_distro = m_shader_warp_slot_issue_distro[ sid ];
    }else{
        gzprintf( visualizer_file, " 0");
    }
    gzprintf(visualizer_file,"\n");

    #define DYNAMIC_WARP_PRINT_RESOLUTION 32
    unsigned total_issued_this_resolution = 0;
    unsigned dynamic_id_issued_sum = 0;
    count = 0;
    gzprintf(visualizer_file, "WarpIssueDynamicIdBreakdown:");
    if(m_shader_dynamic_warp_issue_distro[sid].size() > 0){
        for ( std::vector<unsigned>::const_iterator iter = m_shader_dynamic_warp_issue_distro[ sid ].begin();
              iter != m_shader_dynamic_warp_issue_distro[ sid ].end(); iter++, count++ ) {
            unsigned diff = count < m_last_shader_dynamic_warp_issue_distro.size() ?
                            *iter - m_last_shader_dynamic_warp_issue_distro[ count ] :
                            *iter;
            total_issued_this_resolution += diff;
            if ( ( count + 1 ) % DYNAMIC_WARP_PRINT_RESOLUTION == 0 ) {
                gzprintf( visualizer_file, " %d", total_issued_this_resolution );
                dynamic_id_issued_sum += total_issued_this_resolution;
                total_issued_this_resolution = 0;
            }
        }
        if ( count % DYNAMIC_WARP_PRINT_RESOLUTION != 0 ) {
            gzprintf( visualizer_file, " %d", total_issued_this_resolution );
            dynamic_id_issued_sum += total_issued_this_resolution;
        }
        m_last_shader_dynamic_warp_issue_distro = m_shader_dynamic_warp_issue_distro[ sid ];
        assert( warp_id_issued_sum == dynamic_id_issued_sum );
    }else{
        gzprintf( visualizer_file, " 0");
    }
    gzprintf(visualizer_file,"\n");

    // overall cache miss rates
    gzprintf(visualizer_file, "gpgpu_n_cache_bkconflict: %d\n", gpgpu_n_cache_bkconflict);
    gzprintf(visualizer_file, "gpgpu_n_shmem_bkconflict: %d\n", gpgpu_n_shmem_bkconflict);     


   // instruction count per shader core
   gzprintf(visualizer_file, "shaderinsncount:  ");
   for (unsigned i=0;i<m_config->num_shader();i++) 
      gzprintf(visualizer_file, "%u ", m_num_sim_insn[i] );
   gzprintf(visualizer_file, "\n");
   // warp instruction count per shader core
   gzprintf(visualizer_file, "shaderwarpinsncount:  ");
   for (unsigned i=0;i<m_config->num_shader();i++)
      gzprintf(visualizer_file, "%u ", m_num_sim_winsn[i] );
   gzprintf(visualizer_file, "\n");
   // warp divergence per shader core
   gzprintf(visualizer_file, "shaderwarpdiv: ");
   for (unsigned i=0;i<m_config->num_shader();i++) 
      gzprintf(visualizer_file, "%u ", m_n_diverge[i] );
   gzprintf(visualizer_file, "\n");
}

#define PROGRAM_MEM_START 0xF0000000 /* should be distinct from other memory spaces... 
                                        check ptx_ir.h to verify this does not overlap 
                                        other memory spaces */
void shader_core_ctx::decode()
{
    if( m_inst_fetch_buffer.m_valid ) {
        // Decode instructions and place them into ibuffer
        address_type base_ibuf_pc = m_inst_fetch_buffer.m_pc;
        address_type end_ibuf_pc = base_ibuf_pc + m_inst_fetch_buffer.m_nbytes;
        unsigned ibuf_pos = 0;
        for (address_type tpc = base_ibuf_pc; tpc < end_ibuf_pc;) {
            const warp_inst_t* new_pI = ptx_fetch_inst(tpc);
            if (new_pI) {
                assert(ibuf_pos < m_config->gpgpu_fetch_decode_width);
                m_warp[m_inst_fetch_buffer.m_warp_id].ibuffer_fill(ibuf_pos, new_pI);
                //if (m_sid==0) {
                //    printf("MARIA adding pc %d to instr buffer on warp %d core %d \n",
                //        tpc, m_inst_fetch_buffer.m_warp_id, m_sid);
                //}
                m_warp[m_inst_fetch_buffer.m_warp_id].inc_inst_in_pipeline();
                m_stats->m_num_decoded_insn[m_sid]++;
                if (new_pI->oprnd_type == INT_OP){
                    m_stats->m_num_INTdecoded_insn[m_sid]++;
                } else if(new_pI->oprnd_type == FP_OP) {
                    m_stats->m_num_FPdecoded_insn[m_sid]++;
                }
                tpc += new_pI->isize;
                ibuf_pos++;
            } else {
                // Advance tpc pointer to at least the end of the fetch size
                tpc += m_inst_fetch_buffer.m_nbytes;
            }
        }
        assert(ibuf_pos > 0);
        m_inst_fetch_buffer.m_valid = false;
    }
}

bool shader_core_ctx::EventIsRunning() {
   return !_edgeEventWarpIds.empty();
}

bool shader_core_ctx::PreemptionInProgress() {
   return (_edgeIntState != IDLE && _edgeWid != -1);
}

bool shader_core_ctx::EventIsRunningOnWarp(unsigned wid) {
    for (auto it = _edgeSaveStateList.begin(); it != _edgeSaveStateList.end(); it++) {
        if (it->second->wid() == wid) 
            return true;
    }
    return false;
}

void shader_core_ctx::fetch()
{
    if (( m_gpu->EventIsRunning() && !EventIsRunning() && !PreemptionInProgress() ) && m_config->_edgeEventPriority==3) {
        //EDGE_DPRINT(EdgeDebug, "Event is running on gpu - not fetching any other warps on SM %d \n", get_sid());
        return;
    }
    if( !m_inst_fetch_buffer.m_valid ) {

        // EDGE: Interrupt warp stuff
        bool intWarpVisited = true;
        unsigned EdgeWarpId = _edgeWid;
        //highest prio - warp that currently being preempted, then the event warp (old and new modes)
        bool iWarpRunningInternal = (_iWarpRunning || !m_config->_edgeRunISR && EventIsRunning() || PreemptionInProgress());
        //if preemption is in progress, victim warp should get highest priority. 
        //otherwise, oldest event warp is prioratized
        if (!m_config->_edgeRunISR && !PreemptionInProgress() && EventIsRunning()) {
                EdgeWarpId = _edgeEventWarpIds.front(); //the oldest event warp
        }
        
        if( iWarpRunningInternal && m_config->_edgeIntFetchPriority ) {
            if (m_config->_edgeRunISR) {
                assert( !_edgeDoFlush ); // If interrupt is running now, shouldn't be flushing...
                assert( EdgeWarpId != -1 ); // Should also have a valid interrupt warp id
            }
            intWarpVisited = false;
            bool new_warp = (EdgeWarpId != _edgeFetchLastChosenWarpId);
            _edgeFetchLastChosenWarpId = EdgeWarpId;
            if (new_warp) {
                //EDGE_DPRINT(EdgeDebug, "%lld: FETCH prioritize event warp %d on SM %d \n", gpu_sim_cycle, EdgeWarpId, m_sid);
            }
        }

        // find an active warp with space in instruction buffer that is not already waiting on a cache miss
        // and get next 1-2 instructions from i-cache...

        for( unsigned i=0; i < m_config->max_warps_per_shader; i++ ) {
            unsigned warp_id = (m_last_warp_fetched+1+i) % (m_config->max_warps_per_shader);

            // First prioritize an interrupt warp if neccessary
            if( iWarpRunningInternal && m_config->_edgeIntFetchPriority ) {
                // Interrupt warp is running. Now check if we need to fetch for it
                if( !intWarpVisited ) {
                    warp_id = EdgeWarpId;
                    intWarpVisited = true;
                } else {
                    if ( iWarpRunningInternal && m_config->_edgeEventPriority==3) {
                        //EDGE_DPRINT(EdgeDebug, "Event warp %d is running on SM - not scheduling any other warps on this SM \n", warp_id, m_shader->get_sid());
                        break;
                    }
                    if( warp_id == EdgeWarpId ) { // We already visited the int warp, skip it
                        warp_id = (warp_id+1) % (m_config->max_warps_per_shader);
                    }
                }
            }

            // this code checks if this warp has finished executing and can be reclaimed
            
            bool did_exit=false;
            kernel_info_t* exit_kernel = NULL;
            if( m_warp[warp_id].hardware_done() && !m_scoreboard->pendingWrites(warp_id) && !m_warp[warp_id].done_exit() ) {
                
                for( unsigned t=0; t<m_config->warp_size;t++) {
                    unsigned tid=warp_id*m_config->warp_size+t;
                    if( m_threadState[tid].m_active == true ) {
                        m_threadState[tid].m_active = false; 
                        unsigned cta_id = m_warp[warp_id].get_cta_id();
                        
                        m_not_completed -= 1;
                        
                        // CDP - Concurrent kernrels
                        exit_kernel = &(m_thread[tid]->get_kernel());
                        register_cta_thread_exit(cta_id, &(m_thread[tid]->get_kernel())); 
                        
                        m_active_threads.reset(tid);
                        assert( m_thread[tid]!= NULL );
                        did_exit=true;
                    }
                }

                if( did_exit ) {
                    m_warp[warp_id].set_done_exit();

                    // EDGE: Handle the interrupt warp completing
                    if( m_warp[warp_id].isIntWarp() && m_config->_edgeRunISR) {
                        assert( _iKernel && !m_cta_status[m_warp[warp_id].get_cta_id()] );
                        clearIntSignal(); // Reset ISR state and clear interrupt signal
                    } else {
                        // EDGE: Update completion stats - only do for non interrupt warp
                        _warpIntStatsVector[warp_id].back()->_endCycle = gpu_sim_cycle;
                    }
                }
            }

            // EDGE FIXME: It's now possible that we've selected a warp to interrupt that hasn't finished yet, BUT
            // while flushing the instructions from the pipeline, does finish. Hence, no longer able to complete if
            // we skip fetching this warp before checking the ending conditions. Now, the flush needs to check if the 
            // warp has "done_exit()", such that we can record this warp as free warp.  
            // EDGE: Stop fetching instructions for the interrupt ID to flush
            //if (m_sid==4) {
            //    printf("MARIA_DEBUG m_sid=%d, warp_id=%d, done_exit=%d, EdgeWarpId=%d, _edgeDoFlush=%d inst_in_pipe=%d \n", 
            //        m_sid, warp_id, m_warp[warp_id].done_exit(), EdgeWarpId, _edgeDoFlush, m_warp[warp_id].inst_in_pipeline());
            //}
            if( _edgeDoFlush ) {
                if( warp_id == _edgeWid )
                    continue;
            }


            // this code fetches instructions from the i-cache or generates memory requests
            if( !m_warp[warp_id].functional_done() && !m_warp[warp_id].imiss_pending() && m_warp[warp_id].ibuffer_empty() ) {
                address_type pc  = m_warp[warp_id].get_pc();
                //if (m_sid==5 && warp_id==20) {
                //    printf("MARIA fetching instr for core %d on warp %d. pc=%d \n", m_sid, warp_id, pc);
                //}
                
                // Use the instruction base virtual address specified in
                // the kernel that is currently scheduled to this shader
               
                // EDGE
                //address_type ppc = pc + m_kernel->get_inst_base_vaddr();
                unsigned cid = m_warp[warp_id].get_cta_id(); 
                assert(cid<m_config->max_cta_per_core);
                
                
                assert(_CTAKernelMap[cid]);
                kernel_info_t* k = _CTAKernelMap[cid];
                address_type ppc = pc + k->get_inst_base_vaddr();
                //printf("MARIA DEBUG looking for CTA %d warp %d core %d. kernel %s \n",
                //        cid, warp_id,  m_sid, k->name().c_str());


                // HACK: This assumes that instructions are 8B each
                unsigned nbytes = m_config->gpgpu_fetch_decode_width * 8;
                unsigned offset_in_block = ppc & (m_config->m_L1I_config.get_line_sz()-1);
                if( (offset_in_block+nbytes) > m_config->m_L1I_config.get_line_sz() )
                    nbytes = (m_config->m_L1I_config.get_line_sz()-offset_in_block);

                // TODO: replace with use of allocator
                // mem_fetch *mf = m_mem_fetch_allocator->alloc()
                // HACK: This access will get sent into gem5, so the control
                // header size must be zero, since gem5 packets will assess
                // control header sizing
                mem_access_t acc(INST_ACC_R,ppc,nbytes,false);
                mem_fetch *mf = new mem_fetch(acc,
                                              NULL/*we don't have an instruction yet*/,
                                              0, // Control header size
                                              warp_id,
                                              m_sid,
                                              m_tpc,
                                              m_memory_config );
                std::list<cache_event> events;
                enum cache_request_status status = m_L1I->access( (new_addr_type)ppc, mf, gpu_sim_cycle+gpu_tot_sim_cycle,events);
                if( status == MISS ) {
                    m_warp[warp_id].set_imiss_pending();
                    m_warp[warp_id].set_last_fetch(gpu_sim_cycle);
                } else if( status == HIT ) {
                    m_inst_fetch_buffer = ifetch_buffer_t(pc,nbytes,warp_id);
                    m_warp[warp_id].set_last_fetch(gpu_sim_cycle);
                    delete mf;
                } else {
                    assert( status == RESERVATION_FAIL );
                    delete mf;
                }

                if( iWarpRunningInternal && m_config->_edgeIntFetchPriority ) {
                    if( warp_id != EdgeWarpId )
                        m_last_warp_fetched = warp_id;
                } else {
                    m_last_warp_fetched = warp_id;
                }


                break;
            }
            if (did_exit && m_warp[warp_id].isIntWarp() && !m_config->_edgeRunISR) {
                NewEdgeDoOneKernelCompletion(exit_kernel, m_warp[warp_id].get_cta_id()); //restore hw state when required
            }
            if (did_exit && exit_kernel->isEventKernel() && !_edgeEventWarpIds.empty()) {
                //std::remove(_edgeEventWarpIds.begin(), _edgeEventWarpIds.end() , warp_id); 
                _edgeEventWarpIds.erase(std::find(_edgeEventWarpIds.begin(), _edgeEventWarpIds.end(), warp_id));
                EDGE_DPRINT(EdgeDebug, "%lld: removing warp %d from list of event warps on SM %d \n", gpu_sim_cycle, warp_id, m_sid);
                //if (!_edgeEventWarpIds.empty()) {
                //    printf("_edgeEventWarpIds.front = %d\n", _edgeEventWarpIds.front());
                //    assert(_edgeEventWarpIds.empty());
                //}
            }
        }
    }

    m_L1I->cycle();

    if( m_L1I->access_ready() ) {
        mem_fetch *mf = m_L1I->next_access();
        m_warp[mf->get_wid()].clear_imiss_pending();

        kernel_info_t* k = getWarpKernel( mf->get_wid() ); 
        k->stats()._nWarpInstStalls++; 
        k->stats()._nInstStallCycles += ( gpu_sim_cycle - m_warp[mf->get_wid()].get_last_fetch() );

        delete mf;
    }
}

void shader_core_ctx::func_exec_inst( warp_inst_t &inst )
{
    execute_warp_inst_t(inst);
    if( inst.is_load() || inst.is_store() )
        inst.generate_mem_accesses();
}

// EDGE: Special barrier op
unsigned shader_core_ctx::barrierType(const warp_inst_t* inst)
{
    int warpId = inst->warp_id();
    int threadId = warpId * m_config->warp_size;
    ptx_thread_info* thread = m_thread[threadId];
    const ptx_instruction* ptxInst = thread->func_info()->get_instruction(inst->pc);

    return thread->get_operand_value(ptxInst->dst(), ptxInst->dst(), U32_TYPE, thread, 1);
}

void shader_core_ctx::warp_reaches_barrier(warp_inst_t &inst) {
    
    unsigned bt = barrierType(&inst);
    unsigned warpId = inst.warp_id();
    int ctaId = m_warp[warpId].get_cta_id();

    switch( bt ) {
        case 0:
            // Regular barrier instruction (bar.sync 0;)
            if( _edgeIntState == IWARP_RUNNING ) 
                assert( warpId != _edgeWid ); // Single warp ISR should never hit a barrier
            m_barriers.warp_reaches_barrier(ctaId, warpId);
            break;

        case 1:
            // Special EDGE barrier. Warp waiting for release flag
            if( m_barriers.anyAtBarrier(ctaId) ) {
                assert( m_barriers.isEdgeBarrier(ctaId) );
            } else {
                m_barriers.setEdgeBarrier(ctaId);
            }        
            m_barriers.warp_reaches_barrier(ctaId, warpId);
            _edgeNumEdgeBarriers++;

            break;

        case 2:
            // FIXME: Simply release the barrier
            m_barriers.releaseAllEdgeBarrier();
            _edgeNumEdgeReleaseBarriers++;

            break;
        default:
            printf("EDGE ERROR: Unknown barrier type: %d\n", bt);
            abort();
    }
}

void shader_core_ctx::issue_warp( register_set& pipe_reg_set, const warp_inst_t* next_inst, const active_mask_t &active_mask, unsigned warp_id )
{
    warp_inst_t** pipe_reg = pipe_reg_set.get_free();
    assert(pipe_reg);
    
    m_warp[warp_id].ibuffer_free();
    assert(next_inst->valid());
    **pipe_reg = *next_inst; // static instruction information
    (*pipe_reg)->issue( active_mask, warp_id, gpu_tot_sim_cycle + gpu_sim_cycle, m_warp[warp_id].get_dynamic_warp_id() ); // dynamic instruction information
    m_stats->shader_cycle_distro[2+(*pipe_reg)->active_count()]++;
    func_exec_inst( **pipe_reg );

    m_scoreboard->reserveRegisters(*pipe_reg);
    m_warp[warp_id].set_next_pc(next_inst->pc + next_inst->isize);

    if( (*pipe_reg)->is_load() && (*pipe_reg)->space.get_type() != shared_space ) {
        // EDGE: Increase pending loads in pipeline
        warpIncPendingLoads(warp_id);

        // Push the load into the inflight load queue. Used to replay instructions if
        // needed. 
        m_warp[warp_id].startLoad(**pipe_reg, (*pipe_reg)->pc, m_simt_stack[warp_id]);
    }

    updateSIMTStack(warp_id,*pipe_reg);

    // EDGE: Increase number of warp isntructions issued by this kernel
    getWarpKernel(warp_id)->stats()._nWarpInstIssued++; // Always start from the bottom thread of a warp, so a partial warp should never return a NULL kernel.
}

void shader_core_ctx::issue(){
    if (m_config->gpgpu_cycle_sched_prio) {
        m_scheduler_prio = (m_scheduler_prio + 1) % schedulers.size();
    }
    //really is issue;
    for (unsigned i = 0; i < schedulers.size(); i++) {
        unsigned sched_index = (m_scheduler_prio + i) % schedulers.size();
        schedulers[sched_index]->cycle();
    }
}

shd_warp_t& scheduler_unit::warp(int i){
    return (*m_warp)[i];
}


/**
 * A general function to order things in a Loose Round Robin way. The simplist use of this
 * function would be to implement a loose RR scheduler between all the warps assigned to this core.
 * A more sophisticated usage would be to order a set of "fetch groups" in a RR fashion.
 * In the first case, the templated class variable would be a simple unsigned int representing the
 * warp_id.  In the 2lvl case, T could be a struct or a list representing a set of warp_ids.
 * @param result_list: The resultant list the caller wants returned.  This list is cleared and then populated
 *                     in a loose round robin way
 * @param input_list: The list of things that should be put into the result_list. For a simple scheduler
 *                    this can simply be the m_supervised_warps list.
 * @param last_issued_from_input:  An iterator pointing the last member in the input_list that issued.
 *                                 Since this function orders in a RR fashion, the object pointed
 *                                 to by this iterator will be last in the prioritization list
 * @param num_warps_to_add: The number of warps you want the scheudler to pick between this cycle.
 *                          Normally, this will be all the warps availible on the core, i.e.
 *                          m_supervised_warps.size(). However, a more sophisticated scheduler may wish to
 *                          limit this number. If the number if < m_supervised_warps.size(), then only
 *                          the warps with highest RR priority will be placed in the result_list.
 */
template < class T >
void scheduler_unit::order_lrr( std::vector< T >& result_list,
                                const typename std::vector< T >& input_list,
                                const typename std::vector< T >::const_iterator& last_issued_from_input,
                                unsigned num_warps_to_add )
{
    assert( num_warps_to_add <= input_list.size() );
    result_list.clear();
    typename std::vector< T >::const_iterator iter
        = ( last_issued_from_input ==  input_list.end() ) ? input_list.begin()
                                                          : last_issued_from_input + 1;

    for ( unsigned count = 0;
          count < num_warps_to_add;
          ++iter, ++count) {
        if ( iter ==  input_list.end() ) {
            iter = input_list.begin();
        }
        result_list.push_back( *iter );
    }
}

/**
 * A general function to order things in an priority-based way.
 * The core usage of the function is similar to order_lrr.
 * The explanation of the additional parameters (beyond order_lrr) explains the further extensions.
 * @param ordering: An enum that determines how the age function will be treated in prioritization
 *                  see the definition of OrderingType.
 * @param priority_function: This function is used to sort the input_list.  It is passed to stl::sort as
 *                           the sorting fucntion. So, if you wanted to sort a list of integer warp_ids
 *                           with the oldest warps having the most priority, then the priority_function
 *                           would compare the age of the two warps.
 */
template < class T >
void scheduler_unit::order_by_priority( std::vector< T >& result_list,
                                        const typename std::vector< T >& input_list,
                                        const typename std::vector< T >::const_iterator& last_issued_from_input,
                                        unsigned num_warps_to_add,
                                        OrderingType ordering,
                                        bool (*priority_func)(T lhs, T rhs) )
{
    assert( num_warps_to_add <= input_list.size() );
    result_list.clear();
    typename std::vector< T > temp = input_list;

    if ( ORDERING_GREEDY_THEN_PRIORITY_FUNC == ordering ) {
        T greedy_value = *last_issued_from_input;
        result_list.push_back( greedy_value );

        std::sort( temp.begin(), temp.end(), priority_func );
        typename std::vector< T >::iterator iter = temp.begin();
        for ( unsigned count = 0; count < num_warps_to_add; ++count, ++iter ) {
            if ( *iter != greedy_value ) {
                result_list.push_back( *iter );
            }
        }
    } else if ( ORDERED_PRIORITY_FUNC_ONLY == ordering ) {
        std::sort( temp.begin(), temp.end(), priority_func );
        typename std::vector< T >::iterator iter = temp.begin();
        for ( unsigned count = 0; count < num_warps_to_add; ++count, ++iter ) {
            result_list.push_back( *iter );
        }
    } else {
        fprintf( stderr, "Unknown ordering - %d\n", ordering );
        abort();
    }
}

void scheduler_unit::cycle()
{
    SCHED_DPRINTF( "scheduler_unit::cycle()\n" );
    bool valid_inst = false;  // there was one warp with a valid instruction to issue (didn't require flush due to control hazard)
    bool ready_inst = false;  // of the valid instructions, there was one not waiting for pending register writes
    bool issued_inst = false; // of these we issued one

    order_warps();
    //WarpVector updated_m_next_cycle_prioritized_warps;
    //if (m_shader->m_config->_edgeStopOtherWarpsInPreemptedSm && m_next_cycle_prioritized_warps[0]->isIntWarp()) {
    //    updated_m_next_cycle_prioritized_warps.push_back(m_next_cycle_prioritized_warps[0]);
    //} else {
    //    updated_m_next_cycle_prioritized_warps = m_next_cycle_prioritized_warps;
    //}

    if (( m_shader->get_gpu()->EventIsRunning() && !m_shader->EventIsRunning() && !m_shader->PreemptionInProgress() ) && m_shader->m_config->_edgeEventPriority==3) {
        //EDGE_DPRINT(EdgeDebug, "Event is running on gpu - not scheduling any other warps on SM %d \n", m_shader->get_sid());
        return;
    }
    for ( std::vector< shd_warp_t* >::const_iterator iter = m_next_cycle_prioritized_warps.begin();
          iter != m_next_cycle_prioritized_warps.end();
          iter++ ) {
        // Don't consider warps that are not yet valid
        if ( (*iter) == NULL || (*iter)->done_exit() ) {
            if (*iter!=NULL) {
                //printf("MARIA_DEBUG core %d warp %d done_exit not scheduled\n", get_sid(), (*iter)->get_warp_id());
            }
            continue;
        }
        SCHED_DPRINTF( "Testing (warp_id %u, dynamic_warp_id %u)\n",
                       (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id() );
        unsigned warp_id = (*iter)->get_warp_id();
        unsigned checked=0;
        unsigned issued=0;
        unsigned max_issue = m_shader->m_config->gpgpu_max_insn_issue_per_warp;

        //if neither event warp nor victim warp, stop scheduilng when P3
        if ( ( m_shader->m_config->_edgeEventPriority==3 || m_shader->m_config->_edgeStopOtherWarpsInPreemptedSm==1 ) && 
             ( m_shader->EventIsRunning() && !(*iter)->kernel->isEventKernel() ) && 
             ( m_shader->PreemptionInProgress() && warp_id != m_shader->edgeWid() ) ) {
            //EDGE_DPRINT(EdgeDebug, "Event warp %d is running on core %d - not scheduling any other warps on this SM \n", warp_id, m_shader->get_sid());
            break;
        }

        while( !warp(warp_id).waiting() && !warp(warp_id).ibuffer_empty() && (checked < max_issue) && (checked <= issued) && (issued < max_issue) ) {
            const warp_inst_t *pI = warp(warp_id).ibuffer_next_inst();
            bool valid = warp(warp_id).ibuffer_next_valid();
            bool warp_inst_issued = false;
            unsigned pc,rpc;
            m_simt_stack[warp_id]->get_pdom_stack_top_info(&pc,&rpc);
            SCHED_DPRINTF( "Warp (warp_id %u, dynamic_warp_id %u) has valid instruction (%s)\n",
                           (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id(),
                           ptx_get_insn_str( pc).c_str() );
            //if (m_sid == 4 && warp_id==0) {
            //    printf("MARIA schedule inst on core 4 warp 0 pc=%d\n", pc);
            //}
            //if (m_shader->m_config->_edgeStopOtherWarpsInPreemptedCTA && m_shader->EventIsRunning() && !(*iter)->isIntWarp()) {
            //    //EDGE_DPRINT(EdgeDebug, "Event warp %d is running on core %d - not scheduling any other warps on this SM \n",
            //    //warp_id, m_shader->get_sid());
            //    continue;
            //}
            
            if( pI ) {
                assert(valid);
                if( pc != pI->pc ) {
                    SCHED_DPRINTF( "Warp (warp_id %u, dynamic_warp_id %u) control hazard instruction flush\n",
                                   (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id() );
                    // control hazard
                    warp(warp_id).set_next_pc(pc);
                    warp(warp_id).ibuffer_flush();
                } else {
                    valid_inst = true;
                    if ( !m_scoreboard->checkCollision(warp_id, pI) ) {
                        SCHED_DPRINTF( "Warp (warp_id %u, dynamic_warp_id %u) passes scoreboard\n",
                                       (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id() );
                        ready_inst = true;
                        const active_mask_t &active_mask = m_simt_stack[warp_id]->get_active_mask();
                        assert( warp(warp_id).inst_in_pipeline() );
                        if ( (pI->op == LOAD_OP) || (pI->op == STORE_OP) || (pI->op == MEMORY_BARRIER_OP) || (pI->op == BARRIER_OP) ) {
                            if( m_mem_out->has_free() ) {
                                m_shader->issue_warp(*m_mem_out,pI,active_mask,warp_id);
                                issued++;
                                issued_inst=true;
                                warp_inst_issued = true;
                                if ( (pI->op == MEMORY_BARRIER_OP) || (pI->op == BARRIER_OP) ) {
                                    // Block this warp from issuing instructions
                                    // while completing memory fence operation
                                    // Note: This organization disallows a warp
                                    // from arriving at a bar.sync (BARRIER_OP)
                                    // until after the implicit (membar.cta)
                                    // fence is completed by the LSQ
                                    warp(warp_id).set_membar();
                                }
                            }
                        } else {
                            bool sp_pipe_avail = m_sp_out->has_free();
                            bool sfu_pipe_avail = m_sfu_out->has_free();
                            if( sp_pipe_avail && (pI->op != SFU_OP) ) {
                                // always prefer SP pipe for operations that can use both SP and SFU pipelines
                                m_shader->issue_warp(*m_sp_out,pI,active_mask,warp_id);
                                issued++;
                                issued_inst=true;
                                warp_inst_issued = true;
                            } else if ( (pI->op == SFU_OP) || (pI->op == ALU_SFU_OP) ) {
                                if( sfu_pipe_avail ) {
                                    m_shader->issue_warp(*m_sfu_out,pI,active_mask,warp_id);
                                    issued++;
                                    issued_inst=true;
                                    warp_inst_issued = true;
                                }
                            } 
                        }
                    } else {
                        SCHED_DPRINTF( "Warp (warp_id %u, dynamic_warp_id %u) fails scoreboard\n",
                                       (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id() );
                        if ((*iter)->isIntWarp()) {
                            //m_scoreboard->printContentsForWarp((*iter)->get_warp_id());
                            (*iter)->FailSbCnt++;
                            if ((*iter)->FailSbCnt % 100 == 0) {
                                //m_shader->display_pipeline(stdout,1,0);
                            }
                        }
                    }
                }
            } else if( valid ) {
               // this case can happen after a return instruction in diverged warp
               SCHED_DPRINTF( "Warp (warp_id %u, dynamic_warp_id %u) return from diverged warp flush\n",
                              (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id() );
               warp(warp_id).set_next_pc(pc);
               warp(warp_id).ibuffer_flush();
            }
            if(warp_inst_issued) {
                SCHED_DPRINTF( "Warp (warp_id %u, dynamic_warp_id %u) issued %u instructions\n",
                               (*iter)->get_warp_id(),
                               (*iter)->get_dynamic_warp_id(),
                               issued );
                (*iter)->FailSbCnt = 0;
                do_on_warp_issued( warp_id, issued, iter );
            }
            checked++;
        }
        if ( issued ) {
            // This might be a bit inefficient, but we need to maintain
            // two ordered list for proper scheduler execution.
            // We could remove the need for this loop by associating a
            // supervised_is index with each entry in the m_next_cycle_prioritized_warps
            // vector. For now, just run through until you find the right warp_id
            for ( std::vector< shd_warp_t* >::const_iterator supervised_iter = m_supervised_warps.begin();
                  supervised_iter != m_supervised_warps.end();
                  ++supervised_iter ) {
                if ( *iter == *supervised_iter ) {
                    m_last_supervised_issued = supervised_iter;
                }
            }
            break;
        }
        if ((*iter)->kernel == NULL) {
            EDGE_DPRINT(EdgeErr, "Warp %d on SM %d has a NULL kernel!!! \n", warp_id, m_shader->get_sid());
            abort();
        }
    }

    // issue stall statistics:
    if( !valid_inst ) 
        m_stats->shader_cycle_distro[0]++; // idle or control hazard
    else if( !ready_inst ) 
        m_stats->shader_cycle_distro[1]++; // waiting for RAW hazards (possibly due to memory) 
    else if( !issued_inst ) 
        m_stats->shader_cycle_distro[2]++; // pipeline stalled
}

void scheduler_unit::do_on_warp_issued( unsigned warp_id,
                                        unsigned num_issued,
                                        const std::vector< shd_warp_t* >::const_iterator& prioritized_iter )
{
    m_stats->event_warp_issued( m_shader->get_sid(),
                                warp_id,
                                num_issued,
                                warp(warp_id).get_dynamic_warp_id() );
    warp(warp_id).ibuffer_step();
}

bool scheduler_unit::sort_warps_by_oldest_dynamic_id(shd_warp_t* lhs, shd_warp_t* rhs)
{
    if (rhs && lhs) {
        if ( lhs->done_exit() || lhs->waiting() ) {
            return false;
        } else if ( rhs->done_exit() || rhs->waiting() ) {
            return true;
        } else {
            return lhs->get_dynamic_warp_id() < rhs->get_dynamic_warp_id();
        }
    } else {
        return lhs < rhs;
    }
}

// EDGE

void EdgeScheduler::getSupervisedWarps(WarpVector& dest) const
{
    for( WarpVector::const_iterator it = m_supervised_warps.begin(); 
            it != m_supervised_warps.end(); 
            ++it) {
        dest.push_back(*it);
    }
}

// GTO and then potentially schedule the high priority interrupt warp
void
EdgeScheduler::order_warps()
{
    // First sort by greedy than oldest
    order_by_priority( m_next_cycle_prioritized_warps,
                       m_supervised_warps,
                       m_last_supervised_issued,
                       m_supervised_warps.size(),
                       ORDERING_GREEDY_THEN_PRIORITY_FUNC,
                       scheduler_unit::sort_warps_by_oldest_dynamic_id );

    bool PreemptionInProgress = _victimWarpPriorityId != -1;
    bool EventIsRunning = ( m_shader->m_config->_edgeIntSchedulePriority && ( intSignal() || !m_shader->m_config->_edgeRunISR) && 
                            m_shader->EventIsRunning() && findIntWarp(false, m_shader->GetOldestEventWarpId())!=NULL );

    char* print_str;
    if (PreemptionInProgress && EventIsRunning) { //victim - first prio, then the event
        scheduleEventWarps();
        scheduleVictimWarp();
        print_str = "victim and event warps";
    }
    if (PreemptionInProgress) { //only victim
        scheduleVictimWarp();
        print_str = "victim warp";
    }
    if (EventIsRunning) { //only event
        scheduleEventWarps();
        print_str = "event warps";
    }
    
    if (PreemptionInProgress || EventIsRunning) {
        //EDGE_DPRINT(EdgeDebug, "%lld: SCHEDULE prioritizing %s on SM %d \n", gpu_sim_cycle, print_str, get_sid()); 
    }
    

    // Should we exclude any warps from scheduling?
    if( _excludeWarpId != -1 ) {
        for( WarpVector::iterator it = m_next_cycle_prioritized_warps.begin(); 
                it != m_next_cycle_prioritized_warps.end(); 
                ++it) {
            if( (*it)->get_warp_id() == _excludeWarpId ) {
                m_next_cycle_prioritized_warps.erase(it);
                break;
            }            
        }
    }

}

// Add the victim warp to the head of the priority queue for warp scheduling
void
EdgeScheduler::scheduleVictimWarp()
{
    WarpVector::iterator it;
    shd_warp_t* victimWarp = NULL;

    for( it = m_next_cycle_prioritized_warps.begin(); 
            it != m_next_cycle_prioritized_warps.end(); 
            ++it) {
        if( (*it)->get_warp_id() == _victimWarpPriorityId ) {
            victimWarp = *it;
            break;
        }
    }
  
    // Only one scheduler will have this warp
    if( victimWarp ) {
        // Remove the victim warp and add it to the front of the list
        m_next_cycle_prioritized_warps.erase(it);
        m_next_cycle_prioritized_warps.insert(m_next_cycle_prioritized_warps.begin(), victimWarp);
        //EDGE_DPRINT(EdgeDebug, "%lld: SCHEDULE prioritizing victim warp %d on SM %d \n", 
        //        gpu_sim_cycle, victimWarp->get_warp_id(), get_sid());
    }
}


// Add the iWarp to the head of the priority queue for warp scheduling
void EdgeScheduler::scheduleEventWarps() {
    for (int i=0; i<m_shader->_edgeEventWarpIds.size(); i++) {
        scheduleEventWarp(m_shader->_edgeEventWarpIds[i]);
    }    
}

void EdgeScheduler::scheduleEventWarp(int wid) {
    shd_warp_t* intWarp = findIntWarp(true, wid);    
    // This is only called if this scheduler is responsible for the iWarp, so it SHOULD be here. 
    if ( intWarp == NULL && wid!=m_shader->GetOldestEventWarpId() ) { //we check that the oldest event warp is part of m_next_cycle_prioritized_warps. other warps can still not be schedulable
        return;
    }
    assert( intWarp != NULL ); 
    if( !m_shader->delayIntWarp() ) {
        m_next_cycle_prioritized_warps.insert(m_next_cycle_prioritized_warps.begin(), intWarp);
        //EDGE_DPRINT(EdgeDebug, "%lld: SCHEDULE prioritizing event warp %d on SM %d \n", 
        //    gpu_sim_cycle, intWarp->get_warp_id(), get_sid());
    }
}

shd_warp_t* EdgeScheduler::findIntWarp(bool Erase, int warp_id) 
{
    WarpVector::iterator it;
    for( it = m_next_cycle_prioritized_warps.begin(); 
            it != m_next_cycle_prioritized_warps.end(); it++) {
        if( (m_shader->m_config->_edgeRunISR && (*it)->isIntWarp()) || 
            (!m_shader->m_config->_edgeRunISR && (*it)->get_warp_id() == warp_id) ) {
            shd_warp_t* intWarp = (*it);
            if (Erase) {
                m_next_cycle_prioritized_warps.erase(it);
            }
            return intWarp;
        }
    }
    return NULL;
}

void lrr_scheduler::order_warps()
{
    order_lrr( m_next_cycle_prioritized_warps,
               m_supervised_warps,
               m_last_supervised_issued,
               m_supervised_warps.size() );
}

void gto_scheduler::order_warps()
{
    order_by_priority( m_next_cycle_prioritized_warps,
                       m_supervised_warps,
                       m_last_supervised_issued,
                       m_supervised_warps.size(),
                       ORDERING_GREEDY_THEN_PRIORITY_FUNC,
                       scheduler_unit::sort_warps_by_oldest_dynamic_id );
}

void
two_level_active_scheduler::do_on_warp_issued( unsigned warp_id,
                                               unsigned num_issued,
                                               const std::vector< shd_warp_t* >::const_iterator& prioritized_iter )
{
    scheduler_unit::do_on_warp_issued( warp_id, num_issued, prioritized_iter );
    if ( SCHEDULER_PRIORITIZATION_LRR == m_inner_level_prioritization ) {
        //std::vector< shd_warp_t* > new_active; 
        WarpVector new_active; 
        order_lrr( new_active,
                   m_next_cycle_prioritized_warps,
                   prioritized_iter,
                   m_next_cycle_prioritized_warps.size() );
        m_next_cycle_prioritized_warps = new_active;
    } else {
        fprintf( stderr,
                 "Unimplemented m_inner_level_prioritization: %d\n",
                 m_inner_level_prioritization );
        abort();
    }
}

void two_level_active_scheduler::order_warps()
{
    //Move waiting warps to m_pending_warps
    unsigned num_demoted = 0;
    for (   std::vector< shd_warp_t* >::iterator iter = m_next_cycle_prioritized_warps.begin();
            iter != m_next_cycle_prioritized_warps.end(); ) {
        bool waiting = (*iter)->waiting();
        for (int i=0; i<4; i++){
            const warp_inst_t* inst = (*iter)->ibuffer_next_inst();
            //Is the instruction waiting on a long operation?
            if ( inst && inst->in[i] > 0 && this->m_scoreboard->islongop((*iter)->get_warp_id(), inst->in[i])){
                waiting = true;
            }
        }

        if( waiting ) {
            m_pending_warps.push_back(*iter);
            iter = m_next_cycle_prioritized_warps.erase(iter);
            SCHED_DPRINTF( "DEMOTED warp_id=%d, dynamic_warp_id=%d\n",
                           (*iter)->get_warp_id(),
                           (*iter)->get_dynamic_warp_id() );
            ++num_demoted;
        } else {
            ++iter;
        }
    }

    //If there is space in m_next_cycle_prioritized_warps, promote the next m_pending_warps
    unsigned num_promoted = 0;
    if ( SCHEDULER_PRIORITIZATION_SRR == m_outer_level_prioritization ) {
        while ( m_next_cycle_prioritized_warps.size() < m_max_active_warps ) {
            m_next_cycle_prioritized_warps.push_back(m_pending_warps.front());
            m_pending_warps.pop_front();
            SCHED_DPRINTF( "PROMOTED warp_id=%d, dynamic_warp_id=%d\n",
                           (m_next_cycle_prioritized_warps.back())->get_warp_id(),
                           (m_next_cycle_prioritized_warps.back())->get_dynamic_warp_id() );
            ++num_promoted;
        }
    } else {
        fprintf( stderr,
                 "Unimplemented m_outer_level_prioritization: %d\n",
                 m_outer_level_prioritization );
        abort();
    }
    assert( num_promoted == num_demoted );
}

swl_scheduler::swl_scheduler ( shader_core_stats* stats, shader_core_ctx* shader,
                               Scoreboard* scoreboard, simt_stack** simt,
                               std::vector<shd_warp_t>* warp,
                               register_set* sp_out,
                               register_set* sfu_out,
                               register_set* mem_out,
                               int id,
                               char* config_string )
    : scheduler_unit ( stats, shader, scoreboard, simt, warp, sp_out, sfu_out, mem_out, id )
{
    unsigned m_prioritization_readin;
    int ret = sscanf( config_string,
                      "warp_limiting:%d:%d",
                      &m_prioritization_readin,
                      &m_num_warps_to_limit
                     );
    assert( 2 == ret );
    m_prioritization = (scheduler_prioritization_type)m_prioritization_readin;
    // Currently only GTO is implemented
    assert( m_prioritization == SCHEDULER_PRIORITIZATION_GTO );
    assert( m_num_warps_to_limit <= shader->get_config()->max_warps_per_shader );
}

void swl_scheduler::order_warps()
{
    if ( SCHEDULER_PRIORITIZATION_GTO == m_prioritization ) {
        order_by_priority( m_next_cycle_prioritized_warps,
                           m_supervised_warps,
                           m_last_supervised_issued,
                           MIN( m_num_warps_to_limit, m_supervised_warps.size() ),
                           ORDERING_GREEDY_THEN_PRIORITY_FUNC,
                           scheduler_unit::sort_warps_by_oldest_dynamic_id );
    } else {
        fprintf(stderr, "swl_scheduler m_prioritization = %d\n", m_prioritization);
        abort();
    }
}

void shader_core_ctx::read_operands()
{
}

address_type coalesced_segment(address_type addr, unsigned segment_size_lg2bytes)
{
   return  (addr >> segment_size_lg2bytes);
}

/*
     // calculate the max cta count and cta size for local memory address mapping
    kernel_max_cta_per_shader = m_config->max_cta(kernel);
    unsigned int gpu_cta_size = kernel.threads_per_cta();
    kernel_padded_threads_per_cta = (gpu_cta_size%m_config->warp_size) ? 
        m_config->warp_size*((gpu_cta_size/m_config->warp_size)+1) : 
        gpu_cta_size;
*/

// Returns numbers of addresses in translated_addrs, each addr points to a 4B (32-bit) word
unsigned shader_core_ctx::translate_local_memaddr( address_type localaddr, unsigned tid, unsigned num_shader, unsigned datasize, new_addr_type* translated_addrs )
{
   // During functional execution, each thread sees its own memory space for local memory, but these
   // need to be mapped to a shared address space for timing simulation.  We do that mapping here.

   address_type thread_base = 0;
   unsigned max_concurrent_threads=0;
   if (m_config->gpgpu_local_mem_map) {
      // Dnew = D*N + T%nTpC + nTpC*C
      // N = nTpC*nCpS*nS (max concurent threads)
      // C = nS*K + S (hw cta number per gpu)
      // K = T/nTpC   (hw cta number per core)
      // D = data index
      // T = thread
      // nTpC = number of threads per CTA
      // nCpS = number of CTA per shader
      // 
      // for a given local memory address threads in a CTA map to contiguous addresses,
      // then distribute across memory space by CTAs from successive shader cores first, 
      // then by successive CTA in same shader core

       // EDGE: FIXME: Verify that whatever I just did actually works... Basically, I can't have
       // the calculated kernel_padded_<values> since the iWarp/iCTA/iKernel isn't part of anything...
       // so the address mapping is all off. Just set the mapping to the max, so any thread has its
       // allocated region ready to go.
       //
       // Fixed: This required modifying the Gem5 registration of local memory, such that all local memory
       // is allocated and registered for the legacy mapping to work. Only use 8KB per thread, instead of 
       // the potential 512KB, but this works for now. 
#if 0
      thread_base = 4*(kernel_padded_threads_per_cta * (m_sid + num_shader * (tid / kernel_padded_threads_per_cta))
                       + tid % kernel_padded_threads_per_cta); 
      max_concurrent_threads = kernel_padded_threads_per_cta * kernel_max_cta_per_shader * num_shader;
#else
      // EDGE FIXME:
      //thread_base = 4*((m_config->n_thread_per_shader+m_config->_nIntThreads) * m_sid + tid);
      //max_concurrent_threads = num_shader * (m_config->n_thread_per_shader+m_config->_nIntThreads);

      thread_base = 4*(m_config->n_thread_per_shader * m_sid + tid);
      max_concurrent_threads = num_shader * m_config->n_thread_per_shader;
#endif
   } else {
      panic("gem5-gpu does not work with legacy local mapping!\n");
      // legacy mapping that maps the same address in the local memory space of all threads 
      // to a single contiguous address region 
      thread_base = 4*(m_config->n_thread_per_shader * m_sid + tid);
      max_concurrent_threads = num_shader * m_config->n_thread_per_shader;
   }
   assert( thread_base < 4/*word size*/*max_concurrent_threads );

   // If requested datasize > 4B, split into multiple 4B accesses
   // otherwise do one sub-4 byte memory access
   unsigned num_accesses = 0;

   if(datasize >= 4) {
      // >4B access, split into 4B chunks
      assert(datasize%4 == 0);   // Must be a multiple of 4B
      num_accesses = datasize/4;
      assert(num_accesses <= MAX_ACCESSES_PER_INSN_PER_THREAD); // max 32B
      assert(localaddr%4 == 0); // Address must be 4B aligned - required if accessing 4B per request, otherwise access will overflow into next thread's space
      for(unsigned i=0; i<num_accesses; i++) {
          address_type local_word = localaddr/4 + i;
          address_type linear_address = local_word*max_concurrent_threads*4 + thread_base + get_gpu()->gem5CudaGPU->getLocalBaseVaddr();
          assert(linear_address < get_gpu()->gem5CudaGPU->getLocalBaseVaddr() + (LOCAL_MEM_SIZE_MAX * max_concurrent_threads));
          translated_addrs[i] = linear_address;
      }
   } else {
      // Sub-4B access, do only one access
      assert(datasize > 0);
      num_accesses = 1;
      address_type local_word = localaddr/4;
      address_type local_word_offset = localaddr%4;
      assert( (localaddr+datasize-1)/4  == local_word ); // Make sure access doesn't overflow into next 4B chunk
      address_type linear_address = local_word*max_concurrent_threads*4 + local_word_offset + thread_base + get_gpu()->gem5CudaGPU->getLocalBaseVaddr();
      assert(linear_address < get_gpu()->gem5CudaGPU->getLocalBaseVaddr() + (LOCAL_MEM_SIZE_MAX * max_concurrent_threads));
      translated_addrs[0] = linear_address;
   }
   return num_accesses;
}

/////////////////////////////////////////////////////////////////////////////////////////
int shader_core_ctx::test_res_bus(int latency){
	for(unsigned i=0; i<num_result_bus; i++){
		if(!m_result_bus[i]->test(latency)){return i;}
	}
	return -1;
}

void shader_core_ctx::execute()
{
	for(unsigned i=0; i<num_result_bus; i++){
		*(m_result_bus[i]) >>=1;
	}
    for( unsigned n=0; n < m_num_function_units; n++ ) {
        unsigned multiplier = m_fu[n]->clock_multiplier();
        for( unsigned c=0; c < multiplier; c++ ) 
            m_fu[n]->cycle();
        m_fu[n]->active_lanes_in_pipeline();
        enum pipeline_stage_name_t issue_port = m_issue_port[n];
        register_set& issue_inst = m_pipeline_reg[ issue_port ];
	    warp_inst_t** ready_reg = issue_inst.get_ready();

        if( issue_inst.has_ready() && m_fu[n]->can_issue( **ready_reg ) ) {
            unsigned warp_id = (*ready_reg)->warp_id();
            bool schedule_wb_now = !m_fu[n]->stallable();
            int resbus = -1;
            if( schedule_wb_now && (resbus=test_res_bus( (*ready_reg)->latency ))!=-1 ) {
                assert( (*ready_reg)->latency < MAX_ALU_LATENCY );
                m_result_bus[resbus]->set( (*ready_reg)->latency );
                m_fu[n]->issue( issue_inst );
            } else if( !schedule_wb_now ) {
                m_fu[n]->issue( issue_inst );
            } else {
                // stall issue (cannot reserve result bus)
            }
        }
    }
}

void ldst_unit::print_cache_stats( FILE *fp, unsigned& dl1_accesses, unsigned& dl1_misses ) {
   if( m_L1D ) {
       m_L1D->print( fp, dl1_accesses, dl1_misses );
   }
}

void ldst_unit::get_cache_stats(cache_stats &cs) {
    // Adds stats to 'cs' from each cache
    if(m_L1D)
        cs += m_L1D->get_stats();
    if(m_L1C)
        cs += m_L1C->get_stats();
    if(m_L1T)
        cs += m_L1T->get_stats();

}

void ldst_unit::get_L1D_sub_stats(struct cache_sub_stats &css) const{
    if(m_L1D)
        m_L1D->get_sub_stats(css);
}
void ldst_unit::get_L1C_sub_stats(struct cache_sub_stats &css) const{
    if(m_L1C)
        m_L1C->get_sub_stats(css);
}
void ldst_unit::get_L1T_sub_stats(struct cache_sub_stats &css) const{
    if(m_L1T)
        m_L1T->get_sub_stats(css);
}

// EDGE: Return the kernel that the current warp is running
kernel_info_t* shader_core_ctx::getWarpKernel(int warpId)
{
    return get_hwtid_kernel(warpId * m_config->warp_size);
}


void shader_core_ctx::warp_inst_complete(const warp_inst_t &inst)
{
   #if 0
      printf("[warp_inst_complete] uid=%u core=%u warp=%u pc=%#x @ time=%llu issued@%llu\n", 
             inst.get_uid(), m_sid, inst.warp_id(), inst.pc, gpu_tot_sim_cycle + gpu_sim_cycle, inst.get_issue_cycle()); 
   #endif
  if(inst.op_pipe==SP__OP)
	  m_stats->m_num_sp_committed[m_sid]++;
  else if(inst.op_pipe==SFU__OP)
	  m_stats->m_num_sfu_committed[m_sid]++;
  else if(inst.op_pipe==MEM__OP)
	  m_stats->m_num_mem_committed[m_sid]++;

  if(m_config->gpgpu_clock_gated_lanes==false)
	  m_stats->m_num_sim_insn[m_sid] += m_config->warp_size;
  else
	  m_stats->m_num_sim_insn[m_sid] += inst.active_count();

  m_stats->m_num_sim_winsn[m_sid]++;
  m_gpu->gpu_sim_insn += inst.active_count();
  inst.completed(gpu_tot_sim_cycle + gpu_sim_cycle);

    // EDGE: Update per kernel stats
    kernel_info_t* k = getWarpKernel(inst.warp_id()); 
    k->stats()._nInsn += inst.active_count();

}

void shader_core_ctx::writeback()
{
	unsigned max_committed_thread_instructions=m_config->warp_size * (m_config->pipe_widths[EX_WB]); //from the functional units
	m_stats->m_pipeline_duty_cycle[m_sid]=((float)(m_stats->m_num_sim_insn[m_sid]-m_stats->m_last_num_sim_insn[m_sid]))/max_committed_thread_instructions;

    m_stats->m_last_num_sim_insn[m_sid]=m_stats->m_num_sim_insn[m_sid];
    m_stats->m_last_num_sim_winsn[m_sid]=m_stats->m_num_sim_winsn[m_sid];

    warp_inst_t** preg = m_pipeline_reg[EX_WB].get_ready();
    warp_inst_t* pipe_reg = (preg==NULL)? NULL:*preg;
    while( preg and !pipe_reg->empty() ) {
    	/*
    	 * Right now, the writeback stage drains all waiting instructions
    	 * assuming there are enough ports in the register file or the
    	 * conflicts are resolved at issue.
    	 */
    	/*
    	 * The operand collector writeback can generally generate a stall
    	 * However, here, the pipelines should be un-stallable. This is
    	 * guaranteed because this is the first time the writeback function
    	 * is called after the operand collector's step function, which
    	 * resets the allocations. There is one case which could result in
    	 * the writeback function returning false (stall), which is when
    	 * an instruction tries to modify two registers (GPR and predicate)
    	 * To handle this case, we ignore the return value (thus allowing
    	 * no stalling).
    	 */
        m_operand_collector.writeback(*pipe_reg);
        unsigned warp_id = pipe_reg->warp_id();

        m_scoreboard->releaseRegisters( pipe_reg );
        m_warp[warp_id].dec_inst_in_pipeline();
        warp_inst_complete(*pipe_reg);
        m_gpu->gpu_sim_insn_last_update_sid = m_sid;
        m_gpu->gpu_sim_insn_last_update = gpu_sim_cycle;
        m_last_inst_gpu_sim_cycle = gpu_sim_cycle;
        m_last_inst_gpu_tot_sim_cycle = gpu_tot_sim_cycle;
        pipe_reg->clear();
        preg = m_pipeline_reg[EX_WB].get_ready();
        pipe_reg = (preg==NULL)? NULL:*preg;
    }
}

bool ldst_unit::shared_cycle( warp_inst_t &inst, mem_stage_stall_type &rc_fail, mem_stage_access_type &fail_type)
{
   if( inst.space.get_type() != shared_space )
       return true;

   if(inst.has_dispatch_delay()){
	   m_stats->gpgpu_n_shmem_bank_access[m_sid]++;
   }

   bool stall = inst.dispatch_delay();
   if( stall ) {
       fail_type = S_MEM;
       rc_fail = BK_CONF;
   } else 
       rc_fail = NO_RC_FAIL;
   return !stall; 
}

mem_stage_stall_type
ldst_unit::process_cache_access( cache_t* cache,
                                 new_addr_type address,
                                 warp_inst_t &inst,
                                 std::list<cache_event>& events,
                                 mem_fetch *mf,
                                 enum cache_request_status status )
{
    mem_stage_stall_type result = NO_RC_FAIL;
    bool write_sent = was_write_sent(events);
    bool read_sent = was_read_sent(events);
    if( write_sent ) 
        m_core->inc_store_req( inst.warp_id() );
    if ( status == HIT ) {
        // HACK for gem5-gpu: Reads should not be sent with hit status
        if (read_sent) assert( !read_sent );
        inst.accessq_pop_back();
        if ( inst.is_load() ) {
            for ( unsigned r=0; r < 4; r++)
                if (inst.out[r] > 0)
                    m_pending_writes[inst.warp_id()][inst.out[r]]--; 
        }
        if( !write_sent ) 
            delete mf;
    } else if ( status == RESERVATION_FAIL ) {
        result = COAL_STALL;
        // HACK for gem5-gpu: Reads should not be sent with reservation_fail status
        if (read_sent) assert( !read_sent );
        assert( !write_sent );
        delete mf;
    } else {
        assert( status == MISS || status == HIT_RESERVED );
        //inst.clear_active( access.get_warp_mask() ); // threads in mf writeback when mf returns
        inst.accessq_pop_back();
    }
    if( !inst.accessq_empty() )
        result = BK_CONF;
    return result;
}

mem_stage_stall_type ldst_unit::process_memory_access_queue( cache_t *cache, warp_inst_t &inst )
{
    mem_stage_stall_type result = NO_RC_FAIL;
    if( inst.accessq_empty() )
        return result;

    if( !cache->data_port_free() ) 
        return DATA_PORT_STALL; 

    //const mem_access_t &access = inst.accessq_back();
    mem_fetch *mf = m_mf_allocator->alloc(inst,inst.accessq_back());
    std::list<cache_event> events;
    enum cache_request_status status = cache->access(mf->get_addr(),mf,gpu_sim_cycle+gpu_tot_sim_cycle,events);
    return process_cache_access( cache, mf->get_addr(), inst, events, mf, status );
}

bool ldst_unit::constant_cycle( warp_inst_t &inst, mem_stage_stall_type &rc_fail, mem_stage_access_type &fail_type)
{
    // EDGE: Now removing parameter accesses from GPGPU-Sim
    if( inst.empty() || (inst.space.get_type() == param_space_kernel) )
        return true;
    if( inst.active_count() == 0 ) 
        return true;
    mem_stage_stall_type fail = process_memory_access_queue(m_L1C,inst);
    if (fail != NO_RC_FAIL){ 
        rc_fail = fail; //keep other fails if this didn't fail.
        fail_type = C_MEM;
        if (rc_fail == BK_CONF or rc_fail == COAL_STALL) {
            m_stats->gpgpu_n_cmem_portconflict++; //coal stalls aren't really a bank conflict, but this maintains previous behavior.
        }
    }
    return inst.accessq_empty(); //done if empty.
}

bool ldst_unit::texture_cycle( warp_inst_t &inst, mem_stage_stall_type &rc_fail, mem_stage_access_type &fail_type)
{
   if( inst.empty() || inst.space.get_type() != tex_space )
       return true;
   if( inst.active_count() == 0 ) 
       return true;
   mem_stage_stall_type fail = process_memory_access_queue(m_L1T,inst);
   if (fail != NO_RC_FAIL){ 
      rc_fail = fail; //keep other fails if this didn't fail.
      fail_type = T_MEM;
   }
   return inst.accessq_empty(); //done if empty.
}

bool ldst_unit::memory_cycle( warp_inst_t &inst, mem_stage_stall_type &stall_reason, mem_stage_access_type &access_type )
{
   if( inst.empty() || 
       ((inst.space.get_type() != global_space) &&
        (inst.space.get_type() != local_space) &&
        (inst.space.get_type() != param_space_local)) ) 
       return true;
   if( inst.active_count() == 0 ) 
       return true;
   assert( !inst.accessq_empty() );
   mem_stage_stall_type stall_cond = NO_RC_FAIL;
   const mem_access_t &access = inst.accessq_back();

   bool bypassL1D = false; 
   if ( CACHE_GLOBAL == inst.cache_op || (m_L1D == NULL) ) {
       bypassL1D = true; 
   } else if (inst.space.is_global()) { // global memory access 
       // skip L1 cache if the option is enabled
       if (m_core->get_config()->gmem_skip_L1D) 
           bypassL1D = true; 
   }

   if( bypassL1D ) {
       // bypass L1 cache
       unsigned control_size = inst.is_store() ? WRITE_PACKET_SIZE : READ_PACKET_SIZE;
       unsigned size = access.get_size() + control_size;
       if( m_icnt->full(size, inst.is_store() || inst.isatomic()) ) {
           stall_cond = ICNT_RC_FAIL;
       } else {
           mem_fetch *mf = m_mf_allocator->alloc(inst,access);
           m_icnt->push(mf);
           inst.accessq_pop_back();
           //inst.clear_active( access.get_warp_mask() );
           if( inst.is_load() ) { 
              for( unsigned r=0; r < 4; r++) 
                  if(inst.out[r] > 0) 
                      assert( m_pending_writes[inst.warp_id()][inst.out[r]] > 0 );
           } else if( inst.is_store() ) 
              m_core->inc_store_req( inst.warp_id() );
       }
   } else {
       assert( CACHE_UNDEFINED != inst.cache_op );
       stall_cond = process_memory_access_queue(m_L1D,inst);
   }
   if( !inst.accessq_empty() ) 
       stall_cond = COAL_STALL;
   if (stall_cond != NO_RC_FAIL) {
      stall_reason = stall_cond;
      bool iswrite = inst.is_store();
      if (inst.space.is_local()) 
         access_type = (iswrite)?L_MEM_ST:L_MEM_LD;
      else 
         access_type = (iswrite)?G_MEM_ST:G_MEM_LD;
   }
   return inst.accessq_empty(); 
}

bool ldst_unit::memory_cycle_gem5( warp_inst_t &inst, mem_stage_stall_type &stall_reason, mem_stage_access_type &access_type )
{

    if( inst.empty()) {
        return true;
    }
    //if (m_core->get_kernel()) {
    //    printf("MARIA DEBUG ldst op will be performed for kernel %s\n", m_core->get_kernel()->entry()->get_name().c_str());
   // }
    

    if (inst.space.get_type() != global_space &&
        inst.space.get_type() != const_space &&
        inst.space.get_type() != local_space &&
        inst.space.get_type() != param_space_kernel && /* EDGE: adding kernel parameter memory to Gem5 */ 
        inst.op != BARRIER_OP &&
        inst.op != MEMORY_BARRIER_OP) {
        return memory_cycle(inst, stall_reason, access_type);
    }
    if( inst.active_count() == 0 ) {
        return true;
    }


    // EDGE / CDP Concurrent Kernel - So, what do we do if one kernel is finishing on an SM and another kernel is still 
    // running? Are we flushing the kernel state? Currently Gem5 doesn't support receiving memory requests while a 
    // 'flush' operation is pending (flush is sent to Gem5 when kernel starts to complete). Currently, let's just 
    // stall the running kernels on this SM until the flush completes and the kernel fully completes. 
    
    if( m_core->kernelFinishing() ) {
        stall_reason = INT_STALL_LSQ_FLUSH;
        bool iswrite = inst.is_store();
        if (inst.space.is_local()) {
            access_type = (iswrite)?L_MEM_ST:L_MEM_LD;
        } else {
            access_type = (iswrite)?G_MEM_ST:G_MEM_LD;
        }
        return false;
    }

    // GPGPU-Sim should NOT have generated accesses for gem5-gpu requests
    assert( inst.accessq_empty() );
    mem_stage_stall_type stall_cond = NO_RC_FAIL;

    // Check if this request is an EDGE operation or not. If EDGE operation, 
    // perform actions here and don't send request to the Gem5 memory subsystem. 
    if( m_core->get_gpu()->EDGEOp(m_sid, inst) ) {
        if( inst.is_load() ) {
            for( unsigned r=0; r < 4; r++) {
                if(inst.out[r] > 0) {
                    m_pending_writes[inst.warp_id()][inst.out[r]]++;
                }
            }
        }
        return true;
    }
    
    if( inst.space.get_type() == param_space_kernel ) {
        // If this is the first time we've encountered this instruction (i.e., no stalls before), then
        // modify the parameter access to use the correct address (kernel_param_addr + offset). Otherwise,
        // this instruction has already been updated, but stalled so no need to update again.
        if( !inst.addrModified() ) { 
            inst.setAddrModified();
            for( unsigned i=0; i<m_config->warp_size; ++i ) {
                if( inst.active(i) ) {
                    //inst.set_addr(i, m_core->get_kernel()->getParamMem() + inst.get_addr(i));
                    int tid = (inst.warp_id() * m_config->warp_size) + i;
                   
                    kernel_info_t* k = m_core->get_hwtid_kernel(tid); //m_core->getWarp(inst.warp_id())->kernel;   //m_core->get_hwtid_kernel(tid);
                    assert(k);

                    //printf("MARIA DEBUG updating parammem addr for %p %s. = %p + %p \n", k, k->entry()->get_name().c_str(), k->getParamMem(), inst.get_addr(i));
                    
                    inst.set_addr(i, k->getParamMem() + inst.get_addr(i));
                }
            }
        }
    }

    //printf("MARIA DEBUG sending mem op with addr=%p type = %d kernel=%s  ", inst.get_addr(0), inst.space.get_type(), m_core->getWarpKernel(inst.warp_id())->entry()->get_name().c_str());
    //inst.print_insn(stdout);
    //printf("\n");
    bool rc_fail = m_core->get_gpu()->gem5CudaGPU->getCudaCore(m_core->m_sid)->executeMemOp(inst);

    if (rc_fail) {
        stall_cond = ICNT_RC_FAIL;
    } else {
        if( inst.is_load() ) {
            for( unsigned r=0; r < 4; r++) {
                if(inst.out[r] > 0) {
                    m_pending_writes[inst.warp_id()][inst.out[r]]++;
                }
            }
        }
    }

    if (stall_cond != NO_RC_FAIL) {
        stall_reason = stall_cond;
        bool iswrite = inst.is_store();
        if (inst.space.is_local()) {
            access_type = (iswrite)?L_MEM_ST:L_MEM_LD;
        } else {
            access_type = (iswrite)?G_MEM_ST:G_MEM_LD;
        }
        return false;
    }
    
    return true;
}

bool ldst_unit::response_buffer_full() const
{
    return m_response_fifo.size() >= m_config->ldst_unit_response_queue_size;
}

void ldst_unit::fill( mem_fetch *mf )
{
    mf->set_status(IN_SHADER_LDST_RESPONSE_FIFO,gpu_sim_cycle+gpu_tot_sim_cycle);
    m_response_fifo.push_back(mf);
}

void ldst_unit::flush(){
	// Flush L1D cache
	m_L1D->flush();
}

simd_function_unit::simd_function_unit( const shader_core_config *config )
{ 
    m_config=config;
    m_dispatch_reg = new warp_inst_t(config); 
}


sfu:: sfu(  register_set* result_port, const shader_core_config *config,shader_core_ctx *core  )
    : pipelined_simd_unit(result_port,config,config->max_sfu_latency,core)
{ 
    m_name = "SFU"; 
}

void sfu::issue( register_set& source_reg )
{
    warp_inst_t** ready_reg = source_reg.get_ready();
	//m_core->incexecstat((*ready_reg));

	(*ready_reg)->op_pipe=SFU__OP;
	m_core->incsfu_stat(m_core->get_config()->warp_size,(*ready_reg)->latency);
	pipelined_simd_unit::issue(source_reg);
}

void ldst_unit::active_lanes_in_pipeline(){
	unsigned active_count=pipelined_simd_unit::get_active_lanes_in_pipeline();
	assert(active_count<=m_core->get_config()->warp_size);
	m_core->incfumemactivelanes_stat(active_count);
}
void sp_unit::active_lanes_in_pipeline(){
	unsigned active_count=pipelined_simd_unit::get_active_lanes_in_pipeline();
	assert(active_count<=m_core->get_config()->warp_size);
	m_core->incspactivelanes_stat(active_count);
	m_core->incfuactivelanes_stat(active_count);
	m_core->incfumemactivelanes_stat(active_count);
}

void sfu::active_lanes_in_pipeline(){
	unsigned active_count=pipelined_simd_unit::get_active_lanes_in_pipeline();
	assert(active_count<=m_core->get_config()->warp_size);
	m_core->incsfuactivelanes_stat(active_count);
	m_core->incfuactivelanes_stat(active_count);
	m_core->incfumemactivelanes_stat(active_count);
}

sp_unit::sp_unit( register_set* result_port, const shader_core_config *config,shader_core_ctx *core)
    : pipelined_simd_unit(result_port,config,config->max_sp_latency,core)
{ 
    m_name = "SP "; 
}

void sp_unit :: issue(register_set& source_reg)
{
    warp_inst_t** ready_reg = source_reg.get_ready();
	//m_core->incexecstat((*ready_reg));
	(*ready_reg)->op_pipe=SP__OP;
	m_core->incsp_stat(m_core->get_config()->warp_size,(*ready_reg)->latency);
	pipelined_simd_unit::issue(source_reg);
}


pipelined_simd_unit::pipelined_simd_unit( register_set* result_port, const shader_core_config *config, unsigned max_latency,shader_core_ctx *core )
    : simd_function_unit(config) 
{
    m_result_port = result_port;
    m_pipeline_depth = max_latency;
    m_pipeline_reg = new warp_inst_t*[m_pipeline_depth];
    for( unsigned i=0; i < m_pipeline_depth; i++ ) 
	m_pipeline_reg[i] = new warp_inst_t( config );
    m_core=core;
}


void pipelined_simd_unit::issue( register_set& source_reg )
{
    //move_warp(m_dispatch_reg,source_reg);
    warp_inst_t** ready_reg = source_reg.get_ready();
	m_core->incexecstat((*ready_reg));
	//source_reg.move_out_to(m_dispatch_reg);
	simd_function_unit::issue(source_reg);
}

/*
    virtual void issue( register_set& source_reg )
    {
        //move_warp(m_dispatch_reg,source_reg);
        //source_reg.move_out_to(m_dispatch_reg);
        simd_function_unit::issue(source_reg);
    }
*/

void ldst_unit::init( mem_fetch_interface *icnt,
                      shader_core_mem_fetch_allocator *mf_allocator,
                      shader_core_ctx *core, 
                      opndcoll_rfu_t *operand_collector,
                      Scoreboard *scoreboard,
                      const shader_core_config *config,
                      const memory_config *mem_config,  
                      shader_core_stats *stats,
                      unsigned sid,
                      unsigned tpc )
{
    m_memory_config = mem_config;
    m_icnt = icnt;
    m_mf_allocator=mf_allocator;
    m_core = core;
    m_operand_collector = operand_collector;
    m_scoreboard = scoreboard;
    m_stats = stats;
    m_sid = sid;
    m_tpc = tpc;
    #define STRSIZE 1024
    char L1T_name[STRSIZE];
    char L1C_name[STRSIZE];
    snprintf(L1T_name, STRSIZE, "L1T_%03d", m_sid);
    snprintf(L1C_name, STRSIZE, "L1C_%03d", m_sid);
    m_L1T = new tex_cache(L1T_name,m_config->m_L1T_config,m_sid,get_shader_texture_cache_id(),icnt,IN_L1T_MISS_QUEUE,IN_SHADER_L1T_ROB);
    m_L1C = new read_only_cache(L1C_name,m_config->m_L1C_config,m_sid,get_shader_constant_cache_id(),icnt,IN_L1C_MISS_QUEUE);
    m_L1D = NULL;
    m_mem_rc = NO_RC_FAIL;
    m_num_writeback_clients=5; // = shared memory, global/local (uncached), L1D, L1T, L1C
    m_writeback_arb = 0;
    m_next_global=NULL;
    m_last_inst_gpu_sim_cycle=0;
    m_last_inst_gpu_tot_sim_cycle=0;
}


ldst_unit::ldst_unit( mem_fetch_interface *icnt,
                      shader_core_mem_fetch_allocator *mf_allocator,
                      shader_core_ctx *core, 
                      opndcoll_rfu_t *operand_collector,
                      Scoreboard *scoreboard,
                      const shader_core_config *config,
                      const memory_config *mem_config,  
                      shader_core_stats *stats,
                      unsigned sid,
                      unsigned tpc ) : pipelined_simd_unit(NULL,config,config->gpgpu_shmem_access_latency,core), m_next_wb(config)
{
    init( icnt,
          mf_allocator,
          core, 
          operand_collector,
          scoreboard,
          config, 
          mem_config,  
          stats, 
          sid,
          tpc );
    if( !m_config->m_L1D_config.disabled() ) {
        char L1D_name[STRSIZE];
        snprintf(L1D_name, STRSIZE, "L1D_%03d", m_sid);
        m_L1D = new l1_cache( L1D_name,
                              m_config->m_L1D_config,
                              m_sid,
                              get_shader_normal_cache_id(),
                              m_icnt,
                              m_mf_allocator,
                              IN_L1D_MISS_QUEUE );
    }
}

ldst_unit::ldst_unit( mem_fetch_interface *icnt,
                      shader_core_mem_fetch_allocator *mf_allocator,
                      shader_core_ctx *core, 
                      opndcoll_rfu_t *operand_collector,
                      Scoreboard *scoreboard,
                      const shader_core_config *config,
                      const memory_config *mem_config,  
                      shader_core_stats *stats,
                      unsigned sid,
                      unsigned tpc,
                      l1_cache* new_l1d_cache )
    : pipelined_simd_unit(NULL,config,config->gpgpu_shmem_access_latency,core), m_L1D(new_l1d_cache), m_next_wb(config)
{
    init( icnt,
          mf_allocator,
          core, 
          operand_collector,
          scoreboard,
          config, 
          mem_config,  
          stats, 
          sid,
          tpc );
}

void ldst_unit:: issue( register_set &reg_set )
{
	warp_inst_t* inst = *(reg_set.get_ready());

   // record how many pending register writes/memory accesses there are for this instruction
   assert(inst->empty() == false);
   if (inst->is_load() and inst->space.get_type() != shared_space) {
      unsigned warp_id = inst->warp_id();
      unsigned n_accesses = inst->accessq_count();
      for (unsigned r = 0; r < 4; r++) {
         unsigned reg_id = inst->out[r];
         if (reg_id > 0) {
            m_pending_writes[warp_id][reg_id] += n_accesses;
         }
      }
   }


	inst->op_pipe=MEM__OP;
	// stat collection
	m_core->mem_instruction_stats(*inst);
	m_core->incmem_stat(m_core->get_config()->warp_size,1);
	pipelined_simd_unit::issue(reg_set);
}

void ldst_unit::writeback()
{
    // process next instruction that is going to writeback
    if( !m_next_wb.empty() ) {
        // If you can write into the RF (bank conflict logic)
        if( m_operand_collector->writeback(m_next_wb) ) {
            bool insn_completed = false; 
            // for each output register (up to 4 for vectors)
            for( unsigned r=0; r < 4; r++ ) {
                if( m_next_wb.out[r] > 0 ) {
                    if( m_next_wb.space.get_type() != shared_space ) {

                        int warp_id = m_next_wb.warp_id();
                        if (m_pending_writes[warp_id][m_next_wb.out[r]] == 0) {
                            printf("MARIA DEBUG assertion failed for warp %d on core %d \n", warp_id, m_core->get_sid());
                            assert( m_pending_writes[warp_id][m_next_wb.out[r]] > 0 );
                        }
                        unsigned still_pending = --m_pending_writes[warp_id][m_next_wb.out[r]];
                        if( !still_pending ) {
                            m_pending_writes[warp_id].erase(m_next_wb.out[r]);
                            m_scoreboard->releaseRegister( warp_id, m_next_wb.out[r] );
                            insn_completed = true; 
                            
                            // EDGE: Decrease pending loads from pipeline
                            m_core->warpDecPendingLoads(warp_id);
                            assert( m_core->getWarp(warp_id)->findInFlightLoad( &m_next_wb ) );
                            m_core->getWarp(warp_id)->completeLoad( &m_next_wb );
                        }
                    } else { // shared 
                        m_scoreboard->releaseRegister( m_next_wb.warp_id(), m_next_wb.out[r] );
                        insn_completed = true; 
                    }
                }
            }
            if( insn_completed ) {
                m_core->warp_inst_complete(m_next_wb);
            }
            m_next_wb.clear();
            // signal gem5 that the wb hazard has cleared
            m_core->get_gpu()->gem5CudaGPU->getCudaCore(m_core->m_sid)->writebackClear();
            m_last_inst_gpu_sim_cycle = gpu_sim_cycle;
            m_last_inst_gpu_tot_sim_cycle = gpu_tot_sim_cycle;
        }
    }

    unsigned serviced_client = -1; 
    for( unsigned c = 0; m_next_wb.empty() && (c < m_num_writeback_clients); c++ ) {
        unsigned next_client = (c+m_writeback_arb)%m_num_writeback_clients;
        switch( next_client ) {
        case 0: // shared memory 
            if( !m_pipeline_reg[0]->empty() ) {
                m_next_wb = *m_pipeline_reg[0];
                if(m_next_wb.isatomic()) {
                    m_next_wb.do_atomic();
                    m_core->decrement_atomic_count(m_next_wb.warp_id(), m_next_wb.active_count());
                }
                m_core->dec_inst_in_pipeline(m_pipeline_reg[0]->warp_id());
                m_pipeline_reg[0]->clear();
                serviced_client = next_client; 
            }
            break;
        case 1: // texture response
            if( m_L1T->access_ready() ) {
                mem_fetch *mf = m_L1T->next_access();
                m_next_wb = mf->get_inst();
                delete mf;
                serviced_client = next_client; 
            }
            break;
        case 2: // const cache response
            if( m_L1C->access_ready() ) {
                mem_fetch *mf = m_L1C->next_access();
                m_next_wb = mf->get_inst();
                delete mf;
                serviced_client = next_client; 
            }
            break;
        case 3: // global/local
            if( m_next_global ) {
                panic("This should never execute in gem5-gpu! Writebacks from CudaCore must occur with writebackInst()!");
                m_next_wb = m_next_global->get_inst();
                if( m_next_global->isatomic() ) 
                    m_core->decrement_atomic_count(m_next_global->get_wid(),m_next_global->get_access_warp_mask().count());
                delete m_next_global;
                m_next_global = NULL;
                serviced_client = next_client; 
            }
            break;
        case 4: 
            if( m_L1D && m_L1D->access_ready() ) {
                mem_fetch *mf = m_L1D->next_access();
                m_next_wb = mf->get_inst();
                delete mf;
                serviced_client = next_client; 
            }
            break;
        default: abort();
        }
    }
    // update arbitration priority only if: 
    // 1. the writeback buffer was available 
    // 2. a client was serviced 
    if (serviced_client != (unsigned)-1) {
        m_writeback_arb = (serviced_client + 1) % m_num_writeback_clients; 
    }
}

bool shd_warp_t::isInFlightLoad(const warp_inst_t* inst) 
{
    if (_inFlightLoadQueue.size()==0) {
        return false;
    }
    for(std::list<PendingLoadInst>::iterator it = _inFlightLoadQueue.begin(); 
            it != _inFlightLoadQueue.end(); ++it ) {
        if( inst->get_uid() == (*it)._inst.get_uid() ) {
            return true;
        }
    }
    return false;
}

bool shd_warp_t::isDroppedLoad(const warp_inst_t* inst) 
{
    for(std::list<PendingLoadInst>::iterator it = _replayLoadQueue.begin(); 
            it != _replayLoadQueue.end(); ++it ) {
        if( inst->get_uid() == (*it)._inst.get_uid() ) {
            (*it)._loadReturned++;
            assert( (*it)._loadReturned <= (*it)._inst.active_count() );
            return true;
        }
    }
    return false;
}

bool shader_core_ctx::lateInFlightLoad(const warp_inst_t* inst)
{
    for( std::vector<PendingLoadInst>::iterator it = _lateInFlightLoads.begin();
            it != _lateInFlightLoads.end(); ++it ) {

        if( inst->get_uid() == (*it)._inst.get_uid() ) {
            (*it)._loadReturned++;
            assert( (*it)._loadReturned <= (*it)._inst.active_count() );
            if( (*it)._loadReturned == (*it)._inst.active_count() ) {
                // The full load has completed, so remove it!
                _lateInFlightLoads.erase(it);
            }
            return true;
        }
    }
    return false;
}

bool ldst_unit::dropLoad(const warp_inst_t* inst)
{
    // First things first. Check if this is a really old load that never finished...
    if( m_core->lateInFlightLoad( inst ) ) {
        // If we've already finished the interrupt, restored the warp, and replayed the loads, and THEN the load
        // comes back, check a global late queue to drop it. 
        return true;
    }

    bool wid_match;
    bool EdgeIsBusy;
    if (m_core->m_config->_edgeRunISR) {
        wid_match = m_core->_edgeWid == inst->warp_id();
        EdgeIsBusy = !m_core->_edgeIsFree;
    } else {
        wid_match = m_core->EventIsRunningOnWarp(inst->warp_id()) || (!m_core->_edgeIsFree && m_core->_edgeWid == inst->warp_id());
        EdgeIsBusy = m_core->EventIsRunningOnWarp(inst->warp_id()) || !m_core->_edgeIsFree;
    }
    //printf("MARIA DEBUG dropLoad called with inst.pc=%d core=%d wid=%d EdgeIsBusy=%d wid_match=%d isInFlightLoad=%d \n", 
    //    inst->pc, m_core->get_sid(), inst->warp_id(), EdgeIsBusy, wid_match, m_core->m_warp[inst->warp_id()].isInFlightLoad(inst));
    // First check if we have an interrupted victim warp that matches this instruction warp_id. 
    if( EdgeIsBusy && wid_match && !m_core->m_warp[inst->warp_id()].isInFlightLoad(inst) ) { 
        if (!m_core->m_config->_edgeRunISR) {
            EDGE_DPRINT(EdgeDebug, "Dropping load with PC %d of preempted warp %d on core %d \n",
                inst->pc, inst->warp_id(), m_core->get_sid());
            return true; //need new assertions
        }
        if( !m_core->m_warp[inst->warp_id()].isIntWarp() ) {
            // Then, if this warp hasn't actually been interrupted yet, but is currently flushing, then we should still
            // be modifying the original warp. Note that we could still somehow perform the writeback here instead
            // of replaying this load. But to make things less complicated for now, just drop the load
            assert( !m_core->_edgeSaveState->valid() && 
                    m_core->m_warp[inst->warp_id()].isDroppedLoad(inst) );
        } else {
            assert( m_core->_edgeSaveState->valid() && 
                    m_core->m_warp[inst->warp_id()].isIntWarp() &&
                    m_core->_edgeSaveState->savedWarp()->isDroppedLoad(inst) );
        }
        return true;
    } else {
        return false;
    }
}

bool ldst_unit::loadNextWb(const warp_inst_t* inst)
{
    if( !m_next_wb.empty() && (m_next_wb.get_uid() == inst->get_uid()) )
        return true;
    else
        return false;
}

unsigned ldst_unit::writebackInst(warp_inst_t &inst)
{
    // Need to first check if we've dropped the load because a warp was interrupted. 
    if( dropLoad(&inst) ){
        // Dropping packet
        return 2;
    } else if (m_next_wb.empty()) {
        m_next_wb = inst;
        if( m_next_wb.isatomic() ) {
            m_core->decrement_atomic_count(m_next_wb.warp_id(),m_next_wb.active_count());
        }
    } else if (m_next_wb.get_uid() != inst.get_uid()) {
        return 0; // WB reg full
    }
    return 1;
}

unsigned ldst_unit::clock_multiplier() const
{ 
    return m_config->mem_warp_parts; 
}
/*
void ldst_unit::issue( register_set &reg_set )
{
	warp_inst_t* inst = *(reg_set.get_ready());
   // stat collection
   m_core->mem_instruction_stats(*inst); 

   // record how many pending register writes/memory accesses there are for this instruction 
   assert(inst->empty() == false); 
   if (inst->is_load() and inst->space.get_type() != shared_space) {
      unsigned warp_id = inst->warp_id(); 
      unsigned n_accesses = inst->accessq_count(); 
      for (unsigned r = 0; r < 4; r++) {
         unsigned reg_id = inst->out[r]; 
         if (reg_id > 0) {
            m_pending_writes[warp_id][reg_id] += n_accesses; 
         }
      }
   }

   pipelined_simd_unit::issue(reg_set);
}
*/
void ldst_unit::cycle()
{
   writeback();
   m_operand_collector->step();
   for( unsigned stage=0; (stage+1)<m_pipeline_depth; stage++ ) 
       if( m_pipeline_reg[stage]->empty() && !m_pipeline_reg[stage+1]->empty() )
            move_warp(m_pipeline_reg[stage], m_pipeline_reg[stage+1]);

   if( !m_response_fifo.empty() ) {
       mem_fetch *mf = m_response_fifo.front();
       if (mf->istexture()) {
           if (m_L1T->fill_port_free()) {
               m_L1T->fill(mf,gpu_sim_cycle+gpu_tot_sim_cycle);
               m_response_fifo.pop_front(); 
           }
       } else if (mf->isconst())  {
           if (m_L1C->fill_port_free()) {
               mf->set_status(IN_SHADER_FETCHED,gpu_sim_cycle+gpu_tot_sim_cycle);
               m_L1C->fill(mf,gpu_sim_cycle+gpu_tot_sim_cycle);
               m_response_fifo.pop_front(); 
           }
       } else {
    	   if( mf->get_type() == WRITE_ACK || ( m_config->gpgpu_perfect_mem && mf->get_is_write() )) {
               m_core->store_ack(mf);
               m_response_fifo.pop_front();
               delete mf;
           } else {
               assert( !mf->get_is_write() ); // L1 cache is write evict, allocate line on load miss only

               bool bypassL1D = false; 
               if ( CACHE_GLOBAL == mf->get_inst().cache_op || (m_L1D == NULL) ) {
                   bypassL1D = true; 
               } else if (mf->get_access_type() == GLOBAL_ACC_R || mf->get_access_type() == GLOBAL_ACC_W) { // global memory access 
                   if (m_core->get_config()->gmem_skip_L1D)
                       bypassL1D = true; 
               }
               if( bypassL1D ) {
                   if ( m_next_global == NULL ) {
                       mf->set_status(IN_SHADER_FETCHED,gpu_sim_cycle+gpu_tot_sim_cycle);
                       m_response_fifo.pop_front();
                       m_next_global = mf;
                   }
               } else {
                   if (m_L1D->fill_port_free()) {
                       m_L1D->fill(mf,gpu_sim_cycle+gpu_tot_sim_cycle);
                       m_response_fifo.pop_front();
                   }
               }
           }
       }
   }

   m_L1T->cycle();
   m_L1C->cycle();
   if( m_L1D ) m_L1D->cycle();

   warp_inst_t &pipe_reg = *m_dispatch_reg;
   enum mem_stage_stall_type rc_fail = NO_RC_FAIL;
   mem_stage_access_type type;
   bool done = true;
   done &= shared_cycle(pipe_reg, rc_fail, type);
   done &= constant_cycle(pipe_reg, rc_fail, type);
   done &= texture_cycle(pipe_reg, rc_fail, type);
   done &= memory_cycle_gem5(pipe_reg, rc_fail, type);
   m_mem_rc = rc_fail;

   if (!done) { // log stall types and return
      assert(rc_fail != NO_RC_FAIL);
      m_stats->gpgpu_n_stall_shd_mem++;
      m_stats->gpu_stall_shd_mem_breakdown[type][rc_fail]++;
      return;
   }

   if( !pipe_reg.empty() ) {
       unsigned warp_id = pipe_reg.warp_id();
       if( pipe_reg.is_load() ) {

           if( pipe_reg.space.get_type() == shared_space ) {
               if( m_pipeline_reg[m_pipeline_depth-1]->empty() ) {
                   // new shared memory request
                   move_warp(m_pipeline_reg[m_pipeline_depth-1],m_dispatch_reg);
                   m_dispatch_reg->clear();
               }
           } else {
               //if( pipe_reg.active_count() > 0 ) {
               //    if( !m_operand_collector->writeback(pipe_reg) ) 
               //        return;
               //} 

               bool pending_requests=false;
               for( unsigned r=0; r<4; r++ ) {
                   unsigned reg_id = pipe_reg.out[r];
                   if( reg_id > 0 ) {
                       if( m_pending_writes[warp_id].find(reg_id) != m_pending_writes[warp_id].end() ) {
                           if ( m_pending_writes[warp_id][reg_id] > 0 ) {
                               pending_requests=true;
                               break;
                           } else {
                               // this instruction is done already
                               m_pending_writes[warp_id].erase(reg_id); 
                           }
                       }
                   }
               }
               if( !pending_requests ) {
                   m_core->warp_inst_complete(*m_dispatch_reg);
                   m_scoreboard->releaseRegisters(m_dispatch_reg);

                   // EDGE: If load finishes, decrement pending here
                   m_core->warpDecPendingLoads(warp_id);
 
                   // EDGE: Decrease pending loads from pipeline                                    
                   assert( m_core->getWarp(warp_id)->findInFlightLoad( m_dispatch_reg ) );
                   m_core->getWarp(warp_id)->completeLoad( m_dispatch_reg );
               }

               m_core->dec_inst_in_pipeline(warp_id);
               m_dispatch_reg->clear();
           }
       } else {
           // stores exit pipeline here
           m_core->dec_inst_in_pipeline(warp_id);
           m_core->warp_inst_complete(*m_dispatch_reg);
           m_dispatch_reg->clear();
       }
   }
}

void shader_core_ctx::register_cta_thread_exit( unsigned cta_num, kernel_info_t* kernel )
{
    assert( m_cta_status[cta_num] > 0 );
    m_cta_status[cta_num]--;

    // Check if this is the last thread in the CTA to complete. If so, clean up the resources.
    if (!m_cta_status[cta_num]) {

        m_gpu->UpdateAvgCTATime(gpu_sim_cycle - m_cta_start_time[cta_num]);

        // Should definitely have at least one active CTA on this core, since a thread just completed. 
        assert( m_n_active_cta > 0 );
        m_n_active_cta--;

        // EDGE FIXME
        m_barriers.deallocate_barrier(cta_num);

        shader_CTA_count_unlog(m_sid, 1);
        //      printf("GPGPU-Sim uArch: Shader %d finished CTA #%d (%lld,%lld), %u CTAs running\n", 
        //          m_sid, cta_num, gpu_sim_cycle, gpu_tot_sim_cycle, m_n_active_cta );
        m_gpu->gem5CudaGPU->getCudaCore(m_sid)->record_block_commit(cta_num);

        if( m_config->_edgeEventReserveCta > 0 && kernel->isEventKernel() ) {
            clearReservedEvent(cta_num);
        } else {

            // CDP - Concurrent kernels + EDGE + Gem5
            if( !kernel->isISRKernel() || m_config->isIntDedicated() != 2 ){
                release_shader_resource_1block(cta_num, *kernel);
            } else {
                // Don't release pseudo dedicated resources
                assert( m_occupied_ctas >= 1 );
                m_occupied_ctas--;
                m_occupied_cta_to_hwtid.erase(cta_num);
            }

        }

        assert( kernel->running() );
        kernel->dec_running();
 

        if( !m_gpu->kernel_more_cta_left(kernel) && 
                !kernel->running() && 
                (_kernelFinishing.find(kernel) == _kernelFinishing.end()) ) { 

            // Save stats for this kernel
            if (kernel->isEventKernel() || kernel->isISRKernel()) {
                m_gpu->DecEventRunning();
                kernel->stats()._TotalLatency = gpu_sim_cycle - kernel->edge_get_int_start_cycle();
                EDGE_DPRINT(EdgeDebug, "%lld: CTA %d completed for kernel %s KernelId=%d on Shader %d. Total latency is %d = %d(wait) + %d(run) \n", 
                    gpu_sim_cycle, cta_num, kernel->entry()->get_name().c_str(), kernel->get_uid(), m_sid, kernel->stats()._TotalLatency, 
                    kernel->stats()._TotalLatency-kernel->stats()._nCycles,kernel->stats()._nCycles );     
                    kernel->stats()._edgeCompleted = gpu_sim_cycle;
            }
                       
            m_gpu->pushKernelStats(kernel);
            //printf("MARIA DEBUG kernel %s finished on core %d at cycle %lld\n", kernel->entry()->get_name().c_str(), get_sid(), gpu_sim_cycle + gpu_tot_sim_cycle);
            //EDGE_DPRINT(EdgeDebug, "%lld: CTA %d completed for kernel %s on Shader %d \n", gpu_sim_cycle, cta_num, kernel->entry()->get_name().c_str(), m_sid);     
            if( !kernel->isISRKernel() || kernel->isEventKernel() ) {
                
                if (!kernel->isISRKernel()) {
                    assert( kernel->hasStarted() );
                }
                kernel->finish();
                kernel->startFinishing();

                // Kernel doesn't have any more work, has completed on this shader, and hasn't started finishing yet
                _kernelFinishing[kernel] = false;
                _kernelCompletePending.push(kernel);   // Schedule the freeing for later
            } else {
                // Simply unblock any CPU threads that may be waiting for the GPU to be completely finished
                // EDGE: FIXME: Bypasses the flushing etc, just simply notify any host threads
                m_gpu->gem5CudaGPU->processFinishEdgeEvent();
            }


            // MEMC_EDGE_DEBUG_TEST
            if( m_config->_isMemcConv ) {
                std::string convKernel("_Z20filterActs_YxX_colorILi4ELi32ELi1ELi4ELi1ELb0ELb1EEvPfS0_S0_iiiiiiiiiiffi");
                if( !convKernel.compare(kernel->entry()->get_name()) ) {
                    // Flush all pending kernels that haven't launched yet
                    m_gpu->flushMemcKernels();
                }            
            }


        }
    }
}

void shader_core_ctx::NewEdgeDoOneKernelCompletion(kernel_info_t* kernel, int cid) {
    if (!kernel->isEventKernel()) { //restore hw state when required
        return;
    } 

    //return CTA to pool
    _edgeCtas.push_back(cid);

    if (!kernel->GetEdgeSaveStateValid()) {
        return;
    }

    EDGE_DPRINT(EdgeDebug, "Event kernel finished on shader <%d>. Restoring state. \n", m_sid);
    EdgeSaveState* OriginalState = GetAndEraseEdgeState(kernel->GetEdgeSaveStateId());
    if (OriginalState == NULL) {
        //printf("NewEdgeDoOneKernelCompletion OriginalState==NULL for core %d warp %d ")
        assert(OriginalState);
    }
    
    edgeResetIntState(OriginalState->wid(), OriginalState->cid(), true);
    edgeRestoreHwState(OriginalState);
}

// First part of the kernel completition 
void 
shader_core_ctx::startKernelFinish(kernel_info_t* k)
{
    _kernelFinishing[k] = true;
    m_gpu->gem5CudaGPU->getCudaCore(m_sid)->finishKernel(); // TODO: Should pass kernel k?
}

// Last part of the kernel completion 
void 
shader_core_ctx::finishKernel()
{
    assert( !_kernelCompletePending.empty() );
    kernel_info_t* k = _kernelCompletePending.front();
    _kernelCompletePending.pop();
    assert( !m_gpu->kernel_more_cta_left(k) );     
    
    if( m_n_active_cta == 0 ) {
        printf("GPGPU-Sim uArch: Shader %u empty (last released kernel %u \'%s\').\n", 
                m_sid, k->get_uid(), k->name().c_str() );
        m_kernel = NULL;
    }

    _kernelFinishing.erase(k);

    if( !k->running() ) {
        printf("GPGPU-Sim uArch: GPU detected kernel %u \'%s\' finished on shader %u.\n", 
                k->get_uid(), k->name().c_str(), m_sid );

        k->completeFinishing();

        // EDGE: Print out the GPU stats
        m_gpu->print_stats();

        k->printStats(stdout);

        if(m_kernel == k)
            m_kernel = NULL;
        m_gpu->set_kernel_done( k );        

    }

    fflush(stdout);
}

// Have we already issued the finish request for kernel k?
bool 
shader_core_ctx::kernelFinishIssued(kernel_info_t* k)
{
    assert( _kernelFinishing.find(k) != _kernelFinishing.end() );
    return _kernelFinishing[k];
}

bool
shader_core_ctx::kernelFinishing()
{
    return !_kernelFinishing.empty();
}

#if 0
void shader_core_ctx::start_kernel_finish()
{
    assert(!m_kernel_finishing);
    m_kernel_finishing = true;
    m_gpu->gem5CudaGPU->getCudaCore(m_sid)->finishKernel();
}

// CDP - Concurrent kernrels
// TODO: Figure out how to get the kernel from "register_cta_thread_exit(cta_num, kernel)" to here. 
void shader_core_ctx::finish_kernel()
{
  assert( m_kernel != NULL );
  
  m_kernel->dec_running();
  
  printf("GPGPU-Sim uArch: Shader %u empty (release kernel %u \'%s\').\n", m_sid, m_kernel->get_uid(),
         m_kernel->name().c_str() );

  if( m_kernel->no_more_ctas_to_run() ) { // Should guarantee be true, since we only call finish if no more to run
      if( !m_kernel->running() ) {        // May not be true until the last CTA finishes
          printf("GPGPU-Sim uArch: GPU detected kernel \'%s\' finished on shader %u.\n", m_kernel->name().c_str(), m_sid );
          m_gpu->set_kernel_done( m_kernel );
      }
  }

  m_kernel=NULL;
  m_kernel_finishing = false;
  fflush(stdout);

}
#endif

void gpgpu_sim::shader_print_runtime_stat( FILE *fout ) 
{
    /*
   fprintf(fout, "SHD_INSN: ");
   for (unsigned i=0;i<m_n_shader;i++) 
      fprintf(fout, "%u ",m_sc[i]->get_num_sim_insn());
   fprintf(fout, "\n");
   fprintf(fout, "SHD_THDS: ");
   for (unsigned i=0;i<m_n_shader;i++) 
      fprintf(fout, "%u ",m_sc[i]->get_not_completed());
   fprintf(fout, "\n");
   fprintf(fout, "SHD_DIVG: ");
   for (unsigned i=0;i<m_n_shader;i++) 
      fprintf(fout, "%u ",m_sc[i]->get_n_diverge());
   fprintf(fout, "\n");

   fprintf(fout, "THD_INSN: ");
   for (unsigned i=0; i<m_shader_config->n_thread_per_shader; i++) 
      fprintf(fout, "%d ", m_sc[0]->get_thread_n_insn(i) );
   fprintf(fout, "\n");
   */
}


void gpgpu_sim::shader_print_scheduler_stat( FILE* fout, bool print_dynamic_info ) const
{
    // Print out the stats from the sampling shader core
    const unsigned scheduler_sampling_core = m_shader_config->gpgpu_warp_issue_shader;
    #define STR_SIZE 55
    char name_buff[ STR_SIZE ];
    name_buff[ STR_SIZE - 1 ] = '\0';
    const std::vector< unsigned >& distro
        = print_dynamic_info ?
          m_shader_stats->get_dynamic_warp_issue()[ scheduler_sampling_core ] :
          m_shader_stats->get_warp_slot_issue()[ scheduler_sampling_core ];
    if ( print_dynamic_info ) {
        snprintf( name_buff, STR_SIZE - 1, "dynamic_warp_id" );
    } else {
        snprintf( name_buff, STR_SIZE - 1, "warp_id" );
    }
    fprintf( fout,
             "Shader %d %s issue ditsribution:\n",
             scheduler_sampling_core,
             name_buff );
    const unsigned num_warp_ids = distro.size();
    // First print out the warp ids
    fprintf( fout, "%s:\n", name_buff );
    for ( unsigned warp_id = 0;
          warp_id < num_warp_ids;
          ++warp_id  ) {
        fprintf( fout, "%d, ", warp_id );
    }

    fprintf( fout, "\ndistro:\n" );
    // Then print out the distribution of instuctions issued
    for ( std::vector< unsigned >::const_iterator iter = distro.begin();
          iter != distro.end();
          iter++ ) {
        fprintf( fout, "%d, ", *iter );
    }
    fprintf( fout, "\n" );
}

void gpgpu_sim::shader_print_cache_stats( FILE *fout ) const{

    // L1I
    struct cache_sub_stats total_css;
    struct cache_sub_stats css;

    if(!m_shader_config->m_L1I_config.disabled()){
        total_css.clear();
        css.clear();
        fprintf(fout, "\n========= Core cache stats =========\n");
        fprintf(fout, "L1I_cache:\n");
        for ( unsigned i = 0; i < m_shader_config->n_simt_clusters; ++i ) {
            m_cluster[i]->get_L1I_sub_stats(css);
            total_css += css;
        }
        fprintf(fout, "\tL1I_total_cache_accesses = %u\n", total_css.accesses);
        fprintf(fout, "\tL1I_total_cache_misses = %u\n", total_css.misses);
        if(total_css.accesses > 0){
            fprintf(fout, "\tL1I_total_cache_miss_rate = %.4lf\n", (double)total_css.misses / (double)total_css.accesses);
        }
        fprintf(fout, "\tL1I_total_cache_pending_hits = %u\n", total_css.pending_hits);
        fprintf(fout, "\tL1I_total_cache_reservation_fails = %u\n", total_css.res_fails);
    }

    // L1D
    if(!m_shader_config->m_L1D_config.disabled()){
        total_css.clear();
        css.clear();
        fprintf(fout, "L1D_cache:\n");
        for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++){
            m_cluster[i]->get_L1D_sub_stats(css);

            fprintf( stdout, "\tL1D_cache_core[%d]: Access = %d, Miss = %d, Miss_rate = %.3lf, Pending_hits = %u, Reservation_fails = %u\n",
                     i, css.accesses, css.misses, (double)css.misses / (double)css.accesses, css.pending_hits, css.res_fails);

            total_css += css;
        }
        fprintf(fout, "\tL1D_total_cache_accesses = %u\n", total_css.accesses);
        fprintf(fout, "\tL1D_total_cache_misses = %u\n", total_css.misses);
        if(total_css.accesses > 0){
            fprintf(fout, "\tL1D_total_cache_miss_rate = %.4lf\n", (double)total_css.misses / (double)total_css.accesses);
        }
        fprintf(fout, "\tL1D_total_cache_pending_hits = %u\n", total_css.pending_hits);
        fprintf(fout, "\tL1D_total_cache_reservation_fails = %u\n", total_css.res_fails);
        total_css.print_port_stats(fout, "\tL1D_cache"); 
    }

    // L1C
    if(!m_shader_config->m_L1C_config.disabled()){
        total_css.clear();
        css.clear();
        fprintf(fout, "L1C_cache:\n");
        for ( unsigned i = 0; i < m_shader_config->n_simt_clusters; ++i ) {
            m_cluster[i]->get_L1C_sub_stats(css);
            total_css += css;
        }
        fprintf(fout, "\tL1C_total_cache_accesses = %u\n", total_css.accesses);
        fprintf(fout, "\tL1C_total_cache_misses = %u\n", total_css.misses);
        if(total_css.accesses > 0){
            fprintf(fout, "\tL1C_total_cache_miss_rate = %.4lf\n", (double)total_css.misses / (double)total_css.accesses);
        }
        fprintf(fout, "\tL1C_total_cache_pending_hits = %u\n", total_css.pending_hits);
        fprintf(fout, "\tL1C_total_cache_reservation_fails = %u\n", total_css.res_fails);
    }

    // L1T
    if(!m_shader_config->m_L1T_config.disabled()){
        total_css.clear();
        css.clear();
        fprintf(fout, "L1T_cache:\n");
        for ( unsigned i = 0; i < m_shader_config->n_simt_clusters; ++i ) {
            m_cluster[i]->get_L1T_sub_stats(css);
            total_css += css;
        }
        fprintf(fout, "\tL1T_total_cache_accesses = %u\n", total_css.accesses);
        fprintf(fout, "\tL1T_total_cache_misses = %u\n", total_css.misses);
        if(total_css.accesses > 0){
            fprintf(fout, "\tL1T_total_cache_miss_rate = %.4lf\n", (double)total_css.misses / (double)total_css.accesses);
        }
        fprintf(fout, "\tL1T_total_cache_pending_hits = %u\n", total_css.pending_hits);
        fprintf(fout, "\tL1T_total_cache_reservation_fails = %u\n", total_css.res_fails);
    }
}

void gpgpu_sim::shader_print_l1_miss_stat( FILE *fout ) const
{
   unsigned total_d1_misses = 0, total_d1_accesses = 0;
   for ( unsigned i = 0; i < m_shader_config->n_simt_clusters; ++i ) {
         unsigned custer_d1_misses = 0, cluster_d1_accesses = 0;
         m_cluster[ i ]->print_cache_stats( fout, cluster_d1_accesses, custer_d1_misses );
         total_d1_misses += custer_d1_misses;
         total_d1_accesses += cluster_d1_accesses;
   }
   fprintf( fout, "total_dl1_misses=%d\n", total_d1_misses );
   fprintf( fout, "total_dl1_accesses=%d\n", total_d1_accesses );
   fprintf( fout, "total_dl1_miss_rate= %f\n", (float)total_d1_misses / (float)total_d1_accesses );
   /*
   fprintf(fout, "THD_INSN_AC: ");
   for (unsigned i=0; i<m_shader_config->n_thread_per_shader; i++) 
      fprintf(fout, "%d ", m_sc[0]->get_thread_n_insn_ac(i));
   fprintf(fout, "\n");
   fprintf(fout, "T_L1_Mss: "); //l1 miss rate per thread
   for (unsigned i=0; i<m_shader_config->n_thread_per_shader; i++) 
      fprintf(fout, "%d ", m_sc[0]->get_thread_n_l1_mis_ac(i));
   fprintf(fout, "\n");
   fprintf(fout, "T_L1_Mgs: "); //l1 merged miss rate per thread
   for (unsigned i=0; i<m_shader_config->n_thread_per_shader; i++) 
      fprintf(fout, "%d ", m_sc[0]->get_thread_n_l1_mis_ac(i) - m_sc[0]->get_thread_n_l1_mrghit_ac(i));
   fprintf(fout, "\n");
   fprintf(fout, "T_L1_Acc: "); //l1 access per thread
   for (unsigned i=0; i<m_shader_config->n_thread_per_shader; i++) 
      fprintf(fout, "%d ", m_sc[0]->get_thread_n_l1_access_ac(i));
   fprintf(fout, "\n");

   //per warp
   int temp =0; 
   fprintf(fout, "W_L1_Mss: "); //l1 miss rate per warp
   for (unsigned i=0; i<m_shader_config->n_thread_per_shader; i++) {
      temp += m_sc[0]->get_thread_n_l1_mis_ac(i);
      if (i%m_shader_config->warp_size == (unsigned)(m_shader_config->warp_size-1)) {
         fprintf(fout, "%d ", temp);
         temp = 0;
      }
   }
   fprintf(fout, "\n");
   temp=0;
   fprintf(fout, "W_L1_Mgs: "); //l1 merged miss rate per warp
   for (unsigned i=0; i<m_shader_config->n_thread_per_shader; i++) {
      temp += (m_sc[0]->get_thread_n_l1_mis_ac(i) - m_sc[0]->get_thread_n_l1_mrghit_ac(i) );
      if (i%m_shader_config->warp_size == (unsigned)(m_shader_config->warp_size-1)) {
         fprintf(fout, "%d ", temp);
         temp = 0;
      }
   }
   fprintf(fout, "\n");
   temp =0;
   fprintf(fout, "W_L1_Acc: "); //l1 access per warp
   for (unsigned i=0; i<m_shader_config->n_thread_per_shader; i++) {
      temp += m_sc[0]->get_thread_n_l1_access_ac(i);
      if (i%m_shader_config->warp_size == (unsigned)(m_shader_config->warp_size-1)) {
         fprintf(fout, "%d ", temp);
         temp = 0;
      }
   }
   fprintf(fout, "\n");
   */
}

void warp_inst_t::print( FILE *fout ) const
{
    if (empty() ) {
        fprintf(fout,"bubble\n" );
        return;
    } else 
        fprintf(fout,"0x%04x ", pc );
    fprintf(fout, "w%02d[", m_warp_id);
    for (unsigned j=0; j<m_config->warp_size; j++)
        fprintf(fout, "%c", (active(j)?'1':'0') );
    fprintf(fout, "]: ");
    ptx_print_insn( pc, fout );
    fprintf(fout, "\n");
}
void shader_core_ctx::incexecstat(warp_inst_t *&inst)
{
	if(inst->mem_op==TEX)
		inctex_stat(inst->active_count(),1);

    // Latency numbers for next operations are used to scale the power values
    // for special operations, according observations from microbenchmarking
    // TODO: put these numbers in the xml configuration

	switch(inst->sp_op){
	case INT__OP:
		incialu_stat(inst->active_count(),25);
		break;
	case INT_MUL_OP:
		incimul_stat(inst->active_count(),7.2);
		break;
	case INT_MUL24_OP:
		incimul24_stat(inst->active_count(),4.2);
		break;
	case INT_MUL32_OP:
		incimul32_stat(inst->active_count(),4);
		break;
	case INT_DIV_OP:
		incidiv_stat(inst->active_count(),40);
		break;
	case FP__OP:
		incfpalu_stat(inst->active_count(),1);
		break;
	case FP_MUL_OP:
		incfpmul_stat(inst->active_count(),1.8);
		break;
	case FP_DIV_OP:
		incfpdiv_stat(inst->active_count(),48);
		break;
	case FP_SQRT_OP:
		inctrans_stat(inst->active_count(),25);
		break;
	case FP_LG_OP:
		inctrans_stat(inst->active_count(),35);
		break;
	case FP_SIN_OP:
		inctrans_stat(inst->active_count(),12);
		break;
	case FP_EXP_OP:
		inctrans_stat(inst->active_count(),35);
		break;
	default:
		break;
	}
}
void shader_core_ctx::print_stage(unsigned int stage, FILE *fout ) const
{
   m_pipeline_reg[stage].print(fout);
   //m_pipeline_reg[stage].print(fout);
}

void shader_core_ctx::display_simt_state(FILE *fout, int mask ) const
{
    if ( (mask & 4) && m_config->model == POST_DOMINATOR ) {
       fprintf(fout,"per warp SIMT control-flow state:\n");
       unsigned n = m_config->n_thread_per_shader / m_config->warp_size;
       for (unsigned i=0; i < n; i++) {
          unsigned nactive = 0;
          for (unsigned j=0; j<m_config->warp_size; j++ ) {
             unsigned tid = i*m_config->warp_size + j;
             int done = ptx_thread_done(tid);
             nactive += (ptx_thread_done(tid)?0:1);
             if ( done && (mask & 8) ) {
                if( m_thread[tid] != NULL ) {
                    unsigned done_cycle = m_thread[tid]->donecycle();
                    if ( done_cycle ) {
                       printf("\n w%02u:t%03u: done @ cycle %u", i, tid, done_cycle );
                    }
                }
             }
          }
          if ( nactive == 0 ) {
             continue;
          }
          m_simt_stack[i]->print(fout);
       }
       fprintf(fout,"\n");
    }
}

void ldst_unit::print(FILE *fout) const
{
    fprintf(fout,"LD/ST unit  = ");
    m_dispatch_reg->print(fout);
    if ( m_mem_rc != NO_RC_FAIL ) {
        fprintf(fout,"              LD/ST stall condition: ");
        switch ( m_mem_rc ) {
        case BK_CONF:        fprintf(fout,"BK_CONF"); break;
        case MSHR_RC_FAIL:   fprintf(fout,"MSHR_RC_FAIL"); break;
        case ICNT_RC_FAIL:   fprintf(fout,"ICNT_RC_FAIL"); break;
        case COAL_STALL:     fprintf(fout,"COAL_STALL"); break;
        case WB_ICNT_RC_FAIL: fprintf(fout,"WB_ICNT_RC_FAIL"); break;
        case WB_CACHE_RSRV_FAIL: fprintf(fout,"WB_CACHE_RSRV_FAIL"); break;
        case N_MEM_STAGE_STALL_TYPE: fprintf(fout,"N_MEM_STAGE_STALL_TYPE"); break;
        default: abort();
        }
        fprintf(fout,"\n");
    }
    fprintf(fout,"LD/ST wb    = ");
    m_next_wb.print(fout);
    fprintf(fout, "Last LD/ST writeback @ %llu + %llu (gpu_sim_cycle+gpu_tot_sim_cycle)\n",
                  m_last_inst_gpu_sim_cycle, m_last_inst_gpu_tot_sim_cycle );
    fprintf(fout,"Pending register writes:\n");
    std::map<unsigned/*warp_id*/, std::map<unsigned/*regnum*/,unsigned/*count*/> >::const_iterator w;
    for( w=m_pending_writes.begin(); w!=m_pending_writes.end(); w++ ) {
        unsigned warp_id = w->first;
        const std::map<unsigned/*regnum*/,unsigned/*count*/> &warp_info = w->second;
        if( warp_info.empty() ) 
            continue;
        fprintf(fout,"  w%2u : ", warp_id );
        std::map<unsigned/*regnum*/,unsigned/*count*/>::const_iterator r;
        for( r=warp_info.begin(); r!=warp_info.end(); ++r ) {
            fprintf(fout,"  %u(%u)", r->first, r->second );
        }
        fprintf(fout,"\n");
    }
    m_L1C->display_state(fout);
    m_L1T->display_state(fout);
    if( !m_config->m_L1D_config.disabled() )
    	m_L1D->display_state(fout);
    fprintf(fout,"LD/ST response FIFO (occupancy = %zu):\n", m_response_fifo.size() );
    for( std::list<mem_fetch*>::const_iterator i=m_response_fifo.begin(); i != m_response_fifo.end(); i++ ) {
        const mem_fetch *mf = *i;
        mf->print(fout);
    }
}

void shader_core_ctx::display_pipeline(FILE *fout, int print_mem, int mask ) const
{
   fprintf(fout, "=================================================\n");
   fprintf(fout, "shader %u at cycle %Lu+%Lu (%u threads running)\n", m_sid, 
           gpu_tot_sim_cycle, gpu_sim_cycle, m_not_completed);
   fprintf(fout, "=================================================\n");

   dump_warp_state(fout);
   fprintf(fout,"\n");

   m_L1I->display_state(fout);

   fprintf(fout, "IF/ID       = ");
   if( !m_inst_fetch_buffer.m_valid )
       fprintf(fout,"bubble\n");
   else {
       fprintf(fout,"w%2u : pc = 0x%x, nbytes = %u\n", 
               m_inst_fetch_buffer.m_warp_id,
               m_inst_fetch_buffer.m_pc, 
               m_inst_fetch_buffer.m_nbytes );
   }
   fprintf(fout,"\nibuffer status:\n");
   for( unsigned i=0; i<m_config->max_warps_per_shader; i++) {
       if( !m_warp[i].ibuffer_empty() ) 
           m_warp[i].print_ibuffer(fout);
   }
   fprintf(fout,"\n");
   display_simt_state(fout,mask);
   fprintf(fout, "-------------------------- Scoreboard\n");
   m_scoreboard->printContents();
/*
   fprintf(fout,"ID/OC (SP)  = ");
   print_stage(ID_OC_SP, fout);
   fprintf(fout,"ID/OC (SFU) = ");
   print_stage(ID_OC_SFU, fout);
   fprintf(fout,"ID/OC (MEM) = ");
   print_stage(ID_OC_MEM, fout);
*/
   fprintf(fout, "-------------------------- OP COL\n");
   m_operand_collector.dump(fout);
/* fprintf(fout, "OC/EX (SP)  = ");
   print_stage(OC_EX_SP, fout);
   fprintf(fout, "OC/EX (SFU) = ");
   print_stage(OC_EX_SFU, fout);
   fprintf(fout, "OC/EX (MEM) = ");
   print_stage(OC_EX_MEM, fout);
*/
   fprintf(fout, "-------------------------- Pipe Regs\n");

   for (unsigned i = 0; i < N_PIPELINE_STAGES; i++) {
       fprintf(fout,"--- %s ---\n",pipeline_stage_name_decode[i]);
       print_stage(i,fout);fprintf(fout,"\n");
   }

   fprintf(fout, "-------------------------- Fu\n");
   for( unsigned n=0; n < m_num_function_units; n++ ){
       m_fu[n]->print(fout);
       fprintf(fout, "---------------\n");
   }
   fprintf(fout, "-------------------------- other:\n");

   for(unsigned i=0; i<num_result_bus; i++){
	   std::string bits = m_result_bus[i]->to_string();
	   fprintf(fout, "EX/WB sched[%d]= %s\n", i, bits.c_str() );
   }
   fprintf(fout, "EX/WB      = ");
   print_stage(EX_WB, fout);
   fprintf(fout, "\n");
   fprintf(fout, "Last EX/WB writeback @ %llu + %llu (gpu_sim_cycle+gpu_tot_sim_cycle)\n",
                 m_last_inst_gpu_sim_cycle, m_last_inst_gpu_tot_sim_cycle );

   if( m_active_threads.count() <= 2*m_config->warp_size ) {
       fprintf(fout,"Active Threads : ");
       unsigned last_warp_id = -1;
       for(unsigned tid=0; tid < m_active_threads.size(); tid++ ) {
           unsigned warp_id = tid/m_config->warp_size;
           if( m_active_threads.test(tid) ) {
               if( warp_id != last_warp_id ) {
                   fprintf(fout,"\n  warp %u : ", warp_id );
                   last_warp_id=warp_id;
               }
               fprintf(fout,"%u ", tid );
           }
       }
   }

}

unsigned int shader_core_config::max_cta( const kernel_info_t &k ) const
{
   unsigned threads_per_cta  = k.threads_per_cta();
   const class function_info *kernel = k.entry();
   unsigned int padded_cta_size = threads_per_cta;
   if (padded_cta_size%warp_size) 
      padded_cta_size = ((padded_cta_size/warp_size)+1)*(warp_size);

   //Limit by n_threads/shader
   unsigned int result_thread = n_thread_per_shader / padded_cta_size;

   const struct gpgpu_ptx_sim_kernel_info *kernel_info = ptx_sim_kernel_info(kernel);

   //Limit by shmem/shader
   unsigned int result_shmem = (unsigned)-1;
   if (kernel_info->smem > 0)
      result_shmem = gpgpu_shmem_size / kernel_info->smem;

   //Limit by register count, rounded up to multiple of 4.
   unsigned int result_regs = (unsigned)-1;
   if (kernel_info->regs > 0)
      result_regs = gpgpu_shader_registers / (padded_cta_size * ((kernel_info->regs+3)&~3));

   //Limit by CTA
   unsigned int result_cta = max_cta_per_core;

   // EDGE: Don't include the Interrupt CTA
   //if( isIntDedicated() ) 
   if( _intMode )  // All int modes increase the CTA count
       result_cta -= _edgeEventCtasPerCore;

   unsigned result = result_thread;
   result = gs_min2(result, result_shmem);
   result = gs_min2(result, result_regs);
   result = gs_min2(result, result_cta);

#if 0
   static const struct gpgpu_ptx_sim_kernel_info* last_kinfo = NULL;
   if (last_kinfo != kernel_info) {   //Only print out stats if kernel_info struct changes
      last_kinfo = kernel_info;
      printf ("GPGPU-Sim uArch: CTA/core = %u, limited by:", result);
      if (result == result_thread) printf (" threads");
      if (result == result_shmem) printf (" shmem");
      if (result == result_regs) printf (" regs");
      if (result == result_cta) printf (" cta_limit");
      printf ("\n");
   }
#else 
    static std::map<const struct gpgpu_ptx_sim_kernel_info*, bool> lastKinfo;
    if( lastKinfo.find(kernel_info) == lastKinfo.end() ) { // Only print on the first time we see this kernel being launched
        lastKinfo[kernel_info] = true;
        printf ("GPGPU-Sim uArch: CTA/core = %u, limited by:", result);
        if (result == result_thread) printf (" threads");
        if (result == result_shmem) printf (" shmem");
        if (result == result_regs) printf (" regs");
        if (result == result_cta) printf (" cta_limit");
        printf ("\n");       
    }

#endif

    //gpu_max_cta_per_shader is limited by number of CTAs if not enough to keep all cores busy    
    if( k.num_blocks() < result*num_shader() ) { 
       result = k.num_blocks() / num_shader();
       if (k.num_blocks() % num_shader())
          result++;
    }

    assert( result <= MAX_CTA_PER_SHADER );
    if (result < 1) {
       printf ("GPGPU-Sim uArch: ERROR ** Kernel requires more resources than shader has.\n");
       abort();
    }

    return result;
}

void shader_core_ctx::cycle()
{
	m_stats->shader_cycles[m_sid]++;

    // EDGE: Increment the cycle stats for the iKernel if currently running
    if( m_config->_edgeRunISR && _iKernel && _iKernel->running() )
        _iKernel->stats()._nCycles++;
    
    _warpOccupancyPerCycle += m_occupied_n_threads;
    _registerUtilizationPerCycle += m_occupied_regs;

    writeback();
    execute();
    read_operands();
    issue();
    decode();
    fetch();
}

// Flushes all content of the cache to memory

void shader_core_ctx::cache_flush()
{
   m_ldst_unit->flush();
}

// modifiers
std::list<opndcoll_rfu_t::op_t> opndcoll_rfu_t::arbiter_t::allocate_reads() 
{
   std::list<op_t> result;  // a list of registers that (a) are in different register banks, (b) do not go to the same operand collector

   int input;
   int output;
   int _inputs = m_num_banks;
   int _outputs = m_num_collectors;
   int _square = ( _inputs > _outputs ) ? _inputs : _outputs;
   assert(_square > 0);
   int _pri = (int)m_last_cu;

   // Clear matching
   for ( int i = 0; i < _inputs; ++i ) 
      _inmatch[i] = -1;
   for ( int j = 0; j < _outputs; ++j ) 
      _outmatch[j] = -1;

   for( unsigned i=0; i<m_num_banks; i++) {
      for( unsigned j=0; j<m_num_collectors; j++) {
         assert( i < (unsigned)_inputs );
         assert( j < (unsigned)_outputs );
         _request[i][j] = 0;
      }
      if( !m_queue[i].empty() ) {
         const op_t &op = m_queue[i].front();
         int oc_id = op.get_oc_id();
         assert( i < (unsigned)_inputs );
         assert( oc_id < _outputs );
         _request[i][oc_id] = 1;
      }
      if( m_allocated_bank[i].is_write() ) {
         assert( i < (unsigned)_inputs );
         _inmatch[i] = 0; // write gets priority
      }
   }

   ///// wavefront allocator from booksim... --->
   
   // Loop through diagonals of request matrix

   for ( int p = 0; p < _square; ++p ) {
      output = ( _pri + p ) % _square;

      // Step through the current diagonal
      for ( input = 0; input < _inputs; ++input ) {
          assert( input < _inputs );
          assert( output < _outputs );
         if ( ( output < _outputs ) && 
              ( _inmatch[input] == -1 ) && 
              ( _outmatch[output] == -1 ) &&
              ( _request[input][output]/*.label != -1*/ ) ) {
            // Grant!
            _inmatch[input] = output;
            _outmatch[output] = input;
         }

         output = ( output + 1 ) % _square;
      }
   }

   // Round-robin the priority diagonal
   _pri = ( _pri + 1 ) % _square;

   /// <--- end code from booksim

   m_last_cu = _pri;
   for( unsigned i=0; i < m_num_banks; i++ ) {
      if( _inmatch[i] != -1 ) {
         if( !m_allocated_bank[i].is_write() ) {
            unsigned bank = (unsigned)i;
            op_t &op = m_queue[bank].front();
            result.push_back(op);
            m_queue[bank].pop_front();
         }
      }
   }

   return result;
}

void barrier_set_t::verifyValidCta(unsigned cta_id)
{
   cta_to_warp_t::iterator w=m_cta_to_warps.find(cta_id);
   if( w == m_cta_to_warps.end() ) { // cta is active
      printf("ERROR ** cta_id %u not found in barrier set on cycle %llu+%llu...\n", cta_id, gpu_tot_sim_cycle, gpu_sim_cycle );
      dump();
      abort();
   }
}

bool barrier_set_t::anyAtBarrier( unsigned cta_id )
{
    verifyValidCta(cta_id);
    
    cta_to_warp_t::iterator w=m_cta_to_warps.find(cta_id);

    warp_set_t warps = w->second;
    warp_set_t at_barrier = warps & m_warp_at_barrier;

    return at_barrier.any();
}

bool barrier_set_t::isEdgeBarrier( unsigned cta_id ) 
{
    verifyValidCta(cta_id);
    return _edgeBarrier[cta_id];
}

void barrier_set_t::setEdgeBarrier( unsigned cta_id )
{
    verifyValidCta(cta_id);
    _edgeBarrier[cta_id] = true;
}


void barrier_set_t::releaseEdgeBarrier( unsigned cta_id ) 
{
    verifyValidCta(cta_id);

    cta_to_warp_t::iterator w=m_cta_to_warps.find(cta_id);

    warp_set_t warps_in_cta = w->second;
    warp_set_t at_barrier = warps_in_cta & m_warp_at_barrier;
    warp_set_t active = warps_in_cta & m_warp_active;

    if( at_barrier == active ) { // If all threads are at the barrier, release it
        m_warp_at_barrier &= ~at_barrier;
        _edgeReleaseBarrier[cta_id] = false;        
        _edgeBarrier[cta_id] = false;
    } else { // Otherwise, forward mark the barrier as released when all warps hit it
        _edgeReleaseBarrier[cta_id] = true;
    }
}


bool barrier_set_t::edgeSetVictimBarrier( unsigned cid, unsigned wid )
{
    // Make sure this is a valid CTA with a barrier hit, that the warp belongs to this CTA, that 
    // the warp is at the barrier, and that no warps from this shader have been interrupted yet. 
    verifyValidCta(cid);
    assert( m_cta_to_warps.find(cid)->second.test(wid) );
    assert( _edgeVictimWarpAtBarrier.none() );
    
    if( !m_warp_at_barrier.test(wid) ) 
        return false;    
    
    _edgeVictimWarpAtBarrier.set(wid);
    m_warp_at_barrier.reset(wid);

    return true;
}

bool barrier_set_t::edgeIsVictimBarrierDone( unsigned wid )
{
    return !_edgeVictimWarpAtBarrier.test(wid);
}

void barrier_set_t::edgeRestoreVictimBarrier( unsigned wid )
{
    assert( !edgeIsVictimBarrierDone(wid) );
    assert( !m_warp_at_barrier.test(wid) );

    _edgeVictimWarpAtBarrier.reset(wid);
    assert( !_edgeVictimWarpAtBarrier.any() );

    m_warp_at_barrier.set(wid);
}

// EDGE: FIXME:
barrier_set_t::barrier_set_t( unsigned max_warps_per_core, unsigned max_cta_per_core )
{
   m_max_warps_per_core = max_warps_per_core;
   m_max_cta_per_core = max_cta_per_core;

   if( max_warps_per_core > WARP_PER_CTA_MAX ) {
      printf("ERROR ** increase WARP_PER_CTA_MAX in shader.h from %u to >= %u or warps per cta in gpgpusim.config\n",
             WARP_PER_CTA_MAX, max_warps_per_core );
      exit(1);
   }
   m_warp_active.reset();
   m_warp_at_barrier.reset();

   for( unsigned i=0; i<max_cta_per_core; ++i ) {
       _edgeBarrier.push_back(false);
       _edgeReleaseBarrier.push_back(false);
   }
}

// during cta allocation
void barrier_set_t::allocate_barrier( unsigned cta_id, warp_set_t warps )
{
   assert( cta_id < m_max_cta_per_core );
   cta_to_warp_t::iterator w=m_cta_to_warps.find(cta_id);
   assert( w == m_cta_to_warps.end() ); // cta should not already be active or allocated barrier resources
   m_cta_to_warps[cta_id] = warps;
   //printf("MARIA DEBUG updating addrs %d \n",  *m_cta_to_warps.find(cta_id));
   assert( m_cta_to_warps.size() <= m_max_cta_per_core ); // catch cta's that were not properly deallocated
  
   m_warp_active |= warps;
   m_warp_at_barrier &= ~warps;
}

// during cta deallocation
void barrier_set_t::deallocate_barrier( unsigned cta_id )
{
   cta_to_warp_t::iterator w=m_cta_to_warps.find(cta_id);
   if( w == m_cta_to_warps.end() )
      return;
   warp_set_t warps = w->second;
   warp_set_t at_barrier = warps & m_warp_at_barrier;
   assert( at_barrier.any() == false ); // no warps stuck at barrier
   warp_set_t active = warps & m_warp_active;
   assert( active.any() == false ); // no warps in CTA still running
   m_warp_active &= ~warps;
   m_warp_at_barrier &= ~warps;
   m_cta_to_warps.erase(w);
}

void barrier_set_t::removeWarpFromCta( unsigned warpId, unsigned ctaId ) 
{
    cta_to_warp_t::iterator w = m_cta_to_warps.find(ctaId);
    if( w == m_cta_to_warps.end() )
        return; // Must have only been a single warp in this CTA, so deallocate_barrier has already been called

    // Verify ctaId is responsible for warpId
    warp_set_t warps = w->second;
    assert( warps.test(warpId) );

    // Verify warpId is not active - this means it just exited, so we can take it freely
    warp_set_t active = warps & m_warp_active;
    assert( !active.test(warpId) );

    // Verify warpId is not at a barrier
    warp_set_t atBarrier = warps & m_warp_at_barrier;
    assert( !atBarrier.test(warpId) );

    // Remove warpId from ctaId
    m_cta_to_warps[ctaId].reset(warpId);
    assert( !m_cta_to_warps[ctaId].test(warpId) );
}

// individual warp hits barrier
void barrier_set_t::warp_reaches_barrier( unsigned cta_id, unsigned warp_id )
{
   cta_to_warp_t::iterator w=m_cta_to_warps.find(cta_id);
   
   if( w == m_cta_to_warps.end() ) { // cta is active
      printf("ERROR ** cta_id %u not found in barrier set on cycle %llu+%llu...\n", cta_id, gpu_tot_sim_cycle, gpu_sim_cycle );
      dump();
      abort();
   }
   assert( w->second.test(warp_id) == true ); // warp is in cta

   m_warp_at_barrier.set(warp_id);

   warp_set_t warps_in_cta = w->second;
   warp_set_t at_barrier = warps_in_cta & m_warp_at_barrier;
   warp_set_t active = warps_in_cta & m_warp_active;

    warp_set_t hasVictimWarp = warps_in_cta & _edgeVictimWarpAtBarrier;

    if( _edgeVictimWarpAtBarrier.any() && hasVictimWarp.any() ) {
        at_barrier |= _edgeVictimWarpAtBarrier;
    }


   //if( (at_barrier & active) == active ) {
   if( at_barrier == active ) {
        if( _edgeBarrier[cta_id] ) { // If this is an EDGE barrier
            if( _edgeReleaseBarrier[cta_id] ) { // Only release the warps if the barrier has been released externally
                _edgeReleaseBarrier[cta_id] = false;
                _edgeBarrier[cta_id] = false;
                m_warp_at_barrier &= ~at_barrier;
            } else {
                // Don't release the warps yet
            }
        } else {
            

            // all warps have reached barrier, so release waiting warps...
            m_warp_at_barrier &= ~at_barrier;
            
            // Was a waiting interrupt warp released? If so, notify that the restore shouldn't return to the barrier
            warp_set_t intWarpAtBar = at_barrier & _edgeVictimWarpAtBarrier;
            if( intWarpAtBar.any() ) {
                _edgeVictimWarpAtBarrier.reset();
            }
        }
   }
}

// fetching a warp
bool barrier_set_t::available_for_fetch( unsigned warp_id ) const
{
   return m_warp_active.test(warp_id) && m_warp_at_barrier.test(warp_id);
}

// warp reaches exit 
void barrier_set_t::warp_exit( unsigned warp_id )
{
   // caller needs to verify all threads in warp are done, e.g., by checking PDOM stack to 
   // see it has only one entry during exit_impl()
   m_warp_active.reset(warp_id);

   // test for barrier release 
   cta_to_warp_t::iterator w=m_cta_to_warps.begin(); 
   for (; w != m_cta_to_warps.end(); ++w) {
      if (w->second.test(warp_id) == true) break;
   }

   warp_set_t warps_in_cta = w->second;
   warp_set_t at_barrier = warps_in_cta & m_warp_at_barrier;
   warp_set_t active = warps_in_cta & m_warp_active;

   if( at_barrier == active ) {
      // all warps have reached barrier, so release waiting warps...
      m_warp_at_barrier &= ~at_barrier;
   }
}

void barrier_set_t::intWarpExit( unsigned intWarpId, unsigned intCtaId )
{
    // caller needs to verify all threads in warp are done, e.g., by checking PDOM stack to 
    // see it has only one entry during exit_impl()
    m_warp_active.reset(intWarpId);

    assert( m_cta_to_warps[intCtaId].test(intWarpId) );

    warp_set_t warps_in_cta = m_cta_to_warps[intCtaId];
    warp_set_t at_barrier = warps_in_cta & m_warp_at_barrier;
    warp_set_t active = warps_in_cta & m_warp_active;

    if( at_barrier == active ) {
        // all warps have reached barrier, so release waiting warps...
        m_warp_at_barrier &= ~at_barrier;
    }
}

// assertions
bool barrier_set_t::warp_waiting_at_barrier( unsigned warp_id ) const
{ 
   return m_warp_at_barrier.test(warp_id);
}

bool barrier_set_t::warp_active( unsigned warp_id ) const
{
    return m_warp_active.test(warp_id);
}

void barrier_set_t::set_at_barrier( unsigned warp_id ) 
{
    m_warp_at_barrier.set(warp_id);
}

void barrier_set_t::set_active( unsigned warp_id )
{
    m_warp_active.set(warp_id);
}



void barrier_set_t::dump() const
{
   printf( "barrier set information\n");
   printf( "  m_max_cta_per_core = %u\n",  m_max_cta_per_core );
   printf( "  m_max_warps_per_core = %u\n", m_max_warps_per_core );
   printf( "  cta_to_warps:\n");
   
   cta_to_warp_t::const_iterator i;
   for( i=m_cta_to_warps.begin(); i!=m_cta_to_warps.end(); i++ ) {
      unsigned cta_id = i->first;
      warp_set_t warps = i->second;
      printf("    cta_id %u : %s\n", cta_id, warps.to_string().c_str() );
   }
   printf("  warp_active: %s\n", m_warp_active.to_string().c_str() );
   printf("  warp_at_barrier: %s\n", m_warp_at_barrier.to_string().c_str() );
   fflush(stdout); 
}

void shader_core_ctx::warp_exit( unsigned warp_id )
{
	bool done = true;
	for (	unsigned i = warp_id*get_config()->warp_size;
			i < (warp_id+1)*get_config()->warp_size;
			i++ ) {

//		if(this->m_thread[i]->m_functional_model_thread_state && this->m_thread[i].m_functional_model_thread_state->donecycle()==0) {
//			done = false;
//		}


		if (m_thread[i] && !m_thread[i]->is_done()) done = false;
	}
	//if (m_warp[warp_id].get_n_completed() == get_config()->warp_size)
	//if (this->m_simt_stack[warp_id]->get_num_entries() == 0)
	if (done) {
        if( m_warp[warp_id].isIntWarp() && m_config->_edgeRunISR ) {
            assert( warp_id == _edgeWid );
            m_barriers.intWarpExit( warp_id, _edgeCid );
        } else {
    		m_barriers.warp_exit( warp_id );
        }
    }
}

bool shader_core_ctx::warp_waiting_at_barrier( unsigned warp_id ) const
{
   return m_barriers.warp_waiting_at_barrier(warp_id);
}

bool shader_core_ctx::warp_waiting_at_mem_barrier( unsigned warp_id ) 
{
   if( !m_warp[warp_id].get_membar() ) {
      return false;
   }
   return true;
}

void shader_core_ctx::set_max_cta( const kernel_info_t &kernel ) 
{
    // calculate the max cta count and cta size for local memory address mapping
    kernel_max_cta_per_shader = m_config->max_cta(kernel);
    unsigned int gpu_cta_size = kernel.threads_per_cta();
    kernel_padded_threads_per_cta = (gpu_cta_size%m_config->warp_size) ? 
        m_config->warp_size*((gpu_cta_size/m_config->warp_size)+1) : 
        gpu_cta_size;
}

void shader_core_ctx::decrement_atomic_count( unsigned wid, unsigned n )
{
   assert( m_warp[wid].get_n_atomic() >= n );
   m_warp[wid].dec_n_atomic(n);
}


bool shader_core_ctx::fetch_unit_response_buffer_full() const
{
    return false;
}

void shader_core_ctx::accept_fetch_response( mem_fetch *mf )
{
    mf->set_status(IN_SHADER_FETCHED,gpu_sim_cycle+gpu_tot_sim_cycle);
    m_L1I->fill(mf,gpu_sim_cycle+gpu_tot_sim_cycle);
}

bool shader_core_ctx::ldst_unit_response_buffer_full() const
{
    return m_ldst_unit->response_buffer_full();
}

void shader_core_ctx::accept_ldst_unit_response(mem_fetch * mf) 
{
   m_ldst_unit->fill(mf);
}

void shader_core_ctx::store_ack( class mem_fetch *mf )
{
	assert( mf->get_type() == WRITE_ACK  || ( m_config->gpgpu_perfect_mem && mf->get_is_write() ) );
    unsigned warp_id = mf->get_wid();
    m_warp[warp_id].dec_store_req();
}

void shader_core_ctx::print_cache_stats( FILE *fp, unsigned& dl1_accesses, unsigned& dl1_misses ) {
   m_ldst_unit->print_cache_stats( fp, dl1_accesses, dl1_misses );
}

void shader_core_ctx::get_cache_stats(cache_stats &cs){
    // Adds stats from each cache to 'cs'
    cs += m_L1I->get_stats(); // Get L1I stats
    m_ldst_unit->get_cache_stats(cs); // Get L1D, L1C, L1T stats
}

void shader_core_ctx::get_L1I_sub_stats(struct cache_sub_stats &css) const{
    if(m_L1I)
        m_L1I->get_sub_stats(css);
}
void shader_core_ctx::get_L1D_sub_stats(struct cache_sub_stats &css) const{
    m_ldst_unit->get_L1D_sub_stats(css);
}
void shader_core_ctx::get_L1C_sub_stats(struct cache_sub_stats &css) const{
    m_ldst_unit->get_L1C_sub_stats(css);
}
void shader_core_ctx::get_L1T_sub_stats(struct cache_sub_stats &css) const{
    m_ldst_unit->get_L1T_sub_stats(css);
}

void shader_core_ctx::get_icnt_power_stats(long &n_simt_to_mem, long &n_mem_to_simt) const{
	n_simt_to_mem += m_stats->n_simt_to_mem[m_sid];
	n_mem_to_simt += m_stats->n_mem_to_simt[m_sid];
}

bool shd_warp_t::functional_done() const
{
    return get_n_completed() == m_warp_size;
}

bool shd_warp_t::hardware_done() const
{
    return functional_done() && stores_done() && !inst_in_pipeline(); 
}

bool shd_warp_t::waiting() 
{
    if ( functional_done() ) {
        // waiting to be initialized with a kernel
        return true;
    } else if ( m_shader->warp_waiting_at_barrier(m_warp_id) ) {
        // waiting for other warps in CTA to reach barrier
        return true;
    } else if ( m_shader->warp_waiting_at_mem_barrier(m_warp_id) ) {
        // waiting for memory barrier
        return true;
    } else if ( m_n_atomic > 0 ) {
        // waiting for atomic operation to complete at memory:
        // this stall is not required for accurate timing model, but rather we
        // stall here since if a call/return instruction occurs in the meantime
        // the functional execution of the atomic when it hits DRAM can cause
        // the wrong register to be read.
        return true;
    }
    return false;
}

void shd_warp_t::startLoad(warp_inst_t& inst, address_type pc, simt_stack* stack)
{
    _inFlightLoadQueue.push_back( PendingLoadInst(inst, pc, *stack) );
}

void shd_warp_t::completeLoad( const warp_inst_t* inst ) 
{
    findInFlightLoad(inst, true);
}

bool shd_warp_t::findInFlightLoad( const warp_inst_t* inst, bool remove )
{
    std::list<PendingLoadInst>::iterator it;
    for( it=_inFlightLoadQueue.begin(); it!=_inFlightLoadQueue.end(); ++it ) {
        if( inst->get_uid() == (*it)._inst.get_uid() ) {
            if( remove ) 
                _inFlightLoadQueue.erase(it);
            return true;
        } 
    }
    return false;
}

// When interrupt a warp, if we have any in-flight loads, just drop them and add the instructions
// that generated them to a replay queue. When the interrupt completes and we restore the original
// warp that was interrupted, replay all of the instructions.
void shd_warp_t::dropLoads() 
{
    assert( _globalLoadPending == _inFlightLoadQueue.size() );
    assert( _replayLoadQueue.empty() );

    unsigned n = _inFlightLoadQueue.size();

    for( unsigned i=0; i<n; ++i ) {
        PendingLoadInst& pli = _inFlightLoadQueue.front();

        m_shader->ldstUnit()->releaseRegisters(&pli._inst);
        EDGE_DPRINT(EdgeDebug, "Preparing to drop load with PC %d of preempted warp %d on SM %d \n",
            pli._pc, pli._inst.warp_id(), m_shader->get_sid());

        _replayLoadQueue.push_back(pli);
        _inFlightLoadQueue.pop_front();
    }

    assert( _inFlightLoadQueue.empty() && (_replayLoadQueue.size() == _globalLoadPending) );
    _globalLoadPending = 0;
}

void shader_core_ctx::addLateInFlightLoad(PendingLoadInst& pli)
{
    // First verify that this load hasn't been seen before
    for( std::vector<PendingLoadInst>::iterator it = _lateInFlightLoads.begin();
            it != _lateInFlightLoads.end(); ++it ) {        
        assert( pli._inst.get_uid() != (*it)._inst.get_uid() );
    }

    _lateInFlightLoads.push_back(pli);
}

// EDGE TODO: Verify the simt stack isn't screwed up now...
void shd_warp_t::replayLoads() 
{
    // Should have nothing in the iBuffer if we're replaying loads
    assert( ibuffer_empty() ); 
    assert( !inst_in_pipeline() );

    address_type spc = 0;
    address_type tnpc = 0;

    // Get the replay queue size and start pushing!
    // Probably only necessary to do the first instruction load. Everything else will
    // just fall into place afterwards. Pushing all of the loads gives us the added
    // bonus of not having to re-access the i$ or redecode the instruction.
    unsigned n = _replayLoadQueue.size();
    for( unsigned i=0; i<n; ++i ) {        
        PendingLoadInst& pli = _replayLoadQueue.front(); 
       
        // Make sure we've already received all of the dropped loads... and if not, make sure NONE of them have come back yet
        if( pli._loadReturned != pli._inst.active_count() ) {
            assert( pli._loadReturned == 0 );
            m_shader->addLateInFlightLoad(pli);
            printf("EDGE: warp %d Adding late in-flight load: uid = %d\n", m_warp_id, pli._inst.get_uid()); 
        }

        address_type tpc = pli._inst.pc;
        if( i==0 ) {
            // If this is the first replay instruction, reset the next PC and update the simt stack
            set_next_pc(tpc);
            m_shader->setSimtStack(m_warp_id, &pli._stack);
            m_shader->setThreadPc(m_warp_id, tpc, pli._inst.get_active_mask());
            spc = tpc;
            tnpc = tpc;
            printf("EDGE:"); 
        } else {
            tnpc += 8;
            if( tpc != tnpc ) {
                printf("EDGE Warning: Replay instructions are not consecutive... something else could have happened in between: "
                        "%ld != %ld: wid=%d, n=%d\n", tpc, tnpc, m_warp_id, n);
                m_shader->incBadReplayLoads();
            }
        }

        printf("EDGE: warp %d replaying load: uid = %d, tpc = %ld (%ld), start_pc = %ld, n = %d\n", m_warp_id, pli._inst.get_uid(), tpc, tnpc, spc, n);

        const warp_inst_t* new_pI = ptx_fetch_inst(tpc);
        assert( new_pI );
        ibuffer_fill(i, new_pI);
        inc_inst_in_pipeline();

        m_shader->incReplayLoads();

        _replayLoadQueue.pop_front();
    }
}

void shader_core_ctx::setThreadPc(int wid, address_type npc, const active_mask_t& active)
{
    int startTid = wid*m_config->warp_size;
    int endTid = startTid + m_config->warp_size;
    unsigned localTid = 0;
    for( unsigned i=startTid; i<endTid; ++i ) {
        if( m_thread[i] && active.test(localTid++) )  {
            m_thread[i]->set_npc(npc);
            m_thread[i]->update_pc(); // EDGE FIXME: Note that m_thread[i]->icount will be off... but maybe that's okay?
        }
    }
}

void shader_core_ctx::setSimtStack(int wid, simt_stack* stack)
{
    assert( m_simt_stack[wid] );
    *m_simt_stack[wid] = *stack;
}


void shd_warp_t::print( FILE *fout ) const
{
    if( !done_exit() ) {
        fprintf( fout, "w%02u (%02u) npc: 0x%04x, done:%c%c%c%c:%2u i:%u s:%u a:%u (done: ", 
                m_dynamic_warp_id,
                m_warp_id,
                m_next_pc,
                (functional_done()?'f':' '),
                (stores_done()?'s':' '),
                (inst_in_pipeline()?' ':'i'),
                (done_exit()?'e':' '),
                n_completed,
                m_inst_in_pipeline, 
                m_stores_outstanding,
                m_n_atomic );
        for (unsigned i = m_warp_id*m_warp_size; i < (m_warp_id+1)*m_warp_size; i++ ) {
          if ( m_shader->ptx_thread_done(i) ) fprintf(fout,"1");
          else fprintf(fout,"0");
          if ( (((i+1)%4) == 0) && (i+1) < (m_warp_id+1)*m_warp_size ) 
             fprintf(fout,",");
        }
        fprintf(fout,") ");
        fprintf(fout," active=%s", m_active_threads.to_string().c_str() );
        fprintf(fout," last fetched @ %5llu, ", m_last_fetch);
        //if( m_imiss_pending ) 
        //    fprintf(fout," i-miss pending");
        
        
        fprintf(fout,"imiss_pend: %s, ", (m_imiss_pending ? "yes" : "no") );
        fprintf(fout,"warp_at_barrier: %s, ", (m_shader->warp_waiting_at_barrier(m_warp_id) ? "yes" : "no") );
        fprintf(fout,"warp_at_mem_barrier: %s ", (m_shader->warp_waiting_at_mem_barrier(m_warp_id) ? "yes" : "no") );
        fprintf(fout,"loads pending: %d, ", _globalLoadPending);
        
        fprintf(fout,"\n");
    }
}

void shd_warp_t::print_ibuffer( FILE *fout ) const
{
    fprintf(fout,"  ibuffer[%2u] : ", m_warp_id );
    for( unsigned i=0; i < IBUFFER_SIZE; i++) {
        const inst_t *inst = m_ibuffer[i].m_inst;
        if( inst ) inst->print_insn(fout);
        else if( m_ibuffer[i].m_valid ) 
           fprintf(fout," <invalid instruction> ");
        else fprintf(fout," <empty> ");
    }
    fprintf(fout,"\n");
}

void opndcoll_rfu_t::add_cu_set(unsigned set_id, unsigned num_cu, unsigned num_dispatch){
    m_cus[set_id].reserve(num_cu); //this is necessary to stop pointers in m_cu from being invalid do to a resize;
    for (unsigned i = 0; i < num_cu; i++) {
        m_cus[set_id].push_back(collector_unit_t());
        m_cu.push_back(&m_cus[set_id].back());
    }
    // for now each collector set gets dedicated dispatch units.
    for (unsigned i = 0; i < num_dispatch; i++) {
        m_dispatch_units.push_back(dispatch_unit_t(&m_cus[set_id]));
    }
}


void opndcoll_rfu_t::add_port(port_vector_t & input, port_vector_t & output, uint_vector_t cu_sets)
{
    //m_num_ports++;
    //m_num_collectors += num_collector_units;
    //m_input.resize(m_num_ports);
    //m_output.resize(m_num_ports);
    //m_num_collector_units.resize(m_num_ports);
    //m_input[m_num_ports-1]=input_port;
    //m_output[m_num_ports-1]=output_port;
    //m_num_collector_units[m_num_ports-1]=num_collector_units;
    m_in_ports.push_back(input_port_t(input,output,cu_sets));
}

void opndcoll_rfu_t::init( unsigned num_banks, shader_core_ctx *shader )
{
   m_shader=shader;
   m_arbiter.init(m_cu.size(),num_banks);
   //for( unsigned n=0; n<m_num_ports;n++ ) 
   //    m_dispatch_units[m_output[n]].init( m_num_collector_units[n] );
   m_num_banks = num_banks;
   m_bank_warp_shift = 0; 
   m_warp_size = shader->get_config()->warp_size;
   m_bank_warp_shift = (unsigned)(int) (log(m_warp_size+0.5) / log(2.0));
   assert( (m_bank_warp_shift == 5) || (m_warp_size != 32) );

   for( unsigned j=0; j<m_cu.size(); j++) {
       m_cu[j]->init(j,num_banks,m_bank_warp_shift,shader->get_config(),this);
   }
   m_initialized=true;
}

int register_bank(int regnum, int wid, unsigned num_banks, unsigned bank_warp_shift)
{
   int bank = regnum;
   if (bank_warp_shift)
      bank += wid;
   return bank % num_banks;
}

bool opndcoll_rfu_t::writeback( const warp_inst_t &inst )
{
   assert( !inst.empty() );
   std::list<unsigned> regs = m_shader->get_regs_written(inst);
   std::list<unsigned>::iterator r;
   unsigned n=0;
   for( r=regs.begin(); r!=regs.end();r++,n++ ) {
      unsigned reg = *r;
      unsigned bank = register_bank(reg,inst.warp_id(),m_num_banks,m_bank_warp_shift);
      
      assert( reg != 0 ); // Shouldn't ever be looking at the NULL register...
      if( m_arbiter.bank_idle(bank) ) {
          m_arbiter.allocate_bank_for_write(bank,op_t(&inst,reg,m_num_banks,m_bank_warp_shift));
      } else {
          return false;
      }
   }
   for(unsigned i=0;i<(unsigned)regs.size();i++){
	      if(m_shader->get_config()->gpgpu_clock_gated_reg_file){
	    	  unsigned active_count=0;
	    	  for(unsigned i=0;i<m_shader->get_config()->warp_size;i=i+m_shader->get_config()->n_regfile_gating_group){
	    		  for(unsigned j=0;j<m_shader->get_config()->n_regfile_gating_group;j++){
	    			  if(inst.get_active_mask().test(i+j)){
	    				  active_count+=m_shader->get_config()->n_regfile_gating_group;
	    				  break;
	    			  }
	    		  }
	    	  }
	    	  m_shader->incregfile_writes(active_count);
	      }else{
	    	  m_shader->incregfile_writes(m_shader->get_config()->warp_size);//inst.active_count());
	      }
   }
   return true;
}

void opndcoll_rfu_t::dispatch_ready_cu()
{
   for( unsigned p=0; p < m_dispatch_units.size(); ++p ) {
      dispatch_unit_t &du = m_dispatch_units[p];
      collector_unit_t *cu = du.find_ready();
      if( cu ) {
    	 for(unsigned i=0;i<(cu->get_num_operands()-cu->get_num_regs());i++){
   	      if(m_shader->get_config()->gpgpu_clock_gated_reg_file){
   	    	  unsigned active_count=0;
   	    	  for(unsigned i=0;i<m_shader->get_config()->warp_size;i=i+m_shader->get_config()->n_regfile_gating_group){
   	    		  for(unsigned j=0;j<m_shader->get_config()->n_regfile_gating_group;j++){
   	    			  if(cu->get_active_mask().test(i+j)){
   	    				  active_count+=m_shader->get_config()->n_regfile_gating_group;
   	    				  break;
   	    			  }
   	    		  }
   	    	  }
   	    	  m_shader->incnon_rf_operands(active_count);
   	      }else{
    		 m_shader->incnon_rf_operands(m_shader->get_config()->warp_size);//cu->get_active_count());
   	      }
    	}
         cu->dispatch();
      }
   }
}

void opndcoll_rfu_t::allocate_cu( unsigned port_num )
{
   input_port_t& inp = m_in_ports[port_num];
   for (unsigned i = 0; i < inp.m_in.size(); i++) {
       if( (*inp.m_in[i]).has_ready() ) {
          //find a free cu 
          for (unsigned j = 0; j < inp.m_cu_sets.size(); j++) {
              std::vector<collector_unit_t> & cu_set = m_cus[inp.m_cu_sets[j]];
	      bool allocated = false;
              for (unsigned k = 0; k < cu_set.size(); k++) {
                  if(cu_set[k].is_free()) {
                     collector_unit_t *cu = &cu_set[k];
                     allocated = cu->allocate(inp.m_in[i],inp.m_out[i]);
                     m_arbiter.add_read_requests(cu);
                     break;
                  }
              }
              if (allocated) break; //cu has been allocated, no need to search more.
          }
          break; // can only service a single input, if it failed it will fail for others.
       }
   }
}

void opndcoll_rfu_t::allocate_reads()
{
   // process read requests that do not have conflicts
   std::list<op_t> allocated = m_arbiter.allocate_reads();
   std::map<unsigned,op_t> read_ops;
   for( std::list<op_t>::iterator r=allocated.begin(); r!=allocated.end(); r++ ) {
      const op_t &rr = *r;
      unsigned reg = rr.get_reg();
      unsigned wid = rr.get_wid();
      unsigned bank = register_bank(reg,wid,m_num_banks,m_bank_warp_shift);
      m_arbiter.allocate_for_read(bank,rr);
      read_ops[bank] = rr;
   }
   std::map<unsigned,op_t>::iterator r;
   for(r=read_ops.begin();r!=read_ops.end();++r ) {
      op_t &op = r->second;
      unsigned cu = op.get_oc_id();
      unsigned operand = op.get_operand();
      m_cu[cu]->collect_operand(operand);
      if(m_shader->get_config()->gpgpu_clock_gated_reg_file){
    	  unsigned active_count=0;
    	  for(unsigned i=0;i<m_shader->get_config()->warp_size;i=i+m_shader->get_config()->n_regfile_gating_group){
    		  for(unsigned j=0;j<m_shader->get_config()->n_regfile_gating_group;j++){
    			  if(op.get_active_mask().test(i+j)){
    				  active_count+=m_shader->get_config()->n_regfile_gating_group;
    				  break;
    			  }
    		  }
    	  }
    	  m_shader->incregfile_reads(active_count);
      }else{
    	  m_shader->incregfile_reads(m_shader->get_config()->warp_size);//op.get_active_count());
      }
  }
} 

bool opndcoll_rfu_t::collector_unit_t::ready() const 
{ 
   return (!m_free) && m_not_ready.none() && (*m_output_register).has_free(); 
}

void opndcoll_rfu_t::collector_unit_t::dump(FILE *fp, const shader_core_ctx *shader ) const
{
   if( m_free ) {
      fprintf(fp,"    <free>\n");
   } else {
      m_warp->print(fp);
      for( unsigned i=0; i < MAX_REG_OPERANDS*2; i++ ) {
         if( m_not_ready.test(i) ) {
            std::string r = m_src_op[i].get_reg_string();
            fprintf(fp,"    '%s' not ready\n", r.c_str() );
         }
      }
   }
}

void opndcoll_rfu_t::collector_unit_t::init( unsigned n, 
                                             unsigned num_banks, 
                                             unsigned log2_warp_size,
                                             const core_config *config,
                                             opndcoll_rfu_t *rfu ) 
{ 
   m_rfu=rfu;
   m_cuid=n; 
   m_num_banks=num_banks;
   assert(m_warp==NULL); 
   m_warp = new warp_inst_t(config);
   m_bank_warp_shift=log2_warp_size;
}

bool opndcoll_rfu_t::collector_unit_t::allocate( register_set* pipeline_reg_set, register_set* output_reg_set ) 
{
   assert(m_free);
   assert(m_not_ready.none());
   m_free = false;
   m_output_register = output_reg_set;
   warp_inst_t **pipeline_reg = pipeline_reg_set->get_ready();
   if( (pipeline_reg) and !((*pipeline_reg)->empty()) ) {
      m_warp_id = (*pipeline_reg)->warp_id();
      for( unsigned op=0; op < MAX_REG_OPERANDS; op++ ) {
         int reg_num = (*pipeline_reg)->arch_reg.src[op]; // this math needs to match that used in function_info::ptx_decode_inst
         if( reg_num >= 0 ) { // valid register
            m_src_op[op] = op_t( this, op, reg_num, m_num_banks, m_bank_warp_shift );
            m_not_ready.set(op);
         } else 
            m_src_op[op] = op_t();
      }
      //move_warp(m_warp,*pipeline_reg);
      pipeline_reg_set->move_out_to(m_warp);
      return true;
   }
   return false;
}

void opndcoll_rfu_t::collector_unit_t::dispatch()
{
   assert( m_not_ready.none() );
   //move_warp(*m_output_register,m_warp);
   m_output_register->move_in(m_warp);
   m_free=true;
   m_output_register = NULL;
   for( unsigned i=0; i<MAX_REG_OPERANDS*2;i++)
      m_src_op[i].reset();
}

simt_core_cluster::simt_core_cluster( class gpgpu_sim *gpu, 
                                      unsigned cluster_id, 
                                      const struct shader_core_config *config, 
                                      const struct memory_config *mem_config,
                                      shader_core_stats *stats, 
                                      class memory_stats_t *mstats )
{
    m_config = config;
    m_cta_issue_next_core=m_config->n_simt_cores_per_cluster-1; // this causes first launch to use hw cta 0
    m_cluster_id=cluster_id;
    m_gpu = gpu;
    m_stats = stats;
    m_memory_stats = mstats;
    m_core = new shader_core_ctx*[ config->n_simt_cores_per_cluster ];
    for( unsigned i=0; i < config->n_simt_cores_per_cluster; i++ ) {
        unsigned sid = m_config->cid_to_sid(i,m_cluster_id);
        m_core[i] = new shader_core_ctx(gpu,this,sid,m_cluster_id,config,mem_config,stats);
        m_core_sim_order.push_back(i); 
    }
}

bool simt_core_cluster::isIWarpRunning() const 
{
    for( unsigned i=0; i<m_config->n_simt_cores_per_cluster; ++i ) {
        if( m_core[i]->isIWarpRunning() )
            return true;
    }
    return false;
}

bool simt_core_cluster::intInProgress() const 
{
    for( unsigned i=0; i<m_config->n_simt_cores_per_cluster; ++i ) {
        if( m_core[i]->intInProgress() )
            return true;
    }
    return false;
}

void simt_core_cluster::core_cycle()
{
    for( std::list<unsigned>::iterator it = m_core_sim_order.begin(); it != m_core_sim_order.end(); ++it ) {
        m_core[*it]->cycle();

        // EDGE
        m_core[*it]->edgeIntCycle();
    }

    if (m_config->simt_core_sim_order == 1) {
        m_core_sim_order.splice(m_core_sim_order.end(), m_core_sim_order, m_core_sim_order.begin()); 
    }
}

void simt_core_cluster::reinit()
{
    for( unsigned i=0; i < m_config->n_simt_cores_per_cluster; i++ ) 
        m_core[i]->reinit(0,m_config->n_thread_per_shader,true);
}

unsigned simt_core_cluster::max_cta( const kernel_info_t &kernel )
{
    return m_config->n_simt_cores_per_cluster * m_config->max_cta(kernel);
}

unsigned simt_core_cluster::get_not_completed() const
{
    unsigned not_completed=0;
    for( unsigned i=0; i < m_config->n_simt_cores_per_cluster; i++ ) 
        not_completed += m_core[i]->get_not_completed();
    return not_completed;
}

void simt_core_cluster::print_not_completed( FILE *fp ) const
{
    for( unsigned i=0; i < m_config->n_simt_cores_per_cluster; i++ ) {
        unsigned not_completed=m_core[i]->get_not_completed();
        unsigned sid=m_config->cid_to_sid(i,m_cluster_id);
        fprintf(fp,"%u(%u) ", sid, not_completed );
    }
}

unsigned simt_core_cluster::get_n_active_cta() const
{
    unsigned n=0;
    for( unsigned i=0; i < m_config->n_simt_cores_per_cluster; i++ ) 
        n += m_core[i]->get_n_active_cta();
    return n;
}

unsigned simt_core_cluster::get_n_active_sms() const
{
    unsigned n=0;
    for( unsigned i=0; i < m_config->n_simt_cores_per_cluster; i++ )
        n += m_core[i]->isactive();
    return n;
}

// EDGE: Modified the scheduling and completion conditions to take the iWarp into account. 
// If the iWarp is running on a core, it's still safe to schedule a kernel there. 
// When scheduling a CTA, if the iKernel is already running then there should be max_cta+1 total CTAs to run.
// On completion of a kernel, issue the completion even if the iWarp is running. TODO: Should experiement with this 
//      to see how much it impacts performance of the other process and the iKernel. 
//
// CDP - Concurrent kernels
unsigned simt_core_cluster::issue_block2core()
{
    ////////////////////////////////////////////////////////////////////////////////////////
    //////////// FIXME: Need to include the iKernel stuff here
    ////////////////////////////////////////////////////////////////////////////////////////    
    unsigned num_blocks_issued=0;
    for( unsigned i=0; i < m_config->n_simt_cores_per_cluster; i++ ) {
        unsigned core = (i+m_cta_issue_next_core+1)%m_config->n_simt_cores_per_cluster;
        
        kernel_info_t* kernel;
        
        if( m_config->gpgpu_concurrent_kernel_sm ) { // Concurrent kernel on sm
            // Always select latest issued kernel
            kernel = m_gpu->select_kernel(m_cluster_id);
        } else {
            // First select core kernel, if no more CTAs, get a new kernel only when the core completes
            kernel = m_core[core]->get_kernel();
            if( !m_gpu->kernel_more_cta_left(kernel) ) {
                if( m_core[core]->get_not_completed() == 0 || m_core[core]->onlyIWarpRunning() ) {
                    kernel = m_gpu->select_kernel(m_cluster_id);
                    if( kernel ) 
                        m_core[core]->set_kernel(kernel);
                }
            }
        }

        if( (m_config->_edgeEventReserveSm > 0 || m_config->_edgeEventPreemptSm > 0) && kernel ) {
            assert( !m_config->_edgeEventReserveCta );
            if( ((m_cluster_id < m_config->_edgeEventReserveSm) && kernel->isEventKernel()) || ((m_cluster_id < m_config->_edgeEventPreemptSm) && kernel->isEventKernel()) ||
                    ((m_cluster_id >= m_config->_edgeEventReserveSm) && (!kernel->isEventKernel() || m_config->_edgeUseAllSmsForEventKernels) )) {
                        
                
                    if( m_gpu->kernel_more_cta_left(kernel) && m_core[core]->can_issue_1block(*kernel) ) {
                        if( !kernel->hasStarted() ) {
                            kernel->start();
                            //printf("MARIA DEBUG kernel %s has started at cycle %d at core %d\n", kernel->entry()->get_name().c_str(), gpu_sim_cycle + gpu_tot_sim_cycle, core);
                        }
                        m_core[core]->issue_block2core(*kernel);
                        num_blocks_issued++;
                        m_cta_issue_next_core=core;
                        break;
                    }
            }
        } else {
            if( kernel && (m_config->_edgeEventReserveCta > 0) && kernel->isEventKernel() ) {
                
                if( m_gpu->kernel_more_cta_left(kernel) ) {
                    int availIdx = m_core[core]->freeReservedEventAvailable();
                    if( availIdx != -1 ) {
                        m_core[core]->setReservedEvent(availIdx);
                        if( !kernel->hasStarted() ) {
                            //printf("MARIA DEBUG kernel %s has started at cycle %d at core %d\n", kernel->entry()->get_name().c_str(), gpu_sim_cycle + gpu_tot_sim_cycle, core);
                            kernel->start();
                        }
                        num_blocks_issued++;
                        m_cta_issue_next_core=core;
                        m_core[core]->issueEventBlock2Core(*kernel, availIdx);
                    }
                }

            } else {
                if( m_gpu->kernel_more_cta_left(kernel) && m_core[core]->can_issue_1block(*kernel) ) {
                    if( !kernel->hasStarted() ) {
                        //printf("MARIA DEBUG kernel %s has started at cycle %d at core %d\n", kernel->entry()->get_name().c_str(), gpu_sim_cycle + gpu_tot_sim_cycle, core);
                        kernel->start();
                    }
                    printf("GPGPU-Sim: Shader %d issuing CTA from kernel %s\n", 
                            core, kernel->name().c_str()); 
                    m_core[core]->issue_block2core(*kernel);
                    num_blocks_issued++;
                    m_cta_issue_next_core=core;
                    break;
                } 
            }
            
        }
    }


    // Check for any finished kernels on this cycle and initiate the completion
    for (unsigned i=0; i < m_config->n_simt_cores_per_cluster; i++) {
        // Tayler - CDP - Concurrent kernel
        kernel_info_t* k = m_core[i]->nextKernelToComplete();
        if( k && !m_core[i]->kernelFinishIssued(k) ) 
            m_core[i]->startKernelFinish(k);
    }

    ////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////

    return num_blocks_issued;
}

void simt_core_cluster::cache_flush()
{
    for( unsigned i=0; i < m_config->n_simt_cores_per_cluster; i++ ) 
        m_core[i]->cache_flush();
}

bool simt_core_cluster::icnt_injection_buffer_full(unsigned size, bool write)
{
    unsigned request_size = size;
    if (!write) 
        request_size = READ_PACKET_SIZE;
    return ! ::icnt_has_buffer(m_cluster_id, request_size);
}

void simt_core_cluster::icnt_inject_request_packet(class mem_fetch *mf)
{
    // stats
    if (mf->get_is_write()) m_stats->made_write_mfs++;
    else m_stats->made_read_mfs++;
    switch (mf->get_access_type()) {
    case CONST_ACC_R: m_stats->gpgpu_n_mem_const++; break;
    case TEXTURE_ACC_R: m_stats->gpgpu_n_mem_texture++; break;
    case GLOBAL_ACC_R: m_stats->gpgpu_n_mem_read_global++; break;
    case GLOBAL_ACC_W: m_stats->gpgpu_n_mem_write_global++; break;
    case LOCAL_ACC_R: m_stats->gpgpu_n_mem_read_local++; break;
    case LOCAL_ACC_W: m_stats->gpgpu_n_mem_write_local++; break;
    case INST_ACC_R: m_stats->gpgpu_n_mem_read_inst++; break;
    case L1_WRBK_ACC: m_stats->gpgpu_n_mem_write_global++; break;
    case L2_WRBK_ACC: m_stats->gpgpu_n_mem_l2_writeback++; break;
    case L1_WR_ALLOC_R: m_stats->gpgpu_n_mem_l1_write_allocate++; break;
    case L2_WR_ALLOC_R: m_stats->gpgpu_n_mem_l2_write_allocate++; break;
    default: assert(0);
    }

   // The packet size varies depending on the type of request: 
   // - For write request and atomic request, the packet contains the data 
   // - For read request (i.e. not write nor atomic), the packet only has control metadata
   unsigned int packet_size = mf->size(); 
   if (!mf->get_is_write() && !mf->isatomic()) {
      packet_size = mf->get_ctrl_size(); 
   }
   m_stats->m_outgoing_traffic_stats->record_traffic(mf, packet_size); 
   unsigned destination = mf->get_sub_partition_id();
   mf->set_status(IN_ICNT_TO_MEM,gpu_sim_cycle+gpu_tot_sim_cycle);
   if (!mf->get_is_write() && !mf->isatomic())
      ::icnt_push(m_cluster_id, m_config->mem2device(destination), (void*)mf, mf->get_ctrl_size() );
   else 
      ::icnt_push(m_cluster_id, m_config->mem2device(destination), (void*)mf, mf->size());
}

void simt_core_cluster::icnt_cycle()
{
    if( !m_response_fifo.empty() ) {
        mem_fetch *mf = m_response_fifo.front();
        unsigned cid = m_config->sid_to_cid(mf->get_sid());
        if( mf->get_access_type() == INST_ACC_R ) {
            // instruction fetch response
            if( !m_core[cid]->fetch_unit_response_buffer_full() ) {
                m_response_fifo.pop_front();
                m_core[cid]->accept_fetch_response(mf);
            }
        } else {
            // data response
            if( !m_core[cid]->ldst_unit_response_buffer_full() ) {
                m_response_fifo.pop_front();
                m_memory_stats->memlatstat_read_done(mf);
                m_core[cid]->accept_ldst_unit_response(mf);
            }
        }
    }
    if( m_response_fifo.size() < m_config->n_simt_ejection_buffer_size ) {
        mem_fetch *mf = (mem_fetch*) ::icnt_pop(m_cluster_id);
        if (!mf) 
            return;
        assert(mf->get_tpc() == m_cluster_id);
        assert(mf->get_type() == READ_REPLY || mf->get_type() == WRITE_ACK );

        // The packet size varies depending on the type of request: 
        // - For read request and atomic request, the packet contains the data 
        // - For write-ack, the packet only has control metadata
        unsigned int packet_size = (mf->get_is_write())? mf->get_ctrl_size() : mf->size(); 
        m_stats->m_incoming_traffic_stats->record_traffic(mf, packet_size); 
        mf->set_status(IN_CLUSTER_TO_SHADER_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
        //m_memory_stats->memlatstat_read_done(mf,m_shader_config->max_warps_per_shader);
        m_response_fifo.push_back(mf);
        m_stats->n_mem_to_simt[m_cluster_id] += mf->get_num_flits(false);
    }
}

void simt_core_cluster::get_pdom_stack_top_info( unsigned sid, unsigned tid, unsigned *pc, unsigned *rpc ) const
{
    unsigned cid = m_config->sid_to_cid(sid);
    m_core[cid]->get_pdom_stack_top_info(tid,pc,rpc);
}

void simt_core_cluster::display_pipeline( unsigned sid, FILE *fout, int print_mem, int mask )
{
    m_core[m_config->sid_to_cid(sid)]->display_pipeline(fout,print_mem,mask);

    fprintf(fout,"\n");
    fprintf(fout,"Cluster %u pipeline state\n", m_cluster_id );
    fprintf(fout,"Response FIFO (occupancy = %zu):\n", m_response_fifo.size() );
    for( std::list<mem_fetch*>::const_iterator i=m_response_fifo.begin(); i != m_response_fifo.end(); i++ ) {
        const mem_fetch *mf = *i;
        mf->print(fout);
    }
}

void simt_core_cluster::print_cache_stats( FILE *fp, unsigned& dl1_accesses, unsigned& dl1_misses ) const {
   for ( unsigned i = 0; i < m_config->n_simt_cores_per_cluster; ++i ) {
      m_core[ i ]->print_cache_stats( fp, dl1_accesses, dl1_misses );
   }
}

void simt_core_cluster::get_icnt_stats(long &n_simt_to_mem, long &n_mem_to_simt) const {
	long simt_to_mem=0;
	long mem_to_simt=0;
	for ( unsigned i = 0; i < m_config->n_simt_cores_per_cluster; ++i ) {
		m_core[i]->get_icnt_power_stats(simt_to_mem, mem_to_simt);
	}
	n_simt_to_mem = simt_to_mem;
	n_mem_to_simt = mem_to_simt;
}

void simt_core_cluster::get_cache_stats(cache_stats &cs) const{
    for ( unsigned i = 0; i < m_config->n_simt_cores_per_cluster; ++i ) {
        m_core[i]->get_cache_stats(cs);
    }
}

void simt_core_cluster::get_L1I_sub_stats(struct cache_sub_stats &css) const{
    struct cache_sub_stats temp_css;
    struct cache_sub_stats total_css;
    temp_css.clear();
    total_css.clear();
    for ( unsigned i = 0; i < m_config->n_simt_cores_per_cluster; ++i ) {
        m_core[i]->get_L1I_sub_stats(temp_css);
        total_css += temp_css;
    }
    css = total_css;
}
void simt_core_cluster::get_L1D_sub_stats(struct cache_sub_stats &css) const{
    struct cache_sub_stats temp_css;
    struct cache_sub_stats total_css;
    temp_css.clear();
    total_css.clear();
    for ( unsigned i = 0; i < m_config->n_simt_cores_per_cluster; ++i ) {
        m_core[i]->get_L1D_sub_stats(temp_css);
        total_css += temp_css;
    }
    css = total_css;
}
void simt_core_cluster::get_L1C_sub_stats(struct cache_sub_stats &css) const{
    struct cache_sub_stats temp_css;
    struct cache_sub_stats total_css;
    temp_css.clear();
    total_css.clear();
    for ( unsigned i = 0; i < m_config->n_simt_cores_per_cluster; ++i ) {
        m_core[i]->get_L1C_sub_stats(temp_css);
        total_css += temp_css;
    }
    css = total_css;
}
void simt_core_cluster::get_L1T_sub_stats(struct cache_sub_stats &css) const{
    struct cache_sub_stats temp_css;
    struct cache_sub_stats total_css;
    temp_css.clear();
    total_css.clear();
    for ( unsigned i = 0; i < m_config->n_simt_cores_per_cluster; ++i ) {
        m_core[i]->get_L1T_sub_stats(temp_css);
        total_css += temp_css;
    }
    css = total_css;
}

void shader_core_ctx::checkExecutionStatusAndUpdate(warp_inst_t &inst, unsigned t, unsigned tid)
{
    if( inst.has_callback(t) ) 
       m_warp[inst.warp_id()].inc_n_atomic();
       inc_total_n_atomic();

    if (inst.space.is_local() && (inst.is_load() || inst.is_store())) {
        new_addr_type localaddrs[MAX_ACCESSES_PER_INSN_PER_THREAD];
        unsigned num_addrs;
        num_addrs = translate_local_memaddr( inst.get_addr(t), 
                                             tid, 
                                             m_config->n_simt_clusters*m_config->n_simt_cores_per_cluster,               
                                             inst.data_size, 
                                             (new_addr_type*) localaddrs );
        inst.set_addr(t, (new_addr_type*) localaddrs, num_addrs);
    }
    if ( ptx_thread_done(tid) ) {
        m_warp[inst.warp_id()].set_completed(t);
        m_warp[inst.warp_id()].ibuffer_flush();
    }

    // PC-Histogram Update 
    unsigned warp_id = inst.warp_id(); 
    unsigned pc = inst.pc; 
    for (unsigned t = 0; t < m_config->warp_size; t++) {
        if (inst.active(t)) {
            int tid = warp_id * m_config->warp_size + t; 
            cflog_update_thread_pc(m_sid, tid, pc);  
        }
    }
}

void simt_core_cluster::initEDGE(kernel_info_t* k)
{
    for(unsigned i=0; i<m_config->n_simt_cores_per_cluster; ++i) 
        m_core[i]->initEDGE(k);
}
