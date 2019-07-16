// Copyright (c) 2009-2011, Tor M. Aamodt, Inderpreet Singh, Timothy Rogers,
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



#include "abstract_hardware_model.h"
#include "cuda-sim/memory.h"
#include "cuda-sim/ptx_ir.h"
#include "cuda-sim/ptx-stats.h"
#include "cuda-sim/cuda-sim.h"
#include "gpgpu-sim/gpu-sim.h"
#include "option_parser.h"
#include <algorithm>

#include "gpu/gpgpu-sim/cuda_gpu.hh"

extern gpgpu_sim *g_the_gpu;

unsigned mem_access_t::sm_next_access_uid = 0;   
unsigned warp_inst_t::sm_next_uid = 0;

void move_warp( warp_inst_t *&dst, warp_inst_t *&src )
{
   assert( dst->empty() );
   warp_inst_t* temp = dst;
   dst = src;
   src = temp;
   src->clear();
}


void gpgpu_functional_sim_config::reg_options(class OptionParser * opp)
{
	option_parser_register(opp, "-gpgpu_ptx_use_cuobjdump", OPT_BOOL,
                 &m_ptx_use_cuobjdump,
                 "Use cuobjdump to extract ptx and sass from binaries",
#if (CUDART_VERSION >= 4000)
                 "1"
#else
                 "0"
#endif
                 );
	option_parser_register(opp, "-gpgpu_experimental_lib_support", OPT_BOOL,
	                 &m_experimental_lib_support,
	                 "Try to extract code from cuda libraries [Broken because of unknown cudaGetExportTable]",
	                 "0");
    option_parser_register(opp, "-gpgpu_ptx_convert_to_ptxplus", OPT_BOOL,
                 &m_ptx_convert_to_ptxplus,
                 "Convert SASS (native ISA) to ptxplus and run ptxplus",
                 "0");
    option_parser_register(opp, "-gpgpu_ptx_force_max_capability", OPT_UINT32,
                 &m_ptx_force_max_capability,
                 "Force maximum compute capability",
                 "0");
   option_parser_register(opp, "-gpgpu_ptx_inst_debug_to_file", OPT_BOOL, 
                &g_ptx_inst_debug_to_file, 
                "Dump executed instructions' debug information to file", 
                "0");
   option_parser_register(opp, "-gpgpu_ptx_inst_debug_file", OPT_CSTR, &g_ptx_inst_debug_file, 
                  "Executed instructions' debug output file",
                  "inst_debug.txt");
   option_parser_register(opp, "-gpgpu_ptx_inst_debug_thread_uid", OPT_INT32, &g_ptx_inst_debug_thread_uid, 
               "Thread UID for executed instructions' debug output", 
               "1");
}

void gpgpu_functional_sim_config::ptx_set_tex_cache_linesize(unsigned linesize)
{
   m_texcache_linesize = linesize;
}

gpgpu_t::gpgpu_t( const gpgpu_functional_sim_config &config, CudaGPU *cuda_gpu )
    : gem5CudaGPU(cuda_gpu), m_function_model_config(config)
{
   m_global_mem = NULL; // Accesses to global memory should go through gem5-gpu
   m_tex_mem = new memory_space_impl<8192>("tex",64*1024);
   m_surf_mem = new memory_space_impl<8192>("surf",64*1024);

   m_dev_malloc=GLOBAL_HEAP_START; 

   if(m_function_model_config.get_ptx_inst_debug_to_file() != 0) 
      ptx_inst_debug_file = fopen(m_function_model_config.get_ptx_inst_debug_file(), "w");

   if(cuda_gpu) {
       sharedMemDelay = cuda_gpu->getSharedMemDelay();
   } else {
       sharedMemDelay = 1;
   }
}

address_type line_size_based_tag_func(new_addr_type address, new_addr_type line_size)
{
   //gives the tag for an address based on a given line size
   return address & ~(line_size-1);
}

const char * mem_access_type_str(enum mem_access_type access_type)
{
   #define MA_TUP_BEGIN(X) static const char* access_type_str[] = {
   #define MA_TUP(X) #X
   #define MA_TUP_END(X) };
   MEM_ACCESS_TYPE_TUP_DEF
   #undef MA_TUP_BEGIN
   #undef MA_TUP
   #undef MA_TUP_END

   assert(access_type < NUM_MEM_ACCESS_TYPE); 

   return access_type_str[access_type]; 
}


void warp_inst_t::clear_active( const active_mask_t &inactive ) {
    active_mask_t test = m_warp_active_mask;
    test &= inactive;
    assert( test == inactive ); // verify threads being disabled were active
    m_warp_active_mask &= ~inactive;
}

void warp_inst_t::set_not_active( unsigned lane_id ) {
    m_warp_active_mask.reset(lane_id);
}

void warp_inst_t::set_active( const active_mask_t &active ) {
   m_warp_active_mask = active;
   if( m_isatomic ) {
      for( unsigned i=0; i < m_config->warp_size; i++ ) {
         if( !m_warp_active_mask.test(i) ) {
             m_per_scalar_thread[i].callback.function = NULL;
             m_per_scalar_thread[i].callback.instruction = NULL;
             m_per_scalar_thread[i].callback.thread = NULL;
         }
      }
   }
}

void warp_inst_t::do_atomic(bool forceDo) {
    do_atomic( m_warp_active_mask,forceDo );
}

void warp_inst_t::do_atomic( const active_mask_t& access_mask,bool forceDo ) {
    assert( m_isatomic && (!m_empty||forceDo) );
    for( unsigned i=0; i < m_config->warp_size; i++ )
    {
        if( access_mask.test(i) )
        {
            dram_callback_t &cb = m_per_scalar_thread[i].callback;
            if( cb.thread )
                cb.function(cb.instruction, cb.thread);
        }
    }
}

void warp_inst_t::generate_mem_accesses()
{
    if( empty() || op == MEMORY_BARRIER_OP || m_mem_accesses_created ) 
        return;
    if ( !((op == LOAD_OP) || (op == STORE_OP)) )
        return; 
    if( m_warp_active_mask.count() == 0 ) 
        return; // predicated off

    // In gem5-gpu, global, const and local references go through the gem5-gpu LSQ
    // EDGE: Now adding kernel parameter memory as well to Gem5
    if( space.get_type() == global_space || space.get_type() == const_space || 
            space.get_type() == local_space || space.get_type() == param_space_kernel )
        return;

    const size_t starting_queue_size = m_accessq.size();

    assert( is_load() || is_store() );
    assert( m_per_scalar_thread_valid ); // need address information per thread

    bool is_write = is_store();

    mem_access_type access_type = NUM_MEM_ACCESS_TYPE;
    switch (space.get_type()) {
    case param_space_kernel: 
        access_type = CONST_ACC_R; 
        break;
    case tex_space: 
        access_type = TEXTURE_ACC_R;   
        break;
    case const_space:
    case global_space:       
        access_type = is_write? GLOBAL_ACC_W: GLOBAL_ACC_R;   
        break;
    case local_space:
    case param_space_local:  
        access_type = is_write? LOCAL_ACC_W: LOCAL_ACC_R;   
        break;
    case shared_space: break;
    default: assert(0); break; 
    }

    // Calculate memory accesses generated by this warp
    new_addr_type cache_block_size = 0; // in bytes 

    switch( space.get_type() ) {
    case shared_space: {
        unsigned subwarp_size = m_config->warp_size / m_config->mem_warp_parts;
        unsigned total_accesses=0;
        for( unsigned subwarp=0; subwarp <  m_config->mem_warp_parts; subwarp++ ) {

            // data structures used per part warp 
            std::map<unsigned,std::map<new_addr_type,unsigned> > bank_accs; // bank -> word address -> access count

            // step 1: compute accesses to words in banks
            for( unsigned thread=subwarp*subwarp_size; thread < (subwarp+1)*subwarp_size; thread++ ) {
                if( !active(thread) ) 
                    continue;
                new_addr_type addr = m_per_scalar_thread[thread].memreqaddr[0];
                //FIXME: deferred allocation of shared memory should not accumulate across kernel launches
                //assert( addr < m_config->gpgpu_shmem_size ); 
                unsigned bank = m_config->shmem_bank_func(addr);
                new_addr_type word = line_size_based_tag_func(addr,m_config->WORD_SIZE);
                bank_accs[bank][word]++;
            }

            if (m_config->shmem_limited_broadcast) {
                // step 2: look for and select a broadcast bank/word if one occurs
                bool broadcast_detected = false;
                new_addr_type broadcast_word=(new_addr_type)-1;
                unsigned broadcast_bank=(unsigned)-1;
                std::map<unsigned,std::map<new_addr_type,unsigned> >::iterator b;
                for( b=bank_accs.begin(); b != bank_accs.end(); b++ ) {
                    unsigned bank = b->first;
                    std::map<new_addr_type,unsigned> &access_set = b->second;
                    std::map<new_addr_type,unsigned>::iterator w;
                    for( w=access_set.begin(); w != access_set.end(); ++w ) {
                        if( w->second > 1 ) {
                            // found a broadcast
                            broadcast_detected=true;
                            broadcast_bank=bank;
                            broadcast_word=w->first;
                            break;
                        }
                    }
                    if( broadcast_detected ) 
                        break;
                }
            
                // step 3: figure out max bank accesses performed, taking account of broadcast case
                unsigned max_bank_accesses=0;
                for( b=bank_accs.begin(); b != bank_accs.end(); b++ ) {
                    unsigned bank_accesses=0;
                    std::map<new_addr_type,unsigned> &access_set = b->second;
                    std::map<new_addr_type,unsigned>::iterator w;
                    for( w=access_set.begin(); w != access_set.end(); ++w ) 
                        bank_accesses += w->second;
                    if( broadcast_detected && broadcast_bank == b->first ) {
                        for( w=access_set.begin(); w != access_set.end(); ++w ) {
                            if( w->first == broadcast_word ) {
                                unsigned n = w->second;
                                assert(n > 1); // or this wasn't a broadcast
                                assert(bank_accesses >= (n-1));
                                bank_accesses -= (n-1);
                                break;
                            }
                        }
                    }
                    if( bank_accesses > max_bank_accesses ) 
                        max_bank_accesses = bank_accesses;
                }

                // step 4: accumulate
                total_accesses+= max_bank_accesses;
            } else {
                // step 2: look for the bank with the maximum number of access to different words 
                unsigned max_bank_accesses=0;
                std::map<unsigned,std::map<new_addr_type,unsigned> >::iterator b;
                for( b=bank_accs.begin(); b != bank_accs.end(); b++ ) {
                    max_bank_accesses = std::max(max_bank_accesses, (unsigned)b->second.size());
                }

                // step 3: accumulate
                total_accesses+= max_bank_accesses;
            }
        }
        assert( total_accesses > 0 && total_accesses <= m_config->warp_size );
        cycles = total_accesses * g_the_gpu->sharedMemDelay; // shared memory conflicts modeled as larger initiation interval
        ptx_file_line_stats_add_smem_bank_conflict( pc, total_accesses );
        break;
    }

    case tex_space: 
        cache_block_size = m_config->gpgpu_cache_texl1_linesize;
        break;
    case param_space_kernel:
        cache_block_size = m_config->gpgpu_cache_constl1_linesize; 
        break;

    case const_space: case global_space: case local_space: case param_space_local:
        if( m_config->gpgpu_coalesce_arch == 13 ) {
           if(isatomic())
               memory_coalescing_arch_13_atomic(is_write, access_type);
           else
               memory_coalescing_arch_13(is_write, access_type);
        } else abort();

        break;

    default:
        abort();
    }

    if( cache_block_size ) {
        assert( m_accessq.empty() );
        mem_access_byte_mask_t byte_mask; 
        std::map<new_addr_type,active_mask_t> accesses; // block address -> set of thread offsets in warp
        std::map<new_addr_type,active_mask_t>::iterator a;
        for( unsigned thread=0; thread < m_config->warp_size; thread++ ) {
            if( !active(thread) ) 
                continue;
            new_addr_type addr = m_per_scalar_thread[thread].memreqaddr[0];
            new_addr_type block_address = line_size_based_tag_func(addr,cache_block_size);
            accesses[block_address].set(thread);
            unsigned idx = addr-block_address; 
            for( unsigned i=0; i < data_size; i++ ) 
                byte_mask.set(idx+i);
        }
        for( a=accesses.begin(); a != accesses.end(); ++a ) 
            m_accessq.push_back( mem_access_t(access_type,a->first,cache_block_size,is_write,a->second,byte_mask) );
    }

    if ( space.get_type() == global_space ) {
        ptx_file_line_stats_add_uncoalesced_gmem( pc, m_accessq.size() - starting_queue_size );
    }
    m_mem_accesses_created=true;
}

void warp_inst_t::memory_coalescing_arch_13( bool is_write, mem_access_type access_type )
{
    // see the CUDA manual where it discusses coalescing rules before reading this
    unsigned segment_size = 0;
    unsigned warp_parts = m_config->mem_warp_parts;
    switch( data_size ) {
    case 1: segment_size = 32; break;
    case 2: segment_size = 64; break;
    case 4: case 8: case 16: segment_size = 128; break;
    }
    unsigned subwarp_size = m_config->warp_size / warp_parts;

    for( unsigned subwarp=0; subwarp <  warp_parts; subwarp++ ) {
        std::map<new_addr_type,transaction_info> subwarp_transactions;

        // step 1: find all transactions generated by this subwarp
        for( unsigned thread=subwarp*subwarp_size; thread<subwarp_size*(subwarp+1); thread++ ) {
            if( !active(thread) )
                continue;

            unsigned data_size_coales = data_size;
            unsigned num_accesses = 1;

            if( space.get_type() == local_space || space.get_type() == param_space_local ) {
               // Local memory accesses >4B were split into 4B chunks
               if(data_size >= 4) {
                  data_size_coales = 4;
                  num_accesses = data_size/4;
               }
               // Otherwise keep the same data_size for sub-4B access to local memory
            }


            assert(num_accesses <= MAX_ACCESSES_PER_INSN_PER_THREAD);

            for(unsigned access=0; access<num_accesses; access++) {
                new_addr_type addr = m_per_scalar_thread[thread].memreqaddr[access];
                new_addr_type block_address = line_size_based_tag_func(addr,segment_size);
                unsigned chunk = (addr&127)/32; // which 32-byte chunk within in a 128-byte chunk does this thread access?
                transaction_info &info = subwarp_transactions[block_address];

                // can only write to one segment
                assert(block_address == line_size_based_tag_func(addr+data_size_coales-1,segment_size));

                info.chunks.set(chunk);
                info.active.set(thread);
                unsigned idx = (addr&127);
                for( unsigned i=0; i < data_size_coales; i++ )
                    info.bytes.set(idx+i);
            }
        }

        // step 2: reduce each transaction size, if possible
        std::map< new_addr_type, transaction_info >::iterator t;
        for( t=subwarp_transactions.begin(); t !=subwarp_transactions.end(); t++ ) {
            new_addr_type addr = t->first;
            const transaction_info &info = t->second;

            memory_coalescing_arch_13_reduce_and_send(is_write, access_type, info, addr, segment_size);

        }
    }
}

void warp_inst_t::memory_coalescing_arch_13_atomic( bool is_write, mem_access_type access_type )
{

   assert(space.get_type() == global_space); // Atomics allowed only for global memory

   // see the CUDA manual where it discusses coalescing rules before reading this
   unsigned segment_size = 0;
   unsigned warp_parts = 2;
   switch( data_size ) {
   case 1: segment_size = 32; break;
   case 2: segment_size = 64; break;
   case 4: case 8: case 16: segment_size = 128; break;
   }
   unsigned subwarp_size = m_config->warp_size / warp_parts;

   for( unsigned subwarp=0; subwarp <  warp_parts; subwarp++ ) {
       std::map<new_addr_type,std::list<transaction_info> > subwarp_transactions; // each block addr maps to a list of transactions

       // step 1: find all transactions generated by this subwarp
       for( unsigned thread=subwarp*subwarp_size; thread<subwarp_size*(subwarp+1); thread++ ) {
           if( !active(thread) )
               continue;

           new_addr_type addr = m_per_scalar_thread[thread].memreqaddr[0];
           unsigned block_address = line_size_based_tag_func(addr,segment_size);
           unsigned chunk = (addr&127)/32; // which 32-byte chunk within in a 128-byte chunk does this thread access?

           // can only write to one segment
           assert(block_address == line_size_based_tag_func(addr+data_size-1,segment_size));

           // Find a transaction that does not conflict with this thread's accesses
           bool new_transaction = true;
           std::list<transaction_info>::iterator it;
           transaction_info* info;
           for(it=subwarp_transactions[block_address].begin(); it!=subwarp_transactions[block_address].end(); it++) {
              unsigned idx = (addr&127);
              if(not it->test_bytes(idx,idx+data_size-1)) {
                 new_transaction = false;
                 info = &(*it);
                 break;
              }
           }
           if(new_transaction) {
              // Need a new transaction
              subwarp_transactions[block_address].push_back(transaction_info());
              info = &subwarp_transactions[block_address].back();
           }
           assert(info);

           info->chunks.set(chunk);
           info->active.set(thread);
           unsigned idx = (addr&127);
           for( unsigned i=0; i < data_size; i++ ) {
               assert(!info->bytes.test(idx+i));
               info->bytes.set(idx+i);
           }
       }

       // step 2: reduce each transaction size, if possible
       std::map< new_addr_type, std::list<transaction_info> >::iterator t_list;
       for( t_list=subwarp_transactions.begin(); t_list !=subwarp_transactions.end(); t_list++ ) {
           // For each block addr
           new_addr_type addr = t_list->first;
           const std::list<transaction_info>& transaction_list = t_list->second;

           std::list<transaction_info>::const_iterator t;
           for(t=transaction_list.begin(); t!=transaction_list.end(); t++) {
               // For each transaction
               const transaction_info &info = *t;
               memory_coalescing_arch_13_reduce_and_send(is_write, access_type, info, addr, segment_size);
           }
       }
   }
}

void warp_inst_t::memory_coalescing_arch_13_reduce_and_send( bool is_write, mem_access_type access_type, const transaction_info &info, new_addr_type addr, unsigned segment_size )
{
   assert( (addr & (segment_size-1)) == 0 );

   const std::bitset<4> &q = info.chunks;
   assert( q.count() >= 1 );
   std::bitset<2> h; // halves (used to check if 64 byte segment can be compressed into a single 32 byte segment)

   unsigned size=segment_size;
   if( segment_size == 128 ) {
       bool lower_half_used = q[0] || q[1];
       bool upper_half_used = q[2] || q[3];
       if( lower_half_used && !upper_half_used ) {
           // only lower 64 bytes used
           size = 64;
           if(q[0]) h.set(0);
           if(q[1]) h.set(1);
       } else if ( (!lower_half_used) && upper_half_used ) {
           // only upper 64 bytes used
           addr = addr+64;
           size = 64;
           if(q[2]) h.set(0);
           if(q[3]) h.set(1);
       } else {
           assert(lower_half_used && upper_half_used);
       }
   } else if( segment_size == 64 ) {
       // need to set halves
       if( (addr % 128) == 0 ) {
           if(q[0]) h.set(0);
           if(q[1]) h.set(1);
       } else {
           assert( (addr % 128) == 64 );
           if(q[2]) h.set(0);
           if(q[3]) h.set(1);
       }
   }
   if( size == 64 ) {
       bool lower_half_used = h[0];
       bool upper_half_used = h[1];
       if( lower_half_used && !upper_half_used ) {
           size = 32;
       } else if ( (!lower_half_used) && upper_half_used ) {
           addr = addr+32;
           size = 32;
       } else {
           assert(lower_half_used && upper_half_used);
       }
   }
   m_accessq.push_back( mem_access_t(access_type,addr,size,is_write,info.active,info.bytes) );
}

void warp_inst_t::completed( unsigned long long cycle ) const 
{
   unsigned long long latency = cycle - issue_cycle; 
   assert(latency <= cycle); // underflow detection 
   ptx_file_line_stats_add_latency(pc, latency * active_count());  
}


KernelStats::KernelStats()
{
    reset();
}

KernelStats::KernelStats(const KernelStats& rhs)
{
    _nInsn = rhs._nInsn;
    _nCycles = rhs._nCycles;
    _nInstStallCycles = rhs._nInstStallCycles;
    _nWarpInstStalls = rhs._nWarpInstStalls;
    _nWarpInstIssued = rhs._nWarpInstIssued;
    _TotalLatency = rhs._TotalLatency;
    _edgePreemptionQueueWait = rhs._edgePreemptionQueueWait;
    _edgePreemptionLen = rhs._edgePreemptionLen;
    _totalDataMemAccessLatency = rhs._totalDataMemAccessLatency;
    _numberOfDataLoads = rhs._numberOfDataLoads;
    _totalInstMemAccessLatency = rhs._totalInstMemAccessLatency;
    _numberOfInstLoads = rhs._numberOfInstLoads;
    _edgeCompleted = rhs._edgeCompleted;
}

KernelStats::~KernelStats()
{

}

void KernelStats::reset()
{
    _nInsn = 0;
    _nCycles = 0;
    _nInstStallCycles = 0;
    _nWarpInstStalls = 0;
    _nWarpInstIssued = 0;
    _TotalLatency = 0;
    _edgePreemptionQueueWait = 0;
    _edgePreemptionLen = 0;
    _totalDataMemAccessLatency = 0;
    _numberOfDataLoads = 0;
    _totalInstMemAccessLatency = 0;
    _numberOfInstLoads = 0;
    _edgeCompleted = 0;
}


void KernelStats::operator+=(const KernelStats& rhs)
{
    _nInsn += rhs._nInsn;
    _nCycles += rhs._nCycles;
    _nWarpInstStalls += rhs._nWarpInstStalls;
    _nInstStallCycles += rhs._nInstStallCycles;
    _nWarpInstIssued += rhs._nWarpInstIssued;
    _TotalLatency += rhs._TotalLatency;
    _edgePreemptionQueueWait += rhs._edgePreemptionQueueWait;
    _edgePreemptionLen += rhs._edgePreemptionLen;
    _totalDataMemAccessLatency += rhs._totalDataMemAccessLatency;
    _numberOfDataLoads += rhs._numberOfDataLoads;
    _totalInstMemAccessLatency += rhs._totalInstMemAccessLatency;
    _numberOfInstLoads += rhs._numberOfInstLoads;
}

void KernelStats::print(FILE* f) const
{
    fprintf(f, "kernel_n_insn = %lld\n", _nInsn);
    fprintf(f, "kernel_n_cycles = %lld\n", _nCycles);
    fprintf(f, "kernel_IPC = %.4lf\n", (double)_nInsn / (double)_nCycles);
    fprintf(f, "kernel_n_warp_inst_issued = %lld\n", _nWarpInstIssued);
    fprintf(f, "kernel_total_n_warp_inst_stalls = %lld\n", _nWarpInstStalls);
    fprintf(f, "kernel_avg_inst_stall_cycles_per_warp = %.4lf\n", (double)_nInstStallCycles / (double)_nWarpInstStalls);
    fprintf(f, "event_kernel_cycles_since_interrupt = %lld\n", _TotalLatency);
    fprintf(f, "event_kernel_cycles_wait_in_preemption_queue = %lld\n", _edgePreemptionQueueWait);
    fprintf(f, "event_kernel_preemption_len_in_cycles = %lld\n", _edgePreemptionLen);
    fprintf(f, "averageDataMemAccessLatency = %lld\n", (double)_totalDataMemAccessLatency / (double)_numberOfDataLoads);
    fprintf(f, "numberOfDataLoads = %lld\n", _numberOfDataLoads);
    fprintf(f, "averageInstMemAccessLatency = %lld\n", (double)_totalInstMemAccessLatency / (double)_numberOfInstLoads);
    fprintf(f, "numberOfInstLoads = %lld\n", _numberOfInstLoads);
}

void KernelStats::print(FILE* f, std::string& name) const 
{
    fprintf(f, "%s_n_insn = %lld\n", name.c_str(), _nInsn);
    fprintf(f, "%s_n_cycles = %lld\n", name.c_str(), _nCycles);
    fprintf(f, "%s_IPC = %.4lf\n", name.c_str(), (double)_nInsn / (double)_nCycles);
    fprintf(f, "%s_n_warp_inst_issued = %lld\n", name.c_str(), _nWarpInstIssued);
    fprintf(f, "%s_total_n_warp_inst_stalls = %lld\n", name.c_str(), _nWarpInstStalls);
    fprintf(f, "%s_avg_inst_stall_cycles_per_warp = %.4lf\n", name.c_str(), (double)_nInstStallCycles / (double)_nWarpInstStalls);
    fprintf(f, "%s event_kernel_cycles_since_interrupt = %lld\n", name.c_str(), _TotalLatency);
    fprintf(f, "%s event_kernel_cycles_wait_in_preemption_queue = %lld\n", name.c_str(), _edgePreemptionQueueWait);
    fprintf(f, "%s event_kernel_preemption_len_in_cycles = %lld\n", name.c_str(), _edgePreemptionLen);
}

void KernelStats::print(FILE* f, std::string& name, unsigned num) const 
{
    fprintf(f, "%s_avg_n_insn = %.4lf\n", name.c_str(), (double)_nInsn / (double)num );
    fprintf(f, "%s_avg_n_cycles = %.4lf\n", name.c_str(), (double)_nCycles / (double)num );
    fprintf(f, "%s_avg_IPC = %.4lf\n", name.c_str(), ((double)_nInsn / (double)_nCycles) );
    fprintf(f, "%s_avg_n_warp_inst_issued = %.4lf\n", name.c_str(), (double)_nWarpInstIssued / (double)num );
    fprintf(f, "%s_avg_total_n_warp_inst_stalls = %.4lf\n", name.c_str(), (double)_nWarpInstStalls / (double)num );
    fprintf(f, "%s_avg_avg_inst_stall_cycles_per_warp = %.4lf\n", name.c_str(), ((double)_nInstStallCycles / (double)_nWarpInstStalls) );
    fprintf(f, "%s avg_event_kernel_cycles_since_interrupt = %.4lf\n", name.c_str(), ((double)_TotalLatency / (double)num));
    fprintf(f, "%s avg_event_kernel_cycles_wait_in_preemption_queue = %.4lf\n", name.c_str(), ((double)_edgePreemptionQueueWait / (double)num));
    fprintf(f, "%s avg_event_kernel_preemption_len_in_cycles = %.4lf\n", name.c_str(), ((double)_edgePreemptionLen / (double)num));
}


#define CONV_CTA_NINSN 3800
#define MM_CTA_NINSN 15624
#define BP1_CTA_NINSN 1161
#define BP2_CTA_NINSN 580
//for BFS there is a high variability. 240-3000. 
//Its really hard to know "how many instr left". Taking the maximum
#define BFS1_CTA_NINSN 4590
#define BFS2_CTA_NINSN 450
#define INITMEMC1_CTA_NINSN 448
#define INITMEMC2_CTA_NINSN 4000

//the proper way to implement this function is by extracting the relevant number
//from ptx similat to what is done for regs. this is just a hack to get it working
//for pact rebuttal
unsigned kernel_info_t::GetTotalWarpInsnPerCta() {
    std::string convName("_Z20filterActs_YxX_colorILi4ELi32ELi1ELi4ELi1ELb0ELb1EEvPfS0_S0_iiiiiiiiiiffi"); 
    std::string MmName("matrixMul");
    std::string BPName1("_Z22bpnn_layerforward_CUDAPfS_S_S_ii"); 
    std::string BPName2("_Z24bpnn_adjust_weights_cudaPfiS_iS_S_"); 
    std::string BFSName1("_Z6KernelP4NodePiPbS2_S2_S1_i");   
    std::string BFSName2("_Z7Kernel2PbS_S_S_i");
    std::string initMemcName1("_Z18initDataStructuresv");   
    std::string InitMemcName2("_Z13initHashTableP10SetRequestjj");    
    
    if (!convName.compare(entry()->get_name().c_str())) {
        return CONV_CTA_NINSN;
    }       
    if (!MmName.compare(entry()->get_name().c_str())) {
        return MM_CTA_NINSN;
    }
    if (!BPName1.compare(entry()->get_name().c_str())) {
        return BP1_CTA_NINSN;
    }
    if (!BPName2.compare(entry()->get_name().c_str())) {
        return BP2_CTA_NINSN;
    }
    if (!BFSName1.compare(entry()->get_name().c_str())) {
        return BFS1_CTA_NINSN;
    }
    if (!BFSName2.compare(entry()->get_name().c_str())) {
        return BFS2_CTA_NINSN;
    }
    if (!initMemcName1.compare(entry()->get_name().c_str())) {
        return INITMEMC1_CTA_NINSN;
    }
    if (!InitMemcName2.compare(entry()->get_name().c_str())) {
        return INITMEMC2_CTA_NINSN;
    }
    printf("GetTotalWarpInsnPerCta: Unknown kernel %s, aborting \n", entry()->get_name().c_str());
    abort();
}

void kernel_info_t::printStats(FILE* f) const
{
    fprintf(f, "===============================\n");
    fprintf(f, "Kernel_name = %s\n", m_kernel_entry->get_name().c_str());
    _stats.print(f);
    fprintf(f, "===============================\n");
}

unsigned kernel_info_t::m_next_uid = 1;

kernel_info_t::kernel_info_t( dim3 gridDim, dim3 blockDim, class function_info *entry )
{
    m_kernel_entry=entry;
    m_grid_dim=gridDim;
    m_block_dim=blockDim;
    m_next_cta.x=0;
    m_next_cta.y=0;
    m_next_cta.z=0;
    m_next_tid=m_next_cta;
    m_num_cores_running=0;
    m_uid = m_next_uid++;
    //m_param_mem = new memory_space_impl<8192>("param",64*1024);
    m_param_mem = NULL; // EDGE: No longer allocating/managing parameter memory in GPGPU-Sim, now in Gem5

    // EDGE: Save initial state for reset
    _initialGridDim = gridDim;
    _initialBlockDim = blockDim;

    _eventKernel = false;
    _stats.reset();
    _started = false;
    _finishing = false;

    start_cycle = 0;
    end_cycle = 0;
    launch_cycle = 0;
}

kernel_info_t::kernel_info_t( const kernel_info_t& rhs )
{
    m_uid               = m_next_uid++;
    m_kernel_entry      = rhs.m_kernel_entry;
    m_grid_dim          = rhs._initialGridDim;
    m_block_dim         = rhs._initialBlockDim;
    _initialGridDim     = rhs._initialGridDim;
    _initialBlockDim    = rhs._initialBlockDim;
    m_next_cta.x        = 0;
    m_next_cta.y        = 0;
    m_next_cta.z        = 0;
    m_next_tid          = m_next_cta;
    m_num_cores_running = 0;
    m_param_mem         = rhs.m_param_mem;
    _eventKernel        = rhs._eventKernel;
    _paramMem           = NULL; // This will need to be set afterwards for the new kernel
    _stats              = rhs._stats;
    _started            = rhs._started;
    _finishing          = rhs._finishing;
    start_cycle = rhs.start_cycle;
    end_cycle = rhs.end_cycle;
    launch_cycle = rhs.launch_cycle;

    m_inst_text_base_vaddr = rhs.m_inst_text_base_vaddr;
    m_active_threads.clear();
}

kernel_info_t::kernel_info_t( const kernel_info_t* rhs )
{
    m_uid               = m_next_uid++;
    m_kernel_entry      = rhs->m_kernel_entry;
    m_grid_dim          = rhs->_initialGridDim;
    m_block_dim         = rhs->_initialBlockDim;
    _initialGridDim     = rhs->_initialGridDim;
    _initialBlockDim    = rhs->_initialBlockDim;
    m_next_cta.x        = 0;
    m_next_cta.y        = 0;
    m_next_cta.z        = 0;
    m_next_tid          = m_next_cta;
    m_num_cores_running = 0;
    m_param_mem         = rhs->m_param_mem;
    _eventKernel        = rhs->_eventKernel;
    _paramMem           = NULL; // This will need to be set afterwards for the new kernel
    _stats              = rhs->_stats;
    _started            = rhs->_started;
    _finishing          = rhs->_finishing;
    start_cycle = rhs->start_cycle;
    end_cycle = rhs->end_cycle;
    launch_cycle = rhs->launch_cycle;
    m_inst_text_base_vaddr = rhs->m_inst_text_base_vaddr;
    m_active_threads.clear();
}


kernel_info_t::~kernel_info_t()
{
    assert( m_active_threads.empty() );
    if( m_param_mem ) delete m_param_mem; // EDGE
}


void kernel_info_t::resetKernel()
{
    assert( m_active_threads.empty() );
    m_grid_dim          = _initialGridDim;
    m_block_dim         = _initialBlockDim;
    m_next_cta.x        = 0;
    m_next_cta.y        = 0;
    m_next_cta.z        = 0;
    m_next_tid          = m_next_cta;
    m_num_cores_running = 0;
    _started            = false;
    _finishing          = false;
    start_cycle = 0;
    end_cycle = 0;
    launch_cycle = 0;
    _stats.reset();
}

std::string kernel_info_t::name() const
{
    return m_kernel_entry->get_name();
}

bool kernel_info_t::isISRKernel() const
{
    bool ret = false;
    if( m_kernel_entry ) {
        ret = m_kernel_entry->isISRKernel();
    }
    return (ret || _ISRLikeKernel);
}

simt_stack::simt_stack( unsigned wid, unsigned warpSize)
{
    m_warp_id=wid;
    m_warp_size = warpSize;
    reset();
}

void simt_stack::reset()
{
    m_stack.clear();
}

void simt_stack::launch( address_type start_pc, const simt_mask_t &active_mask )
{
    reset();
    simt_stack_entry new_stack_entry;
    new_stack_entry.m_pc = start_pc;
    new_stack_entry.m_calldepth = 1;
    new_stack_entry.m_active_mask = active_mask;
    new_stack_entry.m_type = STACK_ENTRY_TYPE_NORMAL;
    m_stack.push_back(new_stack_entry);
}

const simt_mask_t &simt_stack::get_active_mask() const
{
    assert(m_stack.size() > 0);
    return m_stack.back().m_active_mask;
}

void simt_stack::get_pdom_stack_top_info( unsigned *pc, unsigned *rpc ) const
{
   assert(m_stack.size() > 0);
   *pc = m_stack.back().m_pc;
   *rpc = m_stack.back().m_recvg_pc;
}

unsigned simt_stack::get_rp() const 
{ 
    assert(m_stack.size() > 0);
    return m_stack.back().m_recvg_pc;
}

void simt_stack::print (FILE *fout) const
{
    for ( unsigned k=0; k < m_stack.size(); k++ ) {
        simt_stack_entry stack_entry = m_stack[k];
        if ( k==0 ) {
            fprintf(fout, "w%02d %1u ", m_warp_id, k );
        } else {
            fprintf(fout, "    %1u ", k );
        }
        for (unsigned j=0; j<m_warp_size; j++)
            fprintf(fout, "%c", (stack_entry.m_active_mask.test(j)?'1':'0') );
        fprintf(fout, " pc: 0x%03x", stack_entry.m_pc );
        if ( stack_entry.m_recvg_pc == (unsigned)-1 ) {
            fprintf(fout," rp: ---- tp: %s cd: %2u ", (stack_entry.m_type==STACK_ENTRY_TYPE_CALL?"C":"N"), stack_entry.m_calldepth );
        } else {
            fprintf(fout," rp: %4u tp: %s cd: %2u ", stack_entry.m_recvg_pc, (stack_entry.m_type==STACK_ENTRY_TYPE_CALL?"C":"N"), stack_entry.m_calldepth );
        }
        if ( stack_entry.m_branch_div_cycle != 0 ) {
            fprintf(fout," bd@%6u ", (unsigned) stack_entry.m_branch_div_cycle );
        } else {
            fprintf(fout," " );
        }
        ptx_print_insn( stack_entry.m_pc, fout );
        fprintf(fout,"\n");
    }
}

void simt_stack::update( simt_mask_t &thread_done, addr_vector_t &next_pc, address_type recvg_pc, op_type next_inst_op )
{
    assert(m_stack.size() > 0);

    assert( next_pc.size() == m_warp_size );

    simt_mask_t  top_active_mask = m_stack.back().m_active_mask;
    address_type top_recvg_pc = m_stack.back().m_recvg_pc;
    address_type top_pc = m_stack.back().m_pc; // the pc of the instruction just executed
    stack_entry_type top_type = m_stack.back().m_type;

    assert(top_active_mask.any());

    const address_type null_pc = -1;
    bool warp_diverged = false;
    address_type new_recvg_pc = null_pc;
    while (top_active_mask.any()) {

        // extract a group of threads with the same next PC among the active threads in the warp
        address_type tmp_next_pc = null_pc;
        simt_mask_t tmp_active_mask;
        for (int i = m_warp_size - 1; i >= 0; i--) {
            if ( top_active_mask.test(i) ) { // is this thread active?
                if (thread_done.test(i)) {
                    top_active_mask.reset(i); // remove completed thread from active mask
                } else if (tmp_next_pc == null_pc) {
                    tmp_next_pc = next_pc[i];
                    tmp_active_mask.set(i);
                    top_active_mask.reset(i);
                } else if (tmp_next_pc == next_pc[i]) {
                    tmp_active_mask.set(i);
                    top_active_mask.reset(i);
                }
            }
        }

        if(tmp_next_pc == null_pc) {
            assert(!top_active_mask.any()); // all threads done
            continue;
        }

        // HANDLE THE SPECIAL CASES FIRST
        if (next_inst_op== CALL_OPS)
        {
            // Since call is not a divergent instruction, all threads should have executed a call instruction
            assert(top_active_mask.any() == false);

            simt_stack_entry new_stack_entry;
            new_stack_entry.m_pc = tmp_next_pc;
            new_stack_entry.m_active_mask = tmp_active_mask;
            new_stack_entry.m_branch_div_cycle = gpu_sim_cycle+gpu_tot_sim_cycle;
            new_stack_entry.m_type = STACK_ENTRY_TYPE_CALL;
            m_stack.push_back(new_stack_entry);
            return;

        } else if(next_inst_op == RET_OPS && top_type==STACK_ENTRY_TYPE_CALL) {
            // pop the CALL Entry
            assert(top_active_mask.any() == false);
            m_stack.pop_back();

            assert(m_stack.size() > 0);
            m_stack.back().m_pc=tmp_next_pc;// set the PC of the stack top entry to return PC from  the call stack;
            // Check if the New top of the stack is reconverging
            if (tmp_next_pc == m_stack.back().m_recvg_pc && m_stack.back().m_type!=STACK_ENTRY_TYPE_CALL)
            {
                assert(m_stack.back().m_type==STACK_ENTRY_TYPE_NORMAL);
                m_stack.pop_back();
            }
            return;
        }

        // discard the new entry if its PC matches with reconvergence PC
        // that automatically reconverges the entry
        // If the top stack entry is CALL, dont reconverge.
        if (tmp_next_pc == top_recvg_pc && (top_type != STACK_ENTRY_TYPE_CALL)) continue;

        // this new entry is not converging
        // if this entry does not include thread from the warp, divergence occurs
        if (top_active_mask.any() && !warp_diverged ) {
            warp_diverged = true;
            // modify the existing top entry into a reconvergence entry in the pdom stack
            new_recvg_pc = recvg_pc;
            if (new_recvg_pc != top_recvg_pc) {
                m_stack.back().m_pc = new_recvg_pc;
                m_stack.back().m_branch_div_cycle = gpu_sim_cycle+gpu_tot_sim_cycle;

                m_stack.push_back(simt_stack_entry());
            }
        }

        // discard the new entry if its PC matches with reconvergence PC
        if (warp_diverged && tmp_next_pc == new_recvg_pc) continue;

        // update the current top of pdom stack
        m_stack.back().m_pc = tmp_next_pc;
        m_stack.back().m_active_mask = tmp_active_mask;
        if (warp_diverged) {
            m_stack.back().m_calldepth = 0;
            m_stack.back().m_recvg_pc = new_recvg_pc;
        } else {
            m_stack.back().m_recvg_pc = top_recvg_pc;
        }

        m_stack.push_back(simt_stack_entry());
    }
    assert(m_stack.size() > 0);
    m_stack.pop_back();


    if (warp_diverged) {
        ptx_file_line_stats_add_warp_divergence(top_pc, 1); 
    }
}

void core_t::execute_warp_inst_t(warp_inst_t &inst, unsigned warpId)
{
    for ( unsigned t=0; t < m_warp_size; t++ ) {
        if( inst.active(t) ) {
            if(warpId==(unsigned (-1)))
                warpId = inst.warp_id();
            unsigned tid=m_warp_size*warpId+t;
            m_thread[tid]->ptx_exec_inst(inst,t);
            
            //virtual function
            checkExecutionStatusAndUpdate(inst,t,tid);
        }
    } 
}

void core_t::writeRegister(const warp_inst_t &inst, unsigned warpSize, unsigned lane_id, char *data) {
    assert(inst.active(lane_id));
    int warpId = inst.warp_id();
    m_thread[warpSize*warpId+lane_id]->writeRegister(inst, lane_id, data);
}
  
bool  core_t::ptx_thread_done( unsigned hw_thread_id ) const  
{
    return ((m_thread[ hw_thread_id ]==NULL) || m_thread[ hw_thread_id ]->is_done());
}
  
void core_t::updateSIMTStack(unsigned warpId, warp_inst_t * inst)
{
    simt_mask_t thread_done;
    addr_vector_t next_pc;
    unsigned wtid = warpId * m_warp_size;
    for (unsigned i = 0; i < m_warp_size; i++) {
        if( ptx_thread_done(wtid+i) ) {
            thread_done.set(i);
            next_pc.push_back( (address_type)-1 );
        } else {
            if( inst->reconvergence_pc == RECONVERGE_RETURN_PC ) 
                inst->reconvergence_pc = get_return_pc(m_thread[wtid+i]);
            next_pc.push_back( m_thread[wtid+i]->get_pc() );
        }
    }
    m_simt_stack[warpId]->update(thread_done,next_pc,inst->reconvergence_pc, inst->op);
}

//! Get the warp to be executed using the data taken form the SIMT stack
warp_inst_t core_t::getExecuteWarp(unsigned warpId)
{
    unsigned pc,rpc;
    m_simt_stack[warpId]->get_pdom_stack_top_info(&pc,&rpc);
    warp_inst_t wi= *ptx_fetch_inst(pc);
    wi.set_active(m_simt_stack[warpId]->get_active_mask());
    return wi;
}

void core_t::deleteSIMTStack()
{
    if ( m_simt_stack ) {
        for (unsigned i = 0; i < m_warp_count; ++i) 
            delete m_simt_stack[i];
        delete[] m_simt_stack;
        m_simt_stack = NULL;
    }
}

void core_t::initilizeSIMTStack(unsigned warp_count, unsigned warp_size)
{ 
    m_simt_stack = new simt_stack*[warp_count];
    for (unsigned i = 0; i < warp_count; ++i) 
        m_simt_stack[i] = new simt_stack(i,warp_size);
    m_warp_size = warp_size;
    m_warp_count = warp_count;
}

void core_t::get_pdom_stack_top_info( unsigned warpId, unsigned *pc, unsigned *rpc ) const
{
    m_simt_stack[warpId]->get_pdom_stack_top_info(pc,rpc);
}

//ugly hack

struct IPv4ParameMem
{
    struct pkt_hdr_normal* packet_buf;
    int* gpu_tbl24;
    int* gpu_tbl8;
    unsigned n;
    int* reg_buffer;
    bool save_regs;
};

// void kernel_info_t::EdgeSetRegSaving(bool save_regs) {
//     std::string ipv4Name("ipv4_fwd_kernel");
//     int save_regs_offset = 0;

//     if (!ipv4Name.compare(entry()->get_name())) {
//         IPv4ParameMem* addr = (IPv4ParameMem*) _paramMem;
//         printf("DEBUG MARIA trying to update addr %lld for save regs. paramem=%lld \n",
//                 addr, _paramMem);
//         Write2GPUMem(addr, save_regs, 1);
//         //addr->save_regs = save_regs;
//     }
// }       
