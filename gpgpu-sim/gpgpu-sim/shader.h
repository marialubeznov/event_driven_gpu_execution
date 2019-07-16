// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung, Andrew Turner,
// Ali Bakhoda 
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

#ifndef SHADER_H
#define SHADER_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <map>
#include <set>
#include <vector>
#include <queue>
#include <list>
#include <bitset>
#include <utility>
#include <algorithm>
#include <deque>

//#include "../cuda-sim/ptx.tab.h"

#include "delayqueue.h"
#include "stack.h"
#include "dram.h"
#include "../abstract_hardware_model.h"
#include "scoreboard.h"
#include "mem_fetch.h"
#include "stats.h"
#include "gpu-cache.h"
#include "traffic_breakdown.h"

#define NO_OP_FLAG            0xFF

/* READ_PACKET_SIZE:
   bytes: 6 address (flit can specify chanel so this gives up to ~2GB/channel, so good for now),
          2 bytes   [shaderid + mshrid](14 bits) + req_size(0-2 bits if req_size variable) - so up to 2^14 = 16384 mshr total 
 */

#define READ_PACKET_SIZE 8

//WRITE_PACKET_SIZE: bytes: 6 address, 2 miscelaneous. 
#define WRITE_PACKET_SIZE 8

#define WRITE_MASK_SIZE 8

//Set a hard limit of 32 CTAs per shader [cuda only has 8]
// EDGE: Setting to 32 + _edgeEventCtasPerCore for the interrupt CTAs
//#define MAX_CTA_PER_SHADER 32
#define MAX_CTA_PER_SHADER 64 

enum EdgeIntStates {
    IDLE,
    SELECT_WARP,
    FLUSH_PIPELINE,
    SAVE_HW_CTX,
    LAUNCH_IWARP,
    IWARP_RUNNING,
    IWARP_COMPLETING,
    RESTORE_HW_CTX,
    NUM_EDGE_INT_STATES
};

typedef unsigned long long EdgeStat_t;

struct PendingLoadInst {
    PendingLoadInst() { _valid = false; _loadReturned = 0; _pc = 0; }
    PendingLoadInst(warp_inst_t& inst, address_type pc, simt_stack& stack) 
    { 
        _valid = true; 
        _loadReturned = 0;
        _inst = inst; 
        _pc = pc; 
        _stack = stack;
    }
    bool _valid;
    unsigned _loadReturned;
    warp_inst_t _inst;
    address_type _pc;
    simt_stack _stack;
};

class EdgeIntStallStats {
public:
    EdgeIntStallStats() : _instInPipeline(0), _iMissPending(0), _atBarrier(0), _atMemBarrier(0),
        _atomics(0), _scoreboardRegPending(0), _pendingLoads(0), _preemptionQueue(0)
    {
    }

    void operator+=(const EdgeIntStallStats& rhs)
    {
        _instInPipeline += rhs._instInPipeline;
        _iMissPending += rhs._iMissPending;
        _atBarrier += rhs._atBarrier;
        _atMemBarrier += rhs._atMemBarrier;
        _atomics += rhs._atomics;
        _scoreboardRegPending += rhs._scoreboardRegPending;
        _pendingLoads += rhs._pendingLoads;
    }

    void printBreakdown(unsigned totalNumStalls)
    {
        float t = (float)totalNumStalls;
        printf( "total_int_warp_stall_breakdown = %.4lf:%.4lf:%.4lf:%.4lf:%.4lf:%.4lf:%.4lf\n",
                ((float)_instInPipeline / t), ((float)_iMissPending / t), 
                ((float)_atBarrier / t), ((float)_atMemBarrier / t), 
                ((float)_atomics / t), ((float)_scoreboardRegPending / t),
                ((float)_pendingLoads / t) );
    }


    EdgeStat_t _instInPipeline;
    EdgeStat_t _iMissPending;
    EdgeStat_t _atBarrier;
    EdgeStat_t _atMemBarrier;
    EdgeStat_t _atomics;
    EdgeStat_t _scoreboardRegPending;
    EdgeStat_t _pendingLoads;
    EdgeStat_t _preemptionQueue;
};




class CTAKernelMap : public std::map<unsigned, kernel_info_t*> {};

class thread_ctx_t {
public:

    thread_ctx_t() {}
    thread_ctx_t( const class thread_ctx_t& rhs )
    {
        m_cta_id = rhs.m_cta_id;
        n_insn = rhs.n_insn;
        n_insn_ac = rhs.n_insn_ac;
        n_l1_mis_ac = rhs.n_l1_mis_ac;
        n_l1_mrghit_ac = rhs.n_l1_mrghit_ac;
        n_l1_access_ac = rhs.n_l1_access_ac;
        m_active = rhs.m_active;
    }

    void reset()
    {
        m_cta_id = -1;
        n_insn = 0;
        m_active = false;
    }

   unsigned m_cta_id; // hardware CTA this thread belongs

   // per thread stats (ac stands for accumulative).
   unsigned n_insn;
   unsigned n_insn_ac;
   unsigned n_l1_mis_ac;
   unsigned n_l1_mrghit_ac;
   unsigned n_l1_access_ac; 

   bool m_active; 
};

class shd_warp_t {
public:
    unsigned FailSbCnt = 0;
    shd_warp_t( class shader_core_ctx *shader, unsigned warp_size, bool isIntWarp = false) 
        : m_shader(shader), m_warp_size(warp_size), _isIntWarp(isIntWarp), _reserved(false)
    {
        m_stores_outstanding=0;
        m_inst_in_pipeline=0;
        reset(); 
    }

    shd_warp_t( const shd_warp_t& rhs ) 
    {
        m_shader = rhs.m_shader;
        m_cta_id = rhs.m_cta_id;
        m_warp_id = rhs.m_warp_id;
        m_warp_size = rhs.m_warp_size;
        m_dynamic_warp_id = rhs.m_dynamic_warp_id;
        m_next_pc = rhs.m_next_pc;
        n_completed = rhs.n_completed;
        m_active_threads = rhs.m_active_threads;
        m_imiss_pending = rhs.m_imiss_pending;
        m_next = rhs.m_next;
        m_n_atomic = rhs.m_n_atomic;
        m_membar = rhs.m_membar;
        m_done_exit = rhs.m_done_exit;
        m_last_fetch = rhs.m_last_fetch;
        m_stores_outstanding = rhs.m_stores_outstanding;
        m_inst_in_pipeline = rhs.m_inst_in_pipeline;
        _isIntWarp = rhs._isIntWarp;
        _globalLoadPending = rhs._globalLoadPending;

        _inFlightLoadQueue = rhs._inFlightLoadQueue;
        _replayLoadQueue = rhs._replayLoadQueue;
        _reserved = rhs._reserved;

        for( unsigned i=0; i<IBUFFER_SIZE; ++i ) {
            m_ibuffer[i].m_inst = rhs.m_ibuffer[i].m_inst;
            m_ibuffer[i].m_valid = rhs.m_ibuffer[i].m_valid;
        }
    }

    void reset()
    {
        assert( m_stores_outstanding==0);
        assert( m_inst_in_pipeline==0);
        m_imiss_pending=false;
        m_warp_id=(unsigned)-1;
        m_dynamic_warp_id = (unsigned)-1;
        n_completed = m_warp_size; 
        m_n_atomic=0;
        m_membar=false;
        m_done_exit=true;
        m_last_fetch=0;
        m_next=0;
        _isIntWarp = false;
        _globalLoadPending = false;
    
        _inFlightLoadQueue.clear();
        _replayLoadQueue.clear();
    }

    void init( address_type start_pc,
               unsigned cta_id,
               unsigned wid,
               const std::bitset<MAX_WARP_SIZE> &active,
               unsigned dynamic_warp_id )
    {
        m_cta_id=cta_id;
        m_warp_id=wid;
        m_dynamic_warp_id=dynamic_warp_id;
        m_next_pc=start_pc;
        assert( n_completed >= active.count() );
        assert( n_completed <= m_warp_size);
        n_completed   -= active.count(); // active threads are not yet completed
        m_active_threads = active;
        m_done_exit=false;
    }

    bool functional_done() const;
    bool waiting(); // not const due to membar
    bool hardware_done() const;

    bool done_exit() const { return m_done_exit; }
    void set_done_exit() { m_done_exit=true; }
    void unset_done_exit() { m_done_exit=false; }

    void print( FILE *fout ) const;
    void print_ibuffer( FILE *fout ) const;

    unsigned get_n_completed() const { return n_completed; }
    void set_completed( unsigned lane ) 
    { 
        assert( m_active_threads.test(lane) );
        m_active_threads.reset(lane);
        n_completed++; 
    }

    unsigned long long get_last_fetch() const { return m_last_fetch; }
    void set_last_fetch( unsigned long long sim_cycle ) { m_last_fetch=sim_cycle; }

    unsigned get_n_atomic() const { return m_n_atomic; }
    void inc_n_atomic() { m_n_atomic++; }
    void dec_n_atomic(unsigned n) { m_n_atomic-=n; }

    void set_membar() { m_membar=true; }
    void clear_membar() { m_membar=false; }
    bool get_membar() const { return m_membar; }
    address_type get_pc() const { return m_next_pc; }
    void set_next_pc( address_type pc ) { m_next_pc = pc; }

    void ibuffer_fill( unsigned slot, const warp_inst_t *pI )
    {
       assert(slot < IBUFFER_SIZE );
       m_ibuffer[slot].m_inst=pI;
       m_ibuffer[slot].m_valid=true;
       m_next=0; 
    }
    bool ibuffer_empty() const
    {
        for( unsigned i=0; i < IBUFFER_SIZE; i++) 
            if(m_ibuffer[i].m_valid) 
                return false;
        return true;
    }

    unsigned num_inst_in_ibuffer() const {
        unsigned res = 0;
        for( unsigned i=0; i < IBUFFER_SIZE; i++) 
            if(m_ibuffer[i].m_valid) 
                res++;
        return res;
    }

    void ibuffer_flush()
    {
        for(unsigned i=0;i<IBUFFER_SIZE;i++) {
            if( m_ibuffer[i].m_valid )
                dec_inst_in_pipeline();
            m_ibuffer[i].m_inst=NULL; 
            m_ibuffer[i].m_valid=false; 
        }
    }
    const warp_inst_t *ibuffer_next_inst() { return m_ibuffer[m_next].m_inst; }
    bool ibuffer_next_valid() { return m_ibuffer[m_next].m_valid; }
    void ibuffer_free()
    {
        m_ibuffer[m_next].m_inst = NULL;
        m_ibuffer[m_next].m_valid = false;
    }
    void ibuffer_step() { m_next = (m_next+1)%IBUFFER_SIZE; }

    bool imiss_pending() const { return m_imiss_pending; }
    void set_imiss_pending() { m_imiss_pending=true; }
    void clear_imiss_pending() { m_imiss_pending=false; }

    bool stores_done() const { return m_stores_outstanding == 0; }
    void inc_store_req() { m_stores_outstanding++; }
    void dec_store_req() 
    {
        assert( m_stores_outstanding > 0 );
        m_stores_outstanding--;
    }

    unsigned num_inst_in_pipeline() const { return m_inst_in_pipeline; }

    bool inst_in_pipeline() const { return m_inst_in_pipeline > 0; }
    void inc_inst_in_pipeline() { 
        m_inst_in_pipeline++; 
        assert(m_inst_in_pipeline);
        //if (m_shader->get_sid()==4 && m_warp_id==0){
        //    printf("MARIA inc_inst_in_pipeline m_inst_in_pipeline=%d\n", m_inst_in_pipeline);
        //}
    }
    void dec_inst_in_pipeline() 
    {
        assert( m_inst_in_pipeline > 0 );
        m_inst_in_pipeline--;
        //if (m_shader->get_sid()==4 && m_warp_id==0){
        //    printf("MARIA dec_inst_in_pipeline m_inst_in_pipeline=%d\n", m_inst_in_pipeline);
        //}
        
    }

    unsigned get_cta_id() const { return m_cta_id; }

    unsigned get_dynamic_warp_id() const { return m_dynamic_warp_id; }
    unsigned get_warp_id() const { return m_warp_id; }

    ///////////////////////////////////////////////////////////////////
    // EDGE
    void     setIntWarp() { _isIntWarp = true; }
    bool     isIntWarp() const { return _isIntWarp; }

    void     setReserved() { _reserved = true; }
    bool     isReserved() const { return _reserved; }

    unsigned pendingLoads() const { return _globalLoadPending; }
    void incPendingLoads() { _globalLoadPending++; }
    void decPendingLoads() { assert( _globalLoadPending > 0 ); _globalLoadPending--; }
   
    void startLoad(warp_inst_t& inst, address_type pc, simt_stack* stack);
    void completeLoad( const warp_inst_t* inst );
    bool findInFlightLoad( const warp_inst_t* inst, bool remove = false );
    void dropLoads();
    void replayLoads();
    bool isDroppedLoad(const warp_inst_t* inst);
    bool isInFlightLoad(const warp_inst_t* inst);
    bool loadNextWb(); 
    ///////////////////////////////////////////////////////////////////
    
    // FIXME:
    void releaseReplay() {
        assert( _inFlightLoadQueue.empty() );
        _replayLoadQueue.clear();
    }

    kernel_info_t* kernel;
    bool isPaused() const {
        return paused;
    }
    void SetPaused() { 
        assert(!paused);
        paused = true; 
    }
    void UnsetPaused() { 
        assert(paused);
        paused = false; 
    }
    void AddPendingBarrierOp(const warp_inst_t* inst) {
        _edgePendingBarrierOps.push_back(inst);
    }
    void RemovePendingBarrierOp() {
        _edgePendingBarrierOps.pop_front();
    }
    bool HasPendingBarrierOps() const {
        return (!_edgePendingBarrierOps.empty());
    }
    bool HasBarrierOpInIbuffer(bool edgeSkipBarrier) const;
private:
    std::list<const warp_inst_t*> _edgePendingBarrierOps;
    bool paused = false;
    // Max number of instructions that can be fetched concurrently per-warp
    static const unsigned IBUFFER_SIZE = 64;
    class shader_core_ctx *m_shader;
    unsigned m_cta_id;
    unsigned m_warp_id;
    unsigned m_warp_size;
    unsigned m_dynamic_warp_id;

    address_type m_next_pc;
    unsigned n_completed;          // number of threads in warp completed
    std::bitset<MAX_WARP_SIZE> m_active_threads;

    bool m_imiss_pending;
    
    struct ibuffer_entry {
       ibuffer_entry() { m_valid = false; m_inst = NULL; }
       const warp_inst_t *m_inst;
       bool m_valid;
    };
    ibuffer_entry m_ibuffer[IBUFFER_SIZE]; 
    unsigned m_next;
                                   
    unsigned m_n_atomic;           // number of outstanding atomic operations 
    bool     m_membar;             // if true, warp is waiting at memory barrier

    bool m_done_exit; // true once thread exit has been registered for threads in this warp

    unsigned long long m_last_fetch;

    unsigned m_stores_outstanding; // number of store requests sent but not yet acknowledged
    unsigned m_inst_in_pipeline;

    // EDGE
    bool    _isIntWarp;
    bool    _reserved;
    unsigned _globalLoadPending; 


    std::list<PendingLoadInst> _inFlightLoadQueue;
    std::list<PendingLoadInst> _replayLoadQueue;
};



inline unsigned hw_tid_from_wid(unsigned wid, unsigned warp_size, unsigned i){return wid * warp_size + i;};
inline unsigned wid_from_hw_tid(unsigned tid, unsigned warp_size){return tid/warp_size;};

// EDGE 64 + 1
const unsigned WARP_PER_CTA_MAX = 65;
typedef std::bitset<WARP_PER_CTA_MAX> warp_set_t;

int register_bank(int regnum, int wid, unsigned num_banks, unsigned bank_warp_shift);

class shader_core_ctx;
class shader_core_config;
class shader_core_stats;

enum scheduler_prioritization_type
{
    SCHEDULER_PRIORITIZATION_LRR = 0, // Loose Round Robin
    SCHEDULER_PRIORITIZATION_SRR, // Strict Round Robin
    SCHEDULER_PRIORITIZATION_GTO, // Greedy Then Oldest
    SCHEDULER_PRIORITIZATION_GTLRR, // Greedy Then Loose Round Robin
    SCHEDULER_PRIORITIZATION_GTY, // Greedy Then Youngest
    SCHEDULER_PRIORITIZATION_OLDEST, // Oldest First
    SCHEDULER_PRIORITIZATION_YOUNGEST, // Youngest First
};

// Each of these corresponds to a string value in the gpgpsim.config file
// For example - to specify the LRR scheudler the config must contain lrr
enum concrete_scheduler
{
    CONCRETE_SCHEDULER_LRR = 0,
    CONCRETE_SCHEDULER_GTO,
    CONCRETE_SCHEDULER_TWO_LEVEL_ACTIVE,
    CONCRETE_SCHEDULER_WARP_LIMITING,
    CONCRETE_SCHEDULER_EDGE,
    NUM_CONCRETE_SCHEDULERS
};

class WarpVector : public std::vector< shd_warp_t* > {};

class scheduler_unit { //this can be copied freely, so can be used in std containers.
public:
    scheduler_unit(shader_core_stats* stats, shader_core_ctx* shader, 
                   Scoreboard* scoreboard, simt_stack** simt, 
                   std::vector<shd_warp_t>* warp, 
                   register_set* sp_out,
                   register_set* sfu_out,
                   register_set* mem_out,
                   int id) 
        : m_supervised_warps(), m_stats(stats), m_shader(shader),
        m_scoreboard(scoreboard), m_simt_stack(simt), /*m_pipeline_reg(pipe_regs),*/ m_warp(warp),
        m_sp_out(sp_out),m_sfu_out(sfu_out),m_mem_out(mem_out), m_id(id){}
    virtual ~scheduler_unit(){}
    virtual void add_supervised_warp_id(int i) {
        m_supervised_warps.push_back(&warp(i));
    }
    virtual void done_adding_supervised_warps() {
        m_last_supervised_issued = m_supervised_warps.end();
    }


    // The core scheduler cycle method is meant to be common between
    // all the derived schedulers.  The scheduler's behaviour can be
    // modified by changing the contents of the m_next_cycle_prioritized_warps list.
    void cycle();

    // These are some common ordering fucntions that the
    // higher order schedulers can take advantage of
    template < typename T >
    void order_lrr( typename std::vector< T >& result_list,
                    const typename std::vector< T >& input_list,
                    const typename std::vector< T >::const_iterator& last_issued_from_input,
                    unsigned num_warps_to_add );
    
    enum OrderingType 
    {
        // The item that issued last is prioritized first then the sorted result
        // of the priority_function
        ORDERING_GREEDY_THEN_PRIORITY_FUNC = 0,
        // No greedy scheduling based on last to issue. Only the priority function determines
        // priority
        ORDERED_PRIORITY_FUNC_ONLY,
        NUM_ORDERING,
    };
    template < typename U >
    void order_by_priority( std::vector< U >& result_list,
                            const typename std::vector< U >& input_list,
                            const typename std::vector< U >::const_iterator& last_issued_from_input,
                            unsigned num_warps_to_add,
                            OrderingType age_ordering,
                            bool (*priority_func)(U lhs, U rhs) );
    static bool sort_warps_by_oldest_dynamic_id(shd_warp_t* lhs, shd_warp_t* rhs);

    // Derived classes can override this function to populate
    // m_supervised_warps with their scheduling policies
    virtual void order_warps() = 0;

    void ClearScoreboard(int wid) { m_scoreboard->resetTable(wid); }

protected:
    virtual void do_on_warp_issued( unsigned warp_id,
                                    unsigned num_issued,
                                    const std::vector< shd_warp_t* >::const_iterator& prioritized_iter );
    inline int get_sid() const;
protected:
    shd_warp_t& warp(int i);

    // This is the prioritized warp list that is looped over each cycle to determine
    // which warp gets to issue.
    WarpVector m_next_cycle_prioritized_warps;

    // The m_supervised_warps list is all the warps this scheduler is supposed to
    // arbitrate between.  This is useful in systems where there is more than
    // one warp scheduler. In a single scheduler system, this is simply all
    // the warps assigned to this core.
    //std::vector< shd_warp_t* > m_supervised_warps;
    WarpVector m_supervised_warps;
   
    // This is the iterator pointer to the last supervised warp you issued
    std::vector< shd_warp_t* >::const_iterator m_last_supervised_issued;
    shader_core_stats *m_stats;
    shader_core_ctx* m_shader;
    // these things should become accessors: but would need a bigger rearchitect of how shader_core_ctx interacts with its parts.
    Scoreboard* m_scoreboard; 
    simt_stack** m_simt_stack;
    //warp_inst_t** m_pipeline_reg;
    std::vector<shd_warp_t>* m_warp;
    register_set* m_sp_out;
    register_set* m_sfu_out;
    register_set* m_mem_out;

    int m_id;
};

/// EDGE
class EdgeScheduler : public scheduler_unit 
{
public:
    EdgeScheduler( shader_core_stats* stats, shader_core_ctx* shader,
                   Scoreboard* scoreboard, simt_stack** simt,
                   std::vector<shd_warp_t>* warp,
                   register_set* sp_out,
                   register_set* sfu_out,
                   register_set* mem_out,
                   int id )
        : scheduler_unit ( stats, shader, scoreboard, simt, warp, sp_out, sfu_out, mem_out, id ), 
            _iSignal(0), _excludeWarpId(-1), _edgeLastChosenWarpId(-1) {}
	virtual ~EdgeScheduler () {}
	virtual void order_warps ();
    virtual void done_adding_supervised_warps() {
        m_last_supervised_issued = m_supervised_warps.begin();
    }

    
    //void     setIntSignal()     { assert( !_iSignal ); _iSignal = 1; }
    //void     clearIntSignal()   { assert( _iSignal ); _iSignal = 0; }
    //unsigned intSignal() const  { return _iSignal; }
    
    void     excludeWarpId(int wid) { assert( _excludeWarpId == -1 ); _excludeWarpId = wid; }
    void     includeWarpId(int wid) { assert( _excludeWarpId == wid ); _excludeWarpId = -1; }

    void     prioritizeVictimWarp(int wid) { 
        assert(std::find(_victimWarpPriorityIds.begin(), _victimWarpPriorityIds.end(), wid) == _victimWarpPriorityIds.end()); 
        _victimWarpPriorityIds.push_back(wid); 
    }
    void     restoreVictimWarp(int wid) { 
        assert(std::find(_victimWarpPriorityIds.begin(), _victimWarpPriorityIds.end(), wid) != _victimWarpPriorityIds.end()); 
        _victimWarpPriorityIds.erase(std::find(_victimWarpPriorityIds.begin(), _victimWarpPriorityIds.end(), wid));
    }

    void getSupervisedWarps(WarpVector& dest) const;

protected:
    void        scheduleEventWarps();
    void        scheduleEventWarp(int wid);

    void        scheduleVictimWarp(int wid);
    void        scheduleVictimWarps();
    shd_warp_t* findIntWarp(bool Erase, int warp_id);

    unsigned    _iSignal;
    int         _excludeWarpId;

    std::vector<int>        _victimWarpPriorityIds;
    int         _edgeLastChosenWarpId;
};

class lrr_scheduler : public scheduler_unit {
public:
	lrr_scheduler ( shader_core_stats* stats, shader_core_ctx* shader,
                    Scoreboard* scoreboard, simt_stack** simt,
                    std::vector<shd_warp_t>* warp,
                    register_set* sp_out,
                    register_set* sfu_out,
                    register_set* mem_out,
                    int id )
	: scheduler_unit ( stats, shader, scoreboard, simt, warp, sp_out, sfu_out, mem_out, id ){}
	virtual ~lrr_scheduler () {}
	virtual void order_warps ();
    virtual void done_adding_supervised_warps() {
        m_last_supervised_issued = m_supervised_warps.end();
    }
};

class gto_scheduler : public scheduler_unit {
public:
	gto_scheduler ( shader_core_stats* stats, shader_core_ctx* shader,
                    Scoreboard* scoreboard, simt_stack** simt,
                    std::vector<shd_warp_t>* warp,
                    register_set* sp_out,
                    register_set* sfu_out,
                    register_set* mem_out,
                    int id )
	: scheduler_unit ( stats, shader, scoreboard, simt, warp, sp_out, sfu_out, mem_out, id ){}
	virtual ~gto_scheduler () {}
	virtual void order_warps ();
    virtual void done_adding_supervised_warps() {
        m_last_supervised_issued = m_supervised_warps.begin();
    }

};

class two_level_active_scheduler : public scheduler_unit {
public:
	two_level_active_scheduler ( shader_core_stats* stats, shader_core_ctx* shader,
                          Scoreboard* scoreboard, simt_stack** simt,
                          std::vector<shd_warp_t>* warp,
                          register_set* sp_out,
                          register_set* sfu_out,
                          register_set* mem_out,
                          int id,
                          char* config_str )
	: scheduler_unit ( stats, shader, scoreboard, simt, warp, sp_out, sfu_out, mem_out, id ),
	  m_pending_warps() 
    {
        unsigned inner_level_readin;
        unsigned outer_level_readin; 
        assert( 3 == sscanf( config_str,
                "two_level_active:%d:%d:%d",
                &m_max_active_warps,
                &inner_level_readin,
                &outer_level_readin) );
        m_inner_level_prioritization=(scheduler_prioritization_type)inner_level_readin;
        m_outer_level_prioritization=(scheduler_prioritization_type)outer_level_readin;
    }
	virtual ~two_level_active_scheduler () {}
    virtual void order_warps();
	void add_supervised_warp_id(int i) {
        if ( m_next_cycle_prioritized_warps.size() < m_max_active_warps ) {
            m_next_cycle_prioritized_warps.push_back( &warp(i) );
        } else {
		    m_pending_warps.push_back(&warp(i));
        }
	}
    virtual void done_adding_supervised_warps() {
        m_last_supervised_issued = m_supervised_warps.begin();
    }

protected:
    virtual void do_on_warp_issued( unsigned warp_id,
                                    unsigned num_issued,
                                    const std::vector< shd_warp_t* >::const_iterator& prioritized_iter );

private:
	std::deque< shd_warp_t* > m_pending_warps;
    scheduler_prioritization_type m_inner_level_prioritization;
    scheduler_prioritization_type m_outer_level_prioritization;
	unsigned m_max_active_warps;
};

// Static Warp Limiting Scheduler
class swl_scheduler : public scheduler_unit {
public:
	swl_scheduler ( shader_core_stats* stats, shader_core_ctx* shader,
                    Scoreboard* scoreboard, simt_stack** simt,
                    std::vector<shd_warp_t>* warp,
                    register_set* sp_out,
                    register_set* sfu_out,
                    register_set* mem_out,
                    int id,
                    char* config_string );
	virtual ~swl_scheduler () {}
	virtual void order_warps ();
    virtual void done_adding_supervised_warps() {
        m_last_supervised_issued = m_supervised_warps.begin();
    }

protected:
    scheduler_prioritization_type m_prioritization;
    unsigned m_num_warps_to_limit;
};



class opndcoll_rfu_t { // operand collector based register file unit
public:
   // constructors
   opndcoll_rfu_t()
   {
      m_num_banks=0;
      m_shader=NULL;
      m_initialized=false;
   }
   void add_cu_set(unsigned cu_set, unsigned num_cu, unsigned num_dispatch);
   typedef std::vector<register_set*> port_vector_t;
   typedef std::vector<unsigned int> uint_vector_t;
   void add_port( port_vector_t & input, port_vector_t & ouput, uint_vector_t cu_sets);
   void init( unsigned num_banks, shader_core_ctx *shader );

   // modifiers
   bool writeback( const warp_inst_t &warp ); // might cause stall 

   void step()
   {
        dispatch_ready_cu();   
        allocate_reads();
        for( unsigned p = 0 ; p < m_in_ports.size(); p++ ) 
            allocate_cu( p );
        process_banks();
   }

   void dump( FILE *fp ) const
   {
      fprintf(fp,"\n");
      fprintf(fp,"Operand Collector State:\n");
      for( unsigned n=0; n < m_cu.size(); n++ ) {
         fprintf(fp,"   CU-%2u: ", n);
         m_cu[n]->dump(fp,m_shader);
      }
      m_arbiter.dump(fp);
   }

   shader_core_ctx *shader_core() { return m_shader; }

private:

   void process_banks()
   {
      m_arbiter.reset_alloction();
   }

   void dispatch_ready_cu();
   void allocate_cu( unsigned port );
   void allocate_reads();

   // types

   class collector_unit_t;

   class op_t {
   public:

      op_t() { m_valid = false; }
      op_t( collector_unit_t *cu, unsigned op, unsigned reg, unsigned num_banks, unsigned bank_warp_shift )
      {
         m_valid = true;
         m_warp=NULL;
         m_cu = cu;
         m_operand = op;
         m_register = reg;
         m_bank = register_bank(reg,cu->get_warp_id(),num_banks,bank_warp_shift);
      }
      op_t( const warp_inst_t *warp, unsigned reg, unsigned num_banks, unsigned bank_warp_shift )
      {
         m_valid=true;
         m_warp=warp;
         m_register=reg;
         m_cu=NULL;
         m_operand = -1;
         m_bank = register_bank(reg,warp->warp_id(),num_banks,bank_warp_shift);
      }

      // accessors
      bool valid() const { return m_valid; }
      unsigned get_reg() const
      {
         assert( m_valid );
         return m_register;
      }
      unsigned get_wid() const
      {
          if( m_warp ) return m_warp->warp_id();
          else if( m_cu ) return m_cu->get_warp_id();
          else abort();
      }
      unsigned get_active_count() const
      {
          if( m_warp ) return m_warp->active_count();
          else if( m_cu ) return m_cu->get_active_count();
          else abort();
      }
      const active_mask_t & get_active_mask()
      {
          if( m_warp ) return m_warp->get_active_mask();
          else if( m_cu ) return m_cu->get_active_mask();
          else abort();
      }
      unsigned get_sp_op() const
      {
          if( m_warp ) return m_warp->sp_op;
          else if( m_cu ) return m_cu->get_sp_op();
          else abort();
      }
      unsigned get_oc_id() const { return m_cu->get_id(); }
      unsigned get_bank() const { return m_bank; }
      unsigned get_operand() const { return m_operand; }
      void dump(FILE *fp) const 
      {
         if(m_cu) 
            fprintf(fp," <R%u, CU:%u, w:%02u> ", m_register,m_cu->get_id(),m_cu->get_warp_id());
         else if( !m_warp->empty() )
            fprintf(fp," <R%u, wid:%02u> ", m_register,m_warp->warp_id() );
      }
      std::string get_reg_string() const
      {
         char buffer[64];
         snprintf(buffer,64,"R%u", m_register);
         return std::string(buffer);
      }

      // modifiers
      void reset() { m_valid = false; }
   private:
      bool m_valid;
      collector_unit_t  *m_cu; 
      const warp_inst_t *m_warp;
      unsigned  m_operand; // operand offset in instruction. e.g., add r1,r2,r3; r2 is oprd 0, r3 is 1 (r1 is dst)
      unsigned  m_register;
      unsigned  m_bank;
   };

   enum alloc_t {
      NO_ALLOC,
      READ_ALLOC,
      WRITE_ALLOC,
   };

   class allocation_t {
   public:
      allocation_t() { m_allocation = NO_ALLOC; }
      bool is_read() const { return m_allocation==READ_ALLOC; }
      bool is_write() const {return m_allocation==WRITE_ALLOC; }
      bool is_free() const {return m_allocation==NO_ALLOC; }
      void dump(FILE *fp) const {
         if( m_allocation == NO_ALLOC ) { fprintf(fp,"<free>"); }
         else if( m_allocation == READ_ALLOC ) { fprintf(fp,"rd: "); m_op.dump(fp); }
         else if( m_allocation == WRITE_ALLOC ) { fprintf(fp,"wr: "); m_op.dump(fp); }
         fprintf(fp,"\n");
      }
      void alloc_read( const op_t &op )  { assert(is_free()); m_allocation=READ_ALLOC; m_op=op; }
      void alloc_write( const op_t &op ) { assert(is_free()); m_allocation=WRITE_ALLOC; m_op=op; }
      void reset() { m_allocation = NO_ALLOC; }
   private:
      enum alloc_t m_allocation;
      op_t m_op;
   };

   class arbiter_t {
   public:
      // constructors
      arbiter_t()
      {
         m_queue=NULL;
         m_allocated_bank=NULL;
         m_allocator_rr_head=NULL;
         _inmatch=NULL;
         _outmatch=NULL;
         _request=NULL;
         m_last_cu=0;
      }
      void init( unsigned num_cu, unsigned num_banks ) 
      { 
         assert(num_cu > 0);
         assert(num_banks > 0);
         m_num_collectors = num_cu;
         m_num_banks = num_banks;
         _inmatch = new int[ m_num_banks ];
         _outmatch = new int[ m_num_collectors ];
         _request = new int*[ m_num_banks ];
         for(unsigned i=0; i<m_num_banks;i++) 
             _request[i] = new int[m_num_collectors];
         m_queue = new std::list<op_t>[num_banks];
         m_allocated_bank = new allocation_t[num_banks];
         m_allocator_rr_head = new unsigned[num_cu];
         for( unsigned n=0; n<num_cu;n++ ) 
            m_allocator_rr_head[n] = n%num_banks;
         reset_alloction();
      }

      // accessors
      void dump(FILE *fp) const
      {
         fprintf(fp,"\n");
         fprintf(fp,"  Arbiter State:\n");
         fprintf(fp,"  requests:\n");
         for( unsigned b=0; b<m_num_banks; b++ ) {
            fprintf(fp,"    bank %u : ", b );
            std::list<op_t>::const_iterator o = m_queue[b].begin();
            for(; o != m_queue[b].end(); o++ ) {
               o->dump(fp);
            }
            fprintf(fp,"\n");
         }
         fprintf(fp,"  grants:\n");
         for(unsigned b=0;b<m_num_banks;b++) {
            fprintf(fp,"    bank %u : ", b );
            m_allocated_bank[b].dump(fp);
         }
         fprintf(fp,"\n");
      }

      // modifiers
      std::list<op_t> allocate_reads(); 

      void add_read_requests( collector_unit_t *cu ) 
      {
         const op_t *src = cu->get_operands();
         for( unsigned i=0; i<MAX_REG_OPERANDS*2; i++) {
            const op_t &op = src[i];
            if( op.valid() ) {
               unsigned bank = op.get_bank();
               m_queue[bank].push_back(op);
            }
         }
      }
      bool bank_idle( unsigned bank ) const
      {
          return m_allocated_bank[bank].is_free();
      }
      void allocate_bank_for_write( unsigned bank, const op_t &op )
      {
         assert( bank < m_num_banks );
         m_allocated_bank[bank].alloc_write(op);
      }
      void allocate_for_read( unsigned bank, const op_t &op )
      {
         assert( bank < m_num_banks );
         m_allocated_bank[bank].alloc_read(op);
      }
      void reset_alloction()
      {
         for( unsigned b=0; b < m_num_banks; b++ ) 
            m_allocated_bank[b].reset();
      }

   private:
      unsigned m_num_banks;
      unsigned m_num_collectors;

      allocation_t *m_allocated_bank; // bank # -> register that wins
      std::list<op_t> *m_queue;

      unsigned *m_allocator_rr_head; // cu # -> next bank to check for request (rr-arb)
      unsigned  m_last_cu; // first cu to check while arb-ing banks (rr)

      int *_inmatch;
      int *_outmatch;
      int **_request;
   };

   class input_port_t {
   public:
       input_port_t(port_vector_t & input, port_vector_t & output, uint_vector_t cu_sets)
       : m_in(input),m_out(output), m_cu_sets(cu_sets)
       {
           assert(input.size() == output.size());
           assert(not m_cu_sets.empty());
       }
   //private:
       port_vector_t m_in,m_out;
       uint_vector_t m_cu_sets;
   };

   class collector_unit_t {
   public:
      // constructors
      collector_unit_t()
      { 
         m_free = true;
         m_warp = NULL;
         m_output_register = NULL;
         m_src_op = new op_t[MAX_REG_OPERANDS*2];
         m_not_ready.reset();
         m_warp_id = -1;
         m_num_banks = 0;
         m_bank_warp_shift = 0;
      }
      // accessors
      bool ready() const;
      const op_t *get_operands() const { return m_src_op; }
      void dump(FILE *fp, const shader_core_ctx *shader ) const;

      unsigned get_warp_id() const { return m_warp_id; }
      unsigned get_active_count() const { return m_warp->active_count(); }
      const active_mask_t & get_active_mask() const { return m_warp->get_active_mask(); }
      unsigned get_sp_op() const { return m_warp->sp_op; }
      unsigned get_id() const { return m_cuid; } // returns CU hw id
      unsigned get_inst_uniq_id() { return m_warp->get_uid(); }

      // modifiers
      void init(unsigned n, 
                unsigned num_banks, 
                unsigned log2_warp_size,
                const core_config *config,
                opndcoll_rfu_t *rfu ); 
      bool allocate( register_set* pipeline_reg, register_set* output_reg );

      void collect_operand( unsigned op )
      {
         m_not_ready.reset(op);
      }
      unsigned get_num_operands() const{
    	  return m_warp->get_num_operands();
      }
      unsigned get_num_regs() const{
    	  return m_warp->get_num_regs();
      }
      void dispatch();
      bool is_free(){return m_free;}

   private:
      bool m_free;
      unsigned m_cuid; // collector unit hw id
      unsigned m_warp_id;
      warp_inst_t  *m_warp;
      register_set* m_output_register; // pipeline register to issue to when ready
      op_t *m_src_op;
      std::bitset<MAX_REG_OPERANDS*2> m_not_ready;
      unsigned m_num_banks;
      unsigned m_bank_warp_shift;
      opndcoll_rfu_t *m_rfu;

   };

   class dispatch_unit_t {
   public:
      dispatch_unit_t(std::vector<collector_unit_t>* cus) 
      { 
         m_last_cu=0;
         m_collector_units=cus;
         m_num_collectors = (*cus).size();
         m_next_cu=0;
      }

      collector_unit_t *find_ready()
      {
         unsigned least_inst_uid = (unsigned)-1;
         collector_unit_t *to_return = NULL;
         // gem5-gpu NOTE: gem5-gpu requires that all memory instructions be
         // issued in-order to the load-store queues to correctly enforce
         // fences. GPGPU-Sim did not have this requirement, so this ready
         // instruction select code is different than GPGPU-Sim.
         for( unsigned n=0; n < m_num_collectors; n++ ) {
            collector_unit_t *cu = &((*m_collector_units)[n]);
            if( !cu->is_free() ) {
               if( cu->get_inst_uniq_id() < least_inst_uid ) {
                  least_inst_uid = cu->get_inst_uniq_id();
                  if( cu->ready() ) {
                     to_return = cu;
                  } else {
                     to_return = NULL;
                  }
               }
            }
         }
         return to_return;
      }

   private:
      unsigned m_num_collectors;
      std::vector<collector_unit_t>* m_collector_units;
      unsigned m_last_cu; // dispatch ready cu's rr
      unsigned m_next_cu;  // for initialization
   };

   // opndcoll_rfu_t data members
   bool m_initialized;

   unsigned m_num_collector_sets;
   //unsigned m_num_collectors;
   unsigned m_num_banks;
   unsigned m_bank_warp_shift;
   unsigned m_warp_size;
   std::vector<collector_unit_t *> m_cu;
   arbiter_t m_arbiter;

   //unsigned m_num_ports;
   //std::vector<warp_inst_t**> m_input;
   //std::vector<warp_inst_t**> m_output;
   //std::vector<unsigned> m_num_collector_units;
   //warp_inst_t **m_alu_port;

   std::vector<input_port_t> m_in_ports;
   typedef std::map<unsigned /* collector set */, std::vector<collector_unit_t> /*collector sets*/ > cu_sets_t;
   cu_sets_t m_cus;
   std::vector<dispatch_unit_t> m_dispatch_units;

   //typedef std::map<warp_inst_t**/*port*/,dispatch_unit_t> port_to_du_t;
   //port_to_du_t                     m_dispatch_units;
   //std::map<warp_inst_t**,std::list<collector_unit_t*> > m_free_cu;
   shader_core_ctx                 *m_shader;
};

class barrier_set_t {
public:
   barrier_set_t( unsigned max_warps_per_core, unsigned max_cta_per_core, shader_core_ctx *m_shader_p );

   // during cta allocation
   void allocate_barrier( unsigned cta_id, warp_set_t warps );

   // during cta deallocation
   void deallocate_barrier( unsigned cta_id );

   typedef std::map<unsigned, warp_set_t >  cta_to_warp_t;

   void removeWarpFromCta( unsigned warpId, unsigned ctaId );

   // individual warp hits barrier
   void warp_reaches_barrier( unsigned cta_id, unsigned warp_id );

   // fetching a warp
   bool available_for_fetch( unsigned warp_id ) const;

   // warp reaches exit 
   void warp_exit( unsigned warp_id );

   // interrupt warp reaches exit
   void intWarpExit( unsigned intWarpId, unsigned intCtaId );

   // assertions
   bool warp_waiting_at_barrier( unsigned warp_id ) const;
   bool warp_active( unsigned warp_id ) const;

    // EDGE
   void set_at_barrier( unsigned warp_id );
   void set_active( unsigned warp_id );
    
    bool anyAtBarrier( unsigned cta_id );
    bool isEdgeBarrier( unsigned cta_id ) ;
    void setEdgeBarrier( unsigned cta_id );

    void releaseEdgeBarrier( unsigned cta_id );
    void releaseAllEdgeBarrier( ) 
    {
       for( cta_to_warp_t::iterator it = m_cta_to_warps.begin(); 
               it != m_cta_to_warps.end(); ++it ) {
            releaseEdgeBarrier(it->first);
       }
    }

    bool edgeSetVictimBarrier( unsigned cid, unsigned wid );
    bool edgeIsVictimBarrierDone( unsigned wid );
    void edgeRestoreVictimBarrier( unsigned wid );

   // debug
   void dump() const;

private:
    void verifyValidCta(unsigned cta_id);

   unsigned m_max_cta_per_core;
   unsigned m_max_warps_per_core;

   cta_to_warp_t m_cta_to_warps; 
   warp_set_t m_warp_active;
   warp_set_t m_warp_at_barrier;

   std::vector<bool> _edgeBarrier;
   std::vector<bool> _edgeReleaseBarrier;

   warp_set_t _edgeVictimWarpAtBarrier;   
   shader_core_ctx                 *m_shader;
};

struct insn_latency_info {
   address_type pc;
   unsigned long latency;
};

struct ifetch_buffer_t {
    ifetch_buffer_t() { m_valid=false; }

    ifetch_buffer_t( address_type pc, unsigned nbytes, unsigned warp_id ) 
    { 
        m_valid=true; 
        m_pc=pc; 
        m_nbytes=nbytes; 
        m_warp_id=warp_id;
    }

    bool m_valid;
    address_type m_pc;
    unsigned m_nbytes;
    unsigned m_warp_id;
};

class shader_core_config;

class simd_function_unit {
public:
    simd_function_unit( const shader_core_config *config );
    ~simd_function_unit() { delete m_dispatch_reg; }

    // modifiers
    virtual void issue( register_set& source_reg ) { source_reg.move_out_to(m_dispatch_reg); occupied.set(m_dispatch_reg->latency);}
    virtual void cycle() = 0;
    virtual void active_lanes_in_pipeline() = 0;

    // accessors
    virtual unsigned clock_multiplier() const { return 1; }
    virtual bool can_issue( const warp_inst_t &inst ) const { return m_dispatch_reg->empty() && !occupied.test(inst.latency); }
    virtual bool stallable() const = 0;
    virtual void print( FILE *fp ) const
    {
        fprintf(fp,"%s dispatch= ", m_name.c_str() );
        m_dispatch_reg->print(fp);
    }
protected:
    std::string m_name;
    const shader_core_config *m_config;
    warp_inst_t *m_dispatch_reg;
    static const unsigned MAX_ALU_LATENCY = 512;
    std::bitset<MAX_ALU_LATENCY> occupied;
};

class pipelined_simd_unit : public simd_function_unit {
public:
    pipelined_simd_unit( register_set* result_port, const shader_core_config *config, unsigned max_latency, shader_core_ctx *core );

    //modifiers
    virtual void cycle() 
    {
        if( !m_pipeline_reg[0]->empty() ){
            m_result_port->move_in(m_pipeline_reg[0]);
        }
        for( unsigned stage=0; (stage+1)<m_pipeline_depth; stage++ ) 
            move_warp(m_pipeline_reg[stage], m_pipeline_reg[stage+1]);
        if( !m_dispatch_reg->empty() ) {
            if( !m_dispatch_reg->dispatch_delay()) {
                int start_stage = m_dispatch_reg->latency - m_dispatch_reg->initiation_interval;
                move_warp(m_pipeline_reg[start_stage],m_dispatch_reg);
            }
        }
        occupied >>=1;
    }
    virtual void issue( register_set& source_reg );
    virtual unsigned get_active_lanes_in_pipeline()
    {
    	active_mask_t active_lanes;
    	active_lanes.reset();
        for( unsigned stage=0; (stage+1)<m_pipeline_depth; stage++ ){
        	if( !m_pipeline_reg[stage]->empty() )
        		active_lanes|=m_pipeline_reg[stage]->get_active_mask();
        }
        return active_lanes.count();
    }
    virtual void active_lanes_in_pipeline() = 0;
/*
    virtual void issue( register_set& source_reg )
    {
        //move_warp(m_dispatch_reg,source_reg);
        //source_reg.move_out_to(m_dispatch_reg);
        simd_function_unit::issue(source_reg);
    }
*/
    // accessors
    virtual bool stallable() const { return false; }
    virtual bool can_issue( const warp_inst_t &inst ) const
    {
        return simd_function_unit::can_issue(inst);
    }
    virtual void print(FILE *fp) const
    {
        simd_function_unit::print(fp);
        for( int s=m_pipeline_depth-1; s>=0; s-- ) {
            if( !m_pipeline_reg[s]->empty() ) { 
                fprintf(fp,"      %s[%2d] ", m_name.c_str(), s );
                m_pipeline_reg[s]->print(fp);
            }
        }
    }
protected:
    unsigned m_pipeline_depth;
    warp_inst_t **m_pipeline_reg;
    register_set *m_result_port;
    class shader_core_ctx *m_core;
};

class sfu : public pipelined_simd_unit
{
public
    :
    sfu( register_set* result_port, const shader_core_config *config, 
            shader_core_ctx *core );
    virtual bool can_issue( const warp_inst_t &inst ) const
    {
        switch(inst.op) {
        case SFU_OP: break;
        case ALU_SFU_OP: break;
        default: return false;
        }
        return pipelined_simd_unit::can_issue(inst);
    }
    virtual void active_lanes_in_pipeline();
    virtual void issue(  register_set& source_reg );
};

class sp_unit : public pipelined_simd_unit
{
public:
    sp_unit( register_set* result_port, const shader_core_config *config, shader_core_ctx *core );
    virtual bool can_issue( const warp_inst_t &inst ) const
    {
        switch(inst.op) {
        case SFU_OP: return false; 
        case LOAD_OP: return false;
        case STORE_OP: return false;
        case MEMORY_BARRIER_OP: return false;
        default: break;
        }
        return pipelined_simd_unit::can_issue(inst);
    }
    virtual void active_lanes_in_pipeline();
    virtual void issue( register_set& source_reg );
};

class simt_core_cluster;
class shader_memory_interface;
class shader_core_mem_fetch_allocator;
class cache_t;

class ldst_unit: public pipelined_simd_unit {
public:
    ldst_unit( mem_fetch_interface *icnt,
               shader_core_mem_fetch_allocator *mf_allocator,
               shader_core_ctx *core, 
               opndcoll_rfu_t *operand_collector,
               Scoreboard *scoreboard,
               const shader_core_config *config, 
               const memory_config *mem_config,  
               class shader_core_stats *stats, 
               unsigned sid, unsigned tpc );

    // modifiers
    virtual void issue( register_set &inst );
    virtual void cycle();
     
    void fill( mem_fetch *mf );
    void flush();
    void writeback();

    /// Inserts this instruction into the writeback stage of the pipeline
    /// Returns true if successful, false if there is an instruction blocking
    // EDGE
    unsigned writebackInst(warp_inst_t &inst);
    bool releaseRegisters(const warp_inst_t* inst);
    bool dropLoad(const warp_inst_t* isnt);
    bool loadNextWb(const warp_inst_t* inst);

    // accessors
    virtual unsigned clock_multiplier() const;

    virtual bool can_issue( const warp_inst_t &inst ) const
    {
        switch(inst.op) {
        case LOAD_OP: break;
        case STORE_OP: break;
        case BARRIER_OP: break;
        case MEMORY_BARRIER_OP: break;
        default: return false;
        }
        return m_dispatch_reg->empty();
    }

    virtual void active_lanes_in_pipeline();
    virtual bool stallable() const { return true; }
    bool response_buffer_full() const;
    void print(FILE *fout) const;
    void print_cache_stats( FILE *fp, unsigned& dl1_accesses, unsigned& dl1_misses );
    void get_cache_stats(unsigned &read_accesses, unsigned &write_accesses, unsigned &read_misses, unsigned &write_misses, unsigned cache_type);
    void get_cache_stats(cache_stats &cs);

    void get_L1D_sub_stats(struct cache_sub_stats &css) const;
    void get_L1C_sub_stats(struct cache_sub_stats &css) const;
    void get_L1T_sub_stats(struct cache_sub_stats &css) const;

    bool warpPendingRegWrites(int warpId);
protected:
    ldst_unit( mem_fetch_interface *icnt,
               shader_core_mem_fetch_allocator *mf_allocator,
               shader_core_ctx *core, 
               opndcoll_rfu_t *operand_collector,
               Scoreboard *scoreboard,
               const shader_core_config *config,
               const memory_config *mem_config,  
               shader_core_stats *stats,
               unsigned sid,
               unsigned tpc,
               l1_cache* new_l1d_cache );
    void init( mem_fetch_interface *icnt,
               shader_core_mem_fetch_allocator *mf_allocator,
               shader_core_ctx *core, 
               opndcoll_rfu_t *operand_collector,
               Scoreboard *scoreboard,
               const shader_core_config *config,
               const memory_config *mem_config,  
               shader_core_stats *stats,
               unsigned sid,
               unsigned tpc );

protected:
   bool shared_cycle( warp_inst_t &inst, mem_stage_stall_type &rc_fail, mem_stage_access_type &fail_type);
   bool constant_cycle( warp_inst_t &inst, mem_stage_stall_type &rc_fail, mem_stage_access_type &fail_type);
   bool texture_cycle( warp_inst_t &inst, mem_stage_stall_type &rc_fail, mem_stage_access_type &fail_type);
   bool memory_cycle( warp_inst_t &inst, mem_stage_stall_type &rc_fail, mem_stage_access_type &fail_type);
   bool memory_cycle_gem5( warp_inst_t &inst, mem_stage_stall_type &rc_fail, mem_stage_access_type &fail_type);

   virtual mem_stage_stall_type process_cache_access( cache_t* cache,
                                                      new_addr_type address,
                                                      warp_inst_t &inst,
                                                      std::list<cache_event>& events,
                                                      mem_fetch *mf,
                                                      enum cache_request_status status );
   mem_stage_stall_type process_memory_access_queue( cache_t *cache, warp_inst_t &inst );

   const memory_config *m_memory_config;
   class mem_fetch_interface *m_icnt;
   shader_core_mem_fetch_allocator *m_mf_allocator;
   class shader_core_ctx *m_core;
   unsigned m_sid;
   unsigned m_tpc;

   tex_cache *m_L1T; // texture cache
   read_only_cache *m_L1C; // constant cache
   l1_cache *m_L1D; // data cache
   std::map<unsigned/*warp_id*/, std::map<unsigned/*regnum*/,unsigned/*count*/> > m_pending_writes;
   std::list<mem_fetch*> m_response_fifo;
   opndcoll_rfu_t *m_operand_collector;
   Scoreboard *m_scoreboard;

   mem_fetch *m_next_global;
   warp_inst_t m_next_wb;
   unsigned m_writeback_arb; // round-robin arbiter for writeback contention between L1T, L1C, shared
   unsigned m_num_writeback_clients;

   enum mem_stage_stall_type m_mem_rc;

   shader_core_stats *m_stats; 

   // for debugging
   unsigned long long m_last_inst_gpu_sim_cycle;
   unsigned long long m_last_inst_gpu_tot_sim_cycle;
};

enum pipeline_stage_name_t {
    ID_OC_SP=0,
    ID_OC_SFU,  
    ID_OC_MEM,  
    OC_EX_SP,
    OC_EX_SFU,
    OC_EX_MEM,
    EX_WB,
    N_PIPELINE_STAGES 
};

const char* const pipeline_stage_name_decode[] = {
    "ID_OC_SP",
    "ID_OC_SFU",  
    "ID_OC_MEM",  
    "OC_EX_SP",
    "OC_EX_SFU",
    "OC_EX_MEM",
    "EX_WB",
    "N_PIPELINE_STAGES" 
};

struct shader_core_config : public core_config
{
    shader_core_config(){
	pipeline_widths_string = NULL;
    }

    void init()
    {
        int ntok = sscanf(gpgpu_shader_core_pipeline_opt,"%d:%d", 
                          &n_thread_per_shader,
                          &warp_size);
        if(ntok != 2) {
           printf("GPGPU-Sim uArch: error while parsing configuration string gpgpu_shader_core_pipeline_opt\n");
           abort();
	    }

        char* toks = new char[100];
        char* tokd = toks;
        strcpy(toks,pipeline_widths_string);

        toks = strtok(toks,",");
        for (unsigned i = 0; i < N_PIPELINE_STAGES; i++) { 
            assert(toks);
            ntok = sscanf(toks,"%d", &pipe_widths[i]);
            assert(ntok == 1); 
            toks = strtok(NULL,",");
        }
        delete[] tokd;

        // EDGE
        _nIntThreads = 0;
        _nIntWarps   = 0;
        _nIntCTAs    = 0;
        if( _intMode ) {
            ntok = sscanf( _edgeIntConfig, "%d:%d", &_nIntCTAs, &_nIntWarps );
            if( ntok != 2 ) {
               printf("GPGPU-Sim uArch: error while parsing configuration string edge_int_config\n");
               abort();
            }
            _nIntThreads = _nIntWarps*warp_size;
            assert( _nIntThreads == warp_size && _nIntWarps == 1 && _nIntCTAs == 1 );

            max_cta_per_core    += _edgeEventCtasPerCore;                                                                     
            if( isIntDedicated() == 1 ) {
                // Increase the maximum values for threads per core to include the interrupt contexts
                n_thread_per_shader += _nIntThreads;                                                                  
            } 
            // pseudo_dedicated doesn't require any additionl threads since it uses existing hardware. However, 
            // each SM needs to pre-reserve the thread/register/shared memory requirements
            
            // Else, we're using an existing warp context, so no need for additional cta/warp/thread contexts
        }

        if (n_thread_per_shader > MAX_THREAD_PER_SM) {
           printf("GPGPU-Sim uArch: Error ** increase MAX_THREAD_PER_SM in abstract_hardware_model.h from %u to %u\n", 
                  MAX_THREAD_PER_SM, n_thread_per_shader);
           abort();
        }

        max_warps_per_shader =  n_thread_per_shader/warp_size;
        assert( !(n_thread_per_shader % warp_size) );
        max_sfu_latency = 512;
        max_sp_latency = 32;
        m_L1I_config.init(m_L1I_config.m_config_string,FuncCachePreferNone);
        m_L1T_config.init(m_L1T_config.m_config_string,FuncCachePreferNone);
        m_L1C_config.init(m_L1C_config.m_config_string,FuncCachePreferNone);
        m_L1D_config.init(m_L1D_config.m_config_string,FuncCachePreferNone);
        gpgpu_cache_texl1_linesize = m_L1T_config.get_line_sz();
        gpgpu_cache_constl1_linesize = m_L1C_config.get_line_sz();
        m_valid = true;

    }

    unsigned isIntDedicated() const
    {
        std::string edgeWarpSel = _edgeWarpSelectionStr;
        if ( _intMode && (edgeWarpSel.find("pseudo_dedicated") != std::string::npos) ) {
            return 2;
        } else if( _intMode && (edgeWarpSel.find("dedicated") != std::string::npos) ) {
            return 1;
        }
        return 0;
    }

    void reg_options(class OptionParser * opp );
    unsigned max_cta( const kernel_info_t &k ) const;
    unsigned num_shader() const { return n_simt_clusters*n_simt_cores_per_cluster; }
    unsigned sid_to_cluster( unsigned sid ) const { return sid / n_simt_cores_per_cluster; }
    unsigned sid_to_cid( unsigned sid )     const { return sid % n_simt_cores_per_cluster; }
    unsigned cid_to_sid( unsigned cid, unsigned cluster_id ) const { return cluster_id*n_simt_cores_per_cluster + cid; }

// data
    char *gpgpu_shader_core_pipeline_opt;
    bool gpgpu_perfect_mem;
    bool gpgpu_clock_gated_reg_file;
    bool gpgpu_clock_gated_lanes;
    enum divergence_support_t model;
    unsigned n_thread_per_shader;
    unsigned n_regfile_gating_group;
    unsigned max_warps_per_shader; 
    unsigned max_cta_per_core; //Limit on number of concurrent CTAs in shader core

    char * gpgpu_scheduler_string;

    char* pipeline_widths_string;
    int pipe_widths[N_PIPELINE_STAGES];

    mutable cache_config m_L1I_config;
    mutable cache_config m_L1T_config;
    mutable cache_config m_L1C_config;
    mutable cache_config m_L1D_config;

    bool gmem_skip_L1D; // on = global memory access always skip the L1 cache 
    
    bool gpgpu_dwf_reg_bankconflict;

    bool gpgpu_cycle_sched_prio;
    int gpgpu_num_sched_per_core;
    int gpgpu_max_insn_issue_per_warp;

    //op collector
    int gpgpu_operand_collector_num_units_sp;
    int gpgpu_operand_collector_num_units_sfu;
    int gpgpu_operand_collector_num_units_mem;
    int gpgpu_operand_collector_num_units_gen;

    unsigned int gpgpu_operand_collector_num_in_ports_sp;
    unsigned int gpgpu_operand_collector_num_in_ports_sfu;
    unsigned int gpgpu_operand_collector_num_in_ports_mem;
    unsigned int gpgpu_operand_collector_num_in_ports_gen;

    unsigned int gpgpu_operand_collector_num_out_ports_sp;
    unsigned int gpgpu_operand_collector_num_out_ports_sfu;
    unsigned int gpgpu_operand_collector_num_out_ports_mem;
    unsigned int gpgpu_operand_collector_num_out_ports_gen;

    int gpgpu_num_sp_units;
    int gpgpu_num_sfu_units;
    int gpgpu_num_mem_units;

    //Shader core resources
    unsigned gpgpu_shader_registers;
    int gpgpu_warpdistro_shader;
    int gpgpu_warp_issue_shader;
    unsigned gpgpu_num_reg_banks;
    bool gpgpu_reg_bank_use_warp_id;
    bool gpgpu_local_mem_map;
    
    unsigned max_sp_latency;
    unsigned max_sfu_latency;
    
    unsigned n_simt_cores_per_cluster;
    unsigned n_simt_clusters;
    unsigned n_simt_ejection_buffer_size;
    unsigned ldst_unit_response_queue_size;

    int simt_core_sim_order; 

    unsigned gpgpu_fetch_decode_width;

    unsigned mem2device(unsigned memid) const { return memid + n_simt_clusters; }
    unsigned num_cluster() const { return n_simt_clusters; }

    // CDP - Concurrent kernels
    bool gpgpu_concurrent_kernel_sm;

    // EDGE - interrupt
    bool                _intMode;
    char*               _edgeIntConfig;
    char*               _edgeWarpSelectionStr;
    unsigned            _nIntThreads;
    unsigned            _nIntWarps;
    unsigned            _nIntCTAs;

    bool                _edgeInternalInt;
    unsigned            _edgeInternalIntPeriod;
    unsigned            _edgeInternalIntDelay;
    unsigned            _edgeInternalEventId;

    bool                _edgeWarmupInt;
    bool                _edgeIntFetchPriority;
    bool                _edgeIntSchedulePriority;
    bool                _edgeIntReserveIcache;

    bool                _edgeVictimWarpHighPriority;
    bool                _edgeFlushIBuffer;
    bool                _edgeReplayLoads;

    unsigned            _edgeEventPriority;
    unsigned            _edgeMaxTimerEvents;
    bool                _edgeTimerEventOnly;
    bool                _edgeSkipBarrier;
    unsigned            _edgePreemptionEnginesPerShader;
    bool                _edgeUseKernelCPIForPreemptionCost;

    // EDGE - event
    unsigned            _edgeEventKernelLaunchLatency;

    unsigned            _edgeEventReserveCta;
    unsigned            _edgeEventReserveSm;
    unsigned            _edgeEventPreemptSm;
    unsigned            _edgePrintStat;
    bool                _edgeSingleSmForIsr;
    unsigned            _edgeSingleSmForIsrIdx;
    bool                _edgeUseAllSmsForEventKernels;
    bool                _edgeRunISR;
    bool                _edgeRunSmallEventAsFastPath;
    bool                _edgeUseIssueBlock2CoreWhenFreeWarp;
    bool                _edgeUseIntCoreId;
    unsigned            _edgeEventCtasPerCore;
    bool                _edgeStopOtherWarpsInPreemptedCTA;
    bool                _edgeStopOtherWarpsInPreemptedSm;
    bool                _edgeEnableRegisterRenamingInsteadBackup;
    unsigned            _edgeEventStartCycle;
    unsigned                _edgeLimitConcurrentEvents;
    bool                _edgeGenRequestsPatternByDelayAndLimit; //for batch only for now
    bool                _edgeDontLaunchEventKernel;
    bool                _edgeLaunchMultiWarpEventKernel;
    bool                _edgeSyncPreemptionEngines;

    bool                _isMemcConv;

};

struct shader_core_stats_pod {

	void* shader_core_stats_pod_start[]; // DO NOT MOVE FROM THE TOP - spaceless pointer to the start of this structure
	unsigned long long *shader_cycles;
    unsigned *m_num_sim_insn; // number of scalar thread instructions committed by this shader core
    unsigned *m_num_sim_winsn; // number of warp instructions committed by this shader core
	unsigned *m_last_num_sim_insn;
	unsigned *m_last_num_sim_winsn;
    unsigned *m_num_decoded_insn; // number of instructions decoded by this shader core
    float *m_pipeline_duty_cycle;
    unsigned *m_num_FPdecoded_insn;
    unsigned *m_num_INTdecoded_insn;
    unsigned *m_num_storequeued_insn;
    unsigned *m_num_loadqueued_insn;
    unsigned *m_num_ialu_acesses;
    unsigned *m_num_fp_acesses;
    unsigned *m_num_imul_acesses;
    unsigned *m_num_tex_inst;
    unsigned *m_num_fpmul_acesses;
    unsigned *m_num_idiv_acesses;
    unsigned *m_num_fpdiv_acesses;
    unsigned *m_num_sp_acesses;
    unsigned *m_num_sfu_acesses;
    unsigned *m_num_trans_acesses;
    unsigned *m_num_mem_acesses;
    unsigned *m_num_sp_committed;
    unsigned *m_num_tlb_hits;
    unsigned *m_num_tlb_accesses;
    unsigned *m_num_sfu_committed;
    unsigned *m_num_mem_committed;
    unsigned *m_read_regfile_acesses;
    unsigned *m_write_regfile_acesses;
    unsigned *m_non_rf_operands;
    unsigned *m_num_imul24_acesses;
    unsigned *m_num_imul32_acesses;
    unsigned *m_active_sp_lanes;
    unsigned *m_active_sfu_lanes;
    unsigned *m_active_fu_lanes;
    unsigned *m_active_fu_mem_lanes;
    unsigned *m_n_diverge;    // number of divergence occurring in this shader
    unsigned gpgpu_n_load_insn;
    unsigned gpgpu_n_store_insn;
    unsigned gpgpu_n_shmem_insn;
    unsigned gpgpu_n_tex_insn;
    unsigned gpgpu_n_const_insn;
    unsigned gpgpu_n_param_insn;
    unsigned gpgpu_n_shmem_bkconflict;
    unsigned gpgpu_n_cache_bkconflict;
    int      gpgpu_n_intrawarp_mshr_merge;
    unsigned gpgpu_n_cmem_portconflict;
    unsigned gpu_stall_shd_mem_breakdown[N_MEM_STAGE_ACCESS_TYPE][N_MEM_STAGE_STALL_TYPE];
    unsigned gpu_reg_bank_conflict_stalls;
    unsigned *shader_cycle_distro;
    unsigned *last_shader_cycle_distro;
    unsigned *num_warps_issuable;
    unsigned gpgpu_n_stall_shd_mem;

    //memory access classification
    int gpgpu_n_mem_read_local;
    int gpgpu_n_mem_write_local;
    int gpgpu_n_mem_texture;
    int gpgpu_n_mem_const;
    int gpgpu_n_mem_read_global;
    int gpgpu_n_mem_write_global;
    int gpgpu_n_mem_read_inst;
    
    int gpgpu_n_mem_l2_writeback;
    int gpgpu_n_mem_l1_write_allocate; 
    int gpgpu_n_mem_l2_write_allocate;

    unsigned made_write_mfs;
    unsigned made_read_mfs;

    unsigned *gpgpu_n_shmem_bank_access;
    long *n_simt_to_mem; // Interconnect power stats
    long *n_mem_to_simt;

};

class shader_core_stats : public shader_core_stats_pod {
public:
    shader_core_stats( const shader_core_config *config )
    {
        m_config = config;
        shader_core_stats_pod *pod = reinterpret_cast< shader_core_stats_pod * > ( this->shader_core_stats_pod_start );
        memset(pod,0,sizeof(shader_core_stats_pod));
        shader_cycles=(unsigned long long *) calloc(config->num_shader(),sizeof(unsigned long long ));
        m_num_sim_insn = (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_num_sim_winsn = (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_last_num_sim_winsn = (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_last_num_sim_insn = (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_pipeline_duty_cycle=(float*) calloc(config->num_shader(),sizeof(float));
        m_num_decoded_insn = (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_num_FPdecoded_insn = (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_num_storequeued_insn=(unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_num_loadqueued_insn=(unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_num_INTdecoded_insn = (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_num_ialu_acesses = (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_num_fp_acesses= (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_num_tex_inst= (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_num_imul_acesses= (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_num_imul24_acesses= (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_num_imul32_acesses= (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_num_fpmul_acesses= (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_num_idiv_acesses= (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_num_fpdiv_acesses= (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_num_sp_acesses= (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_num_sfu_acesses= (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_num_trans_acesses= (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_num_mem_acesses= (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_num_sp_committed= (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_num_tlb_hits=(unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_num_tlb_accesses=(unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_active_sp_lanes= (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_active_sfu_lanes= (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_active_fu_lanes= (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_active_fu_mem_lanes= (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_num_sfu_committed= (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_num_mem_committed= (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_read_regfile_acesses= (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_write_regfile_acesses= (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_non_rf_operands=(unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_n_diverge = (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        shader_cycle_distro = (unsigned*) calloc(config->warp_size+3, sizeof(unsigned));
        last_shader_cycle_distro = (unsigned*) calloc(m_config->warp_size+3, sizeof(unsigned));

        n_simt_to_mem = (long *)calloc(config->num_shader(), sizeof(long));
        n_mem_to_simt = (long *)calloc(config->num_shader(), sizeof(long));

        m_outgoing_traffic_stats = new traffic_breakdown("coretomem"); 
        m_incoming_traffic_stats = new traffic_breakdown("memtocore"); 

        gpgpu_n_shmem_bank_access = (unsigned *)calloc(config->num_shader(), sizeof(unsigned));

        m_shader_dynamic_warp_issue_distro.resize( config->num_shader() );
        m_shader_warp_slot_issue_distro.resize( config->num_shader() );
    }

    ~shader_core_stats()
    {
        delete m_outgoing_traffic_stats; 
        delete m_incoming_traffic_stats; 
        free(m_num_sim_insn); 
        free(m_num_sim_winsn);
        free(m_n_diverge); 
        free(shader_cycle_distro);
        free(last_shader_cycle_distro);
    }

    void new_grid()
    {
    }

    void event_warp_issued( unsigned s_id, unsigned warp_id, unsigned num_issued, unsigned dynamic_warp_id );

    void visualizer_print( gzFile visualizer_file );

    void print( FILE *fout ) const;

    const std::vector< std::vector<unsigned> >& get_dynamic_warp_issue() const
    {
        return m_shader_dynamic_warp_issue_distro;
    }

    const std::vector< std::vector<unsigned> >& get_warp_slot_issue() const
    {
        return m_shader_warp_slot_issue_distro;
    }

    
private:
    const shader_core_config *m_config;

    traffic_breakdown *m_outgoing_traffic_stats; // core to memory partitions
    traffic_breakdown *m_incoming_traffic_stats; // memory partition to core 

    // Counts the instructions issued for each dynamic warp.
    std::vector< std::vector<unsigned> > m_shader_dynamic_warp_issue_distro;
    std::vector<unsigned> m_last_shader_dynamic_warp_issue_distro;
    std::vector< std::vector<unsigned> > m_shader_warp_slot_issue_distro;
    std::vector<unsigned> m_last_shader_warp_slot_issue_distro;

    friend class power_stat_t;
    friend class shader_core_ctx;
    friend class ldst_unit;
    friend class simt_core_cluster;
    friend class scheduler_unit;
    friend class TwoLevelScheduler;
    friend class LooseRoundRobbinScheduler;
};


class shader_core_mem_fetch_allocator : public mem_fetch_allocator {
public:
    shader_core_mem_fetch_allocator( unsigned core_id, unsigned cluster_id, const memory_config *config )
    {
    	m_core_id = core_id;
    	m_cluster_id = cluster_id;
    	m_memory_config = config;
    }
    mem_fetch *alloc( new_addr_type addr, mem_access_type type, unsigned size, bool wr ) const 
    {
    	mem_access_t access( type, addr, size, wr );
    	mem_fetch *mf = new mem_fetch( access, 
    				       NULL,
    				       wr?WRITE_PACKET_SIZE:READ_PACKET_SIZE, 
    				       -1, 
    				       m_core_id, 
    				       m_cluster_id,
    				       m_memory_config );
    	return mf;
    }
    
    mem_fetch *alloc( const warp_inst_t &inst, const mem_access_t &access ) const
    {
        warp_inst_t inst_copy = inst;
        mem_fetch *mf = new mem_fetch(access, 
                                      &inst_copy, 
                                      access.is_write()?WRITE_PACKET_SIZE:READ_PACKET_SIZE,
                                      inst.warp_id(),
                                      m_core_id, 
                                      m_cluster_id, 
                                      m_memory_config);
        return mf;
    }

private:
    unsigned m_core_id;
    unsigned m_cluster_id;
    const memory_config *m_memory_config;
};

class EdgeSaveState
{
public:
    EdgeSaveState(class shader_core_ctx *shader, unsigned warpSize);
    ~EdgeSaveState();

    bool saveState(unsigned wid, unsigned cid, unsigned ctaStatus, shd_warp_t* warp,
            simt_stack* simtStack, ptx_thread_info** threadInfo, thread_ctx_t* threadState,
            std::bitset<MAX_THREAD_PER_SM>& activeThreads, kernel_info_t* kernel, 
            std::bitset<MAX_THREAD_PER_SM>& occupiedHwThreads, 
            bool isActive, bool isAtBarrier);

    bool restoreState(shd_warp_t* warp, simt_stack** simtStack, ptx_thread_info** threadInfo,
            thread_ctx_t* threadState, std::bitset<MAX_THREAD_PER_SM>& activeThreads,
            std::bitset<MAX_THREAD_PER_SM>& occupiedHwThreads,
            bool& isActive, bool& isAtBarrier);

    bool valid() const { return _valid; }
    shd_warp_t* savedWarp() const { return _warp; }
    void setId(unsigned new_id) { id = new_id; }
    unsigned wid() { return _wid; }
    unsigned cid() { return _cid; }
    kernel_info_t* kernel() { return _kernel; };
private:
    bool                _valid;         // Is the warp save state valid
    unsigned            _warpSize;
    unsigned            _wid;           // Warp ID
    unsigned            _cid;           // CTA ID
    unsigned            _ctaStatus;     // CTA Status?

    shd_warp_t*         _warp;
    simt_stack*         _simtStack;
    ptx_thread_info**   _threadInfo;
    thread_ctx_t*       _threadState;
    std::bitset<32>     _activeThreads;
    std::bitset<32>     _occupiedHwThreads;

    bool                _isActive;
    bool                _isAtBarrier;

    kernel_info_t*      _kernel; 
    unsigned id;
};

class shader_preemption_engine {
public:
    shader_preemption_engine(class shader_core_ctx* m_shader_p, 
                             class gpgpu_sim* m_gpu_p, 
                             const struct shader_core_config *m_config_p,
                             std::vector<shd_warp_t>* m_warp_p) { 
        m_shader = m_shader_p;
        m_gpu = m_gpu_p;
        m_config = m_config_p;
        m_warp = m_warp_p;
        edgeIntCycleReset();
    }
    void edgeIntCycleReset();
    void edgeIntCycle();
    void setiKernel(kernel_info_t* kernel) { _iKernel = kernel; }
    EdgeIntStates getEdgeState() { return _edgeIntState; }
    void setEdgeState(EdgeIntStates newState) { _edgeIntState = newState; }
    
    
    unsigned GetWid() { return _edgeWid; }
    unsigned GetCid() { return _edgeCid; }
    bool PreemptionInProgress() { return (_edgeIntState != IDLE && _edgeWid != -1); }
    void StartNewWarpPreemption(kernel_info_t* kernel, bool saveContext, bool launchKernel);
    void EnableRegSavingInKernel();
    unsigned long long getTotalIntSchedStalls() const { return _edgeTotalIntSchedStall; }
    unsigned long long getTotalIntRunCycles() const { return _edgeTotalIntRunCycles; }
    unsigned long long getTotalIntSchedCycles() const { return _edgeTotalIntSchedCycles; }


private:
    shader_core_ctx* m_shader;
    int _edgeWid;       // Current interrupt warp id
    int _edgeCid;       // Current interrupt cta id
    bool _edgeIsFree;   // If current interrupt has a free context 
    bool _edgeDoFlush; 
    bool _edgeDropLoads;
    
    
    EdgeIntStates _edgeIntState;
    gpgpu_sim *m_gpu;
    const struct shader_core_config* m_config;
    //std::vector<kernel_info_t*> _eventKernelQueue;
    
    std::vector<shd_warp_t>* m_warp;
    kernel_info_t* _iKernel;

    unsigned long long _edgeIntWarpSelectCycle;
    unsigned long long _edgeIntLaunchCycle;
    unsigned long long _edgeTotalIntSchedStall = 0;
    unsigned long long _edgeCurrentIntSchedStallCycles = 0;
    unsigned long long _edgeTotalIntRunCycles = 0;
    unsigned long long _edgeTotalIntSchedCycles;
    
    //
    //unsigned startThread;
    //unsigned endThread;
    bool _saveContext;
    bool _launchKernel;

    ///shd_warp_t& warp(int i) { return (*m_warp)[i]; }   
};



class shader_core_ctx : public core_t {
public:
    // creator:
    shader_core_ctx( class gpgpu_sim *gpu,
                     class simt_core_cluster *cluster,
                     unsigned shader_id,
                     unsigned tpc_id,
                     const struct shader_core_config *config,
                     const struct memory_config *mem_config,
                     shader_core_stats *stats);

    ~shader_core_ctx()
    {
        delete _edgeSaveState;     
    }

// used by simt_core_cluster:
    // modifiers
    void cycle();
    void reinit(unsigned start_thread, unsigned end_thread, bool reset_not_completed );
    void issue_block2core( class kernel_info_t &kernel );
    void cache_flush();
    void accept_fetch_response( mem_fetch *mf );
    void accept_ldst_unit_response( class mem_fetch * mf );
    
    // EDGE
    int freeReservedEventAvailable() 
    {
        for( int i=0; i<_edgeActiveReservedCtas.size(); ++i ) { 
            if( _edgeActiveReservedCtas[i] == false )
                return i;
        }
        return -1;
    }

    unsigned numReservedEvents() { return _edgeActiveReservedCtas.size(); }
    bool isReservedEvent(int idx) { return _edgeActiveReservedCtas[idx]; }
    void setReservedEvent(int idx) { _edgeActiveReservedCtas[idx] = true; }
    void clearReservedEvent(int idx) 
    { 
        assert( _edgeActiveReservedCtas[idx] );
        _edgeActiveReservedCtas[idx] = false; 
    }


    void issueEventBlock2Core(class kernel_info_t &kernel, unsigned idx);
    unsigned ldst_unit_wb_inst(warp_inst_t &inst) { return m_ldst_unit->writebackInst(inst); }

    void set_kernel( kernel_info_t *k ) 
    {
        assert(k);
        m_kernel=k; 
        // CDP - Concurrent kernels
        //k->inc_running(); 
        printf("GPGPU-Sim uArch: Shader %d bind to kernel %u \'%s\'\n", m_sid, m_kernel->get_uid(),
                 m_kernel->name().c_str() );
    }

    // Callback from gem5
    std::map<kernel_info_t*,bool> _kernelFinishing;
    void startKernelFinish(kernel_info_t* k);
    void finishKernel();
    bool kernelFinishIssued(kernel_info_t* k);
    bool kernelFinishing();
    bool kernelCompletePending() { return !_kernelCompletePending.empty(); }
    kernel_info_t* nextKernelToComplete() {
        if( kernelCompletePending() )
            return _kernelCompletePending.front();
        else
            return NULL;
    }

#if 0
    bool m_kernel_finishing;
    void start_kernel_finish();
    void finish_kernel();
    bool kernel_finish_issued() { return m_kernel_finishing; }
#endif

    /************************************************************/
    // EDGE
    
    bool edgeReserveEventResources(kernel_info_t* k, unsigned nCtas); 
    

    void initEDGE(kernel_info_t* k);
    void setIntSignal();
    void clearIntSignal();

    void resetOccupiedResources();

    bool intInProgress();
    //bool onlyIWarpRunning() const { return ( isIWarpRunning() && m_not_completed == 32 ); }  

    void scheduleInt();
    unsigned GetEdgeWidHighPrio(bool &fetch_victim_warp, bool &fetch_event_warp, std::list<int>* visitedEventWarps);
    bool selectVictimWarp(int& warpId, kernel_info_t* EventKernel, bool interrupty, bool launchKernel);
    bool selectIntCtaWarpCtx(int& ctaId, int& warpId, kernel_info_t* EventKernel, bool interrupt, bool occupy, bool launchKernel);
    int findFreeCta(); // FIXME: Make private
    void configureIntCtx(int cid, int wid, bool isFree, kernel_info_t* _iKernel);
    int canInterruptWarp(const shd_warp_t* victimWarp, kernel_info_t* EventKernel, bool launchKernel) const;

    bool warpNeedsFlush(int wid);

    unsigned NumFreeIntWarpsAvailable(kernel_info_t* EventKernel);
    unsigned NumVictimWarpsAvailable(kernel_info_t* EventKernel, bool launchKernel);
    std::vector<unsigned> VictimWarpsAvailable(kernel_info_t* EventKernel, bool launchKernel);

    unsigned warpPendingLoads(int wid) const { return m_warp[wid].pendingLoads(); }
    unsigned allWarpsPendingLoads();

    void warpIncPendingLoads(int wid) { m_warp[wid].incPendingLoads(); }
    void warpDecPendingLoads(int wid) { m_warp[wid].decPendingLoads(); }

    shd_warp_t* getWarp(int wid) { return &m_warp[wid]; }
    ldst_unit* ldstUnit() const { return m_ldst_unit; }

    void setSimtStack(int wid, simt_stack* stack);
    void setThreadPc(int wid, address_type npc, const active_mask_t& active);
    void incReplayLoads() { _nReplayLoads++; }
    void incBadReplayLoads() { _nBadReplayLoads++; }

    void addLateInFlightLoad(PendingLoadInst& pli);
    bool lateInFlightLoad(const warp_inst_t* inst); 
    bool loadNextWb(const warp_inst_t* inst) { return m_ldst_unit->loadNextWb(inst); }

    void edgeIntCycle();
    void edgeIntCycleReset();

    void edgeResetIntState(unsigned CurrEdgeWid, unsigned CurrEdgeCid, bool freeThreads);
    void edgeSaveResetHwState(EdgeSaveState* _edgeSaveStateInput, unsigned wid, unsigned cid);
    void edgeRestoreHwState(EdgeSaveState* _edgeSaveStateInput);

    bool edgeWarpPendingRegWrites(int warpId);

    kernel_info_t* getWarpKernel(int warpId); 

    bool delayIntWarp() const;
    void printStats(FILE* f);

    void printICacheUtilization();

    unsigned long long getMinIntSchedStalls() const { return _edgeMinIntSchedStall; }
    unsigned long long getMaxIntSchedStalls() const { return _edgeMaxIntSchedStall; }
    void setMinIntSchedStalls(unsigned long long val) { _edgeMinIntSchedStall = val; }
    void setMaxIntSchedStalls(unsigned long long val) { _edgeMaxIntSchedStall = val; }
    unsigned long long getTotalIntSchedStalls();

    unsigned long long getNumIntFreeWarps() const { return _edgeNumFreeWarpInts; }
    void IncNNumIntFreeWarps() { _edgeNumFreeWarpInts++; }
    unsigned long long getNumIntVictimWarps() const { return _edgeNumVictimWarpInts; }
    void IncNumVictimWarpInts() { _edgeNumVictimWarpInts++; }
    unsigned long long getNumNoFlushVictimWarps() const { return _edgeNumNoFlushVictimWarpInts; }
    unsigned long long getNumIntExitWarps() const { return _edgeNumExitWarpInts; }
    void IncNumExitWarpInts() { _edgeNumExitWarpInts++; }
        
    unsigned long long getNumReplayLoads() const { return _nReplayLoads; }
    unsigned long long getNumBadReplayLoads() const { return _nBadReplayLoads; }

    EdgeIntStallStats getEdgeIntStallStats() const { return _edgeIntStallStats; } 

    unsigned long long getNumEdgeBarriers() const { return _edgeNumEdgeBarriers; }
    unsigned long long getNumEdgeReleaseBarriers() const { return _edgeNumEdgeReleaseBarriers; }

    unsigned long long getTotalIntSchedCycles();

    unsigned long long getTotalIntRunCycles();

    unsigned long long getTotalBarriersSkipped() const { return _edgeSkippedBarriers; }
    unsigned long long getTotalBarriersRestored() const { return _edgeBarriersRestored; }

    unsigned long long getIdleCyclesNum() const { return _edgeNumOfIdleCycles; }
    void incIdleCyclesNum() { _edgeNumOfIdleCycles++; }

    // EDGE special barrier
    unsigned barrierType(const warp_inst_t* inst);

    std::list<unsigned long long> _edgeInterruptAssertCycle;
    std::list<unsigned long long> _edgeInterruptAssertCycleForISR;
    unsigned long long edge_get_int_start_cycle() { 
        if (_edgeInterruptAssertCycle.empty()) {
            printf("EdgeErr: Trying to access _edgeInterruptAssertCycle on core %d but its empty\n", m_sid);
            abort();
        }
        unsigned long long result = _edgeInterruptAssertCycle.front(); 
        _edgeInterruptAssertCycle.pop_front(); return result; 
    }
    unsigned long long edge_get_int_start_cycle_for_isr() { 
        if (_edgeInterruptAssertCycleForISR.empty()) {
            //printf("EdgeErr: Trying to access _edgeInterruptAssertCycle on core %d but its empty\n", m_sid);
            //abort();
            return 0;
        }
        unsigned long long result = _edgeInterruptAssertCycleForISR.front(); 
        _edgeInterruptAssertCycleForISR.pop_front(); return result; 
    }

    /************************************************************/

    // accessors
    bool fetch_unit_response_buffer_full() const;
    bool ldst_unit_response_buffer_full() const;
    unsigned get_not_completed() const { return m_not_completed; }
    unsigned get_n_active_cta() const { return m_n_active_cta; }
    void inc_n_active_cta() { m_n_active_cta++; }

    unsigned isactive() const {if(m_n_active_cta>0) return 1; else return 0;}
    kernel_info_t *get_kernel() { return m_kernel; }
    unsigned get_sid() const {return m_sid;}

// used by functional simulation:
    // modifiers
    virtual void warp_exit( unsigned warp_id );
    
    // accessors
    virtual bool warp_waiting_at_barrier( unsigned warp_id ) const;
    void warp_reaches_barrier(warp_inst_t &inst);
    bool fence_unblock_needed(unsigned warp_id) {
        return m_warp[warp_id].get_membar();
    }
    void complete_fence(unsigned warp_id) {
        assert(m_warp[warp_id].get_membar());
        m_warp[warp_id].clear_membar();
    }
    void get_pdom_stack_top_info( unsigned tid, unsigned *pc, unsigned *rpc ) const;

// used by pipeline timing model components:
    // modifiers
    void mem_instruction_stats(const warp_inst_t &inst);
    void decrement_atomic_count( unsigned wid, unsigned n );
    void inc_store_req( unsigned warp_id) { m_warp[warp_id].inc_store_req(); }
    void dec_inst_in_pipeline( unsigned warp_id ) { m_warp[warp_id].dec_inst_in_pipeline(); } // also used in writeback()
    void store_ack( class mem_fetch *mf );
    bool warp_waiting_at_mem_barrier( unsigned warp_id );
    void set_max_cta( const kernel_info_t &kernel );
    void warp_inst_complete(const warp_inst_t &inst);
    
    // accessors
    std::list<unsigned> get_regs_written( const inst_t &fvt ) const;
    const shader_core_config *get_config() const { return m_config; }
    void print_cache_stats( FILE *fp, unsigned& dl1_accesses, unsigned& dl1_misses );

    void get_cache_stats(cache_stats &cs);
    void get_L1I_sub_stats(struct cache_sub_stats &css) const;
    void get_L1D_sub_stats(struct cache_sub_stats &css) const;
    void get_L1C_sub_stats(struct cache_sub_stats &css) const;
    void get_L1T_sub_stats(struct cache_sub_stats &css) const;

    void get_icnt_power_stats(long &n_simt_to_mem, long &n_mem_to_simt) const;

// debug:
    void display_simt_state(FILE *fout, int mask ) const;
    void display_pipeline( FILE *fout, int print_mem, int mask3bit ) const;

    unsigned get_sid() { return m_sid; }

    void incload_stat() {m_stats->m_num_loadqueued_insn[m_sid]++;}
    void incstore_stat() {m_stats->m_num_storequeued_insn[m_sid]++;}
    void incialu_stat(unsigned active_count,double latency) {
		if(m_config->gpgpu_clock_gated_lanes==false){
		  m_stats->m_num_ialu_acesses[m_sid]=m_stats->m_num_ialu_acesses[m_sid]+active_count*latency
		    + inactive_lanes_accesses_nonsfu(active_count, latency);
		}else {
        m_stats->m_num_ialu_acesses[m_sid]=m_stats->m_num_ialu_acesses[m_sid]+active_count*latency;
		}
	 }
    void inctex_stat(unsigned active_count,double latency){
    	m_stats->m_num_tex_inst[m_sid]=m_stats->m_num_tex_inst[m_sid]+active_count*latency;
    }
    void incimul_stat(unsigned active_count,double latency) {
		if(m_config->gpgpu_clock_gated_lanes==false){
		  m_stats->m_num_imul_acesses[m_sid]=m_stats->m_num_imul_acesses[m_sid]+active_count*latency
		    + inactive_lanes_accesses_nonsfu(active_count, latency);
		}else {
        m_stats->m_num_imul_acesses[m_sid]=m_stats->m_num_imul_acesses[m_sid]+active_count*latency;
		}
	 }
    void incimul24_stat(unsigned active_count,double latency) {
      if(m_config->gpgpu_clock_gated_lanes==false){
   		m_stats->m_num_imul24_acesses[m_sid]=m_stats->m_num_imul24_acesses[m_sid]+active_count*latency
		    + inactive_lanes_accesses_nonsfu(active_count, latency);
		}else {
		  m_stats->m_num_imul24_acesses[m_sid]=m_stats->m_num_imul24_acesses[m_sid]+active_count*latency;
		}
	 }
	 void incimul32_stat(unsigned active_count,double latency) {
		if(m_config->gpgpu_clock_gated_lanes==false){
		  m_stats->m_num_imul32_acesses[m_sid]=m_stats->m_num_imul32_acesses[m_sid]+active_count*latency
			 + inactive_lanes_accesses_sfu(active_count, latency);			
		}else{
		  m_stats->m_num_imul32_acesses[m_sid]=m_stats->m_num_imul32_acesses[m_sid]+active_count*latency;
		}
		//printf("Int_Mul -- Active_count: %d\n",active_count);
	 }
	 void incidiv_stat(unsigned active_count,double latency) {
		if(m_config->gpgpu_clock_gated_lanes==false){
		  m_stats->m_num_idiv_acesses[m_sid]=m_stats->m_num_idiv_acesses[m_sid]+active_count*latency
			 + inactive_lanes_accesses_sfu(active_count, latency); 
		}else {
		  m_stats->m_num_idiv_acesses[m_sid]=m_stats->m_num_idiv_acesses[m_sid]+active_count*latency;
		}
	 }
	 void incfpalu_stat(unsigned active_count,double latency) {
		if(m_config->gpgpu_clock_gated_lanes==false){
		  m_stats->m_num_fp_acesses[m_sid]=m_stats->m_num_fp_acesses[m_sid]+active_count*latency
			 + inactive_lanes_accesses_nonsfu(active_count, latency);
		}else {
        m_stats->m_num_fp_acesses[m_sid]=m_stats->m_num_fp_acesses[m_sid]+active_count*latency;
		} 
	 }
	 void incfpmul_stat(unsigned active_count,double latency) {
		 		// printf("FP MUL stat increament\n");
      if(m_config->gpgpu_clock_gated_lanes==false){
		  m_stats->m_num_fpmul_acesses[m_sid]=m_stats->m_num_fpmul_acesses[m_sid]+active_count*latency
		    + inactive_lanes_accesses_nonsfu(active_count, latency);
		}else {
        m_stats->m_num_fpmul_acesses[m_sid]=m_stats->m_num_fpmul_acesses[m_sid]+active_count*latency;
		}
	 }
	 void incfpdiv_stat(unsigned active_count,double latency) {
		if(m_config->gpgpu_clock_gated_lanes==false){
		  m_stats->m_num_fpdiv_acesses[m_sid]=m_stats->m_num_fpdiv_acesses[m_sid]+active_count*latency
			+ inactive_lanes_accesses_sfu(active_count, latency); 
		}else {
		  m_stats->m_num_fpdiv_acesses[m_sid]=m_stats->m_num_fpdiv_acesses[m_sid]+active_count*latency;
		}
	 }
	 void inctrans_stat(unsigned active_count,double latency) {
		if(m_config->gpgpu_clock_gated_lanes==false){
		  m_stats->m_num_trans_acesses[m_sid]=m_stats->m_num_trans_acesses[m_sid]+active_count*latency
			+ inactive_lanes_accesses_sfu(active_count, latency); 
		}else{
		  m_stats->m_num_trans_acesses[m_sid]=m_stats->m_num_trans_acesses[m_sid]+active_count*latency;
		}
	 }

	 void incsfu_stat(unsigned active_count,double latency) {m_stats->m_num_sfu_acesses[m_sid]=m_stats->m_num_sfu_acesses[m_sid]+active_count*latency;}
	 void incsp_stat(unsigned active_count,double latency) {m_stats->m_num_sp_acesses[m_sid]=m_stats->m_num_sp_acesses[m_sid]+active_count*latency;}
	 void incmem_stat(unsigned active_count,double latency) {
		if(m_config->gpgpu_clock_gated_lanes==false){
		  m_stats->m_num_mem_acesses[m_sid]=m_stats->m_num_mem_acesses[m_sid]+active_count*latency
		    + inactive_lanes_accesses_nonsfu(active_count, latency);
		}else {
		  m_stats->m_num_mem_acesses[m_sid]=m_stats->m_num_mem_acesses[m_sid]+active_count*latency;
		}
	 }
	 void incexecstat(warp_inst_t *&inst);

	 void incregfile_reads(unsigned active_count) {m_stats->m_read_regfile_acesses[m_sid]=m_stats->m_read_regfile_acesses[m_sid]+active_count;}
	 void incregfile_writes(unsigned active_count){m_stats->m_write_regfile_acesses[m_sid]=m_stats->m_write_regfile_acesses[m_sid]+active_count;}
	 void incnon_rf_operands(unsigned active_count){m_stats->m_non_rf_operands[m_sid]=m_stats->m_non_rf_operands[m_sid]+active_count;}

	 void incspactivelanes_stat(unsigned active_count) {m_stats->m_active_sp_lanes[m_sid]=m_stats->m_active_sp_lanes[m_sid]+active_count;}
	 void incsfuactivelanes_stat(unsigned active_count) {m_stats->m_active_sfu_lanes[m_sid]=m_stats->m_active_sfu_lanes[m_sid]+active_count;}
	 void incfuactivelanes_stat(unsigned active_count) {m_stats->m_active_fu_lanes[m_sid]=m_stats->m_active_fu_lanes[m_sid]+active_count;}
	 void incfumemactivelanes_stat(unsigned active_count) {m_stats->m_active_fu_mem_lanes[m_sid]=m_stats->m_active_fu_mem_lanes[m_sid]+active_count;}

	 void inc_simt_to_mem(unsigned n_flits){ m_stats->n_simt_to_mem[m_sid] += n_flits; }
     void edge_inc_launched_background_task_tbs_num() { launched_background_task_tbs_num += 1; };
     unsigned edge_launched_background_task_tbs_num() { return launched_background_task_tbs_num; };
     void edge_inc_launched_event_kernel_tbs_num() { launched_event_kernel_tbs_num += 1; };
     unsigned edge_launched_event_kernel_tbs_num() { return launched_event_kernel_tbs_num; };
     unsigned get_total_n_atomic() { return total_n_atomic; }
     void inc_total_n_atomic() { total_n_atomic++; }

     //MARIA functions for new EDGE
     double GetCTIofCTA(int cta_id);
     bool EventIsRunningOnWarp(unsigned wid);
     bool EventIsRunning();
     bool PreemptionInProgress();
     unsigned NumEventsRunning();
     bool CanRunEdgeEvent(kernel_info_t* eventKernel);
     unsigned TotalInstInPipeline();
     //void NewEdgeSaveStateAndIssueBlock2Core();
     void StopPreemption(bool restoreState, EdgeIntStates currState);
     void NewEdgeDoOneKernelCompletion(kernel_info_t* kernel, int cid);
     
     void ScheduleFastPathEvent(kernel_info_t* kernel);
     void StartNewWarpPreemption(kernel_info_t* kernel);
     void AddEventWarpId(unsigned wid) { _edgeEventWarpIds.push_back(wid); }
     unsigned GetOldestEventWarpId() { return _edgeEventWarpIds.front(); }
     unsigned GetEventWarpId(int idx) { return _edgeEventWarpIds[idx]; }
     shd_warp_t* ChooseVictimWarp(kernel_info_t* EventKernel, bool launchKernel);
     std::list<unsigned> _edgeCtas;

     void EnableRegSavingInKernel();
     bool AreThereFreeRegsOnShader(kernel_info_t* k);
     std::vector<unsigned> _edgeEventWarpIds;
     std::vector< shader_preemption_engine* > _edgePreemptionEngines;
     //void ClearVictimWarpPriorityIds() { 
     //   _victimWarpFetchPriorityIds.clear(); 
     //   _victimWarpIds.clear();
     //}    
     std::list<int>        _victimWarpIds;

     bool isVictimWarp(int wid) const {
        return ( std::find( _victimWarpIds.begin(), _victimWarpIds.end(), wid ) != _victimWarpIds.end() );
     }

    void AddNewEdgeState(unsigned key, EdgeSaveState* new_state) {
       _edgeSaveStateList.insert(std::make_pair(key, new_state));
    }
    EdgeSaveState* GetEdgeState(unsigned key) {
       return _edgeSaveStateList[key];
    }
    EdgeSaveState* GetAndEraseEdgeState(unsigned key) {
       EdgeSaveState* result = _edgeSaveStateList[key];
       _edgeSaveStateList.erase(key);
       return result;
    }
    unsigned long long getNumInts() const { return _edgeNumInts; }
    unsigned long long _edgeNumInts;

    void prioritizeVictimWarp(int wid) {
        for( int i=0; i<m_config->gpgpu_num_sched_per_core; ++i) {
            ((EdgeScheduler*)schedulers[i])->prioritizeVictimWarp(wid);
        }
    }

    void restoreVictimWarp(int wid) {
        for( int i=0; i<m_config->gpgpu_num_sched_per_core; ++i) {
            ((EdgeScheduler*)schedulers[i])->restoreVictimWarp(wid);
        }
    }

    void MarkEventKernelAsDoneHack(int StartWid, int EndWid, int startThread, int endThread);
    int GetNextVictimWarpToVisitInFetch();
    void AddVictimWarp(int wid) {
        assert(wid>=0);
        assert(std::find(_victimWarpIds.begin(), _victimWarpIds.end(), wid) == _victimWarpIds.end());
        _victimWarpIds.push_back(wid); //contains all warps which preemption is in progress
    }

    void AddVisitedVictimWarpForFetch(int wid) { //victim warp is fetched only once per preemption process
        assert(std::find( visitedVictimWarps.begin(), visitedVictimWarps.end(), wid ) == visitedVictimWarps.end());
        visitedVictimWarps.push_back(wid);
    }

    bool RemoveVisitedVictimWarpForFetch(int wid) { //used when preemption ends
        if (std::find( visitedVictimWarps.begin(), visitedVictimWarps.end(), wid ) == visitedVictimWarps.end()) { //can happen sometimes
            return false;
        }
        visitedVictimWarps.erase(std::find( visitedVictimWarps.begin(), visitedVictimWarps.end(), wid ) );
        return true;
    }

    void AddVictimWarpFetchPriorityId(int wid) {
        assert(wid>=0);
        assert(std::find(_victimWarpFetchPriorityIds.begin(), _victimWarpFetchPriorityIds.end(), wid) == _victimWarpFetchPriorityIds.end());
        _victimWarpFetchPriorityIds.push_back(wid); //used only for priority in fetch()
    }

    void RemoveVictimWarpFetchPriorityId(int wid) {
        assert(std::find( _victimWarpFetchPriorityIds.begin(), _victimWarpFetchPriorityIds.end(), wid ) != _victimWarpFetchPriorityIds.end());
        _victimWarpFetchPriorityIds.erase(std::find( _victimWarpFetchPriorityIds.begin(), _victimWarpFetchPriorityIds.end(), wid ) );                
    }
    std::vector<unsigned> active_ctas;
    std::vector<unsigned> _cta_ncycles;
    unsigned EdgeDrainingCost();
    unsigned EdgePreemptionCost(kernel_info_t* eventKernel);
    double GetMaxCPI(std::vector<unsigned> victim_warp_indices, int N);
    bool pendingWrites(int wid) { return m_scoreboard->pendingWrites(wid); }
private:
    std::vector<unsigned> _cta_ninsn;
    std::vector<unsigned> _cta_ninsn_left;
     std::list<int>        visitedVictimWarps;
     int _edgeFetchLastChosenWarpId = -1;
     //std::list<unsigned> _edgeCtas;
     std::list<int>        _victimWarpFetchPriorityIds; //list of warps that require preemption
     

     int _intSigCount = 0;
     unsigned launched_event_kernel_tbs_num = 0;
     unsigned total_n_atomic = 0;
     friend class ldst_unit;
     unsigned launched_background_task_tbs_num = 0;
	 unsigned inactive_lanes_accesses_sfu(unsigned active_count,double latency){
      return  ( ((32-active_count)>>1)*latency) + ( ((32-active_count)>>3)*latency) + ( ((32-active_count)>>3)*latency);
	 }
	 unsigned inactive_lanes_accesses_nonsfu(unsigned active_count,double latency){
      return  ( ((32-active_count)>>1)*latency);
	 }

    int test_res_bus(int latency);
    void init_warps(unsigned cta_id, unsigned start_thread, unsigned end_thread);
    virtual void checkExecutionStatusAndUpdate(warp_inst_t &inst, unsigned t, unsigned tid);
    address_type next_pc( int tid ) const;
    void fetch();

    // CDP - Concurrent Kernels
    void register_cta_thread_exit( unsigned cta_num, kernel_info_t* kernel );

    void decode();
    
    void issue();
    friend class scheduler_unit; //this is needed to use private issue warp.
    friend class EdgeScheduler;
    friend class TwoLevelScheduler;
    friend class LooseRoundRobbinScheduler;
    void issue_warp( register_set& warp, const warp_inst_t *pI, const active_mask_t &active_mask, unsigned warp_id );
    void func_exec_inst( warp_inst_t &inst );

     // Returns numbers of addresses in translated_addrs
    unsigned translate_local_memaddr( address_type localaddr, unsigned tid, unsigned num_shader, unsigned datasize, new_addr_type* translated_addrs );

    void read_operands();
    
    void execute();
    
    void writeback();
    
    // used in display_pipeline():
    void dump_warp_state( FILE *fout ) const;
    void print_stage(unsigned int stage, FILE *fout) const;
    unsigned long long m_last_inst_gpu_sim_cycle;
    unsigned long long m_last_inst_gpu_tot_sim_cycle;

    // general information
    unsigned m_sid; // shader id
    unsigned m_tpc; // texture processor cluster id (aka, node id when using interconnect concentration)
    const shader_core_config *m_config;
    const memory_config *m_memory_config;
    class simt_core_cluster *m_cluster;

    // statistics 
    shader_core_stats *m_stats;

    // CTA scheduling / hardware thread allocation
    unsigned m_n_active_cta; // number of Cooperative Thread Arrays (blocks) currently running on this shader.

    // EDGE: was MAX_CTA_PER_SHADER + 1
    unsigned m_cta_status[MAX_CTA_PER_SHADER]; // CTAs status 
    unsigned m_cta_start_time[MAX_CTA_PER_SHADER];

    unsigned m_not_completed; // number of threads to be completed (==0 when all thread on this core completed) 

    std::vector<kernel_info_t*> _CTAKernelMap;

    // EDGE: was MAX_THREAD_PER_SM + 32 
    std::bitset<MAX_THREAD_PER_SM> m_active_threads;
    
    // thread contexts 
    thread_ctx_t             *m_threadState;
    
    // interconnect interface
    mem_fetch_interface *m_icnt;
    shader_core_mem_fetch_allocator *m_mem_fetch_allocator;
    
    // fetch
    read_only_cache *m_L1I; // instruction cache
    int  m_last_warp_fetched;

    // decode/dispatch
    std::vector<shd_warp_t>   m_warp;   // per warp information array
    barrier_set_t             m_barriers;
    ifetch_buffer_t           m_inst_fetch_buffer;
    std::vector<register_set> m_pipeline_reg;
    Scoreboard               *m_scoreboard;
    opndcoll_rfu_t            m_operand_collector;

    //schedule
    unsigned m_scheduler_prio;
    std::vector<scheduler_unit*>  schedulers;

    // execute
    unsigned m_num_function_units;
    std::vector<pipeline_stage_name_t> m_dispatch_port;
    std::vector<pipeline_stage_name_t> m_issue_port;
    std::vector<simd_function_unit*> m_fu; // stallable pipelines should be last in this array
    ldst_unit *m_ldst_unit;
    static const unsigned MAX_ALU_LATENCY = 512;
    unsigned num_result_bus;
    std::vector< std::bitset<MAX_ALU_LATENCY>* > m_result_bus;

    // used for local address mapping with single kernel launch
    unsigned kernel_max_cta_per_shader;
    unsigned kernel_padded_threads_per_cta;
    // Used for handing out dynamic warp_ids to new warps.
    // the differnece between a warp_id and a dynamic_warp_id
    // is that the dynamic_warp_id is a running number unique to every warp
    // run on this shader, where the warp_id is the static warp slot.
    unsigned m_dynamic_warp_id;

    //EDGE
    EdgeSaveState* _edgeSaveState;
    std::map<unsigned, EdgeSaveState*> _edgeSaveStateList;
    
    std::vector<bool> _edgeActiveReservedCtas;

    //
      
    unsigned long long _edgeNumFreeWarpInts;
    unsigned long long _edgeNumVictimWarpInts;
    unsigned long long _edgeNumNoFlushVictimWarpInts;
    unsigned long long _edgeNumExitWarpInts;

    unsigned long long _nReplayLoads;
    unsigned long long _nBadReplayLoads;
    unsigned long long _edgeNumEdgeBarriers;
    unsigned long long _edgeNumEdgeReleaseBarriers;

    unsigned long long _edgeSkippedBarriers;
    unsigned long long _edgeBarriersRestored;

    unsigned long long _edgeNumOfIdleCycles = 0;

    unsigned long long _edgeMaxIntSchedStall;
    unsigned long long _edgeMinIntSchedStall;

    struct WarpIntStats {
        WarpIntStats() 
        {
            _startCycle = 0;
            _endCycle = 0;
            _nInterrupts = 0;
        }

        unsigned long long _startCycle;
        unsigned long long _endCycle;
        unsigned long long _nInterrupts;
    };

    class WarpIntStatsVector : public std::vector<std::vector<WarpIntStats*>> {};
    WarpIntStatsVector _warpIntStatsVector;


    EdgeIntStallStats _edgeIntStallStats; 

    unsigned long long _warpOccupancyPerCycle;
    unsigned long long _registerUtilizationPerCycle;

    std::vector<PendingLoadInst> _lateInFlightLoads;
    
#if 0
    struct PendingLoadInst {
        PendingLoadInst() { _valid = false; _loadReturned = 0; _pc = 0; }
        PendingLoadInst(warp_inst_t& inst, address_type pc, simt_stack& stack) 
        { 
            _valid = true; 
            _loadReturned = 0;
            _inst = inst; 
            _pc = pc; 
            _stack = stack;
        }
        bool _valid;
        unsigned _loadReturned;
        warp_inst_t _inst;
        address_type _pc;
        simt_stack _stack;
    };
#endif

    // CDP - Concurrent kernels on SM
public:
    bool can_issue_1block(kernel_info_t& kernel);
    bool occupy_shader_resource_1block(kernel_info_t& kernel, bool occupy, bool interrupt);
    void release_shader_resource_1block(unsigned hw_ctaid, kernel_info_t& kernel);
    int find_available_hwtid(unsigned int cta_size, bool occupy);
    int edgeFindAvailableHwtid(unsigned int cta_size, bool occupy);

    kernel_info_t* get_hwtid_kernel(int tid); 

    // EDGEs
    void edgeOccupyShaderResourceIntWhenWarpIsFree(kernel_info_t* k, int _edgeWid);
    void edgeRecordIntStall(int wid);

    unsigned long long getWarpOccupancyPerCycle() const { return _warpOccupancyPerCycle; }
    unsigned long long getRegisterUtilization() const { return _registerUtilizationPerCycle; }
    bool isFreeIntWarpAvailable();
    
    void inc_occupied_ctas() { m_occupied_ctas++; }
    unsigned int get_occupied_shmem() { return m_occupied_shmem; }
    unsigned int get_occupied_n_threads() { return m_occupied_n_threads; } 
    unsigned int get_occupied_regs() { return m_occupied_regs; } 
    unsigned int get_occupied_ctas() { return m_occupied_ctas; } 
private:
    unsigned int m_occupied_n_threads; 
    unsigned int m_occupied_shmem; 
    unsigned int m_occupied_regs;
    unsigned int m_occupied_ctas;
    std::bitset<MAX_THREAD_PER_SM> m_occupied_hwtid;
    std::map<unsigned int, unsigned int> m_occupied_cta_to_hwtid;     

    // kernel completion queue
    std::queue<kernel_info_t*>      _kernelCompletePending;

};

class simt_core_cluster {
public:
    simt_core_cluster( class gpgpu_sim *gpu, 
                       unsigned cluster_id, 
                       const struct shader_core_config *config, 
                       const struct memory_config *mem_config,
                       shader_core_stats *stats,
                       memory_stats_t *mstats );

    void core_cycle();
    void icnt_cycle();

    void reinit();
    unsigned issue_block2core();
    void cache_flush();
    bool icnt_injection_buffer_full(unsigned size, bool write);
    void icnt_inject_request_packet(class mem_fetch *mf);

    // for perfect memory interface
    bool response_queue_full() {
        return ( m_response_fifo.size() >= m_config->n_simt_ejection_buffer_size );
    }
    void push_response_fifo(class mem_fetch *mf) {
        m_response_fifo.push_back(mf);
    }

    void get_pdom_stack_top_info( unsigned sid, unsigned tid, unsigned *pc, unsigned *rpc ) const;
    unsigned max_cta( const kernel_info_t &kernel );
    unsigned get_not_completed() const;
    void print_not_completed( FILE *fp ) const;
    unsigned get_n_active_cta() const;
    unsigned get_n_active_sms() const;
    gpgpu_sim *get_gpu() { return m_gpu; }

    void display_pipeline( unsigned sid, FILE *fout, int print_mem, int mask );
    void print_cache_stats( FILE *fp, unsigned& dl1_accesses, unsigned& dl1_misses ) const;

    void get_cache_stats(cache_stats &cs) const;
    void get_L1I_sub_stats(struct cache_sub_stats &css) const;
    void get_L1D_sub_stats(struct cache_sub_stats &css) const;
    void get_L1C_sub_stats(struct cache_sub_stats &css) const;
    void get_L1T_sub_stats(struct cache_sub_stats &css) const;

    void get_icnt_stats(long &n_simt_to_mem, long &n_mem_to_simt) const;

    shader_core_ctx *get_core(int id_in_cluster) { return m_core[id_in_cluster]; }

    void initEDGE(kernel_info_t* k);
    bool isIWarpRunning() const;
    bool intInProgress();


private:
    unsigned m_cluster_id;
    gpgpu_sim *m_gpu;
    const shader_core_config *m_config;
    shader_core_stats *m_stats;
    memory_stats_t *m_memory_stats;
    shader_core_ctx **m_core;

    unsigned m_cta_issue_next_core;
    std::list<unsigned> m_core_sim_order;
    std::list<mem_fetch*> m_response_fifo;
};

class shader_memory_interface : public mem_fetch_interface {
public:
    shader_memory_interface( shader_core_ctx *core, simt_core_cluster *cluster ) { m_core=core; m_cluster=cluster; }
    virtual bool full( unsigned size, bool write ) const 
    {
        return m_cluster->icnt_injection_buffer_full(size,write);
    }
    virtual void push(mem_fetch *mf) 
    {
    	m_core->inc_simt_to_mem(mf->get_num_flits(true));
        m_cluster->icnt_inject_request_packet(mf);        
    }
private:
    shader_core_ctx *m_core;
    simt_core_cluster *m_cluster;
};

class perfect_memory_interface : public mem_fetch_interface {
public:
    perfect_memory_interface( shader_core_ctx *core, simt_core_cluster *cluster ) { m_core=core; m_cluster=cluster; }
    virtual bool full( unsigned size, bool write) const
    {
        return m_cluster->response_queue_full();
    }
    virtual void push(mem_fetch *mf)
    {
        if ( mf && mf->isatomic() )
            mf->do_atomic(); // execute atomic inside the "memory subsystem"
        m_core->inc_simt_to_mem(mf->get_num_flits(true));
        m_cluster->push_response_fifo(mf);        
    }
private:
    shader_core_ctx *m_core;
    simt_core_cluster *m_cluster;
};

inline int scheduler_unit::get_sid() const { return m_shader->get_sid(); }

#endif /* SHADER_H */
