#ifndef GPU_CACHE_GEM5_H_
#define GPU_CACHE_GEM5_H_

#include "gpu-cache.h"
#include "base/misc.hh"
#include "gpu/gpgpu-sim/cuda_core.hh"
#include "gpu/gpgpu-sim/cuda_gpu.hh"

class l1icache_gem5 : public read_only_cache {
private:    
    gpgpu_t* abstractGPU;
    CudaCore* shaderCore;
    unsigned m_sid;

    new_addr_type _edgeIntStartBlock;
    new_addr_type _edgeIntEndBlock;
    bool          _edgeIntInit;

    bool isIntAddr(unsigned addr) const
    {
        assert( _edgeIntInit );
        return ( (addr >= _edgeIntStartBlock) && (addr < _edgeIntEndBlock) );
    }

public:
    l1icache_gem5(gpgpu_t* _gpu, const char *name, cache_config &config, int core_id, int type_id, mem_fetch_interface *memport, enum mem_fetch_status status);
    enum cache_request_status access(new_addr_type addr, mem_fetch *mf, unsigned time, std::list<cache_event> &events);
    void fill(mem_fetch *mf, unsigned time);
    void cycle();

    void setEdgeInstRange(unsigned startPC, unsigned size) 
    {
        assert( !_edgeIntInit );
        _edgeIntStartBlock = m_config.block_addr(startPC);

        new_addr_type endBlock = m_config.block_addr(startPC + size);
        if( (startPC + size) > endBlock ) 
            endBlock = m_config.block_addr(startPC + size + m_config.get_line_sz());

        _edgeIntEndBlock = endBlock; //m_config.block_addr(startPC + size);
        _edgeIntInit = true;
    }

    void printEdgeUtilization();
};

#endif /* GPU_CACHE_GEM5_H_ */
