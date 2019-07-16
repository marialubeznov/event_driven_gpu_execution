#ifndef __MEMC_H__
#define __MEMC_H__

#include <stdint.h>



#define initGpuShmemPtr(T, h_ptr, symbol, size)                   \
    {                                                             \
        cudaMallocHost((void**)&(h_ptr), size*sizeof(T));         \
        memset((void*)(h_ptr), 0, size*sizeof(T));                \
        cudaMemcpyToSymbol( (symbol), &(h_ptr), sizeof(void*) );  \
    }

#define initGpuGlobals(T, d_ptr, symbol, size)                    \
    {                                                             \
        cudaMalloc(&(d_ptr), size*sizeof(T));                     \
        cudaMemset((d_ptr), 0, size*sizeof(T));                   \
        cudaMemcpyToSymbol( (symbol), &(d_ptr), sizeof(T*));      \
    }

#endif /* __MEMC_H__ */
