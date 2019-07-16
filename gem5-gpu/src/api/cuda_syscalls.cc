// This file created from cuda_runtime_api.h distributed with CUDA 1.1
// Changes Copyright 2009,  Tor M. Aamodt, Ali Bakhoda and George L. Yuan
// University of British Columbia

/*
 * cuda_syscalls.cc
 *
 * Copyright © 2009 by Tor M. Aamodt, Wilson W. L. Fung, Ali Bakhoda,
 * George L. Yuan and the University of British Columbia, Vancouver,
 * BC V6T 1Z4, All Rights Reserved.
 *
 * THIS IS A LEGAL DOCUMENT BY DOWNLOADING GPGPU-SIM, YOU ARE AGREEING TO THESE
 * TERMS AND CONDITIONS.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNERS OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * NOTE: The files libcuda/cuda_runtime_api.c and src/cuda-sim/cuda-math.h
 * are derived from the CUDA Toolset available from http://www.nvidia.com/cuda
 * (property of NVIDIA).  The files benchmarks/BlackScholes/ and
 * benchmarks/template/ are derived from the CUDA SDK available from
 * http://www.nvidia.com/cuda (also property of NVIDIA).  The files from
 * src/intersim/ are derived from Booksim (a simulator provided with the
 * textbook "Principles and Practices of Interconnection Networks" available
 * from http://cva.stanford.edu/books/ppin/). As such, those files are bound by
 * the corresponding legal terms and conditions set forth separately (original
 * copyright notices are left in files from these sources and where we have
 * modified a file our copyright notice appears before the original copyright
 * notice).
 *
 * Using this version of GPGPU-Sim requires a complete installation of CUDA
 * which is distributed seperately by NVIDIA under separate terms and
 * conditions.  To use this version of GPGPU-Sim with OpenCL requires a
 * recent version of NVIDIA's drivers which support OpenCL.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the University of British Columbia nor the names of
 * its contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * 4. This version of GPGPU-SIM is distributed freely for non-commercial use only.
 *
 * 5. No nonprofit user may place any restrictions on the use of this software,
 * including as modified by the user, by any other authorized user.
 *
 * 6. GPGPU-SIM was developed primarily by Tor M. Aamodt, Wilson W. L. Fung,
 * Ali Bakhoda, George L. Yuan, at the University of British Columbia,
 * Vancouver, BC V6T 1Z4
 */

/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

/*
 * Copyright (c) 2011-2013 Mark D. Hill and David A. Wood
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include <iostream>
#include <string>

#ifdef OPENGL_SUPPORT
#define GL_GLEXT_PROTOTYPES
#ifdef __APPLE__
#include <GLUT/glut.h> // Apple's version of GLUT is here
#else
#include <GL/gl.h>
#endif
#endif

#include "api/cuda_syscalls.hh"
#include "api/gpu_syscall_helper.hh"
#include "cpu/thread_context.hh"
#include "cuda-sim/cuda-sim.h"
#include "cuda-sim/ptx_ir.h"
#include "cuda-sim/ptx_loader.h"
#include "cuda-sim/ptx_parser.h"
#include "debug/GPUSyscalls.hh"
#include "gpu/gpgpu-sim/cuda_gpu.hh"
#include "gpgpusim_entrypoint.h"
#include "gpgpu-sim/gpu-sim.h"
#include "stream_manager.h"

#include "edge.h"
//======MARIA TRICK FOR IP FORWARD=======//
//#include "rte_lpm.h"
#include "ipv4.h" 
#include "ipv6.h" 
#include "bfs.h"

#include "util.h" 
//======MARIA TRICK FOR IP FORWARD=======//

#define MAX_STRING_LEN 1000

typedef struct CUstream_st *cudaStream_t;

static int load_static_globals(GPUSyscallHelper *helper, symbol_table *symtab);
static int load_constants(GPUSyscallHelper *helper, symbol_table *symtab);

unsigned g_active_device = 0; // Active CUDA-enabled GPU that runs the code
cudaError_t g_last_cudaError = cudaSuccess;

extern stream_manager *g_stream_manager;


void register_ptx_function(const char *name, function_info *impl)
{
   // TODO: Figure out the best location for this function
}

kernel_info_t *GPGPUCudaPTXSimInitGrid(gpgpu_ptx_sim_arg_list_t args,
                                       Addr simArgMemBase,
                                       struct dim3 gridDim,
                                       struct dim3 blockDim,
                                       function_info* entry,
                                       GPUSyscallHelper& helper,
                                       bool isEvent)
{
    kernel_info_t *result = new kernel_info_t(gridDim,blockDim,entry);
    if (entry == NULL) 
        panic("GPGPU-Sim PTX: ERROR launching kernel -- no PTX implementation found");
    
    unsigned argcount=args.size();
    unsigned argn=1;
    for (gpgpu_ptx_sim_arg_list_t::iterator a = args.begin(); a != args.end(); a++) {
        entry->add_param_data(argcount-argn, &(*a));

        // Store the data in host memory for this argument: FIXME: Probably need to do something for timing here.
        //              Is it really important?
        if( !isEvent ) 
            helper.writeBlob(simArgMemBase + a->m_offset, (uint8_t*)(a->m_start), a->m_nbytes, false);
        argn++;
    }
    entry->gem5Finalize(); // Finalize the parameter values
    g_ptx_kernel_count++;

    return result;
}


kernel_info_t *gpgpu_cuda_ptx_sim_init_grid(gpgpu_ptx_sim_arg_list_t args,
                                            struct dim3 gridDim,
                                            struct dim3 blockDim,
                                            function_info* entry)
{
   kernel_info_t *result = new kernel_info_t(gridDim,blockDim,entry);
   if (entry == NULL) {
       panic("GPGPU-Sim PTX: ERROR launching kernel -- no PTX implementation found");
   }
   unsigned argcount=args.size();
   unsigned argn=1;
   for (gpgpu_ptx_sim_arg_list_t::iterator a = args.begin(); a != args.end(); a++) {
      entry->add_param_data(argcount-argn, &(*a));
      argn++;
   }

   entry->finalize(result->get_param_memory());
   g_ptx_kernel_count++;

   return result;
}

#if defined __APPLE__
#   define __my_func__    __PRETTY_FUNCTION__
#else
# if defined __cplusplus ? __GNUC_PREREQ (2, 6) : __GNUC_PREREQ (2, 4)
#   define __my_func__    __PRETTY_FUNCTION__
# else
#  if defined __STDC_VERSION__ && __STDC_VERSION__ >= 199901L
#   define __my_func__    __func__
#  else
#   define __my_func__    ((__const char *) 0)
#  endif
# endif
#endif

class kernel_config {
  public:
    kernel_config(dim3 GridDim, dim3 BlockDim, size_t sharedMem, struct CUstream_st *stream)
    {
        m_GridDim=GridDim;
        m_BlockDim=BlockDim;
        m_sharedMem=sharedMem;
        m_stream = stream;
    }
    void set_arg(const void *arg, size_t size, size_t offset)
    {
        m_args.push_front(gpgpu_ptx_sim_arg(arg,size,offset));
    }
    dim3 grid_dim() const { return m_GridDim; }
    dim3 block_dim() const { return m_BlockDim; }
    gpgpu_ptx_sim_arg_list_t get_args() { return m_args; }
    struct CUstream_st *get_stream() { return m_stream; }

  private:
    dim3 m_GridDim;
    dim3 m_BlockDim;
    size_t m_sharedMem;
    struct CUstream_st *m_stream;
    gpgpu_ptx_sim_arg_list_t m_args;
};

extern "C" void ptxinfo_addinfo()
{
    if (!strcmp("__cuda_dummy_entry__",get_ptxinfo_kname())) {
      // this string produced by ptxas for empty ptx files (e.g., bandwidth test)
        clear_ptxinfo();
        return;
    }
    CudaGPU *cudaGPU = CudaGPU::getCudaGPU(g_active_device);
    print_ptxinfo();
    cudaGPU->add_ptxinfo(get_ptxinfo_kname(), get_ptxinfo_kinfo());
    clear_ptxinfo();
}

void cuda_not_implemented(const char* func, unsigned line)
{
    fflush(stdout);
    fflush(stderr);
    printf("\n\ngem5-gpu CUDA: Execution error: CUDA API function \"%s()\" has not been implemented yet.\n"
            "                 [gem5-gpu/src/gem5/%s around line %u]\n\n\n",
    func,__FILE__, line);
    fflush(stdout);
    abort();
}

typedef std::map<unsigned,CUevent_st*> event_tracker_t;

int CUevent_st::m_next_event_uid;
event_tracker_t g_timer_events;
std::list<kernel_config> g_cuda_launch_stack;

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

void
cudaMalloc(ThreadContext *tc, gpusyscall_t *call_params)
{
    GPUSyscallHelper helper(tc, call_params);

    Addr sim_devPtr = *((Addr*)helper.getParam(0, true));
    size_t sim_size = *((size_t*)helper.getParam(1));
    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: cudaMalloc(devPtr = %x, size = %d)\n", sim_devPtr, sim_size);

    g_last_cudaError = cudaSuccess;

    CudaGPU *cudaGPU = CudaGPU::getCudaGPU(g_active_device);

    if (!cudaGPU->isManagingGPUMemory()) {
        // Tell CUDA runtime to allocate memory
        cudaError_t to_return = cudaErrorApiFailureBase;
        helper.setReturn((uint8_t*)&to_return, sizeof(cudaError_t));
        return;
    } else {
        Addr addr = cudaGPU->allocateGPUMemory(sim_size);
        helper.writeBlob(sim_devPtr, (uint8_t*)(&addr), sizeof(Addr), true);
        if (addr) {
            g_last_cudaError = cudaSuccess;
        } else {
            g_last_cudaError = cudaErrorMemoryAllocation;
        }
        helper.setReturn((uint8_t*)&g_last_cudaError, sizeof(cudaError_t));
    }
}

void
cudaMallocHost(ThreadContext *tc, gpusyscall_t *call_params) {
    GPUSyscallHelper helper(tc, call_params);

    Addr sim_ptr = *((Addr*)helper.getParam(0, true));
    size_t sim_size = *((size_t*)helper.getParam(1));
    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: cudaMallocHost(ptr = %x, size = %d)\n", sim_ptr, sim_size);

    g_last_cudaError = cudaSuccess;
    // Tell CUDA runtime to allocate memory
    cudaError_t to_return = cudaErrorApiFailureBase;
    helper.setReturn((uint8_t*)&to_return, sizeof(cudaError_t));
}

void
cudaRegisterDeviceMemory(ThreadContext *tc, gpusyscall_t *call_params)
{
    // This GPU syscall is used to initialize tracking of GPU memory so that
    // the GPU can do TLB lookups and if necessary, physical memory allocations
    GPUSyscallHelper helper(tc, call_params);

    Addr sim_devicePtr = *((Addr*)helper.getParam(0, true));
    size_t sim_size = *((size_t*)helper.getParam(1));
    bool isParamMem = *((bool*)helper.getParam(1));

    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: cudaRegisterDeviceMemory(devicePtr = %x, size = %d, parameter memory = %s)\n", 
            sim_devicePtr, sim_size, (isParamMem ? "Yes" : "No"));

    CudaGPU *cudaGPU = CudaGPU::getCudaGPU(g_active_device);
    cudaGPU->registerDeviceMemory(tc, sim_devicePtr, sim_size);
}

void
cudaMallocPitch(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);
}

//__host__ cudaError_t CUDARTAPI cudaMallocArray(struct cudaArray **array, const struct cudaChannelFormatDesc *desc, size_t width, size_t height __dv(1)) {
void
cudaMallocArray(ThreadContext *tc, gpusyscall_t *call_params) {
    cuda_not_implemented(__my_func__,__LINE__);
}

void
cudaFree(ThreadContext *tc, gpusyscall_t *call_params) {
    GPUSyscallHelper helper(tc, call_params);

    Addr sim_devPtr = *((Addr*)helper.getParam(0, true));
    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: cudaFree(devPtr = %x)\n", sim_devPtr);

    CudaGPU *cudaGPU = CudaGPU::getCudaGPU(g_active_device);

    if (!cudaGPU->isManagingGPUMemory()) {
        g_last_cudaError = cudaSuccess;
        // Tell CUDA runtime to free memory
        cudaError_t to_return = cudaErrorApiFailureBase;
        helper.setReturn((uint8_t*)&to_return, sizeof(cudaError_t));
    } else {
        // TODO: Tell SPA to free this memory
        cudaError_t to_return = cudaSuccess;
        helper.setReturn((uint8_t*)&to_return, sizeof(cudaError_t));
    }
}

void
cudaFreeHost(ThreadContext *tc, gpusyscall_t *call_params) {
    GPUSyscallHelper helper(tc, call_params);

    Addr sim_ptr = *((Addr*)helper.getParam(0, true));
    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: cudaFreeHost(ptr = %x)\n", sim_ptr);

    g_last_cudaError = cudaSuccess;
    // Tell CUDA runtime to free memory
    cudaError_t to_return = cudaErrorApiFailureBase;
    helper.setReturn((uint8_t*)&to_return, sizeof(cudaError_t));
}

//__host__ cudaError_t CUDARTAPI cudaFreeArray(struct cudaArray *array){
void
cudaFreeArray(ThreadContext *tc, gpusyscall_t *call_params) {
    cuda_not_implemented(__my_func__,__LINE__);
};


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

void
cudaMemcpy(ThreadContext *tc, gpusyscall_t *call_params) {
    GPUSyscallHelper helper(tc, call_params);

    Addr sim_dst = *((Addr*)helper.getParam(0, true));
    Addr sim_src = *((Addr*)helper.getParam(1, true));
    size_t sim_count = *((size_t*)helper.getParam(2));
    enum cudaMemcpyKind sim_kind = *((enum cudaMemcpyKind*)helper.getParam(3));

    CudaGPU *cudaGPU = CudaGPU::getCudaGPU(g_active_device);

    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: cudaMemcpy(dst = %x, src = %x, count = %d, kind = %s)\n",
            sim_dst, sim_src, sim_count, cudaMemcpyKindStrings[sim_kind]);

    bool suspend = false;
    if (sim_count == 0) {
        g_last_cudaError = cudaSuccess;
        helper.setReturn((uint8_t*)&suspend, sizeof(bool));
        return;
    }

    if (sim_kind == cudaMemcpyHostToDevice) {
        stream_operation mem_op((const void*)sim_src, (size_t)sim_dst, sim_count, 0);
        mem_op.setThreadContext(tc);
        g_stream_manager->push(mem_op);
    } else if (sim_kind == cudaMemcpyDeviceToHost) {
        stream_operation mem_op((size_t)sim_src, (void*)sim_dst, sim_count, 0);
        mem_op.setThreadContext(tc);
        g_stream_manager->push(mem_op);
    } else if (sim_kind == cudaMemcpyDeviceToDevice) {
        stream_operation mem_op((size_t)sim_src, (size_t)sim_dst, sim_count, 0);
        mem_op.setThreadContext(tc);
        g_stream_manager->push(mem_op);
    } else {
        panic("GPGPU-Sim PTX: cudaMemcpy - ERROR : unsupported cudaMemcpyKind\n");
    }

    suspend = cudaGPU->needsToBlock();
    assert(suspend);
    g_last_cudaError = cudaSuccess;
    helper.setReturn((uint8_t*)&suspend, sizeof(bool));
}

//__host__ cudaError_t CUDARTAPI cudaMemcpyToArray(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind) {
void
cudaMemcpyToArray(ThreadContext *tc, gpusyscall_t *call_params) {
    cuda_not_implemented(__my_func__,__LINE__);
}

//__host__ cudaError_t CUDARTAPI cudaMemcpyFromArray(void *dst, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind) {
void
cudaMemcpyFromArray(ThreadContext *tc, gpusyscall_t *call_params) {
    cuda_not_implemented(__my_func__,__LINE__);
}

//__host__ cudaError_t CUDARTAPI cudaMemcpyArrayToArray(struct cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToDevice)) {
void
cudaMemcpyArrayToArray(ThreadContext *tc, gpusyscall_t *call_params) {
    cuda_not_implemented(__my_func__,__LINE__);
}

//__host__ cudaError_t CUDARTAPI cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind) {
void
cudaMemcpy2D(ThreadContext *tc, gpusyscall_t *call_params) {
    cuda_not_implemented(__my_func__,__LINE__);
}

//__host__ cudaError_t CUDARTAPI cudaMemcpy2DToArray(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind) {
void
cudaMemcpy2DToArray(ThreadContext *tc, gpusyscall_t *call_params) {
    cuda_not_implemented(__my_func__,__LINE__);
}

//__host__ cudaError_t CUDARTAPI cudaMemcpy2DFromArray(void *dst, size_t dpitch, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind) {
void
cudaMemcpy2DFromArray(ThreadContext *tc, gpusyscall_t *call_params) {
    cuda_not_implemented(__my_func__,__LINE__);
}

//__host__ cudaError_t CUDARTAPI cudaMemcpy2DArrayToArray(struct cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToDevice)) {
void
cudaMemcpy2DArrayToArray(ThreadContext *tc, gpusyscall_t *call_params) {
    cuda_not_implemented(__my_func__,__LINE__);
}

void
cudaMemcpyToSymbol(ThreadContext *tc, gpusyscall_t *call_params) {
    GPUSyscallHelper helper(tc, call_params);

    Addr sim_symbol = *((Addr*)helper.getParam(0, true));
    Addr sim_src = *((Addr*)helper.getParam(1, true));
    size_t sim_count = *((size_t*)helper.getParam(2));
    size_t sim_offset = *((size_t*)helper.getParam(3));
    enum cudaMemcpyKind sim_kind = *((enum cudaMemcpyKind*)helper.getParam(4));

    CudaGPU *cudaGPU = CudaGPU::getCudaGPU(g_active_device);

    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: cudaMemcpyToSymbol(symbol = %x, src = %x, count = %d, offset = %d, kind = %s)\n",
            sim_symbol, sim_src, sim_count, sim_offset, cudaMemcpyKindStrings[sim_kind]);

    assert(sim_kind == cudaMemcpyHostToDevice);
    stream_operation mem_op((const void*)sim_src, (const char*)sim_symbol, sim_count, sim_offset, NULL);
    mem_op.setThreadContext(tc);
    g_stream_manager->push(mem_op);

    bool suspend = cudaGPU->needsToBlock();
    assert(suspend);
    g_last_cudaError = cudaSuccess;
    helper.setReturn((uint8_t*)&suspend, sizeof(bool));
}

void
cudaMemcpyFromSymbol(ThreadContext *tc, gpusyscall_t *call_params) {
    GPUSyscallHelper helper(tc, call_params);

    Addr sim_dst = *((Addr*)helper.getParam(0, true));
    Addr sim_symbol = *((Addr*)helper.getParam(1, true));
    size_t sim_count = *((size_t*)helper.getParam(2));
    size_t sim_offset = *((size_t*)helper.getParam(3));
    enum cudaMemcpyKind sim_kind = *((enum cudaMemcpyKind*)helper.getParam(4));

    CudaGPU *cudaGPU = CudaGPU::getCudaGPU(g_active_device);

    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: cudaMemcpyToSymbol(symbol = %x, src = %x, count = %d, offset = %d, kind = %s)\n",
            sim_symbol, sim_dst, sim_count, sim_offset, cudaMemcpyKindStrings[sim_kind]);

    assert(sim_kind == cudaMemcpyDeviceToHost);
    stream_operation mem_op((const char*)sim_symbol, (void*)sim_dst, sim_count, sim_offset, NULL);
    mem_op.setThreadContext(tc);
    g_stream_manager->push(mem_op);

    bool suspend = cudaGPU->needsToBlock();
    assert(suspend);
    g_last_cudaError = cudaSuccess;
    helper.setReturn((uint8_t*)&suspend, sizeof(bool));
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

//	__host__ cudaError_t CUDARTAPI cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream)
void
cudaMemcpyAsync(ThreadContext *tc, gpusyscall_t *call_params)
{
    GPUSyscallHelper helper(tc, call_params);

    Addr sim_dst = *((Addr*)helper.getParam(0, true));
    Addr sim_src = *((Addr*)helper.getParam(1, true));
    size_t sim_count = *((size_t*)helper.getParam(2));
    enum cudaMemcpyKind sim_kind = *((enum cudaMemcpyKind*)helper.getParam(3));
    cudaStream_t sim_stream = *((cudaStream_t*)helper.getParam(4));

    CudaGPU *cudaGPU = CudaGPU::getCudaGPU(g_active_device);

    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: cudaMemcpyAsync(dst = %x, src = %x, count = %d, kind = %s, stream = %x)\n",
            sim_dst, sim_src, sim_count, cudaMemcpyKindStrings[sim_kind], sim_stream);

    bool suspend = false;
    if (sim_count == 0) {
        g_last_cudaError = cudaSuccess;
        helper.setReturn((uint8_t*)&suspend, sizeof(bool));
        return;
    }

    if (sim_kind == cudaMemcpyHostToDevice) {
        stream_operation mem_op((const void*)sim_src, (size_t)sim_dst, sim_count, sim_stream);
        mem_op.setThreadContext(tc);
        g_stream_manager->push(mem_op);
    } else if (sim_kind == cudaMemcpyDeviceToHost) {
        stream_operation mem_op((size_t)sim_src, (void*)sim_dst, sim_count, sim_stream);
        mem_op.setThreadContext(tc);
        g_stream_manager->push(mem_op);
    } else if (sim_kind == cudaMemcpyDeviceToDevice) {
        stream_operation mem_op((size_t)sim_src, (size_t)sim_dst, sim_count, sim_stream);
        mem_op.setThreadContext(tc);
        g_stream_manager->push(mem_op);
    } else {
        panic("GPGPU-Sim PTX: cudaMemcpy - ERROR : unsupported cudaMemcpyKind\n");
    }

    // EDGE: Don't need to block here anymore. This is an async call, so the blocking should only occur when the data needs to be used, not because an operation
    // is pending.
    //suspend = cudaGPU->needsToBlock();
    suspend = false;
    g_last_cudaError = cudaSuccess;
    helper.setReturn((uint8_t*)&suspend, sizeof(bool));
    
}

//	__host__ cudaError_t CUDARTAPI cudaMemcpyToArrayAsync(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream)
void
cudaMemcpyToArrayAsync(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);
}

//	__host__ cudaError_t CUDARTAPI cudaMemcpyFromArrayAsync(void *dst, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream)
void
cudaMemcpyFromArrayAsync(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);
}

//	__host__ cudaError_t CUDARTAPI cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream)
void
cudaMemcpy2DAsync(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);
}

void
cudaMemcpy2DToArrayAsync(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);
}

void
cudaMemcpy2DFromArrayAsync(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);
}

void
cudaBlockThread(ThreadContext *tc, gpusyscall_t *call_params)
{
    // Similar to futex in syscalls, except we need to track the variable to be set
    GPUSyscallHelper helper(tc, call_params);
    Addr sim_is_free_ptr = *((Addr*)helper.getParam(0, true));
    CUstream_st* sim_stream = *((CUstream_st**)helper.getParam(1, true));
    bool sim_deviceSync = *((bool*)helper.getParam(2));

    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: cudaBlockThread(tc = %x, is_free_ptr = %x, stream=%x, device sync = %s)\n", 
                        tc, sim_is_free_ptr, sim_stream, (sim_deviceSync ? "true" : "false") );
    
    CudaGPU *cudaGPU = CudaGPU::getCudaGPU(g_active_device);
    if( !sim_deviceSync ) 
        cudaGPU->blockThread(tc, sim_is_free_ptr, sim_stream);
    else 
        cudaGPU->synchronizeDevice(tc, sim_is_free_ptr);
}


void __cudaSleepThread(ThreadContext* tc, gpusyscall_t* call_params)
{
    GPUSyscallHelper helper(tc, call_params);
    Addr sim_is_free_ptr = *((Addr*)helper.getParam(0, true));
    size_t nSleepCycles = *((size_t*)helper.getParam(1));

    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: cudaSleepThread(tc = %x, is_free_ptr = %x, Number of sleep GPU cycles = %ld)\n", 
                    tc, sim_is_free_ptr, nSleepCycles );

    CudaGPU *cudaGPU = CudaGPU::getCudaGPU(g_active_device);

    cudaGPU->sleepThreadNCycles(tc, sim_is_free_ptr, nSleepCycles);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

// void Write2GPUMem(void *devPtr, int value, size_t count)
// {
//     GPUSyscallHelper helper(tc, call_params);

//     Addr sim_mem = devPtr;
//     int sim_c = value;
//     size_t sim_count = count;
//     DPRINTF(GPUSyscalls, "gem5 GPU Syscall: cudaMemset(mem = %x, c = %d, count = %d)\n", sim_mem, sim_c, sim_count);

//     helper.writeBlob(sim_mem, (uint8_t*)(&value), count, true);
// }

void
cudaMemset(ThreadContext *tc, gpusyscall_t *call_params)
{
    GPUSyscallHelper helper(tc, call_params);

    Addr sim_mem = *((Addr*)helper.getParam(0, true));
    int sim_c = *((int*)helper.getParam(1));
    size_t sim_count = *((size_t*)helper.getParam(2));
    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: cudaMemset(mem = %x, c = %d, count = %d)\n", sim_mem, sim_c, sim_count);

    CudaGPU *cudaGPU = CudaGPU::getCudaGPU(g_active_device);

    if (!cudaGPU->isManagingGPUMemory() && !cudaGPU->isAccessingHostPagetable()) {
        // Signal to libcuda that it should handle the memset. This is required
        // if the copy engine may be unable to access the CPU's pagetable to get
        // address translations (unified memory without access host pagetable)
        g_last_cudaError = cudaErrorApiFailureBase;
        helper.setReturn((uint8_t*)&g_last_cudaError, sizeof(cudaError_t));
    } else {
        stream_operation mem_op((size_t)sim_mem, sim_c, sim_count, 0);
        mem_op.setThreadContext(tc);
        g_stream_manager->push(mem_op);
        g_last_cudaError = cudaSuccess;
        helper.setReturn((uint8_t*)&g_last_cudaError, sizeof(cudaError_t));

        bool suspend = cudaGPU->needsToBlock();
        assert(suspend);
    }
}

void
cudaMemset2D(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

void
cudaGetSymbolAddress(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);
}

void
cudaGetSymbolSize(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

void
cudaGetDeviceCount(ThreadContext *tc, gpusyscall_t *call_params)
{
    GPUSyscallHelper helper(tc, call_params);
    Addr sim_count = *((Addr*)helper.getParam(0, true));

    int count = CudaGPU::getNumCudaDevices();
    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: cudaGetDeviceCount(count* = %x) = %d\n", sim_count, count);

    helper.writeBlob(sim_count, (uint8_t*)(&count), sizeof(int));
    g_last_cudaError = cudaSuccess;
}

void
cudaGetDeviceProperties(ThreadContext *tc, gpusyscall_t *call_params)
{
    GPUSyscallHelper helper(tc, call_params);

    Addr sim_prop = *((Addr*)helper.getParam(0, true));
    int sim_device = *((int*)helper.getParam(1));
    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: cudaGetDeviceProperties(prop* = %x, device = %d)\n", sim_prop, sim_device);
    CudaGPU *cudaGPU = CudaGPU::getCudaGPU(g_active_device);
    if (sim_device <= CudaGPU::getNumCudaDevices())  {
        CudaGPU::CudaDeviceProperties *prop = cudaGPU->getDeviceProperties();
        helper.writeBlob(sim_prop, (uint8_t*)(prop), sizeof(CudaGPU::CudaDeviceProperties));
        g_last_cudaError = cudaSuccess;
    } else {
        g_last_cudaError = cudaErrorInvalidDevice;
    }
    helper.setReturn((uint8_t*)&g_last_cudaError, sizeof(cudaError_t));
}

void
cudaChooseDevice(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);
}

void
cudaSetDevice(ThreadContext *tc, gpusyscall_t *call_params)
{
    GPUSyscallHelper helper(tc, call_params);

    int sim_device = *((int*)helper.getParam(0));
    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: cudaSetDevice(device = %d)\n", sim_device);
    if (sim_device <= CudaGPU::getNumCudaDevices()) {
        g_active_device = sim_device;
        g_last_cudaError = cudaSuccess;
    } else {
        g_last_cudaError = cudaErrorInvalidDevice;
    }
    helper.setReturn((uint8_t*)&g_last_cudaError, sizeof(cudaError_t));
}

void
cudaGetDevice(ThreadContext *tc, gpusyscall_t *call_params)
{
    GPUSyscallHelper helper(tc, call_params);

    Addr sim_device = *((Addr*)helper.getParam(0, true));
    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: cudaGetDevice(device = 0x%x)\n", sim_device);
    if (g_active_device <= CudaGPU::getNumCudaDevices()) {
        helper.writeBlob(sim_device, (uint8_t*)&g_active_device, sizeof(int));
        g_last_cudaError = cudaSuccess;
    } else {
        g_last_cudaError = cudaErrorInvalidDevice;
    }
    helper.setReturn((uint8_t*)&g_last_cudaError, sizeof(cudaError_t));
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
*******************************************************************************/

// __host__ cudaError_t CUDARTAPI cudaBindTexture(size_t *offset, const struct textureReference *texref, const void *devPtr, const struct cudaChannelFormatDesc *desc, size_t size __dv(UINT_MAX))
void
cudaBindTexture(ThreadContext *tc, gpusyscall_t *call_params)
{
    GPUSyscallHelper helper(tc, call_params);
    cuda_not_implemented(__my_func__,__LINE__);
}

void
cudaBindTextureToArray(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);
}

void
cudaUnbindTexture(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);
}

void
cudaGetTextureAlignmentOffset(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);
}

void
cudaGetTextureReference(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
*******************************************************************************/

void
cudaGetChannelDesc(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);
}

void
cudaCreateChannelDesc(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
*******************************************************************************/

void
cudaGetLastError(ThreadContext *tc, gpusyscall_t *call_params)
{
    GPUSyscallHelper helper(tc, call_params);
    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: cudaGetLastError()\n");
    helper.setReturn((uint8_t*)&g_last_cudaError, sizeof(cudaError_t));
}

void
cudaGetErrorString(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
*******************************************************************************/




void
cudaConfigureCall(ThreadContext *tc, gpusyscall_t *call_params)
{
    GPUSyscallHelper helper(tc, call_params);

    dim3 sim_gridDim = *((dim3*)helper.getParam(0));
    dim3 sim_blockDim = *((dim3*)helper.getParam(1));
    size_t sim_sharedMem = *((size_t*)helper.getParam(2));
    cudaStream_t sim_stream = *((cudaStream_t*)helper.getParam(3));
    
    if (sim_stream) {
        //panic("gem5-fusion doesn't currently support CUDA streams");
        DPRINTF(GPUSyscalls, "gem5 GPU Syscall: Using non-default stream!\n");
    }

    //assert(!sim_stream); // We do not currently support CUDA streams
    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: cudaConfigureCall(gridDim = (%u,%u,%u), blockDim = (%u,%u,%u), sharedMem = %u, stream = %x)\n", sim_gridDim.x, sim_gridDim.y, sim_gridDim.z, sim_blockDim.x, sim_blockDim.y, sim_blockDim.z, sim_sharedMem, sim_stream);

    g_cuda_launch_stack.push_back(kernel_config(sim_gridDim, sim_blockDim, sim_sharedMem, sim_stream));
    g_last_cudaError = cudaSuccess;
}

void
cudaSetupArgument(ThreadContext *tc, gpusyscall_t *call_params){
    GPUSyscallHelper helper(tc, call_params);

    Addr sim_arg = *((Addr*)helper.getParam(0, true));
    size_t sim_size = *((size_t*)helper.getParam(1));
    size_t sim_offset = *((size_t*)helper.getParam(2));
    const void* arg = new uint8_t[sim_size];
    helper.readBlob(sim_arg, (uint8_t*)arg, sim_size);
    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: cudaSetupArgument(arg = %x, size = %d, offset = %d)\n", sim_arg, sim_size, sim_offset);

    assert(!g_cuda_launch_stack.empty());
    kernel_config &config = g_cuda_launch_stack.back();
    config.set_arg(arg, sim_size, sim_offset);

    g_last_cudaError = cudaSuccess;
}


void
cudaLaunch(ThreadContext *tc, gpusyscall_t *call_params)
{
    GPUSyscallHelper helper(tc, call_params);

    Addr sim_hostFun = *((Addr*)helper.getParam(0, true));
    new_addr_type simParamMem = *((new_addr_type*)helper.getParam(1, true));

    CudaGPU *cudaGPU = CudaGPU::getCudaGPU(g_active_device);
    char *mode = getenv("PTX_SIM_MODE_FUNC");
    if (mode)
        sscanf(mode,"%u", &g_ptx_sim_mode);
    assert(!g_cuda_launch_stack.empty());
    kernel_config config = g_cuda_launch_stack.back();
    struct CUstream_st *stream = config.get_stream();

    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: cudaLaunch(hostFun* = %x)\n", sim_hostFun);
    /*
    kernel_info_t *grid = gpgpu_cuda_ptx_sim_init_grid(config.get_args(), config.grid_dim(), config.block_dim(), 
                                                        cudaGPU->get_kernel((const char*)sim_hostFun));
    */
    kernel_info_t* grid = GPGPUCudaPTXSimInitGrid(config.get_args(), simParamMem, config.grid_dim(), config.block_dim(), 
                                                   cudaGPU->get_kernel((const char*)sim_hostFun), helper, false);

    grid->setParamMem((new_addr_type)simParamMem);    
    grid->set_inst_base_vaddr(cudaGPU->getInstBaseVaddr());
    std::string kname = grid->name();

    if (kname.find("memcGetKernel") != std::string::npos) {
        grid->setEventKernel(2); // HACK: Specify that this Memc CUDA launched kernel is an event kernel
    }

    stream_operation op(grid, g_ptx_sim_mode, stream);
    op.setThreadContext(tc);
    g_stream_manager->push(op);
    g_cuda_launch_stack.pop_back();
    g_last_cudaError = cudaSuccess;
}

size_t getMaxThreadsPerBlock(struct cudaFuncAttributes attr) {
    CudaGPU *cudaGPU = CudaGPU::getCudaGPU(g_active_device);
    CudaGPU::CudaDeviceProperties *prop;

    prop = cudaGPU->getDeviceProperties();

    size_t max = prop->maxThreadsPerBlock;

    if ((prop->regsPerBlock / attr.numRegs) < max) {
        max = prop->regsPerBlock / attr.numRegs;
    }

    return max;
}

void
cudaFuncGetAttributes(ThreadContext *tc, gpusyscall_t *call_params)
{
    GPUSyscallHelper helper(tc, call_params);

    Addr sim_attr = *((Addr*)helper.getParam(0, true));
    Addr sim_hostFun = *((Addr*)helper.getParam(1, true));

    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: cudaFuncGetAttributes(attr* = %x, hostFun* = %x)\n", sim_attr, sim_hostFun);

    CudaGPU *cudaGPU = CudaGPU::getCudaGPU(g_active_device);
    function_info *entry = cudaGPU->get_kernel((const char*)sim_hostFun);

    if (entry) {
        const struct gpgpu_ptx_sim_kernel_info *kinfo = entry->get_kernel_info();
        cudaFuncAttributes attr;
        attr.sharedSizeBytes = kinfo->smem;
        attr.constSizeBytes  = kinfo->cmem;
        attr.localSizeBytes  = kinfo->lmem;
        attr.numRegs         = kinfo->regs;
        attr.maxThreadsPerBlock = getMaxThreadsPerBlock(attr);
        attr.ptxVersion      = kinfo->ptx_version;
        attr.binaryVersion   = kinfo->sm_target;
        helper.writeBlob(sim_attr, (uint8_t*)&attr, sizeof(cudaFuncAttributes));
        g_last_cudaError = cudaSuccess;
    } else {
        g_last_cudaError = cudaErrorInvalidDeviceFunction;
    }

    helper.setReturn((uint8_t*)&g_last_cudaError, sizeof(cudaError_t));
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
*******************************************************************************/

// __host__ cudaError_t CUDARTAPI cudaStreamCreate(cudaStream_t *stream)
void
cudaStreamCreate(ThreadContext *tc, gpusyscall_t *call_params)
{
    GPUSyscallHelper helper(tc, call_params);
    Addr sim_stream = *((Addr*)helper.getParam(0, true));
    
    cudaStream_t stream = new CUstream_st();
    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: cudaStreamCreate <%x>\n", stream);

    // Add stream to GPGPU-Sim
    g_stream_manager->add_stream(stream);

    // Set stream to return to host code
    helper.writeBlob(sim_stream, (uint8_t*)&stream, sizeof(cudaStream_t));

    g_last_cudaError = cudaSuccess;
    helper.setReturn((uint8_t*)&g_last_cudaError, sizeof(cudaError_t));
}

// __host__ cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t stream)
void
cudaStreamDestroy(ThreadContext *tc, gpusyscall_t *call_params)
{
    GPUSyscallHelper helper(tc, call_params);
    CUstream_st* sim_stream = *((CUstream_st**)helper.getParam(0, true));
    
    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: cudaStreamDestroy <%x>\n", sim_stream);
    assert( g_stream_manager->streamEmpty(sim_stream) );
    g_stream_manager->destroy_stream((cudaStream_t)sim_stream);

    g_last_cudaError = cudaSuccess;
    helper.setReturn((uint8_t*)&g_last_cudaError, sizeof(cudaError_t));
}

// __host__ cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream)
void
cudaStreamSynchronize(ThreadContext *tc, gpusyscall_t *call_params)
{
    GPUSyscallHelper helper(tc, call_params);
    cudaStream_t stream = *((cudaStream_t*)helper.getParam(0, true));

    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: cudaStreamSynchronize <%x>\n", stream);
    bool suspend = false;

    if( stream == NULL ) {
        g_last_cudaError = cudaErrorInvalidResourceHandle; // Doesn't really do anything right now
    } else {
        CudaGPU *cudaGPU = CudaGPU::getCudaGPU(g_active_device);
        suspend = cudaGPU->needsToBlock(stream);
        g_last_cudaError = cudaSuccess;
        // Legacy GPGPU-Sim. This will incorrectly stall the CPU
        // stream->synchronize();
    }
    
    helper.setReturn((uint8_t*)&suspend, sizeof(bool));
}

// __host__ cudaError_t CUDARTAPI cudaStreamQuery(cudaStream_t stream)
void
cudaStreamQuery(ThreadContext *tc, gpusyscall_t *call_params)
{
    GPUSyscallHelper helper(tc, call_params);
    cudaStream_t stream = ((cudaStream_t)helper.getParam(0, true));

    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: cudaStreamSynchronize <%x>\n", stream);

    if( stream == NULL ) {
        g_last_cudaError = cudaErrorInvalidResourceHandle;
    } else {
        g_last_cudaError = stream->empty() ? cudaSuccess : cudaErrorNotReady;
    }

    helper.setReturn((uint8_t*)&g_last_cudaError, sizeof(cudaError_t));
}

void cudaDeviceSynchronize(ThreadContext* tc, gpusyscall_t* call_params)
{
    GPUSyscallHelper helper(tc, call_params);

    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: cudaDeviceSynchronize\n");
    bool suspend = false;

    CudaGPU *cudaGPU = CudaGPU::getCudaGPU(g_active_device);
    suspend = cudaGPU->busy();
    g_last_cudaError = cudaSuccess;
    
    helper.setReturn((uint8_t*)&suspend, sizeof(bool));
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
*******************************************************************************/

// __host__ cudaError_t CUDARTAPI cudaEventCreate(cudaEvent_t *event)
void
cudaEventCreate(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);
}

void
cudaEventRecord(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);
}

void
cudaEventQuery(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);
}

void
cudaEventSynchronize(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);
}

void
cudaEventDestroy(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);
}

void
cudaEventElapsedTime(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);
}


/*******************************************************************************
*                                                                              *
*                                   EDGE                                       *
*                                                                              *
*******************************************************************************/
// Need to keep consistent with memcached stuff
#define MAX_KEY_SIZE 32
#define MAX_VAL_SIZE 8
#define BUFFER_SIZE 128

typedef struct SetRequest {
    uint16_t _keyLength;
    uint16_t _valueLength;
    uint8_t _key[MAX_KEY_SIZE];
    uint8_t _value[MAX_VAL_SIZE];
} SetRequest;

typedef struct GpuGetPkt {
    uint8_t _pkt[BUFFER_SIZE];
} GpuGetPkt;


#include <netinet/in_systm.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <netinet/ip6.h>
//#include <net/ethernet.h>     /* the L2 protocols */
#include <sys/time.h>
#include <sys/mman.h>
#include <time.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/udp.h>


int in_cksum(unsigned char *buf, unsigned nbytes, int sum) 
{
  uint i;

  /* Checksum all the pairs of bytes first... */
  for (i = 0; i < (nbytes & ~1U); i += 2) {
    sum += (u_int16_t) ntohs(*((u_int16_t *)(buf + i)));
    /* Add carry. */
    if(sum > 0xFFFF)
      sum -= 0xFFFF;
  }

  /* If there's a single byte left over, checksum it, too.   Network
     byte order is big-endian, so the remaining byte is the high byte. */
  if(i < nbytes) {
    sum += buf [i] << 8;
    /* Add carry. */
    if(sum > 0xFFFF)
      sum -= 0xFFFF;
  }

  return sum;
}

static u_int32_t wrapsum (u_int32_t sum) 
{
  sum = ~sum & 0xFFFF;
  return htons(sum);
}

#define ETH_ALEN 6
//typedef struct ether_header {
//    u_int8_t ether_dhost[ETH_ALEN];
//    u_int8_t ether_shost[ETH_ALEN];
//    u_int16_t ether_type;
//} ether_header;


void setPktHdr(size_t *data){
    int res = 0;

    // FIXME
    int pkt_size = 72; // 42 byte header + 12 byte payload
    u_int32_t src_ip = 0xC0A80002 /* from 192.168.0.2 */;
    u_int32_t dst_ip =  0xC0A80104 /* 192.168.1.4 */;

    struct ether_header *eh = (struct ether_header *)data;
    struct iphdr *iph = (struct iphdr *)((size_t)data + sizeof(struct ether_header));
    struct udphdr *uh = (struct udphdr *)((size_t)data + sizeof(struct ether_header) + sizeof(struct iphdr));

    // Set packet buffer contents
    // Ether
    eh->ether_type = htons(0x0800);
    eh->ether_shost[0] = 0x68;
    eh->ether_shost[1] = 0x05;
    eh->ether_shost[2] = 0xCA;
    eh->ether_shost[3] = 0x13;
    eh->ether_shost[4] = 0xCE;
    eh->ether_shost[5] = 0x79;
    eh->ether_dhost[0] = 0x68;
    eh->ether_dhost[1] = 0x05;
    eh->ether_dhost[2] = 0xCA;
    eh->ether_dhost[3] = 0x1B;
    eh->ether_dhost[4] = 0x1E;
    eh->ether_dhost[5] = 0x66;

    // IP
    iph->ihl = 5;
    iph->version = 4;
    iph->tos = 0;
    iph->tot_len = htons(pkt_size - sizeof(struct ether_header));
    iph->id = htons(9930);
    iph->ttl = 64;
    iph->frag_off = htons(0);
    iph->protocol = IPPROTO_UDP;
    iph->daddr = htonl(dst_ip);
    iph->saddr = htonl(src_ip);
    iph->check = 0;
    iph->check = wrapsum(in_cksum((unsigned char *)iph, sizeof(struct iphdr), 0));

    // UDP
    uh->source = htons(9191);
    uh->dest = htons(9960);
    uh->len = htons(pkt_size - sizeof(struct ether_header) - sizeof(struct iphdr));
    uh->check = 0; /* It must be 0 to compute the checksum */

    return;

}

int readSetRequest(FILE *fp, unsigned char *req, int *num_req){
    int num=0;
    unsigned char *t_req = req;
    unsigned int fint;
    unsigned char fchar;
    int ret = 0;
    int get_hdr_cnt = 0;

    do{
        ret = fscanf(fp, "%02x", &fint);
        if(ret == EOF) return ret;
    
        fchar = (unsigned char)fint;
        if( (fchar != '\r') && (fchar != '\n') ){
            *t_req++ = fchar;
            num++;
        }

    }while( (fchar != '\n') );
   
    *num_req = num;
   
    return 0;
}

unsigned initSetRequests(const char* filename, SetRequest** setReqs, unsigned nSetReqs)
{
    int numSetReqs = 0;
    std::string getReqLine;
    FILE* setFile = fopen(filename, "r");
    if( !setFile ) {
        DPRINTF(GPUSyscalls, "ERROR opening Get trace file");
        abort();
    }

    // Calculate the number of requests
    char ch=0;
    while(!feof( setFile )){
        ch = fgetc( setFile );
        if(ch == '\n') numSetReqs++;

        if( numSetReqs > nSetReqs )
            break;
    }
    rewind(setFile);

    numSetReqs = (numSetReqs < nSetReqs) ? numSetReqs : nSetReqs;
    DPRINTF(GPUSyscalls, "numSetReqs: %d. Allocating %d Bytes\n", numSetReqs, numSetReqs*sizeof(SetRequest));
    SetRequest* sr = (SetRequest*)malloc(numSetReqs*sizeof(SetRequest));

    uint8_t val = 0;
    int sz = 0;
    for( unsigned i=0; i<numSetReqs; ++i ) {
        uint8_t* key = sr[i]._key;

        if( readSetRequest(setFile, (unsigned char *)key, &sz) != -1 ) {
            sr[i]._keyLength = sz;
        } else {
            break;
        }

        for( unsigned j=0; j<MAX_VAL_SIZE; ++j ) {
            sr[i]._value[j] = val++;
        }
    
        sr[i]._valueLength = MAX_VAL_SIZE;
    }
    DPRINTF(GPUSyscalls, "Configure %u/%u SET requests... \n", numSetReqs, numSetReqs);

    *setReqs = sr;
    return numSetReqs;
}

int readGetReq(FILE *fp, unsigned char *req, int *num_req){
    int num=0;
    unsigned char *t_req = req;
    unsigned int fint;
    unsigned char fchar;
    int ret = 0;
    int get_hdr_cnt = 0;

    do{
        ret = fscanf(fp, "%02x", &fint);
        if(ret == EOF) return ret;
        fchar = (unsigned char)fint;

        // Always has an 8 Byte header which may contain 0x0d or 0x0a... just ignore
        if((get_hdr_cnt++ < 7) || (fchar != '\r') || (fchar != '\n')){
            *t_req++ = fchar;
            num++;
        }
    }while((fchar != '\n') || (get_hdr_cnt++ < 7));

    *t_req++ = '\r';
    *t_req++ = '\n';
    *num_req = num;

    return 0;
}

unsigned initGetRequests(const char* filename, GpuGetPkt** getReqs, unsigned nGetReqs, unsigned getBatchSize)
{
    std::string getReqLine;
    unsigned numGetReqs = 0;
    unsigned char *pkt = NULL;

    unsigned bufferAllocSize = 0;

    FILE* getFile = fopen(filename, "r");
    if( !getFile ) {
        DPRINTF(GPUSyscalls, "ERROR opening Get trace file");
        abort();
    }

    // Calculate the number of requests
    DPRINTF(GPUSyscalls, "Calculating GET request file size (up to %d)\n", nGetReqs);
    
    char ch=0;
    while( !feof(getFile) ) {
        ch = fgetc(getFile);
        if(ch == '\n') numGetReqs++;
        if( numGetReqs >= nGetReqs ) break;
    }
    rewind(getFile);

    numGetReqs = (numGetReqs < nGetReqs) ? numGetReqs : nGetReqs;

    // If numGetReqs not a multiple of the batch size, round down
    if( numGetReqs % getBatchSize ) {
        numGetReqs -= (numGetReqs % getBatchSize);
    }
    
    assert( numGetReqs % getBatchSize == 0 );
    bufferAllocSize = numGetReqs * sizeof(GpuGetPkt);
    DPRINTF(GPUSyscalls, "Num GET requests = %d (%d bytes)\n", numGetReqs, bufferAllocSize);
    
    // Allocate memory for packets from the input file (getTraceFilename)
    *getReqs = (GpuGetPkt *)malloc(bufferAllocSize);

    int sz = 0;
    for(unsigned i=0; i<numGetReqs; ++i){
        pkt = (*getReqs)[i]._pkt;

        unsigned char* pkt_start = pkt;
        setPktHdr((size_t *)pkt);
        pkt += 42;

        if(readGetReq(getFile, (unsigned char *)pkt, &sz) != -1){
            // Got something
        }else{
            // EOF
            break;
        }
    }

    // Set the starting batch pointer
    DPRINTF(GPUSyscalls, "Complete! Created a trace with %d elements for a batch size of %d (%d batches)\n", 
            numGetReqs, getBatchSize, (numGetReqs / getBatchSize));

    return numGetReqs;
}

//===================================MARIA FOR IP FORWARD TRICK============================//

int g_port_mask = IPv4_XIA_R2_PORT_MASK;
struct rte_lpm *ipv4_init()
{
    int i, j;
    uint64_t seed = 0xdeadbeef;

    int num_active_ports = ipv4_bitcount(g_port_mask);
    int *port_arr = ipv4_get_active_ports(g_port_mask);

    struct rte_lpm *lpm = rte_lpm_create(0, IPV4_MAX_RULES);

    /**< Read the prefixes from a prefixes file */
    int num_prefixes;
    struct ipv4_prefix *prefix_arr = ipv4_read_prefixes(IPV4_PREFIX_FILE, &num_prefixes);
    

    for(i = 0; i < num_prefixes; i ++) {
        uint32_t prefix_ip = 0;

        for(j = 0; j < IPV4_ADDR_LEN; j ++) {
            prefix_ip += (prefix_arr[i].bytes[j] << (8 * (3 - j)));
        }

        int dst_port = port_arr[fastrand(&seed) % num_active_ports];

        int add_status = rte_lpm_add(lpm, prefix_ip, prefix_arr[i].depth, dst_port);
        if(add_status < 0) {
            printf("ipv4: Failed to add IPv4 prefix %d. Status = %d\n",
                i, add_status);
            exit(-1);
        }

        if(i % 20000 == 0) {
            printf("ipv4: Added %d IPs\n", i);
        }
    }

    //XXX: HACK - Failed lookups should choose a random port. This hack overdoes it and directs *all* lookups to a random ports.
    for (unsigned i = 0; i < RTE_LPM_TBL24_NUM_ENTRIES; i ++) {
    	uint16_t *tbl24_entry = (uint16_t *) &(lpm->tbl24[i]);
        //If this entry does not point to a tbl8, randomize it. 
        if((*tbl24_entry & RTE_LPM_VALID_EXT_ENTRY_BITMASK) != RTE_LPM_VALID_EXT_ENTRY_BITMASK) {
        	*tbl24_entry = i & 3;
       	}
    }    
    return lpm;
}

struct rte_lpm6 *ipv6_init(int portmask,
    struct ipv6_prefix *prefix_arr, int add_prefixes)
{
    int i;

    /**< Get random prefixes */
    int num_prefixes = IPV6_NUM_RAND_PREFIXES;
    prefix_arr = ipv6_gen_rand_prefixes(num_prefixes, portmask);

    if(add_prefixes == 1) {
        struct rte_lpm6_config ipv6_config;
        ipv6_config.max_rules = 100000;
        ipv6_config.number_tbl8s = IPV6_NUM_TBL8;
        assert(num_prefixes < (int) ipv6_config.max_rules);

        struct rte_lpm6 *lpm = rte_lpm6_create(0, &ipv6_config);

        for(i = 0; i < num_prefixes; i ++) {
            /**< Add this prefix to LPM6 */
            struct ipv6_prefix cur_prfx = (prefix_arr)[i];

            int add_status = rte_lpm6_add(lpm,
                cur_prfx.bytes, cur_prfx.depth, cur_prfx.dst_port);

            if(add_status < 0) {
                printf("ipv6: Failed to add IPv6 prefix %d. Status = %d\n",
                    i, add_status);
                exit(-1);
            }

            if(i % 1000 == 0) {
                printf("ipv6: Added prefixes = %d, total = %d\n",
                    i, num_prefixes);
            }
        }

        printf("\tipv6: Done inserting prefixes\n");
        return lpm;
    } else {
        return NULL;
    }
}


//===================================MARIA FOR IP FORWARD TRICK============================//


void __edgeExtra(ThreadContext* tc, gpusyscall_t* call_params)
{
    GPUSyscallHelper helper(tc, call_params);
    int op = *((int*)helper.getParam(0));
    
    cudaError_t ret = cudaSuccess;
    if( op == 6 ) {
        assert( g_stream_manager );
        if( g_stream_manager->streamZeroHasKernelLaunch() )
            ret = cudaErrorNotReady; 
        g_last_cudaError = ret;
        helper.setReturn((uint8_t*)&ret, sizeof(cudaError_t));
        return;
    }

    Addr userMem = *((Addr*)helper.getParam(1, true));
    size_t nReqs = *((size_t*)helper.getParam(2));
    size_t batchSize = *((size_t*)helper.getParam(3));
    Addr simFilenamePtr = *((Addr*)helper.getParam(4, true));
    size_t simFilenameLen = *((size_t*)helper.getParam(5));

   	char filename[256];
   	helper.readBlob(simFilenamePtr, (uint8_t*)filename, simFilenameLen);
   	filename[simFilenameLen] = '\0';

   	DPRINTF(GPUSyscalls, "Performing EDGE Extra op: %d, Addr = %p, nReqs = %ld, batchSize = %ld, filename = %s\n", 
       	    op, userMem, nReqs, batchSize, filename);

   	SetRequest* setReqs;
   	GpuGetPkt* getReqs;
    host_graph* h_graph=(host_graph*) malloc(sizeof(host_graph)) ;
    Points* hp = (Points*)malloc(sizeof(Points));
    
    float* dest = (float*)malloc( nReqs*batchSize*sizeof(float) );
    printf("edge extra : %d and %d and %d \n",nReqs, batchSize, nReqs*batchSize );
   	struct rte_lpm *lpm;
    
   	switch( op ) {
    	case 0:
           	nReqs = initSetRequests(filename, &setReqs, nReqs);
            helper.writeBlob(userMem, (uint8_t*)setReqs, nReqs*sizeof(SetRequest), false);
            break;
        case 1:
            nReqs = initGetRequests(filename, &getReqs, nReqs, batchSize);
            helper.writeBlob(userMem, (uint8_t*)getReqs, nReqs*sizeof(GpuGetPkt), false);
            break;
        case 2: 
    	    lpm = ipv4_init();
        	printf("Performing EDGE Extra op: %d, Addr = %p\n", op, userMem);
        	helper.writeBlob(userMem, (uint8_t*)lpm, nReqs*sizeof(struct rte_lpm), false);
        	break;
        case 3: 
            init_graph_from_file(filename, h_graph);
            printf("Performing EDGE Extra op: %d, Addr = %p\n", op, userMem);
            helper.writeBlob(userMem, (uint8_t*)h_graph, 1*sizeof(struct host_graph), false);
            break;
        case 4:
            read_sep( dest, nReqs, batchSize, nReqs ) ;
            printf("Performing EDGE Extra op: %d, Addr = %p\n", op, userMem);
            helper.writeBlob(userMem, (uint8_t*)dest, nReqs*batchSize*sizeof(float), false);
            break;
        case 5:
            helper.readBlob(userMem, (uint8_t*)hp, sizeof(Points));
            pspeedy( hp  ) ;
            printf("Performing EDGE Extra op: %d, Addr = %p\n", op, userMem);
            helper.writeBlob(userMem, (uint8_t*)hp, sizeof(Points), false);
            break;

        default:
            panic("EDGE: Unknown __edgeExtra op type: %d\n", op);
    }

    g_last_cudaError = ret;
    helper.setReturn((uint8_t*)&ret, sizeof(int));
}

void __edgeExtraipv6(ThreadContext* tc, gpusyscall_t* call_params)
{
    GPUSyscallHelper helper(tc, call_params);
    int op = *((int*)helper.getParam(0));
    
    cudaError_t ret = cudaSuccess;
    if( op == 3 ) {
        assert( g_stream_manager );
        if( g_stream_manager->streamZeroHasKernelLaunch() )
            ret = cudaErrorNotReady; 
        g_last_cudaError = ret;
        helper.setReturn((uint8_t*)&ret, sizeof(cudaError_t));
        return;
    }
    //call_params.arg_lengths[2] = sizeof(int); //portmask
    //call_params.arg_lengths[3] = sizeof(void**); /// pefix_arr
    //call_params.arg_lengths[4] = sizeof(int); //add_prefix
    //call_params.arg_lengths[5] = sizeof(int); // n_requests_per_batch
    //call_params.arg_lengths[6] = sizeof(int); // n_batches

    Addr userMem = *((Addr*)helper.getParam(1, true));
    int portmask = *((int*)helper.getParam(2));
    Addr pefix_arr_user = *((Addr*)helper.getParam(3, true));
    int add_prefix = *((int*)helper.getParam(4));
    int batchSize = *((int*)helper.getParam(5));
    int nReqs = *((int*)helper.getParam(6));

    struct ipv6_prefix *prefix_arr;
    int num_prefixes = IPV6_NUM_RAND_PREFIXES;
    int prefix_mem_size = num_prefixes * sizeof(struct ipv6_prefix);
    printf("%d and %d and %d \n", num_prefixes,sizeof(struct ipv6_prefix), prefix_mem_size );
    printf("%x and %x and %x \n ", pefix_arr_user,prefix_arr,  prefix_mem_size);

    DPRINTF(GPUSyscalls, "Performing EDGE Extra op: %d, Addr = %p, nReqs = %ld, batchSize = %ld \n", 
            op, userMem, nReqs, batchSize);

    //struct rte_lpm6 *ipv6_init(int portmask, struct ipv6_prefix **prefix_arr, int add_prefixes);

    struct rte_lpm6 *lpm;
    
    switch( op ) {
        case 1: 
            lpm = ipv6_init(portmask,prefix_arr, add_prefix);
            printf("Performing EDGE Extra op: %d, Addr = %p Addr = %p '\n", op, userMem, pefix_arr_user);
            printf("%x and %x\n",&prefix_arr, prefix_arr  );
            helper.writeBlob(userMem, (uint8_t*)lpm, nReqs*sizeof(struct rte_lpm6), false);
            helper.writeBlob(pefix_arr_user, (uint8_t*)prefix_arr, prefix_mem_size, false);
            break;
        default:
            panic("EDGE: Unknown __edgeExtra op type: %d\n", op);
    }

    g_last_cudaError = ret;
    helper.setReturn((uint8_t*)&ret, sizeof(int));
}

// Sets an internal GPU timer to triger an event every N cycles, if the next pending event is valid. 
void __edgeScheduleEventTimer(ThreadContext* tc, gpusyscall_t* call_params)
{
    GPUSyscallHelper helper(tc, call_params);
    int simEventID = *((int*)helper.getParam(0));
    unsigned long long simN = *((unsigned long long*)helper.getParam(1));
    //unsigned long long maxEventsNum = *((unsigned long long*)helper.getParam(2));
    cudaError_t ret;

    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: __cudaScheduleEventTimer(), tc = %x, eventID = %d, N = %d\n", 
            tc, simEventID, simN);
    
    if( !CudaGPU::getCudaGPU(g_active_device)->getTheGPU()->EventManager()->scheduleGPUTimerEvent(simEventID, simN) ) {
        DPRINTF(GPUSyscalls, "Error: Invalid eventID = %d\n", simEventID);
        ret = cudaErrorInvalidValue;
    } else {
        ret = cudaSuccess;
    }
    
    g_last_cudaError = ret;
    helper.setReturn((uint8_t*)&ret, sizeof(int));
}

void __edgeScheduleEventTimerBatch(ThreadContext* tc, gpusyscall_t* call_params)
{
    printf("MARIA DEBUG __edgeScheduleEventTimerBatch is called\n");

    GPUSyscallHelper helper(tc, call_params);
    int simEventID = *((int*)helper.getParam(0));
    unsigned long long simNouter = *((unsigned long long*)helper.getParam(1));
    unsigned long long batchN = *((unsigned long long*)helper.getParam(2));
    unsigned long long simNinner = *((unsigned long long*)helper.getParam(3));
    //unsigned long long maxEventsNum = *((unsigned long long*)helper.getParam(2));
    cudaError_t ret;

    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: __cudaScheduleEventTimerBatch(), tc = %x, eventID = %d, Nouter = %d, Batch = %d, Ninner = %d \n", 
            tc, simEventID, simNouter, batchN, simNinner);
    
    if( !CudaGPU::getCudaGPU(g_active_device)->getTheGPU()->EventManager()->scheduleGPUTimerBatchEvent(simEventID, simNouter, batchN, simNinner) ) {
        DPRINTF(GPUSyscalls, "Error: Invalid eventID = %d\n", simEventID);
        ret = cudaErrorInvalidValue;
    } else {
        ret = cudaSuccess;
    }
    
    g_last_cudaError = ret;
    helper.setReturn((uint8_t*)&ret, sizeof(int));
}



/**
 * A map of kernel_info_t* to the allocated parameter memory. Keeps track of which 
 * parameter memory belongs to which kernel so we can free it up at a later time. 
 */
class KernelParamGCList : public std::vector<new_addr_type> {};
KernelParamGCList gKernelParamGCList;

void MarkParamMemForGC(new_addr_type paramMem) 
{
    gKernelParamGCList.push_back(paramMem); // Will be GC'd sometime later from the host side
}

// Calculate the total size in bytes of the arguments for the current kernel
size_t GetArgSize(kernel_config& config)
{
    size_t size = 0;
    gpgpu_ptx_sim_arg_list_t args = config.get_args();
    for (gpgpu_ptx_sim_arg_list_t::iterator a = args.begin(); a != args.end(); ++a) {
        size += (*a).m_nbytes;
    }
    return size;
}






void __cudaGetFreeArgMem(ThreadContext* tc, gpusyscall_t* call_params)
{
    GPUSyscallHelper helper(tc, call_params);

    new_addr_type paramMem = 0;
    if( !gKernelParamGCList.empty() ) {
        paramMem = gKernelParamGCList.front();
        gKernelParamGCList.erase(gKernelParamGCList.begin());
    }

    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: __cudaGetFreeArgMem(), tc = %x, ArgMem = %ld\n", tc, paramMem);
    helper.setReturn((uint8_t*)&paramMem, sizeof(new_addr_type));
}

// Return the total argument size in bytes for the current kernel
void __cudaGetArgSize(ThreadContext* tc, gpusyscall_t* call_params)
{
    GPUSyscallHelper helper(tc, call_params);

    assert(!g_cuda_launch_stack.empty());
    kernel_config config = g_cuda_launch_stack.back();
    size_t ret = GetArgSize(config);

    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: __cudaGetArgSize(), tc = %x, ArgSize = %ld\n", tc, ret);
    helper.setReturn((uint8_t*)&ret, sizeof(size_t));
}


void __cudaEDGEInit(ThreadContext* tc, gpusyscall_t* call_params)
{
    GPUSyscallHelper helper(tc, call_params);
    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: __cudaEDGEInit(), tc = %x\n", tc);
    
    CudaGPU::getCudaGPU(g_active_device)->initISRKernel();

    g_last_cudaError = cudaSuccess;
    cudaError_t ret = cudaSuccess;
    helper.setReturn((uint8_t*)&ret, sizeof(cudaError_t));
}

void __cudaSetupEventArgument(ThreadContext *tc, gpusyscall_t *call_params){
    GPUSyscallHelper helper(tc, call_params);

    size_t sim_size = *((size_t*)helper.getParam(0));
    size_t sim_offset = *((size_t*)helper.getParam(1));
    const void* arg = new uint8_t[sim_size];
    
    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: __cudaSetupEventArgument(size = %d, offset = %d)\n", sim_size, sim_offset);

    if( g_cuda_launch_stack.empty() ) {
        DPRINTF(GPUSyscalls, "gem5 GPU Syscall: First call, created event kernel_config\n");
        dim3 tmp;
        g_cuda_launch_stack.push_back(kernel_config(tmp, tmp, 0, NULL));
    }

    kernel_config &config = g_cuda_launch_stack.back();
    config.set_arg(NULL, sim_size, sim_offset);

    g_last_cudaError = cudaSuccess;
}


void __cudaRegisterEvent(ThreadContext* tc, gpusyscall_t* call_params)
{
    GPUSyscallHelper helper(tc, call_params);
    Addr simEventFun  = *((Addr*)helper.getParam(0, true));
    Addr simEventFun2  = *((Addr*)helper.getParam(1, true));
    dim3 sim_gridDim = *((dim3*)helper.getParam(2));
    dim3 sim_blockDim = *((dim3*)helper.getParam(3));
    size_t sim_sharedMem = *((size_t*)helper.getParam(4));

    CudaGPU *cudaGPU = CudaGPU::getCudaGPU(g_active_device);

    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: __cudaRegisterEvent(), tc = %x\n"
        "\tsimEventFun: %p\n\tsimEventFun2: %p\n\tgrid: (%d,%d,%d)\n\tblock: (%d,%d,%d)\n", 
        tc, simEventFun, simEventFun2, sim_gridDim.x, sim_gridDim.y, sim_gridDim.z, 
        sim_blockDim.x, sim_blockDim.y, sim_blockDim.z );

    function_info* entry = cudaGPU->get_kernel((const char*)simEventFun);
    
    if (entry == NULL) 
        panic("GPGPU-Sim PTX: ERROR launching kernel -- no PTX implementation found");

    kernel_info_t *grid = new kernel_info_t( sim_gridDim, sim_blockDim, entry );
    kernel_info_t *grid2 = NULL;
    function_info* entry2 = NULL;
    if (simEventFun2 != NULL) {
        entry2 = cudaGPU->get_kernel((const char*)simEventFun2);
        grid2 = new kernel_info_t( sim_gridDim, sim_blockDim, entry2 );
    }
    assert(grid2);
    int ret = cudaGPU->getTheGPU()->EventManager()->registerGPUEvent(EDGE_USER_EVENT, grid, grid2);

    g_last_cudaError = cudaSuccess;
    helper.setReturn((uint8_t*)&ret, sizeof(int));
}

void __cudaRegisterBarrier(ThreadContext* tc, gpusyscall_t* call_params)
{
    GPUSyscallHelper helper(tc, call_params);
    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: __cudaRegisterBarrier(), tc = %x, eventID = %d\n", tc);

    CudaGPU *cudaGPU = CudaGPU::getCudaGPU(g_active_device);

    int ret = cudaGPU->getTheGPU()->EventManager()->registerGPUEvent(EDGE_RELEASE_BAR);

    g_last_cudaError = cudaSuccess;
    helper.setReturn((uint8_t*)&ret, sizeof(int));
}

void __cudaConfigureEventParam(ThreadContext* tc, gpusyscall_t* call_params)
{
    GPUSyscallHelper helper(tc, call_params);
    int simEventId = *((int*)helper.getParam(0));
    Addr simParamMem = *((Addr*)helper.getParam(1, true));
    size_t simParamSize = *((size_t*)helper.getParam(2));
    size_t simMaxInFlightKernels = *((size_t*)helper.getParam(3));
    bool child_kernel = *((bool*)helper.getParam(4));

    //assert(!child_kernel && (simParamMem!=NULL));

    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: __cudaConfigureEventParam(), tc = %x\n"
            "eventId = %d, paramMem = %p, paramSize = %ld, maxEvents = %ld\n", 
            tc, simEventId, simParamMem, simParamSize, simMaxInFlightKernels);
    
    CudaGPU *cudaGPU = CudaGPU::getCudaGPU(g_active_device);

    kernel_info_t* eventKernel;
    if (!child_kernel) {
        eventKernel = cudaGPU->getTheGPU()->EventManager()->getKernel(simEventId);
    } else {
        eventKernel = cudaGPU->getTheGPU()->EventManager()->getKernel2(simEventId);
    }
    if( eventKernel == NULL )
        panic("GPGPU-Sim Error: Invalid eventId: %d - could not find event kernel\n", simEventId);

    eventKernel->setEventKernel(1); // Specify that this is an event kernel
    eventKernel->set_inst_base_vaddr(cudaGPU->getInstBaseVaddr()); // FIXME

    if (!child_kernel) {
        eventKernel->setParamMem((new_addr_type)simParamMem);
        if( !cudaGPU->getTheGPU()->EventManager()->configureEvent(simEventId, (new_addr_type)simParamMem, simParamSize, simMaxInFlightKernels) )
            panic("GPGPU-Sim Error: Invalid eventId: %d - could not find event kernel\n", simEventId);
    } else {
        eventKernel->setParamMem(cudaGPU->getTheGPU()->EventManager()->getEvent(simEventId)->getParamBaseAddr());
    }

    assert(eventKernel->getParamMem());
    // Should have a simple config with only the parameter sizes/offsets pending
    assert(!g_cuda_launch_stack.empty());
    kernel_config config = g_cuda_launch_stack.back();
    assert(!config.get_stream());
    
    gpgpu_ptx_sim_arg_list_t args = config.get_args();
    unsigned argCount = args.size();
    unsigned argN = 1;

    for (gpgpu_ptx_sim_arg_list_t::iterator a = args.begin(); a != args.end(); a++) {
        eventKernel->entry()->edgeAddParamMetaData(argCount-argN, &(*a));
        argN++;
    }

    eventKernel->entry()->gem5Finalize(); // Finalize the parameter values
    g_ptx_kernel_count++;

    g_last_cudaError = cudaSuccess;

    helper.setReturn((uint8_t*)&simParamMem, sizeof(void*));
}


void __cudaScheduleEvent(ThreadContext* tc, gpusyscall_t* call_params)
{
    GPUSyscallHelper helper(tc, call_params);
    int simEventID = *((int*)helper.getParam(0, false));

    cudaError_t ret;
    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: __cudaScheduleEvent(), tc = %x, eventID = %d\n", tc, simEventID);
    if( !CudaGPU::getCudaGPU(g_active_device)->getTheGPU()->EventManager()->scheduleGPUEvent(simEventID) ) {
        DPRINTF(GPUSyscalls, "Error: Invalid eventID = %d\n", simEventID);
        ret = cudaErrorInvalidValue;
    } else {
        ret = cudaSuccess;
    }
    
    g_last_cudaError = ret;
    helper.setReturn((uint8_t*)&ret, sizeof(int));
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
*******************************************************************************/

void
cudaThreadExit(ThreadContext *tc, gpusyscall_t *call_params)
{
    // This function should clean-up any/all resources associated with the
    // current device in the passed thread context
    cuda_not_implemented(__my_func__,__LINE__);
}

void
cudaThreadSynchronize(ThreadContext *tc, gpusyscall_t *call_params)
{
    GPUSyscallHelper helper(tc, call_params);
    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: cudaThreadSynchronize(), tc = %x\n", tc);
    CudaGPU *cudaGPU = CudaGPU::getCudaGPU(g_active_device);
    bool suspend = cudaGPU->needsToBlock();
    g_last_cudaError = cudaSuccess;
    helper.setReturn((uint8_t*)&suspend, sizeof(bool));
}

void
__cudaSynchronizeThreads(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);
}

void
deleteFatCudaBinary(__cudaFatCudaBinary* fat_cubin) {
    if (fat_cubin->ident) delete[] fat_cubin->ident;
    if (fat_cubin->ptx) {
        // @TODO: This might need to loop... consider splitting out the
        // CUDA binary read into a separate helper class that tracks the
        // number of ptx_entries that are read in
        if (fat_cubin->ptx->gpuProfileName) delete[] fat_cubin->ptx->gpuProfileName;
        if (fat_cubin->ptx->ptx) delete[] fat_cubin->ptx->ptx;
        delete[] fat_cubin->ptx;
    }
    delete fat_cubin;
}

symbol_table* registering_symtab = NULL;
unsigned registering_fat_cubin_handle = 0;
int registering_allocation_size = -1;
Addr registering_allocation_ptr = 0;
Addr registering_local_alloc_ptr = 0;

unsigned
get_global_and_constant_alloc_size(symbol_table* symtab)
{
    unsigned total_bytes = 0;
    symbol_table::iterator iter;
    for (iter = symtab->global_iterator_begin(); iter != symtab->global_iterator_end(); iter++) {
        symbol* global = *iter;
        total_bytes += global->get_size_in_bytes();
    }

    for (iter = symtab->const_iterator_begin(); iter != symtab->const_iterator_end(); iter++) {
        symbol* constant = *iter;
        total_bytes += constant->get_size_in_bytes();
    }

    return total_bytes;
}

unsigned
get_local_alloc_size(CudaGPU *cudaGPU) {
    unsigned cores = cudaGPU->getDeviceProperties()->multiProcessorCount;
    unsigned threads_per_core = cudaGPU->getMaxThreadsPerMultiprocessor();
    // NOTE: Per technical specs Wikipedia: http://en.wikipedia.org/wiki/CUDA
    // For CUDA GPUs with compute capability 1.x, each thread should be able to
    // access up to 16kB of memory, and for compute capability 2.x+, each
    // thread should be able to access up to 512kB of local memory. Since this
    // could blow out the simulator's memory footprint, here we use 8kB per
    // thread as a more reasonable baseline. This may need to be changed if
    // benchmarks trip on the GPGPU-Sim-side panic of too much local memory
    // usage per thread.
    unsigned max_local_mem_per_thread = 8 * 1024;
    return cores * threads_per_core * max_local_mem_per_thread;
}

void
finalize_global_and_constant_setup(ThreadContext *tc, Addr base_addr, symbol_table* symtab)
{
    CudaGPU *cudaGPU = CudaGPU::getCudaGPU(g_active_device);
    Addr curr_addr = base_addr;
    Addr next_addr = 0;
    symbol_table::iterator iter;
    for (iter = symtab->global_iterator_begin(); iter != symtab->global_iterator_end(); iter++) {
        symbol* global = *iter;
        global->set_address(curr_addr);
        cudaGPU->registerDeviceMemory(tc, curr_addr, global->get_size_in_bytes());
        next_addr = curr_addr + global->get_size_in_bytes();
        if (next_addr - base_addr > registering_allocation_size) {
            panic("Didn't allocate enough global+const memory. Bailing!");
        } else {
            DPRINTF(GPUSyscalls, "GPGPU-Sim PTX: Updated symbol \"%s\" to address range 0x%x to 0x%x\n", global->name(), curr_addr, next_addr-1);
        }
        curr_addr = next_addr;
    }

    for (iter = symtab->const_iterator_begin(); iter != symtab->const_iterator_end(); iter++) {
        symbol* constant = *iter;
        constant->set_address(curr_addr);
        cudaGPU->registerDeviceMemory(tc, curr_addr, constant->get_size_in_bytes());
        next_addr = curr_addr + constant->get_size_in_bytes();
        if (next_addr - base_addr > registering_allocation_size) {
            panic("Didn't allocate enough global+const memory. Bailing!");
        } else {
            DPRINTF(GPUSyscalls, "GPGPU-Sim PTX: Updated symbol \"%s\" to address range 0x%x to 0x%x\n", constant->name(), curr_addr, next_addr-1);
        }
        curr_addr = next_addr;
    }
}

unsigned next_fat_bin_handle = 1;
unsigned source_num = 1;

void registerFatBinaryTop(GPUSyscallHelper *helper, Addr sim_fatCubin, size_t sim_binSize)
{
    CudaGPU *cudaGPU = CudaGPU::getCudaGPU(g_active_device);
    gpgpu_t *gpu = cudaGPU->getTheGPU();

    // Get primary arguments
    __cudaFatCudaBinary* fat_cubin = new __cudaFatCudaBinary;

#if THE_ISA == ARM_ISA
    // Size of fat binary in 32-bit simulated system is 64B
    #define FATBIN_PACKAGE_SIZE 64
    // Add 4B to keep last 64-bit pointer math from reading other stack junk
    uint8_t fatbin_package[FATBIN_PACKAGE_SIZE + 4];
    helper->readBlob(sim_fatCubin, fatbin_package, FATBIN_PACKAGE_SIZE);
    fat_cubin->magic = unpackData<unsigned long>(fatbin_package, 0);
    fat_cubin->version = unpackData<unsigned long>(fatbin_package, 4);
    fat_cubin->gpuInfoVersion = unpackData<unsigned long>(fatbin_package, 8);
    fat_cubin->key = unpackPointer<char*>(fatbin_package, 12);
    fat_cubin->ident = unpackPointer<char*>(fatbin_package, 16);
    fat_cubin->usageMode = unpackPointer<char*>(fatbin_package, 20);
    fat_cubin->ptx = unpackPointer<__cudaFatPtxEntry*>(fatbin_package, 24);
    fat_cubin->cubin = unpackPointer<__cudaFatCubinEntry*>(fatbin_package, 28);
    fat_cubin->debug = unpackPointer<__cudaFatDebugEntry*>(fatbin_package, 32);
    fat_cubin->debugInfo = unpackPointer<void*>(fatbin_package, 36);
    fat_cubin->flags = unpackData<unsigned int>(fatbin_package, 40);
    fat_cubin->exported = unpackPointer<__cudaFatSymbol*>(fatbin_package, 44);
    fat_cubin->imported = unpackPointer<__cudaFatSymbol*>(fatbin_package, 48);
    fat_cubin->dependends = unpackPointer<__cudaFatCudaBinaryRec*>(fatbin_package, 52);
    fat_cubin->characteristic = unpackData<unsigned int>(fatbin_package, 56);
    fat_cubin->elf = unpackPointer<__cudaFatElfEntry*>(fatbin_package, 60);
#elif THE_ISA == X86_ISA
    // x86 64-bit, we can just read directly from memory
    helper->readBlob(sim_fatCubin, (uint8_t*)fat_cubin, sizeof(struct __cudaFatCudaBinaryRec));
#else
    #error Currently gem5-gpu is only known to support x86 and ARM
#endif

    if (sim_binSize < 0) {
        panic("Used wrong __cudaRegisterFatBinary call!!! Did you run the sizeHack.py?");
    }

    // Read in the fat PTX entries
    uint8_t* ptx_entries = NULL;
    __cudaFatPtxEntry* ptx_entry_ptr;
    int ptx_count = 0;
    do {
        uint8_t* temp_ptx_entry_buf = new uint8_t[sizeof(__cudaFatPtxEntry) * (ptx_count + 1)];
        if (ptx_entries) {
            memcpy(temp_ptx_entry_buf, ptx_entries, sizeof(__cudaFatPtxEntry) * ptx_count);
        }

#if THE_ISA == ARM_ISA
        // Size of PTX entry in 32-bit simulated system is 8B
        #define PTXENTRY_PACKAGE_SIZE 8
        // Add 4B to keep last 64-bit pointer math from reading other stack junk
        uint8_t ptx_entry_package[PTXENTRY_PACKAGE_SIZE + 4];
        helper->readBlob(
                (Addr)fat_cubin->ptx + ptx_count * PTXENTRY_PACKAGE_SIZE,
                ptx_entry_package, PTXENTRY_PACKAGE_SIZE);

        __cudaFatPtxEntry *temp_ptx_entry_ptr =
                                    (__cudaFatPtxEntry*)temp_ptx_entry_buf;
        temp_ptx_entry_ptr[ptx_count].gpuProfileName =
                                    unpackPointer<char*>(ptx_entry_package, 0);
        temp_ptx_entry_ptr[ptx_count].ptx =
                                    unpackPointer<char*>(ptx_entry_package, 4);
#elif THE_ISA == X86_ISA
        helper->readBlob((Addr)(fat_cubin->ptx + ptx_count),
                temp_ptx_entry_buf + sizeof(__cudaFatPtxEntry) * ptx_count,
                sizeof(__cudaFatPtxEntry));
#endif
        ptx_entry_ptr = (__cudaFatPtxEntry *)temp_ptx_entry_buf + ptx_count;
        if (ptx_entry_ptr->ptx != 0) {
            DPRINTF(GPUSyscalls, "GPGPU-Sim PTX: Found instruction text segment: %x\n", (Addr)ptx_entry_ptr->ptx);
            cudaGPU->registerDeviceInstText(helper->getThreadContext(), (Addr)ptx_entry_ptr->ptx, sim_binSize);
            uint8_t* ptx_code = new uint8_t[sim_binSize];
            helper->readBlob((Addr)ptx_entry_ptr->ptx, ptx_code, sim_binSize);
            uint8_t* gpu_profile = new uint8_t[MAX_STRING_LEN];
            helper->readString((Addr)ptx_entry_ptr->gpuProfileName, gpu_profile, MAX_STRING_LEN);

            ptx_entry_ptr->ptx = (char*)ptx_code;
            ptx_entry_ptr->gpuProfileName = (char*)gpu_profile;
        }
        ptx_count++;
        if (ptx_entries) delete[] ptx_entries;
        ptx_entries = temp_ptx_entry_buf;
    } while(ptx_entry_ptr->gpuProfileName != 0);
    fat_cubin->ptx = (__cudaFatPtxEntry *)ptx_entries;

    // Read ident member
    Addr ident_addr = (Addr)fat_cubin->ident;
    fat_cubin->ident = new char[MAX_STRING_LEN];
    helper->readString(ident_addr, (uint8_t*)fat_cubin->ident, MAX_STRING_LEN);

    //static unsigned next_fat_bin_handle = 1;
    //static unsigned source_num = 1;
    assert(registering_fat_cubin_handle == 0);
    registering_fat_cubin_handle = next_fat_bin_handle++;
    assert(fat_cubin->version >= 3);
    unsigned num_ptx_versions = 0;
    unsigned max_capability = 0;
    unsigned selected_capability = 0;
    bool found = false;
    unsigned forced_max_capability = gpu->get_config().get_forced_max_capability();
    while (fat_cubin->ptx[num_ptx_versions].gpuProfileName != NULL) {
        unsigned capability = 0;
        sscanf(fat_cubin->ptx[num_ptx_versions].gpuProfileName, "compute_%u", &capability);
        DPRINTF(GPUSyscalls, "GPGPU-Sim PTX: __cudaRegisterFatBinary found PTX versions for '%s', capability = %s\n", fat_cubin->ident, fat_cubin->ptx[num_ptx_versions].gpuProfileName);
        if (forced_max_capability) {
            if (capability > max_capability && capability <= forced_max_capability) {
                found = true;
                max_capability = capability;
                selected_capability = num_ptx_versions;
            }
        } else {
            if (capability > max_capability) {
                found = true;
                max_capability = capability;
                selected_capability = num_ptx_versions;
            }
        }
        num_ptx_versions++;
    }
    if (found) {
        DPRINTF(GPUSyscalls, "GPGPU-Sim PTX: Loading PTX for %s, capability = %s\n",
                fat_cubin->ident, fat_cubin->ptx[selected_capability].gpuProfileName);
        const char *ptx = fat_cubin->ptx[selected_capability].ptx;
        if (gpu->get_config().convert_to_ptxplus()) {
            panic("GPGPU-Sim PTXPLUS: gem5 + GPGPU-Sim does not support PTXPLUS!");
        } else {
            assert(registering_symtab == NULL);
            registering_symtab = gpgpu_ptx_sim_load_ptx_from_string(ptx, source_num);
            cudaGPU->add_binary(registering_symtab, registering_fat_cubin_handle);
            gpgpu_ptxinfo_load_from_string(ptx, source_num);
        }
        source_num++;
        assert(registering_allocation_size == -1);
        registering_allocation_size = get_global_and_constant_alloc_size(registering_symtab);
    } else {
        panic("GPGPU-Sim PTX: warning -- did not find an appropriate PTX in cubin");
    }
    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: __cudaRegisterFatBinary needs %d bytes allocated\n", registering_allocation_size);

    deleteFatCudaBinary(fat_cubin);
}

void
__cudaRegisterFatBinary(ThreadContext *tc, gpusyscall_t *call_params)
{
#if (CUDART_VERSION < 2010)
    panic("GPGPU-Sim PTX: ERROR ** this version of GPGPU-Sim requires CUDA 2.1 or higher\n");
#endif

    GPUSyscallHelper helper(tc, call_params);

    // Get CUDA call simulated parameters
    Addr sim_fatCubin = *((Addr*)helper.getParam(0, true));
    int sim_binSize = *((int*)helper.getParam(1));
    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: __cudaRegisterFatBinary(fatCubin* = %x, binSize = %d)\n", sim_fatCubin, sim_binSize);
    CudaGPU *cudaGPU = CudaGPU::getCudaGPU(g_active_device);

    registerFatBinaryTop(&helper, sim_fatCubin, sim_binSize);

    cudaGPU->saveFatBinaryInfoTop(tc->threadId(), registering_fat_cubin_handle, sim_fatCubin, sim_binSize);

    if (!cudaGPU->isManagingGPUMemory()) {
        helper.setReturn((uint8_t*)&registering_allocation_size, sizeof(int));
    } else {
        assert(!registering_allocation_ptr);
        registering_allocation_ptr = cudaGPU->allocateGPUMemory(registering_allocation_size);
        int zero_allocation = 0;
        helper.setReturn((uint8_t*)&zero_allocation, sizeof(int));
    }
}

unsigned int registerFatBinaryBottom(GPUSyscallHelper *helper, Addr sim_alloc_ptr)
{
    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: __cudaRegisterFatBinaryFinalize(alloc_ptr* = 0x%x)\n", sim_alloc_ptr);

    assert(registering_symtab);
    assert(registering_fat_cubin_handle > 0);
    assert(registering_allocation_size >= 0);
    assert(sim_alloc_ptr || registering_allocation_size == 0);

    if (registering_allocation_size > 0) {
        finalize_global_and_constant_setup(helper->getThreadContext(), sim_alloc_ptr, registering_symtab);
    }

    load_static_globals(helper, registering_symtab);
    load_constants(helper, registering_symtab);

    unsigned int handle = registering_fat_cubin_handle;

    registering_symtab = NULL;
    registering_fat_cubin_handle = 0;
    registering_allocation_size = -1;
    registering_allocation_ptr = 0;

    return handle;
}

void
__cudaRegisterFatBinaryFinalize(ThreadContext *tc, gpusyscall_t *call_params)
{
    GPUSyscallHelper helper(tc, call_params);
    Addr sim_alloc_ptr = *((Addr*)helper.getParam(0, true));

    CudaGPU *cudaGPU = CudaGPU::getCudaGPU(g_active_device);

    cudaGPU->saveFatBinaryInfoBottom(sim_alloc_ptr);

    unsigned int handle;
    if (!cudaGPU->isManagingGPUMemory()) {
        cudaGPU->saveFatBinaryInfoBottom(sim_alloc_ptr);
        handle = registerFatBinaryBottom(&helper, sim_alloc_ptr);
    } else {
        assert(!sim_alloc_ptr);
        assert(registering_allocation_ptr || registering_allocation_size == 0);
        cudaGPU->saveFatBinaryInfoBottom(registering_allocation_ptr);
        handle = registerFatBinaryBottom(&helper, registering_allocation_ptr);
    }

    // TODO: If local memory has been allocated and has been mapped by the CPU
    // thread, register the allocation with the GPU for address translation.
    // if (registering_local_alloc_ptr && !cudaGPU->getAccessHostPagetable()) {
    //    cudaGPU->registerDeviceMemory(tc, registering_local_alloc_ptr, get_local_alloc_size(cudaGPU));
    // }


    // EDGE: Okay! So, previously the local memory was never being registered with the GPU, which caused a translation fail in the TLBs. 
    // Now, we allocated local memory (#cores * #threads_per_core * MAX_LOCAL_MEM_PER_THREAD) and map it into the GPU. The only problem
    // is that every single __cudaRegisterFatBinary call ends up trying to do this. If two binaries happen to use local memory, this results
    // in multiple allocations of the same data size, which is supposed to be shared. So, adding a check for if we've already allocated
    // local memory or not in the GPU. 
    DPRINTF(GPUSyscalls, "EDGE: Registering local memory with GPU. Address = %p, size = %llu\n", registering_local_alloc_ptr, get_local_alloc_size(cudaGPU));
    if( registering_local_alloc_ptr && !cudaGPU->localMemoryRegistered() ) {
        cudaGPU->registerDeviceMemory(tc, registering_local_alloc_ptr, get_local_alloc_size(cudaGPU));
        cudaGPU->registerLocalMemory();
    }


    helper.setReturn((uint8_t*)&handle, sizeof(void**), true);
}

void
__cudaCheckAllocateLocal(ThreadContext *tc, gpusyscall_t *call_params)
{
    GPUSyscallHelper helper(tc, call_params);

    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: __cudaCheckAllocateLocal()\n");

    CudaGPU *cudaGPU = CudaGPU::getCudaGPU(g_active_device);

    assert(registering_symtab);
    if (registering_symtab->get_local_next() > 0 && (registering_local_alloc_ptr == NULL)) {
        unsigned long long local_alloc_size = get_local_alloc_size(cudaGPU);
        if (!cudaGPU->isManagingGPUMemory()) {
            DPRINTF(GPUSyscalls, "gem5 GPU Syscall:      CPU must allocate local: %lluB\n", local_alloc_size);
            helper.setReturn((uint8_t*)&local_alloc_size, sizeof(unsigned long long), false);
        } else {
            DPRINTF(GPUSyscalls, "gem5 GPU Syscall:      GPU allocating local...\n");
            registering_local_alloc_ptr = cudaGPU->allocateGPUMemory(local_alloc_size);
            cudaGPU->setLocalBaseVaddr(registering_local_alloc_ptr);
            cudaGPU->registerDeviceMemory(tc, registering_local_alloc_ptr, local_alloc_size);
            unsigned long long zero_allocation = 0;
            helper.setReturn((uint8_t*)&zero_allocation, sizeof(int));
        }
    }
}

void
__cudaSetLocalAllocation(ThreadContext *tc, gpusyscall_t *call_params) {
    GPUSyscallHelper helper(tc, call_params);
    Addr sim_alloc_ptr = *((Addr*)helper.getParam(0, true));

    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: __cudaSetLocalAllocation(alloc_ptr* = 0x%x)\n", sim_alloc_ptr);

    CudaGPU *cudaGPU = CudaGPU::getCudaGPU(g_active_device);

    // Save the local memory base address
    assert(!cudaGPU->isManagingGPUMemory());
    assert(!registering_local_alloc_ptr);
    registering_local_alloc_ptr = sim_alloc_ptr;
    cudaGPU->setLocalBaseVaddr(registering_local_alloc_ptr);

    // TODO: Need to check if using host or GPU page mappings. If the GPU is
    // not able to access the host's pagetable, then the memory pages need to
    // be mapped for the GPU to access them.
    // global: bool registering_signal_local_map = false;
    // if (!cudaGPU->getAccessHostPagetable()) {
    //     registering_signal_local_map = true;
    //     helper.setReturn((uint8_t*)&signal_map_memory, sizeof(bool));
    // }
}

void
__cudaUnregisterFatBinary(ThreadContext *tc, gpusyscall_t *call_params)
{
    // TODO: If running multiple benchmarks in the same simulation, this will
    // need to be updated to print as appropriate
    CudaGPU *cudaGPU = CudaGPU::getCudaGPU(g_active_device);
    cudaGPU->printPTXFileLineStats();
    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: __cudaUnregisterFatBinary() Faked\n");

    registering_local_alloc_ptr = 0;

    // EDGE: Do we need to unregister the local memory here? Possibly... Depends on what Faked actually does.
    // If local memory is freed then yes, because we'll need to allocate it again. Otherwise, no!
}

void
__cudaRegisterFunction(ThreadContext *tc, gpusyscall_t *call_params)
{
    GPUSyscallHelper helper(tc, call_params);

    Addr sim_fatCubinHandle = *((Addr*)helper.getParam(0, true));
    Addr sim_hostFun = *((Addr*)helper.getParam(1, true));
    Addr sim_deviceFun = *((Addr*)helper.getParam(2));
    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: __cudaRegisterFunction(fatCubinHandle** = %x, hostFun* = %x, deviceFun* = %x)\n",
            sim_fatCubinHandle, sim_hostFun, sim_deviceFun);

    // Read device function name from simulated system memory
    char* device_fun = new char[MAX_STRING_LEN];
    CudaGPU *cudaGPU = CudaGPU::getCudaGPU(g_active_device);
    helper.readString(sim_deviceFun, (uint8_t*)device_fun, MAX_STRING_LEN);

    // Register function
    unsigned fat_cubin_handle = (unsigned)(unsigned long long)sim_fatCubinHandle;
    cudaGPU->register_function(fat_cubin_handle, (const char*)sim_hostFun, device_fun);
    cudaGPU->saveFunctionNames(fat_cubin_handle, (const char*)sim_hostFun, device_fun);
    delete[] device_fun;
}

void register_var(Addr sim_deviceAddress, const char* deviceName, int sim_size, int sim_constant, int sim_global, int sim_ext, Addr sim_hostVar)
{
    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: __cudaRegisterVar(fatCubinHandle** = %x, hostVar* = 0x%x, deviceAddress* = 0x%x, deviceName* = %s, ext = %d, size = %d, constant = %d, global = %d)\n",
            /*sim_fatCubinHandle*/ 0, sim_hostVar, sim_deviceAddress,
            deviceName, sim_ext, sim_size, sim_constant, sim_global);

    if (sim_constant && !sim_global && !sim_ext) {
        gpgpu_ptx_sim_register_const_variable((void*)sim_hostVar, deviceName, sim_size);
    } else if (!sim_constant && !sim_global && !sim_ext) {
        gpgpu_ptx_sim_register_global_variable((void*)sim_hostVar, deviceName, sim_size);
    } else if (sim_constant && !sim_global && sim_ext) {
        DPRINTF(GPUSyscalls, "EDGE: Now including extern constant defined variable mapping... seems to work. Need to double check.\n");
        gpgpu_ptx_sim_register_const_variable((void*)sim_hostVar, deviceName, sim_size);
    } else if (!sim_constant && sim_global && sim_ext) {
        DPRINTF(GPUSyscalls, "EDGE: Now including extern global defined variable mapping... seems to work. Need to double check.\n");
        gpgpu_ptx_sim_register_const_variable((void*)sim_hostVar, deviceName, sim_size);
    } else if (!sim_constant && !sim_global && sim_ext) {
        DPRINTF(GPUSyscalls, "EDGE: Now including extern device defined variable mapping... Need to double check.\n");
        gpgpu_ptx_sim_register_const_variable((void*)sim_hostVar, deviceName, sim_size);
    } else {
        panic("__cudaRegisterVar: Don't know how to register variable!");
    }
}

void __cudaRegisterVar(ThreadContext *tc, gpusyscall_t *call_params)
{
    GPUSyscallHelper helper(tc, call_params);

    // Addr sim_fatCubinHandle = *((Addr*)helper.getParam(0, true));
    Addr sim_hostVar = *((Addr*)helper.getParam(1, true));
    Addr sim_deviceAddress = *((Addr*)helper.getParam(2, true));
    Addr sim_deviceName = *((Addr*)helper.getParam(3, true));
    int sim_ext = *((int*)helper.getParam(4));
    int sim_size = *((int*)helper.getParam(5));
    int sim_constant = *((int*)helper.getParam(6));
    int sim_global = *((int*)helper.getParam(7));

    const char* deviceName = new char[MAX_STRING_LEN];
    CudaGPU *cudaGPU = CudaGPU::getCudaGPU(g_active_device);
    helper.readString(sim_deviceName, (uint8_t*)deviceName, MAX_STRING_LEN);

    cudaGPU->saveVar(sim_deviceAddress, deviceName, sim_size, sim_constant, sim_global, sim_ext, sim_hostVar);

    register_var(sim_deviceAddress, deviceName, sim_size, sim_constant, sim_global, sim_ext, sim_hostVar);
}

//  void __cudaRegisterShared(void **fatCubinHandle, void **devicePtr)
void
__cudaRegisterShared(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);
}

void
__cudaRegisterSharedVar(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);
}

void
__cudaRegisterTexture(ThreadContext *tc, gpusyscall_t *call_params)
{
    GPUSyscallHelper helper(tc, call_params);

    // Addr sim_fatCubinHandle = *((Addr*)helper.getParam(0, true));
    Addr sim_hostVar = *((Addr*)helper.getParam(1));
    // Addr sim_deviceAddress = *((Addr*)helper.getParam(2, true));
    Addr sim_deviceName = *((Addr*)helper.getParam(3, true));
    int sim_dim = *((int*)helper.getParam(4));
    int sim_norm = *((int*)helper.getParam(5));
    int sim_ext = *((int*)helper.getParam(6));
    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: __cudaRegisterTexture(fatCubinHandle** = %x, hostVar* = %x, deviceAddress* = %x, deviceName* = %x, dim = %d, norm = %d, ext = %d)\n",
            /*sim_fatCubinHandle*/ 0, sim_hostVar, /*sim_deviceAddress*/ 0,
            sim_deviceName, sim_dim, sim_norm, sim_ext);

    const char* deviceName = new char[MAX_STRING_LEN];
    CudaGPU *cudaGPU = CudaGPU::getCudaGPU(g_active_device);
    gpgpu_t *gpu = cudaGPU->getTheGPU();
    helper.readString(sim_deviceName, (uint8_t*)deviceName, MAX_STRING_LEN);

    gpu->gpgpu_ptx_sim_bindNameToTexture(deviceName, (const struct textureReference*)sim_hostVar, sim_dim, sim_norm, sim_ext);
    warn("__cudaRegisterTexture implementation is not complete!");
}

void
cudaGLRegisterBufferObject(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);
}

void
cudaGLMapBufferObject(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);
}

void
cudaGLUnmapBufferObject(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);
}

void
cudaGLUnregisterBufferObject(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);
}

#if (CUDART_VERSION >= 2010)

void
cudaHostAlloc(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);
}

void
cudaHostGetDevicePointer(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);
}

void
cudaSetValidDevices(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);
}

void
cudaSetDeviceFlags(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);
}

void
cudaEventCreateWithFlags(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);
}

void
cudaDriverGetVersion(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);
}

void
cudaRuntimeGetVersion(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);
}

#endif

void
cudaGLSetGLDevice(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);
}

void
cudaWGLGetDevice(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);
}

void
__cudaMutexOperation(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);
}

void
__cudaTextureFetch(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);
}

namespace cuda_math {
    uint64_t __cudaMutexOperation(ThreadContext *tc, gpusyscall_t *call_params)
    {
        cuda_not_implemented(__my_func__,__LINE__);
        return 0;
    }

    uint64_t __cudaTextureFetch(ThreadContext *tc, gpusyscall_t *call_params)
    {
        cuda_not_implemented(__my_func__,__LINE__);
        return 0;
    }

    uint64_t __cudaSynchronizeThreads(ThreadContext *tc, gpusyscall_t *call_params)
    {
        //TODO This function should syncronize if we support Asyn kernel calls
        return g_last_cudaError = cudaSuccess;
    }

    void  __cudaTextureFetch(const void *tex, void *index, int integer, void *val) {
        cuda_not_implemented(__my_func__,__LINE__);
    }

    void __cudaMutexOperation(int lock)
    {
        cuda_not_implemented(__my_func__,__LINE__);
    }
}

/// static functions

static int load_static_globals(GPUSyscallHelper *helper, symbol_table *symtab)
{
    DPRINTF(GPUSyscalls, "GPGPU-Sim PTX: loading globals with explicit initializers\n");
    int ng_bytes = 0;
    symbol_table::iterator g = symtab->global_iterator_begin();

    for (; g != symtab->global_iterator_end(); g++) {
        symbol *global = *g;
        if (global->has_initializer()) {
            DPRINTF(GPUSyscalls, "GPGPU-Sim PTX:     initializing '%s'\n", global->name().c_str());
            // unsigned addr = global->get_address();
            const type_info *type = global->type();
            type_info_key ti = type->get_key();
            size_t size;
            int t;
            ti.type_decode(size, t);
            int nbytes = size/8;
            Addr offset = 0;
            std::list<operand_info> init_list = global->get_initializer();
            for (std::list<operand_info>::iterator i = init_list.begin(); i != init_list.end(); i++) {
                operand_info op = *i;
                ptx_reg_t value = op.get_literal_value();

                Addr addr = global->get_address();
                helper->writeBlob(addr + offset, (uint8_t*)&value, nbytes);

                offset += nbytes;
                ng_bytes += nbytes;
            }
            DPRINTF(GPUSyscalls, " wrote %u bytes to \'%s\'\n", offset, global->name());
        }
    }
    DPRINTF(GPUSyscalls, "GPGPU-Sim PTX: finished loading globals (%u bytes total).\n", ng_bytes);
    return ng_bytes;
}

static int load_constants(GPUSyscallHelper *helper, symbol_table *symtab)
{
   DPRINTF(GPUSyscalls, "GPGPU-Sim PTX: loading constants with explicit initializers\n");
   int nc_bytes = 0;
   symbol_table::iterator g = symtab->const_iterator_begin();

   for (; g != symtab->const_iterator_end(); g++) {
      symbol *constant = *g;
      if (constant->is_const() && constant->has_initializer()) {

         // get the constant element data size
         int basic_type;
         size_t num_bits;
         constant->type()->get_key().type_decode(num_bits, basic_type);

         std::list<operand_info> init_list = constant->get_initializer();
         int nbytes_written = 0;
         for (std::list<operand_info>::iterator i = init_list.begin(); i != init_list.end(); ++i) {
            operand_info op = *i;
            ptx_reg_t value = op.get_literal_value();
            int nbytes = num_bits/8;
            switch (op.get_type()) {
            case int_t: assert(nbytes >= 1); break;
            case float_op_t: assert(nbytes == 4); break;
            case double_op_t: assert(nbytes >= 4); break; // account for double DEMOTING
            default:
               panic("Op type not recognized in load_constants"); break;
            }

            Addr addr = constant->get_address() + nbytes_written;
            helper->writeBlob(addr, (uint8_t*)&value, nbytes);

            DPRINTF(GPUSyscalls, " wrote %u bytes to \'%s\'\n", nbytes, constant->name());
            nc_bytes += nbytes;
            nbytes_written += nbytes;
         }
      }
   }
   DPRINTF(GPUSyscalls, "GPGPU-Sim PTX: finished loading constants (%u bytes total).\n", nc_bytes);
   return nc_bytes;
}

// EDGE
void RegisterPTXFile(const std::string& ptxFilename)
{
    std::ifstream fileStream(ptxFilename, std::ios::in);
    if( !fileStream ) {
        std::cout << "Error: file <" << ptxFilename.c_str() << "> does not exist" << std::endl;
        abort();
    }

    std::string ptxStr, line;
    while( getline(fileStream, line) ) {
        ptxStr += (line + "\n");
    }
    fileStream.close();

    std::cout << "============================" << std::endl;
    std::cout << ptxStr << std::endl;
    std::cout << "============================" << std::endl;

    CudaGPU *cudaGPU = CudaGPU::getCudaGPU(g_active_device);

    assert(registering_symtab == NULL);
    registering_fat_cubin_handle = next_fat_bin_handle++;
    registering_symtab = gpgpu_ptx_sim_load_ptx_from_string(ptxStr.c_str(), source_num);
    cudaGPU->add_binary(registering_symtab, registering_fat_cubin_handle);
    gpgpu_ptxinfo_load_from_string(ptxStr.c_str(), source_num);

    registering_fat_cubin_handle = 0;
    registering_symtab = NULL;
    registering_allocation_size = -1;
    registering_allocation_ptr = 0;

    std::cout << std::endl << std::endl;

    source_num++;
}

// bfs file read
void init_graph_from_file(char* fname,host_graph* h_graph)
{
    FILE *fp;

    fp = fopen(fname,"r");
    if(!fp)
    {
        printf("Error Reading graph file\n");
        return;
    }
    
    int no_of_nodes;
    int edge_list_size;
    int source = 0;

    fscanf(fp,"%d",&no_of_nodes);

    

    // allocate host memory
    

    int start, edgeno;   
    // initalize the memory
    for( unsigned int i = 0; i < no_of_nodes; i++) 
    {
        fscanf(fp,"%d %d",&start,&edgeno);
        h_graph->h_graph_nodes[i].starting = start;
        h_graph->h_graph_nodes[i].no_of_edges = edgeno;
        h_graph->h_graph_mask[i]=false;
        h_graph->h_updating_graph_mask[i]=false;
        h_graph->h_graph_visited[i]=false;
        if(i%100==0)
            printf("%d nodes read \n",i);
    }

    //read the source node from the file
    fscanf(fp,"%d",&source);
    source=0;

    //set the source node as true in the mask
    h_graph->h_graph_mask[source]=true;
    h_graph->h_graph_visited[source]=true;

    fscanf(fp,"%d",&edge_list_size);

    int id,cost;
    
    for(int i=0; i < edge_list_size ; i++)
    {
        fscanf(fp,"%d %d",&id,&cost);
        h_graph->h_graph_edges[i] = id;
        if(i%100==0)
            printf("%d edges read \n",i);
    }

    if(fp)
        fclose(fp);    

    printf("Read File\n");

}

void read_sep( float* dest, int dim, int num, int n ) {
    size_t count = 0;
    for( int i = 0; i < num && n > 0; i++ ) {
      for( int k = 0; k < dim; k++ ) {

  dest[i*dim + k] = lrand48()/(float)INT_MAX;
      }
      if(i%100==0)
            printf("read %d done, count = %d  \n",i, count );
      n--;
      count++;
    }
    //return count;
}

void pspeedy(Points *points)
{
    float z= points->z;
  int pid=0; 
  int nproc=1;

  //my block
  long kcenter;
  long bsize = points->num/nproc;
  long k1 = bsize * pid;
  long k2 = k1 + bsize;
  k2 = points->num;

  static float totalcost;
  
  static bool open = false;
  static float* costs; //cost for each thread. 
  static int i;


    printf("Speedy: facility cost %lf\n", z);


  /* create center at first point, send it to itself */
  for( int k = k1; k < k2; k++ )    {
    float distance = dist(points->p[k],points->p[0],points->dim);
    points->p[k].cost = distance * points->p[k].weight;
    points->p[k].assign=0;
  }

  
    kcenter = 1;
    costs = (float*)malloc(sizeof(float)*nproc);
  
   
  if( pid != 0 ) { // we are not the master threads. we wait until a center is opened.
    while(1) {

      if( i >= points->num ) break;
      for( int k = k1; k < k2; k++ )
    {
      float distance = dist(points->p[i],points->p[k],points->dim);
      if( distance*points->p[k].weight < points->p[k].cost )
        {
          points->p[k].cost = distance * points->p[k].weight;
          points->p[k].assign=i;
        }
    }

    } 
  }
  else  { // I am the master thread. I decide whether to open a center and notify others if so. 
    for(i = 1; i < points->num; i++ )  {
      bool to_open = ((float)lrand48()/(float)INT_MAX)<(points->p[i].cost/z);
      if( to_open )  {
    (kcenter)++;

    open = true;

    for( int k = k1; k < k2; k++ )  {
      float distance = dist(points->p[i],points->p[k],points->dim);
      if( distance*points->p[k].weight < points->p[k].cost )  {
        points->p[k].cost = distance * points->p[k].weight;
        points->p[k].assign=i;
      }
    }

    open = false;

      }
    }

    open = true;

  }

  open = false;
  float mytotal = 0;
  for( int k = k1; k < k2; k++ )  {
    mytotal += points->p[k].cost;
  }
  costs[pid] = mytotal;

  // aggregate costs from each thread
  if( pid == 0 )
    {
      totalcost=z*(kcenter);
      for( int i = 0; i < nproc; i++ )
    {
      totalcost += costs[i];
    } 
      free(costs);
    }



      printf("Speedy opened %d facilities for total cost %lf\n",
          kcenter, totalcost);
      printf("Distance Cost %lf\n", totalcost - z*(kcenter));
   
  points->kcenter=kcenter;
  points->totalcost=totalcost;
}

float dist(Point p1, Point p2, int dim)
{
  int i;
  float result=0.0;
  for (i=0;i<dim;i++)
    result += (p1.coord[i] - p2.coord[i])*(p1.coord[i] - p2.coord[i]);

  return(result);
}