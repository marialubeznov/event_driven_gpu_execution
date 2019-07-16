#ifndef _PACKET_SWIZZLE_H_
#define _PACKET_SWIZZLE_H_

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <stddef.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <sstream>
#include <cstring>


#include <cuda.h>
#include <builtin_types.h>
//#include <drvapi_error_string.h>

// includes, project
//#include <helper_cuda_drvapi.h>
//#include <helper_timer.h>
//#include <helper_string.h>
//#include <helper_image.h>

#include "common.h"



//define input ptx file for different platforms
#define PTX_FILE "packet_swizzle_kernel.ptx"
#define CUBIN_FILE "packet_swizzle_kernel.cubin"


struct kernel_args {
    unsigned buffer_size;
    unsigned batch_size;
    unsigned n_batches;
    void* h_packet_buffer; 
    void* h_response_buffer; 
    CUdeviceptr g_packet_buffer;
    CUdeviceptr g_response_buffer;
};



#endif // _PACKET_SWIZZLE_H_
