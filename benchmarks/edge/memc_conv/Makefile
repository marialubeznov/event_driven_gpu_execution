################################################################################
#
# Build script for project
#
################################################################################

# Add source files here
EXECUTABLE	:= memc_conv
# CUDA source files (compiled with cudacc)
#CUFILES		:= network_event.cu isr.cu
CUFILES		:= memc_conv.cu backprop_cuda.cu bfs.cu
# CUDA dependency files
CU_DEPS		:= backprop_cuda_kernel.cu kernel.cu kernel2.cu
# C/C++ source files (compiled with gcc / c++)
CFILES		:= facetrain.c imagenet.c backprop.c 

CFLAGS		:= -DOUTPUT
CUFLAGS		:= -DOUTPUT

################################################################################
# Rules and targets

include ../common/common.mk
