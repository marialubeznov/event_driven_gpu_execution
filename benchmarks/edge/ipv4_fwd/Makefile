################################################################################
#
# Build script for project
#
################################################################################

# Add source files here
EXECUTABLE	:= ip_forward_conv
# CUDA source files (compiled with cudacc)
CUFILES		:= ip_forward_conv.cu backprop_cuda.cu bfs.cu
# CUDA dependency files
CU_DEPS		:= backprop_cuda_kernel.cu kernel.cu kernel2.cu
# C/C++ source files (compiled with gcc / c++)
CFILES		:= util.cpp ipv4.cpp rte_lpm.cpp facetrain.c imagenet.c backprop.c

CFLAGS		:= -DOUTPUT
CUFLAGS		:= -DOUTPUT

################################################################################
# Rules and targets

include ../common/common.mk
