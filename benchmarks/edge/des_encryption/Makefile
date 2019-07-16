################################################################################
#
# Build script for project
#
################################################################################

# Add source files here
EXECUTABLE	:= des_conv
# CUDA source files (compiled with cudacc)
CUFILES		:= des_conv.cu  bfs.cu des.cu backprop_cuda.cu
# CUDA dependency files
CU_DEPS		:= kernel.cu kernel2.cu
# C/C++ source files (compiled with gcc / c++)
CFILES		:= facetrain.c imagenet.c backprop.c

CFLAGS		:= -DOUTPUT
CUFLAGS		:= -DOUTPUT

################################################################################
# Rules and targets

include ../common/common.mk
