#!/bin/sh

TARGET=gem5.opt

if [ $# -gt 0 ]; then
    if [ $1 = "opt" ]; then
        TARGET=gem5.opt
    else
        TARGET=gem5.debug
    fi
fi

scons build/X86_VI_hammer_GPU/$TARGET --default=X86 EXTRAS=../gem5-gpu/src:../gpgpu-sim/ PROTOCOL=VI_hammer GPGPU_SIM=True -j 9
