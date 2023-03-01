#!/bin/bash

set -e
set -x

#git submodule init
#git submodule update --remote

# Volta - 70
# Turing (2080) - 75
# Ampere (3080) - 86
cuda_arch=70

while getopts a: flag
do
    case "${flag}" in
        a) cuda_arch=${OPTARG};;
    esac
done

if [ ! -f "blst/libblst.a" ]; then
    git clone git@github.com:supranational/sppark.git
    git clone git@github.com:supranational/blst.git
    cd blst
    sh build.sh
    cd ..
fi

LIBS="-Lblst -lblst"
INCLUDES="-Iblst/src -Isppark/ff"
FLAGS="-O2 -D__ADX__ -arch=sm_$cuda_arch -Xcompiler -Wno-subobject-linkage -Xcompiler -O3"
OMP="-Xcompiler -fopenmp"

nvcc $LIBS $INCLUDES $FLAGS -c src/tree_builder_device.cu

nvcc $LIBS $INCLUDES -Isppark/util $FLAGS -c src/pre_commit_phase2.cu $OMP

ar rc libpc2.a pre_commit_phase2.o tree_builder_device.o blst/libblst.a

nvcc $LIBS $INCLUDES $FLAGS -o run_pc2 test.cu -L./ -lpc2 $OMP
