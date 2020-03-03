#!/bin/bash

# ---- input
device=daint # laptop
src_dir=$HOME/code/tsgemm
build_dir=$HOME/build/tsgemm
# ----

# dependencies
source $src_dir/scripts/env.sh

mkdir -p $build_dir
cd $build_dir

rm CMakeCache.txt
rm -rf CMakeFiles

cmake $src_dir \
  -D CMAKE_BUILD_TYPE=Release \
  -D HPX_IGNORE_COMPILER_COMPATIBILITY=ON \
  -D CMAKE_PREFIX_PATH=$hpx_dir

make -j8
