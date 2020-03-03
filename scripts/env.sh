#!/bin/sh

if [[ ${device} == daint ]]; then
  module purge
  module load modules craype slurm xalt # cray
  module load daint-mc PrgEnv-gnu CMake intel cray-mpich # dependencies 
  module unload cray-libsci
  export CRAYPE_LINK_TYPE=dynamic
  export CC=`which cc`
  export CXX=`which CC`
  #hpx_dir=/apps/daint/UES/biddisco/build/hpx-debug/lib/cmake/HPX  # Debug
  hpx_dir=$HOME/software/hpx-futures
elif [[ ${device} == laptop ]]; then
  mpi_dir=$(spack location -i mpich)
  mkl_dir=$(spack location -i intel-mkl)/mkl
  hpx_dir=$HOME/software/hpx-futures
  export MKLROOT=$mkl_dir
  export CC=$mpi_dir/bin/mpicc
  export CXX=$mpi_dir/bin/mpicxx
else
  echo "device can be laptop or daint\n!"
  exit 1
fi
