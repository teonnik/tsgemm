  mpi_dir=$(spack location -i mpich)
  mkl_dir=$(spack location -i intel-mkl)/mkl
  hpx_dir=$HOME/software/hpx-futures
  export MKLROOT=$mkl_dir
  export CC=$mpi_dir/bin/mpicc
  export CXX=$mpi_dir/bin/mpicxx
