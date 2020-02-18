# cray
module purge
module load modules
module load craype
module load slurm
module load xalt
export CRAYPE_LINK_TYPE=dynamic

# cpu
module load daint-mc

# compiler
module load PrgEnv-gnu 
export CC=`which cc`
export CXX=`which CC`

# dependencies
module load CMake
module load intel
module load cray-mpich
module unload cray-libsci

# dependencies
#hpx_dir=/apps/daint/UES/biddisco/build/hpx-debug/lib/cmake/HPX  # Debug
hpx_dir=$HOME/software/hpx-futures
