#pragma once

#include <array>
#include <complex>
#include <cstdio>
#include <mpi.h>

namespace tsgemm {

// Map between C++ and MPI types.
template <typename scalar>
MPI_Datatype get_mpi_type();

void check_num_procs(int exp_num_procs);

int get_proc_rank(MPI_Comm comm_cart);

int get_proc_rank(MPI_Comm comm_cart, int row, int col);

std::array<int, 2> get_proc_coords(MPI_Comm comm_cart);

MPI_Comm init_comm_cart(int pgrid_rows, int pgrid_cols);

// Wrap MPI runtime initialization and destruction.
struct mpi_init {
  mpi_init(int argc, char **argv, int thd_required);
  ~mpi_init();
}; // end struct mpi_init

} // end namespace tsgemm
