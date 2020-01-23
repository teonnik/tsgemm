#pragma once

#include <mpi.h>
#include <complex>
#include <cstdio>

namespace tsgemm {

// Map between C++ and MPI types.
//
template <typename scalar>
MPI_Datatype get_mpi_type();

template <>
MPI_Datatype get_mpi_type<float>() {
  return MPI_FLOAT;
};

template <>
MPI_Datatype get_mpi_type<double>() {
  return MPI_DOUBLE;
};

template <>
MPI_Datatype get_mpi_type<std::complex<float>>() {
  return MPI_CXX_FLOAT_COMPLEX;
};

template <>
MPI_Datatype get_mpi_type<std::complex<double>>() {
  return MPI_CXX_DOUBLE_COMPLEX;
};

void check_num_procs(int exp_num_procs) {
  int act_num_procs;
  MPI_Comm_size(MPI_COMM_WORLD, &act_num_procs);
  if (act_num_procs != exp_num_procs) {
    printf("[ERROR] The number of processes doesn't match the process grid!");
    exit(0);
  }
}

int get_proc_rank(MPI_Comm comm_cart) {
  int rank;
  MPI_Comm_rank(comm_cart, &rank);
  return rank;
}

int get_proc_rank(MPI_Comm comm_cart, int row, int col) {
  int rank;
  int coords[2] = {row, col};
  MPI_Cart_rank(comm_cart, coords, &rank);
  return rank;
}

std::array<int, 2> get_proc_coords(MPI_Comm comm_cart) {
  int rank = get_proc_rank(comm_cart);
  int pcoords[2];
  MPI_Cart_coords(comm_cart, rank, 2, pcoords);
  return {pcoords[0], pcoords[1]};
}

// Initializes a grid with Cartesian coordinates.
//
MPI_Comm init_comm_cart(int pgrid_rows, int pgrid_cols) {
  MPI_Comm comm_cart;
  constexpr int ndims = 2;
  int const dims[ndims] = {pgrid_rows, pgrid_cols};
  constexpr int periods[ndims] = {false, false};
  constexpr int reorder = true;
  MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &comm_cart);
  return comm_cart;
}

// Wrap MPI runtime initialization and destruction.
struct mpi_init {
  mpi_init(int argc, char** argv, int thd_required) {
    int thd_provided;
    MPI_Init_thread(&argc, &argv, thd_required, &thd_provided);

    if (thd_required != thd_provided) {
      std::fprintf(stderr, "Provided MPI threading model does not match the required one.\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }

  ~mpi_init() {
    MPI_Finalize();
  }
};  // end struct mpi_init

}  // end namespace tsgemm
