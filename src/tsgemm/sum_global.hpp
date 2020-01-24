#pragma once

#include <mpi.h>
#include <tsgemm/mpi_utils.hpp>
#include <vector>

namespace tsgemm {

template <typename scalar>
scalar sum_global(MPI_Comm comm_cart, std::vector<scalar> const &buffer) {
  using tsgemm::get_proc_rank;

  scalar local_sum = 0;
  for (auto el : buffer) {
    local_sum += el;
  }

  scalar global_sum = 0;
  MPI_Allreduce(&local_sum, &global_sum, 1, tsgemm::get_mpi_type<scalar>(),
                MPI_SUM, comm_cart);

  return global_sum;
}

} // end namespace tsgemm
