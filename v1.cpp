#include <dlaf/communication/communicator.h>
#include <dlaf/matrix.h>
#include <dlaf/mpi_header.h>

#include <mpi.h>
#include <blas.hh>
#include <boost/program_options.hpp>
#include <hpx/dataflow.hpp>
#include <hpx/hpx.hpp>
#include <hpx/hpx_start.hpp>
//#include <hpx/runtime/threads/run_as_hpx_thread.hpp>

#include <complex>
#include <cstdio>
#include <iostream>
#include <vector>

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

// A struct representing a 2D-region.
//
struct span2d {
  int rows;
  int cols;

  // Column-major index
  //
  int idx(int i, int j) const noexcept {
    return i + j * rows;
  }

  std::pair<int, int> coords(int index) const noexcept {
    return {index % rows, index / rows};
  }

  int size() const noexcept {
    return rows * cols;
  }

  int ld() const noexcept {
    return rows;
  }

  // Iterates in column major order.
  // f : (i, j) -> void
  //
  template <typename Func>
  void loop(Func f) const noexcept {
    for (int j = 0; j < cols; ++j) {
      for (int i = 0; i < rows; ++i) {
        f(i, j);
      }
    }
  }

  // Iterates in column major order.
  // f : (this_index, other_index) -> void
  //
  template <typename Func>
  void loop(span2d other, Func f) const noexcept {
    for (int j = 0; j < cols; ++j) {
      for (int i = 0; i < rows; ++i) {
        f(idx(i, j), other.idx(i, j));
      }
    }
  }
};

// Returns the coordinate of the process holding `index` in 2D block-cyclic distribution.
//
int find_pcoord(int index, int blk, int nproc) {
  return (index / blk) % nproc;
}

// Interleaves splits from blocks and tiles.
//
template <typename Predicate, typename Transform>
std::vector<int> splits(int len, int blk, int tile, Predicate filter_f, Transform transf_f) {
  int num_blocks = (len + blk - 1) / blk;
  int num_tiles = (len + tile - 1) / tile;

  std::vector<int> splits;
  splits.reserve(num_tiles + num_blocks);

  int bi = 0;
  int ti = 0;
  while (bi != num_blocks || ti != num_tiles) {
    int blk_delim = bi * blk;
    int tile_delim = ti * tile;

    int delim = (blk_delim > tile_delim) ? tile_delim : blk_delim;
    bi = (blk_delim > tile_delim) ? bi : bi + 1;
    ti = (blk_delim < tile_delim) ? ti : ti + 1;

    if (filter_f(delim))
      splits.push_back(transf_f(delim));
  }
  if (filter_f(len))
    splits.push_back(transf_f(len));

  return splits;
}

std::vector<int> splits(int len, int blk, int tile) {
  auto filter_f = [](int) { return true; };  // no filter
  auto transf_f = [](int x) { return x; };   // identity
  return splits(len, blk, tile, filter_f, transf_f);
}

// nproc - the number of processes along the dimension
// pcoord - the coordinate of the process along the dimension
//
std::vector<int> splits(int len, int blk, int tile, int nproc, int pcoord) {
  auto filter_f = [blk, nproc, pcoord](int delim) { return find_pcoord(delim, blk, nproc) == pcoord; };
  auto transf_f = [blk, nproc](int delim) {
    int b_idx = (delim / blk) / nproc;  // index of the block
    int el_idx = delim % blk;           // index within the block
    return b_idx * blk + el_idx;
  };
  return splits(len, blk, tile, filter_f, transf_f);
}

template <typename scalar>
void tile_gemm(int tile_m, int tile_n, int tile_k, scalar const* a_buffer, int lda,
               scalar const* b_buffer, int ldb, scalar* c_buffer, int ldc) {
  constexpr scalar alpha = 1;
  constexpr scalar beta = 1;
  blas::gemm(blas::Layout::ColMajor, blas::Op::ConjTrans, blas::Op::NoTrans, tile_m, tile_n, tile_k,
             alpha, a_buffer, lda, b_buffer, ldb, beta, c_buffer, ldc);
}

// For each `rank`, assigns a tag equal to the number of previous occurences within the `ranks_map`. This
// ensures that each message sent from `this` process to the `rank` has a unique tag. `ranks_map` and
// `c_pieces` are ordered in column-major, the order of tags within slabs follow that ordering.
//
std::vector<int> init_tags_map(int num_procs, std::vector<int> const& ranks_map) {
  std::vector<int> tags_map(ranks_map.size());
  for (int r = 0; r < num_procs; ++r) {
    int tag = 0;
    for (int i_map = 0; i_map < ranks_map.size(); ++i_map) {
      if (ranks_map[i_map] == r) {
        tags_map[i_map] = tag;
        ++tag;
      }
    }
  }

  return tags_map;
}

std::vector<int> init_ranks_map(std::vector<int> const& c_split_r, std::vector<int> const& c_split_c,
                                int p_r, int p_c, int bs_r, int bs_c) {
  int num_c_seg_r = static_cast<int>(c_split_r.size()) - 1;
  int num_c_seg_c = static_cast<int>(c_split_c.size()) - 1;
  int num_c_pieces = num_c_seg_r * num_c_seg_c;

  std::vector<int> ranks_map(num_c_pieces);

  for (int j = 0; j < num_c_seg_c - 1; ++j) {
    for (int i = 0; i < num_c_seg_r; ++i) {
      int map_idx = i + j * num_c_seg_r;
      int pcoord_r = find_pcoord(c_split_r[i], bs_r, p_r);
      int pcoord_c = find_pcoord(c_split_c[j], bs_c, p_c);
      ranks_map[map_idx] = pcoord_r + pcoord_c * p_r;
    }
  }

  return ranks_map;
}

// In SIRIUS, `A`, `B` and `C` are usually submatrices of bigger matrices. The
// only difference that entails is that the `lld` for `C` might be larger than
// assumed here. Hence writing to `C` might be slightly faster than in SIRIUS.
//
// Assumptions: Tall and skinny `k` >> `m` and `k` >> `n`.
//
// Matrices: `A` (`m x k`), `B` (`k x n`) and `C` (m x n).
//
// `A` is complex conjugated.
//
// `C` is distributed in 2D block-cyclic manner. The 2D process grid is row
// major (the MPI default) with process 0 in the top left corner.
//
// All matrices are distributed in column-major order.
//
// Example usage:
//
//     mpirun -np 1 tsgemm --mnk 100 100 10000 --tile_mnk 64 64 64 --proc_grid 1 1 --blk_dims 32 32
//
int tsgemm_main(int argc, char** argv) {
  using scalar_t = std::complex<double>;
  using hpx::dataflow;
  using hpx::util::unwrapping;
  using hpx::future;
  namespace po = boost::program_options;

  // Input
  //
  po::options_description desc("Allowed options.");

  // clang-format off
  desc.add_options()
    ("mnk",       po::value<std::vector<int>>() ->multitoken(), "mnk")
    ("tile_mnk",  po::value<std::vector<int>>() ->multitoken(), "Tile mnk dimensions.")
    ("proc_grid", po::value<std::vector<int>>() ->multitoken(), "Process grid")
    ("blk_dims",  po::value<std::vector<int>>() ->multitoken(), "Block sizes.")
  ;
  // clang-format on

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  std::vector<int> mnk = vm["mnk"].as<std::vector<int>>();
  std::vector<int> tile_mnk = vm["tile_mnk"].as<std::vector<int>>();
  std::vector<int> proc_grid = vm["proc_grid"].as<std::vector<int>>();
  std::vector<int> blk_dims = vm["blk_dims"].as<std::vector<int>>();

  int m = mnk[0];
  int n = mnk[1];
  int k = mnk[2];
  int tile_m = tile_mnk[0];
  int tile_n = tile_mnk[1];
  int tile_k = tile_mnk[2];
  int p_r = proc_grid[0];
  int p_c = proc_grid[1];
  int bs_r = blk_dims[0];
  int bs_c = blk_dims[1];

  for (int input : {m, n, k, p_r, p_c, bs_r, bs_c, tile_m, tile_n, tile_k}) {
    if (input <= 0) {
      std::cout << "[ERROR] All parameters must be positive!\n";
      std::terminate();
    }
  }

  // Communicator
  //
  MPI_Comm comm_world = MPI_COMM_WORLD;

  // Number of processes
  //
  int p = p_r * p_c;
  int num_procs;
  MPI_Comm_size(comm_world, &num_procs);
  if (num_procs != p) {
    std::cout << "[ERROR] The number of processes doesn't match the process grid!";
    exit(0);
  }

  // Get the coordinates of the grid from the rank.
  //
  MPI_Comm comm_cart;
  constexpr int ndims = 2;
  int const dims[ndims] = {p_r, p_c};
  constexpr int periods[ndims] = {false, false};
  constexpr int reorder = true;
  MPI_Cart_create(comm_world, ndims, dims, periods, reorder, &comm_cart);

  // Process rank and coords
  //
  int r;
  int pcoords[2];
  MPI_Comm_rank(comm_cart, &r);
  MPI_Cart_coords(comm_cart, r, ndims, pcoords);

  // Local distribution of A and B. Only the `k` dimension is split. In SIRIUS, `k_loc` is approximately
  // equally distributed. `k_loc` coincides with `lld` for `A` and `B`. If there is a remainder,
  // distributed it across ranks starting from the `0`-th.
  //
  int k_rem = k % p;
  int k_loc = k / p + ((r < k_rem) ? 1 : 0);

  // Delimiters descibing how C is split locally and globally along columns and rows.
  //
  std::vector<int> c_split_r = splits(m, bs_r, tile_m);
  std::vector<int> c_split_c = splits(n, bs_c, tile_n);
  std::vector<int> slab_split_r = splits(m, bs_r, tile_m, p_r, pcoords[0]);
  std::vector<int> slab_split_c = splits(n, bs_c, tile_n, p_c, pcoords[1]);

  // Local tile grid
  //
  int tgrid_m = (m + tile_m - 1) / tile_m;
  int tgrid_n = (n + tile_n - 1) / tile_n;
  int tgrid_k = (k + tile_k - 1) / tile_k;

  // Local matrix layouts
  //
  span2d a_ts{m, k_loc};
  span2d b_ts{n, k_loc};
  span2d c_ini{m, n};
  span2d c_fin{slab_split_r.back(), slab_split_c.back()};

  // Data for A, B:
  //
  // - all buffers are stored in column-major layout
  // - values in `a_buffer` and `b_buffer` are irrelevant
  // - `c_ini` has to be initialized to zero (for accumulation)
  // - `c_fin` is the local portion of the 2D block cyclic distribution
  // - `c_ini` and `c_fin` have to be initialized to 0.
  //
  std::vector<scalar_t> a_buffer(a_ts.size(), 42);
  std::vector<scalar_t> b_buffer(b_ts.size(), 42);
  std::vector<scalar_t> c_ini_buffer(c_ini.size(), 0);
  std::vector<scalar_t> c_fin_buffer(c_fin.size(), 0);

  // Futures for all tiles in column major order.
  //
  std::vector<future<void>> c_ini_futures(tgrid_m * tgrid_n);
  // std::vector<future<void>> c_fin_tiles(tgrid_m * tgrid_n);

  // GEMM
  // -------------------------------------------------------------------------------------

  // Local gemm
  //
  // - Tiles along the `k` dimension are chained.
  // - `A` is transposed and has similar layout to `B`.
  // - Tile sizes near borders have to be adjusted: `len_x`
  //
  for (int i = 0; i < tgrid_m; ++i) {
    for (int j = 0; j < tgrid_n; ++j) {
      for (int k = 0; k < tgrid_k; ++k) {
        int len_m = tile_m - ((i != tgrid_m - 1) ? 0 : m % tile_m);
        int len_n = tile_n - ((j != tgrid_n - 1) ? 0 : n % tile_n);
        int len_k = tile_k - ((k != tgrid_k - 1) ? 0 : k_loc % tile_k);

        int a_offset = a_ts.idx(k * tile_k, i * tile_m);
        int b_offset = b_ts.idx(k * tile_k, j * tile_n);
        int c_offset = c_ini.idx(i * tile_m, j * tile_n);

        scalar_t const* a_ptr = a_buffer.data() + a_offset;
        scalar_t const* b_ptr = b_buffer.data() + b_offset;
        scalar_t* c_ptr = c_ini_buffer.data() + c_offset;

        int c_tiles_idx = i + j * tgrid_m;
        c_ini_futures[c_tiles_idx] =
            dataflow(unwrapping(tile_gemm<scalar_t>), len_m, len_n, len_k, a_ptr, a_ts.ld(), b_ptr,
                     b_ts.ld(), c_ptr, c_ini.ld(), std::move(c_ini_futures[c_tiles_idx]));
      }
    }
  }

  hpx::wait_all(c_ini_futures);

  // SEND
  // -------------------------------------------------------------------------------------

  // Ranks and tags maps for each piece.
  //
  std::vector<int> ranks_map = init_ranks_map(c_split_r, c_split_r, p_r, p_c, bs_r, bs_c);
  std::vector<int> tags_map = init_tags_map(p, ranks_map);
  span2d pieces_grid{static_cast<int>(c_split_r.size()) - 1, static_cast<int>(c_split_c.size()) - 1};

  // Allocate send buffers.
  //
  std::vector<scalar_t> snd_buffer(c_ini.size());
  std::vector<MPI_Request> snd_reqs(pieces_grid.size());

  // For each piece at (i, j) of C_ini stored locally issue a send to the process it belongs in C_fin.
  //
  int snd_offset = 0;
  auto send_f = [&c_split_c, &c_split_r, &c_ini_buffer, &c_ini, &snd_buffer, &snd_offset, &pieces_grid,
                 &ranks_map, &snd_reqs, &tags_map, &comm_cart](int i, int j) {
    int begin_c = c_split_c[j];
    int begin_r = c_split_r[i];

    span2d piece{c_split_r[i + 1] - begin_r, c_split_c[j + 1] - begin_c};

    int nelems = piece.size();
    int c_ini_offset = c_ini.idx(begin_r, begin_c);

    scalar_t const* c_ptr = c_ini_buffer.data() + c_ini_offset;
    scalar_t* snd_ptr = snd_buffer.data() + snd_offset;

    piece.loop(c_ini, [c_ptr, snd_ptr](int piece_idx, int c_ini_idx) {
      snd_ptr[piece_idx] = c_ptr[c_ini_idx];
    });

    int piece_idx = pieces_grid.idx(i, j);
    int dest_rank = ranks_map[piece_idx];
    auto& snd_req = snd_reqs[piece_idx];
    int tag = tags_map[piece_idx];
    MPI_CALL(MPI_Issend(snd_ptr, nelems, get_mpi_type<scalar_t>(), dest_rank, tag, comm_cart, &snd_req))

    snd_offset += nelems;
  };
  pieces_grid.loop(send_f);

  // RECEIVE
  // -------------------------------------------------------------------------------------

  // Define a grid for pieces in slabs
  //
  span2d slabs_grid{static_cast<int>(slab_split_r.size()) - 1,
                    static_cast<int>(slab_split_c.size()) - 1};

  // Allocate receive buffers
  //
  std::vector<scalar_t> rcv_buffer(p * c_fin.size());
  std::vector<MPI_Request> rcv_reqs(p * slabs_grid.size());

  // For each piece at (i, j) issue a receive from all processes.
  //
  int rcv_offset = 0;
  int rcv_req_idx = 0;
  auto recv_f = [p, &rcv_offset, &rcv_req_idx, &slab_split_r, &slab_split_c, &slabs_grid, &rcv_buffer,
                 &rcv_reqs, &comm_cart](int i, int j) {
    int nelems = (slab_split_c[j + 1] - slab_split_c[j]) * (slab_split_r[i + 1] - slab_split_r[i]);
    int tag = slabs_grid.idx(i, j);

    for (int src_rank = 0; src_rank < p; ++src_rank) {
      scalar_t* rcv_ptr = rcv_buffer.data() + rcv_offset;
      auto& rcv_req = rcv_reqs[rcv_req_idx];

      MPI_CALL(MPI_Irecv(rcv_ptr, nelems, get_mpi_type<scalar_t>(), src_rank, tag, comm_cart, &rcv_req))

      rcv_offset += nelems;
      ++rcv_req_idx;
    }
  };
  slabs_grid.loop(recv_f);

  // TODO: Receives are stalling.

  // Wait for all data to be received.
  //
  MPI_Waitall(static_cast<int>(rcv_reqs.size()), rcv_reqs.data(), MPI_STATUS_IGNORE);

  // For each piece at (i, j) assemble received data from all processes.
  //
  rcv_offset = 0;
  auto assemble_f = [p, bs_r, bs_c, &c_fin, &c_fin_buffer, &rcv_buffer, &slab_split_c, &slab_split_r,
                     &rcv_offset](int i, int j) {
    span2d piece{slab_split_r[i + 1] - slab_split_r[i], slab_split_c[j + 1] - slab_split_c[j]};

    int offset_c = c_fin.idx(i * bs_r, j * bs_c);
    scalar_t* c_ptr = c_fin_buffer.data() + offset_c;

    for (int src_rank = 0; src_rank < p; ++src_rank) {
      scalar_t const* rcv_ptr = rcv_buffer.data() + rcv_offset;

      piece.loop(c_fin, [c_ptr, rcv_ptr](int piece_idx, int c_fin_idx) {
        c_ptr[c_fin_idx] += rcv_ptr[piece_idx];
      });

      rcv_offset += piece.size();
    }
  };
  slabs_grid.loop(assemble_f);

  // Wait until all data is sent
  //
  MPI_Waitall(static_cast<int>(snd_reqs.size()), snd_reqs.data(), MPI_STATUS_IGNORE);

  return hpx::finalize();
}

int main(int argc, char** argv) {
  // Initialize MPI
  //
  int threading_required = MPI_THREAD_SERIALIZED;
  int threading_provided;
  MPI_Init_thread(&argc, &argv, threading_required, &threading_provided);

  if (threading_provided != threading_required) {
    std::fprintf(stderr, "Provided MPI threading model does not match the required one.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // Initialize HPX
  //
  hpx::start(nullptr, argc, argv);
  hpx::runtime* rt = hpx::get_runtime_ptr();
  hpx::util::yield_while([rt]() { return rt->get_state() < hpx::state_running; });

  // Run tsgemm
  //
  auto ret = hpx::run_as_hpx_thread(&tsgemm_main, argc, argv);

  // Stop HPX & MPI
  //
  hpx::stop();
  MPI_Finalize();

  return ret;
}
