#include <dlaf/communication/communicator.h>
#include <dlaf/matrix.h>
#include <dlaf/mpi_header.h>

#include <mpi.h>
#include <blas.hh>
#include <hpx/dataflow.hpp>
#include <hpx/hpx.hpp>
#include <hpx/hpx_start.hpp>
#include <hpx/lcos/local/mutex.hpp>

//#include <hpx/program_options.hpp>
#include <hpx/runtime/threads/run_as_hpx_thread.hpp>

#include <algorithm>
#include <chrono>
#include <complex>
#include <cstdio>
#include <iostream>
#include <mutex>
#include <vector>

/* TODO:
 * - check whether input is positive in triplet and span constructors
 * - write a struct for splits.
 */

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

static auto t_gemm = std::chrono::duration<double>(0);
static auto t_send = std::chrono::duration<double>(0);

// Matrix triplet values.
//
struct triplet {
  int m;
  int n;
  int k;
};

// Returns the coordinate of the process holding `index` in 2D block-cyclic distribution.
//
int find_pcoord(int index, int blk, int nproc) {
  return (index / blk) % nproc;
}

// Interleaves splits from blocks and tiles.
//
std::vector<int> splits(int len, int blk, int tile) {
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

    splits.push_back(delim);
  }
  splits.push_back(len);

  return splits;
}

// nproc - the number of processes along the dimension
// pcoord - the coordinate of the process along the dimension
//
std::vector<int> slab_splits(std::vector<int> const& matrix_splits, int blk, int nproc, int pcoord) {
  int num_matrix_splits = static_cast<int>(matrix_splits.size());
  std::vector<int> slab_splits;
  slab_splits.reserve(num_matrix_splits);

  // Iterates over all delimiters picking those belonging to the process and converting them to local values.
  //
  int last_delim_index = 0;
  for (int i_delim = 0; i_delim < num_matrix_splits - 1; ++i_delim) {
    int delim = matrix_splits[i_delim];
    if (find_pcoord(delim, blk, nproc) == pcoord) {
      int b_idx = (delim / blk) / nproc;  // index of the block
      int el_idx = delim % blk;           // index within the block
      slab_splits.push_back(b_idx * blk + el_idx);
      last_delim_index = i_delim;
    }
  }
  int last_len = matrix_splits[last_delim_index + 1] - matrix_splits[last_delim_index];
  slab_splits.push_back(slab_splits.back() + last_len);

  return slab_splits;
}

template <typename scalar>
void tile_gemm(int tile_m, int tile_n, int tile_k, scalar const* a_buffer, int lda,
               scalar const* b_buffer, int ldb, scalar* c_buffer, int ldc) {
  auto t_begin = std::chrono::high_resolution_clock::now();
  constexpr scalar alpha = 1;
  constexpr scalar beta = 1;
  blas::gemm(blas::Layout::ColMajor, blas::Op::ConjTrans, blas::Op::NoTrans, tile_m, tile_n, tile_k,
             alpha, a_buffer, lda, b_buffer, ldb, beta, c_buffer, ldc);
  auto t_end = std::chrono::high_resolution_clock::now();
  t_gemm += t_end - t_begin;
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

// NOTE: Ranks in row-major order within the process grid. This is the default in MPI.
//
std::vector<int> init_ranks_map(std::vector<int> const& c_split_r, std::vector<int> const& c_split_c,
                                span2d pgrid, span2d blk) {
  int num_c_seg_r = static_cast<int>(c_split_r.size()) - 1;
  int num_c_seg_c = static_cast<int>(c_split_c.size()) - 1;
  int num_c_pieces = num_c_seg_r * num_c_seg_c;

  std::vector<int> ranks_map(num_c_pieces);

  for (int j = 0; j < num_c_seg_c; ++j) {
    for (int i = 0; i < num_c_seg_r; ++i) {
      int map_idx = i + j * num_c_seg_r;
      int pcoord_r = find_pcoord(c_split_r[i], blk.rows, pgrid.rows);
      int pcoord_c = find_pcoord(c_split_c[j], blk.cols, pgrid.cols);
      ranks_map[map_idx] = pcoord_r * pgrid.cols + pcoord_c;
    }
  }

  return ranks_map;
}

void check_num_procs(int exp_num_procs) {
  int act_num_procs;
  MPI_Comm_size(MPI_COMM_WORLD, &act_num_procs);
  if (act_num_procs != exp_num_procs) {
    std::cout << "[ERROR] The number of processes doesn't match the process grid!";
    exit(0);
  }
}

int get_proc_rank(MPI_Comm comm_cart) {
  int rank;
  MPI_Comm_rank(comm_cart, &rank);
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
MPI_Comm init_comm_cart(span2d pgrid) {
  MPI_Comm comm_cart;
  constexpr int ndims = 2;
  int const dims[ndims] = {pgrid.rows, pgrid.cols};
  constexpr int periods[ndims] = {false, false};
  constexpr int reorder = true;
  MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &comm_cart);
  return comm_cart;
}

int tile_len(int tcoord, int tgrid, int len, int tile) {
  int tile_rem = len % tile;
  if (tcoord == tgrid - 1 && tile_rem != 0)
    return tile_rem;
  return tile;
};

// ****************************************************************************************

// Local gemm
//
// - Tiles along the `k` dimension are chained.
// - `A` is transposed and has similar layout to `B`.
// - Tile sizes near borders have to be adjusted: `len_x`
//
template <typename scalar>
std::vector<hpx::future<void>> local_gemm(triplet lmat, triplet tile,
                                          std::vector<scalar> const& a_buffer,
                                          std::vector<scalar> const& b_buffer,
                                          std::vector<scalar>& c_buffer) {
  using hpx::dataflow;
  using hpx::util::unwrapping;
  using hpx::future;

  triplet tgrid{(lmat.m + tile.m - 1) / tile.m, (lmat.n + tile.n - 1) / tile.n,
                (lmat.k + tile.k - 1) / tile.k};
  span2d a_span{lmat.k, lmat.m};
  span2d b_span{lmat.k, lmat.n};
  span2d c_span{lmat.m, lmat.n};

  // Futures for all tiles in column major order.
  //
  std::vector<future<void>> c_ini_futures(tgrid.m * tgrid.n);
  for (int i = 0; i < c_ini_futures.size(); ++i) {
    c_ini_futures[i] = hpx::make_ready_future();
  }

  auto tile_len = [](int tcoord, int tgrid, int len, int tile) {
    int tile_rem = len % tile;
    if (tcoord == tgrid - 1 && tile_rem != 0)
      return tile_rem;
    return tile;
  };
  for (int i = 0; i < tgrid.m; ++i) {
    for (int j = 0; j < tgrid.n; ++j) {
      for (int k = 0; k < tgrid.k; ++k) {
        int len_m = tile_len(i, tgrid.m, lmat.m, tile.m);
        int len_n = tile_len(j, tgrid.n, lmat.n, tile.n);
        int len_k = tile_len(k, tgrid.k, lmat.k, tile.k);

        int a_offset = a_span.idx(k * tile.k, i * tile.m);
        int b_offset = b_span.idx(k * tile.k, j * tile.n);
        int c_offset = c_span.idx(i * tile.m, j * tile.n);

        scalar const* a_ptr = a_buffer.data() + a_offset;
        scalar const* b_ptr = b_buffer.data() + b_offset;
        scalar* c_ptr = c_buffer.data() + c_offset;

        int c_tiles_idx = i + j * tgrid.m;
        c_ini_futures[c_tiles_idx] =
            dataflow(unwrapping(hpx::util::annotated_function(tile_gemm<scalar>, "gemm")), 
            // dataflow(unwrapping(tile_gemm<scalar>), 
                    len_m, len_n,
                     len_k, a_ptr, a_span.ld(), b_ptr, b_span.ld(), c_ptr, c_span.ld(),
                     std::move(c_ini_futures[c_tiles_idx]));
      }
    }
  }

  return c_ini_futures;
}

template <typename scalar>
void issue_tile_sends(MPI_Comm comm_cart, span2d loc_pieces_grid, int i_begin_piece, int j_begin_piece,
                      span2d c_ini, span2d pieces_grid, std::vector<int> const& c_split_r,
                      std::vector<int> const& c_split_c, std::vector<int> const& ranks_map,
                      std::vector<int> const& tags_map, std::vector<scalar> const& c_ini_buffer,
                      std::vector<scalar>& snd_buffer, std::vector<MPI_Request>& snd_reqs,
                      hpx::lcos::local::mutex& mt) {
  auto t_begin = std::chrono::high_resolution_clock::now();

  // For each piece at (i, j) within the tile issue a send to the process it belongs.
  //
  for (int j_loc_piece = 0; j_loc_piece < loc_pieces_grid.cols; ++j_loc_piece) {
    for (int i_loc_piece = 0; i_loc_piece < loc_pieces_grid.rows; ++i_loc_piece) {
      int i_piece = i_begin_piece + i_loc_piece;
      int j_piece = j_begin_piece + j_loc_piece;
      int begin_r = c_split_r[i_piece];
      int begin_c = c_split_c[j_piece];

      span2d piece{c_split_r[i_piece + 1] - begin_r, c_split_c[j_piece + 1] - begin_c};

      int nelems = piece.size();
      int c_ini_offset = c_ini.idx(begin_r, begin_c);

      scalar const* c_ptr = c_ini_buffer.data() + c_ini_offset;
      scalar* snd_ptr = snd_buffer.data() + c_ini_offset;

      piece.loop(c_ini, [c_ptr, snd_ptr](int piece_idx, int c_ini_idx) {
        snd_ptr[piece_idx] = c_ptr[c_ini_idx];
      });

      int piece_idx = pieces_grid.idx(i_piece, j_piece);
      int dest_rank = ranks_map[piece_idx];
      auto& snd_req = snd_reqs[piece_idx];
      int tag = tags_map[piece_idx];

      std::lock_guard<hpx::lcos::local::mutex> lk(mt);
      MPI_CALL(MPI_Issend(snd_ptr, nelems, get_mpi_type<scalar>(), dest_rank, tag, comm_cart, &snd_req))
    }
  }
  auto t_end = std::chrono::high_resolution_clock::now();
  t_send += t_end - t_begin;
}

template <typename scalar>
void issue_sends(MPI_Comm comm_cart, span2d c_tile, span2d c_ini, std::vector<int> const& c_split_r,
                 std::vector<int> const& c_split_c, std::vector<int> const& ranks_map,
                 std::vector<int> const& tags_map, std::vector<scalar> const& c_ini_buffer,
                 std::vector<scalar>& snd_buffer, std::vector<hpx::future<void>>& c_ini_futures,
                 std::vector<MPI_Request>& snd_reqs, hpx::lcos::local::mutex& mt) {
  using hpx::dataflow;
  using hpx::util::unwrapping;

  span2d pieces_grid{static_cast<int>(c_split_r.size()) - 1, static_cast<int>(c_split_c.size()) - 1};

  span2d tgrid{(c_ini.rows + c_tile.rows - 1) / c_tile.rows,
               (c_ini.cols + c_tile.cols - 1) / c_tile.cols};

  for (int j_tile = 0; j_tile < tgrid.cols; ++j_tile) {
    for (int i_tile = 0; i_tile < tgrid.rows; ++i_tile) {
      int idx_tile = tgrid.idx(i_tile, j_tile);

      int len_tile_i = tile_len(i_tile, tgrid.rows, c_ini.rows, c_tile.rows);
      int len_tile_j = tile_len(j_tile, tgrid.cols, c_ini.cols, c_tile.cols);

      int i_delim = i_tile * c_tile.rows;
      int j_delim = j_tile * c_tile.cols;

      int i_begin_piece =
          std::distance(c_split_r.begin(), std::find(c_split_r.begin(), c_split_r.end(), i_delim));
      int j_begin_piece =
          std::distance(c_split_c.begin(), std::find(c_split_c.begin(), c_split_c.end(), j_delim));

      int i_end_piece = std::distance(c_split_r.begin(), std::find(c_split_r.begin(), c_split_r.end(),
                                                                   i_delim + len_tile_i));
      int j_end_piece = std::distance(c_split_c.begin(), std::find(c_split_c.begin(), c_split_c.end(),
                                                                   j_delim + len_tile_j));

      span2d loc_pieces_grid{i_end_piece - i_begin_piece, j_end_piece - j_begin_piece};

      c_ini_futures[idx_tile] =
          dataflow(unwrapping(hpx::util::annotated_function(issue_tile_sends<scalar>, "send")),
          // dataflow(unwrapping(issue_tile_sends<scalar>),
                   comm_cart, loc_pieces_grid, i_begin_piece, j_begin_piece, c_ini, pieces_grid,
                   std::cref(c_split_r), std::cref(c_split_c), std::cref(ranks_map), std::cref(tags_map),
                   std::cref(c_ini_buffer), std::ref(snd_buffer), std::ref(snd_reqs), std::ref(mt),
                   std::move(c_ini_futures[idx_tile]));
    }
  }
}

template <typename scalar>
std::vector<MPI_Request> issue_recvs(MPI_Comm comm_cart, int num_procs,
                                     std::vector<int> const& slab_split_r,
                                     std::vector<int> const& slab_split_c,
                                     std::vector<scalar>& rcv_buffer) {
  // Define a grid for pieces in slabs
  //
  span2d slabs_grid{static_cast<int>(slab_split_r.size()) - 1,
                    static_cast<int>(slab_split_c.size()) - 1};

  std::vector<MPI_Request> rcv_reqs(num_procs * slabs_grid.size());

  // For each piece at (i, j) issue a receive from all processes.
  //
  int rcv_offset = 0;
  int rcv_req_idx = 0;
  auto recv_f = [&](int i, int j) {
    int nelems = (slab_split_c[j + 1] - slab_split_c[j]) * (slab_split_r[i + 1] - slab_split_r[i]);
    int tag = slabs_grid.idx(i, j);

    for (int src_rank = 0; src_rank < num_procs; ++src_rank) {
      scalar* rcv_ptr = rcv_buffer.data() + rcv_offset;
      auto& rcv_req = rcv_reqs[rcv_req_idx];

      MPI_CALL(MPI_Irecv(rcv_ptr, nelems, get_mpi_type<scalar>(), src_rank, tag, comm_cart, &rcv_req))

      rcv_offset += nelems;
      ++rcv_req_idx;
    }
  };
  slabs_grid.loop(recv_f);

  return rcv_reqs;
}

template <typename scalar>
void assemble_rcv_data(int num_procs, std::vector<int> const& slab_split_r,
                       std::vector<int> const& slab_split_c, span2d c_fin,
                       std::vector<scalar> const& rcv_buffer, std::vector<scalar>& c_fin_buffer) {
  // Define a grid for pieces in slabs
  //
  span2d slabs_grid{static_cast<int>(slab_split_r.size()) - 1,
                    static_cast<int>(slab_split_c.size()) - 1};

  // For each piece at (i, j) assemble received data from all processes.
  //
  int rcv_offset = 0;
  auto assemble_f = [&](int i, int j) {
    int begin_r = slab_split_r[i];
    int begin_c = slab_split_c[j];
    span2d piece{slab_split_r[i + 1] - begin_r, slab_split_c[j + 1] - begin_c};

    int offset_c = c_fin.idx(begin_r, begin_c);
    scalar* c_ptr = c_fin_buffer.data() + offset_c;

    for (int src_rank = 0; src_rank < num_procs; ++src_rank) {
      scalar const* rcv_ptr = rcv_buffer.data() + rcv_offset;

      piece.loop(c_fin, [c_ptr, rcv_ptr](int piece_idx, int c_fin_idx) {
        c_ptr[c_fin_idx] += rcv_ptr[piece_idx];
      });

      rcv_offset += piece.size();
    }
  };
  slabs_grid.loop(assemble_f);
}

template <typename scalar>
void print_cfin_sum(MPI_Comm comm_cart, std::vector<scalar> const& c_fin_buffer) {
  scalar local_sum = 0;
  for (auto el : c_fin_buffer) {
    local_sum += el;
  }

  scalar global_sum;
  MPI_Allreduce(&local_sum, &global_sum, 1, get_mpi_type<scalar>(), MPI_SUM, comm_cart);

  int rank = get_proc_rank(comm_cart);
  if (rank == 0) {
    printf("sum=%.6f\n", global_sum);
  }
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
  // using scalar_t = double;
  using clock_t = std::chrono::high_resolution_clock;
  using seconds_t = std::chrono::duration<double>;

  //   // clang-format off
  // desc.add_options()
  //         ("mnk",       po::value<std::array<int,3>>()->default_value({100, 100, 10000}), "mnk")
  //         ("tile_mnk",  po::value<std::array<int,3>>()->default_value({ 64,  64,    64}), "Tile mnk dimensions.")
  //         ("proc_grid", po::value<std::array<int,2>>()->default_value({  1,   1}),        "Process grid")
  //         ("blk_dims",  po::value<std::array<int,2>>()->default_value({ 32,  32}),        "Block sizes.")
  // ;
  // // clang-format on

  // po::variables_map vm;
  // po::store(po::parse_command_line(argc, argv, desc), vm);
  // po::notify(vm);

  // std::array<int, 3> mnk = vm["mnk"].as<std::array<int, 3>>();
  // std::array<int, 3> tile_mnk = vm["tile_mnk"].as<std::array<int, 3>>();
  // std::array<int, 2> proc_grid = vm["proc_grid"].as<std::array<int, 2>>();
  // std::array<int, 2> blk_dims = vm["blk_dims"].as<std::array<int, 2>>();

  // int m = mnk[0];
  // int n = mnk[1];
  // int k = mnk[2];
  // int tile_m = tile_mnk[0];
  // int tile_n = tile_mnk[1];
  // int tile_k = tile_mnk[2];
  // int p_r = proc_grid[0];
  // int p_c = proc_grid[1];
  // int bs_r = blk_dims[0];
  // int bs_c = blk_dims[1];

  // Input
  //
  // clang-format off
  triplet gmat{
      std::stoi(argv[2]),
      std::stoi(argv[3]),
      std::stoi(argv[4])
  };
  triplet tile{
      std::stoi(argv[5]),
      std::stoi(argv[6]),
      std::stoi(argv[7])
  };
  span2d pgrid{
      std::stoi(argv[8]),
      std::stoi(argv[9])
  };
  span2d blk{
      std::stoi(argv[10]),
      std::stoi(argv[11])
  };
  // clang-format on

  MPI_Comm comm_cart = init_comm_cart(pgrid);
  std::array<int, 2> pcoords = get_proc_coords(comm_cart);
  int rank = get_proc_rank(comm_cart);
  int num_procs = pgrid.rows * pgrid.cols;

  // Local distribution of A and B. Only the `k` dimension is split. In SIRIUS, `k_loc` is approximately
  // equally distributed. `k_loc` coincides with `lld` for `A` and `B`. If there is a remainder,
  // distributed it across ranks starting from the `0`-th.
  //
  triplet lmat{gmat.m, gmat.n, gmat.k / num_procs + ((rank < gmat.k % num_procs) ? 1 : 0)};

  // Checks
  //
  check_num_procs(num_procs);

  if (tile.m > lmat.m)
    throw std::invalid_argument("[ERROR] tile_m > m");
  if (tile.n > lmat.n)
    throw std::invalid_argument("[ERROR] tile_n > n");
  if (tile.k > lmat.k)
    throw std::invalid_argument("[ERROR] tile_k > k_loc");

  if (argc != 12)
    throw std::invalid_argument("[ERROR] 10 arguments are expected!");

  // Delimiters descibing how C is split locally and globally along columns and rows.
  //
  std::vector<int> c_split_r = splits(lmat.m, blk.rows, tile.m);
  std::vector<int> c_split_c = splits(lmat.n, blk.cols, tile.n);
  std::vector<int> slab_split_r = slab_splits(c_split_r, blk.rows, pgrid.rows, pcoords[0]);
  std::vector<int> slab_split_c = slab_splits(c_split_c, blk.cols, pgrid.cols, pcoords[1]);

  // Local span of C initial and final layouts
  //
  span2d c_tile{tile.m, tile.n};
  span2d c_ini{lmat.m, lmat.n};
  span2d c_fin{slab_split_r.back(), slab_split_c.back()};

  // Data for A, B:
  //
  // - all buffers are stored in column-major layout
  // - values in `a_buffer` and `b_buffer` are irrelevant
  // - `c_ini` has to be initialized to zero (for accumulation)
  // - `c_fin` is the local portion of the 2D block cyclic distribution
  // - `c_ini` and `c_fin` have to be initialized to 0.
  //
  std::vector<scalar_t> a_buffer(lmat.k * lmat.m, 1);
  std::vector<scalar_t> b_buffer(lmat.k * lmat.n, 1);
  std::vector<scalar_t> c_ini_buffer(c_ini.size(), 0);
  std::vector<scalar_t> c_fin_buffer(c_fin.size(), 0);

  // Comm buffers
  //
  std::vector<scalar_t> snd_buffer(c_ini.size());
  std::vector<scalar_t> rcv_buffer(num_procs * c_fin.size());
  std::vector<int> ranks_map = init_ranks_map(c_split_r, c_split_c, pgrid, blk);
  std::vector<int> tags_map = init_tags_map(num_procs, ranks_map);
  std::vector<MPI_Request> snd_reqs((c_split_r.size() - 1) * (c_split_c.size() - 1));
  hpx::lcos::local::mutex mt;

  // 1. Local multiply
  // 3. Issue receives
  // 2. Issue sends
  // 4. Wait for all data to be received.
  // 5. Assemble received data.
  // 6. Wait until all data is sent
  //
  constexpr int num_iters = 4;
  for (int i = 0; i <= num_iters; ++i) {
    auto t_start = clock_t::now();

    auto t_start_gemm = clock_t::now();
    auto c_ini_futures = local_gemm(lmat, tile, a_buffer, b_buffer, c_ini_buffer);
    auto t_end_gemm = clock_t::now();

    auto t_start_recv = clock_t::now();
    std::vector<MPI_Request> rcv_reqs =
        issue_recvs(comm_cart, num_procs, slab_split_r, slab_split_c, rcv_buffer);
    auto t_end_recv = clock_t::now();

    auto t_start_send = clock_t::now();
    issue_sends(comm_cart, c_tile, c_ini, c_split_r, c_split_c, ranks_map, tags_map, c_ini_buffer,
                snd_buffer, c_ini_futures, snd_reqs, mt);
    auto t_end_send = clock_t::now();

    auto t_start_hpx_wait = clock_t::now();
    hpx::wait_all(c_ini_futures);
    auto t_end_hpx_wait = clock_t::now();

    auto t_start_recv_wait = clock_t::now();
    MPI_Waitall(static_cast<int>(rcv_reqs.size()), rcv_reqs.data(), MPI_STATUS_IGNORE);
    auto t_end_recv_wait = clock_t::now();

    auto t_start_assemble = clock_t::now();
    assemble_rcv_data(num_procs, slab_split_r, slab_split_c, c_fin, rcv_buffer, c_fin_buffer);
    auto t_end_assemble = clock_t::now();

    auto t_start_send_wait = clock_t::now();
    MPI_Waitall(static_cast<int>(snd_reqs.size()), snd_reqs.data(), MPI_STATUS_IGNORE);
    auto t_end_send_wait = clock_t::now();

    auto t_end = clock_t::now();

    if (rank == 0 && i != 0) {
      auto t_tot_dur = seconds_t(t_end - t_start).count();
      auto t_gemm_dur = seconds_t(t_end_gemm - t_start_gemm).count();
      auto t_recv_dur = seconds_t(t_end_recv - t_start_recv).count();
      auto t_send_dur = seconds_t(t_end_send - t_start_send).count();
      auto t_hpx_wait_dur = seconds_t(t_end_hpx_wait - t_start_hpx_wait).count();
      auto t_recv_wait_dur = seconds_t(t_end_recv_wait - t_start_recv_wait).count();
      auto t_assemble_dur = seconds_t(t_end_assemble - t_start_assemble).count();
      auto t_send_wait_dur = seconds_t(t_end_send_wait - t_start_send_wait).count();
      printf("t_tot[s]=%.5f ", t_tot_dur);
      printf("t_gemm/t_send=%.5f ", t_gemm.count() / t_send.count());
      //      printf("t_gemm[s]=%.5f  t_recv[s]=%.5f t_send[s]=%.5f t_hpx_wait[s]=%.5f
      //      t_recv_wait[s]=%.5f t_assemble[s]=%.5f t_send_wait[s]=%.5f     |    ",
      //             t_gemm_dur, t_recv_dur, t_send_dur, t_hpx_wait_dur, t_recv_wait_dur, t_assemble_dur,
      //             t_send_wait_dur);
      printf("m=%d n=%d k=%d tile_m=%d tile_n=%d tile_k=%d pi=%d pj=%d bi=%d bj=%d \n", gmat.m, gmat.n,
             gmat.k, tile.m, tile.n, tile.k, pgrid.rows, pgrid.cols, blk.rows, blk.cols);
    }
  }

  // Simple check
  //
  // print_cfin_sum(comm_cart, c_fin_buffer);

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
  auto ret = hpx::threads::run_as_hpx_thread(tsgemm_main, argc, argv);

  // Stop HPX & MPI
  //
  hpx::stop();
  MPI_Finalize();

  return ret;
}
