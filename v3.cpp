#include <gemm.hpp>
#include <mpi_utils.hpp>

#include <hpx/dataflow.hpp>
#include <hpx/hpx.hpp>
#include <hpx/mpi/mpi_future.hpp>

#include <algorithm>
#include <chrono>
#include <complex>
#include <cstdio>
#include <vector>

// Represents a dimension of length `len` split into segments of length `seg`.
//
struct seg_dim {
  int len;
  int seg;

  // Returns the length of the segment at `seg_index`.
  int seg_len(int seg_idx) const noexcept {
    return el_index(seg_idx + 1) - std::min(el_index(seg_idx), len);
  }

  // Returns the number of segments.
  int num_seg() const noexcept {
    return (len + seg - 1) / seg;
  }

  // Returns the `el_index` of the segment at `seg_index`.
  int el_index(int seg_idx) const noexcept {
    return seg_idx * seg;
  }

  // Returns the index of the segment to which the element belongs.
  int seg_index(int el_idx) const noexcept {
    return el_idx / seg;
  }

  // The reminder segment is the last segment if non-zero
  int rem_seg() const noexcept {
    return len % seg;
  }
};

// 1D block-cyclic distribution with tiles. A dimension of the C matrix.
//
struct c_dim {
  int len;
  int blk;
  int tile;
  int nproc;
  int pcoord;

  // Splits are el_indices where the matrix C is split.
  int next_split_offset(int curr_split_offset) const noexcept {
    return std::min((curr_split_offset / blk + 1) * blk, (curr_split_offset / tile + 1) * tile);
  }

  // A `slab` is a segment made out of `blk` belonging to the current process.
  // `curr_slab_split` is an element index of a split within the slab.
  int next_slab_split_offset(int slab_split_offset) const noexcept {
    int csplit_offset = from_slab_el_index(slab_split_offset);
    int nsplit_offset = next_split_offset(csplit_offset);
    if (el_pcoord(nsplit_offset) != pcoord)
      nsplit_offset += (nproc - 1) * blk;
    return to_slab_el_index(nsplit_offset);
  }

  // Returns the length of the local slab stored at the process
  int slab_len() const noexcept {
    seg_dim blk_dim = seg_dim{len, blk};
    seg_dim proc_dim = seg_dim{blk_dim.num_seg(), nproc};

    int slab = proc_dim.num_seg() * blk;

    int rem_pcoords = proc_dim.rem_seg();
    int rem_len = blk_dim.rem_seg();

    bool last_pcoord = (rem_pcoords == 0) ? pcoord == nproc - 1 : rem_pcoords == pcoord;
    bool missing_pcoord_in_reminder = rem_pcoords != 0 && pcoord > rem_pcoords;

    if (last_pcoord && rem_len != 0)
      return slab - blk + rem_len;

    if (missing_pcoord_in_reminder)
      return slab - blk;

    return slab;
  }

  // Returns the coordinate of the process holding the element.
  int el_pcoord(int el_index) const noexcept {
    return (el_index / blk) % nproc;
  }

  // Map: slab_el_index -> el_index
  int from_slab_el_index(int slab_el_idx) const noexcept {
    int blk_idx = (slab_el_idx / blk) * nproc + pcoord;
    return blk_idx * blk + slab_el_idx % blk;
  }

  // Map: el_index -> slab_el_index
  int to_slab_el_index(int el_idx) const noexcept {
    return slab_blk_index(el_idx) * blk + el_idx % blk;
  }

  // Returns the index of the block holding the element in the slab's frame of reference.
  int slab_blk_index(int el_idx) const noexcept {
    return (el_idx / blk) / nproc;
  }

  seg_dim blk_dim() const noexcept {
    return seg_dim{len, blk};
  }

  seg_dim tile_dim() const noexcept {
    return seg_dim{len, tile};
  }

};  // end struct c_dim

// Column-major map from (rows, cols) to an index
int index_map(int rows, int cols, int ld) noexcept {
  return rows + cols * ld;
}

// Iterates over all pieces in column major order
template <typename RowFunc, typename ColFunc, typename WorkFunc>
void iterate_pieces(RowFunc&& next_row_split, ColFunc&& next_col_split, int rows_end, int cols_end,
                    WorkFunc&& func) noexcept {
  // cs - cols splits
  // rs - rows splits
  for (int cs_begin = 0; cs_begin < cols_end;) {
    int cs_end = std::min(next_col_split(cs_begin), cols_end);
    int cs_len = cs_end - cs_begin;
    for (int rs_begin = 0; rs_begin < rows_end;) {
      int rs_end = std::min(next_row_split(rs_begin), rows_end);
      int rs_len = rs_end - rs_begin;

      // do some work
      func(rs_begin, cs_begin, rs_len, cs_len);

      rs_begin = rs_end;
    }
    cs_begin = cs_end;
  }
}

// Note: accumulate<scalar, 0> is a copy
template <typename scalar, int param>
void accumulate(int rows, int cols, scalar const* in, int ldin, scalar* out, int ldout) noexcept {
  for (int j = 0; j < cols; ++j) {
    for (int i = 0; i < rows; ++i) {
      int out_idx = i + j * ldout;
      int in_idx = i + j * ldin;
      out[out_idx] = in[in_idx] + scalar{param} * out[out_idx];
    }
  }
}

// ****************************************************************************************

// Local gemm
//
// - Tiles along the `k` dimension are chained.
// - `A` is transposed and has similar layout to `B`.
// - Tile sizes near borders have to be adjusted: `len_x`
//
template <typename scalar>
void schedule_local_gemm(seg_dim m_dim, seg_dim n_dim, seg_dim k_dim,
                         std::vector<scalar> const& a_buffer, std::vector<scalar> const& b_buffer,
                         std::vector<scalar>& c_buffer,
                         std::vector<hpx::shared_future<void>>& c_ini_futures) {
  // Futures for all tiles in column major order.

  int lda = k_dim.len;
  int ldb = lda;
  int ldc = m_dim.len;

  for (int i = 0; i < m_dim.num_seg(); ++i) {
    for (int j = 0; j < n_dim.num_seg(); ++j) {
      for (int k = 0; k < k_dim.num_seg(); ++k) {
        int m_el_idx = m_dim.el_index(i);
        int n_el_idx = n_dim.el_index(j);
        int k_el_idx = k_dim.el_index(k);

        int len_m = m_dim.seg_len(i);
        int len_n = n_dim.seg_len(j);
        int len_k = k_dim.seg_len(k);

        int a_offset = index_map(k_el_idx, m_el_idx, lda);
        int b_offset = index_map(k_el_idx, n_el_idx, ldb);
        int c_offset = index_map(m_el_idx, n_el_idx, ldc);

        scalar const* a_ptr = a_buffer.data() + a_offset;
        scalar const* b_ptr = b_buffer.data() + b_offset;
        scalar* c_ptr = c_buffer.data() + c_offset;

        int c_tile_idx = i + j * m_dim.num_seg();
        auto& fut = c_ini_futures[c_tile_idx];
        fut = hpx::dataflow(hpx::util::unwrapping(
                                hpx::util::annotated_function(tsgemm::gemm<scalar>, "gemm")),
                            len_m, len_n, len_k, scalar(1), a_ptr, lda, b_ptr, ldb, scalar(1), c_ptr,
                            ldc, fut);
      }
    }
  }
}

template <typename scalar>
void schedule_offload_and_send(MPI_Comm comm_cart, c_dim const& rows_dim, c_dim const& cols_dim,
                               std::vector<scalar> const& cini_buffer, std::vector<scalar>& send_buffer,
                               std::vector<hpx::shared_future<void>>& gemm_futures,
                               std::vector<hpx::future<void>>& comm_futures) noexcept {
  auto row_split_f = [&rows_dim](int split) { return rows_dim.next_split_offset(split); };
  auto col_split_f = [&cols_dim](int split) { return cols_dim.next_split_offset(split); };

  int num_procs = rows_dim.nproc * cols_dim.nproc;
  std::vector<int> tags(num_procs, 0);

  seg_dim trdim = rows_dim.tile_dim();
  seg_dim tcdim = cols_dim.tile_dim();
  int ld_tgrid = trdim.num_seg();
  int cini_ld = rows_dim.len;

  auto offload_and_send_f = [&](int prow, int pcol, int prlen, int pclen) {
    int tidx = index_map(trdim.seg_index(prow), tcdim.seg_index(pcol), ld_tgrid);

    int send_ld = prlen;
    int offset = index_map(prow, pcol, cini_ld);
    scalar const* cini_ptr = cini_buffer.data() + offset;
    scalar* send_ptr = send_buffer.data() + offset;

    // schedule offload
    auto offload_fut =
        hpx::dataflow(hpx::util::unwrapping(
                          hpx::util::annotated_function(accumulate<scalar, 0>, "offload")),
                      prlen, pclen, cini_ptr, cini_ld, send_ptr, send_ld, gemm_futures[tidx]);

    // schedule send
    int num_elems = prlen * pclen;
    int pcoord_r = rows_dim.el_pcoord(prow);
    int pcoord_c = cols_dim.el_pcoord(pcol);
    int dest_rank = tsgemm::get_proc_rank(comm_cart, pcoord_r, pcoord_c);
    int& tag = tags[dest_rank];
    // printf("%d %d %d %d %d %d %d\n", tidx, prow, pcol, prlen, pclen, dest_rank, tag);

    // capture by value to avoid segfault due to lifetime issues
    auto mpi_launcher = [=](auto&&) {
      return hpx::mpi::async(MPI_Isend, send_ptr, num_elems, tsgemm::get_mpi_type<scalar>(), dest_rank,
                             tag, comm_cart);
    };
    comm_futures.push_back(offload_fut.then(mpi_launcher));
    ++tag;
  };

  iterate_pieces(row_split_f, col_split_f, rows_dim.len, cols_dim.len, offload_and_send_f);
}

template <typename scalar>
void schedule_recv_and_load(MPI_Comm comm_cart, c_dim const& rows_dim, c_dim const& cols_dim,
                            std::vector<scalar>& cfin_buffer, std::vector<scalar>& recv_buffer,
                            std::vector<hpx::future<void>>& comm_futures) noexcept {
  auto row_split_f = [&rows_dim](int split) { return rows_dim.next_slab_split_offset(split); };
  auto col_split_f = [&cols_dim](int split) { return cols_dim.next_slab_split_offset(split); };

  int num_procs = rows_dim.nproc * cols_dim.nproc;
  int cfin_ld = rows_dim.slab_len();
  int rcv_offset = 0;
  int tag = 0;
  auto recv_and_load_f = [&](int prow, int pcol, int prlen, int pclen) {
    int rcv_ld = prlen;
    int num_elems = prlen * pclen;
    int cfin_offset = index_map(prow, pcol, cfin_ld);
    scalar* cfin_ptr = cfin_buffer.data() + cfin_offset;
    for (int src_rank = 0; src_rank < num_procs; ++src_rank) {
      scalar* rcv_ptr = recv_buffer.data() + rcv_offset;
      auto mpi_fut = hpx::mpi::async(MPI_Irecv, rcv_ptr, num_elems, tsgemm::get_mpi_type<scalar>(),
                                     src_rank, tag, comm_cart);
      auto load_fut = hpx::dataflow(hpx::util::unwrapping(
                                        hpx::util::annotated_function(accumulate<scalar, 1>, "load")),
                                    prlen, pclen, rcv_ptr, rcv_ld, cfin_ptr, cfin_ld, mpi_fut);
      comm_futures.push_back(std::move(load_fut));
      // printf("%d %d %d %d %d %d\n", prow, pcol, prlen, pclen, src_rank, tag);

      rcv_offset += num_elems;
    }
    ++tag;
  };

  iterate_pieces(row_split_f, col_split_f, rows_dim.slab_len(), cols_dim.slab_len(), recv_and_load_f);
}

// ****************************************************************************************

template <typename scalar>
void print_cfin_sum(MPI_Comm comm_cart, std::vector<scalar> const& c_fin_buffer) {
  scalar local_sum = 0;
  for (auto el : c_fin_buffer) {
    local_sum += el;
  }

  scalar global_sum;
  MPI_Allreduce(&local_sum, &global_sum, 1, tsgemm::get_mpi_type<scalar>(), MPI_SUM, comm_cart);

  int rank = tsgemm::get_proc_rank(comm_cart);
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
int tsgemm_main(hpx::program_options::variables_map& vm) {
  //  using scalar_t = std::complex<double>;
  using scalar_t = double;
  using clock_t = std::chrono::high_resolution_clock;
  using seconds_t = std::chrono::duration<double>;

  int len_m = vm["len_m"].as<int>();
  int len_n = vm["len_n"].as<int>();
  int len_k = vm["len_k"].as<int>();
  int tile_m = vm["tile_m"].as<int>();
  int tile_n = vm["tile_n"].as<int>();
  int tile_k = vm["tile_k"].as<int>();
  int pgrid_rows = vm["pgrid_rows"].as<int>();
  int pgrid_cols = vm["pgrid_cols"].as<int>();
  int blk_rows = vm["blk_rows"].as<int>();
  int blk_cols = vm["blk_cols"].as<int>();

  printf("len mnk  = %d %d %d\n", len_m, len_n, len_k);
  printf("tile mnk = %d %d %d\n", tile_m, tile_n, tile_k);
  printf("pgrid    = %d %d\n", pgrid_rows, pgrid_cols);
  printf("blk      = %d %d\n", blk_rows, blk_cols);

  MPI_Comm comm_cart = tsgemm::init_comm_cart(pgrid_rows, pgrid_cols);
  std::array<int, 2> pcoords = tsgemm::get_proc_coords(comm_cart);
  int rank = tsgemm::get_proc_rank(comm_cart);
  int num_procs = pgrid_rows * pgrid_cols;

  // Checks
  //
  tsgemm::check_num_procs(num_procs);

  if (tile_m > len_m)
    throw std::invalid_argument("[ERROR] tile_m > m");
  if (tile_n > len_n)
    throw std::invalid_argument("[ERROR] tile_n > n");

  // Local distribution of A and B. Only the `k` dimension is split. In SIRIUS, `k_loc` is approximately
  // equally distributed. `k_loc` coincides with `lld` for `A` and `B`. If there is a remainder,
  // distributed it across ranks starting from the `0`-th.
  //
  int k_loc = len_k / num_procs + ((rank < len_k % num_procs) ? 1 : 0);
  seg_dim k_dim{k_loc, tile_k};
  seg_dim m_dim{len_m, tile_m};
  seg_dim n_dim{len_n, tile_n};

  // Delimiters descibing how C is split locally and globally along columns and rows.
  //
  c_dim c_rows_dim{len_m, blk_rows, tile_m, pgrid_rows, pcoords[0]};
  c_dim c_cols_dim{len_n, blk_cols, tile_n, pgrid_cols, pcoords[1]};

  // Data for A, B:
  //
  // - all buffers are stored in column-major layout
  // - values in `a_buffer` and `b_buffer` are irrelevant
  // - `c_ini` has to be initialized to zero (for accumulation)
  // - `c_fin` is the local portion of the 2D block cyclic distribution
  // - `c_ini` and `c_fin` have to be initialized to 0.
  //
  std::vector<scalar_t> a_buffer(k_loc * len_m, 1);
  std::vector<scalar_t> b_buffer(k_loc * len_n, 1);
  std::vector<scalar_t> cini_buffer(len_m * len_n, 0);
  std::vector<scalar_t> cfin_buffer(c_rows_dim.slab_len() * c_cols_dim.slab_len(), 0);

  // Comm buffers
  //
  std::vector<scalar_t> send_buffer(cini_buffer.size());
  std::vector<scalar_t> recv_buffer(num_procs * cfin_buffer.size());

  // Futures
  //
  int num_tiles = m_dim.num_seg() * n_dim.num_seg();
  std::vector<hpx::shared_future<void>> gemm_futures(num_tiles);
  for (auto& gemm_fut : gemm_futures) {
    gemm_fut = hpx::make_ready_future();
  }

  std::vector<hpx::future<void>> comm_futures;
  int seg_m = std::min(tile_m, blk_rows);
  int seg_n = std::min(tile_n, blk_cols);
  comm_futures.reserve(4 * len_m * len_n / (seg_m * seg_n));

  // Tell the scheduler that we want to handle mpi in the background
  // here we use the provided hpx::mpi::poll function but a user
  // provided function or lambda may be supplied
  //
  auto const sched = hpx::threads::get_self_id_data()->get_scheduler_base();
  sched->set_user_polling_function(&hpx::mpi::poll);
  sched->add_scheduler_mode(hpx::threads::policies::enable_user_polling);

  // 1. Local multiply
  // 3. Issue receives
  // 2. Issue sends
  // 4. Wait for all data to be received.
  // 5. Assemble received data.
  // 6. Wait until all data is sent
  //
  constexpr int num_iters = 4;
  for (int i = 0; i <= num_iters; ++i) {
    auto t0 = clock_t::now();

    auto t0_gemm = clock_t::now();
    schedule_local_gemm(m_dim, n_dim, k_dim, a_buffer, b_buffer, cini_buffer, gemm_futures);
    auto t1_gemm = clock_t::now();

    auto t0_send = clock_t::now();
    schedule_offload_and_send(comm_cart, c_rows_dim, c_cols_dim, cini_buffer, send_buffer, gemm_futures,
                              comm_futures);
    auto t1_send = clock_t::now();

    auto t0_recv = clock_t::now();
    schedule_recv_and_load(comm_cart, c_rows_dim, c_cols_dim, cfin_buffer, recv_buffer, comm_futures);
    auto t1_recv = clock_t::now();

    auto t0_wait = clock_t::now();
    // FIXME: hangs here
    hpx::wait_all(comm_futures);
    auto t1_wait = clock_t::now();

    auto t1 = clock_t::now();

    if (rank == 0 && i != 0) {
      printf("\n ---- Results ---- \n");
      printf("t_tot  [s] = %.5f\n", seconds_t(t1 - t0).count());
      printf("t_gemm [s] = %.5f\n", seconds_t(t1_gemm - t0_gemm).count());
      printf("t_recv [s] = %.5f\n", seconds_t(t1_recv - t0_recv).count());
      printf("t_send [s] = %.5f\n", seconds_t(t1_send - t0_send).count());
      printf("t_wait [s] = %.5f\n", seconds_t(t1_wait - t0_wait).count());
    }
  }

  // Before exiting, shut down the mpi/user polling loop.
  //
  sched->remove_scheduler_mode(hpx::threads::policies::enable_user_polling);

  // Simple check
  print_cfin_sum(comm_cart, cfin_buffer);

  return hpx::finalize();
}

// Example usage:
//
//   mpirun -np 1 tsgemm --len_m      100  --len_n      100  --len_k  10000
//                       --tile_m      64  --tile_n      64  --tile_k    64
//                       --pgrid_rows   1  --pgrid_cols   1
//                       --blk_rows    32  --blk_cols    32
//
int main(int argc, char** argv) {
  // note: MPI_THREAD_SERIALIZED is no longer enough
  auto mpi_handler = tsgemm::mpi_init{argc, argv, MPI_THREAD_MULTIPLE};  // MPI

  // Input
  // note: has to be in main so that hpx knows about the various options
  namespace po = hpx::program_options;
  po::options_description desc("Allowed options.");

  // clang-format off
  desc.add_options()
     ("len_m",      po::value<int>()->default_value( 100), "m dimension")
     ("len_n",      po::value<int>()->default_value( 100), "n dimension")
     ("len_k",      po::value<int>()->default_value(1000), "k dimension")
     ("tile_m",     po::value<int>()->default_value(  64), "tile m dimension")
     ("tile_n",     po::value<int>()->default_value(  64), "tile n dimension")
     ("tile_k",     po::value<int>()->default_value(  64), "tile k dimension")
     ("pgrid_rows", po::value<int>()->default_value(   1), "process grid rows")
     ("pgrid_cols", po::value<int>()->default_value(   1), "process grid columns")
     ("blk_rows",   po::value<int>()->default_value(  32), "block rows")
     ("blk_cols",   po::value<int>()->default_value(  32), "block columns")
  ;
  // clang-format on

  return hpx::init(tsgemm_main, desc, argc, argv);  // HPX
}
