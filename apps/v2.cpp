#include <tsgemm/accumulate.hpp>
#include <tsgemm/gemm.hpp>
#include <tsgemm/geometry.hpp>
#include <tsgemm/input.hpp>
#include <tsgemm/mpi_utils.hpp>
#include <tsgemm/sum_global.hpp>

#include <hpx/dataflow.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/resource_partitioner.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/runtime/threads/executors/pool_executor.hpp>

#include <algorithm>
#include <chrono>
#include <complex>
#include <cstdio>
#include <vector>

// Local gemm
//
// - Tiles along the `k` dimension are chained.
// - `A` is transposed and has similar layout to `B`.
// - Tile sizes near borders have to be adjusted: `len_x`
//
template <typename scalar>
void schedule_local_gemm(tsgemm::seg_dim m_dim, tsgemm::seg_dim n_dim,
                         tsgemm::seg_dim k_dim,
                         std::vector<scalar> const &a_buffer,
                         std::vector<scalar> const &b_buffer,
                         std::vector<scalar> &c_buffer,
                         std::vector<hpx::shared_future<void>> &cini_futures) {
  // Futures for all tiles in column major order.

  using hpx::dataflow;
  using hpx::future;
  using hpx::make_ready_future;
  using hpx::util::annotated_function;
  using std::move;
  using tsgemm::gemm;
  using tsgemm::index_map;

  int lda = k_dim.len;
  int ldb = lda;
  int ldc = m_dim.len;

  // iterate over C tiles in column major order
  for (int j = 0; j < n_dim.num_seg(); ++j) {
    for (int i = 0; i < m_dim.num_seg(); ++i) {
      int m_el_idx = m_dim.el_index(i);
      int n_el_idx = n_dim.el_index(j);
      int len_m = m_dim.seg_len(i);
      int len_n = n_dim.seg_len(j);
      int c_offset = index_map(m_el_idx, n_el_idx, ldc);
      scalar *c_ptr = c_buffer.data() + c_offset;
      future<void> cini_fut = make_ready_future();

      for (int k = 0; k < k_dim.num_seg(); ++k) {
        int k_el_idx = k_dim.el_index(k);
        int len_k = k_dim.seg_len(k);
        int a_offset = index_map(k_el_idx, m_el_idx, lda);
        int b_offset = index_map(k_el_idx, n_el_idx, ldb);
        scalar const *a_ptr = a_buffer.data() + a_offset;
        scalar const *b_ptr = b_buffer.data() + b_offset;

        auto gemm_func = annotated_function(
            [=](auto &&) {
              gemm(len_m, len_n, len_k, scalar(1), a_ptr, lda, b_ptr, ldb,
                   scalar(1), c_ptr, ldc);
            },
            "gemm");
        cini_fut = dataflow(gemm_func, cini_fut);
      }
      cini_futures.push_back(move(cini_fut));
    }
  }
}

template <typename scalar>
void schedule_offload_and_send(
    hpx::threads::executors::pool_executor const &executor, MPI_Comm comm_cart,
    tsgemm::c_dim const &rows_dim, tsgemm::c_dim const &cols_dim,
    std::vector<scalar> const &cini_buffer, std::vector<scalar> &send_buffer,
    std::vector<hpx::shared_future<void>> &gemm_futures,
    std::vector<hpx::future<void>> &comm_futures) noexcept {
  using hpx::dataflow;
  using hpx::util::annotated_function;
  using hpx::util::yield_while;
  using std::move;
  using tsgemm::accumulate;
  using tsgemm::index_map;
  using tsgemm::iterate_pieces;

  auto row_split_f = [&rows_dim](int split) {
    return rows_dim.next_split_offset(split);
  };
  auto col_split_f = [&cols_dim](int split) {
    return cols_dim.next_split_offset(split);
  };

  int num_procs = rows_dim.nproc * cols_dim.nproc;

  tsgemm::seg_dim trdim = rows_dim.tile_dim();
  tsgemm::seg_dim tcdim = cols_dim.tile_dim();
  int ld_tgrid = trdim.num_seg();
  int cini_ld = rows_dim.len;

  std::vector<int> tags(num_procs, 0);
  int snd_offset = 0;
  auto iterate_f = [&](int prow, int pcol, int prlen, int pclen) {
    int tidx =
        index_map(trdim.seg_index(prow), tcdim.seg_index(pcol), ld_tgrid);

    // offload metadata
    int send_ld = prlen;
    int cini_offset = index_map(prow, pcol, cini_ld);
    scalar const *cini_ptr = cini_buffer.data() + cini_offset;
    scalar *send_ptr = send_buffer.data() + snd_offset;

    // send metadata
    int num_elems = prlen * pclen;
    int pcoord_r = rows_dim.el_pcoord(prow);
    int pcoord_c = cols_dim.el_pcoord(pcol);
    int dest_rank = tsgemm::get_proc_rank(comm_cart, pcoord_r, pcoord_c);
    int &tag = tags[dest_rank];

    // capture by value to avoid segfault due to lifetime issues
    auto offload_and_send_f = annotated_function(
        [=](auto &&) {
          // offload
          accumulate<scalar, 0>(prlen, pclen, cini_ptr, cini_ld, send_ptr,
                                send_ld);

          // send
          MPI_Request req;
          {
            //          std::lock_guard<hpx::lcos::local::mutex> lk(mt);
            MPI_Isend(send_ptr, num_elems, tsgemm::get_mpi_type<scalar>(),
                      dest_rank, tag, comm_cart, &req);
          }

          // yield if not sent yet
          yield_while([&req] {
            int flag;
            MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
            return flag == 0;
          });

          // manual yield
          //      int flag = 0;
          //      while (flag == 0) {
          //        MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
          //        hpx::this_thread::yield();
          //      }
        },
        "offload/send");

    // wait on the gemm for the tile
    auto offload_fut =
        dataflow(executor, offload_and_send_f, gemm_futures[tidx]);

    comm_futures.push_back(move(offload_fut));
    ++tag;
    snd_offset += num_elems;
  };

  iterate_pieces(row_split_f, col_split_f, rows_dim.len, cols_dim.len,
                 iterate_f);
}

template <typename scalar>
void schedule_recv_and_load(
    hpx::threads::executors::pool_executor const &executor, MPI_Comm comm_cart,
    tsgemm::c_dim const &rows_dim, tsgemm::c_dim const &cols_dim,
    std::vector<scalar> &cfin_buffer, std::vector<scalar> &recv_buffer,
    std::vector<hpx::future<void>> &comm_futures) noexcept {

  using hpx::dataflow;
  using hpx::util::annotated_function;
  using std::move;
  using tsgemm::accumulate;
  using tsgemm::index_map;
  using tsgemm::iterate_pieces;

  auto row_split_f = [&rows_dim](int split) {
    return rows_dim.next_slab_split_offset(split);
  };
  auto col_split_f = [&cols_dim](int split) {
    return cols_dim.next_slab_split_offset(split);
  };

  int num_procs = rows_dim.nproc * cols_dim.nproc;
  int cfin_ld = rows_dim.slab_len();
  int rcv_offset = 0;
  int tag = 0;
  auto iterate_f = [&](int prow, int pcol, int prlen, int pclen) {
    int recv_ld = prlen;
    int num_elems = prlen * pclen;
    int cfin_offset = index_map(prow, pcol, cfin_ld);
    scalar *cfin_ptr = cfin_buffer.data() + cfin_offset;
    scalar *recv_ptr = recv_buffer.data() + rcv_offset;

    auto recv_and_load_func = annotated_function(
        [=] {
          // recv
          auto recv_reqs = std::make_unique<MPI_Request[]>(num_procs);
          {
            //        std::lock_guard<hpx::lcos::local::mutex> lk(mt);
            for (int src_rank = 0; src_rank < num_procs; ++src_rank) {
              auto &req = recv_reqs[src_rank];
              MPI_Irecv(recv_ptr + src_rank * num_elems, num_elems,
                        tsgemm::get_mpi_type<scalar>(), src_rank, tag,
                        comm_cart, &req);
            }
          }

          // yield if not yet received
          hpx::util::yield_while([num_procs, reqs = std::move(recv_reqs)] {
            int flag;
            MPI_Testall(num_procs, reqs.get(), &flag, MPI_STATUS_IGNORE);
            return flag == 0;
          });

          //      int flag = 0;
          //      while (flag == 0) {
          //        MPI_Testall(num_procs, recv_reqs.get(), &flag,
          //        MPI_STATUS_IGNORE); hpx::this_thread::yield();
          //      }

          // load
          for (int src_rank = 0; src_rank < num_procs; ++src_rank) {
            accumulate<scalar, 1>(prlen, pclen, recv_ptr + src_rank * num_elems,
                                  recv_ld, cfin_ptr, cfin_ld);
          }
        },
        "recv/load");

    auto load_fut = dataflow(executor, recv_and_load_func);

    comm_futures.push_back(move(load_fut));
    rcv_offset += num_procs * num_elems;
    ++tag;
  };

  iterate_pieces(row_split_f, col_split_f, rows_dim.slab_len(),
                 cols_dim.slab_len(), iterate_f);
}

// ****************************************************************************************

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
int hpx_main(hpx::program_options::variables_map &vm) {
  using scalar_t = std::complex<double>;
  using clock_t = std::chrono::high_resolution_clock;
  using seconds_t = std::chrono::duration<double>;

  using hpx::threads::thread_priority_default;
  using hpx::threads::thread_priority_high;
  using hpx::threads::executors::pool_executor;

  // 1. Use mpi pool executor
  // 2. Use default pool with high priority executor
  // 3. Use default pool with default priority executor
#if defined(TSGEMM_USE_MPI_POOL)
  std::string pool_name = "mpi";
  auto priority = thread_priority_default;
#elif defined(TSGEMM_USE_PRIORITIES)
  std::string pool_name = "default";
  auto priority = thread_priority_high;
#else
  std::string pool_name = "default";
  auto priority = thread_priority_default;
#endif
  // an executor that can be used to place work on the MPI pool if it is enabled
  auto mpi_executor = pool_executor(pool_name, priority);

  // hpx::threads::thread_priority_high

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

  MPI_Comm comm_cart = tsgemm::init_comm_cart(pgrid_rows, pgrid_cols);
  std::array<int, 2> pcoords = tsgemm::get_proc_coords(comm_cart);
  int rank = tsgemm::get_proc_rank(comm_cart);
  int num_procs = pgrid_rows * pgrid_cols;

  // Checks
  //
  tsgemm::check_num_procs(num_procs);

  if (tile_m > len_m) {
    throw std::invalid_argument("[ERROR] tile_m > m");
  }
  if (tile_n > len_n) {
    throw std::invalid_argument("[ERROR] tile_n > n");
  }
  if (!(tile_m % blk_rows == 0 || blk_rows % tile_m == 0)) {
    throw std::invalid_argument(
        "[ERROR] tile_m and blk_rows should be multiple of each other.");
  }

  if (!(tile_n % blk_cols == 0 || blk_cols % tile_n == 0)) {
    throw std::invalid_argument(
        "[ERROR] tile_n and blk_cols should be multiple of each other.");
  }

  // Local distribution of A and B. Only the `k` dimension is split. In
  // SIRIUS, `k_loc` is approximately equally distributed. `k_loc` coincides
  // with `lld` for `A` and `B`. If there is a remainder, distributed it
  // across ranks starting from the `0`-th.
  //
  int k_loc = len_k / num_procs + ((rank < len_k % num_procs) ? 1 : 0);
  tsgemm::seg_dim k_dim{k_loc, tile_k};
  tsgemm::c_dim m_dim{len_m, blk_rows, tile_m, pgrid_rows, pcoords[0]};
  tsgemm::c_dim n_dim{len_n, blk_cols, tile_n, pgrid_cols, pcoords[1]};

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
  std::vector<scalar_t> cfin_buffer(m_dim.slab_len() * n_dim.slab_len(), 0);

  // Comm buffers
  //
  std::vector<scalar_t> send_buffer(cini_buffer.size());
  std::vector<scalar_t> recv_buffer(num_procs * cfin_buffer.size());

  // Futures
  //
  int num_tiles = m_dim.tile_dim().num_seg() * n_dim.tile_dim().num_seg();
  int seg_m = std::min(tile_m, blk_rows);
  int seg_n = std::min(tile_n, blk_cols);
  int num_pieces = (len_m + seg_m - 1) * (len_n + seg_n - 1) / (seg_m * seg_n);
  std::vector<hpx::shared_future<void>> gemm_futures;
  std::vector<hpx::future<void>> comm_futures;
  gemm_futures.reserve(num_tiles);
  comm_futures.reserve(2 * num_pieces);

  // mutex
  // hpx::lcos::local::mutex mt;

  // Check if too many non-blocking communications are being issued.
  constexpr int max_comms = 1000;
  // TODO: order the communications by sending `num_comm_cols` columns at once
  // TODO: schedule the gemms associated with the columns first
  //  int m_numcols = (len_m + seg_m - 1) / seg_m;
  //  int num_comm_cols = (max_comms + m_numcols - 1) / m_numcols;
  if (num_pieces > max_comms) {
    printf("[WARNING] There are too many pieces! Increase the block size, tile "
           "size or both!");
  }

  // Setup
  if (rank == 0) {
    printf("len mnk    = %d %d %d\n", len_m, len_n, len_k);
    printf("tile mnk   = %d %d %d\n", tile_m, tile_n, tile_k);
    printf("pgrid      = %d %d\n", pgrid_rows, pgrid_cols);
    printf("blk        = %d %d\n", blk_rows, blk_cols);
    printf("k_loc      = %d\n", k_loc);
    printf("num_pieces = %d\n", num_pieces);
  }

  // 0. Reset buffers
  // 1. Schedule multiply
  // 3. Schedule offloads and receives after multiply
  // 2. Schedule sends and loads
  // 4. Wait for all
  //
  constexpr int num_iters = 5;
  for (int i = 0; i < num_iters; ++i) {

    gemm_futures.clear();
    comm_futures.clear();
    std::fill(std::begin(cini_buffer), std::end(cini_buffer), scalar_t{0});
    std::fill(std::begin(cfin_buffer), std::end(cfin_buffer), scalar_t{0});

    auto t0_tot = clock_t::now();

    auto t0_gemm = clock_t::now();
    schedule_local_gemm(m_dim.tile_dim(), n_dim.tile_dim(), k_dim, a_buffer,
                        b_buffer, cini_buffer, gemm_futures);
    auto t1_gemm = clock_t::now();

    auto t0_send = clock_t::now();
    schedule_offload_and_send(mpi_executor, comm_cart, m_dim, n_dim,
                              cini_buffer, send_buffer, gemm_futures,
                              comm_futures);
    auto t1_send = clock_t::now();

    auto t0_recv = clock_t::now();
    schedule_recv_and_load(mpi_executor, comm_cart, m_dim, n_dim, cfin_buffer,
                           recv_buffer, comm_futures);
    auto t1_recv = clock_t::now();

    auto t0_wait = clock_t::now();
    hpx::wait_all(comm_futures);
    auto t1_wait = clock_t::now();

    auto t1_tot = clock_t::now();

    if (rank == 0) {
      // clang-format off
      printf("%d: t_tot  [s] = %.5f\n", i, seconds_t(t1_tot - t0_tot).count());
      //printf("%d: t_gemm [s] = %.5f\n", i, seconds_t(t1_gemm - t0_gemm).count());
      //printf("%d: t_recv [s] = %.5f\n", i, seconds_t(t1_recv - t0_recv).count());
      //printf("%d: t_send [s] = %.5f\n", i, seconds_t(t1_send - t0_send).count());
      //printf("%d: t_wait [s] = %.5f\n", i, seconds_t(t1_wait - t0_wait).count());
      // clang-format on
    }
  }

  // Simple check
  // std::stringstream ss;
  // using tsgemm::sum_global;
  // ss << "cini sum = " << sum_global(comm_cart, cini_buffer) << '\n';
  // ss << "send sum = " << sum_global(comm_cart, send_buffer) << '\n';
  // ss << "recv sum = " << sum_global(comm_cart, recv_buffer) << '\n';
  // ss << "cfin sum = " << sum_global(comm_cart, cfin_buffer) << '\n';
  // if (rank == 0)
  //  std::cout << ss.str() << '\n';

  return hpx::finalize();
}

// Example usage:
//
//   mpirun -np 1 tsgemm --len_m      100  --len_n      100  --len_k  10000
//                       --tile_m      64  --tile_n      64  --tile_k    64
//                       --pgrid_rows   1  --pgrid_cols   1
//                       --blk_rows    32  --blk_cols    32
//
int main(int argc, char **argv) {

  // init MPI
  auto mpi_handler = tsgemm::mpi_init{argc, argv, MPI_THREAD_MULTIPLE};

  // declare options before creating resource partitioner
  auto desc_cmdline = tsgemm::init_desc();

#ifdef TSGEMM_USE_MPI_POOL
  // NB.
  // thread pools must be declared before starting the runtime

  // Create resource partitioner
  hpx::resource::partitioner rp(desc_cmdline, argc, argv);

  // create a thread pool that is not "default" that we will use for MPI work
  rp.create_thread_pool("mpi");

  // add (enabled) PUs on the first core to it
  rp.add_resource(rp.numa_domains()[0].cores()[0].pus(), "mpi");
#endif

  // init HPX
  return hpx::init(desc_cmdline, argc, argv);
}
