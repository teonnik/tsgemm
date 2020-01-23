#pragma once
#include <algorithm>


namespace tsgemm {

// Represents a dimension of length `len` split into segments of length `seg`.
//
struct seg_dim {
  int len;
  int seg;

  // Returns the length of the segment at `seg_index`.
  int seg_len(int seg_idx) const noexcept {
    return std::min(el_index(seg_idx + 1), len) - el_index(seg_idx);
  }

  // Returns the number of segments.
  int num_seg() const noexcept { return (len + seg - 1) / seg; }

  // Returns the `el_index` of the segment at `seg_index`.
  int el_index(int seg_idx) const noexcept { return seg_idx * seg; }

  // Returns the index of the segment to which the element belongs.
  int seg_index(int el_idx) const noexcept { return el_idx / seg; }

  // The reminder segment is the last segment if non-zero
  int rem_seg() const noexcept { return len % seg; }
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
    return std::min((curr_split_offset / blk + 1) * blk,
                    (curr_split_offset / tile + 1) * tile);
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
    bool last_pcoord = pcoord == ((rem_pcoords == 0) ? nproc : rem_pcoords) - 1;
    bool missing_pcoord_in_rem = rem_pcoords != 0 && pcoord > rem_pcoords - 1;

    if (last_pcoord && rem_len != 0)
      return slab - blk + rem_len;
    if (missing_pcoord_in_rem)
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

  // Returns the index of the block holding the element in the slab's frame of
  // reference.
  int slab_blk_index(int el_idx) const noexcept {
    return (el_idx / blk) / nproc;
  }

  seg_dim blk_dim() const noexcept { return seg_dim{len, blk}; }

  seg_dim tile_dim() const noexcept { return seg_dim{len, tile}; }

}; // end struct c_dim

// Column-major map from (rows, cols) to an index
int index_map(int rows, int cols, int ld) noexcept { return rows + cols * ld; }

// Iterates over all pieces in column major order
template <typename RowFunc, typename ColFunc, typename WorkFunc>
void iterate_pieces(RowFunc &&next_row_split, ColFunc &&next_col_split,
                    int rows_end, int cols_end, WorkFunc &&func) noexcept {
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

} // end namespace tsgemm
