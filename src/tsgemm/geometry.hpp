#pragma once
#include <algorithm>

namespace tsgemm {

// Represents a dimension of length `len` split into segments of length `seg`.
//
struct seg_dim {
  int len;
  int seg;

  // Returns the length of the segment at `seg_index`.
  int seg_len(int seg_idx) const noexcept;

  // Returns the number of segments.
  int num_seg() const noexcept;

  // Returns the `el_index` of the segment at `seg_index`.
  int el_index(int seg_idx) const noexcept; 

  // Returns the index of the segment to which the element belongs.
  int seg_index(int el_idx) const noexcept;

  // The reminder segment is the last segment if non-zero
  int rem_seg() const noexcept;
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
  int next_split_offset(int curr_split_offset) const noexcept;

  // A `slab` is a segment made out of `blk` belonging to the current process.
  // `curr_slab_split` is an element index of a split within the slab.
  int next_slab_split_offset(int slab_split_offset) const noexcept; 

  // Returns the length of the local slab stored at the process
  int slab_len() const noexcept;

  // Returns the coordinate of the process holding the element.
  int el_pcoord(int el_index) const noexcept;

  // Map: slab_el_index -> el_index
  int from_slab_el_index(int slab_el_idx) const noexcept;

  // Map: el_index -> slab_el_index
  int to_slab_el_index(int el_idx) const noexcept;

  // Returns the index of the block holding the element in the slab's frame of
  // reference.
  int slab_blk_index(int el_idx) const noexcept;

  seg_dim blk_dim() const noexcept;

  seg_dim tile_dim() const noexcept;

}; // end struct c_dim

// Column-major map from (rows, cols) to an index
int index_map(int rows, int cols, int ld) noexcept;

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
