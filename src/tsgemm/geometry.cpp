#include <tsgemm/geometry.hpp>

#include <algorithm>

namespace tsgemm {

int seg_dim::seg_len(int seg_idx) const noexcept {
  return std::min(el_index(seg_idx + 1), len) - el_index(seg_idx);
}

int seg_dim::num_seg() const noexcept { return (len + seg - 1) / seg; }

int seg_dim::el_index(int seg_idx) const noexcept { return seg_idx * seg; }

int seg_dim::seg_index(int el_idx) const noexcept { return el_idx / seg; }

int seg_dim::rem_seg() const noexcept { return len % seg; }

int c_dim::next_split_offset(int curr_split_offset) const noexcept {
  return std::min((curr_split_offset / blk + 1) * blk,
                  (curr_split_offset / tile + 1) * tile);
}

int c_dim::next_slab_split_offset(int slab_split_offset) const noexcept {
  int csplit_offset = from_slab_el_index(slab_split_offset);
  int nsplit_offset = next_split_offset(csplit_offset);
  if (el_pcoord(nsplit_offset) != pcoord)
    nsplit_offset += (nproc - 1) * blk;
  return to_slab_el_index(nsplit_offset);
}

int c_dim::slab_len() const noexcept {
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

int c_dim::el_pcoord(int el_index) const noexcept {
  return (el_index / blk) % nproc;
}

int c_dim::from_slab_el_index(int slab_el_idx) const noexcept {
  int blk_idx = (slab_el_idx / blk) * nproc + pcoord;
  return blk_idx * blk + slab_el_idx % blk;
}

int c_dim::to_slab_el_index(int el_idx) const noexcept {
  return slab_blk_index(el_idx) * blk + el_idx % blk;
}

int c_dim::slab_blk_index(int el_idx) const noexcept {
  return (el_idx / blk) / nproc;
}

seg_dim c_dim::blk_dim() const noexcept { return seg_dim{len, blk}; }

seg_dim c_dim::tile_dim() const noexcept { return seg_dim{len, tile}; }

int index_map(int rows, int cols, int ld) noexcept { return rows + cols * ld; }

} // end namespace tsgemm
