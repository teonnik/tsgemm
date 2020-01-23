#pragma once

namespace tsgemm {

// Note: accumulate<scalar, 0> is a copy
template <typename scalar, int param>
void accumulate(int rows, int cols, scalar const *in, int ldin, scalar *out,
                int ldout) noexcept {
  for (int j = 0; j < cols; ++j) {
    for (int i = 0; i < rows; ++i) {
      int out_idx = i + j * ldout;
      int in_idx = i + j * ldin;
      out[out_idx] = in[in_idx] + scalar{param} * out[out_idx];
    }
  }
}

} // end namespace tsgemm
