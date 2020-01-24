#pragma once

namespace tsgemm {

// definition
template <typename scalar>
void gemm(const int M, const int N, const int K, const scalar alpha,
          const scalar *A, const int lda, const scalar *B, const int ldb,
          const scalar beta, scalar *C, const int ldc);

} // end namespace tsgemm
