#include <tsgemm/gemm.hpp>

#include <complex>
#include <mkl.h>

#define TSGEMM_GEMM_DEF(scalar) \
void gemm (const int M, const int N, const int K, const scalar alpha, \
           const scalar *A, const int lda, const scalar *B, const int ldb, \
           const scalar beta, scalar *C, const int ldc) 

#define TSGEMM_MKL_CALL(func, alpha, beta) \
  func(CBLAS_LAYOUT::CblasColMajor, CBLAS_TRANSPOSE::CblasConjTrans, \
       CBLAS_TRANSPOSE::CblasNoTrans, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

namespace tsgemm {

namespace detail {

// overload set
TSGEMM_GEMM_DEF(float)                { TSGEMM_MKL_CALL(cblas_sgemm,  alpha,  beta) }
TSGEMM_GEMM_DEF(double)               { TSGEMM_MKL_CALL(cblas_dgemm,  alpha,  beta) }
TSGEMM_GEMM_DEF(std::complex<float>)  { TSGEMM_MKL_CALL(cblas_cgemm, &alpha, &beta) } 
TSGEMM_GEMM_DEF(std::complex<double>) { TSGEMM_MKL_CALL(cblas_zgemm, &alpha, &beta) }

} // end namespace detail

// definition
template <typename scalar> TSGEMM_GEMM_DEF(scalar) { detail::gemm(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc); } 

// specializations
template TSGEMM_GEMM_DEF(float);
template TSGEMM_GEMM_DEF(double);
template TSGEMM_GEMM_DEF(std::complex<float>);
template TSGEMM_GEMM_DEF(std::complex<double>);

} // end namespace tsgemm
