#pragma once

#include <cuda_runtime.h>

namespace bf16_c500_tk_cute_local::cute_tk::kernel {

template <typename BodyPolicy, int Threads, typename T, typename Tc,
          typename Tscal, bool IsBetaZero, bool HasOneDimBias,
          typename Pattern>
__global__ void gemm_pattern_kernel(const void *A, const void *B, void *C,
                                    int M, int N, int K, int lda, int ldb,
                                    int ldc, Tscal alpha, Tscal beta,
                                    const void *bias) {
    BodyPolicy::template run<T, Tc, Tscal, IsBetaZero, HasOneDimBias, Pattern>(
        A, B, C, M, N, K, lda, ldb, ldc, alpha, beta, bias, blockIdx.x,
        blockIdx.y);
}

template <typename BodyPolicy, int Threads, typename T, typename Tc,
          typename Tscal, bool IsBetaZero, bool HasOneDimBias,
          typename Pattern>
inline void launch_gemm_pattern_kernel(dim3 grid_dim, const void *A,
                                       const void *B, void *C, int M, int N,
                                       int K, int lda, int ldb, int ldc,
                                       Tscal alpha, Tscal beta,
                                       const void *bias) {
    gemm_pattern_kernel<BodyPolicy, Threads, T, Tc, Tscal, IsBetaZero,
                        HasOneDimBias, Pattern>
        <<<grid_dim, Threads>>>(A, B, C, M, N, K, lda, ldb, ldc, alpha, beta,
                                bias);
}

} // namespace bf16_c500_tk_cute_local::cute_tk::kernel
