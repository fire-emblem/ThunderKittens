#pragma once

#include "../kernel/layoutc_mainloop.cuh"

namespace bf16_c500_tk_cute_local::cute_tk::kernel {

template <typename T, typename Tc, typename Tscal, bool IsBetaZero,
          bool HasOneDimBias>
__global__ void cute_tk_bf16_continuousc_128x128x128_stage4(
    const void *A, const void *B, void *C, int M, int N, int K, int lda,
    int ldb, int ldc, Tscal alpha, Tscal beta, const void *bias) {
    ::bf16_c500_tk_local::kernel::tk_local_b16_128x128x128_stage4_device<
        T, Tc, Tscal, IsBetaZero, HasOneDimBias, true>(
        A, B, C, M, N, K, lda, ldb, ldc, alpha, beta, bias, blockIdx.x,
        blockIdx.y);
}

} // namespace bf16_c500_tk_cute_local::cute_tk::kernel
