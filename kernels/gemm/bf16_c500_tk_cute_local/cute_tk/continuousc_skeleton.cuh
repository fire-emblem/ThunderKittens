#pragma once

#include "../kernel/layoutc_mainloop.cuh"

namespace bf16_c500_tk_cute_local::cute_tk::kernel {

struct continuousc_stage4_body {
    template <typename T, typename Tc, typename Tscal, bool IsBetaZero,
              bool HasOneDimBias, typename Pattern>
    __device__ __forceinline__ static void run(
        const void *A, const void *B, void *C, int M, int N, int K, int lda,
        int ldb, int ldc, Tscal alpha, Tscal beta, const void *bias, int bidx,
        int bidy) {
        ::bf16_c500_tk_local::kernel::tk_local_b16_128x128x128_stage4_device<
            T, Tc, Tscal, IsBetaZero, HasOneDimBias, true>(
            A, B, C, M, N, K, lda, ldb, ldc, alpha, beta, bias, bidx, bidy);
    }
};

} // namespace bf16_c500_tk_cute_local::cute_tk::kernel
