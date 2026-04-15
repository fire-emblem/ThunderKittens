#pragma once

namespace bf16_c500_tk_cute_local::cute_tk::kernel {

template <typename Impl, bool SupportsOneDimBias>
struct tile128_stage4_body_template {
    template <typename T, typename Tc, typename Tscal, bool IsBetaZero,
              bool HasOneDimBias, typename Pattern>
    __device__ __forceinline__ static void run(
        const void *A, const void *B, void *C, int M, int N, int K, int lda,
        int ldb, int ldc, Tscal alpha, Tscal beta, const void *bias, int bidx,
        int bidy) {
        static_assert(Pattern::tile_shape::tile_m == 128 &&
                          Pattern::tile_shape::tile_n == 128 &&
                          Pattern::tile_shape::tile_k == 128,
                      "tile128_stage4_body_template requires a 128x128x128 tile");
        static_assert(Pattern::schedule_policy::stage_count == 4,
                      "tile128_stage4_body_template requires a stage4 schedule");
        static_assert(!HasOneDimBias || SupportsOneDimBias,
                      "this tile128/stage4 body does not support one-dim bias");

        Impl::template run_stage4<T, Tc, Tscal, IsBetaZero, HasOneDimBias,
                                  Pattern>(A, B, C, M, N, K, lda, ldb, ldc,
                                           alpha, beta, bias, bidx, bidy);
    }
};

} // namespace bf16_c500_tk_cute_local::cute_tk::kernel
