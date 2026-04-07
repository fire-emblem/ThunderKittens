#pragma once

#include <cuda_runtime.h>

#include "../kernel/layoutc_mainloop.cuh"
#include "tile_contract.cuh"

namespace bf16_c500_tk_local::contracts {

struct launch_contract {
    using tile = tile_contract;

    static constexpr const char *family_name = "tk_local_bf16_layoutc_128x128x128_stage4";
    static constexpr float alpha = 1.0f;
    static constexpr float beta = 0.0f;

    static inline dim3 grid(int m, int n) {
        return dim3(m / tile::tile_m, n / tile::tile_n);
    }

    template <typename T, typename Tc, typename Tscal, bool IsBetaZero, bool HasOneDimBias>
    static inline void launch(
        dim3 grid, const void *a, const void *b, void *c, int m, int n, int k, int lda, int ldb, int ldc,
        Tscal alpha_value, Tscal beta_value, const void *bias = nullptr) {
        kernel::tk_local_bf16_layoutc_128x128x128_stage4<T, Tc, Tscal, IsBetaZero, HasOneDimBias>
            <<<grid, tile::threads>>>(a, b, c, m, n, k, lda, ldb, ldc, alpha_value, beta_value, bias);
    }
};

} // namespace bf16_c500_tk_local::contracts
