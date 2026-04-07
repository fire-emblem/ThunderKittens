#pragma once

#include <cuda_runtime.h>

#include "../contracts/layout_contract.cuh"
#include "../contracts/stage_contract.cuh"
#include "../contracts/tile_contract.cuh"
#include "../host/layout_traits.cuh"
#include "../kernel/layoutc_mainloop.cuh"

namespace bf16_c500_tk_local::families {

struct bf16_layoutc_128x128x128_stage4 {
    using tile = contracts::tile_contract;
    using stage = contracts::stage_contract;
    using layout = contracts::layout_contract;
    using host_layout = host::layoutc_host_traits;

    static constexpr const char *family_name = "tk_local_bf16_layoutc_128x128x128_stage4";
    static constexpr float alpha = 1.0f;
    static constexpr float beta = 0.0f;
    static constexpr bool requires_zero_init = false;

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

} // namespace bf16_c500_tk_local::families
