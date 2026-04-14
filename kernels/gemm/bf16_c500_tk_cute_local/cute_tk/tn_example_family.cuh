#pragma once

#include <cuda_runtime.h>

#include "../host/tn_example_host_traits.cuh"
#include "tn_example_skeleton.cuh"

namespace bf16_c500_tk_cute_local::cute_tk::families {

struct tn_example_bf16_128x128x128_stage4_family {
    using host_layout = ::bf16_c500_tk_local::host::tn_example_host_traits;
    static constexpr const char *family_name =
        "cute_tk_tn_example_bf16_128x128x128_stage4";
    static constexpr float alpha = 1.0f;
    static constexpr float beta = 0.0f;
    static constexpr bool requires_zero_init = false;

    static inline dim3 grid(int m, int n) {
        return dim3((m + 127) / 128, (n + 127) / 128);
    }

    static inline bool supports_runtime_shape(int m, int n, int k) {
        return m > 0 && n > 0 && k > 0 && (k % 128) == 0;
    }

    template <typename T, typename Tc, typename Tscal, bool IsBetaZero,
              bool HasOneDimBias>
    static inline void launch(dim3 grid_dim, const void *a, const void *b,
                              void *c, int m, int n, int k, int lda, int ldb,
                              int ldc, Tscal alpha_value, Tscal beta_value,
                              const void *bias = nullptr) {
        (void)bias;
        static_assert(!HasOneDimBias,
                      "tn_example family does not support one-dim bias");
        ::bf16_c500_tk_cute_local::cute_tk::kernel::
            hgemm_tn_128x128x128_4m1n8k_256t<T, Tc, Tscal, IsBetaZero>
            <<<grid_dim, 256>>>(a, b, c, m, n, k, lda, ldb, ldc, alpha_value,
                                beta_value);
    }
};

} // namespace bf16_c500_tk_cute_local::cute_tk::families

namespace bf16_c500_tk_cute_local::cute_tk {

using tn_example_bf16_stage4_family =
    ::bf16_c500_tk_cute_local::cute_tk::families::
        tn_example_bf16_128x128x128_stage4_family;

} // namespace bf16_c500_tk_cute_local::cute_tk
