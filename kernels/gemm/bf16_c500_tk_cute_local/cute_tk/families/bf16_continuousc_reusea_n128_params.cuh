#pragma once

#include <cuda_runtime.h>

#include "../host/layout_traits.cuh"
#include "../../kernel/continuousc_reusea_n128.cuh"

namespace bf16_c500_tk_local::families {

template <int NTile, int APerWarp, int SplitN, int SplitK>
struct bf16_continuousc_reusea_n_params {
    using host_layout = host::continuousc_host_traits;
    static constexpr const char *family_name =
        "tk_local_bf16_continuousc_reusea_n_params";
    static constexpr float alpha = 1.0f;
    static constexpr float beta = 0.0f;
    static constexpr bool requires_zero_init = (SplitK > 1);

    static inline dim3 grid(int m, int n) {
        (void)n;
        return dim3((m / 16 / (256 / 64 * APerWarp)) * SplitN * SplitK);
    }

    template <typename T, typename Tc, typename Tscal, bool IsBetaZero,
              bool HasOneDimBias>
    static inline void launch(dim3 grid_dim, const void *a, const void *b,
                              void *c, int m, int n, int k, int lda, int ldb,
                              int ldc, Tscal alpha_value, Tscal beta_value,
                              const void *bias = nullptr) {
        (void)lda;
        (void)ldb;
        (void)ldc;
        kernel::tk_local_continuousc_reusea_n<T, Tc, Tscal, NTile, APerWarp,
                                              SplitN, SplitK, IsBetaZero,
                                              HasOneDimBias>
            <<<grid_dim, 256>>>(reinterpret_cast<T *>(const_cast<void *>(a)),
                                reinterpret_cast<T *>(const_cast<void *>(b)),
                                reinterpret_cast<Tc *>(c), m, n, k,
                                alpha_value, beta_value,
                                reinterpret_cast<Tc *>(const_cast<void *>(bias)));
    }
};

} // namespace bf16_c500_tk_local::families
