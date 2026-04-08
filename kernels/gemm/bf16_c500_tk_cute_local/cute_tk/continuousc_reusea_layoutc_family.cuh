#pragma once

#include <cstdlib>
#include <iostream>

#include <cuda_runtime.h>

#include "../host/layout_traits.cuh"
#include "continuousc_reusea_skeleton.cuh"
#include "policies.cuh"

namespace bf16_c500_tk_cute_local::cute_tk::families {

template <typename TileShape, typename StagePolicy, int APerWarp, int SplitN,
          int SplitK>
struct continuousc_reusea_layoutc_family {
    using host_layout = ::bf16_c500_tk_local::host::layoutc_host_traits;
    static constexpr const char *family_name =
        "cute_tk_continuousc_reusea_layoutc_n_params";
    static constexpr float alpha = 1.0f;
    static constexpr float beta = 0.0f;
    static constexpr bool requires_zero_init = (SplitK > 1);
    static constexpr int NTile = TileShape::tile_n;
    static constexpr int StageCount = StagePolicy::stage_count;

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
        using schedule_t =
            ::bf16_c500_tk_cute_local::cute_tk::continuousc_reusea_schedule<
                NTile, APerWarp, SplitN, SplitK, StageCount>;
        if (!schedule_t::valid_k_partition(k)) {
            std::cerr << "cute_tk continuousc_reusea_layoutc invalid stage configuration: "
                      << "k=" << k << " stage_count=" << StageCount
                      << " split_k=" << SplitK << std::endl;
            std::exit(1);
        }
        ::bf16_c500_tk_cute_local::cute_tk::kernel::
            cute_tk_continuousc_reusea_n<T, Tc, Tscal, StageCount, NTile, APerWarp,
                                         SplitN, SplitK, IsBetaZero,
                                         HasOneDimBias, false>
            <<<grid_dim, 256>>>(
                reinterpret_cast<T *>(const_cast<void *>(a)),
                reinterpret_cast<T *>(const_cast<void *>(b)),
                reinterpret_cast<Tc *>(c), m, n, k, alpha_value, beta_value,
                reinterpret_cast<Tc *>(const_cast<void *>(bias)));
    }
};

template <int NTile, int APerWarp, int SplitN, int SplitK>
using continuousc_reusea_layoutc_n_params =
    continuousc_reusea_layoutc_family<tile_shape_policy<128, NTile, 128>,
                                      stage_4, APerWarp, SplitN, SplitK>;

} // namespace bf16_c500_tk_cute_local::cute_tk::families
