#pragma once

#include <cuda_runtime.h>

#include "../host/layout_traits.cuh"
#include "composition/family_pattern.cuh"
#include "layoutc_square_candidates.cuh"
#include "square_tt_tile256x256x64_skeleton.cuh"

namespace bf16_c500_tk_cute_local::cute_tk::families {

template <typename TileShape, typename StagePolicy>
struct square_tt_tile256x256x64_family
    : ::bf16_c500_tk_cute_local::cute_tk::family_pattern<
          ::bf16_c500_tk_cute_local::cute_tk::square_tt_semantic_tag,
          TileShape, ::bf16_c500_tk_cute_local::cute_tk::layoutc_layout_atom,
          ::bf16_c500_tk_cute_local::cute_tk::layoutc_stage4_schedule,
          ::bf16_c500_tk_cute_local::cute_tk::default_stage_layout_atom> {
    using host_layout = ::bf16_c500_tk_local::host::square_tt_host_traits;
    static constexpr const char *family_name =
        "cute_tk_square_tt_tile256x256x64_stage4";
    static constexpr float alpha = 1.0f;
    static constexpr float beta = 0.0f;
    static constexpr bool requires_zero_init = false;
    static constexpr bool implemented = true;

    static_assert(TileShape::tile_m == 256 && TileShape::tile_n == 256 &&
                      TileShape::tile_k == 64,
                  "square_tt_tile256x256x64_family requires 256x256x64 tile");
    static_assert(StagePolicy::stage_count == 4,
                  "square_tt_tile256x256x64_family currently supports only stage4");

    static inline dim3 grid(int m, int n) {
        return dim3((n + TileShape::tile_n - 1) / TileShape::tile_n,
                    (m + TileShape::tile_m - 1) / TileShape::tile_m);
    }

    static inline bool supports_runtime_shape(int m, int n, int k) {
        return (m == 256 && n == 256 && k == 64) ||
               (m == 2048 && n == 2048 && k == 2048) ||
               (m == 4096 && n == 4096 && k == 4096);
    }

    template <typename T, typename Tc, typename Tscal, bool IsBetaZero,
              bool HasOneDimBias>
    static inline void launch(dim3 grid_dim, const void *a, const void *b,
                              void *c, int m, int n, int k, int lda, int ldb,
                              int ldc, Tscal alpha_value, Tscal beta_value,
                              const void *bias = nullptr) {
        ::bf16_c500_tk_cute_local::cute_tk::kernel::
            cute_tk_bf16_square_tt_tile256x256x64_stage4<T, Tc, Tscal, IsBetaZero,
                                                     HasOneDimBias>
            <<<grid_dim,
               ::bf16_c500_tk_cute_local::cute_tk::square_tt_tile256x256x64_traits::threads,
               ::bf16_c500_tk_cute_local::cute_tk::square_tt_tile256x256x64_traits::
                   a_smem_double_buffer_bytes>>>(
                a, b, c, m, n, k, lda, ldb, ldc, alpha_value, beta_value, bias);
    }
};

} // namespace bf16_c500_tk_cute_local::cute_tk::families

namespace bf16_c500_tk_cute_local::cute_tk {

using square_tt_tile256x256x64_stage4_family_t =
    ::bf16_c500_tk_cute_local::cute_tk::families::square_tt_tile256x256x64_family<
        tile_256x256x64, stage_4>;

} // namespace bf16_c500_tk_cute_local::cute_tk
