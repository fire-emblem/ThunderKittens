#pragma once

#include <cuda_runtime.h>

#include "../contracts/layout_contract.cuh"
#include "../contracts/stage_contract.cuh"
#include "../contracts/tile_contract.cuh"
#include "../host/layout_traits.cuh"
#include "composition/family_pattern.cuh"
#include "primitives/structure/stage_layout_atom.cuh"
#include "layoutc_skeleton.cuh"
#include "policies.cuh"

namespace bf16_c500_tk_cute_local::cute_tk::families {

template <typename TileShape, typename StagePolicy,
          typename GeometryAtom = ::bf16_c500_tk_cute_local::cute_tk::layoutc_layout_atom,
          typename SchedulePolicy = ::bf16_c500_tk_cute_local::cute_tk::layoutc_stage4_schedule,
          typename StageLayoutAtom = ::bf16_c500_tk_cute_local::cute_tk::default_stage_layout_atom>
struct layoutc_family
    : ::bf16_c500_tk_cute_local::cute_tk::family_pattern<
          ::bf16_c500_tk_cute_local::cute_tk::layoutc_semantic_tag, TileShape,
          GeometryAtom, SchedulePolicy, StageLayoutAtom> {
    using pattern =
        ::bf16_c500_tk_cute_local::cute_tk::family_pattern<
            ::bf16_c500_tk_cute_local::cute_tk::layoutc_semantic_tag,
            TileShape, GeometryAtom, SchedulePolicy, StageLayoutAtom>;
    using tile = ::bf16_c500_tk_local::contracts::tile_contract;
    using stage = ::bf16_c500_tk_local::contracts::stage_contract;
    using layout = ::bf16_c500_tk_local::contracts::layout_contract;
    using tile_shape = typename pattern::tile_shape;
    using geometry_atom = typename pattern::geometry_atom;
    using schedule_policy = typename pattern::schedule_policy;
    using stage_layout_atom = typename pattern::stage_layout_atom;
    using host_layout = typename pattern::host_layout;

    static_assert(TileShape::tile_m == 128 && TileShape::tile_n == 128 &&
                      TileShape::tile_k == 128,
                  "cute_tk layoutc_family currently supports only 128x128x128");
    static_assert(StagePolicy::stage_count == 4,
                  "cute_tk layoutc_family currently supports only stage4");

    static constexpr const char *family_name =
        "cute_tk_layoutc_tile128x128x128_stage4";
    static constexpr float alpha = 1.0f;
    static constexpr float beta = 0.0f;
    static constexpr bool requires_zero_init = false;

    static inline dim3 grid(int m, int n) {
        return dim3((m + tile::tile_m - 1) / tile::tile_m,
                    (n + tile::tile_n - 1) / tile::tile_n);
    }

    static inline bool supports_runtime_shape(int m, int n, int k) {
        return m > 0 && n > 0 && (k % tile::tile_k) == 0;
    }

    template <typename T, typename Tc, typename Tscal, bool IsBetaZero,
              bool HasOneDimBias>
    static inline void launch(dim3 grid_dim, const void *a, const void *b,
                              void *c, int m, int n, int k, int lda, int ldb,
                              int ldc, Tscal alpha_value, Tscal beta_value,
                              const void *bias = nullptr) {
        ::bf16_c500_tk_cute_local::cute_tk::kernel::
            cute_tk_bf16_layoutc_tile128x128x128_stage4_family_t<
                T, Tc, Tscal, IsBetaZero, HasOneDimBias,
                pattern><<<grid_dim, tile::threads>>>(
                a, b, c, m, n, k, lda, ldb, ldc, alpha_value, beta_value, bias);
    }
};

using layoutc_tile128x128x128_stage4_family_t =
    layoutc_family<tile_128x128x128, stage_4>;
using layoutc_tile128x128x128_stage4_swizzled_tn_geometry_family_t =
    layoutc_family<tile_128x128x128, stage_4,
                   ::bf16_c500_tk_cute_local::cute_tk::swizzled_tn_layout_atom>;

} // namespace bf16_c500_tk_cute_local::cute_tk::families

namespace bf16_c500_tk_cute_local::cute_tk {

using layoutc_tile128x128x128_stage4_family_t =
    ::bf16_c500_tk_cute_local::cute_tk::families::
        layoutc_tile128x128x128_stage4_family_t;
using layoutc_tile128x128x128_stage4_swizzled_tn_geometry_family_t =
    ::bf16_c500_tk_cute_local::cute_tk::families::
        layoutc_tile128x128x128_stage4_swizzled_tn_geometry_family_t;

} // namespace bf16_c500_tk_cute_local::cute_tk
