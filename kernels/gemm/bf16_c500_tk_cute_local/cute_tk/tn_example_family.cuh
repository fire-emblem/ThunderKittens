#pragma once

#include <cuda_runtime.h>

#include "composition/family_pattern.cuh"
#include "composition/gemm_kernel_template.cuh"
#include "primitives/structure/geometry_atom.cuh"
#include "policies.cuh"
#include "primitives/structure/stage_layout_atom.cuh"
#include "tn_example_skeleton.cuh"

namespace bf16_c500_tk_cute_local::cute_tk::families {

template <typename GeometryAtom,
          typename SchedulePolicy = ::bf16_c500_tk_cute_local::cute_tk::tn_example_stage4_schedule,
          typename TileShape = ::bf16_c500_tk_cute_local::cute_tk::tile_128x128x128,
          typename StageLayoutAtom = ::bf16_c500_tk_cute_local::cute_tk::default_stage_layout_atom>
struct swizzled_tn_family
    : ::bf16_c500_tk_cute_local::cute_tk::family_pattern<
          ::bf16_c500_tk_cute_local::cute_tk::swizzled_tn_semantic_tag,
          TileShape, GeometryAtom, SchedulePolicy, StageLayoutAtom> {
    using pattern =
        ::bf16_c500_tk_cute_local::cute_tk::family_pattern<
            ::bf16_c500_tk_cute_local::cute_tk::swizzled_tn_semantic_tag,
            TileShape, GeometryAtom, SchedulePolicy, StageLayoutAtom>;
    using geometry_atom = typename pattern::geometry_atom;
    using host_layout = typename pattern::host_layout;
    using schedule_policy = typename pattern::schedule_policy;
    using tile_shape = typename pattern::tile_shape;
    using stage_layout_atom = typename pattern::stage_layout_atom;
    static constexpr const char *family_name = "cute_tk_swizzled_tn_generic";
    static constexpr float alpha = 1.0f;
    static constexpr float beta = 0.0f;
    static constexpr bool requires_zero_init = false;

    static inline dim3 grid(int m, int n) {
        return dim3((m + tile_shape::tile_m - 1) / tile_shape::tile_m,
                    (n + tile_shape::tile_n - 1) / tile_shape::tile_n);
    }

    static inline bool supports_runtime_shape(int m, int n, int k) {
        return m > 0 && n > 0 && k > 0 && (k % tile_shape::tile_k) == 0;
    }

    template <typename T, typename Tc, typename Tscal, bool IsBetaZero,
              bool HasOneDimBias>
    static inline void launch(dim3 grid_dim, const void *a, const void *b,
                              void *c, int m, int n, int k, int lda, int ldb,
                              int ldc, Tscal alpha_value, Tscal beta_value,
                              const void *bias = nullptr) {
        ::bf16_c500_tk_cute_local::cute_tk::kernel::
            launch_gemm_pattern_kernel<
                ::bf16_c500_tk_cute_local::cute_tk::kernel::swizzled_tn_stage4_body,
                256, T, Tc, Tscal, IsBetaZero, HasOneDimBias, pattern>(
                grid_dim, a, b, c, m, n, k, lda, ldb, ldc, alpha_value,
                beta_value, bias);
    }
};

struct swizzled_tn_tile128x128x128_stage4_family
    : swizzled_tn_family<::bf16_c500_tk_cute_local::cute_tk::swizzled_tn_layout_atom,
                         ::bf16_c500_tk_cute_local::cute_tk::tn_example_stage4_schedule,
                         ::bf16_c500_tk_cute_local::cute_tk::tile_128x128x128> {
    static constexpr const char *family_name =
        "cute_tk_swizzled_tn_tile128x128x128_stage4";
};




} // namespace bf16_c500_tk_cute_local::cute_tk::families

namespace bf16_c500_tk_cute_local::cute_tk {

using swizzled_tn_tile128x128x128_stage4_family_t =
    ::bf16_c500_tk_cute_local::cute_tk::families::
        swizzled_tn_tile128x128x128_stage4_family;

} // namespace bf16_c500_tk_cute_local::cute_tk
