#pragma once

#include <cuda_runtime.h>

#include "layout_atom.cuh"
#include "policies.cuh"
#include "tn_example_skeleton.cuh"

namespace bf16_c500_tk_cute_local::cute_tk::families {

template <typename GeometryAtom,
          typename SchedulePolicy = ::bf16_c500_tk_cute_local::cute_tk::tn_example_stage4_schedule>
struct tn_example_family {
    using geometry_atom = GeometryAtom;
    using host_layout = typename geometry_atom::host_layout;
    using geometry_provider = typename geometry_atom::provider;
    using schedule_policy = SchedulePolicy;
    static constexpr const char *family_name = "cute_tk_tn_example_generic";
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
            hgemm_tn_128x128x128_4m1n8k_256t<T, Tc, Tscal, IsBetaZero,
                                            geometry_provider, schedule_policy>
            <<<grid_dim, 256>>>(a, b, c, m, n, k, lda, ldb, ldc, alpha_value,
                                beta_value);
    }
};

struct tn_example_bf16_128x128x128_stage4_family
    : tn_example_family<::bf16_c500_tk_cute_local::cute_tk::tn_example_swizzled_layout_atom,
                        ::bf16_c500_tk_cute_local::cute_tk::tn_example_stage4_schedule> {
    static constexpr const char *family_name =
        "cute_tk_tn_example_bf16_128x128x128_stage4";
};

struct tn_example_linear_geom_bf16_128x128x128_stage4_family
    : tn_example_family<::bf16_c500_tk_cute_local::cute_tk::tn_example_linear_layout_atom,
                        ::bf16_c500_tk_cute_local::cute_tk::tn_example_stage4_schedule> {
    static constexpr const char *family_name =
        "cute_tk_tn_example_linear_geom_bf16_128x128x128_stage4";

    static inline bool supports_runtime_shape(int m, int n, int k) {
        return (m == 1664 && n == 1024 && k == 16384) ||
               (m == 2048 && n == 2048 && k == 2048) ||
               (m == 4096 && n == 4096 && k == 4096);
    }
};

struct tn_example_conservative_bf16_128x128x128_stage4_family
    : tn_example_family<::bf16_c500_tk_cute_local::cute_tk::tn_example_swizzled_layout_atom,
                        ::bf16_c500_tk_cute_local::cute_tk::tn_example_stage4_conservative_schedule> {
    static constexpr const char *family_name =
        "cute_tk_tn_example_conservative_bf16_128x128x128_stage4";

    static inline bool supports_runtime_shape(int m, int n, int k) {
        return (m == 1664 && n == 1024 && k == 16384) ||
               (m == 2048 && n == 2048 && k == 2048) ||
               (m == 4096 && n == 4096 && k == 4096);
    }
};

struct layoutc_tn_tuned_bf16_128x128x128_stage4_family
    : tn_example_family<::bf16_c500_tk_cute_local::cute_tk::tn_example_swizzled_layout_atom,
                        ::bf16_c500_tk_cute_local::cute_tk::tn_example_stage4_schedule> {
    static constexpr const char *family_name =
        "cute_tk_layoutc_tn_tuned_bf16_128x128x128_stage4";

    static inline bool supports_runtime_shape(int m, int n, int k) {
        return (m == 1664 && n == 1024 && k == 16384) ||
               (m == 2048 && n == 2048 && k == 2048) ||
               (m == 4096 && n == 4096 && k == 4096);
    }
};

} // namespace bf16_c500_tk_cute_local::cute_tk::families

namespace bf16_c500_tk_cute_local::cute_tk {

using tn_example_bf16_stage4_family =
    ::bf16_c500_tk_cute_local::cute_tk::families::
        tn_example_bf16_128x128x128_stage4_family;
using tn_example_linear_geom_bf16_stage4_family =
    ::bf16_c500_tk_cute_local::cute_tk::families::
        tn_example_linear_geom_bf16_128x128x128_stage4_family;
using tn_example_conservative_bf16_stage4_family =
    ::bf16_c500_tk_cute_local::cute_tk::families::
        tn_example_conservative_bf16_128x128x128_stage4_family;
using layoutc_tn_tuned_bf16_stage4_family =
    ::bf16_c500_tk_cute_local::cute_tk::families::
        layoutc_tn_tuned_bf16_128x128x128_stage4_family;

} // namespace bf16_c500_tk_cute_local::cute_tk
