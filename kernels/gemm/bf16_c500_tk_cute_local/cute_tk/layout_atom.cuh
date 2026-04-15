#pragma once

#include "../host/layout_traits.cuh"
#include "../host/tn_example_host_traits.cuh"
#include "../kernel/layoutc_geometry.cuh"
#include "tn_example_geometry.cuh"

namespace bf16_c500_tk_cute_local::cute_tk {

template <typename HostLayoutT, typename GeometryProviderT>
struct geometry_atom {
    using host_layout = HostLayoutT;
    using provider = GeometryProviderT;

    template <typename ALdgType, typename BLdgType, typename ALdsType,
              typename BLdsType, typename... Args>
    __host__ __device__ static auto make_stage_geometry(Args... args) {
        return provider::template make<ALdgType, BLdgType, ALdsType, BLdsType>(
            args...);
    }
};

struct current_layoutc_geometry_provider {
    template <typename ALdgType, typename BLdgType, typename ALdsType,
              typename BLdsType>
    __host__ __device__ static auto make(int tid, int lane, int slot, int lda,
                                         int n) {
        return ::bf16_c500_tk_local::kernel::make_layoutc_stage_geometry<
            ALdgType, BLdgType, ALdsType, BLdsType, __maca_bfloat16>(
            tid, lane, slot, lda, n);
    }
};

struct tn_example_swizzled_geometry_provider {
    template <typename ALdgType, typename BLdgType, typename ALdsType,
              typename BLdsType>
    __device__ __forceinline__ static auto make(int tid, int lane, int slot,
                                                int lda, int ldb, int m_a,
                                                int n_b) {
        return ::bf16_c500_tk_cute_local::cute_tk::kernel::
            tn_example_swizzled_geometry::template make<ALdgType, BLdgType,
                                                        ALdsType, BLdsType>(
                tid, lane, slot, lda, ldb, m_a, n_b);
    }
};

struct tn_example_linear_geometry_provider {
    template <typename ALdgType, typename BLdgType, typename ALdsType,
              typename BLdsType>
    __device__ __forceinline__ static auto make(int tid, int lane, int slot,
                                                int lda, int ldb, int m_a,
                                                int n_b) {
        return ::bf16_c500_tk_cute_local::cute_tk::kernel::
            tn_example_linear_geometry::template make<ALdgType, BLdgType,
                                                      ALdsType, BLdsType>(
                tid, lane, slot, lda, ldb, m_a, n_b);
    }
};

using layoutc_layout_atom =
    geometry_atom<::bf16_c500_tk_local::host::layoutc_host_traits,
                  current_layoutc_geometry_provider>;
using continuousc_layout_atom =
    geometry_atom<::bf16_c500_tk_local::host::continuousc_host_traits,
                  current_layoutc_geometry_provider>;
using tn_example_swizzled_layout_atom =
    geometry_atom<::bf16_c500_tk_local::host::tn_example_host_traits,
                  tn_example_swizzled_geometry_provider>;
using tn_example_linear_layout_atom =
    geometry_atom<::bf16_c500_tk_local::host::tn_example_host_traits,
                  tn_example_linear_geometry_provider>;

} // namespace bf16_c500_tk_cute_local::cute_tk
