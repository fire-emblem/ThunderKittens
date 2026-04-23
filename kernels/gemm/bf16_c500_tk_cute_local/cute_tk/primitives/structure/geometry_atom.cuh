#pragma once

// Backward compatibility layer - delegates to kernel/gemm/geometry.cuh
// This file will be removed after migration is complete

#include "../layout/stage_geometry.cuh"
#include "../../kernel/gemm/geometry.cuh"
#include "../../host/layout_traits.cuh"
#include "../../host/tn_example_host_traits.cuh"

namespace bf16_c500_tk_cute_local::cute_tk {

// Abstract geometry atom - wraps geometry providers
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

// Concrete geometry providers - kernel-specific
using column_major_c_geometry_provider = ::bf16_c500_tk_cute_local::kernel::gemm::column_major_c_geometry_t;
using continuous_c_geometry_provider = ::bf16_c500_tk_cute_local::kernel::gemm::continuous_c_geometry_t;
using swizzled_tn_geometry_provider = ::bf16_c500_tk_cute_local::kernel::gemm::swizzled_tn_geometry_t;

// Legacy aliases with layoutc/continuousc naming (kernel-specific)
using layoutc_layout_atom =
    geometry_atom<::bf16_c500_tk_local::host::layoutc_host_traits,
                  column_major_c_geometry_provider>;
using continuousc_layout_atom =
    geometry_atom<::bf16_c500_tk_local::host::continuousc_host_traits,
                  continuous_c_geometry_provider>;
using swizzled_tn_layout_atom =
    geometry_atom<::bf16_c500_tk_local::host::tn_example_host_traits,
                  swizzled_tn_geometry_provider>;

} // namespace bf16_c500_tk_cute_local::cute_tk
