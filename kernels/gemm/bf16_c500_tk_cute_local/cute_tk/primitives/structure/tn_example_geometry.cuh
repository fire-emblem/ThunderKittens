#pragma once

// Backward compatibility layer - delegates to kernel/gemm/geometry.cuh
// This file will be removed after migration is complete

#include "../../kernel/gemm/geometry.cuh"

namespace bf16_c500_tk_cute_local::cute_tk::kernel {

// Legacy alias - use swizzled_tn_geometry_t for TN layout
template <typename ALdgType, typename BLdgType, typename ALdsType,
          typename BLdsType>
using tn_example_stage_geometry = ::bf16_c500_tk_cute_local::primitives::swizzled_tn_geometry_t<
    ALdgType, BLdgType, ALdsType, BLdsType>;

// Legacy alias - use swizzled_tn_geometry_t instead
using swizzled_tn_geometry = ::bf16_c500_tk_cute_local::kernel::gemm::swizzled_tn_geometry_t;

} // namespace bf16_c500_tk_cute_local::cute_tk::kernel