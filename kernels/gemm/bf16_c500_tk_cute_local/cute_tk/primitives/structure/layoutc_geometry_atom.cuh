#pragma once

// Backward compatibility layer - delegates to kernel/gemm/geometry.cuh
// This file will be removed after migration is complete

#include "../../kernel/gemm/geometry.cuh"

namespace bf16_c500_tk_cute_local::cute_tk::primitives {

// Legacy alias - use kernel::gemm::column_major_c_geometry_t instead
template <typename ALdgType, typename BLdgType, typename ALdsType,
          typename BLdsType, typename T>
using layoutc_stage_geometry = ::bf16_c500_tk_cute_local::primitives::stage_geometry_t<
    ALdgType, BLdgType, ALdsType, BLdsType>;

// Legacy factory function
template <typename ALdgType, typename BLdgType, typename ALdsType,
          typename BLdsType, typename T>
__device__ __forceinline__ auto
make_layoutc_stage_geometry(int tid, int lane, int slot, int lda, int n) {
    return ::bf16_c500_tk_cute_local::kernel::gemm::column_major_c_geometry_t::make<
        ALdgType, BLdgType, ALdsType, BLdsType>(tid, lane, slot, lda, n);
}

} // namespace bf16_c500_tk_cute_local::cute_tk::primitives

// Backward compatibility: alias in kernel namespace
namespace bf16_c500_tk_local::kernel {
template <typename ALdgType, typename BLdgType, typename ALdsType,
          typename BLdsType, typename T>
using layoutc_stage_geometry =
    ::bf16_c500_tk_cute_local::primitives::stage_geometry_t<
        ALdgType, BLdgType, ALdsType, BLdsType>;

template <typename ALdgType, typename BLdgType, typename ALdsType,
          typename BLdsType, typename T>
__device__ __forceinline__ auto
make_layoutc_stage_geometry(int tid, int lane, int slot, int lda, int n) {
    return ::bf16_c500_tk_cute_local::kernel::gemm::column_major_c_geometry_t::make<
        ALdgType, BLdgType, ALdsType, BLdsType>(tid, lane, slot, lda, n);
}
} // namespace bf16_c500_tk_local::kernel
