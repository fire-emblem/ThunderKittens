#pragma once

#include "mxc_builtins.cuh"

#include <type_traits>

namespace bf16_c500_tk_cute_local::arch {

__device__ __forceinline__ float4_native mma_16x16x16_bf16(uint a0, uint a1, uint b0, uint b1, float4_native c) {
    return __builtin_mxc_mma_16x16x16bf16(uint2_native{a0, a1}, uint2_native{b0, b1}, c);
}

__device__ __forceinline__ float4_native mma_16x16x16_f16(uint a0, uint a1, uint b0, uint b1, float4_native c) {
    return __builtin_mxc_mma_16x16x16f16(uint2_native{a0, a1}, uint2_native{b0, b1}, c);
}

template <typename T, bool SwapAB = false>
__device__ __forceinline__ float4_native mma_16x16x16_b16(uint a0, uint a1, uint b0, uint b1, float4_native c) {
    const uint2_native a = SwapAB ? uint2_native{b0, b1} : uint2_native{a0, a1};
    const uint2_native b = SwapAB ? uint2_native{a0, a1} : uint2_native{b0, b1};
    if constexpr (std::is_same_v<T, half_native>) {
        return __builtin_mxc_mma_16x16x16f16(a, b, c);
    } else {
        return __builtin_mxc_mma_16x16x16bf16(a, b, c);
    }
}

} // namespace bf16_c500_tk_cute_local::arch
