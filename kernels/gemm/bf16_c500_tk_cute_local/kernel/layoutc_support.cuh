#pragma once

#include <maca.h>
#include <maca_bfloat16.h>
#include <maca_fp16.h>
#include <mc_runtime.h>

#include <cmath>

#include "../cute_tk/primitives/async_copy.cuh"
#include "../cute_tk/primitives/mma.cuh"
#include "../cute_tk/primitives/mxc_builtins.cuh"
#include "../cute_tk/primitives/sync.cuh"

namespace bf16_c500_tk_local::kernel {

#define arrive_gvmcnt(num) ::bf16_c500_tk_local::primitives::arrive_gvmcnt<(num)>();
#define arrive_bsmcnt(num) ::bf16_c500_tk_local::primitives::arrive_bsmcnt<(num)>();

#define LDG_B128_BSM_NO_PREDICATOR(saddr, gaddr)                               \
    ::bf16_c500_tk_local::primitives::ldg_b128_bsm_no_predicator(             \
        reinterpret_cast<void *>(saddr), reinterpret_cast<void *>(gaddr));
#define LDG_B128_BSM_WITH_PREDICATOR(saddr, gaddr, cmp_op1, cmp_op2, cmp_type) \
    ::bf16_c500_tk_local::primitives::ldg_b128_bsm_with_predicator<cmp_type>(  \
        reinterpret_cast<void *>(saddr), reinterpret_cast<void *>(gaddr),      \
        cmp_op1, cmp_op2);
#define LDG_B128_BSM_WITH_PREDICATOR_NORET0(saddr, gaddr, cmp_op1, cmp_op2, cmp_type) \
    ::bf16_c500_tk_local::primitives::ldg_b128_bsm_with_predicator_noret0<cmp_type>(   \
        reinterpret_cast<void *>(saddr), reinterpret_cast<void *>(gaddr),               \
        cmp_op1, cmp_op2);
#define LDG_B64_BSM_NO_PREDICATOR(saddr, gaddr)                                \
    ::bf16_c500_tk_local::primitives::ldg_b64_bsm_no_predicator(               \
        reinterpret_cast<void *>(saddr), reinterpret_cast<void *>(gaddr));
#define LDG_B64_BSM_with_PREDICATOR(saddr, gaddr, cmp_op1, cmp_op2, cmp_type)  \
    ::bf16_c500_tk_local::primitives::ldg_b64_bsm_with_predicator<cmp_type>(   \
        reinterpret_cast<void *>(saddr), reinterpret_cast<void *>(gaddr),      \
        cmp_op1, cmp_op2);
#define LDG_B64_BSM_WITH_PREDICATOR_NORET0(saddr, gaddr, cmp_op1, cmp_op2, cmp_type)   \
    ::bf16_c500_tk_local::primitives::ldg_b64_bsm_with_predicator_noret0<cmp_type>(    \
        reinterpret_cast<void *>(saddr), reinterpret_cast<void *>(gaddr),               \
        cmp_op1, cmp_op2);

using FLOAT4 = primitives::float4_native;

template <typename T, bool SwapAB = false>
__forceinline__ __device__ FLOAT4 mma_16x16x16b16(uint a0, uint a1, uint b0,
                                                  uint b1, FLOAT4 C) {
    return ::bf16_c500_tk_local::primitives::mma_16x16x16_b16<T, SwapAB>(a0, a1, b0, b1, C);
}

template <typename T, typename BFragType, typename AFragType>
__forceinline__ __device__ FLOAT4 accumulate_layoutc_kgroup(
    const BFragType (&b_frag)[4],
    const AFragType (&a_frag)[4],
    FLOAT4 c) {
    c = mma_16x16x16b16<T, true>(b_frag[0][0], b_frag[0][1], a_frag[0][0],
                                 a_frag[0][1], c);
    c = mma_16x16x16b16<T, true>(b_frag[0][2], b_frag[0][3], a_frag[0][2],
                                 a_frag[0][3], c);
    c = mma_16x16x16b16<T, true>(b_frag[1][0], b_frag[1][1], a_frag[1][0],
                                 a_frag[1][1], c);
    c = mma_16x16x16b16<T, true>(b_frag[1][2], b_frag[1][3], a_frag[1][2],
                                 a_frag[1][3], c);
    c = mma_16x16x16b16<T, true>(b_frag[2][0], b_frag[2][1], a_frag[2][0],
                                 a_frag[2][1], c);
    c = mma_16x16x16b16<T, true>(b_frag[2][2], b_frag[2][3], a_frag[2][2],
                                 a_frag[2][3], c);
    c = mma_16x16x16b16<T, true>(b_frag[3][0], b_frag[3][1], a_frag[3][0],
                                 a_frag[3][1], c);
    c = mma_16x16x16b16<T, true>(b_frag[3][2], b_frag[3][3], a_frag[3][2],
                                 a_frag[3][3], c);
    return c;
}

} // namespace bf16_c500_tk_local::kernel
