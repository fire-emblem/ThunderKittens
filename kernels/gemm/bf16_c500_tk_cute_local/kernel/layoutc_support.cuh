#pragma once

#include <maca.h>
#include <maca_bfloat16.h>
#include <maca_fp16.h>
#include <mc_runtime.h>

#include <cmath>

#include "../cute_tk/primitives/pipeline/copy_atom.cuh"
#include "../cute_tk/primitives/compute/mma_atom.cuh"
#include "../cute_tk/primitives/mxc_builtins.cuh"
#include "../cute_tk/primitives/pipeline/sync_atom.cuh"

namespace bf16_c500_tk_local::kernel {

// Legacy macros for hand-scheduled mainloops — delegate to cute_tk atoms
#define arrive_gvmcnt(num) \
    ::bf16_c500_tk_cute_local::cute_tk::sync_atom::arrive_gvmcnt<(num)>();
#define arrive_bsmcnt(num) \
    ::bf16_c500_tk_local::primitives::arrive_bsmcnt<(num)>();

#define LDG_B128_BSM_NO_PREDICATOR(saddr, gaddr)                               \
    ::bf16_c500_tk_cute_local::cute_tk::copy_atom::issue_b128_bsm_no_pred(     \
        saddr, gaddr);
#define LDG_B128_BSM_WITH_PREDICATOR(saddr, gaddr, cmp_op1, cmp_op2, cmp_type) \
    ::bf16_c500_tk_cute_local::cute_tk::copy_atom::issue_b128_bsm_pred<cmp_type>( \
        saddr, gaddr, cmp_op1, cmp_op2);
#define LDG_B128_BSM_WITH_PREDICATOR_NORET0(saddr, gaddr, cmp_op1, cmp_op2, cmp_type) \
    ::bf16_c500_tk_cute_local::cute_tk::copy_atom::load_b_stage_pred_noret0(  \
        saddr, gaddr, cmp_op1, cmp_op2);
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

using FLOAT4 = ::bf16_c500_tk_cute_local::cute_tk::mma_atom::float4_t;

template <typename T, bool SwapAB = false>
__forceinline__ __device__ FLOAT4 mma_16x16x16b16(uint a0, uint a1, uint b0,
                                                  uint b1, FLOAT4 C) {
    if constexpr (SwapAB) {
        return ::bf16_c500_tk_cute_local::cute_tk::mma_atom::fma_pair<T>(
            b0, b1, a0, a1, C);
    } else {
        return ::bf16_c500_tk_cute_local::cute_tk::mma_atom::fma_pair<T>(
            a0, a1, b0, b1, C);
    }
}

template <typename T, typename BFragType, typename AFragType>
__forceinline__ __device__ FLOAT4 accumulate_layoutc_kgroup(
    const BFragType (&b_frag)[4],
    const AFragType (&a_frag)[4],
    FLOAT4 c) {
    return ::bf16_c500_tk_cute_local::cute_tk::mma_atom::accumulate_kgroup<T>(
        b_frag, a_frag, c);
}

} // namespace bf16_c500_tk_local::kernel
