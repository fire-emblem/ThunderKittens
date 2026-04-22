#pragma once

#include <maca.h>

#include "../cute_tk/contracts/stage_contract.cuh"
#include "../cute_tk/primitives/pipeline/fragment_atom.cuh"
#include "../cute_tk/primitives/pipeline/issue_order_atom.cuh"

namespace bf16_c500_tk_local::kernel {

template <typename LdsType>
__device__ __forceinline__ LdsType load_layoutc_fragment_from_shared(
    uint8_t *wsm_lds,
    const int (&lds_offset)[4],
    int idx) {
    return ::bf16_c500_tk_cute_local::cute_tk::fragment_atom::load_from_shared<LdsType>(
        wsm_lds, lds_offset, idx);
}

template <typename ALdgType, typename BLdgType, typename T,
          typename StageContract =
              ::bf16_c500_tk_local::contracts::stage_contract>
__device__ __forceinline__ void issue_layoutc_prologue(
    uint8_t *wsm_ldg,
    uint8_t *a_ptr,
    uint8_t *b_ptr,
    const int (&a_ldg_offset)[2][4],
    const int (&b_ldg_offset)[2][4],
    int k,
    int n,
    int start_col) {
    constexpr int stage_count = StageContract::stage_count;
#pragma unroll
    for (int stage_i = 0; stage_i < stage_count; ++stage_i) {
        ::bf16_c500_tk_cute_local::cute_tk::issue_order_atom::
            template issue_ab_stage_pred<StageContract, ALdgType, BLdgType, T>(
                wsm_ldg, a_ptr, b_ptr, a_ldg_offset, b_ldg_offset, stage_i, 0,
                k / (sizeof(ALdgType) / sizeof(T)), start_col + stage_i * 16,
                n, start_col + (4 + stage_i) * 16, n);
    }
}

template <typename ALdsType, typename BLdsType,
          typename StageContract =
              ::bf16_c500_tk_local::contracts::stage_contract>
__device__ __forceinline__ void prime_layoutc_registers(
    ALdsType (&a)[4][4],
    BLdsType (&b)[4][4],
    uint8_t *wsm_lds,
    const int (&a_lds_offset)[4],
    const int (&b_lds_offset)[4]) {
    ::bf16_c500_tk_cute_local::cute_tk::fragment_atom::prime_registers<ALdsType, BLdsType, StageContract>(
        a, b, wsm_lds, a_lds_offset, b_lds_offset);
}

template <typename ALdsType, typename BLdsType,
          typename StageContract =
              ::bf16_c500_tk_local::contracts::stage_contract>
__device__ __forceinline__ void reload_layoutc_stage_from_shared(
    ALdsType (&a)[4][4],
    BLdsType (&b)[4][4],
    int lds_idx,
    uint8_t *wsm_lds2,
    const int (&a_lds_offset)[4],
    const int (&b_lds_offset)[4]) {
    ::bf16_c500_tk_cute_local::cute_tk::fragment_atom::reload_stage<ALdsType, BLdsType, StageContract>(
        a, b, lds_idx, wsm_lds2, a_lds_offset, b_lds_offset);
}

} // namespace bf16_c500_tk_local::kernel
