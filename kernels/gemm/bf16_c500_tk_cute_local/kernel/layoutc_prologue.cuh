#pragma once

#include <maca.h>

namespace bf16_c500_tk_local::kernel {

template <typename LdsType>
__device__ __forceinline__ LdsType load_layoutc_fragment_from_shared(
    uint8_t *wsm_lds,
    const int (&lds_offset)[4],
    int idx) {
    return *reinterpret_cast<LdsType *>(wsm_lds + lds_offset[idx]);
}

template <typename ALdgType, typename BLdgType, typename T>
__device__ __forceinline__ void issue_layoutc_prologue(
    uint8_t *wsm_ldg,
    uint8_t *a_ptr,
    uint8_t *b_ptr,
    const int (&a_ldg_offset)[2][4],
    const int (&b_ldg_offset)[2][4],
    int k,
    int n,
    int start_col) {
    constexpr int stage_count = 4;
#pragma unroll
    for (int stage_i = 0; stage_i < stage_count; ++stage_i) {
        __builtin_mxc_ldg_b128_bsm_predicator(
            wsm_ldg + 0x4000 * stage_i + 0x0000, a_ptr + a_ldg_offset[0][stage_i],
            0, true, true, false, true, 0, k / (sizeof(ALdgType) / sizeof(T)),
            MACA_ICMP_SLT);
        __builtin_mxc_ldg_b128_bsm_predicator(
            wsm_ldg + 0x4000 * stage_i + 0x1000, a_ptr + a_ldg_offset[1][stage_i],
            0, true, true, false, true, 0, k / (sizeof(ALdgType) / sizeof(T)),
            MACA_ICMP_SLT);
        __builtin_mxc_ldg_b128_bsm_predicator(
            wsm_ldg + 0x4000 * stage_i + 0x2000, b_ptr + b_ldg_offset[0][stage_i],
            0, true, true, false, true, start_col + stage_i * 16, n,
            MACA_ICMP_SLT);
        __builtin_mxc_ldg_b128_bsm_predicator(
            wsm_ldg + 0x4000 * stage_i + 0x3000, b_ptr + b_ldg_offset[1][stage_i],
            0, true, true, false, true, start_col + (4 + stage_i) * 16, n,
            MACA_ICMP_SLT);
    }
}

template <typename ALdsType, typename BLdsType>
__device__ __forceinline__ void prime_layoutc_registers(
    ALdsType (&a)[4][4],
    BLdsType (&b)[4][4],
    uint8_t *wsm_lds,
    const int (&a_lds_offset)[4],
    const int (&b_lds_offset)[4]) {
    __builtin_mxc_arrive_gvmcnt(12);
    __builtin_mxc_barrier_inst();

    a[0][0] = load_layoutc_fragment_from_shared<ALdsType>(wsm_lds, a_lds_offset, 0);
    a[0][1] = load_layoutc_fragment_from_shared<ALdsType>(wsm_lds, a_lds_offset, 1);
    a[0][2] = load_layoutc_fragment_from_shared<ALdsType>(wsm_lds, a_lds_offset, 2);
    a[0][3] = load_layoutc_fragment_from_shared<ALdsType>(wsm_lds, a_lds_offset, 3);

    b[0][0] = load_layoutc_fragment_from_shared<BLdsType>(wsm_lds, b_lds_offset, 0);
    b[0][1] = load_layoutc_fragment_from_shared<BLdsType>(wsm_lds, b_lds_offset, 1);
    b[0][2] = load_layoutc_fragment_from_shared<BLdsType>(wsm_lds, b_lds_offset, 2);
    b[0][3] = load_layoutc_fragment_from_shared<BLdsType>(wsm_lds, b_lds_offset, 3);

    __builtin_mxc_arrive_gvmcnt(8);
    __builtin_mxc_barrier_inst();

    a[1][0] = load_layoutc_fragment_from_shared<ALdsType>(wsm_lds + 0x4000, a_lds_offset, 0);
    a[1][1] = load_layoutc_fragment_from_shared<ALdsType>(wsm_lds + 0x4000, a_lds_offset, 1);
    a[1][2] = load_layoutc_fragment_from_shared<ALdsType>(wsm_lds + 0x4000, a_lds_offset, 2);
    a[1][3] = load_layoutc_fragment_from_shared<ALdsType>(wsm_lds + 0x4000, a_lds_offset, 3);

    b[1][0] = load_layoutc_fragment_from_shared<BLdsType>(wsm_lds + 0x4000, b_lds_offset, 0);
    b[1][1] = load_layoutc_fragment_from_shared<BLdsType>(wsm_lds + 0x4000, b_lds_offset, 1);
    b[1][2] = load_layoutc_fragment_from_shared<BLdsType>(wsm_lds + 0x4000, b_lds_offset, 2);
    b[1][3] = load_layoutc_fragment_from_shared<BLdsType>(wsm_lds + 0x4000, b_lds_offset, 3);
}

template <typename ALdsType, typename BLdsType>
__device__ __forceinline__ void reload_layoutc_stage_from_shared(
    ALdsType (&a)[4][4],
    BLdsType (&b)[4][4],
    int lds_idx,
    uint8_t *wsm_lds2,
    const int (&a_lds_offset)[4],
    const int (&b_lds_offset)[4]) {
    a[lds_idx][0] = load_layoutc_fragment_from_shared<ALdsType>(wsm_lds2, a_lds_offset, 0);
    a[lds_idx][1] = load_layoutc_fragment_from_shared<ALdsType>(wsm_lds2, a_lds_offset, 1);
    a[lds_idx][2] = load_layoutc_fragment_from_shared<ALdsType>(wsm_lds2, a_lds_offset, 2);
    a[lds_idx][3] = load_layoutc_fragment_from_shared<ALdsType>(wsm_lds2, a_lds_offset, 3);
    b[lds_idx][0] = load_layoutc_fragment_from_shared<BLdsType>(wsm_lds2, b_lds_offset, 0);
    b[lds_idx][1] = load_layoutc_fragment_from_shared<BLdsType>(wsm_lds2, b_lds_offset, 1);
    b[lds_idx][2] = load_layoutc_fragment_from_shared<BLdsType>(wsm_lds2, b_lds_offset, 2);
    b[lds_idx][3] = load_layoutc_fragment_from_shared<BLdsType>(wsm_lds2, b_lds_offset, 3);
}

} // namespace bf16_c500_tk_local::kernel
