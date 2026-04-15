#pragma once

#include <maca.h>

#include "layoutc_prologue.cuh"
#include "layoutc_support.cuh"

namespace bf16_c500_tk_local::kernel {

template <typename T, int Stage, typename FLOAT4, typename ALdsType,
          typename BLdsType>
__device__ __forceinline__ void accumulate_layoutc_stage_tiles(
    FLOAT4 (&c_f32)[4][4],
    ALdsType (&a)[Stage][4],
    BLdsType (&b)[Stage][4],
    int stage_i) {
    for (int i = 0; i < stage_i; ++i) {
        c_f32[stage_i][i] =
            accumulate_layoutc_kgroup<T>(b[i], a[stage_i], c_f32[stage_i][i]);
    }
    for (int i = 0; i <= stage_i; ++i) {
        c_f32[i][stage_i] =
            accumulate_layoutc_kgroup<T>(b[stage_i], a[i], c_f32[i][stage_i]);
    }
}

template <typename T, int Stage, typename FLOAT4, typename ALdsType,
          typename BLdsType, typename ALdgType, typename BLdgType>
__device__ __forceinline__ void run_layoutc_tail_iteration(
    FLOAT4 (&c_f32)[4][4],
    ALdsType (&a)[Stage][4],
    BLdsType (&b)[Stage][4],
    uint8_t *wsm_ldg,
    uint8_t *wsm_lds,
    const int (&a_lds_offset)[4],
    const int (&b_lds_offset)[4],
    const int (&a_ldg_offset)[2][4],
    const int (&b_ldg_offset)[2][4],
    uint8_t *a_ptr,
    uint8_t *b_ptr,
    int stage_i,
    int k_remaining,
    int n,
    int start_col) {
    const int lds_idx = (stage_i + 1) % Stage;
    uint8_t *wsm_lds2 = wsm_lds + (0x4000 * lds_idx);

    accumulate_layoutc_stage_tiles<T, Stage>(c_f32, a, b, stage_i);

    arrive_gvmcnt(4 * (Stage - 2));
    __builtin_mxc_barrier_inst();

    LDG_B128_BSM_WITH_PREDICATOR(wsm_ldg + 0x4000 * stage_i + 0x0000,
                                 a_ptr + a_ldg_offset[0][stage_i], 0,
                                 k_remaining / (sizeof(ALdgType) / sizeof(T)),
                                 MACA_ICMP_SLT);
    LDG_B128_BSM_WITH_PREDICATOR(wsm_ldg + 0x4000 * stage_i + 0x1000,
                                 a_ptr + a_ldg_offset[1][stage_i], 0,
                                 k_remaining / (sizeof(ALdgType) / sizeof(T)),
                                 MACA_ICMP_SLT);
    LDG_B128_BSM_WITH_PREDICATOR(wsm_ldg + 0x4000 * stage_i + 0x2000,
                                 b_ptr + b_ldg_offset[0][stage_i],
                                 start_col + stage_i * 16, n, MACA_ICMP_SLT);
    LDG_B128_BSM_WITH_PREDICATOR(
        wsm_ldg + 0x4000 * stage_i + 0x3000, b_ptr + b_ldg_offset[1][stage_i],
        start_col + stage_i * 16 + 64, n, MACA_ICMP_SLT);

    reload_layoutc_stage_from_shared(a, b, lds_idx, wsm_lds2, a_lds_offset,
                                     b_lds_offset);
}

template <int Stage>
__device__ __forceinline__ void arrive_layoutc_drain_barrier(int stage_i) {
    if (stage_i == 0) {
        arrive_gvmcnt(4 * (Stage - 2 - 0));
        __builtin_mxc_barrier_inst();
    } else if (stage_i == 1) {
        arrive_gvmcnt(4 * (Stage - 2 - 1));
        __builtin_mxc_barrier_inst();
    } else if (stage_i == 2) {
        arrive_gvmcnt(4 * (Stage - 2 - 2));
        __builtin_mxc_barrier_inst();
    }
}

template <typename T, int Stage, typename FLOAT4, typename ALdsType,
          typename BLdsType>
__device__ __forceinline__ void drain_layoutc_tail(
    FLOAT4 (&c_f32)[4][4],
    ALdsType (&a)[Stage][4],
    BLdsType (&b)[Stage][4],
    uint8_t *wsm_lds,
    const int (&a_lds_offset)[4],
    const int (&b_lds_offset)[4]) {
#pragma unroll
    for (int stage_i = 0; stage_i < Stage; ++stage_i) {
        const int lds_idx = (stage_i + 1) % Stage;
        uint8_t *wsm_lds2 = wsm_lds + (0x4000 * lds_idx);

        accumulate_layoutc_stage_tiles<T, Stage>(c_f32, a, b, stage_i);
        arrive_layoutc_drain_barrier<Stage>(stage_i);

        if (stage_i < Stage - 1) {
            reload_layoutc_stage_from_shared(a, b, lds_idx, wsm_lds2,
                                             a_lds_offset, b_lds_offset);
        }
    }
}

} // namespace bf16_c500_tk_local::kernel
