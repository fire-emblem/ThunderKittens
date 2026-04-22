#pragma once

#include <maca.h>

#include "../cute_tk/primitives/pipeline/tail_atom.cuh"
#include "layoutc_prologue.cuh"

namespace bf16_c500_tk_local::kernel {

template <typename T, int Stage, typename FLOAT4, typename ALdsType,
          typename BLdsType>
__device__ __forceinline__ void accumulate_layoutc_stage_tiles(
    FLOAT4 (&c_f32)[4][4],
    ALdsType (&a)[Stage][4],
    BLdsType (&b)[Stage][4],
    int stage_i) {
    ::bf16_c500_tk_cute_local::cute_tk::tail_atom::accumulate_stage_tiles<T, Stage>(
        c_f32, a, b, stage_i);
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
    ::bf16_c500_tk_cute_local::cute_tk::tail_atom::run_tail_iteration<T, Stage, FLOAT4,
        ALdsType, BLdsType, ALdgType, BLdgType>(
            c_f32, a, b, wsm_ldg, wsm_lds, a_lds_offset, b_lds_offset,
            a_ldg_offset, b_ldg_offset, a_ptr, b_ptr, stage_i, k_remaining,
            n, start_col);
}

template <int Stage>
__device__ __forceinline__ void arrive_layoutc_drain_barrier(int stage_i) {
    ::bf16_c500_tk_cute_local::cute_tk::tail_atom::arrive_drain_barrier<Stage>(stage_i);
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
    ::bf16_c500_tk_cute_local::cute_tk::tail_atom::drain_tail<T, Stage>(
        c_f32, a, b, wsm_lds, a_lds_offset, b_lds_offset);
}

} // namespace bf16_c500_tk_local::kernel
