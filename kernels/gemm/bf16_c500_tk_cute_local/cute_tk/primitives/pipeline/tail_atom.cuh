#pragma once

#include "../compute/mma_atom.cuh"
#include "copy_atom.cuh"
#include "fragment_atom.cuh"
#include "schedule_atom.cuh"
#include "sync_atom.cuh"

namespace bf16_c500_tk_cute_local::cute_tk {

struct tail_atom {
    template <typename T, int Stage, typename FLOAT4, typename ALdsType,
              typename BLdsType>
    __device__ __forceinline__ static void accumulate_stage_tiles(
        FLOAT4 (&c_f32)[4][4],
        ALdsType (&a)[Stage][4],
        BLdsType (&b)[Stage][4],
        int stage_i) {
        for (int i = 0; i < stage_i; ++i) {
            c_f32[stage_i][i] =
                mma_atom::accumulate_kgroup<T>(b[i], a[stage_i], c_f32[stage_i][i]);
        }
        for (int i = 0; i <= stage_i; ++i) {
            c_f32[i][stage_i] =
                mma_atom::accumulate_kgroup<T>(b[stage_i], a[i], c_f32[i][stage_i]);
        }
    }

    template <typename T, int Stage, typename FLOAT4, typename ALdsType,
              typename BLdsType, typename ALdgType, typename BLdgType>
    __device__ __forceinline__ static void run_tail_iteration(
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
        constexpr int stage_count = Stage;
        const int lds_idx = (stage_i + 1) % Stage;
        uint8_t *wsm_lds2 = wsm_lds + (0x4000 * lds_idx);

        accumulate_stage_tiles<T, Stage>(c_f32, a, b, stage_i);

        sync_atom::wait_gmem_async<4 * (Stage - 2)>();
        sync_atom::barrier();

        copy_atom::issue_b128_bsm_pred<MACA_ICMP_SLT>(
            wsm_ldg + 0x4000 * stage_i + 0x0000,
            a_ptr + a_ldg_offset[0][stage_i], 0,
            k_remaining / (sizeof(ALdgType) / sizeof(T)));
        copy_atom::issue_b128_bsm_pred<MACA_ICMP_SLT>(
            wsm_ldg + 0x4000 * stage_i + 0x1000,
            a_ptr + a_ldg_offset[1][stage_i], 0,
            k_remaining / (sizeof(ALdgType) / sizeof(T)));
        copy_atom::issue_b128_bsm_pred<MACA_ICMP_SLT>(
            wsm_ldg + 0x4000 * stage_i + 0x2000,
            b_ptr + b_ldg_offset[0][stage_i],
            start_col + stage_i * 16, n);
        copy_atom::issue_b128_bsm_pred<MACA_ICMP_SLT>(
            wsm_ldg + 0x4000 * stage_i + 0x3000,
            b_ptr + b_ldg_offset[1][stage_i],
            start_col + stage_i * 16 + 64, n);

        fragment_atom::reload_stage(a, b, lds_idx, wsm_lds2, a_lds_offset,
                                    b_lds_offset);
    }

    template <int Stage>
    __device__ __forceinline__ static void arrive_drain_barrier(int stage_i) {
        schedule_atom::template wait_tail_stage<Stage>(stage_i);
    }

    template <typename T, int Stage, typename FLOAT4, typename ALdsType,
              typename BLdsType>
    __device__ __forceinline__ static void drain_tail(
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

            accumulate_stage_tiles<T, Stage>(c_f32, a, b, stage_i);
            arrive_drain_barrier<Stage>(stage_i);

            if (stage_i < Stage - 1) {
                fragment_atom::reload_stage(a, b, lds_idx, wsm_lds2,
                                            a_lds_offset, b_lds_offset);
            }
        }
    }
};

} // namespace bf16_c500_tk_cute_local::cute_tk
