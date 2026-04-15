#pragma once

#include <maca.h>
#include <maca_bfloat16.h>
#include <maca_fp16.h>
#include <mc_runtime.h>

#include "../kernel/layoutc_support.cuh"
#include "primitives/pipeline/copy_atom.cuh"
#include "primitives/epilogue/epilogue_atom.cuh"
#include "primitives/pipeline/mainloop_atom.cuh"
#include "primitives/compute/mma_atom.cuh"
#include "policies.cuh"
#include "primitives/pipeline/schedule_atom.cuh"
#include "primitives/pipeline/sync_atom.cuh"

namespace bf16_c500_tk_cute_local::cute_tk::kernel {

using FLOAT4 = ::bf16_c500_tk_cute_local::cute_tk::mma_atom::float4_t;

template <typename T, typename Tc, typename Taccum, int StageCount, int NTile,
          int APerWarp, int SplitN, int SplitK, bool IsBetaZero,
          bool HasOneDimBias, bool OutputContinuous = true>
__global__ void __launch_bounds__(256) cute_tk_continuousc_reusea_n(
    T *A, T *B, Tc *C, int m, int n, int k, Taccum alpha, Taccum beta,
    Tc *bias) {
    static_assert(IsBetaZero, "tk local benchmark path assumes beta == 0");
    static_assert(!HasOneDimBias, "tk local benchmark path does not support bias yet");

    constexpr int Stages = StageCount;
    constexpr int ElementsPerAccess = 8;
    constexpr int RowThreadsPerMma = 16;
    constexpr int ColThreadsPerMma = 4;
    constexpr int ElementsPerThreadPerMma = 4;
    constexpr int BlockDimX = 256;
    constexpr int WarpPerBlock = BlockDimX / 64;
    constexpr int NumCycleB = NTile / RowThreadsPerMma;
    constexpr int SharedNumCycleB = (NumCycleB + SplitN - 1) / SplitN;
    constexpr int SharedArriveCount =
        (SharedNumCycleB + WarpPerBlock - 1) / WarpPerBlock;
    using schedule_t =
        continuousc_reusea_schedule<NTile, APerWarp, SplitN, SplitK, Stages>;

    using UINT4 = __NATIVE_VECTOR__(4, uint);
    using INT128 = __NATIVE_VECTOR__(4, int);

    const auto schedule = schedule_t::make(m, k);

    const int warpIdInBlock = schedule.warp_id_in_block;
    const int laneId = schedule.lane_id;
    const int quarterWarpId = schedule.quarter_warp_id;
    const int quarterLaneId = schedule.quarter_lane_id;
    const int warpRowsGroupBegin = schedule.warp_rows_group_begin;
    const int splitNStart = schedule.split_n_start;
    const int end = schedule.end;

    int A_offset = schedule_t::initial_a_offset(schedule, k);
    int B_offset = schedule_t::initial_b_offset(schedule);
    int C_offset[APerWarp];
    for (int i = 0; i < APerWarp; ++i) {
        C_offset[i] = schedule_t::initial_c_offset(schedule, i, m);
    }

    UINT4 *A_ptr[APerWarp];
    for (int i = 0; i < APerWarp; ++i) {
        A_ptr[i] = reinterpret_cast<UINT4 *>(A) + A_offset +
                   i * (RowThreadsPerMma * (k / ElementsPerAccess));
    }
    UINT4 *B_ptr = reinterpret_cast<UINT4 *>(B) + B_offset;
    __shared__ UINT4 sharedTmpB[Stages][SharedNumCycleB][64];
    UINT4 *shared_ptr =
        reinterpret_cast<UINT4 *>(&sharedTmpB[0][0][0]) + laneId;

    const int splitNumCycleB = schedule_t::split_num_cycle_b(schedule);

    UINT4 tmpA[Stages][APerWarp];
    UINT4 tmpB[SharedNumCycleB];
    FLOAT4 C_f32[SharedNumCycleB][APerWarp];

    for (int i = 0; i < SharedNumCycleB; ++i) {
        for (int idxA = 0; idxA < APerWarp; ++idxA) {
            C_f32[i][idxA] = {0.0f, 0.0f, 0.0f, 0.0f};
        }
    }

    if (end == 0) {
        for (int idxA = 0; idxA < splitNumCycleB; ++idxA) {
            A_ptr[idxA] = reinterpret_cast<UINT4 *>(A);
        }
    }

    int A_ptr_offset = 0;
    int B_ptr_offset = 0;
    int shared_ptr_offset = 0;

#pragma unroll
    for (int t = 0; t < Stages; ++t) {
        asm("/* Stop compiler reordering (begin) */");
        for (int idxA = 0; idxA < APerWarp; ++idxA) {
            tmpA[t][idxA] =
                copy_atom::load_gmem_128(A_ptr[idxA] + A_ptr_offset);
        }
        A_ptr_offset += 64;
#pragma unroll
        for (int j = 0; j < SharedNumCycleB; j += WarpPerBlock) {
            const int pred = (j + warpIdInBlock) < splitNumCycleB ? 1 : 0;
            copy_atom::load_b_stage_pred_noret0(
                reinterpret_cast<uint8_t *>(shared_ptr + shared_ptr_offset +
                                            (j + warpIdInBlock) * 64),
                B_ptr + B_ptr_offset + (j + warpIdInBlock) * 64, pred, 1);
        }
        shared_ptr_offset += SharedNumCycleB * 64;
        B_ptr_offset += NumCycleB * 64;
    }

    for (int iter = 0; iter < end - 1; ++iter) {
        shared_ptr_offset = 0;
#pragma unroll
        for (int t = 0; t < Stages; ++t) {
            asm("/* Stop compiler reordering (loop) */");
            schedule_atom::template wait_reusea_steady<SharedArriveCount,
                                                       APerWarp, Stages>();

            mainloop_atom::template load_b_fragments<SharedNumCycleB>(
                sharedTmpB[t], laneId, tmpB);
            mainloop_atom::template compute_stage<T, Stages, SharedNumCycleB,
                                                  APerWarp>(C_f32, tmpA, t,
                                                            tmpB);
            sync_atom::barrier();

            for (int idxA = 0; idxA < APerWarp; ++idxA) {
                tmpA[t][idxA] =
                    copy_atom::load_gmem_128(A_ptr[idxA] + A_ptr_offset);
            }
            A_ptr_offset += 64;
#pragma unroll
            for (int j = 0; j < SharedNumCycleB; j += WarpPerBlock) {
                const int pred = (j + warpIdInBlock) < splitNumCycleB ? 1 : 0;
                copy_atom::load_b_stage_pred_noret0(
                    reinterpret_cast<uint8_t *>(shared_ptr + shared_ptr_offset +
                                                (j + warpIdInBlock) * 64),
                    B_ptr + B_ptr_offset + (j + warpIdInBlock) * 64, pred, 1);
            }
            shared_ptr_offset += SharedNumCycleB * 64;
            B_ptr_offset += NumCycleB * 64;
        }
    }

#pragma unroll
    for (int t = 0; t < Stages; ++t) {
        asm("/* Stop compiler reordering (end loop) */");
        schedule_atom::template wait_reusea_drain<SharedArriveCount, Stages>(t);

        mainloop_atom::template load_b_fragments<SharedNumCycleB>(sharedTmpB[t],
                                                                  laneId, tmpB);
        mainloop_atom::template compute_stage<T, Stages, SharedNumCycleB,
                                              APerWarp>(C_f32, tmpA, t, tmpB);
    }

    if (end == 0) return;

    for (int j = 0; j < SharedNumCycleB && j < splitNumCycleB; ++j) {
        for (int idxA = 0; idxA < APerWarp; ++idxA) {
            if constexpr (OutputContinuous) {
                epilogue_atom::template store_continuousc_fragment<
                    Tc, Taccum, FLOAT4, IsBetaZero, SplitK>(
                    reinterpret_cast<Tc *>(C), C_f32[j][idxA], m, n,
                    C_offset[idxA],
                    quarterLaneId + (splitNStart + j) * RowThreadsPerMma,
                    idxA, j, quarterWarpId, quarterLaneId, alpha, beta);
            } else {
                epilogue_atom::template store_layoutc_fragment<Tc, Taccum,
                                                               FLOAT4,
                                                               IsBetaZero>(
                    reinterpret_cast<Tc *>(C), C_f32[j][idxA], m, n,
                    warpRowsGroupBegin, splitNStart, idxA, j, quarterWarpId,
                    quarterLaneId, alpha, beta);
            }
            C_offset[idxA] = schedule_t::advance_c_offset(C_offset[idxA], m);
        }
    }
}

} // namespace bf16_c500_tk_cute_local::cute_tk::kernel
