#pragma once

#include "copy_atom.cuh"
#include "mma_atom.cuh"
#include "sync_atom.cuh"

namespace bf16_c500_tk_cute_local::cute_tk {

struct mainloop_atom {
    template <int SharedNumCycleB, int WarpPerBlock, int APerWarp,
              typename SharedPtr, typename BPtr, typename APtrArray,
              typename AFragArray>
    __device__ __forceinline__ static void prefetch_stage(
        int stage_idx,
        int split_num_cycle_b,
        int warp_id_in_block,
        int num_cycle_b,
        int &a_ptr_offset,
        int &b_ptr_offset,
        int &shared_ptr_offset,
        SharedPtr shared_ptr,
        BPtr b_ptr,
        APtrArray const &a_ptr,
        AFragArray &tmp_a) {
        (void)stage_idx;
        asm("/* Stop compiler reordering (begin) */");
        for (int idx_a = 0; idx_a < APerWarp; ++idx_a) {
            tmp_a[stage_idx][idx_a] =
                copy_atom::load_gmem_128(a_ptr[idx_a] + a_ptr_offset);
        }
        a_ptr_offset += 64;
#pragma unroll
        for (int j = 0; j < SharedNumCycleB; j += WarpPerBlock) {
            const int pred = (j + warp_id_in_block) < split_num_cycle_b ? 1 : 0;
            copy_atom::load_b_stage_pred_noret0(
                reinterpret_cast<uint8_t *>(shared_ptr + shared_ptr_offset +
                                            (j + warp_id_in_block) * 64),
                b_ptr + b_ptr_offset + (j + warp_id_in_block) * 64, pred, 1);
        }
        shared_ptr_offset += SharedNumCycleB * 64;
        b_ptr_offset += num_cycle_b * 64;
    }

    template <int SharedNumCycleB, typename SharedStage, typename BFragArray>
    __device__ __forceinline__ static void load_b_fragments(
        SharedStage const &shared_stage,
        int lane_id,
        BFragArray &tmp_b) {
        for (int j = 0; j < SharedNumCycleB; ++j) {
            tmp_b[j] = shared_stage[j][lane_id];
        }
    }

    template <typename T, int SharedNumCycleB, int APerWarp, typename AccArray,
              typename AFragArray, typename BFragArray>
    __device__ __forceinline__ static void compute_stage(
        AccArray &c_f32,
        AFragArray const &tmp_a,
        int stage_idx,
        BFragArray const &tmp_b) {
        mma_atom::template accumulate_reusea_tile<T, SharedNumCycleB, APerWarp>(
            c_f32, tmp_a, stage_idx, tmp_b);
    }

    template <int SharedArriveCount, int APerWarp, int Stages>
    __device__ __forceinline__ static void wait_steady() {
        sync_atom::template wait_gmem_async<(SharedArriveCount + APerWarp) *
                                            (Stages - 1)>();
    }

    template <int SharedArriveCount, int Stages>
    __device__ __forceinline__ static void wait_drain(int stage_idx) {
        if (stage_idx == 0) {
            sync_atom::template wait_gmem_async<(SharedArriveCount + 1) *
                                                (Stages - 1)>();
        } else if (stage_idx == 1) {
            sync_atom::template wait_gmem_async<(SharedArriveCount + 1) *
                                                (Stages - 2)>();
        } else if (stage_idx == 2) {
            sync_atom::template wait_gmem_async<(SharedArriveCount + 1) *
                                                (Stages - 3)>();
        } else {
            sync_atom::template wait_gmem_async<(SharedArriveCount + 1) *
                                                (Stages - 4)>();
        }
    }
};

} // namespace bf16_c500_tk_cute_local::cute_tk
