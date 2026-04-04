#pragma once

#include "../async_primitives.cuh"
#include "../copy_atoms.cuh"
#include "../layouts/lds_offsets.cuh"
#include "../layouts/operand_layouts.cuh"
#include "bf16_contracts.cuh"
#include "bf16_epilogue.cuh"

namespace kittens::arch::c500::gemm {

using bf16_mainloop_atom = mma_bf16_16x16x16_fp32;
using bf16_shared_tile_a = st_bf<bf16_contracts::kWarpM, bf16_contracts::kStageK>;
using bf16_shared_tile_b = st_bf<bf16_contracts::kStageK, bf16_contracts::kWarpN>;
using bf16_shared_tile_c = st_bf<bf16_contracts::kWarpM, bf16_contracts::kWarpN>;
using bf16_reg_tile_c = rt_fl<bf16_contracts::kWarpM, bf16_contracts::kWarpN>;

constexpr int kAtomsM = bf16_contracts::kWarpM / bf16_mainloop_atom::M;
constexpr int kAtomsN = bf16_contracts::kWarpN / bf16_mainloop_atom::N;
constexpr int kStageAtomsK = bf16_contracts::kStageK / bf16_mainloop_atom::K;

using bf16_frag_a = fragment_a<bf16_mainloop_atom>;
using bf16_frag_b = fragment_b<bf16_mainloop_atom>;
using bf16_frag_c = fragment_c<bf16_mainloop_atom>;

__device__ inline void zero_accumulators(bf16_frag_c (&acc)[kAtomsM][kAtomsN]) {
#pragma unroll
    for (int m = 0; m < kAtomsM; ++m) {
#pragma unroll
        for (int n = 0; n < kAtomsN; ++n) {
#pragma unroll
            for (int r = 0; r < bf16_mainloop_atom::c_registers; ++r) {
                acc[m][n].reg[r] = 0.0f;
            }
        }
    }
}

template<typename SharedA, typename SharedB>
__device__ inline void mma_tile_stage(bf16_frag_c (&acc)[kAtomsM][kAtomsN],
                                      const SharedA &a_tile,
                                      const SharedB &b_tile) {
    bf16_frag_a a_frag[kAtomsM][kStageAtomsK];
    bf16_frag_b b_frag[kStageAtomsK][kAtomsN];

#pragma unroll
    for (int m = 0; m < kAtomsM; ++m) {
#pragma unroll
        for (int k = 0; k < kStageAtomsK; ++k) {
            load_a<bf16_mainloop_atom>(a_frag[m][k], a_tile, m * bf16_mainloop_atom::M, k * bf16_mainloop_atom::K);
        }
    }

#pragma unroll
    for (int k = 0; k < kStageAtomsK; ++k) {
#pragma unroll
        for (int n = 0; n < kAtomsN; ++n) {
            load_b<bf16_mainloop_atom>(b_frag[k][n], b_tile, k * bf16_mainloop_atom::K, n * bf16_mainloop_atom::N);
        }
    }

#pragma unroll
    for (int m = 0; m < kAtomsM; ++m) {
#pragma unroll
        for (int n = 0; n < kAtomsN; ++n) {
#pragma unroll
            for (int k = 0; k < kStageAtomsK; ++k) {
                bf16_frag_c next;
                mma<bf16_mainloop_atom>(next, a_frag[m][k], b_frag[k][n], acc[m][n]);
                acc[m][n] = next;
            }
        }
    }
}

template<typename LoadGroup, typename GlobalA, typename GlobalB>
__device__ inline void issue_stage_async(bf16_shared_tile_a (&a_s)[bf16_contracts::kStages][bf16_contracts::kLoadGroups],
                                         bf16_shared_tile_b (&b_s)[bf16_contracts::kStages][bf16_contracts::kLoadGroups],
                                         const GlobalA &a,
                                         const GlobalB &b,
                                         int stage_slot,
                                         int load_id,
                                         int warp_row,
                                         int warp_col,
                                         int k_stage) {
    LoadGroup::template load<2, true>(
        a_s[stage_slot][load_id], a, kittens::coord<bf16_shared_tile_a>{warp_row + load_id, k_stage});
    LoadGroup::template load<2, true>(
        b_s[stage_slot][load_id], b, kittens::coord<bf16_shared_tile_b>{k_stage, warp_col + load_id});
}

template<typename OutputTile>
__device__ inline void export_accumulators(OutputTile &dst,
                                           const bf16_frag_c (&acc)[kAtomsM][kAtomsN]) {
#pragma unroll
    for (int m = 0; m < kAtomsM; ++m) {
#pragma unroll
        for (int n = 0; n < kAtomsN; ++n) {
            store_epilogue(dst, acc[m][n], m, n);
        }
    }
}

template<int M, int N, int K, typename Globals>
__device__ inline void run_bf16_mainloop(const Globals &g) {
    using load_group = kittens::group<(bf16_contracts::kNumWorkers / bf16_contracts::kLoadGroups)>;
    constexpr int kLoadBlocks = bf16_contracts::kNumWorkers / load_group::GROUP_WARPS;

    static_assert(K % bf16_contracts::kBlockK == 0,
                  "Task 4 minimal bf16_c500 mainloop requires K to be a multiple of 128.");

    const int workerid = kittens::warpid();
    const int row_worker = workerid / bf16_contracts::kWaveN;
    const int col_worker = workerid % bf16_contracts::kWaveN;
    const int load_id = load_group::groupid();

    const int warp_row = bf16_contracts::kLoadGroups * blockIdx.y;
    const int warp_col = bf16_contracts::kLoadGroups * blockIdx.x;

    __shared__ bf16_shared_tile_a a_s[bf16_contracts::kStages][kLoadBlocks];
    __shared__ bf16_shared_tile_b b_s[bf16_contracts::kStages][kLoadBlocks];

    bf16_frag_c acc[kAtomsM][kAtomsN];
    bf16_reg_tile_c out;
    zero_accumulators(acc);

    constexpr int num_k_stages = K / bf16_contracts::kStageK;

    issue_stage_async<load_group>(a_s, b_s, g.a, g.b, 0, load_id, warp_row, warp_col, 0);
    __syncthreads();

    for (int k_stage = 0; k_stage < num_k_stages; ++k_stage) {
        const int stage_slot = k_stage % bf16_contracts::kStages;
        const int next_stage = k_stage + 1;

        if (next_stage < num_k_stages) {
            issue_stage_async<load_group>(a_s, b_s, g.a, g.b, next_stage % bf16_contracts::kStages, load_id,
                                          warp_row, warp_col, next_stage);
        }

        mma_tile_stage(acc, a_s[stage_slot][row_worker], b_s[stage_slot][col_worker]);

        if (next_stage < num_k_stages) {
            __syncthreads();
        }
    }

    export_accumulators(out, acc);
    kittens::warp::store(g.c, out, {0, 0, warp_row + row_worker, warp_col + col_worker});
}

} // namespace kittens::arch::c500::gemm
