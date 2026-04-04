#pragma once

#include "../async.cuh"
#include "../copy_atoms.cuh"
#include "../layouts/lds_offsets.cuh"
#include "../layouts/operand_layouts.cuh"
#include "bf16_contracts.cuh"
#include "bf16_epilogue.cuh"

namespace kittens::arch::c500::gemm {

using bf16_mainloop_atom = mma_bf16_16x16x16_fp32;
using bf16_shared_tile_a = st_bf<bf16_contracts::kWarpM, bf16_contracts::kBlockK>;
using bf16_shared_tile_b = st_bf<bf16_contracts::kBlockK, bf16_contracts::kWarpN>;
using bf16_shared_tile_c = st_bf<bf16_contracts::kWarpM, bf16_contracts::kWarpN>;
using bf16_reg_tile_c = rt_fl<bf16_contracts::kWarpM, bf16_contracts::kWarpN>;

constexpr int kAtomsM = bf16_contracts::kWarpM / bf16_mainloop_atom::M;
constexpr int kAtomsN = bf16_contracts::kWarpN / bf16_mainloop_atom::N;
constexpr int kAtomsK = bf16_contracts::kBlockK / bf16_mainloop_atom::K;

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
__device__ inline void mma_tile(bf16_frag_c (&acc)[kAtomsM][kAtomsN],
                                const SharedA &a_tile,
                                const SharedB &b_tile) {
    bf16_frag_a a_frag[kAtomsM][kAtomsK];
    bf16_frag_b b_frag[kAtomsK][kAtomsN];

#pragma unroll
    for (int m = 0; m < kAtomsM; ++m) {
#pragma unroll
        for (int k = 0; k < kAtomsK; ++k) {
            load_a<bf16_mainloop_atom>(a_frag[m][k], a_tile, m * bf16_mainloop_atom::M, k * bf16_mainloop_atom::K);
        }
    }

#pragma unroll
    for (int k = 0; k < kAtomsK; ++k) {
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
            for (int k = 0; k < kAtomsK; ++k) {
                bf16_frag_c next;
                mma<bf16_mainloop_atom>(next, a_frag[m][k], b_frag[k][n], acc[m][n]);
                acc[m][n] = next;
            }
        }
    }
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

    __shared__ bf16_shared_tile_a a_s[kLoadBlocks];
    __shared__ bf16_shared_tile_b b_s[kLoadBlocks];

    bf16_frag_c acc[kAtomsM][kAtomsN];
    bf16_reg_tile_c out;
    zero_accumulators(acc);

    constexpr int num_k_tiles = K / bf16_contracts::kBlockK;
    for (int tile = 0; tile < num_k_tiles; ++tile) {
        load_group::load<2, true>(a_s[load_id], g.a, {warp_row + load_id, tile});
        load_group::load<2, true>(b_s[load_id], g.b, {tile, warp_col + load_id});
        __syncthreads();

        mma_tile(acc, a_s[row_worker], b_s[col_worker]);

        if (tile + 1 < num_k_tiles) {
            __syncthreads();
        }
    }

    export_accumulators(out, acc);
    kittens::warp::store(g.c, out, {0, 0, warp_row + row_worker, warp_col + col_worker});
}

} // namespace kittens::arch::c500::gemm
