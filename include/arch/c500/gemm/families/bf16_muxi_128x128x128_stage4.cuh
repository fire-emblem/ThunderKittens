#pragma once

#include "../bf16_contracts.cuh"
#include "../contracts/bf16_muxi_bank_contract.cuh"
#include "../contracts/bf16_muxi_frontier_contract.cuh"
#include "../primitives/bf16_layouta_native_stage.cuh"
#include "../schedulers/bf16_muxi_layouta_stage4_scheduler.cuh"
#include "../../primitives/pipeline.cuh"
#include "bf16_balanced_128x128x128_stage4.cuh"

#ifndef BF16_C500_EXPERIMENTAL_LAYOUTA_NATIVE
#define BF16_C500_EXPERIMENTAL_LAYOUTA_NATIVE 0
#endif

namespace kittens::arch::c500::gemm::families {

struct bf16_muxi_128x128x128_stage4 {
    using fallback_family = bf16_balanced_128x128x128_stage4;
    using contracts = fallback_family::contracts;
    using bank_contract = kittens::arch::c500::gemm::contracts::bf16_muxi_bank_contract;
    using frontier_contract = kittens::arch::c500::gemm::contracts::bf16_muxi_frontier_contract;
    using scheduler = kittens::arch::c500::gemm::schedulers::bf16_muxi_layouta_stage4_scheduler;
    using atom = fallback_family::atom;
    using shared_tile_a = fallback_family::shared_tile_a;
    using shared_tile_b = fallback_family::shared_tile_b;
    using shared_tile_c = fallback_family::shared_tile_c;
    using reg_tile_c = fallback_family::reg_tile_c;
    using frag_a = fallback_family::frag_a;
    using frag_b = fallback_family::frag_b;
    using frag_c = fallback_family::frag_c;

    static constexpr int kAtomsM = fallback_family::kAtomsM;
    static constexpr int kAtomsN = fallback_family::kAtomsN;

    __device__ static inline void zero_accumulators(frag_c (&acc)[kAtomsM][kAtomsN]) {
        fallback_family::zero_accumulators(acc);
    }

    template<typename GlobalC>
    __device__ static inline void store_accumulators_layouta(const GlobalC &dst,
                                                             const frag_c (&acc)[kAtomsM][kAtomsN],
                                                             int block_row,
                                                             int block_col,
                                                             int row_group,
                                                             int col_group) {
        fallback_family::store_accumulators_layouta(dst, acc, block_row, block_col, row_group, col_group);
    }

    template<int M, int N, int K, typename Globals>
    __device__ static inline void run(const Globals &g) {
        fallback_family::template run<M, N, K>(g);
    }

    template<int M, int N, int K, typename Globals>
    __device__ static inline void run_layouta(const Globals &g) {
        static_assert(K % contracts::kStageK == 0,
                      "C500 muxi layoutA mainloop requires K to be a multiple of 32.");
        static constexpr int kResidentBanks = bank_contract::kResidentBanks;
        static constexpr int kAccRows = frontier_contract::kAccRows;
        static constexpr int kAccCols = frontier_contract::kAccCols;
        static_assert(kResidentBanks == 4);

        if constexpr (!BF16_C500_EXPERIMENTAL_LAYOUTA_NATIVE || (K % 128) != 0 || K < 256) {
            fallback_family::template run_layouta<M, N, K>(g);
            return;
        } else {
            using native_vec = kittens::arch::c500::gemm::primitives::bf16_layouta_native_vec;
            using native_acc = kittens::arch::c500::gemm::primitives::bf16_layouta_native_acc;

            const int tid = threadIdx.x;
            const int slot = kittens::warpid();
            const int lane = kittens::laneid();
            const int quarter_lane = lane & 15;
            const int quarter_warp = lane >> 4;
            const int row_group = slot / contracts::kWaveN;
            const int col_group = slot % contracts::kWaveN;

            constexpr int kTileKNative = 128;
            constexpr int kStagesNative = contracts::kStages;
            constexpr int kVecElems = sizeof(native_vec) / sizeof(bf16);
            constexpr int kNativeStageTransactions = 4;

            const int lda_vec = g.a.template stride<2>() / kVecElems;
            const int ldb_vec = g.b.template stride<2>() / kVecElems;

            auto *a_ptr = reinterpret_cast<uint8_t *>(g.a.raw_ptr);
            auto *b_ptr = reinterpret_cast<uint8_t *>(g.b.raw_ptr);

            const int start_row = blockIdx.y * contracts::kBlockM;
            const int start_col = blockIdx.x * contracts::kBlockN;

            a_ptr += static_cast<size_t>(start_row) * g.a.template stride<2>() * sizeof(bf16);
            b_ptr += static_cast<size_t>(start_col) * g.b.template stride<2>() * sizeof(bf16);

            int a_ldg_offset[2][4];
            a_ldg_offset[0][0] = (tid + 16 * lda_vec * 0) * static_cast<int>(sizeof(native_vec));
            a_ldg_offset[0][1] = (tid + 16 * lda_vec * 1) * static_cast<int>(sizeof(native_vec));
            a_ldg_offset[0][2] = (tid + 16 * lda_vec * 2) * static_cast<int>(sizeof(native_vec));
            a_ldg_offset[0][3] = (tid + 16 * lda_vec * 3) * static_cast<int>(sizeof(native_vec));
            a_ldg_offset[1][0] = (tid + 16 * lda_vec * 4) * static_cast<int>(sizeof(native_vec));
            a_ldg_offset[1][1] = (tid + 16 * lda_vec * 5) * static_cast<int>(sizeof(native_vec));
            a_ldg_offset[1][2] = (tid + 16 * lda_vec * 6) * static_cast<int>(sizeof(native_vec));
            a_ldg_offset[1][3] = (tid + 16 * lda_vec * 7) * static_cast<int>(sizeof(native_vec));

            const int b_row_offset =
                (quarter_lane * g.b.template stride<2>() / kVecElems) + slot * 4 + quarter_warp;

            int b_ldg_offset[2][4];
            b_ldg_offset[0][0] = b_row_offset * static_cast<int>(sizeof(native_vec)) +
                                 0 * 16 * g.b.template stride<2>() * sizeof(bf16);
            b_ldg_offset[0][1] = b_row_offset * static_cast<int>(sizeof(native_vec)) +
                                 1 * 16 * g.b.template stride<2>() * sizeof(bf16);
            b_ldg_offset[0][2] = b_row_offset * static_cast<int>(sizeof(native_vec)) +
                                 2 * 16 * g.b.template stride<2>() * sizeof(bf16);
            b_ldg_offset[0][3] = b_row_offset * static_cast<int>(sizeof(native_vec)) +
                                 3 * 16 * g.b.template stride<2>() * sizeof(bf16);
            b_ldg_offset[1][0] = b_row_offset * static_cast<int>(sizeof(native_vec)) +
                                 4 * 16 * g.b.template stride<2>() * sizeof(bf16);
            b_ldg_offset[1][1] = b_row_offset * static_cast<int>(sizeof(native_vec)) +
                                 5 * 16 * g.b.template stride<2>() * sizeof(bf16);
            b_ldg_offset[1][2] = b_row_offset * static_cast<int>(sizeof(native_vec)) +
                                 6 * 16 * g.b.template stride<2>() * sizeof(bf16);
            b_ldg_offset[1][3] = b_row_offset * static_cast<int>(sizeof(native_vec)) +
                                 7 * 16 * g.b.template stride<2>() * sizeof(bf16);

            int a_lds_offset[4];
            int b_lds_offset[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                a_lds_offset[i] = (lane + (slot / 2) * (0x1000 / static_cast<int>(sizeof(native_vec))) +
                                   i * (0x400 / static_cast<int>(sizeof(native_vec)))) *
                                  static_cast<int>(sizeof(native_vec));
                b_lds_offset[i] = (lane + (0x2000 / static_cast<int>(sizeof(native_vec))) +
                                   (slot & 1) * (0x1000 / static_cast<int>(sizeof(native_vec))) +
                                   i * (0x400 / static_cast<int>(sizeof(native_vec)))) *
                                  static_cast<int>(sizeof(native_vec));
            }

            __shared__ KITTENS_ALIGN_AS(16) uint8_t wsm[0x10000];
            auto *wsm_ldg = wsm + slot * 0x400;
            auto *wsm_lds = wsm;

            auto issue_native_stage = [&](int stage_slot, int global_stage) {
                const int stage_off = 0x4000 * stage_slot;
                kittens::arch::c500::detail::ldg_b128_bsm_pred(
                    wsm_ldg + stage_off + 0x0000,
                    a_ptr + a_ldg_offset[0][global_stage & 3],
                    0,
                    K / kVecElems);
                kittens::arch::c500::detail::ldg_b128_bsm_pred(
                    wsm_ldg + stage_off + 0x1000,
                    a_ptr + a_ldg_offset[1][global_stage & 3],
                    0,
                    K / kVecElems);
                kittens::arch::c500::detail::ldg_b128_bsm_pred(
                    wsm_ldg + stage_off + 0x2000,
                    b_ptr + b_ldg_offset[0][global_stage & 3],
                    start_col + global_stage * 16,
                    N);
                kittens::arch::c500::detail::ldg_b128_bsm_pred(
                    wsm_ldg + stage_off + 0x3000,
                    b_ptr + b_ldg_offset[1][global_stage & 3],
                    start_col + (4 + global_stage) * 16,
                    N);
            };

            native_acc acc_native[kAccRows][kAccCols];
            kittens::arch::c500::gemm::primitives::zero_native_accumulators(acc_native);

            constexpr int num_k_tiles = K / kTileKNative;
            constexpr int prefetch_tiles = num_k_tiles < kStagesNative ? num_k_tiles : kStagesNative;

#pragma unroll
            for (int i = 0; i < prefetch_tiles; ++i) {
                issue_native_stage(i, i);
            }
            kittens::arch::c500::primitives::wait_stage_prefix<kNativeStageTransactions>(prefetch_tiles);

            for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
                const int stage_slot = k_tile % kStagesNative;
                const int stage_off = 0x4000 * stage_slot;
                kittens::arch::c500::gemm::primitives::bf16_layouta_native_stage_operands ops{};
                const int stage_offsets[4] = {stage_off, stage_off, stage_off, stage_off};
                kittens::arch::c500::gemm::primitives::load_native_stage_operands_for_rows(
                    ops, wsm_lds, stage_offsets, a_lds_offset, b_lds_offset);

                const int refill_tile = k_tile + prefetch_tiles;
                if (refill_tile < num_k_tiles) {
                    a_ptr += 128 * sizeof(bf16);
                    b_ptr += 128 * sizeof(bf16);
                    issue_native_stage(stage_slot, refill_tile);
                }

                kittens::arch::c500::gemm::primitives::consume_native_stage_full(acc_native, ops);

                if (k_tile + 1 < num_k_tiles) {
                    const int remaining_after_consume = num_k_tiles - (k_tile + 1);
                    const int wait_window = min(prefetch_tiles - 1, max(0, remaining_after_consume - 1));
                    kittens::arch::c500::primitives::wait_stage_window<kNativeStageTransactions>(wait_window);
                }
            }

            frag_c acc[kAtomsM][kAtomsN];
#pragma unroll
            for (int m = 0; m < kAccRows; ++m) {
#pragma unroll
                for (int n = 0; n < kAccCols; ++n) {
#pragma unroll
                    for (int r = 0; r < atom::c_registers; ++r) {
                        acc[m][n].reg[r] = acc_native[m][n][r];
                    }
                }
            }

            store_accumulators_layouta(g.c,
                                       acc,
                                       blockIdx.y * contracts::kBlockM,
                                       blockIdx.x * contracts::kBlockN,
                                       row_group,
                                       col_group);
        }
    }
};

template<int M, int N, int K, typename Globals>
__device__ inline void run_bf16_muxi_128x128x128_stage4(const Globals &g) {
    bf16_muxi_128x128x128_stage4::template run<M, N, K>(g);
}

template<int M, int N, int K, typename Globals>
__device__ inline void run_bf16_muxi_128x128x128_stage4_layouta(const Globals &g) {
    bf16_muxi_128x128x128_stage4::template run_layouta<M, N, K>(g);
}

} // namespace kittens::arch::c500::gemm::families
