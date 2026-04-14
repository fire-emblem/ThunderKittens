#pragma once

#include "../bf16_contracts.cuh"
#include "../bf16_epilogue.cuh"
#include "../bf16_operand_stage.cuh"
#include "../bf16_stage_primitives.cuh"
#include "../../primitives/pipeline.cuh"

#ifndef BF16_C500_EXPERIMENTAL_LAYOUTA_NATIVE
#define BF16_C500_EXPERIMENTAL_LAYOUTA_NATIVE 0
#endif

namespace kittens::arch::c500::gemm::families {

struct bf16_balanced_128x128x128_stage4 {
    using contracts = kittens::arch::c500::gemm::contracts::bf16_balanced_128x128x128_stage4;
    using atom = kittens::arch::c500::gemm::bf16_stage_atom;
    using shared_tile_a = st_bf<contracts::kWarpM, contracts::kStageK>;
    using shared_tile_b = st_bf<contracts::kStageK, contracts::kWarpN>;
    using shared_tile_c = st_bf<contracts::kWarpM, contracts::kWarpN>;
    using reg_tile_c = rt_fl<contracts::kWarpM, contracts::kWarpN>;
    using frag_a = fragment_a<atom>;
    using frag_b = fragment_b<atom>;
    using frag_c = fragment_c<atom>;

    static constexpr int kAtomsM = contracts::kWarpM / atom::M;
    static constexpr int kAtomsN = contracts::kWarpN / atom::N;
    static constexpr int kStageAtomsK = contracts::kStageK / atom::K;
    static constexpr int kLoadGroupThreads =
        (contracts::kNumWorkers / contracts::kLoadGroups) * contracts::kWaveSize;
    static constexpr int kAsyncElemsPerTransfer = sizeof(float4) / sizeof(bf16);
    static constexpr int kAsyncTransactionsPerStage =
        (contracts::kWarpM * contracts::kStageK) / (kLoadGroupThreads * kAsyncElemsPerTransfer) +
        (contracts::kStageK * contracts::kWarpN) / (kLoadGroupThreads * kAsyncElemsPerTransfer);
    static constexpr int kOperandStageAsyncTransactions =
        contracts::operand_layout::kAsyncTransactionCount;

    __device__ static inline void zero_accumulators(frag_c (&acc)[kAtomsM][kAtomsN]) {
#pragma unroll
        for (int m = 0; m < kAtomsM; ++m) {
#pragma unroll
            for (int n = 0; n < kAtomsN; ++n) {
#pragma unroll
                for (int r = 0; r < atom::c_registers; ++r) {
                    acc[m][n].reg[r] = 0.0f;
                }
            }
        }
    }

    template<typename OutputTile>
    __device__ static inline void export_accumulators(OutputTile &dst, const frag_c (&acc)[kAtomsM][kAtomsN]) {
#pragma unroll
        for (int m = 0; m < kAtomsM; ++m) {
#pragma unroll
            for (int n = 0; n < kAtomsN; ++n) {
                store_epilogue(dst, acc[m][n], m, n);
            }
        }
    }

    template<typename GlobalC>
    __device__ static inline void store_accumulators_layouta(const GlobalC &dst,
                                                             const frag_c (&acc)[kAtomsM][kAtomsN],
                                                             int block_row,
                                                             int block_col,
                                                             int row_group,
                                                             int col_group) {
        const int lane = kittens::laneid();
        const int lane_row = lane & 0x0f;
        const int lane_group = lane >> 4;

#pragma unroll
        for (int m = 0; m < kAtomsM; ++m) {
            const int row = block_row + row_group * 16 + m * 32 + lane_row;
#pragma unroll
            for (int n = 0; n < kAtomsN; ++n) {
                const int col_base = block_col + col_group * 16 + n * 32 + lane_group * 4;
#pragma unroll
                for (int reg = 0; reg < atom::c_registers; ++reg) {
                    dst.raw_ptr[row * dst.template stride<2>() + col_base + reg] = __float2bfloat16(acc[m][n].reg[reg]);
                }
            }
        }
    }

    __device__ static inline void wait_stage_window(int outstanding_stages) {
        kittens::arch::c500::primitives::wait_stage_window<kAsyncTransactionsPerStage>(outstanding_stages);
    }

    template<int RemainingOutstanding = 0>
    __device__ static inline void wait_operand_stage_window() {
        kittens::arch::c500::primitives::wait_until<RemainingOutstanding>();
    }

    template<typename Globals>
    __device__ static inline async_token<kAsyncTransactionsPerStage>
    issue_stage_async(bf16_stage_ring &ring,
                      const Globals &g,
                      int load_id,
                      int warp_row,
                      int warp_col,
                      int stage_slot,
                      int k_stage) {
        return kittens::arch::c500::gemm::issue_ab_stage_async(ring,
                                                               g.a,
                                                               g.b,
                                                               stage_slot,
                                                               load_id,
                                                               warp_row,
                                                               warp_col,
                                                               k_stage);
    }

    template<typename Globals>
    __device__ static inline async_token<1>
    issue_stage_async_segment(bf16_stage_ring &ring,
                              const Globals &g,
                              int load_id,
                              int warp_row,
                              int warp_col,
                              int stage_slot,
                              int k_stage,
                              int segment) {
        return kittens::arch::c500::gemm::issue_ab_stage_async_segment(
            ring, g.a, g.b, stage_slot, load_id, warp_row, warp_col, k_stage, segment);
    }

    __device__ static inline void consume_stage(const bf16_stage_ring &ring,
                                                int stage_slot,
                                                int row_worker,
                                                int col_worker,
                                                frag_c (&acc)[kAtomsM][kAtomsN]) {
        mma_raw_stage_aligned_tile_bridge(ring, stage_slot, row_worker, col_worker, acc);
    }

    template<int MBegin, int MEnd, int NBegin, int NEnd>
    __device__ static inline void consume_stage_quadrant(const bf16_stage_ring &ring,
                                                         int stage_slot,
                                                         int row_worker,
                                                         int col_worker,
                                                         frag_c (&acc)[kAtomsM][kAtomsN]) {
        mma_raw_stage_aligned_tile_bridge_quadrant<MBegin, MEnd, NBegin, NEnd>(
            ring, stage_slot, row_worker, col_worker, acc);
    }

    template<int Stages, typename GlobalA, typename GlobalBLayoutA>
    __device__ static inline async_token<2 * kOperandStageAsyncTransactions>
    issue_operand_stage_async_layouta(bf16_operand_cta_stage_ring<Stages> &ring,
                                      const GlobalA &a,
                                      const GlobalBLayoutA &b,
                                      int stage) {
        return combine(kittens::arch::c500::gemm::issue_a_operand_stage_async(ring, a, stage),
                       kittens::arch::c500::gemm::issue_b_operand_stage_async_layouta(ring, b, stage));
    }

    template<int Stages, typename GlobalA, typename GlobalBLayoutA>
    __device__ static inline async_token<2 * kOperandStageAsyncTransactions>
    issue_operand_stage_async_layouta_aligned(bf16_operand_cta_stage_ring<Stages> &ring,
                                              const GlobalA &a,
                                              const GlobalBLayoutA &b,
                                              int stage) {
        return combine(kittens::arch::c500::gemm::issue_a_operand_stage_async_aligned(ring, a, stage),
                       kittens::arch::c500::gemm::issue_b_operand_stage_async_layouta_aligned(ring, b, stage));
    }

    template<int Stages>
    __device__ static inline void consume_operand_stage(frag_c (&acc)[kAtomsM][kAtomsN],
                                                        const bf16_operand_cta_stage_ring<Stages> &ring,
                                                        int stage_slot,
                                                        int row_group,
                                                        int col_group) {
        mma_operand_stage(acc, ring, stage_slot, row_group, col_group);
    }

    template<int Stages>
    __device__ static inline void mma_operand_stage(frag_c (&acc)[kAtomsM][kAtomsN],
                                                    const bf16_operand_cta_stage_ring<Stages> &ring,
                                                    int stage_slot,
                                                    int row_group,
                                                    int col_group) {
        const int lane = kittens::laneid();

#pragma unroll
        for (int mma_k = 0; mma_k < kStageAtomsK; ++mma_k) {
            const int kg = mma_k * 2;
            bf16_operand_vec a_words[kAtomsM];
            bf16_operand_vec b_words[kAtomsN];

#pragma unroll
            for (int m = 0; m < kAtomsM; ++m) {
                a_words[m] = load_cta_a_operand_words(ring, stage_slot, row_group, m, kg, lane);
            }

#pragma unroll
            for (int n = 0; n < kAtomsN; ++n) {
                b_words[n] = load_cta_b_operand_words(ring, stage_slot, col_group, n, kg, lane);
            }

            frag_a a_frag[kAtomsM];
            frag_b b_frag[kAtomsN];

#pragma unroll
            for (int m = 0; m < kAtomsM; ++m) {
                a_frag[m] = make_a_operand_fragment(a_words[m], 0);
            }

#pragma unroll
            for (int n = 0; n < kAtomsN; ++n) {
                b_frag[n] = make_b_operand_fragment(b_words[n], 0);
            }

#pragma unroll
            for (int m = 0; m < kAtomsM; ++m) {
#pragma unroll
                for (int n = 0; n < kAtomsN; ++n) {
                    frag_c next{};
                    mma<atom>(next, a_frag[m], b_frag[n], acc[m][n]);
                    acc[m][n] = next;
                }
            }
        }
    }

    template<int M, int N, int K, typename Globals>
    __device__ static inline void run(const Globals &g) {
        using load_group = kittens::group<(contracts::kNumWorkers / contracts::kLoadGroups)>;

        static_assert(K % contracts::kStageK == 0,
                      "C500 bf16 mainloop requires K to be a multiple of 32.");

        const int workerid = kittens::warpid();
        const int row_worker = workerid / contracts::kWaveN;
        const int col_worker = workerid % contracts::kWaveN;
        const int load_id = load_group::groupid();

        const int warp_row = contracts::kLoadGroups * blockIdx.y;
        const int warp_col = contracts::kLoadGroups * blockIdx.x;

        __shared__ bf16_stage_ring ring;

        frag_c acc[kAtomsM][kAtomsN];
        reg_tile_c out;
        zero_accumulators(acc);

        constexpr int num_k_stages = K / contracts::kStageK;
        constexpr int kPrefetchStages = num_k_stages < contracts::kStages ? num_k_stages : contracts::kStages;
#pragma unroll
        for (int prefetch = 0; prefetch < kPrefetchStages; ++prefetch) {
            issue_stage_async(ring, g, load_id, warp_row, warp_col, prefetch, prefetch);
        }
        wait_stage_window(kPrefetchStages - 1);
        __syncthreads();

        for (int k_stage = 0; k_stage < num_k_stages; ++k_stage) {
            const int stage_slot = k_stage % contracts::kStages;
            const int next_stage = k_stage + kPrefetchStages;
            const bool has_next = next_stage < num_k_stages;

            consume_stage(ring, stage_slot, row_worker, col_worker, acc);

            if (has_next) {
                issue_stage_async(ring, g, load_id, warp_row, warp_col, stage_slot, next_stage);
            }

            const int remaining_after_current = num_k_stages - (k_stage + 1);
            if (remaining_after_current > 0) {
                const int outstanding_window = min(kPrefetchStages - 1, max(0, remaining_after_current - 1));
                wait_stage_window(outstanding_window);
                __syncthreads();
            }
        }

        export_accumulators(out, acc);
        kittens::warp::store(g.c, out, {0, 0, warp_row + row_worker, warp_col + col_worker});
    }

    template<int M, int N, int K, typename Globals>
    __device__ static inline void run_layouta(const Globals &g) {
        static_assert(K % contracts::kStageK == 0,
                      "C500 layoutA mainloop requires K to be a multiple of 32.");

        if constexpr (BF16_C500_EXPERIMENTAL_LAYOUTA_NATIVE && (K % 128) == 0 && K >= 256) {
            using native_vec = __NATIVE_VECTOR__(4, uint32_t);
            using native_acc = __NATIVE_VECTOR__(4, float);
            using native_pair = __NATIVE_VECTOR__(2, uint32_t);

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
            b_ldg_offset[0][0] = b_row_offset * static_cast<int>(sizeof(native_vec)) + 0 * 16 * g.b.template stride<2>() * sizeof(bf16);
            b_ldg_offset[0][1] = b_row_offset * static_cast<int>(sizeof(native_vec)) + 1 * 16 * g.b.template stride<2>() * sizeof(bf16);
            b_ldg_offset[0][2] = b_row_offset * static_cast<int>(sizeof(native_vec)) + 2 * 16 * g.b.template stride<2>() * sizeof(bf16);
            b_ldg_offset[0][3] = b_row_offset * static_cast<int>(sizeof(native_vec)) + 3 * 16 * g.b.template stride<2>() * sizeof(bf16);
            b_ldg_offset[1][0] = b_row_offset * static_cast<int>(sizeof(native_vec)) + 4 * 16 * g.b.template stride<2>() * sizeof(bf16);
            b_ldg_offset[1][1] = b_row_offset * static_cast<int>(sizeof(native_vec)) + 5 * 16 * g.b.template stride<2>() * sizeof(bf16);
            b_ldg_offset[1][2] = b_row_offset * static_cast<int>(sizeof(native_vec)) + 6 * 16 * g.b.template stride<2>() * sizeof(bf16);
            b_ldg_offset[1][3] = b_row_offset * static_cast<int>(sizeof(native_vec)) + 7 * 16 * g.b.template stride<2>() * sizeof(bf16);

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

            native_acc acc_native[4][4];
#pragma unroll
            for (int m = 0; m < 4; ++m) {
#pragma unroll
                for (int n = 0; n < 4; ++n) {
                    acc_native[m][n] = native_acc{0.f, 0.f, 0.f, 0.f};
                }
            }

            constexpr int num_k_tiles = K / kTileKNative;
            constexpr int prefetch_tiles = num_k_tiles < kStagesNative ? num_k_tiles : kStagesNative;

#pragma unroll
            for (int i = 0; i < prefetch_tiles; ++i) {
                issue_native_stage(i, i);
            }
            kittens::arch::c500::primitives::wait_stage_prefix<4>(prefetch_tiles);

            for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
                const int stage_slot = k_tile % kStagesNative;
                const int stage_off = 0x4000 * stage_slot;
                native_vec a_frag[4];
                native_vec b_frag[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    a_frag[i] = *reinterpret_cast<native_vec *>(wsm_lds + stage_off + a_lds_offset[i]);
                    b_frag[i] = *reinterpret_cast<native_vec *>(wsm_lds + stage_off + b_lds_offset[i]);
                }

                const int refill_tile = k_tile + prefetch_tiles;
                if (refill_tile < num_k_tiles) {
                    a_ptr += 128 * sizeof(bf16);
                    b_ptr += 128 * sizeof(bf16);
                    issue_native_stage(stage_slot, refill_tile);
                }

#pragma unroll
                for (int m = 0; m < 4; ++m) {
#pragma unroll
                    for (int n = 0; n < 4; ++n) {
                        acc_native[m][n] = __builtin_mxc_mma_16x16x16bf16(
                            native_pair{b_frag[n][0], b_frag[n][1]},
                            native_pair{a_frag[m][0], a_frag[m][1]},
                            acc_native[m][n]);
                        acc_native[m][n] = __builtin_mxc_mma_16x16x16bf16(
                            native_pair{b_frag[n][2], b_frag[n][3]},
                            native_pair{a_frag[m][2], a_frag[m][3]},
                            acc_native[m][n]);
                    }
                }

                if (k_tile + 1 < num_k_tiles) {
                    const int remaining_after_consume = num_k_tiles - (k_tile + 1);
                    const int wait_window = min(prefetch_tiles - 1, max(0, remaining_after_consume - 1));
                    kittens::arch::c500::primitives::wait_stage_window<4>(wait_window);
                }
            }

            frag_c acc[kAtomsM][kAtomsN];
#pragma unroll
            for (int m = 0; m < 4; ++m) {
#pragma unroll
                for (int n = 0; n < 4; ++n) {
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
            return;
        }

        const int workerid = kittens::warpid();
        const int row_group = workerid / contracts::kWaveN;
        const int col_group = workerid % contracts::kWaveN;

        __shared__ bf16_operand_cta_stage_ring<1> ring;

        frag_c acc[kAtomsM][kAtomsN];
        zero_accumulators(acc);

        constexpr int num_k_stages = K / contracts::kStageK;

        for (int k_stage = 0; k_stage < num_k_stages; ++k_stage) {
            auto a_tile = g.a;
            auto b_tile = g.b;
            a_tile.raw_ptr = &g.a.raw_ptr[(blockIdx.y * contracts::kBlockM) * g.a.template stride<2>() +
                                          k_stage * contracts::kStageK];
            b_tile.raw_ptr = &g.b.raw_ptr[(blockIdx.x * contracts::kBlockN) * g.b.template stride<2>() +
                                          k_stage * contracts::kStageK];
            auto tok = issue_operand_stage_async_layouta_aligned(ring, a_tile, b_tile, 0);
            (void)tok;
            wait_operand_stage_window<>();
            __syncthreads();
            consume_operand_stage(acc, ring, 0, row_group, col_group);
            __syncthreads();
        }

        store_accumulators_layouta(g.c,
                                   acc,
                                   blockIdx.y * contracts::kBlockM,
                                   blockIdx.x * contracts::kBlockN,
                                   row_group,
                                   col_group);
    }
};

template<int M, int N, int K, typename Globals>
__device__ inline void run_bf16_balanced_128x128x128_stage4(const Globals &g) {
    bf16_balanced_128x128x128_stage4::template run<M, N, K>(g);
}

template<int M, int N, int K, typename Globals>
__device__ inline void run_bf16_balanced_128x128x128_stage4_layouta(const Globals &g) {
    bf16_balanced_128x128x128_stage4::template run_layouta<M, N, K>(g);
}

} // namespace kittens::arch::c500::gemm::families
