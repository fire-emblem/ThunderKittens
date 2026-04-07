#pragma once

#include "../bf16_contracts.cuh"
#include "../bf16_epilogue.cuh"
#include "../bf16_operand_stage.cuh"
#include "../bf16_stage_primitives.cuh"

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
        switch (outstanding_stages) {
            case 0:
                kittens::arch::c500::wait_until<0>();
                break;
            case 1:
                kittens::arch::c500::wait_until<kAsyncTransactionsPerStage>();
                break;
            case 2:
                kittens::arch::c500::wait_until<2 * kAsyncTransactionsPerStage>();
                break;
            case 3:
                kittens::arch::c500::wait_until<3 * kAsyncTransactionsPerStage>();
                break;
            default:
                kittens::arch::c500::wait_until<0>();
                break;
        }
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
            issue_ab_stage_async(ring, g.a, g.b, prefetch, load_id, warp_row, warp_col, prefetch);
        }
        wait_stage_window(kPrefetchStages - 1);
        __syncthreads();

        for (int k_stage = 0; k_stage < num_k_stages; ++k_stage) {
            const int stage_slot = k_stage % contracts::kStages;
            const int next_stage = k_stage + kPrefetchStages;
            const bool has_next = next_stage < num_k_stages;

            mma_raw_stage_aligned_tile_bridge(ring, stage_slot, row_worker, col_worker, acc);

            if (has_next) {
                issue_ab_stage_async(ring, g.a, g.b, stage_slot, load_id, warp_row, warp_col, next_stage);
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
            kittens::arch::c500::wait(tok);
            __syncthreads();
            mma_operand_stage(acc, ring, 0, row_group, col_group);
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
