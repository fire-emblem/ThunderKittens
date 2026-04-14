#pragma once

#include "../contracts/bf16_layoutc_native_contracts.cuh"
#include "../primitives/bf16_layoutc_native_primitives.cuh"
#include "../../async_primitives.cuh"
#include "../../primitives/pipeline.cuh"

namespace kittens::arch::c500::gemm::families {

struct bf16_c500_layoutc_128x128x128_stage4 {
    using contracts = kittens::arch::c500::gemm::contracts::bf16_layoutc_native_128x128x128_stage4;
    using native_vec = kittens::arch::c500::gemm::primitives::bf16_layoutc_native_vec;
    using native_acc = kittens::arch::c500::gemm::primitives::bf16_layoutc_native_acc;

    static constexpr int kAccRows = 4;
    static constexpr int kAccCols = 4;
    static constexpr int kTileKNative = 128;
    static constexpr int kStagesNative = contracts::kStages;
    static constexpr int kVecElems = sizeof(native_vec) / sizeof(bf16);
    static constexpr int kNativeStageTransactions = 4;

    __device__ static inline void store_layoutc(bf16 *dst,
                                                int n,
                                                const native_acc (&acc)[kAccRows][kAccCols],
                                                int start_row,
                                                int start_col,
                                                int slot,
                                                int lane) {
        using c_store_vec = __NATIVE_VECTOR__(sizeof(bf16), uint32_t);
        auto *c_ptr = reinterpret_cast<c_store_vec *>(dst);
        const int quarter_warp = lane >> 4;
        const int quarter_lane = lane & 15;
        const int warp_store_offset =
            ((quarter_warp > 1 ? (quarter_warp + 30) : quarter_warp)) + quarter_lane * 2;
        const int warp_rows_group_begin = start_row / 16 + slot / 2 * 4;
        const int warp_cols_group_begin = start_col / 16 + (slot & 1) * 4;

#pragma unroll
        for (int j = 0; j < kAccCols; ++j) {
#pragma unroll
            for (int i = 0; i < kAccRows; ++i) {
                const size_t c_offset =
                    ((warp_rows_group_begin + i) / 2) * (4 * 8 * 16 / 4) * (n / 16) + warp_store_offset +
                    ((warp_rows_group_begin + i) % 2) * 64 + (warp_cols_group_begin + j) * 2 * 64;
                bf16 packed[4];
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                    packed[t] = __float2bfloat16(acc[i][j][t]);
                }
                c_ptr[c_offset] = *reinterpret_cast<c_store_vec *>(&packed[0]);
            }
        }
    }

    template<int M, int N, int K, typename Globals>
    __device__ static inline void run_layoutc(const Globals &g) {
        static_assert(M % contracts::kBlockM == 0, "M must be a multiple of 128.");
        static_assert(N % contracts::kBlockN == 0, "N must be a multiple of 128.");
        static_assert(K % kTileKNative == 0, "K must be a multiple of 128.");

        const int tid = threadIdx.x;
        const int slot = __builtin_mxc_readfirstlane(tid / contracts::kWaveSize);
        const int lane = tid & (contracts::kWaveSize - 1);
        const int start_row = blockIdx.x * contracts::kBlockM;
        const int start_col = blockIdx.y * contracts::kBlockN;

        auto *a_ptr = reinterpret_cast<uint8_t *>(g.a_native) +
                      static_cast<size_t>(start_row) * (K / kVecElems) * sizeof(native_vec);
        auto *b_ptr = reinterpret_cast<uint8_t *>(g.b_native) +
                      static_cast<size_t>(start_col / 128) * 64 * (128 / 16) * sizeof(native_vec);

        int a_ldg_offset[2][4];
        a_ldg_offset[0][0] = (tid + 16 * (K / kVecElems) * 0) * static_cast<int>(sizeof(native_vec));
        a_ldg_offset[0][1] = (tid + 16 * (K / kVecElems) * 1) * static_cast<int>(sizeof(native_vec));
        a_ldg_offset[0][2] = (tid + 16 * (K / kVecElems) * 2) * static_cast<int>(sizeof(native_vec));
        a_ldg_offset[0][3] = (tid + 16 * (K / kVecElems) * 3) * static_cast<int>(sizeof(native_vec));
        a_ldg_offset[1][0] = (tid + 16 * (K / kVecElems) * 4) * static_cast<int>(sizeof(native_vec));
        a_ldg_offset[1][1] = (tid + 16 * (K / kVecElems) * 5) * static_cast<int>(sizeof(native_vec));
        a_ldg_offset[1][2] = (tid + 16 * (K / kVecElems) * 6) * static_cast<int>(sizeof(native_vec));
        a_ldg_offset[1][3] = (tid + 16 * (K / kVecElems) * 7) * static_cast<int>(sizeof(native_vec));

        const int b_row_offset = lane + slot * 64 * (N / 16);
        int b_ldg_offset[2][4];
        b_ldg_offset[0][0] = (b_row_offset + 64 * 0) * static_cast<int>(sizeof(native_vec));
        b_ldg_offset[0][1] = (b_row_offset + 64 * 1) * static_cast<int>(sizeof(native_vec));
        b_ldg_offset[0][2] = (b_row_offset + 64 * 2) * static_cast<int>(sizeof(native_vec));
        b_ldg_offset[0][3] = (b_row_offset + 64 * 3) * static_cast<int>(sizeof(native_vec));
        b_ldg_offset[1][0] = (b_row_offset + 64 * 4) * static_cast<int>(sizeof(native_vec));
        b_ldg_offset[1][1] = (b_row_offset + 64 * 5) * static_cast<int>(sizeof(native_vec));
        b_ldg_offset[1][2] = (b_row_offset + 64 * 6) * static_cast<int>(sizeof(native_vec));
        b_ldg_offset[1][3] = (b_row_offset + 64 * 7) * static_cast<int>(sizeof(native_vec));

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

        auto issue_native_bank = [&](int bank_slot) {
            const int stage_off = 0x4000 * bank_slot;
            kittens::arch::c500::gemm::primitives::ldg_b128_bsm_pred<MACA_ICMP_SLT>(
                wsm_ldg + stage_off + 0x0000,
                a_ptr + a_ldg_offset[0][bank_slot],
                0,
                K / kVecElems);
            kittens::arch::c500::gemm::primitives::ldg_b128_bsm_pred<MACA_ICMP_SLT>(
                wsm_ldg + stage_off + 0x1000,
                a_ptr + a_ldg_offset[1][bank_slot],
                0,
                K / kVecElems);
            kittens::arch::c500::gemm::primitives::ldg_b128_bsm_pred<MACA_ICMP_SLT>(
                wsm_ldg + stage_off + 0x2000,
                b_ptr + b_ldg_offset[0][bank_slot],
                start_col + bank_slot * 16,
                N);
            kittens::arch::c500::gemm::primitives::ldg_b128_bsm_pred<MACA_ICMP_SLT>(
                wsm_ldg + stage_off + 0x3000,
                b_ptr + b_ldg_offset[1][bank_slot],
                start_col + (4 + bank_slot) * 16,
                N);
        };

        native_acc acc_native[kAccRows][kAccCols];
        kittens::arch::c500::gemm::primitives::zero_layoutc_native_accumulators(acc_native);

        constexpr int num_k_tiles = K / kTileKNative;
        for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
#pragma unroll
            for (int bank = 0; bank < kStagesNative; ++bank) {
                issue_native_bank(bank);
            }

            kittens::arch::c500::gemm::primitives::bf16_layoutc_native_stage_operands ops{};
            kittens::arch::c500::primitives::wait_until<12>();
#pragma unroll
            for (int kg = 0; kg < 4; ++kg) {
                ops.a[0][kg] = *reinterpret_cast<native_vec *>(wsm_lds + 0x0000 + a_lds_offset[kg]);
                ops.b[0][kg] = *reinterpret_cast<native_vec *>(wsm_lds + 0x0000 + b_lds_offset[kg]);
            }
            kittens::arch::c500::primitives::wait_until<8>();
#pragma unroll
            for (int kg = 0; kg < 4; ++kg) {
                ops.a[1][kg] = *reinterpret_cast<native_vec *>(wsm_lds + 0x4000 + a_lds_offset[kg]);
                ops.b[1][kg] = *reinterpret_cast<native_vec *>(wsm_lds + 0x4000 + b_lds_offset[kg]);
            }
            kittens::arch::c500::primitives::wait_until<4>();
#pragma unroll
            for (int kg = 0; kg < 4; ++kg) {
                ops.a[2][kg] = *reinterpret_cast<native_vec *>(wsm_lds + 0x8000 + a_lds_offset[kg]);
                ops.b[2][kg] = *reinterpret_cast<native_vec *>(wsm_lds + 0x8000 + b_lds_offset[kg]);
            }
            kittens::arch::c500::primitives::wait_until<0>();
#pragma unroll
            for (int kg = 0; kg < 4; ++kg) {
                ops.a[3][kg] = *reinterpret_cast<native_vec *>(wsm_lds + 0xC000 + a_lds_offset[kg]);
                ops.b[3][kg] = *reinterpret_cast<native_vec *>(wsm_lds + 0xC000 + b_lds_offset[kg]);
            }
            kittens::arch::c500::gemm::primitives::consume_layoutc_native_stage_full(acc_native, ops);
            a_ptr += (128 / 8) * 16 * sizeof(native_vec);
            b_ptr += static_cast<size_t>(16) * N * sizeof(native_vec);
        }

        store_layoutc(g.c, N, acc_native, start_row, start_col, slot, lane);
    }
};

template<int M, int N, int K, typename Globals>
__device__ inline void run_bf16_c500_layoutc_128x128x128_stage4(const Globals &g) {
    bf16_c500_layoutc_128x128x128_stage4::template run_layoutc<M, N, K>(g);
}

} // namespace kittens::arch::c500::gemm::families
