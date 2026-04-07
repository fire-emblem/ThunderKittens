#pragma once

#include "../async_primitives.cuh"
#include "../mma_atoms.cuh"
#include "contracts/bf16_balanced_contracts.cuh"
#include "contracts/bf16_balanced_operand_layout.cuh"

namespace kittens::arch::c500::gemm {

using bf16_operand_atom = mma_bf16_16x16x16_fp32;
using bf16_operand_vec = __NATIVE_VECTOR__(4, uint32_t);
using bf16_operand_frag_a = fragment_a<bf16_operand_atom>;
using bf16_operand_frag_b = fragment_b<bf16_operand_atom>;
using bf16_balanced_contract = contracts::bf16_balanced_128x128x128_stage4;
using bf16_balanced_operand_layout = contracts::bf16_balanced_operand_layout;

__host__ __device__ inline uint32_t pack_operand_pair(bf16 lo, bf16 hi) {
    const bf16_2 pair{lo, hi};
    return *reinterpret_cast<const uint32_t *>(&pair);
}

using bf16_balanced_operand_coords = bf16_balanced_operand_layout;

template<int Stages>
struct bf16_operand_stage_ring {
    static constexpr int kStages = Stages;
    static constexpr int kAtomsMN = bf16_balanced_operand_layout::kAtomsMN;
    static constexpr int kKGroups = bf16_balanced_operand_layout::kKGroups;
    static constexpr int kWaveSize = bf16_balanced_contract::kWaveSize;
    static constexpr int kVecBytes = sizeof(bf16_operand_vec);
    static constexpr int kOperandBytes = kAtomsMN * kKGroups * kWaveSize * kVecBytes;
    static constexpr int kAStageOffset = 0;
    static constexpr int kBStageOffset = kOperandBytes;
    static constexpr int kStageBytes = 2 * kOperandBytes;

    alignas(16) uint8_t bytes[kStages * kStageBytes];
};

using bf16_operand_stage_ring_1 = bf16_operand_stage_ring<1>;
using bf16_operand_stage_ring_4 = bf16_operand_stage_ring<bf16_balanced_contract::kStages>;

template<int Stages>
struct bf16_operand_cta_stage_ring {
    static constexpr int kStages = Stages;
    static constexpr int kAGroupCount = bf16_balanced_operand_layout::kAGroupCount;
    static constexpr int kBGroupCount = bf16_balanced_operand_layout::kBGroupCount;
    static constexpr int kAtomsMN = bf16_operand_stage_ring<Stages>::kAtomsMN;
    static constexpr int kKGroups = bf16_operand_stage_ring<Stages>::kKGroups;
    static constexpr int kWaveSize = bf16_operand_stage_ring<Stages>::kWaveSize;
    static constexpr int kVecBytes = bf16_operand_stage_ring<Stages>::kVecBytes;
    static constexpr int kGroupBytes = kAtomsMN * kKGroups * kWaveSize * kVecBytes;
    static constexpr int kAStageOffset = 0;
    static constexpr int kBStageOffset = kAGroupCount * kGroupBytes;
    static constexpr int kStageBytes = (kAGroupCount + kBGroupCount) * kGroupBytes;

    alignas(16) uint8_t bytes[kStages * kStageBytes];
};

using bf16_operand_cta_stage_ring_1 = bf16_operand_cta_stage_ring<1>;
using bf16_operand_cta_stage_ring_4 = bf16_operand_cta_stage_ring<bf16_balanced_contract::kStages>;

template<int Stages>
__host__ __device__ static constexpr int operand_stage_offset(int stage) {
    return stage * bf16_operand_stage_ring<Stages>::kStageBytes;
}

template<int Stages>
__host__ __device__ static constexpr int operand_cta_stage_offset(int stage) {
    return stage * bf16_operand_cta_stage_ring<Stages>::kStageBytes;
}

template<int Stages>
__host__ __device__ static constexpr int operand_a_stage_offset(int stage) {
    return operand_stage_offset<Stages>(stage) + bf16_operand_stage_ring<Stages>::kAStageOffset;
}

template<int Stages>
__host__ __device__ static constexpr int operand_b_stage_offset(int stage) {
    return operand_stage_offset<Stages>(stage) + bf16_operand_stage_ring<Stages>::kBStageOffset;
}

template<int Stages>
__host__ __device__ static constexpr int operand_a_offset(int stage, int m, int k_group, int lane) {
    return operand_a_stage_offset<Stages>(stage) +
           (((m * bf16_operand_stage_ring<Stages>::kKGroups + k_group) * bf16_operand_stage_ring<Stages>::kWaveSize) + lane) *
               bf16_operand_stage_ring<Stages>::kVecBytes;
}

template<int Stages>
__host__ __device__ static constexpr int operand_b_offset(int stage, int n, int k_group, int lane) {
    return operand_b_stage_offset<Stages>(stage) +
           (((n * bf16_operand_stage_ring<Stages>::kKGroups + k_group) * bf16_operand_stage_ring<Stages>::kWaveSize) + lane) *
               bf16_operand_stage_ring<Stages>::kVecBytes;
}

template<int Stages>
__host__ __device__ static constexpr int operand_cta_a_offset(int stage,
                                                              int row_group,
                                                              int m,
                                                              int k_group,
                                                              int lane) {
    return operand_cta_stage_offset<Stages>(stage) +
           bf16_operand_cta_stage_ring<Stages>::kAStageOffset +
           (((((row_group * bf16_operand_cta_stage_ring<Stages>::kAtomsMN) + m) *
               bf16_operand_cta_stage_ring<Stages>::kKGroups +
               k_group) *
                  bf16_operand_cta_stage_ring<Stages>::kWaveSize) +
             lane) *
               bf16_operand_cta_stage_ring<Stages>::kVecBytes;
}

template<int Stages>
__host__ __device__ static constexpr int operand_cta_b_offset(int stage,
                                                              int col_group,
                                                              int n,
                                                              int k_group,
                                                              int lane) {
    return operand_cta_stage_offset<Stages>(stage) +
           bf16_operand_cta_stage_ring<Stages>::kBStageOffset +
           (((((col_group * bf16_operand_cta_stage_ring<Stages>::kAtomsMN) + n) *
               bf16_operand_cta_stage_ring<Stages>::kKGroups +
               k_group) *
                  bf16_operand_cta_stage_ring<Stages>::kWaveSize) +
             lane) *
               bf16_operand_cta_stage_ring<Stages>::kVecBytes;
}

template<int Stages>
__device__ inline void store_a_operand_words(bf16_operand_stage_ring<Stages> &ring,
                                             int stage,
                                             int m,
                                             int k_group,
                                             int lane,
                                             const bf16_operand_vec &words) {
    const uint32_t shared =
        static_cast<uint32_t>(__cvta_generic_to_shared(&ring.bytes[0])) + operand_a_offset<Stages>(stage, m, k_group, lane);
    *reinterpret_cast<bf16_operand_vec *>(__cvta_shared_to_generic(shared)) = words;
}

template<int Stages>
__device__ inline void store_b_operand_words(bf16_operand_stage_ring<Stages> &ring,
                                             int stage,
                                             int n,
                                             int k_group,
                                             int lane,
                                             const bf16_operand_vec &words) {
    const uint32_t shared =
        static_cast<uint32_t>(__cvta_generic_to_shared(&ring.bytes[0])) + operand_b_offset<Stages>(stage, n, k_group, lane);
    *reinterpret_cast<bf16_operand_vec *>(__cvta_shared_to_generic(shared)) = words;
}

template<int Stages>
__device__ inline void store_cta_a_operand_words(bf16_operand_cta_stage_ring<Stages> &ring,
                                                 int stage,
                                                 int row_group,
                                                 int m,
                                                 int k_group,
                                                 int lane,
                                                 const bf16_operand_vec &words) {
    const uint32_t shared =
        static_cast<uint32_t>(__cvta_generic_to_shared(&ring.bytes[0])) +
        operand_cta_a_offset<Stages>(stage, row_group, m, k_group, lane);
    *reinterpret_cast<bf16_operand_vec *>(__cvta_shared_to_generic(shared)) = words;
}

template<int Stages>
__device__ inline void store_cta_b_operand_words(bf16_operand_cta_stage_ring<Stages> &ring,
                                                 int stage,
                                                 int col_group,
                                                 int n,
                                                 int k_group,
                                                 int lane,
                                                 const bf16_operand_vec &words) {
    const uint32_t shared =
        static_cast<uint32_t>(__cvta_generic_to_shared(&ring.bytes[0])) +
        operand_cta_b_offset<Stages>(stage, col_group, n, k_group, lane);
    *reinterpret_cast<bf16_operand_vec *>(__cvta_shared_to_generic(shared)) = words;
}

template<int Stages>
__device__ inline bf16_operand_vec load_a_operand_words(const bf16_operand_stage_ring<Stages> &ring,
                                                        int stage,
                                                        int m,
                                                        int k_group,
                                                        int lane) {
    const uint32_t shared =
        static_cast<uint32_t>(__cvta_generic_to_shared(&ring.bytes[0])) + operand_a_offset<Stages>(stage, m, k_group, lane);
    return *reinterpret_cast<const bf16_operand_vec *>(__cvta_shared_to_generic(shared));
}

template<int Stages>
__device__ inline bf16_operand_vec load_b_operand_words(const bf16_operand_stage_ring<Stages> &ring,
                                                        int stage,
                                                        int n,
                                                        int k_group,
                                                        int lane) {
    const uint32_t shared =
        static_cast<uint32_t>(__cvta_generic_to_shared(&ring.bytes[0])) + operand_b_offset<Stages>(stage, n, k_group, lane);
    return *reinterpret_cast<const bf16_operand_vec *>(__cvta_shared_to_generic(shared));
}

template<int Stages>
__device__ inline bf16_operand_vec load_cta_a_operand_words(const bf16_operand_cta_stage_ring<Stages> &ring,
                                                            int stage,
                                                            int row_group,
                                                            int m,
                                                            int k_group,
                                                            int lane) {
    const uint32_t shared =
        static_cast<uint32_t>(__cvta_generic_to_shared(&ring.bytes[0])) +
        operand_cta_a_offset<Stages>(stage, row_group, m, k_group, lane);
    return *reinterpret_cast<const bf16_operand_vec *>(__cvta_shared_to_generic(shared));
}

template<int Stages>
__device__ inline bf16_operand_vec load_cta_b_operand_words(const bf16_operand_cta_stage_ring<Stages> &ring,
                                                            int stage,
                                                            int col_group,
                                                            int n,
                                                            int k_group,
                                                            int lane) {
    const uint32_t shared =
        static_cast<uint32_t>(__cvta_generic_to_shared(&ring.bytes[0])) +
        operand_cta_b_offset<Stages>(stage, col_group, n, k_group, lane);
    return *reinterpret_cast<const bf16_operand_vec *>(__cvta_shared_to_generic(shared));
}

__device__ inline fragment_a<bf16_operand_atom> make_a_operand_fragment(const bf16_operand_vec &words, int half_k) {
    fragment_a<bf16_operand_atom> frag{};
    frag.reg[0] = words[bf16_balanced_operand_layout::kPackedHalfCount * half_k + 0];
    frag.reg[1] = words[bf16_balanced_operand_layout::kPackedHalfCount * half_k + 1];
    return frag;
}

__device__ inline fragment_b<bf16_operand_atom> make_b_operand_fragment(const bf16_operand_vec &words, int half_k) {
    fragment_b<bf16_operand_atom> frag{};
    frag.reg[0] = words[bf16_balanced_operand_layout::kPackedHalfCount * half_k + 0];
    frag.reg[1] = words[bf16_balanced_operand_layout::kPackedHalfCount * half_k + 1];
    return frag;
}

__host__ __device__ inline bf16_operand_vec pack_a_operand_words(const bf16_operand_frag_a &lo,
                                                                 const bf16_operand_frag_a &hi) {
    return bf16_operand_vec{lo.reg[0], lo.reg[1], hi.reg[0], hi.reg[1]};
}

__host__ __device__ inline bf16_operand_vec pack_b_operand_words(const bf16_operand_frag_b &lo,
                                                                 const bf16_operand_frag_b &hi) {
    return bf16_operand_vec{lo.reg[0], lo.reg[1], hi.reg[0], hi.reg[1]};
}

template<int Stages, typename GlobalA>
__device__ inline async_token<bf16_balanced_operand_layout::kAsyncTransactionCount>
issue_a_operand_stage_async(bf16_operand_cta_stage_ring<Stages> &ring, const GlobalA &a, int stage) {
    const int warp = kittens::warpid();
    const int row_group = warp / bf16_balanced_contract::kWaveN;
    const int col_group = warp % bf16_balanced_contract::kWaveN;
    const int lane = kittens::laneid();

    if (col_group == 0) {
#pragma unroll
        for (int m = 0; m < bf16_operand_cta_stage_ring<Stages>::kAtomsMN; ++m) {
#pragma unroll
            for (int kg = 0; kg < bf16_operand_cta_stage_ring<Stages>::kKGroups; ++kg) {
                const int row = row_group * bf16_balanced_operand_layout::kLaneMn +
                                m * bf16_balanced_operand_layout::kAtomRowStride +
                                bf16_balanced_operand_coords::lane_mn(lane);
                const int tile_k = kg * bf16_balanced_operand_layout::kGroupTileK;
                bf16_operand_vec words;
                words[0] = pack_operand_pair(
                    a.raw_ptr[row * a.template stride<2>() + tile_k + bf16_balanced_operand_coords::lane_group(lane) + 0],
                    a.raw_ptr[row * a.template stride<2>() + tile_k + bf16_balanced_operand_coords::lane_group(lane) +
                              bf16_balanced_operand_layout::kLaneGroupKStride]);
                words[1] = pack_operand_pair(
                    a.raw_ptr[row * a.template stride<2>() + tile_k + bf16_balanced_operand_coords::lane_group(lane) +
                              2 * bf16_balanced_operand_layout::kLaneGroupKStride],
                    a.raw_ptr[row * a.template stride<2>() + tile_k + bf16_balanced_operand_coords::lane_group(lane) +
                              3 * bf16_balanced_operand_layout::kLaneGroupKStride]);
                words[2] = pack_operand_pair(
                    a.raw_ptr[row * a.template stride<2>() + tile_k + bf16_balanced_operand_coords::lane_group(lane) +
                              bf16_balanced_operand_layout::kLaneGroupKStride],
                    a.raw_ptr[row * a.template stride<2>() + tile_k + bf16_balanced_operand_coords::lane_group(lane) +
                              2 * bf16_balanced_operand_layout::kLaneGroupKStride]);
                words[3] = pack_operand_pair(
                    a.raw_ptr[row * a.template stride<2>() + tile_k + bf16_balanced_operand_coords::lane_group(lane) +
                              3 * bf16_balanced_operand_layout::kLaneGroupKStride],
                    a.raw_ptr[row * a.template stride<2>() + tile_k + bf16_balanced_operand_coords::lane_group(lane) +
                              4 * bf16_balanced_operand_layout::kLaneGroupKStride]);
                store_cta_a_operand_words(ring, stage, row_group, m, kg, lane, words);
            }
        }
    }

    return {};
}

template<int Stages, typename GlobalBLayoutA>
__device__ inline async_token<bf16_balanced_operand_layout::kAsyncTransactionCount>
issue_b_operand_stage_async_layouta(bf16_operand_cta_stage_ring<Stages> &ring, const GlobalBLayoutA &b, int stage) {
    const int warp = kittens::warpid();
    const int row_group = warp / bf16_balanced_contract::kWaveN;
    const int col_group = warp % bf16_balanced_contract::kWaveN;
    const int lane = kittens::laneid();

    if (row_group == 0) {
#pragma unroll
        for (int n = 0; n < bf16_operand_cta_stage_ring<Stages>::kAtomsMN; ++n) {
#pragma unroll
            for (int kg = 0; kg < bf16_operand_cta_stage_ring<Stages>::kKGroups; ++kg) {
                const int row = col_group * bf16_balanced_operand_layout::kLaneMn +
                                n * bf16_balanced_operand_layout::kAtomColStride +
                                bf16_balanced_operand_coords::lane_mn(lane);
                const int tile_k = kg * bf16_balanced_operand_layout::kGroupTileK;
                bf16_operand_vec words;
                words[0] = pack_operand_pair(
                    b.raw_ptr[row * b.template stride<2>() + tile_k + bf16_balanced_operand_coords::lane_group(lane) + 0],
                    b.raw_ptr[row * b.template stride<2>() + tile_k + bf16_balanced_operand_coords::lane_group(lane) +
                              bf16_balanced_operand_layout::kLaneGroupKStride]);
                words[1] = pack_operand_pair(
                    b.raw_ptr[row * b.template stride<2>() + tile_k + bf16_balanced_operand_coords::lane_group(lane) +
                              2 * bf16_balanced_operand_layout::kLaneGroupKStride],
                    b.raw_ptr[row * b.template stride<2>() + tile_k + bf16_balanced_operand_coords::lane_group(lane) +
                              3 * bf16_balanced_operand_layout::kLaneGroupKStride]);
                words[2] = pack_operand_pair(
                    b.raw_ptr[row * b.template stride<2>() + tile_k + bf16_balanced_operand_coords::lane_group(lane) +
                              bf16_balanced_operand_layout::kLaneGroupKStride],
                    b.raw_ptr[row * b.template stride<2>() + tile_k + bf16_balanced_operand_coords::lane_group(lane) +
                              2 * bf16_balanced_operand_layout::kLaneGroupKStride]);
                words[3] = pack_operand_pair(
                    b.raw_ptr[row * b.template stride<2>() + tile_k + bf16_balanced_operand_coords::lane_group(lane) +
                              3 * bf16_balanced_operand_layout::kLaneGroupKStride],
                    b.raw_ptr[row * b.template stride<2>() + tile_k + bf16_balanced_operand_coords::lane_group(lane) +
                              4 * bf16_balanced_operand_layout::kLaneGroupKStride]);
                store_cta_b_operand_words(ring, stage, col_group, n, kg, lane, words);
            }
        }
    }

    return {};
}

template<int Stages, typename GlobalA>
__device__ inline async_token<bf16_balanced_operand_layout::kAsyncTransactionCount>
issue_a_operand_stage_async_aligned(bf16_operand_cta_stage_ring<Stages> &ring, const GlobalA &a, int stage) {
    const int warp = kittens::warpid();
    const int row_group = warp / bf16_balanced_contract::kWaveN;
    const int col_group = warp % bf16_balanced_contract::kWaveN;
    const int lane = kittens::laneid();

    if (col_group == 0) {
#pragma unroll
        for (int m = 0; m < bf16_operand_cta_stage_ring<Stages>::kAtomsMN; ++m) {
#pragma unroll
            for (int mma_k = 0; mma_k < bf16_balanced_operand_layout::kAlignedMmaSteps; ++mma_k) {
                const int kg = mma_k * bf16_balanced_operand_layout::kAlignedKGroupStride;
                const int row = row_group * bf16_balanced_operand_layout::kLaneMn +
                                m * bf16_balanced_operand_layout::kAtomRowStride +
                                bf16_balanced_operand_coords::lane_mn(lane);
                const int tile_k = mma_k * bf16_balanced_operand_layout::kAlignedTileK;
                bf16_operand_vec words{};
                words[0] = pack_operand_pair(
                    a.raw_ptr[row * a.template stride<2>() + tile_k + bf16_balanced_operand_coords::lane_group(lane) + 0],
                    a.raw_ptr[row * a.template stride<2>() + tile_k + bf16_balanced_operand_coords::lane_group(lane) +
                              bf16_balanced_operand_layout::kLaneGroupKStride]);
                words[1] = pack_operand_pair(
                    a.raw_ptr[row * a.template stride<2>() + tile_k + bf16_balanced_operand_coords::lane_group(lane) +
                              2 * bf16_balanced_operand_layout::kLaneGroupKStride],
                    a.raw_ptr[row * a.template stride<2>() + tile_k + bf16_balanced_operand_coords::lane_group(lane) +
                              3 * bf16_balanced_operand_layout::kLaneGroupKStride]);
                store_cta_a_operand_words(ring, stage, row_group, m, kg, lane, words);
            }
        }
    }

    return {};
}

template<int Stages, typename GlobalBLayoutA>
__device__ inline async_token<bf16_balanced_operand_layout::kAsyncTransactionCount>
issue_b_operand_stage_async_layouta_aligned(bf16_operand_cta_stage_ring<Stages> &ring, const GlobalBLayoutA &b, int stage) {
    const int warp = kittens::warpid();
    const int row_group = warp / bf16_balanced_contract::kWaveN;
    const int col_group = warp % bf16_balanced_contract::kWaveN;
    const int lane = kittens::laneid();

    if (row_group == 0) {
#pragma unroll
        for (int n = 0; n < bf16_operand_cta_stage_ring<Stages>::kAtomsMN; ++n) {
#pragma unroll
            for (int mma_k = 0; mma_k < bf16_balanced_operand_layout::kAlignedMmaSteps; ++mma_k) {
                const int kg = mma_k * bf16_balanced_operand_layout::kAlignedKGroupStride;
                const int row = col_group * bf16_balanced_operand_layout::kLaneMn +
                                n * bf16_balanced_operand_layout::kAtomColStride +
                                bf16_balanced_operand_coords::lane_mn(lane);
                const int tile_k = mma_k * bf16_balanced_operand_layout::kAlignedTileK;
                bf16_operand_vec words{};
                words[0] = pack_operand_pair(
                    b.raw_ptr[row * b.template stride<2>() + tile_k + bf16_balanced_operand_coords::lane_group(lane) + 0],
                    b.raw_ptr[row * b.template stride<2>() + tile_k + bf16_balanced_operand_coords::lane_group(lane) +
                              bf16_balanced_operand_layout::kLaneGroupKStride]);
                words[1] = pack_operand_pair(
                    b.raw_ptr[row * b.template stride<2>() + tile_k + bf16_balanced_operand_coords::lane_group(lane) +
                              2 * bf16_balanced_operand_layout::kLaneGroupKStride],
                    b.raw_ptr[row * b.template stride<2>() + tile_k + bf16_balanced_operand_coords::lane_group(lane) +
                              3 * bf16_balanced_operand_layout::kLaneGroupKStride]);
                store_cta_b_operand_words(ring, stage, col_group, n, kg, lane, words);
            }
        }
    }

    return {};
}

} // namespace kittens::arch::c500::gemm
