#pragma once

#include "../async_primitives.cuh"
#include "../mma_atoms.cuh"
#include "../layouts/lds_offsets.cuh"
#include "../layouts/operand_layouts.cuh"
#include "bf16_operand_stage.cuh"

namespace kittens::arch::c500::gemm {

using bf16_stage_atom = mma_bf16_16x16x16_fp32;
using bf16_stage_vec = __NATIVE_VECTOR__(4, uint32_t);

struct bf16_stage_ring {
    alignas(16) uint8_t bytes[bf16_128x128x128_stage_layout::kStages *
                              bf16_128x128x128_stage_layout::kStageBytes];
};

template<typename GlobalA>
__device__ inline async_token<1> issue_a_stage_async_part(bf16_stage_ring &ring,
                                                          const GlobalA &a,
                                                          int stage,
                                                          int load_group,
                                                          int warp_row,
                                                          int k_stage,
                                                          int part) {
    constexpr int kRows = 64;
    constexpr int kCols = 32;
    constexpr int kElemsPerTransfer = sizeof(float4) / sizeof(typename GlobalA::dtype);
    constexpr int kMemcpyPerRow = kCols / kElemsPerTransfer;
    constexpr int kGroupThreads = 128;
    constexpr int kTotalCalls = (kRows * kCols) / (kGroupThreads * kElemsPerTransfer);
    static_assert(kTotalCalls == 2, "C500 balanced stage A expects two async transactions per load group.");

    const int lane = threadIdx.x % kGroupThreads;
    auto *src_ptr = &a.raw_ptr[((warp_row + load_group) * kRows) * a.template stride<2>() +
                               k_stage * kCols];

    const int load_idx = part * kGroupThreads + lane;
    const int row = load_idx / kMemcpyPerRow;
    const int col = (load_idx * kElemsPerTransfer) % kCols;
    const uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(&ring.bytes[0])) +
                         bf16_128x128x128_stage_layout::a_group_offset(stage, load_group) +
                         (row * kCols + col) * sizeof(typename GlobalA::dtype);
    kittens::arch::c500::detail::ldg_b128_bsm_no_pred(
        __cvta_shared_to_generic(dst),
        src_ptr + row * a.template stride<2>() + col);

    return {};
}

template<typename GlobalB>
__device__ inline async_token<1> issue_b_stage_async_part(bf16_stage_ring &ring,
                                                          const GlobalB &b,
                                                          int stage,
                                                          int load_group,
                                                          int warp_col,
                                                          int k_stage,
                                                          int part) {
    constexpr int kRows = 32;
    constexpr int kCols = 64;
    constexpr int kElemsPerTransfer = sizeof(float4) / sizeof(typename GlobalB::dtype);
    constexpr int kMemcpyPerRow = kCols / kElemsPerTransfer;
    constexpr int kGroupThreads = 128;
    constexpr int kTotalCalls = (kRows * kCols) / (kGroupThreads * kElemsPerTransfer);
    static_assert(kTotalCalls == 2, "C500 balanced stage B expects two async transactions per load group.");

    const int lane = threadIdx.x % kGroupThreads;
    auto *src_ptr = &b.raw_ptr[(k_stage * kRows) * b.template stride<2>() +
                               (warp_col + load_group) * kCols];

    const int load_idx = part * kGroupThreads + lane;
    const int row = load_idx / kMemcpyPerRow;
    const int col = (load_idx * kElemsPerTransfer) % kCols;
    const uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(&ring.bytes[0])) +
                         bf16_128x128x128_stage_layout::b_group_offset(stage, load_group) +
                         (row * kCols + col) * sizeof(typename GlobalB::dtype);
    kittens::arch::c500::detail::ldg_b128_bsm_no_pred(
        __cvta_shared_to_generic(dst),
        src_ptr + row * b.template stride<2>() + col);

    return {};
}

template<typename GlobalA>
__device__ inline async_token<2> issue_a_stage_async(bf16_stage_ring &ring,
                                                     const GlobalA &a,
                                                     int stage,
                                                     int load_group,
                                                     int warp_row,
                                                     int k_stage) {
    return combine(issue_a_stage_async_part(ring, a, stage, load_group, warp_row, k_stage, 0),
                   issue_a_stage_async_part(ring, a, stage, load_group, warp_row, k_stage, 1));
}

template<typename GlobalB>
__device__ inline async_token<2> issue_b_stage_async(bf16_stage_ring &ring,
                                                     const GlobalB &b,
                                                     int stage,
                                                     int load_group,
                                                     int warp_col,
                                                     int k_stage) {
    return combine(issue_b_stage_async_part(ring, b, stage, load_group, warp_col, k_stage, 0),
                   issue_b_stage_async_part(ring, b, stage, load_group, warp_col, k_stage, 1));
}

template<typename GlobalA, typename GlobalB>
__device__ inline async_token<4> issue_ab_stage_async(bf16_stage_ring &ring,
                                                      const GlobalA &a,
                                                      const GlobalB &b,
                                                      int stage,
                                                      int load_group,
                                                      int warp_row,
                                                      int warp_col,
                                                      int k_stage) {
    return combine(issue_a_stage_async(ring, a, stage, load_group, warp_row, k_stage),
                   issue_b_stage_async(ring, b, stage, load_group, warp_col, k_stage));
}

template<typename GlobalA, typename GlobalB>
__device__ inline async_token<1> issue_ab_stage_async_segment(bf16_stage_ring &ring,
                                                              const GlobalA &a,
                                                              const GlobalB &b,
                                                              int stage,
                                                              int load_group,
                                                              int warp_row,
                                                              int warp_col,
                                                              int k_stage,
                                                              int segment) {
    switch (segment) {
        case 0:
            return issue_a_stage_async_part(ring, a, stage, load_group, warp_row, k_stage, 0);
        case 1:
            return issue_a_stage_async_part(ring, a, stage, load_group, warp_row, k_stage, 1);
        case 2:
            return issue_b_stage_async_part(ring, b, stage, load_group, warp_col, k_stage, 0);
        default:
            return issue_b_stage_async_part(ring, b, stage, load_group, warp_col, k_stage, 1);
    }
}

__device__ inline uint32_t stage_base_offset(int stage) {
    return bf16_128x128x128_stage_layout::stage_offset(stage);
}

__device__ inline bf16_stage_vec load_stage_a_words(const bf16_stage_ring &ring,
                                                    int stage,
                                                    int thread_idx,
                                                    int k_group) {
    const uint32_t shared =
        static_cast<uint32_t>(__cvta_generic_to_shared(&ring.bytes[0])) +
        stage_base_offset(stage) +
        lds_offset_a(thread_idx, k_group);
    return *reinterpret_cast<const bf16_stage_vec *>(__cvta_shared_to_generic(shared));
}

__device__ inline bf16_stage_vec load_stage_b_words(const bf16_stage_ring &ring,
                                                    int stage,
                                                    int thread_idx,
                                                    int k_group) {
    const uint32_t shared =
        static_cast<uint32_t>(__cvta_generic_to_shared(&ring.bytes[0])) +
        stage_base_offset(stage) +
        lds_offset_b(thread_idx, k_group);
    return *reinterpret_cast<const bf16_stage_vec *>(__cvta_shared_to_generic(shared));
}

__device__ inline bf16 raw_stage_lds_bf16(uint32_t shared_base, int elem_idx) {
    bf16 value;
    kittens::move<bf16>::lds(value, shared_base + elem_idx * static_cast<int>(sizeof(bf16)));
    return value;
}

__device__ inline uint32_t raw_stage_a_shared_base(const bf16_stage_ring &ring, int stage, int row_group) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(&ring.bytes[0])) +
           bf16_128x128x128_stage_layout::a_group_offset(stage, row_group);
}

__device__ inline uint32_t raw_stage_b_shared_base(const bf16_stage_ring &ring, int stage, int col_group) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(&ring.bytes[0])) +
           bf16_128x128x128_stage_layout::b_group_offset(stage, col_group);
}

__device__ inline bf16_operand_frag_a raw_stage_load_a_fragment(const bf16_stage_ring &ring,
                                                                int stage,
                                                                int row_group,
                                                                int m,
                                                                int tile_k) {
    const int lane = kittens::laneid();
    const int row = m * bf16_stage_atom::M + (lane & 0x0f);
    const int col_group = lane >> 4;
    const uint32_t a_shared = raw_stage_a_shared_base(ring, stage, row_group);

    bf16_operand_frag_a frag{};
    frag.reg[0] = kittens::arch::c500::detail::pack_pair(
        raw_stage_lds_bf16(a_shared, row * 32 + tile_k + col_group + 0),
        raw_stage_lds_bf16(a_shared, row * 32 + tile_k + col_group + 4));
    frag.reg[1] = kittens::arch::c500::detail::pack_pair(
        raw_stage_lds_bf16(a_shared, row * 32 + tile_k + col_group + 8),
        raw_stage_lds_bf16(a_shared, row * 32 + tile_k + col_group + 12));
    return frag;
}

__device__ inline bf16_operand_frag_b raw_stage_load_b_fragment(const bf16_stage_ring &ring,
                                                                int stage,
                                                                int col_group,
                                                                int n,
                                                                int tile_k) {
    const int lane = kittens::laneid();
    const int row_group_k = lane >> 4;
    const int col = n * bf16_stage_atom::N + (lane & 0x0f);
    const uint32_t b_shared = raw_stage_b_shared_base(ring, stage, col_group);

    bf16_operand_frag_b frag{};
    frag.reg[0] = kittens::arch::c500::detail::pack_pair(
        raw_stage_lds_bf16(b_shared, (tile_k + row_group_k + 0) * 64 + col),
        raw_stage_lds_bf16(b_shared, (tile_k + row_group_k + 4) * 64 + col));
    frag.reg[1] = kittens::arch::c500::detail::pack_pair(
        raw_stage_lds_bf16(b_shared, (tile_k + row_group_k + 8) * 64 + col),
        raw_stage_lds_bf16(b_shared, (tile_k + row_group_k + 12) * 64 + col));
    return frag;
}

__device__ inline bf16_operand_vec bridge_raw_stage_a_to_operand(const bf16_stage_ring &ring,
                                                                 int stage,
                                                                 int row_group,
                                                                 int m,
                                                                 int k_group,
                                                                 int lane) {
    (void)lane;
    const auto lo = raw_stage_load_a_fragment(ring, stage, row_group, m, k_group * 8 + 0);
    const auto hi = raw_stage_load_a_fragment(ring, stage, row_group, m, k_group * 8 + 4);
    return pack_a_operand_words(lo, hi);
}

__device__ inline bf16_operand_vec bridge_raw_stage_b_to_operand(const bf16_stage_ring &ring,
                                                                 int stage,
                                                                 int col_group,
                                                                 int n,
                                                                 int k_group,
                                                                 int lane) {
    (void)lane;
    const auto lo = raw_stage_load_b_fragment(ring, stage, col_group, n, k_group * 8 + 0);
    const auto hi = raw_stage_load_b_fragment(ring, stage, col_group, n, k_group * 8 + 4);
    return pack_b_operand_words(lo, hi);
}

__device__ inline fragment_c<bf16_stage_atom> mma_raw_stage_bridge(const bf16_stage_ring &ring,
                                                                   int stage,
                                                                   int row_group,
                                                                   int col_group,
                                                                   int m,
                                                                   int n,
                                                                   const fragment_c<bf16_stage_atom> &acc) {
    fragment_c<bf16_stage_atom> out = acc;
#pragma unroll
    for (int kg = 0; kg < bf16_native_stage_layout::kKGroups; ++kg) {
        const auto a_words = bridge_raw_stage_a_to_operand(ring, stage, row_group, m, kg, kittens::laneid());
        const auto b_words = bridge_raw_stage_b_to_operand(ring, stage, col_group, n, kg, kittens::laneid());
        fragment_c<bf16_stage_atom> next{};
        mma<bf16_stage_atom>(next, make_a_operand_fragment(a_words, 0), make_b_operand_fragment(b_words, 0), out);
        out = next;
        mma<bf16_stage_atom>(next, make_a_operand_fragment(a_words, 1), make_b_operand_fragment(b_words, 1), out);
        out = next;
    }
    return out;
}

__device__ inline bf16_operand_vec bridge_raw_stage_a_to_operand_aligned(const bf16_stage_ring &ring,
                                                                         int stage,
                                                                         int row_group,
                                                                         int m,
                                                                         int mma_k,
                                                                         int lane) {
    (void)lane;
    const auto lo = raw_stage_load_a_fragment(ring, stage, row_group, m, mma_k * 16);
    bf16_operand_vec words{};
    words[0] = lo.reg[0];
    words[1] = lo.reg[1];
    return words;
}

__device__ inline bf16_operand_vec bridge_raw_stage_b_to_operand_aligned(const bf16_stage_ring &ring,
                                                                         int stage,
                                                                         int col_group,
                                                                         int n,
                                                                         int mma_k,
                                                                         int lane) {
    (void)lane;
    const auto lo = raw_stage_load_b_fragment(ring, stage, col_group, n, mma_k * 16);
    bf16_operand_vec words{};
    words[0] = lo.reg[0];
    words[1] = lo.reg[1];
    return words;
}

__device__ inline fragment_c<bf16_stage_atom> mma_raw_stage_aligned_bridge(const bf16_stage_ring &ring,
                                                                           int stage,
                                                                           int row_group,
                                                                           int col_group,
                                                                           int m,
                                                                           int n,
                                                                           const fragment_c<bf16_stage_atom> &acc) {
    fragment_c<bf16_stage_atom> out = acc;
#pragma unroll
    for (int mma_k = 0; mma_k < 2; ++mma_k) {
        const auto a_words = bridge_raw_stage_a_to_operand_aligned(ring, stage, row_group, m, mma_k, kittens::laneid());
        const auto b_words = bridge_raw_stage_b_to_operand_aligned(ring, stage, col_group, n, mma_k, kittens::laneid());
        fragment_c<bf16_stage_atom> next{};
        mma<bf16_stage_atom>(next, make_a_operand_fragment(a_words, 0), make_b_operand_fragment(b_words, 0), out);
        out = next;
    }
    return out;
}

template<int MAtoms = bf16_native_stage_layout::kMAtoms, int NAtoms = bf16_native_stage_layout::kNAtoms>
__device__ inline void mma_raw_stage_tile_bridge(const bf16_stage_ring &ring,
                                                 int stage,
                                                 int row_group,
                                                 int col_group,
                                                 fragment_c<bf16_stage_atom> (&acc)[MAtoms][NAtoms]) {
#pragma unroll
    for (int m = 0; m < MAtoms; ++m) {
#pragma unroll
        for (int n = 0; n < NAtoms; ++n) {
            acc[m][n] = mma_raw_stage_bridge(ring, stage, row_group, col_group, m, n, acc[m][n]);
        }
    }
}

template<int MAtoms = bf16_native_stage_layout::kMAtoms, int NAtoms = bf16_native_stage_layout::kNAtoms>
__device__ inline void mma_raw_stage_aligned_tile_bridge(const bf16_stage_ring &ring,
                                                         int stage,
                                                         int row_group,
                                                         int col_group,
                                                         fragment_c<bf16_stage_atom> (&acc)[MAtoms][NAtoms]) {
#pragma unroll
    for (int m = 0; m < MAtoms; ++m) {
#pragma unroll
        for (int n = 0; n < NAtoms; ++n) {
            acc[m][n] = mma_raw_stage_aligned_bridge(ring, stage, row_group, col_group, m, n, acc[m][n]);
        }
    }
}

template<int MBegin,
         int MEnd,
         int NBegin,
         int NEnd,
         int MAtoms = bf16_native_stage_layout::kMAtoms,
         int NAtoms = bf16_native_stage_layout::kNAtoms>
__device__ inline void mma_raw_stage_aligned_tile_bridge_quadrant(
    const bf16_stage_ring &ring,
    int stage,
    int row_group,
    int col_group,
    fragment_c<bf16_stage_atom> (&acc)[MAtoms][NAtoms]) {
#pragma unroll
    for (int m = MBegin; m < MEnd; ++m) {
#pragma unroll
        for (int n = NBegin; n < NEnd; ++n) {
            acc[m][n] = mma_raw_stage_aligned_bridge(ring, stage, row_group, col_group, m, n, acc[m][n]);
        }
    }
}

__device__ inline fragment_a<bf16_stage_atom> make_a_fragment(const bf16_stage_vec &words, int mma_k) {
    fragment_a<bf16_stage_atom> frag{};
    frag.reg[0] = words[2 * mma_k + 0];
    frag.reg[1] = words[2 * mma_k + 1];
    return frag;
}

__device__ inline fragment_b<bf16_stage_atom> make_b_fragment(const bf16_stage_vec &words, int mma_k) {
    fragment_b<bf16_stage_atom> frag{};
    frag.reg[0] = words[2 * mma_k + 0];
    frag.reg[1] = words[2 * mma_k + 1];
    return frag;
}

} // namespace kittens::arch::c500::gemm
