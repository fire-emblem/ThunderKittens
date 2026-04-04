#pragma once

#include "../mma_atoms.cuh"
#include "../layouts/lds_offsets.cuh"
#include "../layouts/operand_layouts.cuh"

namespace kittens::arch::c500::gemm {

using bf16_stage_atom = mma_bf16_16x16x16_fp32;
using bf16_stage_vec = __NATIVE_VECTOR__(4, uint32_t);

struct bf16_stage_ring {
    alignas(16) uint8_t bytes[bf16_128x128x128_stage_layout::kStages *
                              bf16_128x128x128_stage_layout::kStageBytes];
};

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
