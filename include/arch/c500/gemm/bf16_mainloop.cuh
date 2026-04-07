#pragma once

#include "bf16_contracts.cuh"
#include "families/bf16_balanced_128x128x128_stage4.cuh"

namespace kittens::arch::c500::gemm {

using bf16_mainloop_family = families::bf16_balanced_128x128x128_stage4;
using bf16_mainloop_atom = bf16_mainloop_family::atom;
using bf16_shared_tile_a = bf16_mainloop_family::shared_tile_a;
using bf16_shared_tile_b = bf16_mainloop_family::shared_tile_b;
using bf16_shared_tile_c = bf16_mainloop_family::shared_tile_c;
using bf16_reg_tile_c = bf16_mainloop_family::reg_tile_c;
using bf16_frag_a = bf16_mainloop_family::frag_a;
using bf16_frag_b = bf16_mainloop_family::frag_b;
using bf16_frag_c = bf16_mainloop_family::frag_c;

constexpr int kAtomsM = bf16_mainloop_family::kAtomsM;
constexpr int kAtomsN = bf16_mainloop_family::kAtomsN;
constexpr int kStageAtomsK = bf16_mainloop_family::kStageAtomsK;
constexpr int kOperandStageAsyncTransactions = bf16_mainloop_family::kOperandStageAsyncTransactions;

__device__ inline void zero_accumulators(bf16_frag_c (&acc)[kAtomsM][kAtomsN]) {
    bf16_mainloop_family::zero_accumulators(acc);
}

template<int Stages, typename GlobalA, typename GlobalBLayoutA>
__device__ inline async_token<2 * kOperandStageAsyncTransactions>
issue_operand_stage_async_layouta(bf16_operand_cta_stage_ring<Stages> &ring,
                                  const GlobalA &a,
                                  const GlobalBLayoutA &b,
                                  int stage) {
    return bf16_mainloop_family::template issue_operand_stage_async_layouta<Stages>(ring, a, b, stage);
}

template<int Stages, typename GlobalA, typename GlobalBLayoutA>
__device__ inline async_token<2 * kOperandStageAsyncTransactions>
issue_operand_stage_async_layouta_aligned(bf16_operand_cta_stage_ring<Stages> &ring,
                                          const GlobalA &a,
                                          const GlobalBLayoutA &b,
                                          int stage) {
    return bf16_mainloop_family::template issue_operand_stage_async_layouta_aligned<Stages>(ring, a, b, stage);
}

template<int M, int N, int K, typename Globals>
__device__ inline void run_bf16_mainloop(const Globals &g) {
    families::run_bf16_balanced_128x128x128_stage4<M, N, K>(g);
}

template<int M, int N, int K, typename Globals>
__device__ inline void run_bf16_mainloop_layouta(const Globals &g) {
    families::run_bf16_balanced_128x128x128_stage4_layouta<M, N, K>(g);
}

} // namespace kittens::arch::c500::gemm
