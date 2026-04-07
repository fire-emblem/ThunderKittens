#pragma once

#include "../families/bf16_balanced_128x128x128_stage4.cuh"

namespace kittens::arch::c500::gemm::dispatch {

using bf16_default_family = families::bf16_balanced_128x128x128_stage4;

template<int M, int N, int K, typename Globals>
__device__ inline void run_bf16(const Globals &g) {
    bf16_default_family::template run<M, N, K>(g);
}

template<int M, int N, int K, typename Globals>
__device__ inline void run_bf16_layouta(const Globals &g) {
    bf16_default_family::template run_layouta<M, N, K>(g);
}

} // namespace kittens::arch::c500::gemm::dispatch
