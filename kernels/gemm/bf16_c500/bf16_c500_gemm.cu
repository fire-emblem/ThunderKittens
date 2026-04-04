#include <cuda_runtime.h>

#include <iostream>

#include "../common.cuh"
#include "kittens.cuh"
#include "arch/c500/gemm/bf16_contracts.cuh"

#ifdef KITTENS_C500
#ifndef __grid_constant__
#define __grid_constant__
#endif
#endif

using namespace kittens;

namespace bf16_c500 {

#ifndef BF16_C500_PROBLEM_M
#define BF16_C500_PROBLEM_M 4096
#endif
#ifndef BF16_C500_PROBLEM_N
#define BF16_C500_PROBLEM_N 4096
#endif
#ifndef BF16_C500_PROBLEM_K
#define BF16_C500_PROBLEM_K 4096
#endif

using contracts = kittens::arch::c500::gemm::bf16_contracts;

template<int M, int K>
using a_gl = gl<bf16, 1, 1, M, K>;
template<int K, int N>
using b_gl = gl<bf16, 1, 1, K, N>;
template<int M, int N>
using c_gl = gl<bf16, 1, 1, M, N>;

template<int M, int N, int K>
struct gemm_globals {
    a_gl<M, K> a;
    b_gl<K, N> b;
    c_gl<M, N> c;
};

template<int M, int N, int K>
__global__ __launch_bounds__(contracts::kThreads) void gemm_kernel(const __grid_constant__ int) {}

int run() {
    constexpr int M = BF16_C500_PROBLEM_M;
    constexpr int N = BF16_C500_PROBLEM_N;
    constexpr int K = BF16_C500_PROBLEM_K;

    std::cout << "bf16_c500 TK GEMM" << std::endl;
    std::cout << "Problem size: M=" << M << ", N=" << N << ", K=" << K << std::endl;
    std::cout << "Skeleton only: mainloop and epilogue land in Task 4." << std::endl;
    return 0;
}

} // namespace bf16_c500

int main() { return bf16_c500::run(); }
