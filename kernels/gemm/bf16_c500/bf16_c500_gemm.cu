#include <cuda_runtime.h>

#include <cstdlib>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../common.cuh"
#include "kittens.cuh"
#include "arch/c500/gemm/bf16_contracts.cuh"
#include "arch/c500/gemm/bf16_mainloop.cuh"

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
using shared_tileA = kittens::arch::c500::gemm::bf16_shared_tile_a;
using shared_tileB = kittens::arch::c500::gemm::bf16_shared_tile_b;
using shared_tileC = kittens::arch::c500::gemm::bf16_shared_tile_c;

template<int M, int K>
using a_gl = gl<bf16, 1, 1, M, K, shared_tileA>;
template<int K, int N>
using b_gl = gl<bf16, 1, 1, K, N, shared_tileB>;
template<int M, int N>
using c_gl = gl<bf16, 1, 1, M, N, shared_tileC>;

template<int M, int N, int K>
struct gemm_globals {
    a_gl<M, K> a;
    b_gl<K, N> b;
    c_gl<M, N> c;
};

template<int M, int N, int K>
__host__ gemm_globals<M, N, K> gemm_init(bf16 *d_a, bf16 *d_b, bf16 *d_c) {
    a_gl<M, K> a_arg{d_a, nullptr, nullptr, nullptr, nullptr};
    b_gl<K, N> b_arg{d_b, nullptr, nullptr, nullptr, nullptr};
    c_gl<M, N> c_arg{d_c, nullptr, nullptr, nullptr, nullptr};
    return {a_arg, b_arg, c_arg};
}

template<int M, int N, int K>
__global__ __launch_bounds__(contracts::kThreads)
void gemm_kernel(const __grid_constant__ gemm_globals<M, N, K> g) {
    kittens::arch::c500::gemm::run_bf16_mainloop<M, N, K>(g);
}

template<int M, int N, int K>
__host__ void launch_gemm(bf16 *a, bf16 *b, bf16 *c) {
    static_assert(M % contracts::kBlockM == 0 && N % contracts::kBlockN == 0,
                  "Task 4 minimal bf16_c500 path assumes M and N are multiples of 128.");
    static_assert(K % contracts::kBlockK == 0,
                  "Task 4 minimal bf16_c500 path assumes K is a multiple of 128.");

    const dim3 grid(N / contracts::kBlockN, M / contracts::kBlockM);
    auto g = gemm_init<M, N, K>(a, b, c);
    gemm_kernel<M, N, K><<<grid, contracts::kThreads>>>(g);
}

int run() {
    constexpr int M = BF16_C500_PROBLEM_M;
    constexpr int N = BF16_C500_PROBLEM_N;
    constexpr int K = BF16_C500_PROBLEM_K;
    constexpr float kMaxRelativeThreshold = 0.01f;

    static_assert(M % contracts::kBlockM == 0, "M must be a multiple of 128.");
    static_assert(N % contracts::kBlockN == 0, "N must be a multiple of 128.");
    static_assert(K % contracts::kBlockK == 0, "K must be a multiple of 128.");

    auto require = [](cudaError_t status, const char *what) {
        if (status != cudaSuccess) {
            std::cerr << what << ": " << cudaGetErrorString(status) << std::endl;
            std::exit(1);
        }
    };

    std::cout << "bf16_c500 TK GEMM" << std::endl;
    std::cout << "Problem size: M=" << M << ", N=" << N << ", K=" << K << std::endl;

    const size_t size_a = static_cast<size_t>(M) * K;
    const size_t size_b = static_cast<size_t>(K) * N;
    const size_t size_c = static_cast<size_t>(M) * N;

    __nv_bfloat16 *d_a = nullptr, *d_b = nullptr, *d_c = nullptr, *d_ref = nullptr;
    require(cudaMalloc(&d_a, size_a * sizeof(__nv_bfloat16)), "cudaMalloc(A)");
    require(cudaMalloc(&d_b, size_b * sizeof(__nv_bfloat16)), "cudaMalloc(B)");
    require(cudaMalloc(&d_c, size_c * sizeof(__nv_bfloat16)), "cudaMalloc(C)");
    require(cudaMalloc(&d_ref, size_c * sizeof(__nv_bfloat16)), "cudaMalloc(C_ref)");

    fill<__nv_bfloat16, FillMode::RANDOM>(d_a, size_a, 2024, -1.0f, 1.0f);
    fill<__nv_bfloat16, FillMode::RANDOM>(d_b, size_b, 2025, -1.0f, 1.0f);
    fill<__nv_bfloat16, FillMode::CONSTANT>(d_c, size_c, 0.0f);
    fill<__nv_bfloat16, FillMode::CONSTANT>(d_ref, size_c, 0.0f);
    require(cudaDeviceSynchronize(), "fill synchronize");

    reference_gemm<__nv_bfloat16, __nv_bfloat16, false>(d_ref, d_a, d_b, M, N, K);
    require(cudaGetLastError(), "reference_gemm launch");
    require(cudaDeviceSynchronize(), "reference_gemm synchronize");

    cudaEvent_t start, stop;
    require(cudaEventCreate(&start), "cudaEventCreate(start)");
    require(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    fill<__nv_bfloat16, FillMode::CONSTANT>(d_c, size_c, 0.0f);
    require(cudaDeviceSynchronize(), "clear output synchronize");

    require(cudaEventRecord(start), "cudaEventRecord(start)");
    launch_gemm<M, N, K>(reinterpret_cast<bf16 *>(d_a),
                         reinterpret_cast<bf16 *>(d_b),
                         reinterpret_cast<bf16 *>(d_c));
    require(cudaGetLastError(), "gemm launch");
    require(cudaEventRecord(stop), "cudaEventRecord(stop)");
    require(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
    require(cudaDeviceSynchronize(), "gemm synchronize");

    float elapsed_ms = 0.0f;
    require(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");

    std::vector<__nv_bfloat16> h_out(size_c);
    std::vector<__nv_bfloat16> h_ref(size_c);
    require(cudaMemcpy(h_out.data(), d_c, size_c * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), "copy output");
    require(cudaMemcpy(h_ref.data(), d_ref, size_c * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), "copy reference");

    double abs_sum = 0.0;
    double err_sum = 0.0;
    float abs_max = 0.0f;
    float err_max = 0.0f;
    for (size_t i = 0; i < size_c; ++i) {
        const float got = __bfloat162float(h_out[i]);
        const float ref = __bfloat162float(h_ref[i]);
        const float abs_err = std::abs(got - ref);
        const float rel_err = abs_err / std::max(1.0f, std::abs(ref));
        abs_sum += abs_err;
        err_sum += rel_err;
        abs_max = std::max(abs_max, abs_err);
        err_max = std::max(err_max, rel_err);
    }

    const bool pass = err_max <= kMaxRelativeThreshold;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Kernel runtime: " << elapsed_ms << " ms" << std::endl;
    std::cout << "abs mean:      " << (abs_sum / size_c) << std::endl;
    std::cout << "abs max:       " << abs_max << std::endl;
    std::cout << "err mean:      " << (err_sum / size_c) << std::endl;
    std::cout << "err max:       " << err_max << std::endl;
    std::cout << "Status: " << (pass ? "PASS" : "FAIL") << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_ref);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return pass ? 0 : 1;
}

} // namespace bf16_c500

int main() { return bf16_c500::run(); }
