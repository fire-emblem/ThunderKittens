#include <cuda_runtime.h>
#include <maca.h>
#include <maca_bfloat16.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../common.cuh"
#include "contracts/launch_contract.cuh"
#include "host/layout_pack.cuh"
#include "host/reference.cuh"
#include "kernel/layoutc_epilogue.cuh"

namespace bf16_c500_tk_local {

using bf16 = __maca_bfloat16;
using launch_contract = contracts::launch_contract;

#ifndef BF16_C500_MUXI_NATIVE_M
#define BF16_C500_MUXI_NATIVE_M 4096
#endif
#ifndef BF16_C500_MUXI_NATIVE_N
#define BF16_C500_MUXI_NATIVE_N 4096
#endif
#ifndef BF16_C500_MUXI_NATIVE_K
#define BF16_C500_MUXI_NATIVE_K 4096
#endif
#ifndef BF16_C500_MUXI_NATIVE_WARMUP_ITERS
#define BF16_C500_MUXI_NATIVE_WARMUP_ITERS 5
#endif
#ifndef BF16_C500_MUXI_NATIVE_PROFILE_ITERS
#define BF16_C500_MUXI_NATIVE_PROFILE_ITERS 20
#endif

inline void require(cudaError_t status, const char *what) {
    if (status != cudaSuccess) {
        std::cerr << what << ": " << cudaGetErrorString(status) << std::endl;
        std::exit(1);
    }
}

int run() {
    constexpr int M = BF16_C500_MUXI_NATIVE_M;
    constexpr int N = BF16_C500_MUXI_NATIVE_N;
    constexpr int K = BF16_C500_MUXI_NATIVE_K;
    constexpr int warmup_iters = BF16_C500_MUXI_NATIVE_WARMUP_ITERS;
    constexpr int profile_iters = BF16_C500_MUXI_NATIVE_PROFILE_ITERS;
    constexpr float alpha = launch_contract::alpha;
    constexpr float beta = launch_contract::beta;

    static_assert(M % 128 == 0 && N % 128 == 0 && K % 128 == 0);

    const size_t size_a = static_cast<size_t>(M) * K;
    const size_t size_b = static_cast<size_t>(K) * N;
    const size_t size_c = static_cast<size_t>(M) * N;

    std::vector<__nv_bfloat16> h_a(size_a);
    std::vector<__nv_bfloat16> h_b(size_b);
    for (size_t i = 0; i < size_a; ++i) {
        const float value = static_cast<float>((static_cast<int>(i * 17 + 3) % 1024) - 512) / 512.0f;
        h_a[i] = __float2bfloat16(value);
    }
    for (size_t i = 0; i < size_b; ++i) {
        const float value = static_cast<float>((static_cast<int>(i * 29 + 5) % 1024) - 512) / 512.0f;
        h_b[i] = __float2bfloat16(value);
    }

    auto h_a_native = bf16_c500_tk_local::host::make_a_native<M, K>(h_a);
    auto h_b_native = bf16_c500_tk_local::host::make_b_native<K, N>(h_b);

    __nv_bfloat16 *d_a_row = nullptr;
    __nv_bfloat16 *d_b_row = nullptr;
    bf16 *d_a_native = nullptr;
    bf16 *d_b_native = nullptr;
    bf16 *d_c = nullptr;
    __nv_bfloat16 *d_ref = nullptr;
    require(cudaMalloc(&d_a_row, size_a * sizeof(__nv_bfloat16)), "cudaMalloc(A_row)");
    require(cudaMalloc(&d_b_row, size_b * sizeof(__nv_bfloat16)), "cudaMalloc(B_row)");
    require(cudaMalloc(&d_a_native, size_a * sizeof(bf16)), "cudaMalloc(A_native)");
    require(cudaMalloc(&d_b_native, size_b * sizeof(bf16)), "cudaMalloc(B_native)");
    require(cudaMalloc(&d_c, size_c * sizeof(bf16)), "cudaMalloc(C)");
    require(cudaMalloc(&d_ref, size_c * sizeof(__nv_bfloat16)), "cudaMalloc(C_ref)");

    require(cudaMemcpy(d_a_row, h_a.data(), size_a * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "copy A_row");
    require(cudaMemcpy(d_b_row, h_b.data(), size_b * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "copy B_row");
    require(cudaMemcpy(d_a_native, h_a_native.data(), size_a * sizeof(bf16), cudaMemcpyHostToDevice), "copy A_native");
    require(cudaMemcpy(d_b_native, h_b_native.data(), size_b * sizeof(bf16), cudaMemcpyHostToDevice), "copy B_native");
    bf16_c500_tk_local::host::fill<bf16, bf16_c500_tk_local::host::FillMode::CONSTANT>(d_c, size_c, 0.0f);
    bf16_c500_tk_local::host::fill<__nv_bfloat16, bf16_c500_tk_local::host::FillMode::CONSTANT>(d_ref, size_c, 0.0f);
    require(cudaDeviceSynchronize(), "init sync");

    bf16_c500_tk_local::host::reference_gemm<__nv_bfloat16, __nv_bfloat16, true>(d_ref, d_b_row, d_a_row, N, M, K);
    require(cudaGetLastError(), "reference launch");
    require(cudaDeviceSynchronize(), "reference sync");

    const dim3 grid = launch_contract::grid(M, N);
    auto launch = [&]() {
        launch_contract::launch<bf16, bf16, float, true, false>(
            grid, d_a_native, d_b_native, d_c, M, N, K, K, K, M, alpha, beta, nullptr);
    };

    int l2_cache_size = 0;
    require(cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, 0), "cudaDeviceGetAttribute(L2)");
    const size_t l2_clear_elems = std::max<size_t>(1, (static_cast<size_t>(l2_cache_size) * 3) / sizeof(int));
    int *l2_clear = nullptr;
    require(cudaMalloc(&l2_clear, l2_clear_elems * sizeof(int)), "cudaMalloc(l2_clear)");

    for (int i = 0; i < warmup_iters; ++i) {
        cudaMemset(l2_clear, 0, l2_clear_elems * sizeof(int));
        launch();
    }
    require(cudaDeviceSynchronize(), "warmup sync");

    std::vector<cudaEvent_t> starts(profile_iters), stops(profile_iters);
    std::vector<float> milliseconds(profile_iters, 0.0f);
    for (int i = 0; i < profile_iters; ++i) {
        cudaMemset(l2_clear, 0, l2_clear_elems * sizeof(int));
        require(cudaEventCreate(&starts[i]), "cudaEventCreate(start)");
        require(cudaEventCreate(&stops[i]), "cudaEventCreate(stop)");
        require(cudaEventRecord(starts[i]), "cudaEventRecord(start)");
        launch();
        require(cudaGetLastError(), "kernel launch");
        require(cudaEventRecord(stops[i]), "cudaEventRecord(stop)");
        require(cudaEventSynchronize(stops[i]), "cudaEventSynchronize(stop)");
    }

    double total_ms = 0.0;
    for (int i = 0; i < profile_iters; ++i) {
        require(cudaEventElapsedTime(&milliseconds[i], starts[i], stops[i]), "cudaEventElapsedTime");
        total_ms += milliseconds[i];
        cudaEventDestroy(starts[i]);
        cudaEventDestroy(stops[i]);
    }

    const double runtime_ms = total_ms / profile_iters;
    const double runtime_s = runtime_ms / 1000.0;
    const double flops = 2.0 * static_cast<double>(M) * N * K;
    const double tflops = (flops / 1e12) / runtime_s;

    std::vector<bf16> h_out(size_c);
    std::vector<__nv_bfloat16> h_ref(size_c);
    require(cudaMemcpy(h_out.data(), d_c, size_c * sizeof(bf16), cudaMemcpyDeviceToHost), "copy output");
    require(cudaMemcpy(h_ref.data(), d_ref, size_c * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), "copy ref");

    double abs_sum = 0.0;
    double err_sum = 0.0;
    float abs_max = 0.0f;
    float err_max = 0.0f;
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < M; ++col) {
            const float got = bf16_c500_tk_local::host::load_layoutc_logical<M, N>(h_out, row, col);
            const float ref = __bfloat162float(h_ref[static_cast<size_t>(row) * M + col]);
            const float abs_err = std::abs(got - ref);
            const float rel_err = abs_err / std::max(1.0f, std::abs(ref));
            abs_sum += abs_err;
            err_sum += rel_err;
            abs_max = std::max(abs_max, abs_err);
            err_max = std::max(err_max, rel_err);
        }
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "bf16_c500 tk-local standalone" << std::endl;
    std::cout << "Family: " << launch_contract::family_name << std::endl;
    std::cout << "Problem size: M=" << M << ", N=" << N << ", K=" << K << std::endl;
    std::cout << "Average runtime: " << runtime_ms << " ms" << std::endl;
    std::cout << "Performance: " << tflops << " TFLOP/s" << std::endl;
    std::cout << "abs mean:      " << (abs_sum / size_c) << std::endl;
    std::cout << "abs max:       " << abs_max << std::endl;
    std::cout << "err mean:      " << (err_sum / size_c) << std::endl;
    std::cout << "err max:       " << err_max << std::endl;

    cudaFree(d_a_row);
    cudaFree(d_b_row);
    cudaFree(d_a_native);
    cudaFree(d_b_native);
    cudaFree(d_c);
    cudaFree(d_ref);
    cudaFree(l2_clear);
    return 0;
}

} // namespace bf16_c500_tk_local

int main() { return bf16_c500_tk_local::run(); }
