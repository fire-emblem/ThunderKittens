#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <iomanip>
#include <iostream>
#include <vector>

#include "../common.cuh"
#include "cute_tk/mainloop.cuh"
#include "host/layout_traits.cuh"

namespace bf16_c500_tk_cute_local::probe {

using local_t = __maca_bfloat16;
using ref_t = __nv_bfloat16;
using family = cute_tk::square_tt_256x256x64_stage4_family;

#ifndef CUTE_TK_SQUARE_TT_PROBE_M
#define CUTE_TK_SQUARE_TT_PROBE_M 256
#endif
#ifndef CUTE_TK_SQUARE_TT_PROBE_N
#define CUTE_TK_SQUARE_TT_PROBE_N 256
#endif
#ifndef CUTE_TK_SQUARE_TT_PROBE_K
#define CUTE_TK_SQUARE_TT_PROBE_K 64
#endif

inline void require(cudaError_t status, const char *what) {
    if (status != cudaSuccess) {
        std::cerr << what << ": " << cudaGetErrorString(status) << std::endl;
        std::exit(1);
    }
}

template <typename T>
inline T host_from_float(float value) {
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __float2bfloat16(value);
    } else {
        return static_cast<T>(value);
    }
}

int run() {
    constexpr int M = CUTE_TK_SQUARE_TT_PROBE_M;
    constexpr int N = CUTE_TK_SQUARE_TT_PROBE_N;
    constexpr int K = CUTE_TK_SQUARE_TT_PROBE_K;
    constexpr int warmup = 1;
    constexpr int profile = 10;
    constexpr float alpha = family::alpha;
    constexpr float beta = family::beta;

    std::vector<ref_t> h_a(static_cast<size_t>(M) * K);
    std::vector<ref_t> h_b(static_cast<size_t>(N) * K);
    for (size_t i = 0; i < h_a.size(); ++i) {
        float value = static_cast<float>((static_cast<int>(i * 17 + 3) % 1024) - 512) / 512.0f;
        h_a[i] = host_from_float<ref_t>(value);
    }
    for (size_t i = 0; i < h_b.size(); ++i) {
        float value = static_cast<float>((static_cast<int>(i * 29 + 5) % 1024) - 512) / 512.0f;
        h_b[i] = host_from_float<ref_t>(value);
    }

    auto h_a_native =
        ::bf16_c500_tk_local::host::square_tt_host_traits::template pack_a_runtime<local_t>(
            M, K, h_a);
    auto h_b_native =
        ::bf16_c500_tk_local::host::square_tt_host_traits::template pack_b_runtime<local_t>(
            K, N, h_b);

    ref_t *d_a_row = nullptr;
    ref_t *d_b_row = nullptr;
    local_t *d_a_native = nullptr;
    local_t *d_b_native = nullptr;
    local_t *d_c = nullptr;
    ref_t *d_ref = nullptr;
    require(cudaMalloc(&d_a_row, h_a.size() * sizeof(ref_t)), "cudaMalloc(A_row)");
    require(cudaMalloc(&d_b_row, h_b.size() * sizeof(ref_t)), "cudaMalloc(B_row)");
    require(cudaMalloc(&d_a_native, h_a_native.size() * sizeof(local_t)),
            "cudaMalloc(A_native)");
    require(cudaMalloc(&d_b_native, h_b_native.size() * sizeof(local_t)),
            "cudaMalloc(B_native)");
    require(cudaMalloc(&d_c, static_cast<size_t>(M) * N * sizeof(local_t)),
            "cudaMalloc(C)");
    require(cudaMalloc(&d_ref, static_cast<size_t>(M) * N * sizeof(ref_t)),
            "cudaMalloc(C_ref)");

    require(cudaMemcpy(d_a_row, h_a.data(), h_a.size() * sizeof(ref_t),
                       cudaMemcpyHostToDevice), "copy A_row");
    require(cudaMemcpy(d_b_row, h_b.data(), h_b.size() * sizeof(ref_t),
                       cudaMemcpyHostToDevice), "copy B_row");
    require(cudaMemcpy(d_a_native, h_a_native.data(),
                       h_a_native.size() * sizeof(local_t),
                       cudaMemcpyHostToDevice), "copy A_native");
    require(cudaMemcpy(d_b_native, h_b_native.data(),
                       h_b_native.size() * sizeof(local_t),
                       cudaMemcpyHostToDevice), "copy B_native");

    fill<local_t, FillMode::CONSTANT>(d_c, static_cast<size_t>(M) * N, 0.0f);
    fill<ref_t, FillMode::CONSTANT>(d_ref, static_cast<size_t>(M) * N, 0.0f);
    require(cudaDeviceSynchronize(), "init sync");

    reference_gemm<ref_t, ref_t, true>(d_ref, d_b_row, d_a_row, N, M, K);
    require(cudaDeviceSynchronize(), "reference sync");

    auto launch = [&]() {
        family::template launch<local_t, local_t, float, true, false>(
            family::grid(M, N), d_a_native, d_b_native, d_c, M, N, K, K, K, M,
            alpha, beta, nullptr);
    };

    for (int i = 0; i < warmup; ++i) {
        fill<local_t, FillMode::CONSTANT>(d_c, static_cast<size_t>(M) * N, 0.0f);
        launch();
    }
    require(cudaDeviceSynchronize(), "warmup sync");

    cudaEvent_t start, stop;
    require(cudaEventCreate(&start), "event start");
    require(cudaEventCreate(&stop), "event stop");
    require(cudaEventRecord(start), "record start");
    for (int i = 0; i < profile; ++i) {
        fill<local_t, FillMode::CONSTANT>(d_c, static_cast<size_t>(M) * N, 0.0f);
        launch();
    }
    require(cudaEventRecord(stop), "record stop");
    require(cudaEventSynchronize(stop), "sync stop");

    float ms = 0.0f;
    require(cudaEventElapsedTime(&ms, start, stop), "elapsed");
    double runtime_ms = static_cast<double>(ms) / profile;
    double tflops =
        (2.0 * static_cast<double>(M) * N * K / 1e12) / (runtime_ms / 1000.0);

    std::vector<local_t> h_out(static_cast<size_t>(M) * N);
    std::vector<ref_t> h_ref(static_cast<size_t>(M) * N);
    require(cudaMemcpy(h_out.data(), d_c, h_out.size() * sizeof(local_t),
                       cudaMemcpyDeviceToHost), "copy out");
    require(cudaMemcpy(h_ref.data(), d_ref, h_ref.size() * sizeof(ref_t),
                       cudaMemcpyDeviceToHost), "copy ref");

    double abs_sum = 0.0;
    double err_sum = 0.0;
    float abs_max = 0.0f;
    float err_max = 0.0f;
    double abs_sum_t = 0.0;
    double err_sum_t = 0.0;
    float abs_max_t = 0.0f;
    float err_max_t = 0.0f;
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < M; ++col) {
            const float got =
                ::bf16_c500_tk_local::host::square_tt_host_traits::template load_c_runtime<local_t>(
                    h_out, M, N, row, col);
            const float ref = static_cast<float>(
                h_ref[static_cast<size_t>(row) * M + col]);
            const float abs_err = std::abs(got - ref);
            const float rel_err = abs_err / std::max(1.0f, std::abs(ref));
            abs_sum += abs_err;
            err_sum += rel_err;
            abs_max = std::max(abs_max, abs_err);
            err_max = std::max(err_max, rel_err);

            const float ref_t = static_cast<float>(
                h_ref[static_cast<size_t>(col) * M + row]);
            const float abs_err_t = std::abs(got - ref_t);
            const float rel_err_t = abs_err_t / std::max(1.0f, std::abs(ref_t));
            abs_sum_t += abs_err_t;
            err_sum_t += rel_err_t;
            abs_max_t = std::max(abs_max_t, abs_err_t);
            err_max_t = std::max(err_max_t, rel_err_t);
        }
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "cute_tk square_tt_tile256x256x64 probe" << std::endl;
    std::cout << "Problem size: M=" << M << ", N=" << N << ", K=" << K << std::endl;
    std::cout << "Average runtime: " << runtime_ms << " ms" << std::endl;
    std::cout << "Performance: " << tflops << " TFLOP/s" << std::endl;
    std::cout << "abs mean:      " << (abs_sum / (static_cast<size_t>(M) * N))
              << std::endl;
    std::cout << "abs max:       " << abs_max << std::endl;
    std::cout << "err mean:      " << (err_sum / (static_cast<size_t>(M) * N))
              << std::endl;
    std::cout << "err max:       " << err_max << std::endl;
    std::cout << "transpose abs mean: "
              << (abs_sum_t / (static_cast<size_t>(M) * N)) << std::endl;
    std::cout << "transpose abs max:  " << abs_max_t << std::endl;
    std::cout << "transpose err mean: "
              << (err_sum_t / (static_cast<size_t>(M) * N)) << std::endl;
    std::cout << "transpose err max:  " << err_max_t << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a_row);
    cudaFree(d_b_row);
    cudaFree(d_a_native);
    cudaFree(d_b_native);
    cudaFree(d_c);
    cudaFree(d_ref);
    return 0;
}

} // namespace bf16_c500_tk_cute_local::probe

int main() { return bf16_c500_tk_cute_local::probe::run(); }
