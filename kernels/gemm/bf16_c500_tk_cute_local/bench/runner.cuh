#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <maca.h>
#include <maca_bfloat16.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <type_traits>
#include <vector>

#include "../../common.cuh"
#include "../host/reference.cuh"
#include "../mainloop.cuh"

namespace bf16_c500_tk_local::bench {

inline void require(cudaError_t status, const char *what) {
    if (status != cudaSuccess) {
        std::cerr << what << ": " << cudaGetErrorString(status) << std::endl;
        std::exit(1);
    }
}

inline bool use_muxi_timing_mode() {
    if (const char *value = std::getenv("TK_LOCAL_TIMING_MODE")) {
        return std::string_view(value) == "muxi";
    }
    return false;
}

template <typename Family>
inline bool runtime_shape_supported(int m, int n, int k) {
    if constexpr (requires { Family::supports_runtime_shape(m, n, k); }) {
        return Family::supports_runtime_shape(m, n, k);
    }
    return (m % 128) == 0 && (n % 128) == 0 && (k % 128) == 0;
}

template <typename T>
inline T host_from_float(float value) {
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __float2bfloat16(value);
    } else if constexpr (std::is_same_v<T, __half>) {
        return __float2half(value);
    } else {
        return static_cast<T>(value);
    }
}

template <typename Case>
int run_case() {
    constexpr int M = Case::m;
    constexpr int N = Case::n;
    constexpr int K = Case::k;
    constexpr int warmup_iters = Case::warmup_iters;
    constexpr int profile_iters = Case::profile_iters;
    using family = typename Case::family;
    using host_layout = typename family::host_layout;
    using local_t = typename Case::local_t;
    using ref_t = typename Case::ref_t;
    constexpr float alpha = family::alpha;
    constexpr float beta = family::beta;

    static_assert(M % 128 == 0 && N % 128 == 0 && K % 128 == 0);

    const size_t size_a = static_cast<size_t>(M) * K;
    const size_t size_b = static_cast<size_t>(K) * N;
    const size_t size_c = static_cast<size_t>(M) * N;

    std::vector<ref_t> h_a(size_a);
    std::vector<ref_t> h_b(size_b);
    for (size_t i = 0; i < size_a; ++i) {
        const float value =
            static_cast<float>((static_cast<int>(i * 17 + 3) % 1024) - 512) /
            512.0f;
        h_a[i] = host_from_float<ref_t>(value);
    }
    for (size_t i = 0; i < size_b; ++i) {
        const float value =
            static_cast<float>((static_cast<int>(i * 29 + 5) % 1024) - 512) /
            512.0f;
        h_b[i] = host_from_float<ref_t>(value);
    }

    auto h_a_native = host_layout::template pack_a_typed<local_t, ref_t, M, K>(h_a);
    auto h_b_native = host_layout::template pack_b_typed<local_t, ref_t, K, N>(h_b);

    ref_t *d_a_row = nullptr;
    ref_t *d_b_row = nullptr;
    local_t *d_a_native = nullptr;
    local_t *d_b_native = nullptr;
    local_t *d_c = nullptr;
    ref_t *d_ref = nullptr;
    require(cudaMalloc(&d_a_row, size_a * sizeof(ref_t)),
            "cudaMalloc(A_row)");
    require(cudaMalloc(&d_b_row, size_b * sizeof(ref_t)),
            "cudaMalloc(B_row)");
    require(cudaMalloc(&d_a_native, size_a * sizeof(local_t)),
            "cudaMalloc(A_native)");
    require(cudaMalloc(&d_b_native, size_b * sizeof(local_t)),
            "cudaMalloc(B_native)");
    require(cudaMalloc(&d_c, size_c * sizeof(local_t)), "cudaMalloc(C)");
    require(cudaMalloc(&d_ref, size_c * sizeof(ref_t)),
            "cudaMalloc(C_ref)");

    require(cudaMemcpy(d_a_row, h_a.data(), size_a * sizeof(ref_t),
                       cudaMemcpyHostToDevice),
            "copy A_row");
    require(cudaMemcpy(d_b_row, h_b.data(), size_b * sizeof(ref_t),
                       cudaMemcpyHostToDevice),
            "copy B_row");
    require(cudaMemcpy(d_a_native, h_a_native.data(), size_a * sizeof(local_t),
                       cudaMemcpyHostToDevice),
            "copy A_native");
    require(cudaMemcpy(d_b_native, h_b_native.data(), size_b * sizeof(local_t),
                       cudaMemcpyHostToDevice),
            "copy B_native");
    host::fill<local_t, host::FillMode::CONSTANT>(d_c, size_c, 0.0f);
    host::fill<ref_t, host::FillMode::CONSTANT>(d_ref, size_c, 0.0f);
    require(cudaDeviceSynchronize(), "init sync");

    host::reference_gemm<ref_t, ref_t, true>(d_ref, d_b_row, d_a_row, N, M, K);
    require(cudaGetLastError(), "reference launch");
    require(cudaDeviceSynchronize(), "reference sync");

    const dim3 grid = family::grid(M, N);
    auto launch = [&]() {
        if constexpr (family::requires_zero_init) {
            host::fill<local_t, host::FillMode::CONSTANT>(d_c, size_c, 0.0f);
        }
        family::template launch<local_t, local_t, float, true, false>(
            grid, d_a_native, d_b_native, d_c, M, N, K, K, K, M, alpha, beta,
            nullptr);
    };

    const bool muxi_timing = use_muxi_timing_mode();
    int *l2_clear = nullptr;
    size_t l2_clear_elems = 0;
    if (!muxi_timing) {
        int l2_cache_size = 0;
        require(cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, 0),
                "cudaDeviceGetAttribute(L2)");
        l2_clear_elems =
            std::max<size_t>(1, (static_cast<size_t>(l2_cache_size) * 3) /
                                    sizeof(int));
        require(cudaMalloc(&l2_clear, l2_clear_elems * sizeof(int)),
                "cudaMalloc(l2_clear)");
    }

    for (int i = 0; i < warmup_iters; ++i) {
        if (!muxi_timing) {
            cudaMemset(l2_clear, 0, l2_clear_elems * sizeof(int));
        }
        launch();
    }
    require(cudaDeviceSynchronize(), "warmup sync");

    double runtime_ms = 0.0;
    if (muxi_timing) {
        cudaEvent_t start, stop;
        require(cudaEventCreate(&start), "cudaEventCreate(start)");
        require(cudaEventCreate(&stop), "cudaEventCreate(stop)");
        require(cudaEventRecord(start), "cudaEventRecord(start)");
        for (int i = 0; i < profile_iters; ++i) {
            launch();
            require(cudaGetLastError(), "kernel launch");
        }
        require(cudaEventRecord(stop), "cudaEventRecord(stop)");
        require(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
        float milliseconds = 0.0f;
        require(cudaEventElapsedTime(&milliseconds, start, stop),
                "cudaEventElapsedTime");
        runtime_ms = static_cast<double>(milliseconds) / profile_iters;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    } else {
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
            require(cudaEventElapsedTime(&milliseconds[i], starts[i], stops[i]),
                    "cudaEventElapsedTime");
            total_ms += milliseconds[i];
            cudaEventDestroy(starts[i]);
            cudaEventDestroy(stops[i]);
        }
        runtime_ms = total_ms / profile_iters;
    }
    const double runtime_s = runtime_ms / 1000.0;
    const double flops = 2.0 * static_cast<double>(M) * N * K;
    const double tflops = (flops / 1e12) / runtime_s;

    std::vector<local_t> h_out(size_c);
    std::vector<ref_t> h_ref(size_c);
    require(cudaMemcpy(h_out.data(), d_c, size_c * sizeof(local_t),
                       cudaMemcpyDeviceToHost),
            "copy output");
    require(cudaMemcpy(h_ref.data(), d_ref, size_c * sizeof(ref_t),
                       cudaMemcpyDeviceToHost),
            "copy ref");

    double abs_sum = 0.0;
    double err_sum = 0.0;
    float abs_max = 0.0f;
    float err_max = 0.0f;
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < M; ++col) {
            const float got =
                host_layout::template load_c_typed<local_t, M, N>(h_out, row, col);
            const float ref = host::cast_native_to_float(
                h_ref[static_cast<size_t>(row) * M + col]);
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
    std::cout << "Case: " << Case::case_name << std::endl;
    std::cout << "Family: " << family::family_name << std::endl;
    std::cout << "Problem size: M=" << M << ", N=" << N << ", K=" << K
              << std::endl;
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
    if (l2_clear != nullptr) {
        cudaFree(l2_clear);
    }
    return 0;
}

template <typename Family, typename LocalT, typename RefT>
int run_runtime_case(const char *case_name, int M, int N, int K,
                     int warmup_iters, int profile_iters) {
    using family = Family;
    using host_layout = typename family::host_layout;
    constexpr float alpha = family::alpha;
    constexpr float beta = family::beta;

    if (!runtime_shape_supported<family>(M, N, K)) {
        std::cerr << "runtime shape unsupported by family: M=" << M
                  << " N=" << N << " K=" << K << std::endl;
        return 1;
    }

    const size_t size_a = static_cast<size_t>(M) * K;
    const size_t size_b = static_cast<size_t>(K) * N;
    const size_t size_c = static_cast<size_t>(M) * N;

    std::vector<RefT> h_a(size_a);
    std::vector<RefT> h_b(size_b);
    for (size_t i = 0; i < size_a; ++i) {
        const float value =
            static_cast<float>((static_cast<int>(i * 17 + 3) % 1024) - 512) /
            512.0f;
        h_a[i] = host_from_float<RefT>(value);
    }
    for (size_t i = 0; i < size_b; ++i) {
        const float value =
            static_cast<float>((static_cast<int>(i * 29 + 5) % 1024) - 512) /
            512.0f;
        h_b[i] = host_from_float<RefT>(value);
    }

    auto h_a_native = host_layout::template pack_a_runtime<LocalT>(M, K, h_a);
    auto h_b_native = host_layout::template pack_b_runtime<LocalT>(K, N, h_b);

    RefT *d_a_row = nullptr;
    RefT *d_b_row = nullptr;
    LocalT *d_a_native = nullptr;
    LocalT *d_b_native = nullptr;
    LocalT *d_c = nullptr;
    RefT *d_ref = nullptr;
    require(cudaMalloc(&d_a_row, size_a * sizeof(RefT)), "cudaMalloc(A_row)");
    require(cudaMalloc(&d_b_row, size_b * sizeof(RefT)), "cudaMalloc(B_row)");
    require(cudaMalloc(&d_a_native, size_a * sizeof(LocalT)),
            "cudaMalloc(A_native)");
    require(cudaMalloc(&d_b_native, size_b * sizeof(LocalT)),
            "cudaMalloc(B_native)");
    require(cudaMalloc(&d_c, size_c * sizeof(LocalT)), "cudaMalloc(C)");
    require(cudaMalloc(&d_ref, size_c * sizeof(RefT)), "cudaMalloc(C_ref)");

    require(cudaMemcpy(d_a_row, h_a.data(), size_a * sizeof(RefT),
                       cudaMemcpyHostToDevice),
            "copy A_row");
    require(cudaMemcpy(d_b_row, h_b.data(), size_b * sizeof(RefT),
                       cudaMemcpyHostToDevice),
            "copy B_row");
    require(cudaMemcpy(d_a_native, h_a_native.data(), size_a * sizeof(LocalT),
                       cudaMemcpyHostToDevice),
            "copy A_native");
    require(cudaMemcpy(d_b_native, h_b_native.data(), size_b * sizeof(LocalT),
                       cudaMemcpyHostToDevice),
            "copy B_native");
    host::fill<LocalT, host::FillMode::CONSTANT>(d_c, size_c, 0.0f);
    host::fill<RefT, host::FillMode::CONSTANT>(d_ref, size_c, 0.0f);
    require(cudaDeviceSynchronize(), "init sync");

    host::reference_gemm<RefT, RefT, true>(d_ref, d_b_row, d_a_row, N, M, K);
    require(cudaGetLastError(), "reference launch");
    require(cudaDeviceSynchronize(), "reference sync");

    const dim3 grid = family::grid(M, N);
    auto launch = [&]() {
        if constexpr (family::requires_zero_init) {
            host::fill<LocalT, host::FillMode::CONSTANT>(d_c, size_c, 0.0f);
        }
        family::template launch<LocalT, LocalT, float, true, false>(
            grid, d_a_native, d_b_native, d_c, M, N, K, K, K, M, alpha, beta,
            nullptr);
    };

    const bool muxi_timing = use_muxi_timing_mode();
    int *l2_clear = nullptr;
    size_t l2_clear_elems = 0;
    if (!muxi_timing) {
        int l2_cache_size = 0;
        require(cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, 0),
                "cudaDeviceGetAttribute(L2)");
        l2_clear_elems =
            std::max<size_t>(1, (static_cast<size_t>(l2_cache_size) * 3) /
                                    sizeof(int));
        require(cudaMalloc(&l2_clear, l2_clear_elems * sizeof(int)),
                "cudaMalloc(l2_clear)");
    }

    for (int i = 0; i < warmup_iters; ++i) {
        if (!muxi_timing) {
            cudaMemset(l2_clear, 0, l2_clear_elems * sizeof(int));
        }
        launch();
    }
    require(cudaDeviceSynchronize(), "warmup sync");

    double runtime_ms = 0.0;
    if (muxi_timing) {
        cudaEvent_t start, stop;
        require(cudaEventCreate(&start), "cudaEventCreate(start)");
        require(cudaEventCreate(&stop), "cudaEventCreate(stop)");
        require(cudaEventRecord(start), "cudaEventRecord(start)");
        for (int i = 0; i < profile_iters; ++i) {
            launch();
            require(cudaGetLastError(), "kernel launch");
        }
        require(cudaEventRecord(stop), "cudaEventRecord(stop)");
        require(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
        float milliseconds = 0.0f;
        require(cudaEventElapsedTime(&milliseconds, start, stop),
                "cudaEventElapsedTime");
        runtime_ms = static_cast<double>(milliseconds) / profile_iters;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    } else {
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
            require(cudaEventElapsedTime(&milliseconds[i], starts[i], stops[i]),
                    "cudaEventElapsedTime");
            total_ms += milliseconds[i];
            cudaEventDestroy(starts[i]);
            cudaEventDestroy(stops[i]);
        }
        runtime_ms = total_ms / profile_iters;
    }
    const double runtime_s = runtime_ms / 1000.0;
    const double flops = 2.0 * static_cast<double>(M) * N * K;
    const double tflops = (flops / 1e12) / runtime_s;

    std::vector<LocalT> h_out(size_c);
    std::vector<RefT> h_ref(size_c);
    require(cudaMemcpy(h_out.data(), d_c, size_c * sizeof(LocalT),
                       cudaMemcpyDeviceToHost),
            "copy output");
    require(cudaMemcpy(h_ref.data(), d_ref, size_c * sizeof(RefT),
                       cudaMemcpyDeviceToHost),
            "copy ref");

    double abs_sum = 0.0;
    double err_sum = 0.0;
    float abs_max = 0.0f;
    float err_max = 0.0f;
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < M; ++col) {
            const float got =
                host_layout::template load_c_runtime<LocalT>(h_out, M, N, row, col);
            const float ref = host::cast_native_to_float(
                h_ref[static_cast<size_t>(row) * M + col]);
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
    std::cout << "Case: " << case_name << std::endl;
    std::cout << "Family: " << family::family_name << std::endl;
    std::cout << "Problem size: M=" << M << ", N=" << N << ", K=" << K
              << std::endl;
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
    if (l2_clear != nullptr) {
        cudaFree(l2_clear);
    }
    return 0;
}

} // namespace bf16_c500_tk_local::bench
