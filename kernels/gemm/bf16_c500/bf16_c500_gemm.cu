#include <cuda_runtime.h>

#include <cstdlib>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#ifndef BF16_C500_USE_LAYOUTA_NATIVE
#define BF16_C500_USE_LAYOUTA_NATIVE 0
#endif
#ifndef BF16_C500_USE_LAYOUTC_NATIVE
#define BF16_C500_USE_LAYOUTC_NATIVE 0
#endif
#if BF16_C500_USE_LAYOUTA_NATIVE && BF16_C500_USE_LAYOUTC_NATIVE
#error "Select only one native C500 path."
#endif

#include "../common.cuh"
#include "kittens.cuh"
#include "arch/c500/gemm/dispatch/bf16_dispatch.cuh"

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
#ifndef BF16_C500_WARMUP_ITERS
#define BF16_C500_WARMUP_ITERS 25
#endif
#ifndef BF16_C500_PROFILE_ITERS
#define BF16_C500_PROFILE_ITERS 100
#endif
using dispatch = kittens::arch::c500::gemm::dispatch::bf16_default_family;
using contracts = dispatch::contracts;
#if BF16_C500_USE_LAYOUTC_NATIVE
using layoutc_dispatch = kittens::arch::c500::gemm::dispatch::bf16_layoutc_native_family;
using layoutc_contracts = layoutc_dispatch::contracts;
#endif
using shared_tileA = dispatch::shared_tile_a;
using shared_tileB = dispatch::shared_tile_b;
using shared_tileC = dispatch::shared_tile_c;

template<int M, int K>
using a_gl = gl<bf16, 1, 1, M, K, shared_tileA>;
template<int K, int N>
using b_gl = gl<bf16, 1, 1, K, N, shared_tileB>;
template<int N, int K>
using b_layouta_gl = gl<bf16, 1, 1, N, K>;
template<int M, int N>
using c_gl = gl<bf16, 1, 1, M, N, shared_tileC>;

template<int M, int N, int K>
struct gemm_globals {
    a_gl<M, K> a;
    b_gl<K, N> b;
    c_gl<M, N> c;
};

template<int M, int N, int K>
struct gemm_globals_layouta {
    a_gl<M, K> a;
    b_layouta_gl<N, K> b;
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
__host__ gemm_globals_layouta<M, N, K> gemm_init_layouta(bf16 *d_a, bf16 *d_b_layouta, bf16 *d_c) {
    a_gl<M, K> a_arg{d_a, nullptr, nullptr, nullptr, nullptr};
    b_layouta_gl<N, K> b_arg{d_b_layouta, nullptr, nullptr, nullptr, nullptr};
    c_gl<M, N> c_arg{d_c, nullptr, nullptr, nullptr, nullptr};
    return {a_arg, b_arg, c_arg};
}

#if BF16_C500_USE_LAYOUTC_NATIVE
template<int M, int N, int K>
struct gemm_globals_layoutc {
    bf16 *a_native;
    bf16 *b_native;
    bf16 *c;
};

template<int M, int N, int K>
__host__ gemm_globals_layoutc<M, N, K> gemm_init_layoutc(bf16 *d_a_native, bf16 *d_b_native, bf16 *d_c) {
    return {d_a_native, d_b_native, d_c};
}
#endif

template<int M, int N, int K>
__global__ __launch_bounds__(contracts::kThreads)
void gemm_kernel(const __grid_constant__ gemm_globals<M, N, K> g) {
    kittens::arch::c500::gemm::dispatch::run_bf16<M, N, K>(g);
}

template<int M, int N, int K>
__global__ __launch_bounds__(contracts::kThreads)
void gemm_kernel_layouta(const __grid_constant__ gemm_globals_layouta<M, N, K> g) {
    kittens::arch::c500::gemm::dispatch::run_bf16_layouta<M, N, K>(g);
}

#if BF16_C500_USE_LAYOUTC_NATIVE
template<int M, int N, int K>
__global__ __launch_bounds__(layoutc_contracts::kThreads)
void gemm_kernel_layoutc(const __grid_constant__ gemm_globals_layoutc<M, N, K> g) {
    kittens::arch::c500::gemm::dispatch::run_bf16_layoutc<M, N, K>(g);
}
#endif

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

template<int M, int N, int K>
__host__ void launch_gemm_layouta(bf16 *a, bf16 *b_layouta, bf16 *c) {
    static_assert(M % contracts::kBlockM == 0 && N % contracts::kBlockN == 0,
                  "C500 layoutA path assumes M and N are multiples of 128.");
    static_assert(K % contracts::kStageK == 0,
                  "C500 layoutA path assumes K is a multiple of 32.");

    const dim3 grid(N / contracts::kBlockN, M / contracts::kBlockM);
    auto g = gemm_init_layouta<M, N, K>(a, b_layouta, c);
    gemm_kernel_layouta<M, N, K><<<grid, contracts::kThreads>>>(g);
}

#if BF16_C500_USE_LAYOUTC_NATIVE
template<int M, int N, int K>
__host__ void launch_gemm_layoutc(bf16 *a_native, bf16 *b_native, bf16 *c) {
    static_assert(M % layoutc_contracts::kBlockM == 0 && N % layoutc_contracts::kBlockN == 0,
                  "C500 layoutC path assumes M and N are multiples of 128.");
    static_assert(K % layoutc_contracts::kBlockK == 0,
                  "C500 layoutC path assumes K is a multiple of 128.");

    const dim3 grid(M / layoutc_contracts::kBlockM, N / layoutc_contracts::kBlockN);
    auto g = gemm_init_layoutc<M, N, K>(a_native, b_native, c);
    gemm_kernel_layoutc<M, N, K><<<grid, layoutc_contracts::kThreads>>>(g);
}
#endif

template<int N, int K>
std::vector<bf16> make_b_layouta(const std::vector<__nv_bfloat16> &row_major_b) {
    std::vector<bf16> layouta(static_cast<size_t>(N) * K);
    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            layouta[static_cast<size_t>(n) * K + k] = row_major_b[static_cast<size_t>(k) * N + n];
        }
    }
    return layouta;
}

#if BF16_C500_USE_LAYOUTC_NATIVE
template<int M, int K>
std::vector<bf16> make_a_layoutc_native(const std::vector<__nv_bfloat16> &row_major_a) {
    static_assert(M % 16 == 0 && K % 8 == 0, "layoutC A requires M%16==0 and K%8==0");
    std::vector<bf16> native(static_cast<size_t>(M) * K);
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            const size_t dst =
                ((((static_cast<size_t>(m) / 16) * (K / 8) + (k / 8)) * 16 + (m % 16)) * 8 + (k % 8));
            native[dst] = row_major_a[static_cast<size_t>(m) * K + k];
        }
    }
    return native;
}

template<int K, int N>
std::vector<bf16> make_b_layoutc_native(const std::vector<__nv_bfloat16> &row_major_b) {
    static_assert(N % 16 == 0 && K % 32 == 0, "layoutC B requires N%16==0 and K%32==0");
    std::vector<bf16> native(static_cast<size_t>(K) * N);
    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            const size_t dst =
                (((((static_cast<size_t>(k) / 32) * (N / 16) + (n / 16)) * 4 + ((k % 32) / 8)) * 16 + (n % 16)) * 8 +
                 (k % 8));
            native[dst] = row_major_b[static_cast<size_t>(k) * N + n];
        }
    }
    return native;
}
#endif

int run() {
    constexpr int M = BF16_C500_PROBLEM_M;
    constexpr int N = BF16_C500_PROBLEM_N;
    constexpr int K = BF16_C500_PROBLEM_K;
    constexpr float kMaxRelativeThreshold = 0.01f;

    static_assert(M % contracts::kBlockM == 0, "M must be a multiple of 128.");
    static_assert(N % contracts::kBlockN == 0, "N must be a multiple of 128.");
#if BF16_C500_USE_LAYOUTC_NATIVE
    static_assert(K % layoutc_contracts::kBlockK == 0, "K must be a multiple of 128.");
#elif BF16_C500_USE_LAYOUTA_NATIVE
    static_assert(K % contracts::kStageK == 0, "K must be a multiple of 32.");
#else
    static_assert(K % contracts::kBlockK == 0, "K must be a multiple of 128.");
#endif

    auto require = [](cudaError_t status, const char *what) {
        if (status != cudaSuccess) {
            std::cerr << what << ": " << cudaGetErrorString(status) << std::endl;
            std::exit(1);
        }
    };

    std::cout << "bf16_c500 TK GEMM" << std::endl;
    std::cout << "Problem size: M=" << M << ", N=" << N << ", K=" << K << std::endl;
#if BF16_C500_USE_LAYOUTC_NATIVE
    std::cout << "Path: layoutC native" << std::endl;
#elif BF16_C500_USE_LAYOUTA_NATIVE
    std::cout << "Path: layoutA native" << std::endl;
#else
    std::cout << "Path: balanced transitional" << std::endl;
#endif

    const size_t size_a = static_cast<size_t>(M) * K;
    const size_t size_b = static_cast<size_t>(K) * N;
    const size_t size_a_layoutc = static_cast<size_t>(M) * K;
    const size_t size_b_layoutc = static_cast<size_t>(K) * N;
    const size_t size_b_layouta = static_cast<size_t>(N) * K;
    const size_t size_c = static_cast<size_t>(M) * N;

    __nv_bfloat16 *d_a = nullptr, *d_b = nullptr, *d_c = nullptr, *d_ref = nullptr;
    require(cudaMalloc(&d_a, size_a * sizeof(__nv_bfloat16)), "cudaMalloc(A)");
    require(cudaMalloc(&d_b, size_b * sizeof(__nv_bfloat16)), "cudaMalloc(B)");
    require(cudaMalloc(&d_c, size_c * sizeof(__nv_bfloat16)), "cudaMalloc(C)");
    require(cudaMalloc(&d_ref, size_c * sizeof(__nv_bfloat16)), "cudaMalloc(C_ref)");
#if BF16_C500_USE_LAYOUTC_NATIVE
    bf16 *d_a_layoutc = nullptr;
    bf16 *d_b_layoutc = nullptr;
    require(cudaMalloc(&d_a_layoutc, size_a_layoutc * sizeof(bf16)), "cudaMalloc(A_layoutC)");
    require(cudaMalloc(&d_b_layoutc, size_b_layoutc * sizeof(bf16)), "cudaMalloc(B_layoutC)");
#elif BF16_C500_USE_LAYOUTA_NATIVE
    bf16 *d_b_layouta = nullptr;
    require(cudaMalloc(&d_b_layouta, size_b_layouta * sizeof(bf16)), "cudaMalloc(B_layoutA)");
#endif

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
    require(cudaMemcpy(d_a, h_a.data(), size_a * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "copy A");
    require(cudaMemcpy(d_b, h_b.data(), size_b * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "copy B");
#if BF16_C500_USE_LAYOUTC_NATIVE
    auto h_a_layoutc = make_a_layoutc_native<M, K>(h_a);
    auto h_b_layoutc = make_b_layoutc_native<K, N>(h_b);
    require(cudaMemcpy(d_a_layoutc, h_a_layoutc.data(), size_a_layoutc * sizeof(bf16), cudaMemcpyHostToDevice),
            "copy A_layoutC");
    require(cudaMemcpy(d_b_layoutc, h_b_layoutc.data(), size_b_layoutc * sizeof(bf16), cudaMemcpyHostToDevice),
            "copy B_layoutC");
#elif BF16_C500_USE_LAYOUTA_NATIVE
    auto h_b_layouta = make_b_layouta<N, K>(h_b);
    require(cudaMemcpy(d_b_layouta, h_b_layouta.data(), size_b_layouta * sizeof(bf16), cudaMemcpyHostToDevice), "copy B_layoutA");
#endif
    fill<__nv_bfloat16, FillMode::CONSTANT>(d_c, size_c, 0.0f);
    fill<__nv_bfloat16, FillMode::CONSTANT>(d_ref, size_c, 0.0f);
    require(cudaDeviceSynchronize(), "fill synchronize");

    reference_gemm<__nv_bfloat16, __nv_bfloat16, false>(d_ref, d_a, d_b, M, N, K);
    require(cudaGetLastError(), "reference_gemm launch");
    require(cudaDeviceSynchronize(), "reference_gemm synchronize");

    fill<__nv_bfloat16, FillMode::CONSTANT>(d_c, size_c, 0.0f);
    require(cudaDeviceSynchronize(), "clear output synchronize");

    int l2_cache_size = 0;
    require(cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, 0), "cudaDeviceGetAttribute(L2)");
    const size_t l2_clear_elems = std::max<size_t>(1, (size_t(l2_cache_size) * 3) / sizeof(int));
    int *l2_clear = nullptr;
    require(cudaMalloc(&l2_clear, l2_clear_elems * sizeof(int)), "cudaMalloc(l2_clear)");

    constexpr int warmup_iters = BF16_C500_WARMUP_ITERS;
    constexpr int profiling_iters = BF16_C500_PROFILE_ITERS;

    for (int i = 0; i < warmup_iters; ++i) {
        cudaMemset(l2_clear, 0, l2_clear_elems * sizeof(int));
#if BF16_C500_USE_LAYOUTC_NATIVE
        launch_gemm_layoutc<M, N, K>(d_a_layoutc, d_b_layoutc, reinterpret_cast<bf16 *>(d_c));
#elif BF16_C500_USE_LAYOUTA_NATIVE
        launch_gemm_layouta<M, N, K>(reinterpret_cast<bf16 *>(d_a),
                                     reinterpret_cast<bf16 *>(d_b_layouta),
                                     reinterpret_cast<bf16 *>(d_c));
#else
        launch_gemm<M, N, K>(reinterpret_cast<bf16 *>(d_a),
                             reinterpret_cast<bf16 *>(d_b),
                             reinterpret_cast<bf16 *>(d_c));
#endif
    }
    require(cudaDeviceSynchronize(), "gemm synchronize");

    std::vector<cudaEvent_t> starts(profiling_iters), stops(profiling_iters);
    std::vector<float> milliseconds(profiling_iters, 0.0f);
    for (int i = 0; i < profiling_iters; ++i) {
        cudaMemset(l2_clear, 0, l2_clear_elems * sizeof(int));
        require(cudaEventCreate(&starts[i]), "cudaEventCreate(start)");
        require(cudaEventCreate(&stops[i]), "cudaEventCreate(stop)");
        require(cudaEventRecord(starts[i]), "cudaEventRecord(start)");
#if BF16_C500_USE_LAYOUTC_NATIVE
        launch_gemm_layoutc<M, N, K>(d_a_layoutc, d_b_layoutc, reinterpret_cast<bf16 *>(d_c));
#elif BF16_C500_USE_LAYOUTA_NATIVE
        launch_gemm_layouta<M, N, K>(reinterpret_cast<bf16 *>(d_a),
                                     reinterpret_cast<bf16 *>(d_b_layouta),
                                     reinterpret_cast<bf16 *>(d_c));
#else
        launch_gemm<M, N, K>(reinterpret_cast<bf16 *>(d_a),
                             reinterpret_cast<bf16 *>(d_b),
                             reinterpret_cast<bf16 *>(d_c));
#endif
        require(cudaGetLastError(), "gemm launch");
        require(cudaEventRecord(stops[i]), "cudaEventRecord(stop)");
        require(cudaEventSynchronize(stops[i]), "cudaEventSynchronize(stop)");
    }

    double total_milliseconds = 0.0;
    for (int i = 0; i < profiling_iters; ++i) {
        require(cudaEventElapsedTime(&milliseconds[i], starts[i], stops[i]), "cudaEventElapsedTime");
        total_milliseconds += milliseconds[i];
    }
    const double runtime_ms = total_milliseconds / profiling_iters;
    const double runtime_s = runtime_ms / 1000.0;
    const double flops = 2.0 * static_cast<double>(M) * N * K;
    const double tflops = (flops / 1e12) / runtime_s;

    std::vector<__nv_bfloat16> h_out(size_c);
    std::vector<__nv_bfloat16> h_ref(size_c);
    require(cudaMemcpy(h_out.data(), d_c, size_c * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), "copy output");
    require(cudaMemcpy(h_ref.data(), d_ref, size_c * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), "copy reference");

#if BF16_C500_USE_LAYOUTC_NATIVE
    auto reference_host = [&](int row, int col, bool transpose_b) {
        float acc = 0.0f;
        for (int kk = 0; kk < K; ++kk) {
            const float a = __bfloat162float(h_a[static_cast<size_t>(row) * K + kk]);
            const float b = transpose_b
                                ? __bfloat162float(h_b[static_cast<size_t>(col) * K + kk])
                                : __bfloat162float(h_b[static_cast<size_t>(kk) * N + col]);
            acc += a * b;
        }
        return acc;
    };
    auto output_host = [&](int row, int col, bool transpose_out) {
        if (transpose_out) {
            return __bfloat162float(h_out[static_cast<size_t>(col) * M + row]);
        }
        return __bfloat162float(h_out[static_cast<size_t>(row) * N + col]);
    };
#endif

    double abs_sum = 0.0;
    double err_sum = 0.0;
    float abs_max = 0.0f;
    float err_max = 0.0f;
    for (size_t i = 0; i < size_c; ++i) {
        float got = 0.0f;
#if BF16_C500_USE_LAYOUTC_NATIVE
        const size_t row = i / N;
        const size_t col = i % N;
        got = output_host(static_cast<int>(row), static_cast<int>(col), true);
#else
        got = __bfloat162float(h_out[i]);
#endif
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
    std::cout << "Average runtime: " << runtime_ms << " ms" << std::endl;
    std::cout << "Performance: " << tflops << " TFLOP/s" << std::endl;
    std::cout << "abs mean:      " << (abs_sum / size_c) << std::endl;
    std::cout << "abs max:       " << abs_max << std::endl;
    std::cout << "err mean:      " << (err_sum / size_c) << std::endl;
    std::cout << "err max:       " << err_max << std::endl;
#if BF16_C500_USE_LAYOUTC_NATIVE
    if (M <= 256 && N <= 256 && K <= 256) {
        auto check_mode = [&](bool transpose_out, bool transpose_b, const char *label) {
            double local_abs = 0.0;
            double local_rel = 0.0;
            float local_abs_max = 0.0f;
            float local_rel_max = 0.0f;
            for (int row = 0; row < M; ++row) {
                for (int col = 0; col < N; ++col) {
                    const float got_local = output_host(row, col, transpose_out);
                    const float ref_local = reference_host(row, col, transpose_b);
                    const float abs_local = std::abs(got_local - ref_local);
                    const float rel_local = abs_local / std::max(1.0f, std::abs(ref_local));
                    local_abs += abs_local;
                    local_rel += rel_local;
                    local_abs_max = std::max(local_abs_max, abs_local);
                    local_rel_max = std::max(local_rel_max, rel_local);
                }
            }
            std::cout << label << " abs mean: " << (local_abs / size_c)
                      << " err mean: " << (local_rel / size_c)
                      << " abs max: " << local_abs_max
                      << " err max: " << local_rel_max << std::endl;
        };
        check_mode(false, false, "probe[out=MN,b=KN]");
        check_mode(true, false, "probe[out=NM,b=KN]");
        check_mode(false, true, "probe[out=MN,b=NK]");
        check_mode(true, true, "probe[out=NM,b=NK]");
    }
#endif
    std::cout << "Status: " << (pass ? "PASS" : "FAIL") << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
#if BF16_C500_USE_LAYOUTC_NATIVE
    cudaFree(d_a_layoutc);
    cudaFree(d_b_layoutc);
#elif BF16_C500_USE_LAYOUTA_NATIVE
    cudaFree(d_b_layouta);
#endif
    cudaFree(d_c);
    cudaFree(d_ref);
    cudaFree(l2_clear);
    for (int i = 0; i < profiling_iters; ++i) {
        cudaEventDestroy(starts[i]);
        cudaEventDestroy(stops[i]);
    }
    return pass ? 0 : 1;
}

} // namespace bf16_c500

int main() { return bf16_c500::run(); }
