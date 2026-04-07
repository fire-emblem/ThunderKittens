#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mcr/mc_runtime.h>
#include <mcblas/mcblas.h>

#include "../common.cuh"

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error in " << __FILE__ << " line " << __LINE__  \
                      << ": " << cudaGetErrorString(err) << std::endl;         \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

#define CHECK_MCBLAS(call)                                                     \
    do {                                                                       \
        mcblasStatus_t status = call;                                          \
        if (status != MCBLAS_STATUS_SUCCESS) {                                 \
            std::cerr << "mcBLAS error in " << __FILE__ << " line "            \
                      << __LINE__ << ": " << status << std::endl;              \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

namespace tk_local_mcblas_runtime {

inline int env_int(const char *name, int fallback) {
    if (const char *value = std::getenv(name)) {
        return std::atoi(value);
    }
    return fallback;
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

template <typename T>
inline float host_to_float(T value) {
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __bfloat162float(value);
    } else if constexpr (std::is_same_v<T, __half>) {
        return __half2float(value);
    } else {
        return static_cast<float>(value);
    }
}

template <typename LocalT>
static inline macaDataType_t mcblas_dtype();

template <>
inline macaDataType_t mcblas_dtype<kittens::bf16>() {
    return MACA_R_16BF;
}

template <>
inline macaDataType_t mcblas_dtype<__half>() {
    return MACA_R_16F;
}

template <typename LocalT>
static inline void mcblas_gemm(mcblasHandle_t handle, const LocalT *A,
                               const LocalT *B, LocalT *D, int M, int N,
                               int K) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CHECK_MCBLAS(mcblasGemmEx(handle, MCBLAS_OP_T, MCBLAS_OP_N, N, M, K,
                              &alpha, B, mcblas_dtype<LocalT>(), K, A,
                              mcblas_dtype<LocalT>(), K, &beta, D,
                              mcblas_dtype<LocalT>(), N, MCBLAS_COMPUTE_32F,
                              MCBLAS_GEMM_DEFAULT_TENSOR_OP));
}

template <typename LocalT, typename RefT>
int run(const char *case_name) {
    const int M = env_int("TK_MCBLAS_M", 4096);
    const int N = env_int("TK_MCBLAS_N", 4096);
    const int K = env_int("TK_MCBLAS_K", 4096);
    const int warmup_iters = env_int("TK_MCBLAS_WARMUP", 1);
    const int profile_iters = env_int("TK_MCBLAS_PROFILE", 3);

    const size_t size_A = static_cast<size_t>(M) * K;
    const size_t size_B = static_cast<size_t>(K) * N;
    const size_t size_D = static_cast<size_t>(M) * N;

    std::vector<LocalT> h_A(size_A);
    std::vector<LocalT> h_B(size_B);
    for (size_t i = 0; i < size_A; ++i) {
        const float value =
            static_cast<float>((static_cast<int>(i * 17 + 3) % 1024) - 512) /
            512.0f;
        h_A[i] = host_from_float<LocalT>(value);
    }
    for (size_t i = 0; i < size_B; ++i) {
        const float value =
            static_cast<float>((static_cast<int>(i * 29 + 5) % 1024) - 512) /
            512.0f;
        h_B[i] = host_from_float<LocalT>(value);
    }

    LocalT *d_A = nullptr;
    LocalT *d_B = nullptr;
    LocalT *d_D = nullptr;
    RefT *d_ref = nullptr;
    CHECK_CUDA(cudaMalloc(&d_A, size_A * sizeof(LocalT)));
    CHECK_CUDA(cudaMalloc(&d_B, size_B * sizeof(LocalT)));
    CHECK_CUDA(cudaMalloc(&d_D, size_D * sizeof(LocalT)));
    CHECK_CUDA(cudaMalloc(&d_ref, size_D * sizeof(RefT)));
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), size_A * sizeof(LocalT),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), size_B * sizeof(LocalT),
                          cudaMemcpyHostToDevice));
    fill<LocalT, FillMode::CONSTANT>(d_D, size_D, 0.0f);
    fill<RefT, FillMode::CONSTANT>(d_ref, size_D, 0.0f);
    CHECK_CUDA(cudaDeviceSynchronize());

    reference_gemm<LocalT, RefT, true>(d_ref, d_B, d_A, N, M, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    mcblasHandle_t handle;
    CHECK_MCBLAS(mcblasCreate(&handle));
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_MCBLAS(mcblasSetStream(handle, reinterpret_cast<mcStream_t>(stream)));

    for (int i = 0; i < warmup_iters; ++i) {
        mcblas_gemm(handle, d_A, d_B, d_D, M, N, K);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < profile_iters; ++i) {
        mcblas_gemm(handle, d_A, d_B, d_D, M, N, K);
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    float milliseconds = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    const double runtime_ms = static_cast<double>(milliseconds) / profile_iters;
    const double runtime_s = runtime_ms / 1000.0;
    const double flops = 2.0 * static_cast<double>(M) * N * K;
    const double tflops = (flops / 1e12) / runtime_s;

    std::vector<LocalT> h_D(size_D);
    std::vector<RefT> h_ref(size_D);
    CHECK_CUDA(cudaMemcpy(h_D.data(), d_D, size_D * sizeof(LocalT),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_ref.data(), d_ref, size_D * sizeof(RefT),
                          cudaMemcpyDeviceToHost));

    double abs_sum = 0.0;
    float abs_max = 0.0f;
    double err_sum = 0.0;
    float err_max = 0.0f;
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < M; ++col) {
            const float got = host_to_float(h_D[static_cast<size_t>(row) * M + col]);
            const float ref = host_to_float(h_ref[static_cast<size_t>(row) * M + col]);
            const float abs_err = std::abs(got - ref);
            const float rel_err = abs_err / std::max(1.0f, std::abs(ref));
            abs_sum += abs_err;
            err_sum += rel_err;
            abs_max = std::max(abs_max, abs_err);
            err_max = std::max(err_max, rel_err);
        }
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "mcBLAS runtime GEMM" << std::endl;
    std::cout << "Case: " << case_name << std::endl;
    std::cout << "Problem size: M=" << M << ", N=" << N << ", K=" << K
              << std::endl;
    std::cout << "Average runtime: " << runtime_ms << " ms" << std::endl;
    std::cout << "Performance: " << tflops << " TFLOP/s" << std::endl;
    std::cout << "abs mean:      " << (abs_sum / size_D) << std::endl;
    std::cout << "abs max:       " << abs_max << std::endl;
    std::cout << "err mean:      " << (err_sum / size_D) << std::endl;
    std::cout << "err max:       " << err_max << std::endl;

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_MCBLAS(mcblasDestroy(handle));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_D));
    CHECK_CUDA(cudaFree(d_ref));
    return 0;
}

} // namespace tk_local_mcblas_runtime

int main() {
#ifdef TK_MCBLAS_USE_FP16
    return tk_local_mcblas_runtime::run<__half, __half>("mcblas_runtime_fp16");
#else
    return tk_local_mcblas_runtime::run<kittens::bf16, kittens::bf16>(
        "mcblas_runtime_bf16");
#endif
}
