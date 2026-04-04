/***************************************************************************************************
 * mcBLAS BF16 GEMM Benchmark
 *
 * D = A * B
 * A: RowMajor (M x K), B: ColMajor (N x K), D: RowMajor (M x N)
 * Accumulator: FP32, Output: BF16
 **************************************************************************************************/

#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <mcr/mc_runtime.h>
#include <mcblas/mcblas.h>

#include "../../common.cuh"

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << " line " << __LINE__ \
                      << ": " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CHECK_MCBLAS(call) \
    do { \
        mcblasStatus_t status = call; \
        if (status != MCBLAS_STATUS_SUCCESS) { \
            std::cerr << "mcBLAS error in " << __FILE__ << " line " << __LINE__ \
                      << ": " << status << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#ifndef BF16_MCBLAS_PROBLEM_M
#define BF16_MCBLAS_PROBLEM_M 4096
#endif
#ifndef BF16_MCBLAS_PROBLEM_N
#define BF16_MCBLAS_PROBLEM_N 4096
#endif
#ifndef BF16_MCBLAS_PROBLEM_K
#define BF16_MCBLAS_PROBLEM_K 4096
#endif
#ifndef BF16_MCBLAS_WARMUP_ITERS
#define BF16_MCBLAS_WARMUP_ITERS 25
#endif
#ifndef BF16_MCBLAS_PROFILE_ITERS
#define BF16_MCBLAS_PROFILE_ITERS 100
#endif

static inline void mcblas_gemm(mcblasHandle_t handle,
                               const kittens::bf16 *A,
                               const kittens::bf16 *B,
                               kittens::bf16 *D,
                               int M,
                               int N,
                               int K) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    CHECK_MCBLAS(mcblasGemmEx(handle,
                              MCBLAS_OP_T,
                              MCBLAS_OP_N,
                              N,
                              M,
                              K,
                              &alpha,
                              B,
                              MACA_R_16BF,
                              K,
                              A,
                              MACA_R_16BF,
                              K,
                              &beta,
                              D,
                              MACA_R_16BF,
                              N,
                              MCBLAS_COMPUTE_32F,
                              MCBLAS_GEMM_DEFAULT_TENSOR_OP));
}

static inline void benchmark(int M, int N, int K) {
    sleep_ms(500);

    mcblasHandle_t handle;
    CHECK_MCBLAS(mcblasCreate(&handle));

    std::cout << "Problem size: M=" << M << ", N=" << N << ", K=" << K << std::endl;

    int l2_cache_size = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, 0));
    const size_t arg_size =
        2 * (size_t(M) * K + size_t(N) * K + size_t(M) * N);
    const size_t ideal_arg_size = size_t(l2_cache_size) * 3;
    const int arg_group_count =
        (arg_size > ideal_arg_size) ? 1 : int(ideal_arg_size / arg_size) + 1;

    std::vector<kittens::bf16 *> blocks_A(arg_group_count);
    std::vector<kittens::bf16 *> blocks_B(arg_group_count);
    std::vector<kittens::bf16 *> blocks_D(arg_group_count);
    kittens::bf16 *block_D_ref = nullptr;

    const size_t size_A = size_t(M) * K;
    const size_t size_B = size_t(K) * N;
    const size_t size_D = size_t(M) * N;

    CHECK_CUDA(cudaMalloc(&block_D_ref, size_D * sizeof(kittens::bf16)));

    uint64_t seed = 2024;
    for (int i = 0; i < arg_group_count; ++i) {
        CHECK_CUDA(cudaMalloc(&blocks_A[i], size_A * sizeof(kittens::bf16)));
        CHECK_CUDA(cudaMalloc(&blocks_B[i], size_B * sizeof(kittens::bf16)));
        CHECK_CUDA(cudaMalloc(&blocks_D[i], size_D * sizeof(kittens::bf16)));
        fill<kittens::bf16, FillMode::RANDOM>(blocks_A[i], size_A, seed + i * 100, -1.0f, 1.0f);
        fill<kittens::bf16, FillMode::RANDOM>(blocks_B[i], size_B, seed + i * 100 + 1, -1.0f, 1.0f);
        fill<kittens::bf16, FillMode::CONSTANT>(blocks_D[i], size_D, 0.0f);
    }
    fill<kittens::bf16, FillMode::CONSTANT>(block_D_ref, size_D, 0.0f);
    CHECK_CUDA(cudaDeviceSynchronize());

    reference_gemm<kittens::bf16, kittens::bf16>(block_D_ref, blocks_A[0], blocks_B[0], M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_MCBLAS(mcblasSetStream(handle, reinterpret_cast<mcStream_t>(stream)));

    for (int i = 0; i < BF16_MCBLAS_WARMUP_ITERS; ++i) {
        const int idx = i % arg_group_count;
        mcblas_gemm(handle, blocks_A[idx], blocks_B[idx], blocks_D[idx], M, N, K);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < BF16_MCBLAS_PROFILE_ITERS; ++i) {
        const int idx = i % arg_group_count;
        mcblas_gemm(handle, blocks_A[idx], blocks_B[idx], blocks_D[idx], M, N, K);
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    float milliseconds = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    const double runtime_ms = static_cast<double>(milliseconds) / BF16_MCBLAS_PROFILE_ITERS;
    const double runtime_s = runtime_ms / 1000.0;
    const int64_t flops = int64_t(2) * M * N * K;
    const double tflops = (double(flops) / 1e12) / runtime_s;

    std::cout << "Average runtime: " << runtime_ms << " ms" << std::endl;
    std::cout << "Performance: " << tflops << " TFLOP/s" << std::endl;

    fill<kittens::bf16, FillMode::CONSTANT>(blocks_D[0], size_D, 0.0f);
    mcblas_gemm(handle, blocks_A[0], blocks_B[0], blocks_D[0], M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());
    check_correctness(blocks_D[0], block_D_ref, size_D);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaStreamDestroy(stream));

    for (int i = 0; i < arg_group_count; ++i) {
        CHECK_CUDA(cudaFree(blocks_A[i]));
        CHECK_CUDA(cudaFree(blocks_B[i]));
        CHECK_CUDA(cudaFree(blocks_D[i]));
    }
    CHECK_CUDA(cudaFree(block_D_ref));
    CHECK_MCBLAS(mcblasDestroy(handle));
}

int main() {
    std::cout << "mcBLAS BF16 GEMM" << std::endl;
    benchmark(BF16_MCBLAS_PROBLEM_M, BF16_MCBLAS_PROBLEM_N, BF16_MCBLAS_PROBLEM_K);
    return 0;
}
