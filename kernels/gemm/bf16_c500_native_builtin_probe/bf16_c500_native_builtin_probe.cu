#include <cuda_runtime.h>
#include <maca.h>
#include <maca_bfloat16.h>

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include "/data/muxi_native_layout_kernels/csrc/muxi_hgemm_layoutA.cuh"

namespace c500_native_builtin_probe {

using bf16 = __maca_bfloat16;
using muxi_layout_kernels::layoutA_hgemm_tn_128x128x128_4m1n8k_256t;
using uint2_native = __NATIVE_VECTOR__(2, uint32_t);
using float4_native = __NATIVE_VECTOR__(4, float);

#ifndef BF16_C500_NATIVE_WARMUP_ITERS
#define BF16_C500_NATIVE_WARMUP_ITERS 5
#endif
#ifndef BF16_C500_NATIVE_PROFILE_ITERS
#define BF16_C500_NATIVE_PROFILE_ITERS 20
#endif
#ifndef BF16_C500_NATIVE_M
#define BF16_C500_NATIVE_M 4096
#endif
#ifndef BF16_C500_NATIVE_N
#define BF16_C500_NATIVE_N 4096
#endif
#ifndef BF16_C500_NATIVE_K
#define BF16_C500_NATIVE_K 4096
#endif
#ifndef BF16_C500_NATIVE_MMA_INNER_ITERS
#define BF16_C500_NATIVE_MMA_INNER_ITERS 2048
#endif

constexpr int kTileM = 128;
constexpr int kTileN = 128;
constexpr int kTileK = 32;
constexpr int kGridX = BF16_C500_NATIVE_M / 128;
constexpr int kGridY = BF16_C500_NATIVE_N / 128;
constexpr int kMicroM = kGridY * kTileM;
constexpr int kMicroN = kGridX * kTileN;
constexpr int kMicroK = BF16_C500_NATIVE_K;

__device__ inline uint32_t pack_bf16_pair(bf16 lo, bf16 hi) {
    struct pair_t {
        bf16 x;
        bf16 y;
    };
    const pair_t v{lo, hi};
    return *reinterpret_cast<const uint32_t *>(&v);
}

__global__ void native_gmem_mma_kernel(const bf16 *A, const bf16 *B, float *sink, int inner_iters) {
    const int lane = threadIdx.x & 63;
    const int linear_block = blockIdx.y * gridDim.x + blockIdx.x;
    const size_t a_base = (static_cast<size_t>(linear_block) * 256 + threadIdx.x) * 8;
    const size_t b_base = (static_cast<size_t>(linear_block) * 256 + threadIdx.x) * 8;

    float4_native acc[4][4];
#pragma unroll
    for (int m = 0; m < 4; ++m) {
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            acc[m][n] = float4_native{0.f, 0.f, 0.f, 0.f};
        }
    }

    for (int iter = 0; iter < inner_iters; ++iter) {
        const size_t iter_a = (a_base + static_cast<size_t>(iter & 7) * 8) % (static_cast<size_t>(kMicroM) * kMicroK - 8);
        const size_t iter_b = (b_base + static_cast<size_t>((iter + 3) & 7) * 8) % (static_cast<size_t>(kMicroN) * kMicroK - 8);
        const uint32_t a0 = pack_bf16_pair(A[iter_a + 0], A[iter_a + 1]);
        const uint32_t a1 = pack_bf16_pair(A[iter_a + 2], A[iter_a + 3]);
        const uint32_t b0 = pack_bf16_pair(B[iter_b + 0], B[iter_b + 1]);
        const uint32_t b1 = pack_bf16_pair(B[iter_b + 2], B[iter_b + 3]);
        const uint2_native a{a0, a1};
        const uint2_native b{b0, b1};
#pragma unroll
        for (int mma_k = 0; mma_k < 2; ++mma_k) {
#pragma unroll
            for (int m = 0; m < 4; ++m) {
#pragma unroll
                for (int n = 0; n < 4; ++n) {
                    acc[m][n] = __builtin_mxc_mma_16x16x16bf16(b, a, acc[m][n]);
                }
            }
        }
    }

    float sum = 0.f;
#pragma unroll
    for (int m = 0; m < 4; ++m) {
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            sum += acc[m][n][0] + acc[m][n][1] + acc[m][n][2] + acc[m][n][3];
        }
    }
    sink[linear_block * blockDim.x + threadIdx.x] = sum;
}

__global__ void native_gmem_mma_store_kernel(const bf16 *A,
                                             const bf16 *B,
                                             float *out,
                                             float *sink,
                                             int inner_iters) {
    const int lane = threadIdx.x & 63;
    const int linear_block = blockIdx.y * gridDim.x + blockIdx.x;
    const size_t a_base = (static_cast<size_t>(linear_block) * 256 + threadIdx.x) * 8;
    const size_t b_base = (static_cast<size_t>(linear_block) * 256 + threadIdx.x) * 8;

    float4_native acc[4][4];
#pragma unroll
    for (int m = 0; m < 4; ++m) {
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            acc[m][n] = float4_native{0.f, 0.f, 0.f, 0.f};
        }
    }

    for (int iter = 0; iter < inner_iters; ++iter) {
        const size_t iter_a = (a_base + static_cast<size_t>(iter & 7) * 8) % (static_cast<size_t>(kMicroM) * kMicroK - 8);
        const size_t iter_b = (b_base + static_cast<size_t>((iter + 3) & 7) * 8) % (static_cast<size_t>(kMicroN) * kMicroK - 8);
        const uint32_t a0 = pack_bf16_pair(A[iter_a + 0], A[iter_a + 1]);
        const uint32_t a1 = pack_bf16_pair(A[iter_a + 2], A[iter_a + 3]);
        const uint32_t b0 = pack_bf16_pair(B[iter_b + 0], B[iter_b + 1]);
        const uint32_t b1 = pack_bf16_pair(B[iter_b + 2], B[iter_b + 3]);
        const uint2_native a{a0, a1};
        const uint2_native b{b0, b1};
#pragma unroll
        for (int mma_k = 0; mma_k < 2; ++mma_k) {
#pragma unroll
            for (int m = 0; m < 4; ++m) {
#pragma unroll
                for (int n = 0; n < 4; ++n) {
                    acc[m][n] = __builtin_mxc_mma_16x16x16bf16(b, a, acc[m][n]);
                }
            }
        }
    }

    const size_t out_base = (static_cast<size_t>(linear_block) * blockDim.x + threadIdx.x) * 64;
    float sum = 0.f;
#pragma unroll
    for (int m = 0; m < 4; ++m) {
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            out[out_base + ((m * 4 + n) * 4 + 0)] = acc[m][n][0];
            out[out_base + ((m * 4 + n) * 4 + 1)] = acc[m][n][1];
            out[out_base + ((m * 4 + n) * 4 + 2)] = acc[m][n][2];
            out[out_base + ((m * 4 + n) * 4 + 3)] = acc[m][n][3];
            sum += acc[m][n][0] + acc[m][n][1] + acc[m][n][2] + acc[m][n][3];
        }
    }
    sink[linear_block * blockDim.x + threadIdx.x] = sum + static_cast<float>(lane);
}

inline void require(cudaError_t status, const char *what) {
    if (status != cudaSuccess) {
        std::cerr << what << ": " << cudaGetErrorString(status) << std::endl;
        std::exit(1);
    }
}

inline size_t l2_flush_elems() {
    int l2_cache_size = 0;
    require(cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, 0), "cudaDeviceGetAttribute(L2)");
    return std::max<size_t>(1, (static_cast<size_t>(l2_cache_size) * 3) / sizeof(int));
}

template<typename LaunchFn>
double time_kernel_ms(LaunchFn launch, int warmup_iters, int profile_iters) {
    const size_t flush_count = l2_flush_elems();
    int *flush_buf = nullptr;
    require(cudaMalloc(&flush_buf, flush_count * sizeof(int)), "cudaMalloc(l2_flush)");

    for (int i = 0; i < warmup_iters; ++i) {
        cudaMemset(flush_buf, 0, flush_count * sizeof(int));
        launch();
    }
    require(cudaDeviceSynchronize(), "warmup synchronize");

    std::vector<cudaEvent_t> starts(profile_iters), stops(profile_iters);
    std::vector<float> ms(profile_iters, 0.0f);
    for (int i = 0; i < profile_iters; ++i) {
        cudaMemset(flush_buf, 0, flush_count * sizeof(int));
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
        require(cudaEventElapsedTime(&ms[i], starts[i], stops[i]), "cudaEventElapsedTime");
        total_ms += ms[i];
        cudaEventDestroy(starts[i]);
        cudaEventDestroy(stops[i]);
    }
    cudaFree(flush_buf);
    return total_ms / profile_iters;
}

void init_bf16(std::vector<bf16> &buf, int mul, int add) {
    for (size_t i = 0; i < buf.size(); ++i) {
        const float value = static_cast<float>((static_cast<int>(i) * mul + add) % 1024 - 512) / 512.0f;
        buf[i] = bf16(value);
    }
}

int run() {
    constexpr int M = BF16_C500_NATIVE_M;
    constexpr int N = BF16_C500_NATIVE_N;
    constexpr int K = BF16_C500_NATIVE_K;
    constexpr int warmup_iters = BF16_C500_NATIVE_WARMUP_ITERS;
    constexpr int profile_iters = BF16_C500_NATIVE_PROFILE_ITERS;
    constexpr int mma_inner_iters = BF16_C500_NATIVE_MMA_INNER_ITERS;
    constexpr dim3 grid(M / 128, N / 128, 1);
    static_assert(M % 128 == 0 && N % 128 == 0 && K % 128 == 0, "M/N/K must be multiples of 128.");

    const size_t size_a = static_cast<size_t>(M) * K;
    const size_t size_b = static_cast<size_t>(N) * K;
    const size_t size_c = static_cast<size_t>(M) * N;

    bf16 *d_a = nullptr;
    bf16 *d_b = nullptr;
    bf16 *d_c = nullptr;
    float *d_sink = nullptr;
    float *d_out = nullptr;
    require(cudaMalloc(&d_a, size_a * sizeof(bf16)), "cudaMalloc(A)");
    require(cudaMalloc(&d_b, size_b * sizeof(bf16)), "cudaMalloc(B)");
    require(cudaMalloc(&d_c, size_c * sizeof(bf16)), "cudaMalloc(C)");
    require(cudaMalloc(&d_sink, static_cast<size_t>(grid.x * grid.y * 256) * sizeof(float)), "cudaMalloc(sink)");
    require(cudaMalloc(&d_out, static_cast<size_t>(grid.x * grid.y * 256 * 64) * sizeof(float)), "cudaMalloc(out)");

    std::vector<bf16> h_a(size_a);
    std::vector<bf16> h_b(size_b);
    init_bf16(h_a, 17, 3);
    init_bf16(h_b, 29, 5);
    require(cudaMemcpy(d_a, h_a.data(), size_a * sizeof(bf16), cudaMemcpyHostToDevice), "copy A");
    require(cudaMemcpy(d_b, h_b.data(), size_b * sizeof(bf16), cudaMemcpyHostToDevice), "copy B");
    require(cudaMemset(d_c, 0, size_c * sizeof(bf16)), "clear C");

    const double flops = 2.0 * static_cast<double>(M) * N * K;
    const double gmem_mma_flops = static_cast<double>(grid.x * grid.y) * mma_inner_iters * 2.0 *
                                  (2.0 * 128.0 * 128.0 * 32.0);
    const auto gmem_mma_launch = [&]() {
        native_gmem_mma_kernel<<<grid, 256>>>(d_a, d_b, d_sink, mma_inner_iters);
    };
    const auto gmem_mma_store_launch = [&]() {
        native_gmem_mma_store_kernel<<<grid, 256>>>(d_a, d_b, d_out, d_sink, mma_inner_iters);
    };
    const auto launch = [&]() {
        layoutA_hgemm_tn_128x128x128_4m1n8k_256t<bf16, bf16, float, true, false>
            <<<grid, 256>>>(d_a, d_b, d_c, M, N, K, K, K, N, 1.0f, 0.0f, nullptr);
    };

    const double gmem_mma_runtime_ms = time_kernel_ms(gmem_mma_launch, warmup_iters, profile_iters);
    const double gmem_mma_store_runtime_ms = time_kernel_ms(gmem_mma_store_launch, warmup_iters, profile_iters);
    const double runtime_ms = time_kernel_ms(launch, warmup_iters, profile_iters);
    const double gmem_mma_runtime_s = gmem_mma_runtime_ms / 1000.0;
    const double gmem_mma_store_runtime_s = gmem_mma_store_runtime_ms / 1000.0;
    const double runtime_s = runtime_ms / 1000.0;
    const double gmem_mma_tflops = (gmem_mma_flops / 1.0e12) / gmem_mma_runtime_s;
    const double gmem_mma_store_tflops = (gmem_mma_flops / 1.0e12) / gmem_mma_store_runtime_s;
    const double tflops = (flops / 1.0e12) / runtime_s;

    std::vector<bf16> h_c(8);
    require(cudaMemcpy(h_c.data(), d_c, h_c.size() * sizeof(bf16), cudaMemcpyDeviceToHost), "copy C sample");

    int sm_count = 0;
    require(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0), "cudaDeviceGetAttribute(SM)");

    std::cout << "C500 native builtin BF16 pipeline probe" << std::endl;
    std::cout << "Kernel: muxi layoutA native pipeline" << std::endl;
    std::cout << "Timing: cudaEvent device time only" << std::endl;
    std::cout << "Shape: M=" << M << " N=" << N << " K=" << K << ", SM count=" << sm_count << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "native_gmem_mma runtime_ms=" << gmem_mma_runtime_ms
              << " tflops=" << gmem_mma_tflops << std::endl;
    std::cout << "native_gmem_mma_store runtime_ms=" << gmem_mma_store_runtime_ms
              << " tflops=" << gmem_mma_store_tflops << std::endl;
    std::cout << "native_pipeline runtime_ms=" << runtime_ms
              << " tflops=" << tflops << std::endl;
    std::cout << "c_sample " << static_cast<float>(h_c[0]) << " "
              << static_cast<float>(h_c[1]) << " "
              << static_cast<float>(h_c[2]) << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_sink);
    cudaFree(d_out);
    return 0;
}

} // namespace c500_native_builtin_probe

int main() { return c500_native_builtin_probe::run(); }
