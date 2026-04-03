#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "../common.cuh"
#include "kittens.cuh"

using namespace kittens;

namespace bf16_ampere_small {

constexpr int NUM_WORKERS = 1;
constexpr int BLOCK_SIZE = NUM_WORKERS * kittens::WARP_THREADS;

template <int TILE_M, int TILE_N, int TILE_K, int M, int N, int K>
struct kernel_layout {
    using shared_tileA = st_bf<TILE_M, TILE_K>;
    using shared_tileB = st_bf<TILE_K, TILE_N>;
    using shared_tileC = st_bf<TILE_M, TILE_N>;
    using reg_tileA = rt_bf<TILE_M, TILE_K>;
    using reg_tileB = rt_bf<TILE_K, TILE_N, ducks::rt_layout::col>;
    using reg_tileC = rt_fl<TILE_M, TILE_N>;
    using a_gl = gl<bf16, 1, 1, M, K, shared_tileA>;
    using b_gl = gl<bf16, 1, 1, K, N, shared_tileB>;
    using c_gl = gl<bf16, 1, 1, M, N, shared_tileC>;
    struct globals {
        a_gl a;
        b_gl b;
        c_gl c;
    };
};

template <int TILE_M, int TILE_N, int TILE_K, int M, int N, int K>
__host__ typename kernel_layout<TILE_M, TILE_N, TILE_K, M, N, K>::globals gemm_init(
    bf16 *d_A, bf16 *d_B, bf16 *d_C) {
    using layout = kernel_layout<TILE_M, TILE_N, TILE_K, M, N, K>;
    typename layout::a_gl a_arg{d_A, nullptr, nullptr, nullptr, nullptr};
    typename layout::b_gl b_arg{d_B, nullptr, nullptr, nullptr, nullptr};
    typename layout::c_gl c_arg{d_C, nullptr, nullptr, nullptr, nullptr};
    return {a_arg, b_arg, c_arg};
}

template <int TILE_M, int TILE_N, int TILE_K, int M, int N, int K>
__global__ __launch_bounds__(BLOCK_SIZE, 1) void gemm_kernel(
    const __grid_constant__ typename kernel_layout<TILE_M, TILE_N, TILE_K, M, N, K>::globals g) {
    using layout = kernel_layout<TILE_M, TILE_N, TILE_K, M, N, K>;
    const int tile_row = blockIdx.y;
    const int tile_col = blockIdx.x;

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int *)&__shm[0]);
    typename layout::shared_tileA &a_s = al.allocate<typename layout::shared_tileA>();
    typename layout::shared_tileB &b_s = al.allocate<typename layout::shared_tileB>();
    typename layout::reg_tileA ar_bf;
    typename layout::reg_tileB br_bf;
    typename layout::reg_tileC cr_fl;

    kittens::warp::zero(cr_fl);

    const int num_k_tiles = K / TILE_K;
    for (int inner = 0; inner < num_k_tiles; ++inner) {
        kittens::warp::load(a_s, g.a, {0, 0, tile_row, inner});
        kittens::warp::load(b_s, g.b, {0, 0, inner, tile_col});
        __syncthreads();
        kittens::warp::load(ar_bf, a_s);
        kittens::warp::load(br_bf, b_s);
        kittens::warp::mma_AB(cr_fl, ar_bf, br_bf, cr_fl);
        __syncthreads();
    }

    kittens::warp::store(g.c, cr_fl, {0, 0, tile_row, tile_col});
}

template <int TILE_M, int TILE_N, int TILE_K, int M, int N, int K>
__host__ void launch_gemm(bf16 *A, bf16 *B, bf16 *C) {
    const dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    auto g = gemm_init<TILE_M, TILE_N, TILE_K, M, N, K>(A, B, C);

    unsigned long mem_size = 20000;
    cudaFuncSetAttribute(gemm_kernel<TILE_M, TILE_N, TILE_K, M, N, K>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    gemm_kernel<TILE_M, TILE_N, TILE_K, M, N, K><<<grid, BLOCK_SIZE, mem_size>>>(g);
}

}  // namespace bf16_ampere_small

struct ShapeSpec {
    int m;
    int n;
    int k;
};

static bool parse_arg_int(char **begin, char **end, const std::string &flag, int &value) {
    for (char **it = begin; it != end; ++it) {
        if (flag != *it || it + 1 == end) {
            continue;
        }
        value = std::atoi(*(it + 1));
        return true;
    }
    return false;
}

static bool has_flag(char **begin, char **end, const std::string &flag) {
    for (char **it = begin; it != end; ++it) {
        if (flag == *it) {
            return true;
        }
    }
    return false;
}

static void print_usage(const char *program) {
    std::cout << "Usage: " << program << " [--m M --n N --k K] [--no-check] [--micro|--no-micro]" << std::endl;
    std::cout << "Supported shapes:" << std::endl;
    std::cout << "  512x512x512" << std::endl;
    std::cout << "  1024x1024x1024" << std::endl;
}

template <int M, int N, int K>
int run_benchmark(bool verify, bool use_micro_tile) {
    std::cout << "bf16_ampere_small TK GEMM" << std::endl;
    std::cout << "Problem size: M=" << M << ", N=" << N << ", K=" << K << std::endl;

    const size_t size_a = static_cast<size_t>(M) * K;
    const size_t size_b = static_cast<size_t>(K) * N;
    const size_t size_c = static_cast<size_t>(M) * N;

    __nv_bfloat16 *d_A = nullptr, *d_B = nullptr, *d_C = nullptr, *d_C_ref = nullptr;
    cudaMalloc(&d_A, size_a * sizeof(__nv_bfloat16));
    cudaMalloc(&d_B, size_b * sizeof(__nv_bfloat16));
    cudaMalloc(&d_C, size_c * sizeof(__nv_bfloat16));
    cudaMalloc(&d_C_ref, size_c * sizeof(__nv_bfloat16));

    fill<__nv_bfloat16, FillMode::RANDOM>(d_A, size_a, 2024, -1.0f, 1.0f);
    fill<__nv_bfloat16, FillMode::RANDOM>(d_B, size_b, 2025, -1.0f, 1.0f);
    fill<__nv_bfloat16, FillMode::CONSTANT>(d_C, size_c, 0.0f);
    fill<__nv_bfloat16, FillMode::CONSTANT>(d_C_ref, size_c, 0.0f);
    cudaDeviceSynchronize();

    if (verify) {
        reference_gemm<__nv_bfloat16, __nv_bfloat16, false>(d_C_ref, d_A, d_B, M, N, K);
        cudaDeviceSynchronize();
    }

    constexpr int warmup_iters = 25;
    constexpr int profiling_iters = 200;

    for (int i = 0; i < warmup_iters; ++i) {
        if (use_micro_tile) {
            bf16_ampere_small::launch_gemm<32, 32, 32, M, N, K>(reinterpret_cast<bf16 *>(d_A),
                                                                reinterpret_cast<bf16 *>(d_B),
                                                                reinterpret_cast<bf16 *>(d_C));
        } else {
            bf16_ampere_small::launch_gemm<64, 64, 32, M, N, K>(reinterpret_cast<bf16 *>(d_A),
                                                                reinterpret_cast<bf16 *>(d_B),
                                                                reinterpret_cast<bf16 *>(d_C));
        }
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < profiling_iters; ++i) {
        if (use_micro_tile) {
            bf16_ampere_small::launch_gemm<32, 32, 32, M, N, K>(reinterpret_cast<bf16 *>(d_A),
                                                                reinterpret_cast<bf16 *>(d_B),
                                                                reinterpret_cast<bf16 *>(d_C));
        } else {
            bf16_ampere_small::launch_gemm<64, 64, 32, M, N, K>(reinterpret_cast<bf16 *>(d_A),
                                                                reinterpret_cast<bf16 *>(d_B),
                                                                reinterpret_cast<bf16 *>(d_C));
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    const double runtime_ms = static_cast<double>(milliseconds) / profiling_iters;
    const double runtime_s = runtime_ms / 1000.0;
    const double flops = 2.0 * static_cast<double>(M) * N * K;
    const double tflops = (flops / 1e12) / runtime_s;

    std::cout << "Average runtime: " << runtime_ms << " ms" << std::endl;
    std::cout << "Performance: " << tflops << " TFLOP/s" << std::endl;

    if (verify) {
        if (use_micro_tile) {
            bf16_ampere_small::launch_gemm<32, 32, 32, M, N, K>(reinterpret_cast<bf16 *>(d_A),
                                                                reinterpret_cast<bf16 *>(d_B),
                                                                reinterpret_cast<bf16 *>(d_C));
        } else {
            bf16_ampere_small::launch_gemm<64, 64, 32, M, N, K>(reinterpret_cast<bf16 *>(d_A),
                                                                reinterpret_cast<bf16 *>(d_B),
                                                                reinterpret_cast<bf16 *>(d_C));
        }
        cudaDeviceSynchronize();
        check_correctness(d_C, d_C_ref, size_c);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_C_ref);
    return 0;
}

int main(int argc, char **argv) {
    ShapeSpec shape{512, 512, 512};
    parse_arg_int(argv + 1, argv + argc, "--m", shape.m);
    parse_arg_int(argv + 1, argv + argc, "--n", shape.n);
    parse_arg_int(argv + 1, argv + argc, "--k", shape.k);
    const bool verify = !has_flag(argv + 1, argv + argc, "--no-check");
    bool use_micro_tile = false;

    if (has_flag(argv + 1, argv + argc, "--help")) {
        print_usage(argv[0]);
        return 0;
    }

    if (shape.m == 512 && shape.n == 512 && shape.k == 512) {
        use_micro_tile = true;
    }
    if (has_flag(argv + 1, argv + argc, "--micro")) {
        use_micro_tile = true;
    }
    if (has_flag(argv + 1, argv + argc, "--no-micro")) {
        use_micro_tile = false;
    }

    if (shape.m == 512 && shape.n == 512 && shape.k == 512) {
        return run_benchmark<512, 512, 512>(verify, use_micro_tile);
    }
    if (shape.m == 1024 && shape.n == 1024 && shape.k == 1024) {
        return run_benchmark<1024, 1024, 1024>(verify, use_micro_tile);
    }

    std::cerr << "Unsupported shape M=" << shape.m << ", N=" << shape.n << ", K=" << shape.k << std::endl;
    print_usage(argv[0]);
    return 1;
}
