#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "../common.cuh"
#include "kittens.cuh"

using namespace kittens;

namespace bf16_ampere {

constexpr int MMA_M = 64;
constexpr int MMA_N = 64;
constexpr int MMA_K = 32;

constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 32;

constexpr int NUM_WORKERS = 4;
constexpr int PIPE_STAGES = 2;
constexpr int LOAD_GROUPS = 2;
constexpr int BLOCK_SIZE = NUM_WORKERS * kittens::WARP_THREADS;

using shared_tileA = st_bf<MMA_M, MMA_K>;
using shared_tileB = st_bf<MMA_K, MMA_N>;
using shared_tileC = st_bf<MMA_M, MMA_N>;

using reg_tileA = rt_bf<MMA_M, MMA_K>;
using reg_tileB = rt_bf<MMA_K, MMA_N, ducks::rt_layout::col>;
using reg_tileC = rt_fl<MMA_M, MMA_N>;

template <int M, int K>
using a_gl = gl<bf16, 1, 1, M, K, shared_tileA>;
template <int K, int N>
using b_gl = gl<bf16, 1, 1, K, N, shared_tileB>;
template <int M, int N>
using c_gl = gl<bf16, 1, 1, M, N, shared_tileC>;

template <int M, int N, int K>
struct gemm_globals {
    a_gl<M, K> a;
    b_gl<K, N> b;
    c_gl<M, N> c;
};

template <int M, int N, int K>
__host__ gemm_globals<M, N, K> gemm_init(bf16 *d_A, bf16 *d_B, bf16 *d_C) {
    a_gl<M, K> a_arg{d_A, nullptr, nullptr, nullptr, nullptr};
    b_gl<K, N> b_arg{d_B, nullptr, nullptr, nullptr, nullptr};
    c_gl<M, N> c_arg{d_C, nullptr, nullptr, nullptr, nullptr};
    return {a_arg, b_arg, c_arg};
}

template <int M, int N, int K>
__global__ __launch_bounds__(BLOCK_SIZE, 1) void gemm_kernel(
    const __grid_constant__ gemm_globals<M, N, K> g) {
    using load_group = kittens::group<(NUM_WORKERS / LOAD_GROUPS)>;

    const int workerid = kittens::warpid();
    const int row_worker = workerid / 2;
    const int col_worker = workerid % 2;
    const int load_id = load_group::groupid();
    constexpr int LOAD_BLOCKS = NUM_WORKERS / load_group::GROUP_WARPS;

    const int warp_row = LOAD_GROUPS * blockIdx.y;
    const int warp_col = LOAD_GROUPS * blockIdx.x;

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int *)&__shm[0]);

    shared_tileA (&a_s)[LOAD_BLOCKS][PIPE_STAGES] =
        al.allocate<shared_tileA, LOAD_BLOCKS, PIPE_STAGES>();
    shared_tileB (&b_s)[LOAD_BLOCKS][PIPE_STAGES] =
        al.allocate<shared_tileB, LOAD_BLOCKS, PIPE_STAGES>();
    reg_tileA ar_bf;
    reg_tileB br_bf;
    reg_tileC cr_fl;

    kittens::warp::zero(cr_fl);

    const int num_k_tiles = K / MMA_K;
    int tic = 0;

    load_group::load_async<2, true>(a_s[load_id][tic], g.a, {warp_row + load_id, 0});
    load_group::load_async<2, true>(b_s[load_id][tic], g.b, {0, warp_col + load_id});

    for (int inner = 0; inner < num_k_tiles; ++inner, tic = (tic + 1) % PIPE_STAGES) {
        const int next_load_idx = inner + 1;
        if (next_load_idx < num_k_tiles) {
            const int next_tic = (tic + 1) % PIPE_STAGES;
            load_group::load_async<2, true>(a_s[load_id][next_tic], g.a,
                                            {warp_row + load_id, next_load_idx});
            load_group::load_async<2, true>(b_s[load_id][next_tic], g.b,
                                            {next_load_idx, warp_col + load_id});
            load_async_wait<2>();
        } else {
            load_async_wait();
        }

        __syncthreads();
        kittens::warp::load(ar_bf, a_s[row_worker][tic]);
        kittens::warp::load(br_bf, b_s[col_worker][tic]);
        kittens::warp::mma_AB(cr_fl, ar_bf, br_bf, cr_fl);
    }

    kittens::warp::store(g.c, cr_fl, {0, 0, warp_row + row_worker, warp_col + col_worker});
}

template <int M, int N, int K>
__host__ void launch_gemm(bf16 *A, bf16 *B, bf16 *C) {
    const dim3 grid((N + (BLOCK_N * LOAD_GROUPS) - 1) / (BLOCK_N * LOAD_GROUPS),
                    (M + (BLOCK_M * LOAD_GROUPS) - 1) / (BLOCK_M * LOAD_GROUPS));
    gemm_globals<M, N, K> g = gemm_init<M, N, K>(A, B, C);

    unsigned long mem_size = 50000;
    cudaFuncSetAttribute(gemm_kernel<M, N, K>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    gemm_kernel<M, N, K><<<grid, BLOCK_SIZE, mem_size>>>(g);
}

}  // namespace bf16_ampere

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
    std::cout << "Usage: " << program << " [--m M --n N --k K] [--no-check]" << std::endl;
    std::cout << "Supported shapes:" << std::endl;
    std::cout << "  512x512x512" << std::endl;
    std::cout << "  1024x1024x1024" << std::endl;
    std::cout << "  2048x2048x2048" << std::endl;
    std::cout << "  4096x4096x4096" << std::endl;
    std::cout << "  4096x8192x4096" << std::endl;
    std::cout << "  8192x4096x4096" << std::endl;
}

template <int M, int N, int K>
int run_benchmark(bool verify) {

    std::cout << "bf16_ampere TK GEMM" << std::endl;
    std::cout << "Problem size: M=" << M << ", N=" << N << ", K=" << K << std::endl;

    const size_t size_a = static_cast<size_t>(M) * K;
    const size_t size_b = static_cast<size_t>(K) * N;
    const size_t size_c = static_cast<size_t>(M) * N;

    int l2_cache_size = 0;
    cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, 0);
    const size_t l2_clear_elems = std::max<size_t>(1, (size_t(l2_cache_size) * 3) / sizeof(int));

    __nv_bfloat16 *d_A = nullptr, *d_B = nullptr, *d_C = nullptr, *d_C_ref = nullptr;
    int *l2_clear = nullptr;

    cudaMalloc(&d_A, size_a * sizeof(__nv_bfloat16));
    cudaMalloc(&d_B, size_b * sizeof(__nv_bfloat16));
    cudaMalloc(&d_C, size_c * sizeof(__nv_bfloat16));
    cudaMalloc(&d_C_ref, size_c * sizeof(__nv_bfloat16));
    cudaMalloc(&l2_clear, l2_clear_elems * sizeof(int));

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
    constexpr int profiling_iters = 100;

    for (int i = 0; i < warmup_iters; ++i) {
        cudaMemset(l2_clear, 0, l2_clear_elems * sizeof(int));
        bf16_ampere::launch_gemm<M, N, K>(reinterpret_cast<bf16 *>(d_A),
                                          reinterpret_cast<bf16 *>(d_B),
                                          reinterpret_cast<bf16 *>(d_C));
    }
    cudaDeviceSynchronize();

    std::vector<cudaEvent_t> starts(profiling_iters), stops(profiling_iters);
    std::vector<float> milliseconds(profiling_iters, 0.0f);
    for (int i = 0; i < profiling_iters; ++i) {
        cudaMemset(l2_clear, 0, l2_clear_elems * sizeof(int));
        cudaEventCreate(&starts[i]);
        cudaEventCreate(&stops[i]);
        cudaEventRecord(starts[i]);
        bf16_ampere::launch_gemm<M, N, K>(reinterpret_cast<bf16 *>(d_A),
                                          reinterpret_cast<bf16 *>(d_B),
                                          reinterpret_cast<bf16 *>(d_C));
        cudaEventRecord(stops[i]);
        cudaEventSynchronize(stops[i]);
    }

    double total_milliseconds = 0.0;
    for (int i = 0; i < profiling_iters; ++i) {
        cudaEventElapsedTime(&milliseconds[i], starts[i], stops[i]);
        total_milliseconds += milliseconds[i];
    }
    const double runtime_ms = total_milliseconds / profiling_iters;
    const double runtime_s = runtime_ms / 1000.0;
    const double flops = 2.0 * static_cast<double>(M) * N * K;
    const double tflops = (flops / 1e12) / runtime_s;

    std::cout << "Average runtime: " << runtime_ms << " ms" << std::endl;
    std::cout << "Performance: " << tflops << " TFLOP/s" << std::endl;

    if (verify) {
        bf16_ampere::launch_gemm<M, N, K>(reinterpret_cast<bf16 *>(d_A),
                                          reinterpret_cast<bf16 *>(d_B),
                                          reinterpret_cast<bf16 *>(d_C));
        cudaDeviceSynchronize();
        check_correctness(d_C, d_C_ref, size_c);
    }

    for (int i = 0; i < profiling_iters; ++i) {
        cudaEventDestroy(starts[i]);
        cudaEventDestroy(stops[i]);
    }
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_C_ref);
    cudaFree(l2_clear);

    return 0;
}

int main(int argc, char **argv) {
    ShapeSpec shape{4096, 4096, 4096};
    parse_arg_int(argv + 1, argv + argc, "--m", shape.m);
    parse_arg_int(argv + 1, argv + argc, "--n", shape.n);
    parse_arg_int(argv + 1, argv + argc, "--k", shape.k);
    const bool verify = !has_flag(argv + 1, argv + argc, "--no-check");

    if (has_flag(argv + 1, argv + argc, "--help")) {
        print_usage(argv[0]);
        return 0;
    }

    if (shape.m == 2048 && shape.n == 2048 && shape.k == 2048) {
        return run_benchmark<2048, 2048, 2048>(verify);
    }
    if (shape.m == 512 && shape.n == 512 && shape.k == 512) {
        return run_benchmark<512, 512, 512>(verify);
    }
    if (shape.m == 1024 && shape.n == 1024 && shape.k == 1024) {
        return run_benchmark<1024, 1024, 1024>(verify);
    }
    if (shape.m == 4096 && shape.n == 4096 && shape.k == 4096) {
        return run_benchmark<4096, 4096, 4096>(verify);
    }
    if (shape.m == 4096 && shape.n == 8192 && shape.k == 4096) {
        return run_benchmark<4096, 8192, 4096>(verify);
    }
    if (shape.m == 8192 && shape.n == 4096 && shape.k == 4096) {
        return run_benchmark<8192, 4096, 4096>(verify);
    }

    std::cerr << "Unsupported shape M=" << shape.m << ", N=" << shape.n << ", K=" << shape.k << std::endl;
    print_usage(argv[0]);
    return 1;
}
