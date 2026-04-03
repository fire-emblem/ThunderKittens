#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "../common.cuh"
#include "kittens.cuh"

using namespace kittens;

namespace bf16_ampere_256 {

constexpr int MMA_M = 64;
constexpr int MMA_N = 64;
constexpr int MMA_K = 32;

constexpr int NUM_WORKERS = 8;
constexpr int PIPE_STAGES = 4;
constexpr int LOAD_GROUPS_A = 4;
constexpr int LOAD_GROUPS_B = 2;
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
    using load_group_a = kittens::group<(NUM_WORKERS / LOAD_GROUPS_A)>;
    using load_group_b = kittens::group<(NUM_WORKERS / LOAD_GROUPS_B)>;

    const int workerid = kittens::warpid();
    const int row_worker = workerid / 2;
    const int col_worker = workerid % 2;
    const int load_id_a = load_group_a::groupid();
    const int load_id_b = load_group_b::groupid();
    constexpr int LOAD_BLOCKS_A = NUM_WORKERS / load_group_a::GROUP_WARPS;
    constexpr int LOAD_BLOCKS_B = NUM_WORKERS / load_group_b::GROUP_WARPS;

    const int warp_row = LOAD_GROUPS_A * blockIdx.y;
    const int warp_col = LOAD_GROUPS_B * blockIdx.x;

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int *)&__shm[0]);

    shared_tileA (&a_s)[LOAD_BLOCKS_A][PIPE_STAGES] =
        al.allocate<shared_tileA, LOAD_BLOCKS_A, PIPE_STAGES>();
    shared_tileB (&b_s)[LOAD_BLOCKS_B][PIPE_STAGES] =
        al.allocate<shared_tileB, LOAD_BLOCKS_B, PIPE_STAGES>();
    reg_tileA ar_bf;
    reg_tileB br_bf;
    reg_tileC cr_fl;

    kittens::warp::zero(cr_fl);

    const int num_k_tiles = K / MMA_K;
    int tic = 0;

    load_group_a::load_async<2, false>(a_s[load_id_a][tic], g.a, {warp_row + load_id_a, 0});
    load_group_b::load_async<2, false>(b_s[load_id_b][tic], g.b, {0, warp_col + load_id_b});

    for (int inner = 0; inner < num_k_tiles; ++inner, tic = (tic + 1) % PIPE_STAGES) {
        const int next_load_idx = inner + 1;
        if (next_load_idx < num_k_tiles) {
            const int next_tic = (tic + 1) % PIPE_STAGES;
            load_group_a::load_async<2, false>(a_s[load_id_a][next_tic], g.a,
                                               {warp_row + load_id_a, next_load_idx});
            load_group_b::load_async<2, false>(b_s[load_id_b][next_tic], g.b,
                                               {next_load_idx, warp_col + load_id_b});
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
    const dim3 grid((N + (64 * LOAD_GROUPS_B) - 1) / (64 * LOAD_GROUPS_B),
                    (M + (64 * LOAD_GROUPS_A) - 1) / (64 * LOAD_GROUPS_A));
    gemm_globals<M, N, K> g = gemm_init<M, N, K>(A, B, C);
    unsigned long mem_size = 100000;
    cudaFuncSetAttribute(gemm_kernel<M, N, K>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    gemm_kernel<M, N, K><<<grid, BLOCK_SIZE, mem_size>>>(g);
}

}  // namespace bf16_ampere_256

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

template <int M, int N, int K>
int run_benchmark(bool verify) {
    std::cout << "bf16_ampere_256 TK GEMM" << std::endl;
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
        bf16_ampere_256::launch_gemm<M, N, K>(reinterpret_cast<bf16 *>(d_A),
                                              reinterpret_cast<bf16 *>(d_B),
                                              reinterpret_cast<bf16 *>(d_C));
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < profiling_iters; ++i) {
        bf16_ampere_256::launch_gemm<M, N, K>(reinterpret_cast<bf16 *>(d_A),
                                              reinterpret_cast<bf16 *>(d_B),
                                              reinterpret_cast<bf16 *>(d_C));
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
        bf16_ampere_256::launch_gemm<M, N, K>(reinterpret_cast<bf16 *>(d_A),
                                              reinterpret_cast<bf16 *>(d_B),
                                              reinterpret_cast<bf16 *>(d_C));
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
    ShapeSpec shape{1024, 1024, 1024};
    parse_arg_int(argv + 1, argv + argc, "--m", shape.m);
    parse_arg_int(argv + 1, argv + argc, "--n", shape.n);
    parse_arg_int(argv + 1, argv + argc, "--k", shape.k);
    const bool verify = !has_flag(argv + 1, argv + argc, "--no-check");
    if (shape.m == 1024 && shape.n == 1024 && shape.k == 1024) {
        return run_benchmark<1024, 1024, 1024>(verify);
    }
    std::cerr << "Unsupported shape" << std::endl;
    return 1;
}
