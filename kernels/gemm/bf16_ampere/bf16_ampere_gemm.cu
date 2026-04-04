#include <cuda_runtime.h>

#include <iostream>
#include <vector>

#include "../common.cuh"
#include "kittens.cuh"

#ifdef KITTENS_C500
#ifndef __grid_constant__
#define __grid_constant__
#endif
#endif

using namespace kittens;

namespace bf16_ampere {

#ifndef BF16_AMPERE_PROBLEM_M
#define BF16_AMPERE_PROBLEM_M 4096
#endif
#ifndef BF16_AMPERE_PROBLEM_N
#define BF16_AMPERE_PROBLEM_N 4096
#endif
#ifndef BF16_AMPERE_PROBLEM_K
#define BF16_AMPERE_PROBLEM_K 4096
#endif
#ifndef BF16_AMPERE_WARMUP_ITERS
#define BF16_AMPERE_WARMUP_ITERS 25
#endif
#ifndef BF16_AMPERE_PROFILE_ITERS
#define BF16_AMPERE_PROFILE_ITERS 100
#endif
#ifndef BF16_AMPERE_C500_PIPE_STAGES
#define BF16_AMPERE_C500_PIPE_STAGES 1
#endif

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

#ifdef KITTENS_C500
using c500_atom = kittens::arch::c500::mma_bf16_16x16x16_fp32;
using c500_frag_a = kittens::arch::c500::fragment_a<c500_atom>;
using c500_frag_b = kittens::arch::c500::fragment_b<c500_atom>;
using c500_frag_c = kittens::arch::c500::fragment_c<c500_atom>;

constexpr int C500_ATOMS_M = MMA_M / c500_atom::M;
constexpr int C500_ATOMS_N = MMA_N / c500_atom::N;
constexpr int C500_ATOMS_K = MMA_K / c500_atom::K;

__device__ inline void c500_zero_accumulators(c500_frag_c (&acc)[C500_ATOMS_M][C500_ATOMS_N]) {
#pragma unroll
    for (int m = 0; m < C500_ATOMS_M; ++m) {
#pragma unroll
        for (int n = 0; n < C500_ATOMS_N; ++n) {
#pragma unroll
            for (int r = 0; r < c500_atom::c_registers; ++r) {
                acc[m][n].reg[r] = 0.0f;
            }
        }
    }
}

template<typename SharedA, typename SharedB>
__device__ inline void c500_mma_tile(c500_frag_c (&acc)[C500_ATOMS_M][C500_ATOMS_N],
                                     const SharedA &a_tile,
                                     const SharedB &b_tile) {
    c500_frag_a a_frag[C500_ATOMS_M][C500_ATOMS_K];
    c500_frag_b b_frag[C500_ATOMS_K][C500_ATOMS_N];

#pragma unroll
    for (int m = 0; m < C500_ATOMS_M; ++m) {
#pragma unroll
        for (int k = 0; k < C500_ATOMS_K; ++k) {
            kittens::arch::c500::load_a<c500_atom>(a_frag[m][k], a_tile, m * c500_atom::M, k * c500_atom::K);
        }
    }

#pragma unroll
    for (int k = 0; k < C500_ATOMS_K; ++k) {
#pragma unroll
        for (int n = 0; n < C500_ATOMS_N; ++n) {
            kittens::arch::c500::load_b<c500_atom>(b_frag[k][n], b_tile, k * c500_atom::K, n * c500_atom::N);
        }
    }

#pragma unroll
    for (int m = 0; m < C500_ATOMS_M; ++m) {
#pragma unroll
        for (int n = 0; n < C500_ATOMS_N; ++n) {
#pragma unroll
            for (int k = 0; k < C500_ATOMS_K; ++k) {
                c500_frag_c next;
                kittens::arch::c500::mma<c500_atom>(next, a_frag[m][k], b_frag[k][n], acc[m][n]);
                acc[m][n] = next;
            }
        }
    }
}

__device__ inline void c500_export_accumulators(reg_tileC &dst,
                                                const c500_frag_c (&acc)[C500_ATOMS_M][C500_ATOMS_N]) {
#pragma unroll
    for (int m = 0; m < C500_ATOMS_M; ++m) {
#pragma unroll
        for (int n = 0; n < C500_ATOMS_N; ++n) {
            const int lane_id = kittens::laneid();
            const int row = lane_id & 0x0f;
            const int lane_group = lane_id >> 4;
            const float r0 = acc[m][n].reg[0];
            const float r1 = acc[m][n].reg[1];
            const float r2 = acc[m][n].reg[2];
            const float r3 = acc[m][n].reg[3];
            const float src0_r0 = __shfl_sync(0xffffffffffffffffull, r0, row + 0 * 16, 64);
            const float src1_r0 = __shfl_sync(0xffffffffffffffffull, r0, row + 1 * 16, 64);
            const float src2_r0 = __shfl_sync(0xffffffffffffffffull, r0, row + 2 * 16, 64);
            const float src3_r0 = __shfl_sync(0xffffffffffffffffull, r0, row + 3 * 16, 64);
            const float src0_r1 = __shfl_sync(0xffffffffffffffffull, r1, row + 0 * 16, 64);
            const float src1_r1 = __shfl_sync(0xffffffffffffffffull, r1, row + 1 * 16, 64);
            const float src2_r1 = __shfl_sync(0xffffffffffffffffull, r1, row + 2 * 16, 64);
            const float src3_r1 = __shfl_sync(0xffffffffffffffffull, r1, row + 3 * 16, 64);
            const float src0_r2 = __shfl_sync(0xffffffffffffffffull, r2, row + 0 * 16, 64);
            const float src1_r2 = __shfl_sync(0xffffffffffffffffull, r2, row + 1 * 16, 64);
            const float src2_r2 = __shfl_sync(0xffffffffffffffffull, r2, row + 2 * 16, 64);
            const float src3_r2 = __shfl_sync(0xffffffffffffffffull, r2, row + 3 * 16, 64);
            const float src0_r3 = __shfl_sync(0xffffffffffffffffull, r3, row + 0 * 16, 64);
            const float src1_r3 = __shfl_sync(0xffffffffffffffffull, r3, row + 1 * 16, 64);
            const float src2_r3 = __shfl_sync(0xffffffffffffffffull, r3, row + 2 * 16, 64);
            const float src3_r3 = __shfl_sync(0xffffffffffffffffull, r3, row + 3 * 16, 64);

            if (lane_group == 0) {
                dst.tiles[m][n].data[0].x = src0_r0;
                dst.tiles[m][n].data[0].y = src1_r0;
                dst.tiles[m][n].data[1].x = src2_r0;
                dst.tiles[m][n].data[1].y = src3_r0;
            } else if (lane_group == 1) {
                dst.tiles[m][n].data[0].x = src0_r1;
                dst.tiles[m][n].data[0].y = src1_r1;
                dst.tiles[m][n].data[1].x = src2_r1;
                dst.tiles[m][n].data[1].y = src3_r1;
            } else if (lane_group == 2) {
                dst.tiles[m][n].data[0].x = src0_r2;
                dst.tiles[m][n].data[0].y = src1_r2;
                dst.tiles[m][n].data[1].x = src2_r2;
                dst.tiles[m][n].data[1].y = src3_r2;
            } else {
                dst.tiles[m][n].data[0].x = src0_r3;
                dst.tiles[m][n].data[0].y = src1_r3;
                dst.tiles[m][n].data[1].x = src2_r3;
                dst.tiles[m][n].data[1].y = src3_r3;
            }
        }
    }
}
#endif

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
#ifdef KITTENS_C500
__global__ __launch_bounds__(BLOCK_SIZE) void gemm_kernel(
#else
__global__ __launch_bounds__(BLOCK_SIZE, 1) void gemm_kernel(
#endif
    const __grid_constant__ gemm_globals<M, N, K> g) {
    using load_group = kittens::group<(NUM_WORKERS / LOAD_GROUPS)>;
#ifdef KITTENS_C500
    constexpr int k_pipe_stages = BF16_AMPERE_C500_PIPE_STAGES;
#else
    constexpr int k_pipe_stages = PIPE_STAGES;
#endif

    const int workerid = kittens::warpid();
    const int row_worker = workerid / 2;
    const int col_worker = workerid % 2;
    const int load_id = load_group::groupid();
    constexpr int LOAD_BLOCKS = NUM_WORKERS / load_group::GROUP_WARPS;

    const int warp_row = LOAD_GROUPS * blockIdx.y;
    const int warp_col = LOAD_GROUPS * blockIdx.x;

    __shared__ shared_tileA a_s[LOAD_BLOCKS][k_pipe_stages];
    __shared__ shared_tileB b_s[LOAD_BLOCKS][k_pipe_stages];
#ifdef KITTENS_C500
    c500_frag_c cr_native[C500_ATOMS_M][C500_ATOMS_N];
    reg_tileA ar_bf;
    reg_tileB br_bf;
#else
    reg_tileA ar_bf;
    reg_tileB br_bf;
#endif
    reg_tileC cr_fl;

#ifdef KITTENS_C500
    c500_zero_accumulators(cr_native);
#else
    kittens::warp::zero(cr_fl);
#endif

    const int num_k_tiles = K / MMA_K;
#ifdef KITTENS_C500
    static_assert(k_pipe_stages >= 1 && k_pipe_stages <= PIPE_STAGES,
                  "C500 pipe stages must be between 1 and PIPE_STAGES.");
    if constexpr (k_pipe_stages == 1) {
        for (int inner = 0; inner < num_k_tiles; ++inner) {
            load_group::load<2, true>(a_s[load_id][0], g.a, {warp_row + load_id, inner});
            load_group::load<2, true>(b_s[load_id][0], g.b, {inner, warp_col + load_id});
            __syncthreads();
            c500_mma_tile(cr_native, a_s[row_worker][0], b_s[col_worker][0]);
            if (inner + 1 < num_k_tiles) {
                __syncthreads();
            }
        }
    } else {
        int tic = 0;
        load_group::load<2, true>(a_s[load_id][tic], g.a, {warp_row + load_id, 0});
        load_group::load<2, true>(b_s[load_id][tic], g.b, {0, warp_col + load_id});
        __syncthreads();
        for (int inner = 0; inner < num_k_tiles; ++inner) {
            c500_mma_tile(cr_native, a_s[row_worker][tic], b_s[col_worker][tic]);

            const int next_load_idx = inner + 1;
            if (next_load_idx < num_k_tiles) {
                const int next_tic = (tic + 1) % k_pipe_stages;
                load_group::load<2, true>(a_s[load_id][next_tic], g.a,
                                          {warp_row + load_id, next_load_idx});
                load_group::load<2, true>(b_s[load_id][next_tic], g.b,
                                          {next_load_idx, warp_col + load_id});
                __syncthreads();
                tic = next_tic;
            }
        }
    }

    c500_export_accumulators(cr_fl, cr_native);
#else
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
#endif

    kittens::warp::store(g.c, cr_fl, {0, 0, warp_row + row_worker, warp_col + col_worker});
}

template <int M, int N, int K>
__host__ void launch_gemm(bf16 *A, bf16 *B, bf16 *C) {
    const dim3 grid((N + (BLOCK_N * LOAD_GROUPS) - 1) / (BLOCK_N * LOAD_GROUPS),
                    (M + (BLOCK_M * LOAD_GROUPS) - 1) / (BLOCK_M * LOAD_GROUPS));
    gemm_globals<M, N, K> g = gemm_init<M, N, K>(A, B, C);

    gemm_kernel<M, N, K><<<grid, BLOCK_SIZE>>>(g);
}

}  // namespace bf16_ampere

int main() {
    constexpr int M = BF16_AMPERE_PROBLEM_M;
    constexpr int N = BF16_AMPERE_PROBLEM_N;
    constexpr int K = BF16_AMPERE_PROBLEM_K;

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

    reference_gemm<__nv_bfloat16, __nv_bfloat16, false>(d_C_ref, d_A, d_B, M, N, K);
    cudaDeviceSynchronize();

    constexpr int warmup_iters = BF16_AMPERE_WARMUP_ITERS;
    constexpr int profiling_iters = BF16_AMPERE_PROFILE_ITERS;

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

    bf16_ampere::launch_gemm<M, N, K>(reinterpret_cast<bf16 *>(d_A),
                                      reinterpret_cast<bf16 *>(d_B),
                                      reinterpret_cast<bf16 *>(d_C));
    cudaDeviceSynchronize();
    check_correctness(d_C, d_C_ref, size_c);

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
