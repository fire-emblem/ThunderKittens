#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../common.cuh"
#include "kittens.cuh"
#include "arch/c500/copy_atoms.cuh"
#include "arch/c500/gemm/bf16_contracts.cuh"
#include "arch/c500/gemm/bf16_mainloop.cuh"
#include "arch/c500/gemm/bf16_operand_stage.cuh"
#include "arch/c500/gemm/bf16_stage_primitives.cuh"

#ifdef KITTENS_C500
#ifndef __grid_constant__
#define __grid_constant__
#endif
#endif

using namespace kittens;

namespace bf16_c500_perf_probe {

using contracts = kittens::arch::c500::gemm::bf16_contracts;
using family = kittens::arch::c500::gemm::bf16_mainloop_family;
using atom = kittens::arch::c500::gemm::bf16_mainloop_atom;
using frag_a = kittens::arch::c500::fragment_a<atom>;
using frag_b = kittens::arch::c500::fragment_b<atom>;
using frag_c = kittens::arch::c500::fragment_c<atom>;
using shared_tileA = kittens::arch::c500::gemm::bf16_shared_tile_a;
using shared_tileB = kittens::arch::c500::gemm::bf16_shared_tile_b;
using shared_tileC = kittens::arch::c500::gemm::bf16_shared_tile_c;
using raw_stage_ring = kittens::arch::c500::gemm::bf16_stage_ring;
using operand_vec = kittens::arch::c500::gemm::bf16_operand_vec;

constexpr int kAtomsM = kittens::arch::c500::gemm::kAtomsM;
constexpr int kAtomsN = kittens::arch::c500::gemm::kAtomsN;
constexpr int kGridX = 16;
constexpr int kGridY = 16;
constexpr int kMicroBlocks = kGridX * kGridY;
constexpr int kMicroM = kGridY * contracts::kBlockM;
constexpr int kMicroN = kGridX * contracts::kBlockN;
constexpr int kMicroK = contracts::kStageK * 8;

#ifndef BF16_C500_PERF_WARMUP_ITERS
#define BF16_C500_PERF_WARMUP_ITERS 5
#endif
#ifndef BF16_C500_PERF_PROFILE_ITERS
#define BF16_C500_PERF_PROFILE_ITERS 20
#endif
#ifndef BF16_C500_PERF_INNER_ITERS
#define BF16_C500_PERF_INNER_ITERS 2048
#endif
#ifndef BF16_C500_PERF_FULL_GEMM_M
#define BF16_C500_PERF_FULL_GEMM_M 4096
#endif
#ifndef BF16_C500_PERF_FULL_GEMM_N
#define BF16_C500_PERF_FULL_GEMM_N 4096
#endif
#ifndef BF16_C500_PERF_FULL_GEMM_K
#define BF16_C500_PERF_FULL_GEMM_K 4096
#endif

template<int M, int K>
using a_gl = gl<bf16, 1, 1, M, K, shared_tileA>;
template<int K, int N>
using b_gl = gl<bf16, 1, 1, K, N, shared_tileB>;
template<int M, int N>
using c_gl = gl<bf16, 1, 1, M, N, shared_tileC>;

template<int M, int N, int K>
struct gemm_globals {
    a_gl<M, K> a;
    b_gl<K, N> b;
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
__global__ __launch_bounds__(contracts::kThreads)
void full_gemm_kernel(const __grid_constant__ gemm_globals<M, N, K> g) {
    kittens::arch::c500::gemm::run_bf16_mainloop<M, N, K>(g);
}

template<int M, int N, int K>
__host__ void launch_full_gemm(bf16 *a, bf16 *b, bf16 *c) {
    const dim3 grid(N / contracts::kBlockN, M / contracts::kBlockM);
    auto g = gemm_init<M, N, K>(a, b, c);
    full_gemm_kernel<M, N, K><<<grid, contracts::kThreads>>>(g);
}

__device__ inline void zero_acc(frag_c (&acc)[kAtomsM][kAtomsN]) {
#pragma unroll
    for (int m = 0; m < kAtomsM; ++m) {
#pragma unroll
        for (int n = 0; n < kAtomsN; ++n) {
#pragma unroll
            for (int i = 0; i < atom::c_registers; ++i) {
                acc[m][n].reg[i] = 0.0f;
            }
        }
    }
}

__device__ inline uint32_t pack_bf16_pair(float lo, float hi) {
    return kittens::arch::c500::gemm::pack_operand_pair(__float2bfloat16(lo), __float2bfloat16(hi));
}

__device__ inline operand_vec make_operand_words(int lane, int atom_idx, int half_seed) {
    const float base = static_cast<float>((lane + 1) * (atom_idx + 3) + half_seed * 7);
    return operand_vec{
        pack_bf16_pair(base + 0.0f, base + 1.0f),
        pack_bf16_pair(base + 2.0f, base + 3.0f),
        pack_bf16_pair(base + 4.0f, base + 5.0f),
        pack_bf16_pair(base + 6.0f, base + 7.0f)
    };
}

__device__ inline float checksum_acc(const frag_c (&acc)[kAtomsM][kAtomsN]) {
    float sum = 0.0f;
#pragma unroll
    for (int m = 0; m < kAtomsM; ++m) {
#pragma unroll
        for (int n = 0; n < kAtomsN; ++n) {
#pragma unroll
            for (int i = 0; i < atom::c_registers; ++i) {
                sum += acc[m][n].reg[i];
            }
        }
    }
    return sum;
}

__device__ inline void load_stage_operands_to_regs(const raw_stage_ring &ring,
                                                   int stage_slot,
                                                   int row_group,
                                                   int col_group,
                                                   operand_vec (&a_words)[contracts::kStageK / atom::K][kAtomsM],
                                                   operand_vec (&b_words)[contracts::kStageK / atom::K][kAtomsN]) {
    const int lane = kittens::laneid();

#pragma unroll
    for (int mma_k = 0; mma_k < contracts::kStageK / atom::K; ++mma_k) {
#pragma unroll
        for (int m = 0; m < kAtomsM; ++m) {
            a_words[mma_k][m] =
                kittens::arch::c500::gemm::bridge_raw_stage_a_to_operand_aligned(ring, stage_slot, row_group, m, mma_k, lane);
        }
#pragma unroll
        for (int n = 0; n < kAtomsN; ++n) {
            b_words[mma_k][n] =
                kittens::arch::c500::gemm::bridge_raw_stage_b_to_operand_aligned(ring, stage_slot, col_group, n, mma_k, lane);
        }
    }
}

__device__ inline void mma_operands_from_regs(const operand_vec (&a_words)[contracts::kStageK / atom::K][kAtomsM],
                                              const operand_vec (&b_words)[contracts::kStageK / atom::K][kAtomsN],
                                              frag_c (&acc)[kAtomsM][kAtomsN]) {
    frag_a a_frag[kAtomsM];
    frag_b b_frag[kAtomsN];

#pragma unroll
    for (int mma_k = 0; mma_k < contracts::kStageK / atom::K; ++mma_k) {
#pragma unroll
        for (int m = 0; m < kAtomsM; ++m) {
            a_frag[m] = kittens::arch::c500::gemm::make_a_operand_fragment(a_words[mma_k][m], 0);
        }
#pragma unroll
        for (int n = 0; n < kAtomsN; ++n) {
            b_frag[n] = kittens::arch::c500::gemm::make_b_operand_fragment(b_words[mma_k][n], 0);
        }
#pragma unroll
        for (int m = 0; m < kAtomsM; ++m) {
#pragma unroll
            for (int n = 0; n < kAtomsN; ++n) {
                frag_c next{};
                kittens::arch::c500::mma<atom>(next, a_frag[m], b_frag[n], acc[m][n]);
                acc[m][n] = next;
            }
        }
    }
}

__device__ inline void seed_raw_stage(raw_stage_ring &ring) {
    const int tid = threadIdx.x;

#pragma unroll
    for (int group = 0; group < contracts::kLoadGroups; ++group) {
        auto *a_ptr = reinterpret_cast<bf16 *>(ring.bytes +
                                               kittens::arch::c500::gemm::bf16_128x128x128_stage_layout::a_group_offset(0, group));
        auto *b_ptr = reinterpret_cast<bf16 *>(ring.bytes +
                                               kittens::arch::c500::gemm::bf16_128x128x128_stage_layout::b_group_offset(0, group));

        for (int idx = tid; idx < contracts::kWarpM * contracts::kStageK; idx += blockDim.x) {
            const int row = idx / contracts::kStageK;
            const int col = idx % contracts::kStageK;
            const float value = static_cast<float>((group + 1) * 1000 + row * contracts::kStageK + col);
            a_ptr[idx] = __float2bfloat16(value / 1024.0f);
        }

        for (int idx = tid; idx < contracts::kStageK * contracts::kWarpN; idx += blockDim.x) {
            const int row = idx / contracts::kWarpN;
            const int col = idx % contracts::kWarpN;
            const float value = static_cast<float>((group + 1) * 2000 + row * contracts::kWarpN + col);
            b_ptr[idx] = __float2bfloat16(value / 1024.0f);
        }
    }
}

__global__ __launch_bounds__(contracts::kThreads)
void mma_only_kernel(float *sink, int inner_iters) {
    const int lane = kittens::laneid();
    frag_c acc[kAtomsM][kAtomsN];
    frag_a a_frag[kAtomsM];
    frag_b b_frag[kAtomsN];
    operand_vec a_words[kAtomsM];
    operand_vec b_words[kAtomsN];

    zero_acc(acc);

#pragma unroll
    for (int m = 0; m < kAtomsM; ++m) {
        a_words[m] = make_operand_words(lane, m, 0);
    }
#pragma unroll
    for (int n = 0; n < kAtomsN; ++n) {
        b_words[n] = make_operand_words(lane, n + 8, 1);
    }

    for (int iter = 0; iter < inner_iters; ++iter) {
#pragma unroll
        for (int mma_k = 0; mma_k < contracts::kStageK / atom::K; ++mma_k) {
#pragma unroll
            for (int m = 0; m < kAtomsM; ++m) {
                a_frag[m] = kittens::arch::c500::gemm::make_a_operand_fragment(a_words[m], mma_k);
            }
#pragma unroll
            for (int n = 0; n < kAtomsN; ++n) {
                b_frag[n] = kittens::arch::c500::gemm::make_b_operand_fragment(b_words[n], mma_k);
            }
#pragma unroll
            for (int m = 0; m < kAtomsM; ++m) {
#pragma unroll
                for (int n = 0; n < kAtomsN; ++n) {
                    frag_c next{};
                    kittens::arch::c500::mma<atom>(next, a_frag[m], b_frag[n], acc[m][n]);
                    acc[m][n] = next;
                }
            }
        }
    }

    const int linear_block = blockIdx.y * gridDim.x + blockIdx.x;
    sink[linear_block * blockDim.x + threadIdx.x] = checksum_acc(acc);
}

__global__ __launch_bounds__(contracts::kThreads)
void mma_smem_kernel(float *sink, int inner_iters) {
    __shared__ raw_stage_ring ring;

    const int worker = kittens::warpid();
    const int row_group = worker / contracts::kWaveN;
    const int col_group = worker % contracts::kWaveN;

    frag_c acc[kAtomsM][kAtomsN];
    zero_acc(acc);

    seed_raw_stage(ring);
    __syncthreads();

    for (int iter = 0; iter < inner_iters; ++iter) {
        kittens::arch::c500::gemm::mma_raw_stage_aligned_tile_bridge(ring, 0, row_group, col_group, acc);
    }

    const int linear_block = blockIdx.y * gridDim.x + blockIdx.x;
    sink[linear_block * blockDim.x + threadIdx.x] = checksum_acc(acc);
}

template<int ActiveStages>
__global__ __launch_bounds__(contracts::kThreads)
void mma_gmem_smem_multistage_kernel(const __grid_constant__ gemm_globals<kMicroM, kMicroN, kMicroK> g,
                                     float *sink,
                                     int inner_iters) {
    __shared__ raw_stage_ring ring;

    const int worker = kittens::warpid();
    const int row_group = worker / contracts::kWaveN;
    const int col_group = worker % contracts::kWaveN;
    const int load_group = worker / 2;
    const int warp_row = contracts::kLoadGroups * blockIdx.y;
    const int warp_col = contracts::kLoadGroups * blockIdx.x;

    frag_c acc[kAtomsM][kAtomsN];
    operand_vec a_words[contracts::kStageK / atom::K][kAtomsM];
    operand_vec b_words[contracts::kStageK / atom::K][kAtomsN];
    zero_acc(acc);

    static_assert(ActiveStages >= 2 && ActiveStages <= contracts::kStages,
                  "Perf probe only supports 2-stage or 4-stage multistage tests.");
    constexpr int kGlobalStages = kMicroK / contracts::kStageK;

#pragma unroll
    for (int stage = 0; stage < ActiveStages; ++stage) {
        family::issue_stage_async(ring, g, load_group, warp_row, warp_col, stage, stage);
    }
    family::wait_stage_window(ActiveStages - 1);
    __syncthreads();

    for (int iter = 0; iter < inner_iters; ++iter) {
        const int current_stage = iter % kGlobalStages;
        const int stage_slot = current_stage % ActiveStages;
        const int next_k_stage = (current_stage + ActiveStages) % kGlobalStages;

        load_stage_operands_to_regs(ring, stage_slot, row_group, col_group, a_words, b_words);
        family::issue_stage_async(ring, g, load_group, warp_row, warp_col, stage_slot, next_k_stage);
        mma_operands_from_regs(a_words, b_words, acc);
        kittens::arch::c500::wait_until<0>();
        __syncthreads();
    }

    const int linear_block = blockIdx.y * gridDim.x + blockIdx.x;
    sink[linear_block * blockDim.x + threadIdx.x] = checksum_acc(acc);
}

struct timed_result {
    double runtime_ms = 0.0;
    double tflops = 0.0;
};

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
timed_result time_kernel(LaunchFn launch, double total_flops, int warmup_iters, int profile_iters) {
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

    const double runtime_ms = total_ms / profile_iters;
    const double runtime_s = runtime_ms / 1000.0;
    return {runtime_ms, (total_flops / 1.0e12) / runtime_s};
}

template<typename T>
void init_bf16(std::vector<T> &buf, int mul, int add) {
    for (size_t i = 0; i < buf.size(); ++i) {
        const float value = static_cast<float>((static_cast<int>(i) * mul + add) % 1024 - 512) / 512.0f;
        buf[i] = __float2bfloat16(value);
    }
}

int run() {
    constexpr int warmup_iters = BF16_C500_PERF_WARMUP_ITERS;
    constexpr int profile_iters = BF16_C500_PERF_PROFILE_ITERS;
    constexpr int inner_iters = BF16_C500_PERF_INNER_ITERS;
    constexpr int full_m = BF16_C500_PERF_FULL_GEMM_M;
    constexpr int full_n = BF16_C500_PERF_FULL_GEMM_N;
    constexpr int full_k = BF16_C500_PERF_FULL_GEMM_K;
    constexpr dim3 micro_grid(kGridX, kGridY);

    static_assert(full_m % contracts::kBlockM == 0, "Full GEMM M must be a multiple of 128.");
    static_assert(full_n % contracts::kBlockN == 0, "Full GEMM N must be a multiple of 128.");
    static_assert(full_k % contracts::kStageK == 0, "Full GEMM K must be a multiple of 32.");

    int sm_count = 0;
    require(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0), "cudaDeviceGetAttribute(SM)");

    const size_t micro_sink_elems = static_cast<size_t>(kMicroBlocks) * contracts::kThreads;
    float *d_sink = nullptr;
    require(cudaMalloc(&d_sink, micro_sink_elems * sizeof(float)), "cudaMalloc(sink)");

    const size_t micro_a_elems = static_cast<size_t>(kMicroM) * kMicroK;
    const size_t micro_b_elems = static_cast<size_t>(kMicroK) * kMicroN;
    bf16 *d_micro_a = nullptr;
    bf16 *d_micro_b = nullptr;
    require(cudaMalloc(&d_micro_a, micro_a_elems * sizeof(bf16)), "cudaMalloc(micro_a)");
    require(cudaMalloc(&d_micro_b, micro_b_elems * sizeof(bf16)), "cudaMalloc(micro_b)");

    std::vector<bf16> h_micro_a(micro_a_elems);
    std::vector<bf16> h_micro_b(micro_b_elems);
    init_bf16(h_micro_a, 17, 3);
    init_bf16(h_micro_b, 29, 5);
    require(cudaMemcpy(d_micro_a, h_micro_a.data(), micro_a_elems * sizeof(bf16), cudaMemcpyHostToDevice), "copy micro A");
    require(cudaMemcpy(d_micro_b, h_micro_b.data(), micro_b_elems * sizeof(bf16), cudaMemcpyHostToDevice), "copy micro B");

    const size_t full_a_elems = static_cast<size_t>(full_m) * full_k;
    const size_t full_b_elems = static_cast<size_t>(full_k) * full_n;
    const size_t full_c_elems = static_cast<size_t>(full_m) * full_n;
    bf16 *d_full_a = nullptr;
    bf16 *d_full_b = nullptr;
    bf16 *d_full_c = nullptr;
    require(cudaMalloc(&d_full_a, full_a_elems * sizeof(bf16)), "cudaMalloc(full_a)");
    require(cudaMalloc(&d_full_b, full_b_elems * sizeof(bf16)), "cudaMalloc(full_b)");
    require(cudaMalloc(&d_full_c, full_c_elems * sizeof(bf16)), "cudaMalloc(full_c)");

    std::vector<bf16> h_full_a(full_a_elems);
    std::vector<bf16> h_full_b(full_b_elems);
    init_bf16(h_full_a, 11, 7);
    init_bf16(h_full_b, 13, 9);
    require(cudaMemcpy(d_full_a, h_full_a.data(), full_a_elems * sizeof(bf16), cudaMemcpyHostToDevice), "copy full A");
    require(cudaMemcpy(d_full_b, h_full_b.data(), full_b_elems * sizeof(bf16), cudaMemcpyHostToDevice), "copy full B");
    require(cudaMemset(d_full_c, 0, full_c_elems * sizeof(bf16)), "clear full C");

    const double micro_flops = static_cast<double>(kMicroBlocks) * inner_iters *
                               (2.0 * contracts::kBlockM * contracts::kBlockN * contracts::kStageK);
    const double full_flops = 2.0 * static_cast<double>(full_m) * full_n * full_k;

    const auto mma_only = time_kernel(
        [&]() {
            mma_only_kernel<<<micro_grid, contracts::kThreads>>>(d_sink, inner_iters);
        },
        micro_flops,
        warmup_iters,
        profile_iters);

    const auto mma_smem = time_kernel(
        [&]() {
            mma_smem_kernel<<<micro_grid, contracts::kThreads>>>(d_sink, inner_iters);
        },
        micro_flops,
        warmup_iters,
        profile_iters);

    auto micro_globals = gemm_init<kMicroM, kMicroN, kMicroK>(d_micro_a, d_micro_b, reinterpret_cast<bf16 *>(d_full_c));
    const auto mma_gmem_smem_2stage = time_kernel(
        [&]() {
            mma_gmem_smem_multistage_kernel<2><<<micro_grid, contracts::kThreads>>>(micro_globals, d_sink, inner_iters);
        },
        micro_flops,
        warmup_iters,
        profile_iters);

    const auto mma_gmem_smem_4stage = time_kernel(
        [&]() {
            mma_gmem_smem_multistage_kernel<4><<<micro_grid, contracts::kThreads>>>(micro_globals, d_sink, inner_iters);
        },
        micro_flops,
        warmup_iters,
        profile_iters);

    const auto full_gemm = time_kernel(
        [&]() {
            launch_full_gemm<full_m, full_n, full_k>(d_full_a, d_full_b, d_full_c);
        },
        full_flops,
        warmup_iters,
        profile_iters);

    std::vector<float> h_sink(8, 0.0f);
    require(cudaMemcpy(h_sink.data(), d_sink, h_sink.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy sink");

    std::cout << "C500 BF16 perf probe" << std::endl;
    std::cout << "Timing: cudaEvent device time only" << std::endl;
    std::cout << "SM count: " << sm_count << std::endl;
    std::cout << "Microbench grid: " << kGridX << "x" << kGridY
              << " blocks, threads/block=" << contracts::kThreads
              << ", inner_iters=" << inner_iters << std::endl;
    std::cout << "Full GEMM shape: " << full_m << "x" << full_n << "x" << full_k << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "mma_only       runtime_ms=" << mma_only.runtime_ms
              << " tflops=" << mma_only.tflops << std::endl;
    std::cout << "mma_smem       runtime_ms=" << mma_smem.runtime_ms
              << " tflops=" << mma_smem.tflops << std::endl;
    std::cout << "mma_gmem_2s    runtime_ms=" << mma_gmem_smem_2stage.runtime_ms
              << " tflops=" << mma_gmem_smem_2stage.tflops << std::endl;
    std::cout << "mma_gmem_4s    runtime_ms=" << mma_gmem_smem_4stage.runtime_ms
              << " tflops=" << mma_gmem_smem_4stage.tflops << std::endl;
    std::cout << "full_gemm      runtime_ms=" << full_gemm.runtime_ms
              << " tflops=" << full_gemm.tflops << std::endl;
    std::cout << "sink_sample    " << h_sink[0] << " " << h_sink[1] << " " << h_sink[2] << std::endl;

    cudaFree(d_sink);
    cudaFree(d_micro_a);
    cudaFree(d_micro_b);
    cudaFree(d_full_a);
    cudaFree(d_full_b);
    cudaFree(d_full_c);
    return 0;
}

} // namespace bf16_c500_perf_probe

int main() { return bf16_c500_perf_probe::run(); }
