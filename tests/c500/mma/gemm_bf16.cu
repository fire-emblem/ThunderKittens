#include "testing_flags.cuh"

#ifdef TEST_C500_MMA_GEMM_BF16

#include "testing_commons.cuh"
#include "../../../kernels/gemm/common.cuh"

namespace c500::mma::gemm_bf16 {

namespace {

constexpr int kM = 128;
constexpr int kN = 128;
constexpr int kK = 32;
constexpr int kWarpM = 64;
constexpr int kWarpN = 64;
constexpr int kLoadGroups = 2;
constexpr int kNumWorkers = 4;
constexpr int kBlockSize = kNumWorkers * kittens::WARP_THREADS;

using atom = kittens::arch::c500::mma_bf16_16x16x16_fp32;
using shared_tileA = kittens::st_bf<kWarpM, kK>;
using shared_tileB = kittens::st_bf<kK, kWarpN>;
using shared_tileC = kittens::st_bf<kWarpM, kWarpN>;
using reg_tileC = kittens::rt_fl<kWarpM, kWarpN>;
using frag_a = kittens::arch::c500::fragment_a<atom>;
using frag_b = kittens::arch::c500::fragment_b<atom>;
using frag_c = kittens::arch::c500::fragment_c<atom>;

constexpr int kAtomsM = kWarpM / atom::M;
constexpr int kAtomsN = kWarpN / atom::N;
constexpr int kAtomsK = kK / atom::K;

template <int M, int K>
using a_gl = kittens::gl<kittens::bf16, 1, 1, M, K, shared_tileA>;
template <int K, int N>
using b_gl = kittens::gl<kittens::bf16, 1, 1, K, N, shared_tileB>;
template <int M, int N>
using c_gl = kittens::gl<kittens::bf16, 1, 1, M, N, shared_tileC>;

struct gemm_globals {
    a_gl<kM, kK> a;
    b_gl<kK, kN> b;
    c_gl<kM, kN> c;
};

__device__ inline void zero_accumulators(frag_c (&acc)[kAtomsM][kAtomsN]) {
#pragma unroll
    for (int m = 0; m < kAtomsM; ++m) {
#pragma unroll
        for (int n = 0; n < kAtomsN; ++n) {
#pragma unroll
            for (int r = 0; r < atom::c_registers; ++r) {
                acc[m][n].reg[r] = 0.0f;
            }
        }
    }
}

template<typename SharedA, typename SharedB>
__device__ inline void mma_tile(frag_c (&acc)[kAtomsM][kAtomsN],
                                const SharedA &a_tile,
                                const SharedB &b_tile) {
    frag_a a_frag[kAtomsM][kAtomsK];
    frag_b b_frag[kAtomsK][kAtomsN];

#pragma unroll
    for (int m = 0; m < kAtomsM; ++m) {
#pragma unroll
        for (int k = 0; k < kAtomsK; ++k) {
            kittens::arch::c500::load_a<atom>(a_frag[m][k], a_tile, m * atom::M, k * atom::K);
        }
    }

#pragma unroll
    for (int k = 0; k < kAtomsK; ++k) {
#pragma unroll
        for (int n = 0; n < kAtomsN; ++n) {
            kittens::arch::c500::load_b<atom>(b_frag[k][n], b_tile, k * atom::K, n * atom::N);
        }
    }

#pragma unroll
    for (int m = 0; m < kAtomsM; ++m) {
#pragma unroll
        for (int n = 0; n < kAtomsN; ++n) {
#pragma unroll
            for (int k = 0; k < kAtomsK; ++k) {
                frag_c next;
                kittens::arch::c500::mma<atom>(next, a_frag[m][k], b_frag[k][n], acc[m][n]);
                acc[m][n] = next;
            }
        }
    }
}

__device__ inline void export_accumulators(reg_tileC &dst,
                                           const frag_c (&acc)[kAtomsM][kAtomsN]) {
#pragma unroll
    for (int m = 0; m < kAtomsM; ++m) {
#pragma unroll
        for (int n = 0; n < kAtomsN; ++n) {
            kittens::arch::c500::store_c<atom>(dst, acc[m][n], m, n);
        }
    }
}

__global__ void gemm_smoke_kernel(const gemm_globals g) {
    using load_group = kittens::group<(kNumWorkers / kLoadGroups)>;
    constexpr int kLoadBlocks = kNumWorkers / load_group::GROUP_WARPS;

    const int workerid = kittens::warpid();
    const int row_worker = workerid / 2;
    const int col_worker = workerid % 2;
    const int load_id = load_group::groupid();

    __shared__ shared_tileA a_s[kLoadBlocks][1];
    __shared__ shared_tileB b_s[kLoadBlocks][1];

    frag_c acc[kAtomsM][kAtomsN];
    reg_tileC out;

    zero_accumulators(acc);

    load_group::load<2, true>(a_s[load_id][0], g.a, {load_id, 0});
    load_group::load<2, true>(b_s[load_id][0], g.b, {0, load_id});
    __syncthreads();

    mma_tile(acc, a_s[row_worker][0], b_s[col_worker][0]);
    export_accumulators(out, acc);
    kittens::warp::store(g.c, out, {0, 0, row_worker, col_worker});
}

bool run_smoke(test_data &results) {
    test_info info{"c500_mma_gemm_bf16_128x128x32_smoke", test_result::FAILED};

    kittens::bf16 *d_a = nullptr;
    kittens::bf16 *d_b = nullptr;
    kittens::bf16 *d_c = nullptr;
    kittens::bf16 *d_ref = nullptr;

    cudaMalloc(&d_a, kM * kK * sizeof(kittens::bf16));
    cudaMalloc(&d_b, kK * kN * sizeof(kittens::bf16));
    cudaMalloc(&d_c, kM * kN * sizeof(kittens::bf16));
    cudaMalloc(&d_ref, kM * kN * sizeof(kittens::bf16));
    CudaCheckError();

    fill<__nv_bfloat16, FillMode::RANDOM>(reinterpret_cast<__nv_bfloat16 *>(d_a), kM * kK, 2024, -1.0f, 1.0f);
    fill<__nv_bfloat16, FillMode::RANDOM>(reinterpret_cast<__nv_bfloat16 *>(d_b), kK * kN, 2025, -1.0f, 1.0f);
    fill<__nv_bfloat16, FillMode::CONSTANT>(reinterpret_cast<__nv_bfloat16 *>(d_c), kM * kN, 0.0f);
    fill<__nv_bfloat16, FillMode::CONSTANT>(reinterpret_cast<__nv_bfloat16 *>(d_ref), kM * kN, 0.0f);
    CudaCheckError();

    reference_gemm<__nv_bfloat16, __nv_bfloat16, false>(reinterpret_cast<__nv_bfloat16 *>(d_ref),
                                                        reinterpret_cast<__nv_bfloat16 *>(d_a),
                                                        reinterpret_cast<__nv_bfloat16 *>(d_b),
                                                        kM, kN, kK);
    CudaCheckError();
    cudaDeviceSynchronize();
    CudaCheckError();

    gemm_globals g{
        a_gl<kM, kK>{d_a, nullptr, nullptr, nullptr, nullptr},
        b_gl<kK, kN>{d_b, nullptr, nullptr, nullptr, nullptr},
        c_gl<kM, kN>{d_c, nullptr, nullptr, nullptr, nullptr}
    };

    gemm_smoke_kernel<<<1, kBlockSize>>>(g);
    CudaCheckError();
    cudaDeviceSynchronize();
    CudaCheckError();

    std::vector<float> empty_input;
    std::vector<float> ref_out(kM * kN, 0.0f);
    std::vector<kittens::bf16> h_ref(kM * kN);
    std::vector<kittens::bf16> h_out(kM * kN);
    cudaMemcpy(h_ref.data(), d_ref, kM * kN * sizeof(kittens::bf16), cudaMemcpyDeviceToHost);
    CudaCheckError();
    cudaMemcpy(h_out.data(), d_c, kM * kN * sizeof(kittens::bf16), cudaMemcpyDeviceToHost);
    CudaCheckError();
    for (int i = 0; i < kM * kN; ++i) {
        ref_out[i] = __bfloat162float(h_ref[i]);
    }

    bool good = true;
    int bad_idx = -1;
    for (int i = 0; i < kM * kN; ++i) {
        const float got = __bfloat162float(h_out[i]);
        if (std::abs(ref_out[i] - got) > 0.02f) {
            good = false;
            bad_idx = i;
            break;
        }
    }

    if (!good) {
        const int row = bad_idx / kN;
        const int col = bad_idx % kN;
        std::cout << "INFO: first GEMM smoke mismatch at row=" << row
                  << ", col=" << col
                  << ", ref=" << ref_out[bad_idx]
                  << ", out=" << __bfloat162float(h_out[bad_idx]) << std::endl;
    }

    info.result = validate(d_a, d_c, empty_input, ref_out, info.label, kN, 0.02f);
    results.push_back(info);

    cudaFree(d_b);
    cudaFree(d_ref);
    CudaCheckError();
    return info.result == test_result::PASSED;
}

} // namespace

void tests(test_data &results) {
    std::cout << " ----- Starting ops/c500/mma/gemm_bf16 tests! -----\n" << std::endl;
    run_smoke(results);
    std::cout << "INFO: C500 bf16 GEMM smoke coverage exercises the native 64x64x32 hot path with shared->native->mma->export.\n" << std::endl;
}

} // namespace c500::mma::gemm_bf16

#endif
