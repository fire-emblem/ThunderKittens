#include "testing_flags.cuh"

#ifdef TEST_C500_GEMM_RAW_VECTOR_GEMM_PROBE

#include "testing_commons.cuh"
#include "../../../kernels/gemm/common.cuh"

#include "arch/c500/gemm/bf16_contracts.cuh"
#include "arch/c500/gemm/bf16_epilogue.cuh"
#include "arch/c500/gemm/bf16_mainloop.cuh"
#include "arch/c500/gemm/bf16_stage_primitives.cuh"

namespace c500::mma::raw_vector_gemm_probe {

namespace {

constexpr int kM = 128;
constexpr int kN = 128;
constexpr int kK = 32;

using contracts = kittens::arch::c500::gemm::bf16_contracts;
using stage_ring = kittens::arch::c500::gemm::bf16_stage_ring;
using atom = kittens::arch::c500::mma_bf16_16x16x16_fp32;
using frag_c = kittens::arch::c500::fragment_c<atom>;
using reg_tile_c = kittens::arch::c500::gemm::bf16_reg_tile_c;
using shared_tile_c = kittens::arch::c500::gemm::bf16_shared_tile_c;
using raw_frag_a = kittens::arch::c500::fragment_a<atom>;
using raw_frag_b = kittens::arch::c500::fragment_b<atom>;

template<int M, int K>
using a_gl = kittens::gl<kittens::bf16, 1, 1, M, K>;
template<int K, int N>
using b_gl = kittens::gl<kittens::bf16, 1, 1, K, N>;
template<int M, int N>
using c_gl = kittens::gl<kittens::bf16, 1, 1, M, N, shared_tile_c>;

struct gemm_globals {
    a_gl<kM, kK> a;
    b_gl<kK, kN> b;
    c_gl<kM, kN> c;
};

constexpr int kAtomTiles = 4;
constexpr int kAtomK = 2;
constexpr int kCandidateMasks = kAtomTiles * kAtomK;

__device__ inline bool same_frag(const raw_frag_a &lhs, const raw_frag_a &rhs) {
    return lhs.reg[0] == rhs.reg[0] && lhs.reg[1] == rhs.reg[1];
}

__device__ inline bool same_frag(const raw_frag_b &lhs, const raw_frag_b &rhs) {
    return lhs.reg[0] == rhs.reg[0] && lhs.reg[1] == rhs.reg[1];
}

__device__ inline void zero_accumulators(frag_c (&acc)[4][4]) {
#pragma unroll
    for (int m = 0; m < 4; ++m) {
#pragma unroll
        for (int n = 0; n < 4; ++n) {
#pragma unroll
            for (int r = 0; r < atom::c_registers; ++r) {
                acc[m][n].reg[r] = 0.0f;
            }
        }
    }
}

__device__ inline void export_accumulators(reg_tile_c &dst,
                                           const frag_c (&acc)[4][4]) {
#pragma unroll
    for (int m = 0; m < 4; ++m) {
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            kittens::arch::c500::gemm::store_epilogue(dst, acc[m][n], m, n);
        }
    }
}

__global__ __launch_bounds__(contracts::kThreads)
void raw_vector_probe_kernel(const __grid_constant__ gemm_globals g,
                             uint32_t *a_match_masks,
                             uint32_t *b_match_masks) {
    __shared__ stage_ring ring;

    const int workerid = kittens::warpid();
    const int row_worker = workerid / contracts::kWaveN;
    const int col_worker = workerid % contracts::kWaveN;
    const int load_group = workerid / 2;
    const uint32_t ring_base = static_cast<uint32_t>(__cvta_generic_to_shared(&ring.bytes[0]));

    auto tok = kittens::arch::c500::gemm::issue_ab_stage_async(ring, g.a, g.b, 0, load_group, 0, 0, 0);
    kittens::arch::c500::wait(tok);
    __syncthreads();

    const uint32_t a_shared = ring_base + kittens::arch::c500::gemm::bf16_128x128x128_stage_layout::a_group_offset(0, row_worker);
    const uint32_t b_shared = ring_base + kittens::arch::c500::gemm::bf16_128x128x128_stage_layout::b_group_offset(0, col_worker);

    frag_c acc[4][4];
    reg_tile_c out;
    zero_accumulators(acc);

    auto a_words0 = kittens::arch::c500::gemm::load_stage_a_words(ring, 0, threadIdx.x, 0);
    auto a_words1 = kittens::arch::c500::gemm::load_stage_a_words(ring, 0, threadIdx.x, 1);
    auto a_words2 = kittens::arch::c500::gemm::load_stage_a_words(ring, 0, threadIdx.x, 2);
    auto a_words3 = kittens::arch::c500::gemm::load_stage_a_words(ring, 0, threadIdx.x, 3);
    auto b_words0 = kittens::arch::c500::gemm::load_stage_b_words(ring, 0, threadIdx.x, 0);
    auto b_words1 = kittens::arch::c500::gemm::load_stage_b_words(ring, 0, threadIdx.x, 1);
    auto b_words2 = kittens::arch::c500::gemm::load_stage_b_words(ring, 0, threadIdx.x, 2);
    auto b_words3 = kittens::arch::c500::gemm::load_stage_b_words(ring, 0, threadIdx.x, 3);

    const auto a_frag00 = kittens::arch::c500::gemm::make_a_fragment(a_words0, 0);
    const auto a_frag01 = kittens::arch::c500::gemm::make_a_fragment(a_words0, 1);
    const auto a_frag10 = kittens::arch::c500::gemm::make_a_fragment(a_words1, 0);
    const auto a_frag11 = kittens::arch::c500::gemm::make_a_fragment(a_words1, 1);
    const auto a_frag20 = kittens::arch::c500::gemm::make_a_fragment(a_words2, 0);
    const auto a_frag21 = kittens::arch::c500::gemm::make_a_fragment(a_words2, 1);
    const auto a_frag30 = kittens::arch::c500::gemm::make_a_fragment(a_words3, 0);
    const auto a_frag31 = kittens::arch::c500::gemm::make_a_fragment(a_words3, 1);

    const auto b_frag00 = kittens::arch::c500::gemm::make_b_fragment(b_words0, 0);
    const auto b_frag01 = kittens::arch::c500::gemm::make_b_fragment(b_words0, 1);
    const auto b_frag10 = kittens::arch::c500::gemm::make_b_fragment(b_words1, 0);
    const auto b_frag11 = kittens::arch::c500::gemm::make_b_fragment(b_words1, 1);
    const auto b_frag20 = kittens::arch::c500::gemm::make_b_fragment(b_words2, 0);
    const auto b_frag21 = kittens::arch::c500::gemm::make_b_fragment(b_words2, 1);
    const auto b_frag30 = kittens::arch::c500::gemm::make_b_fragment(b_words3, 0);
    const auto b_frag31 = kittens::arch::c500::gemm::make_b_fragment(b_words3, 1);

    const raw_frag_a a0[2] = {a_frag00, a_frag01};
    const raw_frag_a a1[2] = {a_frag10, a_frag11};
    const raw_frag_a a2[2] = {a_frag20, a_frag21};
    const raw_frag_a a3[2] = {a_frag30, a_frag31};
    const raw_frag_b b0[2] = {b_frag00, b_frag01};
    const raw_frag_b b1[2] = {b_frag10, b_frag11};
    const raw_frag_b b2[2] = {b_frag20, b_frag21};
    const raw_frag_b b3[2] = {b_frag30, b_frag31};

    const raw_frag_a *a_rows[4] = {a0, a1, a2, a3};
    const raw_frag_b *b_cols[4] = {b0, b1, b2, b3};

#pragma unroll
    for (int m = 0; m < kAtomTiles; ++m) {
#pragma unroll
        for (int kk = 0; kk < kAtomK; ++kk) {
            const raw_frag_a scalar_a =
                kittens::arch::c500::gemm::raw_stage_load_a_fragment(ring, 0, row_worker, m, kk * atom::K);

            uint32_t mask = 0;
#pragma unroll
            for (int cand = 0; cand < kAtomTiles; ++cand) {
#pragma unroll
                for (int pair = 0; pair < kAtomK; ++pair) {
                    if (same_frag(scalar_a, a_rows[cand][pair])) {
                        mask |= (1u << (cand * kAtomK + pair));
                    }
                }
            }
            a_match_masks[(threadIdx.x * kAtomTiles + m) * kAtomK + kk] = mask;
        }
    }

#pragma unroll
    for (int n = 0; n < kAtomTiles; ++n) {
#pragma unroll
        for (int kk = 0; kk < kAtomK; ++kk) {
            const raw_frag_b scalar_b =
                kittens::arch::c500::gemm::raw_stage_load_b_fragment(ring, 0, col_worker, n, kk * atom::K);

            uint32_t mask = 0;
#pragma unroll
            for (int cand = 0; cand < kAtomTiles; ++cand) {
#pragma unroll
                for (int pair = 0; pair < kAtomK; ++pair) {
                    if (same_frag(scalar_b, b_cols[cand][pair])) {
                        mask |= (1u << (cand * kAtomK + pair));
                    }
                }
            }
            b_match_masks[(threadIdx.x * kAtomTiles + n) * kAtomK + kk] = mask;
        }
    }

#pragma unroll
    for (int m = 0; m < 4; ++m) {
#pragma unroll
        for (int n = 0; n < 4; ++n) {
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                frag_c next;
                kittens::arch::c500::mma<atom>(next, a_rows[m][kk], b_cols[n][kk], acc[m][n]);
                acc[m][n] = next;
            }
        }
    }

    export_accumulators(out, acc);
    kittens::warp::store(g.c, out, {0, 0, row_worker, col_worker});
}

bool run_probe(test_data &results) {
    test_info info{"c500_gemm_bf16_raw_vector_mma_smoke", test_result::FAILED};

    kittens::bf16 *d_a = nullptr;
    kittens::bf16 *d_b = nullptr;
    kittens::bf16 *d_c = nullptr;
    kittens::bf16 *d_ref = nullptr;
    uint32_t *d_a_match = nullptr;
    uint32_t *d_b_match = nullptr;

    cudaMalloc(&d_a, kM * kK * sizeof(kittens::bf16));
    cudaMalloc(&d_b, kK * kN * sizeof(kittens::bf16));
    cudaMalloc(&d_c, kM * kN * sizeof(kittens::bf16));
    cudaMalloc(&d_ref, kM * kN * sizeof(kittens::bf16));
    cudaMalloc(&d_a_match, contracts::kThreads * kCandidateMasks * sizeof(uint32_t));
    cudaMalloc(&d_b_match, contracts::kThreads * kCandidateMasks * sizeof(uint32_t));
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
    cudaDeviceSynchronize();
    CudaCheckError();

    gemm_globals g{
        a_gl<kM, kK>{d_a, nullptr, nullptr, nullptr, nullptr},
        b_gl<kK, kN>{d_b, nullptr, nullptr, nullptr, nullptr},
        c_gl<kM, kN>{d_c, nullptr, nullptr, nullptr, nullptr}
    };

    raw_vector_probe_kernel<<<1, contracts::kThreads>>>(g, d_a_match, d_b_match);
    cudaDeviceSynchronize();
    CudaCheckError();

    std::vector<float> out(kM * kN, 0.0f);
    std::vector<float> ref_out(kM * kN, 0.0f);
    std::vector<uint32_t> h_a_match(contracts::kThreads * kCandidateMasks, 0);
    std::vector<uint32_t> h_b_match(contracts::kThreads * kCandidateMasks, 0);
    std::vector<kittens::bf16> h_out(kM * kN);
    std::vector<kittens::bf16> h_ref(kM * kN);
    cudaMemcpy(h_a_match.data(), d_a_match, h_a_match.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b_match.data(), d_b_match, h_b_match.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    CudaCheckError();
    cudaMemcpy(h_out.data(), d_c, kM * kN * sizeof(kittens::bf16), cudaMemcpyDeviceToHost);
    CudaCheckError();
    cudaMemcpy(h_ref.data(), d_ref, kM * kN * sizeof(kittens::bf16), cudaMemcpyDeviceToHost);
    CudaCheckError();
    for (int i = 0; i < kM * kN; ++i) {
        out[i] = __bfloat162float(h_out[i]);
        ref_out[i] = __bfloat162float(h_ref[i]);
    }

    bool good = true;
    int first_i = -1;
    for (int i = 0; i < kM * kN; ++i) {
        if (fabsf(ref_out[i] - out[i]) > 0.02f) {
            good = false;
            first_i = i;
            break;
        }
    }

    std::cout << "test `" << info.label << "`";
    if (good) {
        std::cout << " -- PASSED" << std::endl;
        info.result = test_result::PASSED;
    } else {
        bool printed_a = false;
        bool printed_b = false;
        for (int tid = 0; tid < contracts::kThreads && (!printed_a || !printed_b); ++tid) {
            for (int tile = 0; tile < kAtomTiles && (!printed_a || !printed_b); ++tile) {
                for (int kk = 0; kk < kAtomK && (!printed_a || !printed_b); ++kk) {
                    const int idx = (tid * kAtomTiles + tile) * kAtomK + kk;
                    if (!printed_a && h_a_match[idx] == 0) {
                        std::cout << "first A fragment with no raw-vector match tid=" << tid
                                  << " tile_m=" << tile
                                  << " mma_k=" << kk
                                  << std::endl;
                        printed_a = true;
                    }
                    if (!printed_b && h_b_match[idx] == 0) {
                        std::cout << "first B fragment with no raw-vector match tid=" << tid
                                  << " tile_n=" << tile
                                  << " mma_k=" << kk
                                  << std::endl;
                        printed_b = true;
                    }
                }
            }
        }
        const int row = first_i / kN;
        const int col = first_i % kN;
        std::cout << " ----- ALERT! FAILED test `" << info.label << "` -----" << std::endl;
        std::cout << "first mismatch row=" << row
                  << " col=" << col
                  << " ref=" << ref_out[first_i]
                  << " got=" << out[first_i]
                  << std::endl;
    }

    results.push_back(info);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_ref);
    cudaFree(d_a_match);
    cudaFree(d_b_match);
    CudaCheckError();
    return info.result == test_result::PASSED;
}

} // namespace

void tests(test_data &results) {
    std::cout << " ----- Starting ops/c500/mma/raw_vector_gemm_probe tests! -----\n" << std::endl;
    run_probe(results);
}

} // namespace c500::mma::raw_vector_gemm_probe

#endif
