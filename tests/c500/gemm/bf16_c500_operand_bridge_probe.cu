#include "testing_flags.cuh"

#ifdef TEST_C500_GEMM_OPERAND_BRIDGE_PROBE

#include <vector>

#include "testing_commons.cuh"

#include "arch/c500/gemm/bf16_mainloop.cuh"
#include "arch/c500/gemm/bf16_operand_stage.cuh"
#include "arch/c500/gemm/bf16_stage_primitives.cuh"

namespace c500::mma::operand_bridge_probe {

namespace {

using namespace kittens;
using contracts = kittens::arch::c500::gemm::bf16_contracts;
using atom = kittens::arch::c500::mma_bf16_16x16x16_fp32;
using raw_stage_ring = kittens::arch::c500::gemm::bf16_stage_ring;
using cta_operand_ring = kittens::arch::c500::gemm::bf16_operand_cta_stage_ring_1;
using frag_a = kittens::arch::c500::fragment_a<atom>;
using frag_b = kittens::arch::c500::fragment_b<atom>;

constexpr int kM = 128;
constexpr int kN = 128;
constexpr int kK = 32;
constexpr int kAtomTiles = 4;
constexpr int kKGroups = 4;
constexpr int kHalfK = 2;
constexpr int kThreads = contracts::kThreads;
constexpr int kGroupCount = 2;
constexpr int kWordCount = kGroupCount * kAtomTiles * kKGroups * kittens::WARP_THREADS;

template<int M, int K>
using a_gl = kittens::gl<kittens::bf16, 1, 1, M, K>;
template<int K, int N>
using b_gl = kittens::gl<kittens::bf16, 1, 1, K, N>;

__device__ inline bool same_frag(const frag_a &lhs, const frag_a &rhs) {
    return lhs.reg[0] == rhs.reg[0] && lhs.reg[1] == rhs.reg[1];
}

__device__ inline bool same_frag(const frag_b &lhs, const frag_b &rhs) {
    return lhs.reg[0] == rhs.reg[0] && lhs.reg[1] == rhs.reg[1];
}

__host__ __device__ inline int word_index(int group, int tile, int kg, int lane) {
    return (((group * kAtomTiles) + tile) * kKGroups + kg) * kittens::WARP_THREADS + lane;
}

__global__ void pack_operand_words_kernel(const __grid_constant__ a_gl<kM, kK> a,
                                          const __grid_constant__ b_gl<kK, kN> b,
                                          kittens::arch::c500::gemm::bf16_operand_vec *a_words_out,
                                          kittens::arch::c500::gemm::bf16_operand_vec *b_words_out) {
    __shared__ raw_stage_ring raw_ring;

    const int warp = kittens::warpid();
    const int row_worker = warp / contracts::kWaveN;
    const int col_worker = warp % contracts::kWaveN;
    const int load_group = warp / 2;
    const int lane = kittens::laneid();

    auto tok = kittens::arch::c500::gemm::issue_ab_stage_async(raw_ring, a, b, 0, load_group, 0, 0, 0);
    kittens::arch::c500::wait(tok);
    __syncthreads();

    if (col_worker == 0) {
#pragma unroll
        for (int m = 0; m < kAtomTiles; ++m) {
#pragma unroll
            for (int kg = 0; kg < kKGroups; ++kg) {
                const frag_a lo =
                    kittens::arch::c500::gemm::raw_stage_load_a_fragment(raw_ring, 0, row_worker, m, kg * 8 + 0);
                const frag_a hi =
                    kittens::arch::c500::gemm::raw_stage_load_a_fragment(raw_ring, 0, row_worker, m, kg * 8 + 4);
                a_words_out[word_index(row_worker, m, kg, lane)] =
                    kittens::arch::c500::gemm::pack_a_operand_words(lo, hi);
            }
        }
    }

    if (row_worker == 0) {
#pragma unroll
        for (int n = 0; n < kAtomTiles; ++n) {
#pragma unroll
            for (int kg = 0; kg < kKGroups; ++kg) {
                const frag_b lo =
                    kittens::arch::c500::gemm::raw_stage_load_b_fragment(raw_ring, 0, col_worker, n, kg * 8 + 0);
                const frag_b hi =
                    kittens::arch::c500::gemm::raw_stage_load_b_fragment(raw_ring, 0, col_worker, n, kg * 8 + 4);
                b_words_out[word_index(col_worker, n, kg, lane)] =
                    kittens::arch::c500::gemm::pack_b_operand_words(lo, hi);
            }
        }
    }
}

__global__ void operand_bridge_verify_kernel(const kittens::arch::c500::gemm::bf16_operand_vec *a_words_in,
                                             const kittens::arch::c500::gemm::bf16_operand_vec *b_words_in,
                                             uint32_t *a_ok,
                                             uint32_t *b_ok) {
    __shared__ cta_operand_ring operand_ring;

    const int warp = kittens::warpid();
    const int row_worker = warp / contracts::kWaveN;
    const int col_worker = warp % contracts::kWaveN;
    const int lane = kittens::laneid();

    if (col_worker == 0) {
#pragma unroll
        for (int m = 0; m < kAtomTiles; ++m) {
#pragma unroll
            for (int kg = 0; kg < kKGroups; ++kg) {
                kittens::arch::c500::gemm::store_cta_a_operand_words(
                    operand_ring, 0, row_worker, m, kg, lane, a_words_in[word_index(row_worker, m, kg, lane)]);
            }
        }
    }

    if (row_worker == 0) {
#pragma unroll
        for (int n = 0; n < kAtomTiles; ++n) {
#pragma unroll
            for (int kg = 0; kg < kKGroups; ++kg) {
                kittens::arch::c500::gemm::store_cta_b_operand_words(
                    operand_ring, 0, col_worker, n, kg, lane, b_words_in[word_index(col_worker, n, kg, lane)]);
            }
        }
    }
    __syncthreads();

#pragma unroll
    for (int m = 0; m < kAtomTiles; ++m) {
#pragma unroll
        for (int kg = 0; kg < kKGroups; ++kg) {
#pragma unroll
            for (int half_k = 0; half_k < kHalfK; ++half_k) {
                const auto ref_words = a_words_in[word_index(row_worker, m, kg, lane)];
                const auto ref = kittens::arch::c500::gemm::make_a_operand_fragment(ref_words, half_k);
                const auto words =
                    kittens::arch::c500::gemm::load_cta_a_operand_words(operand_ring, 0, row_worker, m, kg, lane);
                const auto got = kittens::arch::c500::gemm::make_a_operand_fragment(words, half_k);
                const int idx = (((threadIdx.x * kAtomTiles) + m) * kKGroups + kg) * kHalfK + half_k;
                a_ok[idx] = same_frag(ref, got) ? 1u : 0u;
            }
        }
    }

#pragma unroll
    for (int n = 0; n < kAtomTiles; ++n) {
#pragma unroll
        for (int kg = 0; kg < kKGroups; ++kg) {
#pragma unroll
            for (int half_k = 0; half_k < kHalfK; ++half_k) {
                const auto ref_words = b_words_in[word_index(col_worker, n, kg, lane)];
                const auto ref = kittens::arch::c500::gemm::make_b_operand_fragment(ref_words, half_k);
                const auto words =
                    kittens::arch::c500::gemm::load_cta_b_operand_words(operand_ring, 0, col_worker, n, kg, lane);
                const auto got = kittens::arch::c500::gemm::make_b_operand_fragment(words, half_k);
                const int idx = (((threadIdx.x * kAtomTiles) + n) * kKGroups + kg) * kHalfK + half_k;
                b_ok[idx] = same_frag(ref, got) ? 1u : 0u;
            }
        }
    }
}

bool run_probe(test_data &results) {
    test_info info{"c500_gemm_bf16_operand_bridge_contract", test_result::FAILED};

    std::vector<bf16> h_a(kM * kK);
    std::vector<bf16> h_b(kK * kN);
    for (int i = 0; i < kM * kK; ++i) h_a[i] = __float2bfloat16(static_cast<float>(i + 1));
    for (int i = 0; i < kK * kN; ++i) h_b[i] = __float2bfloat16(static_cast<float>(10000 + i + 1));

    bf16 *d_a = nullptr;
    bf16 *d_b = nullptr;
    uint32_t *d_a_ok = nullptr;
    uint32_t *d_b_ok = nullptr;
    kittens::arch::c500::gemm::bf16_operand_vec *d_a_words = nullptr;
    kittens::arch::c500::gemm::bf16_operand_vec *d_b_words = nullptr;
    const size_t count = static_cast<size_t>(kThreads) * kAtomTiles * kKGroups * kHalfK;
    std::vector<uint32_t> h_a_ok(count, 0);
    std::vector<uint32_t> h_b_ok(count, 0);

    cudaMalloc(&d_a, h_a.size() * sizeof(bf16));
    cudaMalloc(&d_b, h_b.size() * sizeof(bf16));
    cudaMalloc(&d_a_ok, count * sizeof(uint32_t));
    cudaMalloc(&d_b_ok, count * sizeof(uint32_t));
    cudaMalloc(&d_a_words, kWordCount * sizeof(kittens::arch::c500::gemm::bf16_operand_vec));
    cudaMalloc(&d_b_words, kWordCount * sizeof(kittens::arch::c500::gemm::bf16_operand_vec));
    CudaCheckError();

    cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(bf16), cudaMemcpyHostToDevice);
    CudaCheckError();

    pack_operand_words_kernel<<<1, kThreads>>>(a_gl<kM, kK>{d_a, nullptr, nullptr, nullptr, nullptr},
                                               b_gl<kK, kN>{d_b, nullptr, nullptr, nullptr, nullptr},
                                               d_a_words,
                                               d_b_words);
    CudaCheckError();

    operand_bridge_verify_kernel<<<1, kThreads>>>(d_a_words, d_b_words, d_a_ok, d_b_ok);
    CudaCheckError();

    cudaMemcpy(h_a_ok.data(), d_a_ok, count * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b_ok.data(), d_b_ok, count * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    CudaCheckError();

    bool good = true;
    for (size_t i = 0; i < count; ++i) {
        if (h_a_ok[i] == 0 || h_b_ok[i] == 0) {
            good = false;
            const int tid = i / (kAtomTiles * kKGroups * kHalfK);
            const int rem0 = i % (kAtomTiles * kKGroups * kHalfK);
            const int tile = rem0 / (kKGroups * kHalfK);
            const int rem1 = rem0 % (kKGroups * kHalfK);
            const int kg = rem1 / kHalfK;
            const int half_k = rem1 % kHalfK;
            std::cout << "first mismatch tid=" << tid
                      << " tile=" << tile
                      << " kg=" << kg
                      << " half=" << half_k
                      << " a_ok=" << h_a_ok[i]
                      << " b_ok=" << h_b_ok[i]
                      << std::endl;
            break;
        }
    }

    std::cout << "test `" << info.label << "`";
    if (good) {
        std::cout << " -- PASSED" << std::endl;
        info.result = test_result::PASSED;
    } else {
        std::cout << " ----- ALERT! FAILED test `" << info.label << "` -----" << std::endl;
    }

    results.push_back(info);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_a_ok);
    cudaFree(d_b_ok);
    cudaFree(d_a_words);
    cudaFree(d_b_words);
    CudaCheckError();
    return good;
}

} // namespace

void tests(test_data &results) {
    std::cout << " ----- Starting ops/c500/mma/operand_bridge_probe tests! -----\n" << std::endl;
    run_probe(results);
}

} // namespace c500::mma::operand_bridge_probe

#endif
