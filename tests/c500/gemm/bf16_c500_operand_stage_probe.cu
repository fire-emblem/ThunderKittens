#include "testing_flags.cuh"

#ifdef TEST_C500_GEMM_OPERAND_STAGE_PROBE

#include <vector>

#include "testing_commons.cuh"

#include "arch/c500/gemm/bf16_mainloop.cuh"
#include "arch/c500/gemm/bf16_operand_stage.cuh"
#include "arch/c500/gemm/bf16_stage_primitives.cuh"

namespace c500::mma::operand_stage_probe {

namespace {

using namespace kittens;
using contracts = kittens::arch::c500::gemm::bf16_contracts;
using atom = kittens::arch::c500::mma_bf16_16x16x16_fp32;
using raw_stage_ring = kittens::arch::c500::gemm::bf16_stage_ring;
using frag_a = kittens::arch::c500::fragment_a<atom>;
using frag_b = kittens::arch::c500::fragment_b<atom>;
using frag_c = kittens::arch::c500::fragment_c<atom>;

constexpr int kM = 128;
constexpr int kN = 128;
constexpr int kK = 32;
constexpr int kAtomTiles = 4;
constexpr int kKGroups = 4;
constexpr int kHalfK = 2;
constexpr int kThreads = contracts::kThreads;

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

__device__ inline bool same_words(const kittens::arch::c500::gemm::bf16_operand_vec &lhs,
                                  const kittens::arch::c500::gemm::bf16_operand_vec &rhs) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (lhs[i] != rhs[i]) return false;
    }
    return true;
}

__global__ void operand_stage_probe_kernel(const __grid_constant__ a_gl<kM, kK> a,
                                           const __grid_constant__ b_gl<kK, kN> b,
                                           uint32_t *a_ok,
                                           uint32_t *b_ok) {
    __shared__ raw_stage_ring raw_ring;

    const int warp = kittens::warpid();
    const int row_worker = warp / contracts::kWaveN;
    const int col_worker = warp % contracts::kWaveN;
    const int load_group = warp / 2;

    auto tok = kittens::arch::c500::gemm::issue_ab_stage_async(raw_ring, a, b, 0, load_group, 0, 0, 0);
    kittens::arch::c500::wait(tok);
    __syncthreads();

#pragma unroll
    for (int m = 0; m < kAtomTiles; ++m) {
#pragma unroll
        for (int kg = 0; kg < kKGroups; ++kg) {
            const frag_a ref_lo =
                kittens::arch::c500::gemm::raw_stage_load_a_fragment(raw_ring, 0, row_worker, m, kg * 8 + 0);
            const frag_a ref_hi =
                kittens::arch::c500::gemm::raw_stage_load_a_fragment(raw_ring, 0, row_worker, m, kg * 8 + 4);
            const auto words = kittens::arch::c500::gemm::pack_a_operand_words(ref_lo, ref_hi);
#pragma unroll
            for (int half_k = 0; half_k < kHalfK; ++half_k) {
                const frag_a ref =
                    kittens::arch::c500::gemm::raw_stage_load_a_fragment(raw_ring, 0, row_worker, m, kg * 8 + half_k * 4);
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
            const frag_b ref_lo =
                kittens::arch::c500::gemm::raw_stage_load_b_fragment(raw_ring, 0, col_worker, n, kg * 8 + 0);
            const frag_b ref_hi =
                kittens::arch::c500::gemm::raw_stage_load_b_fragment(raw_ring, 0, col_worker, n, kg * 8 + 4);
            const auto words = kittens::arch::c500::gemm::pack_b_operand_words(ref_lo, ref_hi);
#pragma unroll
            for (int half_k = 0; half_k < kHalfK; ++half_k) {
                const frag_b ref =
                    kittens::arch::c500::gemm::raw_stage_load_b_fragment(raw_ring, 0, col_worker, n, kg * 8 + half_k * 4);
                const auto got = kittens::arch::c500::gemm::make_b_operand_fragment(words, half_k);
                const int idx = (((threadIdx.x * kAtomTiles) + n) * kKGroups + kg) * kHalfK + half_k;
                b_ok[idx] = same_frag(ref, got) ? 1u : 0u;
            }
        }
    }
}

__global__ void raw_stage_native_bridge_probe_kernel(const __grid_constant__ a_gl<kM, kK> a,
                                                     const __grid_constant__ b_gl<kK, kN> b,
                                                     uint32_t *a_ok,
                                                     uint32_t *b_ok) {
    __shared__ raw_stage_ring raw_ring;

    const int warp = kittens::warpid();
    const int row_worker = warp / contracts::kWaveN;
    const int col_worker = warp % contracts::kWaveN;
    const int load_group = warp / 2;
    const int lane = kittens::laneid();

    auto tok = kittens::arch::c500::gemm::issue_ab_stage_async(raw_ring, a, b, 0, load_group, 0, 0, 0);
    kittens::arch::c500::wait(tok);
    __syncthreads();

#pragma unroll
    for (int m = 0; m < kAtomTiles; ++m) {
#pragma unroll
        for (int kg = 0; kg < kKGroups; ++kg) {
            const auto got =
                kittens::arch::c500::gemm::bridge_raw_stage_a_to_operand(raw_ring, 0, row_worker, m, kg, lane);
            const frag_a ref_lo =
                kittens::arch::c500::gemm::raw_stage_load_a_fragment(raw_ring, 0, row_worker, m, kg * 8 + 0);
            const frag_a ref_hi =
                kittens::arch::c500::gemm::raw_stage_load_a_fragment(raw_ring, 0, row_worker, m, kg * 8 + 4);
            const auto ref = kittens::arch::c500::gemm::pack_a_operand_words(ref_lo, ref_hi);
            a_ok[(threadIdx.x * kAtomTiles + m) * kKGroups + kg] = same_words(got, ref) ? 1u : 0u;
        }
    }

#pragma unroll
    for (int n = 0; n < kAtomTiles; ++n) {
#pragma unroll
        for (int kg = 0; kg < kKGroups; ++kg) {
            const auto got =
                kittens::arch::c500::gemm::bridge_raw_stage_b_to_operand(raw_ring, 0, col_worker, n, kg, lane);
            const frag_b ref_lo =
                kittens::arch::c500::gemm::raw_stage_load_b_fragment(raw_ring, 0, col_worker, n, kg * 8 + 0);
            const frag_b ref_hi =
                kittens::arch::c500::gemm::raw_stage_load_b_fragment(raw_ring, 0, col_worker, n, kg * 8 + 4);
            const auto ref = kittens::arch::c500::gemm::pack_b_operand_words(ref_lo, ref_hi);
            b_ok[(threadIdx.x * kAtomTiles + n) * kKGroups + kg] = same_words(got, ref) ? 1u : 0u;
        }
    }
}

__global__ void raw_stage_native_mma_probe_kernel(const __grid_constant__ a_gl<kM, kK> a,
                                                  const __grid_constant__ b_gl<kK, kN> b,
                                                  uint32_t *ok_out) {
    __shared__ raw_stage_ring raw_ring;

    const int warp = kittens::warpid();
    const int row_worker = warp / contracts::kWaveN;
    const int col_worker = warp % contracts::kWaveN;
    const int load_group = warp / 2;

    auto tok = kittens::arch::c500::gemm::issue_ab_stage_async(raw_ring, a, b, 0, load_group, 0, 0, 0);
    kittens::arch::c500::wait(tok);
    __syncthreads();

    bool good = true;
#pragma unroll
    for (int m = 0; m < kAtomTiles && good; ++m) {
#pragma unroll
        for (int n = 0; n < kAtomTiles && good; ++n) {
            frag_c ref_acc{};
            frag_c got_acc{};
#pragma unroll
            for (int i = 0; i < atom::c_registers; ++i) {
                ref_acc.reg[i] = 0.0f;
                got_acc.reg[i] = 0.0f;
            }
#pragma unroll
            for (int kg = 0; kg < kKGroups; ++kg) {
                const frag_a ref_a_lo =
                    kittens::arch::c500::gemm::raw_stage_load_a_fragment(raw_ring, 0, row_worker, m, kg * 8 + 0);
                const frag_a ref_a_hi =
                    kittens::arch::c500::gemm::raw_stage_load_a_fragment(raw_ring, 0, row_worker, m, kg * 8 + 4);
                const frag_b ref_b_lo =
                    kittens::arch::c500::gemm::raw_stage_load_b_fragment(raw_ring, 0, col_worker, n, kg * 8 + 0);
                const frag_b ref_b_hi =
                    kittens::arch::c500::gemm::raw_stage_load_b_fragment(raw_ring, 0, col_worker, n, kg * 8 + 4);
                frag_c next_ref{};
                kittens::arch::c500::mma<atom>(next_ref, ref_a_lo, ref_b_lo, ref_acc);
                ref_acc = next_ref;
                kittens::arch::c500::mma<atom>(next_ref, ref_a_hi, ref_b_hi, ref_acc);
                ref_acc = next_ref;
            }

            auto next_got = kittens::arch::c500::gemm::mma_raw_stage_bridge(raw_ring, 0, row_worker, col_worker, m, n, got_acc);

#pragma unroll
            for (int i = 0; i < atom::c_registers; ++i) {
                if (ref_acc.reg[i] != next_got.reg[i]) {
                    good = false;
                    break;
                }
            }
        }
    }

    ok_out[threadIdx.x] = good ? 1u : 0u;
}

__global__ void raw_stage_native_tile_probe_kernel(const __grid_constant__ a_gl<kM, kK> a,
                                                   const __grid_constant__ b_gl<kK, kN> b,
                                                   uint32_t *ok_out) {
    __shared__ raw_stage_ring raw_ring;

    const int warp = kittens::warpid();
    const int row_worker = warp / contracts::kWaveN;
    const int col_worker = warp % contracts::kWaveN;
    const int load_group = warp / 2;

    auto tok = kittens::arch::c500::gemm::issue_ab_stage_async(raw_ring, a, b, 0, load_group, 0, 0, 0);
    kittens::arch::c500::wait(tok);
    __syncthreads();

    frag_c ref_acc[kAtomTiles][kAtomTiles];
    frag_c got_acc[kAtomTiles][kAtomTiles];
#pragma unroll
    for (int m = 0; m < kAtomTiles; ++m) {
#pragma unroll
        for (int n = 0; n < kAtomTiles; ++n) {
#pragma unroll
            for (int i = 0; i < atom::c_registers; ++i) {
                ref_acc[m][n].reg[i] = 0.0f;
                got_acc[m][n].reg[i] = 0.0f;
            }
        }
    }

#pragma unroll
    for (int m = 0; m < kAtomTiles; ++m) {
#pragma unroll
        for (int n = 0; n < kAtomTiles; ++n) {
#pragma unroll
            for (int kg = 0; kg < kKGroups; ++kg) {
                const frag_a ref_a_lo =
                    kittens::arch::c500::gemm::raw_stage_load_a_fragment(raw_ring, 0, row_worker, m, kg * 8 + 0);
                const frag_a ref_a_hi =
                    kittens::arch::c500::gemm::raw_stage_load_a_fragment(raw_ring, 0, row_worker, m, kg * 8 + 4);
                const frag_b ref_b_lo =
                    kittens::arch::c500::gemm::raw_stage_load_b_fragment(raw_ring, 0, col_worker, n, kg * 8 + 0);
                const frag_b ref_b_hi =
                    kittens::arch::c500::gemm::raw_stage_load_b_fragment(raw_ring, 0, col_worker, n, kg * 8 + 4);
                frag_c next_ref{};
                kittens::arch::c500::mma<atom>(next_ref, ref_a_lo, ref_b_lo, ref_acc[m][n]);
                ref_acc[m][n] = next_ref;
                kittens::arch::c500::mma<atom>(next_ref, ref_a_hi, ref_b_hi, ref_acc[m][n]);
                ref_acc[m][n] = next_ref;
            }
        }
    }

    kittens::arch::c500::gemm::mma_raw_stage_tile_bridge(raw_ring, 0, row_worker, col_worker, got_acc);

    bool good = true;
#pragma unroll
    for (int m = 0; m < kAtomTiles && good; ++m) {
#pragma unroll
        for (int n = 0; n < kAtomTiles && good; ++n) {
#pragma unroll
            for (int i = 0; i < atom::c_registers; ++i) {
                if (ref_acc[m][n].reg[i] != got_acc[m][n].reg[i]) {
                    good = false;
                    break;
                }
            }
        }
    }

    ok_out[threadIdx.x] = good ? 1u : 0u;
}

__global__ void raw_stage_native_aligned_tile_probe_kernel(const __grid_constant__ a_gl<kM, kK> a,
                                                           const __grid_constant__ b_gl<kK, kN> b,
                                                           uint32_t *ok_out) {
    __shared__ raw_stage_ring raw_ring;

    const int warp = kittens::warpid();
    const int row_worker = warp / contracts::kWaveN;
    const int col_worker = warp % contracts::kWaveN;
    const int load_group = warp / 2;

    auto tok = kittens::arch::c500::gemm::issue_ab_stage_async(raw_ring, a, b, 0, load_group, 0, 0, 0);
    kittens::arch::c500::wait(tok);
    __syncthreads();

    frag_c ref_acc[kAtomTiles][kAtomTiles];
    frag_c got_acc[kAtomTiles][kAtomTiles];
#pragma unroll
    for (int m = 0; m < kAtomTiles; ++m) {
#pragma unroll
        for (int n = 0; n < kAtomTiles; ++n) {
#pragma unroll
            for (int i = 0; i < atom::c_registers; ++i) {
                ref_acc[m][n].reg[i] = 0.0f;
                got_acc[m][n].reg[i] = 0.0f;
            }
        }
    }

    for (int m = 0; m < kAtomTiles; ++m) {
        for (int n = 0; n < kAtomTiles; ++n) {
            for (int mma_k = 0; mma_k < 2; ++mma_k) {
                const frag_a ref_a =
                    kittens::arch::c500::gemm::raw_stage_load_a_fragment(raw_ring, 0, row_worker, m, mma_k * 16);
                const frag_b ref_b =
                    kittens::arch::c500::gemm::raw_stage_load_b_fragment(raw_ring, 0, col_worker, n, mma_k * 16);
                frag_c next_ref{};
                kittens::arch::c500::mma<atom>(next_ref, ref_a, ref_b, ref_acc[m][n]);
                ref_acc[m][n] = next_ref;
            }
        }
    }
    kittens::arch::c500::gemm::mma_raw_stage_aligned_tile_bridge(raw_ring, 0, row_worker, col_worker, got_acc);

    bool good = true;
#pragma unroll
    for (int m = 0; m < kAtomTiles && good; ++m) {
#pragma unroll
        for (int n = 0; n < kAtomTiles && good; ++n) {
#pragma unroll
            for (int i = 0; i < atom::c_registers; ++i) {
                if (ref_acc[m][n].reg[i] != got_acc[m][n].reg[i]) {
                    good = false;
                    break;
                }
            }
        }
    }

    ok_out[threadIdx.x] = good ? 1u : 0u;
}

bool run_probe(test_data &results) {
    test_info info{"c500_gemm_bf16_operand_stage_contract", test_result::FAILED};

    std::vector<bf16> h_a(kM * kK);
    std::vector<bf16> h_b(kK * kN);
    for (int i = 0; i < kM * kK; ++i) h_a[i] = __float2bfloat16(static_cast<float>(i + 1));
    for (int i = 0; i < kK * kN; ++i) h_b[i] = __float2bfloat16(static_cast<float>(10000 + i + 1));

    bf16 *d_a = nullptr;
    bf16 *d_b = nullptr;
    uint32_t *d_a_ok = nullptr;
    uint32_t *d_b_ok = nullptr;
    const size_t count = static_cast<size_t>(kThreads) * kAtomTiles * kKGroups * kHalfK;
    std::vector<uint32_t> h_a_ok(count, 0);
    std::vector<uint32_t> h_b_ok(count, 0);

    cudaMalloc(&d_a, h_a.size() * sizeof(bf16));
    cudaMalloc(&d_b, h_b.size() * sizeof(bf16));
    cudaMalloc(&d_a_ok, count * sizeof(uint32_t));
    cudaMalloc(&d_b_ok, count * sizeof(uint32_t));
    CudaCheckError();

    cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(bf16), cudaMemcpyHostToDevice);
    CudaCheckError();

    operand_stage_probe_kernel<<<1, kThreads>>>(a_gl<kM, kK>{d_a, nullptr, nullptr, nullptr, nullptr},
                                                b_gl<kK, kN>{d_b, nullptr, nullptr, nullptr, nullptr},
                                                d_a_ok,
                                                d_b_ok);
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
    CudaCheckError();
    return good;
}

bool run_raw_stage_native_bridge_probe(test_data &results) {
    test_info info{"c500_gemm_bf16_raw_stage_native_bridge_contract", test_result::FAILED};

    std::vector<bf16> h_a(kM * kK);
    std::vector<bf16> h_b(kK * kN);
    for (int i = 0; i < kM * kK; ++i) h_a[i] = __float2bfloat16(static_cast<float>(i + 1));
    for (int i = 0; i < kK * kN; ++i) h_b[i] = __float2bfloat16(static_cast<float>(10000 + i + 1));

    bf16 *d_a = nullptr;
    bf16 *d_b = nullptr;
    uint32_t *d_a_ok = nullptr;
    uint32_t *d_b_ok = nullptr;
    const size_t count = static_cast<size_t>(kThreads) * kAtomTiles * kKGroups;
    std::vector<uint32_t> h_a_ok(count, 0);
    std::vector<uint32_t> h_b_ok(count, 0);

    cudaMalloc(&d_a, h_a.size() * sizeof(bf16));
    cudaMalloc(&d_b, h_b.size() * sizeof(bf16));
    cudaMalloc(&d_a_ok, count * sizeof(uint32_t));
    cudaMalloc(&d_b_ok, count * sizeof(uint32_t));
    CudaCheckError();

    cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(bf16), cudaMemcpyHostToDevice);
    CudaCheckError();

    raw_stage_native_bridge_probe_kernel<<<1, kThreads>>>(a_gl<kM, kK>{d_a, nullptr, nullptr, nullptr, nullptr},
                                                          b_gl<kK, kN>{d_b, nullptr, nullptr, nullptr, nullptr},
                                                          d_a_ok,
                                                          d_b_ok);
    CudaCheckError();

    cudaMemcpy(h_a_ok.data(), d_a_ok, count * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b_ok.data(), d_b_ok, count * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    CudaCheckError();

    bool good = true;
    for (size_t i = 0; i < count; ++i) {
        if (h_a_ok[i] == 0 || h_b_ok[i] == 0) {
            good = false;
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
    CudaCheckError();
    return good;
}

bool run_raw_stage_native_mma_probe(test_data &results) {
    test_info info{"c500_gemm_bf16_raw_stage_native_mma_contract", test_result::FAILED};

    std::vector<bf16> h_a(kM * kK);
    std::vector<bf16> h_b(kK * kN);
    for (int i = 0; i < kM * kK; ++i) h_a[i] = __float2bfloat16(static_cast<float>(i + 1));
    for (int i = 0; i < kK * kN; ++i) h_b[i] = __float2bfloat16(static_cast<float>(10000 + i + 1));

    bf16 *d_a = nullptr;
    bf16 *d_b = nullptr;
    uint32_t *d_ok = nullptr;
    std::vector<uint32_t> h_ok(kThreads, 0);

    cudaMalloc(&d_a, h_a.size() * sizeof(bf16));
    cudaMalloc(&d_b, h_b.size() * sizeof(bf16));
    cudaMalloc(&d_ok, h_ok.size() * sizeof(uint32_t));
    CudaCheckError();

    cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(bf16), cudaMemcpyHostToDevice);
    CudaCheckError();

    raw_stage_native_mma_probe_kernel<<<1, kThreads>>>(a_gl<kM, kK>{d_a, nullptr, nullptr, nullptr, nullptr},
                                                       b_gl<kK, kN>{d_b, nullptr, nullptr, nullptr, nullptr},
                                                       d_ok);
    CudaCheckError();
    cudaMemcpy(h_ok.data(), d_ok, h_ok.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    CudaCheckError();

    bool good = true;
    for (int tid = 0; tid < kThreads; ++tid) {
        if (h_ok[tid] == 0) {
            good = false;
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
    cudaFree(d_ok);
    CudaCheckError();
    return good;
}

bool run_raw_stage_native_tile_probe(test_data &results) {
    test_info info{"c500_gemm_bf16_raw_stage_native_tile_contract", test_result::FAILED};

    std::vector<bf16> h_a(kM * kK);
    std::vector<bf16> h_b(kK * kN);
    for (int i = 0; i < kM * kK; ++i) h_a[i] = __float2bfloat16(static_cast<float>(i + 1));
    for (int i = 0; i < kK * kN; ++i) h_b[i] = __float2bfloat16(static_cast<float>(10000 + i + 1));

    bf16 *d_a = nullptr;
    bf16 *d_b = nullptr;
    uint32_t *d_ok = nullptr;
    std::vector<uint32_t> h_ok(kThreads, 0);

    cudaMalloc(&d_a, h_a.size() * sizeof(bf16));
    cudaMalloc(&d_b, h_b.size() * sizeof(bf16));
    cudaMalloc(&d_ok, h_ok.size() * sizeof(uint32_t));
    CudaCheckError();

    cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(bf16), cudaMemcpyHostToDevice);
    CudaCheckError();

    raw_stage_native_tile_probe_kernel<<<1, kThreads>>>(a_gl<kM, kK>{d_a, nullptr, nullptr, nullptr, nullptr},
                                                        b_gl<kK, kN>{d_b, nullptr, nullptr, nullptr, nullptr},
                                                        d_ok);
    CudaCheckError();
    cudaMemcpy(h_ok.data(), d_ok, h_ok.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    CudaCheckError();

    bool good = true;
    for (int tid = 0; tid < kThreads; ++tid) {
        if (h_ok[tid] == 0) {
            good = false;
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
    cudaFree(d_ok);
    CudaCheckError();
    return good;
}

bool run_raw_stage_native_aligned_tile_probe(test_data &results) {
    test_info info{"c500_gemm_bf16_raw_stage_native_aligned_tile_contract", test_result::FAILED};

    std::vector<bf16> h_a(kM * kK);
    std::vector<bf16> h_b(kK * kN);
    for (int i = 0; i < kM * kK; ++i) h_a[i] = __float2bfloat16(static_cast<float>(i + 1));
    for (int i = 0; i < kK * kN; ++i) h_b[i] = __float2bfloat16(static_cast<float>(10000 + i + 1));

    bf16 *d_a = nullptr;
    bf16 *d_b = nullptr;
    uint32_t *d_ok = nullptr;
    std::vector<uint32_t> h_ok(kThreads, 0);

    cudaMalloc(&d_a, h_a.size() * sizeof(bf16));
    cudaMalloc(&d_b, h_b.size() * sizeof(bf16));
    cudaMalloc(&d_ok, h_ok.size() * sizeof(uint32_t));
    CudaCheckError();

    cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(bf16), cudaMemcpyHostToDevice);
    CudaCheckError();

    raw_stage_native_aligned_tile_probe_kernel<<<1, kThreads>>>(a_gl<kM, kK>{d_a, nullptr, nullptr, nullptr, nullptr},
                                                                b_gl<kK, kN>{d_b, nullptr, nullptr, nullptr, nullptr},
                                                                d_ok);
    CudaCheckError();
    cudaMemcpy(h_ok.data(), d_ok, h_ok.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    CudaCheckError();

    bool good = true;
    for (int tid = 0; tid < kThreads; ++tid) {
        if (h_ok[tid] == 0) {
            good = false;
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
    cudaFree(d_ok);
    CudaCheckError();
    return good;
}

} // namespace

void tests(test_data &results) {
    std::cout << " ----- Starting ops/c500/mma/operand_stage_probe tests! -----\n" << std::endl;
    run_probe(results);
    run_raw_stage_native_bridge_probe(results);
    run_raw_stage_native_mma_probe(results);
    run_raw_stage_native_tile_probe(results);
    run_raw_stage_native_aligned_tile_probe(results);
}

} // namespace c500::mma::operand_stage_probe

#endif
