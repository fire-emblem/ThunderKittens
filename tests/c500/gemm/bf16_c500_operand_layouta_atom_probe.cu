#include "testing_flags.cuh"

#ifdef TEST_C500_GEMM_OPERAND_LAYOUTA_ATOM_PROBE

#include <vector>

#include "testing_commons.cuh"

#include "../../../kernels/gemm/common.cuh"
#include "arch/c500/gemm/bf16_mainloop.cuh"
#include "arch/c500/gemm/bf16_operand_stage.cuh"
#include "arch/c500/gemm/bf16_stage_primitives.cuh"

namespace c500::mma::operand_layouta_atom_probe {

namespace {

using namespace kittens;
using contracts = kittens::arch::c500::gemm::bf16_contracts;
using atom = kittens::arch::c500::mma_bf16_16x16x16_fp32;
using operand_ring = kittens::arch::c500::gemm::bf16_operand_cta_stage_ring_1;
using frag_a = kittens::arch::c500::fragment_a<atom>;
using frag_b = kittens::arch::c500::fragment_b<atom>;
using frag_c = kittens::arch::c500::fragment_c<atom>;
using operand_vec = kittens::arch::c500::gemm::bf16_operand_vec;
using reg_tile_c = kittens::arch::c500::gemm::bf16_reg_tile_c;

constexpr int kM = 128;
constexpr int kN = 128;
constexpr int kK = 128;
constexpr int kAtomTiles = 4;
constexpr int kKGroups = 4;
constexpr int kHalfK = 2;
constexpr int kThreads = contracts::kThreads;
constexpr int kRegsC = atom::c_registers;

template<int M, int K>
using a_gl = kittens::gl<kittens::bf16, 1, 1, M, K>;
template<int K, int N>
using b_row_gl = kittens::gl<kittens::bf16, 1, 1, K, N>;
template<int N, int K>
using b_layouta_gl = kittens::gl<kittens::bf16, 1, 1, N, K>;

__device__ inline bool same_acc(const frag_c &lhs, const frag_c &rhs) {
#pragma unroll
    for (int i = 0; i < kRegsC; ++i) {
        if (lhs.reg[i] != rhs.reg[i]) return false;
    }
    return true;
}

__host__ __device__ inline uint32_t pack_pair(bf16 lo, bf16 hi) {
    const bf16_2 pair{lo, hi};
    return *reinterpret_cast<const uint32_t *>(&pair);
}

__device__ inline frag_a make_ref_a_fragment(const a_gl<kM, kK> &a,
                                             int row_group,
                                             int lane,
                                             int m,
                                             int kg,
                                             int half) {
    const int row = row_group * 16 + m * 32 + (lane & 0x0f);
    const int col_group = lane >> 4;
    const int tile_k = kg * 8 + half * 4;
    frag_a frag{};
    frag.reg[0] = pack_pair(a.raw_ptr[row * a.template stride<2>() + tile_k + col_group + 0],
                            a.raw_ptr[row * a.template stride<2>() + tile_k + col_group + 4]);
    frag.reg[1] = pack_pair(a.raw_ptr[row * a.template stride<2>() + tile_k + col_group + 8],
                            a.raw_ptr[row * a.template stride<2>() + tile_k + col_group + 12]);
    return frag;
}

__device__ inline frag_b make_ref_b_fragment(const b_row_gl<kK, kN> &b,
                                             int col_group,
                                             int lane,
                                             int n,
                                             int kg,
                                             int half) {
    const int row_group_k = lane >> 4;
    const int col = col_group * 16 + n * 32 + (lane & 0x0f);
    const int tile_k = kg * 8 + half * 4;
    frag_b frag{};
    frag.reg[0] = pack_pair(b.raw_ptr[(tile_k + row_group_k + 0) * b.template stride<2>() + col],
                            b.raw_ptr[(tile_k + row_group_k + 4) * b.template stride<2>() + col]);
    frag.reg[1] = pack_pair(b.raw_ptr[(tile_k + row_group_k + 8) * b.template stride<2>() + col],
                            b.raw_ptr[(tile_k + row_group_k + 12) * b.template stride<2>() + col]);
    return frag;
}

__device__ inline bool same_acc_exact(const frag_c &lhs, const frag_c &rhs) {
#pragma unroll
    for (int i = 0; i < kRegsC; ++i) {
        if (lhs.reg[i] != rhs.reg[i]) return false;
    }
    return true;
}

__global__ void operand_layouta_stage_probe_kernel(const __grid_constant__ a_gl<kM, kK> a,
                                                   const __grid_constant__ b_row_gl<kK, kN> b_row,
                                                   const __grid_constant__ b_layouta_gl<kN, kK> b_layouta,
                                                   uint32_t *ok_out) {
    __shared__ operand_ring op_ring;

    const int warp = kittens::warpid();
    const int row_group = warp / contracts::kWaveN;
    const int col_group = warp % contracts::kWaveN;
    const int lane = kittens::laneid();

    auto op_tok = kittens::arch::c500::gemm::issue_operand_stage_async_layouta(op_ring, a, b_layouta, 0);
    kittens::arch::c500::wait(op_tok);
    __syncthreads();

    bool good = true;
#pragma unroll
    for (int m = 0; m < kAtomTiles && good; ++m) {
#pragma unroll
        for (int n = 0; n < kAtomTiles && good; ++n) {
            frag_c ref_acc{};
            frag_c op_acc{};
#pragma unroll
            for (int i = 0; i < kRegsC; ++i) {
                ref_acc.reg[i] = 0.0f;
                op_acc.reg[i] = 0.0f;
            }

#pragma unroll
            for (int mma_k = 0; mma_k < 2; ++mma_k) {
                const frag_a ref_a = make_ref_a_fragment(a, row_group, lane, m, mma_k * 2, 0);
                const frag_b ref_b = make_ref_b_fragment(b_row, col_group, lane, n, mma_k * 2, 0);
                const auto a_words =
                    kittens::arch::c500::gemm::load_cta_a_operand_words(op_ring, 0, row_group, m, mma_k * 2, lane);
                const auto b_words =
                    kittens::arch::c500::gemm::load_cta_b_operand_words(op_ring, 0, col_group, n, mma_k * 2, lane);

                frag_c next_ref{};
                frag_c next_op{};
                kittens::arch::c500::mma<atom>(next_ref, ref_a, ref_b, ref_acc);
                kittens::arch::c500::mma<atom>(next_op,
                                               kittens::arch::c500::gemm::make_a_operand_fragment(a_words, 0),
                                               kittens::arch::c500::gemm::make_b_operand_fragment(b_words, 0),
                                               op_acc);
                ref_acc = next_ref;
                op_acc = next_op;
            }

            if (!same_acc_exact(ref_acc, op_acc)) {
                good = false;
            }
        }
    }

    ok_out[threadIdx.x] = good ? 1u : 0u;
}

template<int M, int N>
using c_gl = kittens::gl<kittens::bf16, 1, 1, M, N, kittens::arch::c500::gemm::bf16_shared_tile_c>;

__global__ void operand_layouta_store_probe_kernel(const __grid_constant__ a_gl<kM, kK> a,
                                                   const __grid_constant__ b_layouta_gl<kN, kK> b_layouta,
                                                   const __grid_constant__ c_gl<kM, kN> c) {
    __shared__ operand_ring op_ring;

    const int workerid = kittens::warpid();
    const int row_group = workerid / contracts::kWaveN;
    const int col_group = workerid % contracts::kWaveN;

    auto tok = kittens::arch::c500::gemm::issue_operand_stage_async_layouta_aligned(op_ring, a, b_layouta, 0);
    kittens::arch::c500::wait(tok);
    __syncthreads();

    frag_c acc[kittens::arch::c500::gemm::kAtomsM][kittens::arch::c500::gemm::kAtomsN];
    reg_tile_c out;
    kittens::arch::c500::gemm::zero_accumulators(acc);
    kittens::arch::c500::gemm::mma_operand_stage(acc, op_ring, 0, row_group, col_group);
    kittens::arch::c500::gemm::export_accumulators(out, acc);
    kittens::warp::store(c, out, {0, 0, row_group, col_group});
}

__global__ void operand_layouta_atom_probe_kernel(const __grid_constant__ a_gl<kM, kK> a,
                                                  const __grid_constant__ b_row_gl<kK, kN> b_row,
                                                  const __grid_constant__ b_layouta_gl<kN, kK> b_layouta,
                                                  uint32_t *ok_out,
                                                  int *detail_out) {
    __shared__ operand_ring op_ring;

    const int warp = kittens::warpid();
    const int row_group = warp / contracts::kWaveN;
    const int col_group = warp % contracts::kWaveN;
    const int load_group = warp / 2;
    const int lane = kittens::laneid();

    auto op_tok = kittens::arch::c500::gemm::issue_operand_stage_async_layouta(op_ring, a, b_layouta, 0);
    kittens::arch::c500::wait(op_tok);
    __syncthreads();

    bool good = true;
    bool reported = false;

#pragma unroll
    for (int m = 0; m < kAtomTiles && good; ++m) {
#pragma unroll
        for (int n = 0; n < kAtomTiles && good; ++n) {
            frag_c ref_acc{};
            frag_c op_acc{};
#pragma unroll
            for (int i = 0; i < kRegsC; ++i) {
                ref_acc.reg[i] = 0.0f;
                op_acc.reg[i] = 0.0f;
            }

#pragma unroll
            for (int kg = 0; kg < kKGroups; ++kg) {
                const frag_a ref_a_lo = make_ref_a_fragment(a, row_group, lane, m, kg, 0);
                const frag_a ref_a_hi = make_ref_a_fragment(a, row_group, lane, m, kg, 1);
                const frag_b ref_b_lo = make_ref_b_fragment(b_row, col_group, lane, n, kg, 0);
                const frag_b ref_b_hi = make_ref_b_fragment(b_row, col_group, lane, n, kg, 1);

                const operand_vec a_words =
                    kittens::arch::c500::gemm::load_cta_a_operand_words(op_ring, 0, row_group, m, kg, lane);
                const operand_vec b_words =
                    kittens::arch::c500::gemm::load_cta_b_operand_words(op_ring, 0, col_group, n, kg, lane);
                const operand_vec ref_a_words = kittens::arch::c500::gemm::pack_a_operand_words(ref_a_lo, ref_a_hi);
                const operand_vec ref_b_words = kittens::arch::c500::gemm::pack_b_operand_words(ref_b_lo, ref_b_hi);

                bool word_match = true;
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    if (a_words[w] != ref_a_words[w]) {
                        word_match = false;
                        if (!reported && atomicCAS(&detail_out[0], -1, threadIdx.x) == -1) {
                            detail_out[1] = 0;
                            detail_out[2] = m;
                            detail_out[3] = n;
                            detail_out[4] = kg;
                            detail_out[5] = w;
                        }
                        break;
                    }
                    if (b_words[w] != ref_b_words[w]) {
                        word_match = false;
                        if (!reported && atomicCAS(&detail_out[0], -1, threadIdx.x) == -1) {
                            detail_out[1] = 1;
                            detail_out[2] = m;
                            detail_out[3] = n;
                            detail_out[4] = kg;
                            detail_out[5] = w;
                        }
                        break;
                    }
                }
                if (!word_match) {
                    good = false;
                    reported = true;
                    break;
                }

                frag_c next_ref{};
                frag_c next_op{};
                kittens::arch::c500::mma<atom>(next_ref, ref_a_lo, ref_b_lo, ref_acc);
                kittens::arch::c500::mma<atom>(next_op,
                                               kittens::arch::c500::gemm::make_a_operand_fragment(a_words, 0),
                                               kittens::arch::c500::gemm::make_b_operand_fragment(b_words, 0),
                                               op_acc);
                ref_acc = next_ref;
                op_acc = next_op;

                kittens::arch::c500::mma<atom>(next_ref, ref_a_hi, ref_b_hi, ref_acc);
                kittens::arch::c500::mma<atom>(next_op,
                                               kittens::arch::c500::gemm::make_a_operand_fragment(a_words, 1),
                                               kittens::arch::c500::gemm::make_b_operand_fragment(b_words, 1),
                                               op_acc);
                ref_acc = next_ref;
                op_acc = next_op;
            }

            if (!same_acc(ref_acc, op_acc)) {
                good = false;
                if (!reported && atomicCAS(&detail_out[0], -1, threadIdx.x) == -1) {
                    detail_out[1] = 2;
                    detail_out[2] = m;
                    detail_out[3] = n;
                    detail_out[4] = -1;
                    detail_out[5] = -1;
                }
                reported = true;
            }
        }
    }

    ok_out[threadIdx.x] = good ? 1u : 0u;
}

bool run_probe(test_data &results) {
    test_info info{"c500_gemm_bf16_operand_layouta_atom_contract", test_result::FAILED};

    std::vector<bf16> h_a(kM * kK);
    std::vector<bf16> h_b_row(kK * kN);
    std::vector<bf16> h_b_layouta(kN * kK);
    for (int i = 0; i < kM * kK; ++i) h_a[i] = __float2bfloat16(static_cast<float>((i % 251) + 1));
    for (int i = 0; i < kK * kN; ++i) h_b_row[i] = __float2bfloat16(static_cast<float>((i % 241) + 3));
    for (int k = 0; k < kK; ++k) {
        for (int n = 0; n < kN; ++n) {
            h_b_layouta[n * kK + k] = h_b_row[k * kN + n];
        }
    }

    bf16 *d_a = nullptr;
    bf16 *d_b_row = nullptr;
    bf16 *d_b_layouta = nullptr;
    uint32_t *d_ok = nullptr;
    int *d_detail = nullptr;
    std::vector<uint32_t> h_ok(kThreads, 0);
    std::vector<int> h_detail(6, -1);

    cudaMalloc(&d_a, h_a.size() * sizeof(bf16));
    cudaMalloc(&d_b_row, h_b_row.size() * sizeof(bf16));
    cudaMalloc(&d_b_layouta, h_b_layouta.size() * sizeof(bf16));
    cudaMalloc(&d_ok, h_ok.size() * sizeof(uint32_t));
    cudaMalloc(&d_detail, h_detail.size() * sizeof(int));
    CudaCheckError();

    cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_row, h_b_row.data(), h_b_row.size() * sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_layouta, h_b_layouta.data(), h_b_layouta.size() * sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_detail, h_detail.data(), h_detail.size() * sizeof(int), cudaMemcpyHostToDevice);
    CudaCheckError();

    operand_layouta_atom_probe_kernel<<<1, kThreads>>>(a_gl<kM, kK>{d_a, nullptr, nullptr, nullptr, nullptr},
                                                       b_row_gl<kK, kN>{d_b_row, nullptr, nullptr, nullptr, nullptr},
                                                       b_layouta_gl<kN, kK>{d_b_layouta, nullptr, nullptr, nullptr, nullptr},
                                                       d_ok,
                                                       d_detail);
    CudaCheckError();
    cudaMemcpy(h_ok.data(), d_ok, h_ok.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_detail.data(), d_detail, h_detail.size() * sizeof(int), cudaMemcpyDeviceToHost);
    CudaCheckError();

    bool good = true;
    for (int tid = 0; tid < kThreads; ++tid) {
        if (h_ok[tid] == 0) {
            good = false;
            std::cout << "first mismatch tid=" << tid
                      << " kind=" << h_detail[1]
                      << " m=" << h_detail[2]
                      << " n=" << h_detail[3]
                      << " kg=" << h_detail[4]
                      << " word=" << h_detail[5]
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
    cudaFree(d_b_row);
    cudaFree(d_b_layouta);
    cudaFree(d_ok);
    cudaFree(d_detail);
    CudaCheckError();
    return good;
}

bool run_stage_probe(test_data &results) {
    test_info info{"c500_gemm_bf16_operand_layouta_stage_contract", test_result::FAILED};

    std::vector<bf16> h_a(kM * kK);
    std::vector<bf16> h_b_row(kK * kN);
    std::vector<bf16> h_b_layouta(kN * kK);
    for (int i = 0; i < kM * kK; ++i) h_a[i] = __float2bfloat16(static_cast<float>((i % 251) + 1));
    for (int i = 0; i < kK * kN; ++i) h_b_row[i] = __float2bfloat16(static_cast<float>((i % 241) + 3));
    for (int k = 0; k < kK; ++k) {
        for (int n = 0; n < kN; ++n) {
            h_b_layouta[n * kK + k] = h_b_row[k * kN + n];
        }
    }

    bf16 *d_a = nullptr;
    bf16 *d_b_row = nullptr;
    bf16 *d_b_layouta = nullptr;
    uint32_t *d_ok = nullptr;
    std::vector<uint32_t> h_ok(kThreads, 0);

    cudaMalloc(&d_a, h_a.size() * sizeof(bf16));
    cudaMalloc(&d_b_row, h_b_row.size() * sizeof(bf16));
    cudaMalloc(&d_b_layouta, h_b_layouta.size() * sizeof(bf16));
    cudaMalloc(&d_ok, h_ok.size() * sizeof(uint32_t));
    CudaCheckError();

    cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_row, h_b_row.data(), h_b_row.size() * sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_layouta, h_b_layouta.data(), h_b_layouta.size() * sizeof(bf16), cudaMemcpyHostToDevice);
    CudaCheckError();

    operand_layouta_stage_probe_kernel<<<1, kThreads>>>(a_gl<kM, kK>{d_a, nullptr, nullptr, nullptr, nullptr},
                                                        b_row_gl<kK, kN>{d_b_row, nullptr, nullptr, nullptr, nullptr},
                                                        b_layouta_gl<kN, kK>{d_b_layouta, nullptr, nullptr, nullptr, nullptr},
                                                        d_ok);
    CudaCheckError();
    cudaMemcpy(h_ok.data(), d_ok, h_ok.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    CudaCheckError();

    bool good = true;
    for (int tid = 0; tid < kThreads; ++tid) {
        if (h_ok[tid] == 0) {
            good = false;
            std::cout << "first stage mismatch tid=" << tid << std::endl;
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
    cudaFree(d_b_row);
    cudaFree(d_b_layouta);
    cudaFree(d_ok);
    CudaCheckError();
    return good;
}

bool run_store_probe(test_data &results) {
    test_info info{"c500_gemm_bf16_operand_layouta_store_contract", test_result::FAILED};
    constexpr int kRefK = 32;

    std::vector<bf16> h_a(kM * kK);
    std::vector<bf16> h_b_row(kK * kN);
    std::vector<bf16> h_b_layouta(kN * kK);
    for (int i = 0; i < kM * kK; ++i) h_a[i] = __float2bfloat16(static_cast<float>((i % 251) + 1));
    for (int i = 0; i < kK * kN; ++i) h_b_row[i] = __float2bfloat16(static_cast<float>((i % 241) + 3));
    for (int k = 0; k < kK; ++k) {
        for (int n = 0; n < kN; ++n) {
            h_b_layouta[n * kK + k] = h_b_row[k * kN + n];
        }
    }

    bf16 *d_a = nullptr;
    bf16 *d_b_row = nullptr;
    bf16 *d_b_layouta = nullptr;
    bf16 *d_c = nullptr;
    bf16 *d_ref = nullptr;
    std::vector<bf16> h_c(kM * kN);
    std::vector<bf16> h_ref(kM * kN);

    cudaMalloc(&d_a, h_a.size() * sizeof(bf16));
    cudaMalloc(&d_b_row, h_b_row.size() * sizeof(bf16));
    cudaMalloc(&d_b_layouta, h_b_layouta.size() * sizeof(bf16));
    cudaMalloc(&d_c, h_c.size() * sizeof(bf16));
    cudaMalloc(&d_ref, h_ref.size() * sizeof(bf16));
    CudaCheckError();

    cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_row, h_b_row.data(), h_b_row.size() * sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_layouta, h_b_layouta.data(), h_b_layouta.size() * sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemset(d_c, 0, h_c.size() * sizeof(bf16));
    cudaMemset(d_ref, 0, h_ref.size() * sizeof(bf16));
    CudaCheckError();

    reference_gemm<__nv_bfloat16, __nv_bfloat16, false>(reinterpret_cast<__nv_bfloat16 *>(d_ref),
                                                        reinterpret_cast<__nv_bfloat16 *>(d_a),
                                                        reinterpret_cast<__nv_bfloat16 *>(d_b_row),
                                                        kM, kN, kRefK);
    cudaDeviceSynchronize();
    CudaCheckError();

    operand_layouta_store_probe_kernel<<<1, kThreads>>>(a_gl<kM, kK>{d_a, nullptr, nullptr, nullptr, nullptr},
                                                        b_layouta_gl<kN, kK>{d_b_layouta, nullptr, nullptr, nullptr, nullptr},
                                                        c_gl<kM, kN>{d_c, nullptr, nullptr, nullptr, nullptr});
    CudaCheckError();
    cudaDeviceSynchronize();
    CudaCheckError();

    cudaMemcpy(h_c.data(), d_c, h_c.size() * sizeof(bf16), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ref.data(), d_ref, h_ref.size() * sizeof(bf16), cudaMemcpyDeviceToHost);
    CudaCheckError();

    bool good = true;
    for (int i = 0; i < kM * kN; ++i) {
        if (h_c[i] != h_ref[i]) {
            good = false;
            std::cout << "first store mismatch idx=" << i
                      << " got=" << __bfloat162float(h_c[i])
                      << " ref=" << __bfloat162float(h_ref[i]) << std::endl;
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
    cudaFree(d_b_row);
    cudaFree(d_b_layouta);
    cudaFree(d_c);
    cudaFree(d_ref);
    CudaCheckError();
    return good;
}

} // namespace

void tests(test_data &results) {
    std::cout << " ----- Starting ops/c500/mma/operand_layouta_atom_probe tests! -----\n" << std::endl;
    run_probe(results);
    run_stage_probe(results);
    run_store_probe(results);
}

} // namespace c500::mma::operand_layouta_atom_probe

#endif
