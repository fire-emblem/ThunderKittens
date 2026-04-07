#include "testing_flags.cuh"

#ifdef TEST_C500_GEMM_OPERAND_B_LAYOUTA_DIRECT_ASYNC_PROBE

#include <vector>

#include "testing_commons.cuh"

#include "arch/c500/gemm/bf16_operand_stage.cuh"

namespace c500::mma::operand_b_layouta_direct_async_probe {

namespace {

using namespace kittens;
using coords = kittens::arch::c500::gemm::bf16_balanced_operand_coords;
using cta_ring = kittens::arch::c500::gemm::bf16_operand_cta_stage_ring_1;

template<int N, int K>
using b_layouta_gl = kittens::gl<kittens::bf16, 1, 1, N, K>;

constexpr int kN = 128;
constexpr int kK = 128;
constexpr int kThreads = 256;
constexpr int kWordsPerVec = 4;

__host__ __device__ inline uint32_t pack_pair(bf16 lo, bf16 hi) {
    const bf16_2 pair{lo, hi};
    return *reinterpret_cast<const uint32_t *>(&pair);
}

__global__ void operand_b_layouta_direct_async_probe_kernel(const __grid_constant__ b_layouta_gl<kN, kK> b,
                                                            uint32_t *b_out) {
    __shared__ cta_ring ring;

    auto tok = kittens::arch::c500::gemm::issue_b_operand_stage_async_layouta(ring, b, 0);
    kittens::arch::c500::wait(tok);
    __syncthreads();

    const int warp = kittens::warpid();
    const int col_group = warp % 2;
    const int lane = kittens::laneid();

#pragma unroll
    for (int n = 0; n < 4; ++n) {
#pragma unroll
        for (int kg = 0; kg < 4; ++kg) {
            const auto words = kittens::arch::c500::gemm::load_cta_b_operand_words(ring, 0, col_group, n, kg, lane);
            const int base = ((((threadIdx.x * 4) + n) * 4 + kg) * kWordsPerVec);
            for (int w = 0; w < kWordsPerVec; ++w) b_out[base + w] = words[w];
        }
    }
}

bool run_probe(test_data &results) {
    test_info info{"c500_gemm_bf16_operand_b_layouta_direct_async_contract", test_result::FAILED};

    std::vector<bf16> h_b(kN * kK);
    for (int c = 0; c < kN; ++c) {
        for (int r = 0; r < kK; ++r) {
            h_b[c * kK + r] = __float2bfloat16(static_cast<float>(10000 + c * kK + r + 1));
        }
    }

    bf16 *d_b = nullptr;
    std::vector<uint32_t> h_b_out(kThreads * 4 * 4 * kWordsPerVec, 0);
    uint32_t *d_b_out = nullptr;

    cudaMalloc(&d_b, h_b.size() * sizeof(bf16));
    cudaMalloc(&d_b_out, h_b_out.size() * sizeof(uint32_t));
    CudaCheckError();

    cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(bf16), cudaMemcpyHostToDevice);
    CudaCheckError();

    operand_b_layouta_direct_async_probe_kernel<<<1, kThreads>>>(b_layouta_gl<kN, kK>{d_b, nullptr, nullptr, nullptr, nullptr},
                                                                 d_b_out);
    CudaCheckError();

    cudaMemcpy(h_b_out.data(), d_b_out, h_b_out.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    CudaCheckError();

    bool good = true;
    for (int tid = 0; tid < kThreads && good; ++tid) {
        const int warp = tid / 64;
        const int col_group = warp % 2;
        const int lane = tid & 63;
        for (int n = 0; n < 4 && good; ++n) {
            const int b_col = col_group * 16 + n * 32 + coords::lane_mn(lane);
            for (int kg = 0; kg < 4 && good; ++kg) {
                const int b_row = coords::k_elem(kg, lane);
                const uint32_t expect[4] = {
                    pack_pair(h_b[b_col * kK + (b_row + 0)], h_b[b_col * kK + (b_row + 1)]),
                    pack_pair(h_b[b_col * kK + (b_row + 2)], h_b[b_col * kK + (b_row + 3)]),
                    pack_pair(h_b[b_col * kK + (b_row + 4)], h_b[b_col * kK + (b_row + 5)]),
                    pack_pair(h_b[b_col * kK + (b_row + 6)], h_b[b_col * kK + (b_row + 7)]),
                };
                const int base = ((((tid * 4) + n) * 4 + kg) * kWordsPerVec);
                for (int w = 0; w < kWordsPerVec; ++w) {
                    if (h_b_out[base + w] != expect[w]) {
                        good = false;
                        std::cout << "B mismatch tid=" << tid << " n=" << n << " kg=" << kg << " word=" << w
                                  << " expect=0x" << std::hex << expect[w]
                                  << " got=0x" << h_b_out[base + w] << std::dec << std::endl;
                        break;
                    }
                }
            }
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

    cudaFree(d_b);
    cudaFree(d_b_out);
    CudaCheckError();
    return good;
}

} // namespace

void tests(test_data &results) {
    std::cout << " ----- Starting ops/c500/mma/operand_b_layouta_direct_async_probe tests! -----\n" << std::endl;
    run_probe(results);
}

} // namespace c500::mma::operand_b_layouta_direct_async_probe

#endif
