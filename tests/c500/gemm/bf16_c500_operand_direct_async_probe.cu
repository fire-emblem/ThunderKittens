#include "testing_flags.cuh"

#ifdef TEST_C500_GEMM_OPERAND_DIRECT_ASYNC_PROBE

#include <vector>

#include "testing_commons.cuh"

#include "arch/c500/gemm/bf16_operand_stage.cuh"

namespace c500::mma::operand_direct_async_probe {

namespace {

using namespace kittens;
using coords = kittens::arch::c500::gemm::bf16_balanced_operand_coords;
using cta_ring = kittens::arch::c500::gemm::bf16_operand_cta_stage_ring_1;
using operand_vec = kittens::arch::c500::gemm::bf16_operand_vec;

template<int M, int K>
using a_gl = kittens::gl<kittens::bf16, 1, 1, M, K>;

constexpr int kM = 128;
constexpr int kK = 128;
constexpr int kThreads = 256;
constexpr int kWordsPerVec = 4;

__host__ __device__ inline uint32_t pack_pair(bf16 lo, bf16 hi) {
    const bf16_2 pair{lo, hi};
    return *reinterpret_cast<const uint32_t *>(&pair);
}

__global__ void operand_direct_async_probe_kernel(const __grid_constant__ a_gl<kM, kK> a,
                                                  uint32_t *a_out) {
    __shared__ cta_ring ring;

    auto tok = kittens::arch::c500::gemm::issue_a_operand_stage_async(ring, a, 0);
    kittens::arch::c500::wait(tok);
    __syncthreads();

    const int warp = kittens::warpid();
    const int row_group = warp / 2;
    const int lane = kittens::laneid();

#pragma unroll
    for (int m = 0; m < 4; ++m) {
#pragma unroll
        for (int kg = 0; kg < 4; ++kg) {
            const auto words = kittens::arch::c500::gemm::load_cta_a_operand_words(ring, 0, row_group, m, kg, lane);
            const int base = ((((threadIdx.x * 4) + m) * 4 + kg) * kWordsPerVec);
            for (int w = 0; w < kWordsPerVec; ++w) a_out[base + w] = words[w];
        }
    }
}

bool run_probe(test_data &results) {
    test_info info{"c500_gemm_bf16_operand_a_direct_async_contract", test_result::FAILED};

    std::vector<bf16> h_a(kM * kK);
    for (int i = 0; i < kM * kK; ++i) h_a[i] = __float2bfloat16(static_cast<float>(i + 1));

    bf16 *d_a = nullptr;
    std::vector<uint32_t> h_a_out(kThreads * 4 * 4 * kWordsPerVec, 0);
    uint32_t *d_a_out = nullptr;

    cudaMalloc(&d_a, h_a.size() * sizeof(bf16));
    cudaMalloc(&d_a_out, h_a_out.size() * sizeof(uint32_t));
    CudaCheckError();

    cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(bf16), cudaMemcpyHostToDevice);
    CudaCheckError();

    operand_direct_async_probe_kernel<<<1, kThreads>>>(a_gl<kM, kK>{d_a, nullptr, nullptr, nullptr, nullptr}, d_a_out);
    CudaCheckError();

    cudaMemcpy(h_a_out.data(), d_a_out, h_a_out.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    CudaCheckError();

    bool good = true;
    for (int tid = 0; tid < kThreads && good; ++tid) {
        const int warp = tid / 64;
        const int row_group = warp / 2;
        const int lane = tid & 63;
        for (int m = 0; m < 4 && good; ++m) {
            const int a_row = row_group * 16 + m * 32 + coords::lane_mn(lane);
            for (int kg = 0; kg < 4 && good; ++kg) {
                const int a_col = coords::k_elem(kg, lane);
                const uint32_t expect[4] = {
                    pack_pair(h_a[a_row * kK + a_col + 0], h_a[a_row * kK + a_col + 1]),
                    pack_pair(h_a[a_row * kK + a_col + 2], h_a[a_row * kK + a_col + 3]),
                    pack_pair(h_a[a_row * kK + a_col + 4], h_a[a_row * kK + a_col + 5]),
                    pack_pair(h_a[a_row * kK + a_col + 6], h_a[a_row * kK + a_col + 7]),
                };
                const int base = ((((tid * 4) + m) * 4 + kg) * kWordsPerVec);
                for (int w = 0; w < kWordsPerVec; ++w) {
                    if (h_a_out[base + w] != expect[w]) {
                        good = false;
                        std::cout << "A mismatch tid=" << tid << " m=" << m << " kg=" << kg << " word=" << w
                                  << " expect=0x" << std::hex << expect[w]
                                  << " got=0x" << h_a_out[base + w] << std::dec << std::endl;
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

    cudaFree(d_a);
    cudaFree(d_a_out);
    CudaCheckError();
    return good;
}

} // namespace

void tests(test_data &results) {
    std::cout << " ----- Starting ops/c500/mma/operand_direct_async_probe tests! -----\n" << std::endl;
    run_probe(results);
}

} // namespace c500::mma::operand_direct_async_probe

#endif
