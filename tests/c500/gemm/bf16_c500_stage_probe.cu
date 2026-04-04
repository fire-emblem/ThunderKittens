#include "testing_flags.cuh"

#ifdef TEST_C500_GEMM_STAGE_PROBE

#include <vector>

#include "testing_commons.cuh"

#include "arch/c500/gemm/bf16_stage_primitives.cuh"

namespace c500::mma::stage_probe {

namespace {

using stage_layout = kittens::arch::c500::gemm::bf16_128x128x128_stage_layout;
using stage_ring = kittens::arch::c500::gemm::bf16_stage_ring;
using stage_vec = kittens::arch::c500::gemm::bf16_stage_vec;

constexpr int kThreads = stage_layout::kThreads;
constexpr int kStages = stage_layout::kStages;
constexpr int kKGroups = 4;
constexpr int kWordsPerVec = 4;

__host__ __device__ inline uint32_t make_word(int operand, int stage, int k_group, int tid, int word) {
    return (static_cast<uint32_t>(operand) << 28) |
           (static_cast<uint32_t>(stage) << 24) |
           (static_cast<uint32_t>(k_group) << 20) |
           (static_cast<uint32_t>(tid & 0xff) << 8) |
           static_cast<uint32_t>(word);
}

__host__ __device__ inline int canonical_a_tid(int tid) {
    const int slot = tid / 64;
    return (slot & 1) ? (tid - 64) : tid;
}

__host__ __device__ inline int canonical_b_tid(int tid) {
    const int slot = tid / 64;
    return (slot >= 2) ? (tid - 128) : tid;
}

__global__ void raw_stage_probe_kernel(uint32_t *a_out, uint32_t *b_out) {
    __shared__ stage_ring ring;

    for (int stage = 0; stage < kStages; ++stage) {
#pragma unroll
        for (int kg = 0; kg < kKGroups; ++kg) {
            const uint32_t a_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&ring.bytes[0])) +
                                    stage_layout::stage_offset(stage) +
                                    kittens::arch::c500::gemm::lds_offset_a(threadIdx.x, kg);
            const uint32_t b_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&ring.bytes[0])) +
                                    stage_layout::stage_offset(stage) +
                                    kittens::arch::c500::gemm::lds_offset_b(threadIdx.x, kg);

            if ((threadIdx.x / 64) != 1 && (threadIdx.x / 64) != 3) {
                auto *a_ptr = reinterpret_cast<stage_vec *>(__cvta_shared_to_generic(a_addr));
                (*a_ptr)[0] = make_word(0, stage, kg, threadIdx.x, 0);
                (*a_ptr)[1] = make_word(0, stage, kg, threadIdx.x, 1);
                (*a_ptr)[2] = make_word(0, stage, kg, threadIdx.x, 2);
                (*a_ptr)[3] = make_word(0, stage, kg, threadIdx.x, 3);
            }

            if ((threadIdx.x / 64) < 2) {
                auto *b_ptr = reinterpret_cast<stage_vec *>(__cvta_shared_to_generic(b_addr));
                (*b_ptr)[0] = make_word(1, stage, kg, threadIdx.x, 0);
                (*b_ptr)[1] = make_word(1, stage, kg, threadIdx.x, 1);
                (*b_ptr)[2] = make_word(1, stage, kg, threadIdx.x, 2);
                (*b_ptr)[3] = make_word(1, stage, kg, threadIdx.x, 3);
            }
        }
    }
    __syncthreads();

    for (int stage = 0; stage < kStages; ++stage) {
#pragma unroll
        for (int kg = 0; kg < kKGroups; ++kg) {
            const auto a_words = kittens::arch::c500::gemm::load_stage_a_words(ring, stage, threadIdx.x, kg);
            const auto b_words = kittens::arch::c500::gemm::load_stage_b_words(ring, stage, threadIdx.x, kg);

            const int vec_idx = ((stage * kKGroups) + kg) * kThreads + threadIdx.x;
            for (int word = 0; word < kWordsPerVec; ++word) {
                a_out[vec_idx * kWordsPerVec + word] = a_words[word];
                b_out[vec_idx * kWordsPerVec + word] = b_words[word];
            }
        }
    }
}

bool run_stage_contract(test_data &results) {
    test_info info{"c500_gemm_bf16_stage_raw_load_contract", test_result::FAILED};

    constexpr int kVecCount = kStages * kKGroups * kThreads;
    const size_t count = static_cast<size_t>(kVecCount) * kWordsPerVec;

    uint32_t *d_a = nullptr;
    uint32_t *d_b = nullptr;
    std::vector<uint32_t> h_a(count, 0);
    std::vector<uint32_t> h_b(count, 0);
    cudaMalloc(&d_a, count * sizeof(uint32_t));
    cudaMalloc(&d_b, count * sizeof(uint32_t));
    CudaCheckError();

    raw_stage_probe_kernel<<<1, kThreads>>>(d_a, d_b);
    CudaCheckError();

    cudaMemcpy(h_a.data(), d_a, count * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b.data(), d_b, count * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    CudaCheckError();

    bool good = true;
    for (int stage = 0; stage < kStages && good; ++stage) {
        for (int kg = 0; kg < kKGroups && good; ++kg) {
            for (int tid = 0; tid < kThreads && good; ++tid) {
                const int vec_idx = ((stage * kKGroups) + kg) * kThreads + tid;
                for (int word = 0; word < kWordsPerVec; ++word) {
                    const uint32_t expect_a = make_word(0, stage, kg, canonical_a_tid(tid), word);
                    const uint32_t expect_b = make_word(1, stage, kg, canonical_b_tid(tid), word);
                    if (h_a[vec_idx * kWordsPerVec + word] != expect_a ||
                        h_b[vec_idx * kWordsPerVec + word] != expect_b) {
                        std::cout << "first mismatch at stage=" << stage
                                  << " kg=" << kg
                                  << " tid=" << tid
                                  << " word=" << word
                                  << " expect_a=0x" << std::hex << expect_a
                                  << " got_a=0x" << h_a[vec_idx * kWordsPerVec + word]
                                  << " expect_b=0x" << expect_b
                                  << " got_b=0x" << h_b[vec_idx * kWordsPerVec + word]
                                  << std::dec << std::endl;
                        good = false;
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
    cudaFree(d_b);
    CudaCheckError();
    return good;
}

} // namespace

void tests(test_data &results) {
    std::cout << " ----- Starting ops/c500/mma/stage_probe tests! -----\n" << std::endl;
    run_stage_contract(results);
}

} // namespace c500::mma::stage_probe

#endif
