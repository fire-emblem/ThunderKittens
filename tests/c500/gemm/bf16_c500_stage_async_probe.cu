#include "testing_flags.cuh"

#ifdef TEST_C500_GEMM_STAGE_ASYNC_PROBE

#include <vector>

#include "testing_commons.cuh"

#include "arch/c500/gemm/bf16_stage_primitives.cuh"
#include "arch/c500/primitives/pipeline.cuh"

namespace c500::mma::stage_async_probe {

namespace {

using namespace kittens;
using stage_ring = kittens::arch::c500::gemm::bf16_stage_ring;
template<int M, int K>
using a_gl = kittens::gl<kittens::bf16, 1, 1, M, K>;
template<int K, int N>
using b_gl = kittens::gl<kittens::bf16, 1, 1, K, N>;

constexpr int kM = 128;
constexpr int kN = 128;
constexpr int kK = 32;
constexpr int kRowsA = 64;
constexpr int kColsA = 32;
constexpr int kRowsB = 32;
constexpr int kColsB = 64;
constexpr int kWordsPerVec = 4;
constexpr int kThreads = 256;

__host__ __device__ inline uint32_t pack_pair(bf16 lo, bf16 hi) {
    const bf16_2 pair{lo, hi};
    return *reinterpret_cast<const uint32_t *>(&pair);
}

__global__ void stage_async_probe_kernel(const __grid_constant__ a_gl<kM, kK> a,
                                         const __grid_constant__ b_gl<kK, kN> b,
                                         uint32_t *a_out,
                                         uint32_t *b_out) {
    __shared__ stage_ring ring;

    const int warp = kittens::warpid();
    const int load_group = warp / 2;

    auto tok = kittens::arch::c500::gemm::issue_ab_stage_async(ring, a, b, 0, load_group, 0, 0, 0);
    kittens::arch::c500::wait(tok);
    __syncthreads();

    for (int kg = 0; kg < 4; ++kg) {
        const auto a_words = kittens::arch::c500::gemm::load_stage_a_words(ring, 0, threadIdx.x, kg);
        const auto b_words = kittens::arch::c500::gemm::load_stage_b_words(ring, 0, threadIdx.x, kg);
        const int base = (threadIdx.x * 4 + kg) * kWordsPerVec;
        for (int w = 0; w < kWordsPerVec; ++w) {
            a_out[base + w] = a_words[w];
            b_out[base + w] = b_words[w];
        }
    }
}

__global__ void stage_async_window_probe_kernel(const __grid_constant__ a_gl<kM, 2 * kK> a,
                                                const __grid_constant__ b_gl<2 * kK, kN> b,
                                                uint32_t *a_stage0,
                                                uint32_t *a_stage1) {
    __shared__ stage_ring ring;

    const int warp = kittens::warpid();
    const int load_group = warp / 2;

    auto tok0 = kittens::arch::c500::gemm::issue_ab_stage_async(ring, a, b, 0, load_group, 0, 0, 0);
    auto tok1 = kittens::arch::c500::gemm::issue_ab_stage_async(ring, a, b, 1, load_group, 0, 0, 1);
    (void)tok0;
    (void)tok1;
    kittens::arch::c500::primitives::wait_stage_window<4>(1);
    __syncthreads();

    const auto words0 = kittens::arch::c500::gemm::load_stage_a_words(ring, 0, threadIdx.x, 0);
    const auto words1 = kittens::arch::c500::gemm::load_stage_a_words(ring, 1, threadIdx.x, 0);
    const int base = threadIdx.x * kWordsPerVec;
    for (int w = 0; w < kWordsPerVec; ++w) {
        a_stage0[base + w] = words0[w];
        a_stage1[base + w] = words1[w];
    }
}

__host__ __device__ inline int canonical_a_load_group(int tid) {
    return (tid / 64) / 2;
}

__host__ __device__ inline int canonical_b_load_group(int tid) {
    return (tid / 64) & 1;
}

__host__ __device__ inline int a_local_load_idx(int lane64, int kg) {
    return (kg / 2) * 128 + (kg & 1) * 64 + lane64;
}

__host__ __device__ inline int b_local_load_idx(int lane64, int kg) {
    return (kg / 2) * 128 + (kg & 1) * 64 + lane64;
}

bool run_probe(test_data &results) {
    test_info info{"c500_gemm_bf16_stage_async_raw_vector_contract", test_result::FAILED};

    std::vector<bf16> h_a(kM * kK);
    std::vector<bf16> h_b(kK * kN);
    for (int i = 0; i < kM * kK; ++i) h_a[i] = __float2bfloat16(static_cast<float>(i + 1));
    for (int i = 0; i < kK * kN; ++i) h_b[i] = __float2bfloat16(static_cast<float>(10000 + i + 1));

    bf16 *d_a = nullptr;
    bf16 *d_b = nullptr;
    uint32_t *d_a_out = nullptr;
    uint32_t *d_b_out = nullptr;
    std::vector<uint32_t> h_a_out(kThreads * 4 * kWordsPerVec, 0);
    std::vector<uint32_t> h_b_out(kThreads * 4 * kWordsPerVec, 0);
    cudaMalloc(&d_a, h_a.size() * sizeof(bf16));
    cudaMalloc(&d_b, h_b.size() * sizeof(bf16));
    cudaMalloc(&d_a_out, h_a_out.size() * sizeof(uint32_t));
    cudaMalloc(&d_b_out, h_b_out.size() * sizeof(uint32_t));
    CudaCheckError();

    cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(bf16), cudaMemcpyHostToDevice);
    CudaCheckError();

    stage_async_probe_kernel<<<1, kThreads>>>(a_gl<kM, kK>{d_a, nullptr, nullptr, nullptr, nullptr},
                                              b_gl<kK, kN>{d_b, nullptr, nullptr, nullptr, nullptr},
                                              d_a_out, d_b_out);
    CudaCheckError();
    cudaMemcpy(h_a_out.data(), d_a_out, h_a_out.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b_out.data(), d_b_out, h_b_out.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    CudaCheckError();

    bool good = true;
    for (int tid = 0; tid < kThreads && good; ++tid) {
        const int lane64 = tid & 63;
        const int a_group = canonical_a_load_group(tid);
        const int b_group = canonical_b_load_group(tid);
        for (int kg = 0; kg < 4 && good; ++kg) {
            const int a_idx = a_local_load_idx(lane64, kg);
            const int a_row = a_group * kRowsA + a_idx / (kColsA / 8);
            const int a_vec = a_idx % (kColsA / 8);
            const int b_idx = b_local_load_idx(lane64, kg);
            const int b_row = b_idx / (kColsB / 8);
            const int b_vec = b_idx % (kColsB / 8);
            const int b_col_base = b_group * kColsB + b_vec * 8;
            const uint32_t expect_a[4] = {
                pack_pair(h_a[a_row * kK + a_vec * 8 + 0], h_a[a_row * kK + a_vec * 8 + 1]),
                pack_pair(h_a[a_row * kK + a_vec * 8 + 2], h_a[a_row * kK + a_vec * 8 + 3]),
                pack_pair(h_a[a_row * kK + a_vec * 8 + 4], h_a[a_row * kK + a_vec * 8 + 5]),
                pack_pair(h_a[a_row * kK + a_vec * 8 + 6], h_a[a_row * kK + a_vec * 8 + 7]),
            };
            const uint32_t expect_b[4] = {
                pack_pair(h_b[b_row * kN + b_col_base + 0], h_b[b_row * kN + b_col_base + 1]),
                pack_pair(h_b[b_row * kN + b_col_base + 2], h_b[b_row * kN + b_col_base + 3]),
                pack_pair(h_b[b_row * kN + b_col_base + 4], h_b[b_row * kN + b_col_base + 5]),
                pack_pair(h_b[b_row * kN + b_col_base + 6], h_b[b_row * kN + b_col_base + 7]),
            };
            const int base = (tid * 4 + kg) * kWordsPerVec;
            for (int w = 0; w < kWordsPerVec; ++w) {
                if (h_a_out[base + w] != expect_a[w] || h_b_out[base + w] != expect_b[w]) {
                    good = false;
                    std::cout << "first mismatch tid=" << tid
                              << " kg=" << kg
                              << " word=" << w
                              << " a expect=0x" << std::hex << expect_a[w]
                              << " got=0x" << h_a_out[base + w]
                              << " b expect=0x" << expect_b[w]
                              << " got=0x" << h_b_out[base + w]
                              << std::dec << std::endl;
                    break;
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
    cudaFree(d_a_out);
    cudaFree(d_b_out);
    CudaCheckError();
    return good;
}

bool run_wait_window_probe(test_data &results) {
    test_info info{"c500_gemm_bf16_stage_async_wait_window_contract", test_result::FAILED};

    std::vector<bf16> h_a(kM * 2 * kK);
    std::vector<bf16> h_b(2 * kK * kN);
    for (int i = 0; i < kM * 2 * kK; ++i) h_a[i] = __float2bfloat16(static_cast<float>(20000 + i + 1));
    for (int i = 0; i < 2 * kK * kN; ++i) h_b[i] = __float2bfloat16(static_cast<float>(30000 + i + 1));

    bf16 *d_a = nullptr;
    bf16 *d_b = nullptr;
    uint32_t *d_a_stage0 = nullptr;
    uint32_t *d_a_stage1 = nullptr;
    std::vector<uint32_t> h_a_stage0(kThreads * kWordsPerVec, 0);
    std::vector<uint32_t> h_a_stage1(kThreads * kWordsPerVec, 0);
    cudaMalloc(&d_a, h_a.size() * sizeof(bf16));
    cudaMalloc(&d_b, h_b.size() * sizeof(bf16));
    cudaMalloc(&d_a_stage0, h_a_stage0.size() * sizeof(uint32_t));
    cudaMalloc(&d_a_stage1, h_a_stage1.size() * sizeof(uint32_t));
    CudaCheckError();

    cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(bf16), cudaMemcpyHostToDevice);
    CudaCheckError();

    stage_async_window_probe_kernel<<<1, kThreads>>>(a_gl<kM, 2 * kK>{d_a, nullptr, nullptr, nullptr, nullptr},
                                                     b_gl<2 * kK, kN>{d_b, nullptr, nullptr, nullptr, nullptr},
                                                     d_a_stage0,
                                                     d_a_stage1);
    CudaCheckError();
    cudaMemcpy(h_a_stage0.data(), d_a_stage0, h_a_stage0.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_a_stage1.data(), d_a_stage1, h_a_stage1.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    CudaCheckError();

    bool good = true;
    for (int tid = 0; tid < kThreads && good; ++tid) {
        const int lane64 = tid & 63;
        const int a_group = canonical_a_load_group(tid);
        const int a_idx = a_local_load_idx(lane64, 0);
        const int a_row = a_group * kRowsA + a_idx / (kColsA / 8);
        const int a_vec = a_idx % (kColsA / 8);
        const uint32_t expect_stage0[4] = {
            pack_pair(h_a[a_row * (2 * kK) + a_vec * 8 + 0], h_a[a_row * (2 * kK) + a_vec * 8 + 1]),
            pack_pair(h_a[a_row * (2 * kK) + a_vec * 8 + 2], h_a[a_row * (2 * kK) + a_vec * 8 + 3]),
            pack_pair(h_a[a_row * (2 * kK) + a_vec * 8 + 4], h_a[a_row * (2 * kK) + a_vec * 8 + 5]),
            pack_pair(h_a[a_row * (2 * kK) + a_vec * 8 + 6], h_a[a_row * (2 * kK) + a_vec * 8 + 7]),
        };
        const uint32_t expect_stage1[4] = {
            pack_pair(h_a[a_row * (2 * kK) + kK + a_vec * 8 + 0], h_a[a_row * (2 * kK) + kK + a_vec * 8 + 1]),
            pack_pair(h_a[a_row * (2 * kK) + kK + a_vec * 8 + 2], h_a[a_row * (2 * kK) + kK + a_vec * 8 + 3]),
            pack_pair(h_a[a_row * (2 * kK) + kK + a_vec * 8 + 4], h_a[a_row * (2 * kK) + kK + a_vec * 8 + 5]),
            pack_pair(h_a[a_row * (2 * kK) + kK + a_vec * 8 + 6], h_a[a_row * (2 * kK) + kK + a_vec * 8 + 7]),
        };
        const int base = tid * kWordsPerVec;
        for (int w = 0; w < kWordsPerVec; ++w) {
            if (h_a_stage0[base + w] != expect_stage0[w] || h_a_stage1[base + w] != expect_stage1[w]) {
                good = false;
                std::cout << "wait-window mismatch tid=" << tid
                          << " word=" << w
                          << " s0 expect=0x" << std::hex << expect_stage0[w]
                          << " got=0x" << h_a_stage0[base + w]
                          << " s1 expect=0x" << expect_stage1[w]
                          << " got=0x" << h_a_stage1[base + w]
                          << std::dec << std::endl;
                break;
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
    cudaFree(d_a_stage0);
    cudaFree(d_a_stage1);
    CudaCheckError();
    return good;
}

} // namespace

void tests(test_data &results) {
    std::cout << " ----- Starting ops/c500/mma/stage_async_probe tests! -----\n" << std::endl;
    run_probe(results);
    run_wait_window_probe(results);
}

} // namespace c500::mma::stage_async_probe

#endif
