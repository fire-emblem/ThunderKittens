#include "testing_flags.cuh"

#ifdef TEST_C500_MEMORY_RAW_GEMM_ASYNC

#include <vector>

#include "testing_commons.cuh"
#include "arch/c500/async_primitives.cuh"
#include "kittens.cuh"

namespace c500::memory::raw_gemm_async {

namespace {

using namespace kittens;

constexpr int kBlockThreads = 256;
constexpr int kLoadThreads = 128;
constexpr int kLoadBlocks = 2;
constexpr int kRowsA = 64;
constexpr int kColsA = 32;
constexpr int kRowsB = 32;
constexpr int kColsB = 64;
constexpr int kElemsA = kRowsA * kColsA;
constexpr int kElemsB = kRowsB * kColsB;
constexpr int kBytesA = kElemsA * sizeof(bf16);
constexpr int kBytesB = kElemsB * sizeof(bf16);

using global_tile_a = gl<bf16, 1, 1, 128, 32, st_bf<64, 32>>;
using global_tile_b = gl<bf16, 1, 1, 32, 128, st_bf<32, 64>>;

__device__ inline uint32_t raw_a_addr(uint8_t *smem, int tile_idx) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(smem + tile_idx * kBytesA));
}

__device__ inline uint32_t raw_b_addr(uint8_t *smem, int tile_idx) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(smem + tile_idx * kBytesB));
}

__global__ void raw_async_probe_kernel(const __grid_constant__ global_tile_a a_src,
                                       const __grid_constant__ global_tile_b b_src,
                                       uint16_t *a_out,
                                       uint16_t *b_out) {
    __shared__ KITTENS_ALIGN_AS(16) uint8_t a_smem[kLoadBlocks * kBytesA];
    __shared__ KITTENS_ALIGN_AS(16) uint8_t b_smem[kLoadBlocks * kBytesB];

    const int load_id = threadIdx.x / kLoadThreads;
    const int lane = threadIdx.x % kLoadThreads;

    constexpr int elem_per_memcpy = sizeof(float4) / sizeof(bf16);
    constexpr int a_memcpy_per_row = kColsA / elem_per_memcpy;
    constexpr int b_memcpy_per_row = kColsB / elem_per_memcpy;
    constexpr int a_total_calls = kElemsA / (kLoadThreads * elem_per_memcpy);
    constexpr int b_total_calls = kElemsB / (kLoadThreads * elem_per_memcpy);

    auto *a_ptr = reinterpret_cast<const bf16 *>(&a_src[coord<st_bf<64, 32>>{load_id, 0}.template unit_coord<2, 3>()]);
    auto *b_ptr = reinterpret_cast<const bf16 *>(&b_src[coord<st_bf<32, 64>>{0, load_id}.template unit_coord<2, 3>()]);
    const uint32_t a_shared = raw_a_addr(a_smem, load_id);
    const uint32_t b_shared = raw_b_addr(b_smem, load_id);

#pragma unroll
    for (int i = 0; i < a_total_calls; ++i) {
        const int load_idx = i * kLoadThreads + lane;
        const int row = load_idx / a_memcpy_per_row;
        const int col = (load_idx * elem_per_memcpy) % kColsA;
        const uint32_t dst = a_shared + (row * kColsA + col) * sizeof(bf16);
        kittens::arch::c500::detail::ldg_b128_bsm_no_pred(__cvta_shared_to_generic(dst),
                                                          a_ptr + row * a_src.template stride<2>() + col);
    }

#pragma unroll
    for (int i = 0; i < b_total_calls; ++i) {
        const int load_idx = i * kLoadThreads + lane;
        const int row = load_idx / b_memcpy_per_row;
        const int col = (load_idx * elem_per_memcpy) % kColsB;
        const uint32_t dst = b_shared + (row * kColsB + col) * sizeof(bf16);
        kittens::arch::c500::detail::ldg_b128_bsm_no_pred(__cvta_shared_to_generic(dst),
                                                          b_ptr + row * b_src.template stride<2>() + col);
    }

    kittens::arch::c500::wait(kittens::arch::c500::async_token<a_total_calls + b_total_calls>{});
    __syncthreads();

    for (int idx = threadIdx.x; idx < kLoadBlocks * kElemsA; idx += kBlockThreads) {
        const int tile = idx / kElemsA;
        const int elem = idx % kElemsA;
        bf16 value;
        kittens::move<bf16>::lds(value, raw_a_addr(a_smem, tile) + elem * sizeof(bf16));
        a_out[idx] = reinterpret_cast<const uint16_t &>(value);
    }

    for (int idx = threadIdx.x; idx < kLoadBlocks * kElemsB; idx += kBlockThreads) {
        const int tile = idx / kElemsB;
        const int elem = idx % kElemsB;
        bf16 value;
        kittens::move<bf16>::lds(value, raw_b_addr(b_smem, tile) + elem * sizeof(bf16));
        b_out[idx] = reinterpret_cast<const uint16_t &>(value);
    }
}

bool compare_exact(const std::string &label,
                   const std::vector<uint16_t> &got,
                   const std::vector<uint16_t> &expected,
                   test_data &results) {
    test_info info{label, test_result::FAILED};
    bool ok = true;
    for (size_t i = 0; i < got.size(); ++i) {
        if (got[i] != expected[i]) {
            ok = false;
            std::cout << "test `" << label << "` ----- ALERT! FAILED test `" << label << "` -----" << std::endl;
            std::cout << "first mismatch idx=" << i
                      << " expected=0x" << std::hex << expected[i]
                      << " actual=0x" << got[i]
                      << std::dec << std::endl;
            break;
        }
    }
    if (ok) {
        std::cout << "test `" << label << "` -- PASSED" << std::endl;
        info.result = test_result::PASSED;
    }
    results.push_back(info);
    return ok;
}

} // namespace

void tests(test_data &results) {
    std::cout << " ----- Starting ops/c500/memory/raw_gemm_async tests! -----\n" << std::endl;

    std::vector<bf16> h_a(kLoadBlocks * kElemsA);
    std::vector<bf16> h_b(kLoadBlocks * kElemsB);
    for (size_t i = 0; i < h_a.size(); ++i) h_a[i] = __float2bfloat16(static_cast<float>(i + 1));
    for (size_t i = 0; i < h_b.size(); ++i) h_b[i] = __float2bfloat16(static_cast<float>(10000 + i + 1));

    bf16 *d_a = nullptr;
    bf16 *d_b = nullptr;
    uint16_t *d_a_out = nullptr;
    uint16_t *d_b_out = nullptr;
    cudaMalloc(&d_a, h_a.size() * sizeof(bf16));
    cudaMalloc(&d_b, h_b.size() * sizeof(bf16));
    cudaMalloc(&d_a_out, h_a.size() * sizeof(uint16_t));
    cudaMalloc(&d_b_out, h_b.size() * sizeof(uint16_t));
    CudaCheckError();

    cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(bf16), cudaMemcpyHostToDevice);
    CudaCheckError();

    global_tile_a a_desc{d_a, nullptr, nullptr, nullptr, nullptr};
    global_tile_b b_desc{d_b, nullptr, nullptr, nullptr, nullptr};
    raw_async_probe_kernel<<<1, kBlockThreads>>>(a_desc, b_desc, d_a_out, d_b_out);
    CudaCheckError();

    std::vector<uint16_t> h_a_out(h_a.size(), 0);
    std::vector<uint16_t> h_b_out(h_b.size(), 0);
    std::vector<uint16_t> h_a_ref(h_a.size(), 0);
    std::vector<uint16_t> h_b_ref(h_b.size(), 0);
    cudaMemcpy(h_a_out.data(), d_a_out, h_a_out.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b_out.data(), d_b_out, h_b_out.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    CudaCheckError();

    for (size_t i = 0; i < h_a.size(); ++i) h_a_ref[i] = reinterpret_cast<const uint16_t &>(h_a[i]);
    for (int tile = 0; tile < kLoadBlocks; ++tile) {
        for (int row = 0; row < kRowsB; ++row) {
            for (int col = 0; col < kColsB; ++col) {
                const int out_idx = tile * kElemsB + row * kColsB + col;
                const int src_idx = row * (kLoadBlocks * kColsB) + tile * kColsB + col;
                h_b_ref[out_idx] = reinterpret_cast<const uint16_t &>(h_b[src_idx]);
            }
        }
    }

    compare_exact("c500_raw_gemm_async_a", h_a_out, h_a_ref, results);
    compare_exact("c500_raw_gemm_async_b", h_b_out, h_b_ref, results);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_a_out);
    cudaFree(d_b_out);
    CudaCheckError();
}

} // namespace c500::memory::raw_gemm_async

#endif
