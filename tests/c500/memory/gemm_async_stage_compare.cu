#include "testing_flags.cuh"

#ifdef TEST_C500_MEMORY

#include <vector>

#include "testing_commons.cuh"
#include "arch/c500/async_primitives.cuh"
#include "kittens.cuh"

namespace c500::memory::gemm_async_stage_compare {

namespace {

using namespace kittens;

using load_group = kittens::group<2>;
using shared_tile_a = st_bf<64, 32>;
using shared_tile_b = st_bf<32, 64>;
using global_tile_a = gl<bf16, 1, 1, 128, 32, shared_tile_a>;
using global_tile_b = gl<bf16, 1, 1, 32, 128, shared_tile_b>;

constexpr int kBlockThreads = 256;
constexpr int kLoadBlocks = 2;
constexpr int kTileAElems = shared_tile_a::rows * shared_tile_a::cols;
constexpr int kTileBElems = shared_tile_b::rows * shared_tile_b::cols;

template<typename ST>
__device__ inline uint16_t logical_read_u16(const ST &tile, int row, int col) {
    const uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(&tile.data[0]));
    kittens::bf16 value;
    kittens::move<kittens::bf16>::lds(value, tile.idx(smem, {row, col}));
    return reinterpret_cast<const uint16_t &>(value);
}

template<typename SharedA, typename SharedB>
__device__ inline auto issue_async_pair(SharedA &a_dst,
                                        SharedB &b_dst,
                                        const global_tile_a &a_src,
                                        const global_tile_b &b_src,
                                        int load_id) {
    auto a_tok = kittens::arch::c500::load_async_tile<load_group::GROUP_THREADS, 2, true>(
        a_dst, a_src, kittens::coord<SharedA>{load_id, 0});
    auto b_tok = kittens::arch::c500::load_async_tile<load_group::GROUP_THREADS, 2, true>(
        b_dst, b_src, kittens::coord<SharedB>{0, load_id});
    return kittens::arch::c500::combine(a_tok, b_tok);
}

__global__ void gemm_stage_compare_kernel(const __grid_constant__ global_tile_a a_src,
                                          const __grid_constant__ global_tile_b b_src,
                                          uint16_t *a_sync_out,
                                          uint16_t *a_async_out,
                                          uint16_t *b_sync_out,
                                          uint16_t *b_async_out) {
    __shared__ shared_tile_a a_sync[kLoadBlocks][1];
    __shared__ shared_tile_a a_async[kLoadBlocks][1];
    __shared__ shared_tile_b b_sync[kLoadBlocks][1];
    __shared__ shared_tile_b b_async[kLoadBlocks][1];

    const int load_id = load_group::groupid();

    load_group::load<2, true>(a_sync[load_id][0], a_src, kittens::coord<shared_tile_a>{load_id, 0});
    load_group::load<2, true>(b_sync[load_id][0], b_src, kittens::coord<shared_tile_b>{0, load_id});

    auto tok = issue_async_pair(a_async[load_id][0], b_async[load_id][0], a_src, b_src, load_id);
    kittens::arch::c500::wait(tok);
    __syncthreads();

    for (int idx = threadIdx.x; idx < kTileAElems; idx += kBlockThreads) {
        const int row = idx / shared_tile_a::cols;
        const int col = idx % shared_tile_a::cols;
        a_sync_out[idx] = logical_read_u16(a_sync[0][0], row, col);
        a_async_out[idx] = logical_read_u16(a_async[0][0], row, col);
        a_sync_out[kTileAElems + idx] = logical_read_u16(a_sync[1][0], row, col);
        a_async_out[kTileAElems + idx] = logical_read_u16(a_async[1][0], row, col);
    }

    for (int idx = threadIdx.x; idx < kTileBElems; idx += kBlockThreads) {
        const int row = idx / shared_tile_b::cols;
        const int col = idx % shared_tile_b::cols;
        b_sync_out[idx] = logical_read_u16(b_sync[0][0], row, col);
        b_async_out[idx] = logical_read_u16(b_async[0][0], row, col);
        b_sync_out[kTileBElems + idx] = logical_read_u16(b_sync[1][0], row, col);
        b_async_out[kTileBElems + idx] = logical_read_u16(b_async[1][0], row, col);
    }
}

__global__ void gemm_stage_compare_serial_kernel(const __grid_constant__ global_tile_a a_src,
                                                 const __grid_constant__ global_tile_b b_src,
                                                 uint16_t *a_async_out,
                                                 uint16_t *b_async_out) {
    __shared__ shared_tile_a a_async[kLoadBlocks][1];
    __shared__ shared_tile_b b_async[kLoadBlocks][1];

    const int load_id = load_group::groupid();

    auto a_tok = kittens::arch::c500::load_async_tile<load_group::GROUP_THREADS, 2, true>(
        a_async[load_id][0], a_src, kittens::coord<shared_tile_a>{load_id, 0});
    kittens::arch::c500::wait(a_tok);
    __syncthreads();

    auto b_tok = kittens::arch::c500::load_async_tile<load_group::GROUP_THREADS, 2, true>(
        b_async[load_id][0], b_src, kittens::coord<shared_tile_b>{0, load_id});
    kittens::arch::c500::wait(b_tok);
    __syncthreads();

    for (int idx = threadIdx.x; idx < kTileAElems; idx += kBlockThreads) {
        const int row = idx / shared_tile_a::cols;
        const int col = idx % shared_tile_a::cols;
        a_async_out[idx] = logical_read_u16(a_async[0][0], row, col);
        a_async_out[kTileAElems + idx] = logical_read_u16(a_async[1][0], row, col);
    }

    for (int idx = threadIdx.x; idx < kTileBElems; idx += kBlockThreads) {
        const int row = idx / shared_tile_b::cols;
        const int col = idx % shared_tile_b::cols;
        b_async_out[idx] = logical_read_u16(b_async[0][0], row, col);
        b_async_out[kTileBElems + idx] = logical_read_u16(b_async[1][0], row, col);
    }
}

bool compare_buffers(const std::string &label,
                     const std::vector<uint16_t> &lhs,
                     const std::vector<uint16_t> &rhs,
                     test_info &info) {
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (lhs[i] != rhs[i]) {
            std::cout << "test `" << label << "` ----- ALERT! FAILED test `" << label << "` -----" << std::endl;
            std::cout << "first mismatch idx=" << i
                      << " expected=0x" << std::hex << lhs[i]
                      << " actual=0x" << rhs[i]
                      << std::dec << std::endl;
            info.result = test_result::FAILED;
            return false;
        }
    }
    std::cout << "test `" << label << "` -- PASSED" << std::endl;
    info.result = test_result::PASSED;
    return true;
}

} // namespace

void tests(test_data &results) {
    std::cout << " ----- Starting ops/c500/memory/gemm_async_stage_compare tests! -----\n" << std::endl;

    std::vector<bf16> h_a(kLoadBlocks * kTileAElems);
    std::vector<bf16> h_b(kLoadBlocks * kTileBElems);
    for (size_t i = 0; i < h_a.size(); ++i) {
        h_a[i] = __float2bfloat16(static_cast<float>(i + 1));
    }
    for (size_t i = 0; i < h_b.size(); ++i) {
        h_b[i] = __float2bfloat16(static_cast<float>(10000 + i + 1));
    }

    bf16 *d_a = nullptr;
    bf16 *d_b = nullptr;
    uint16_t *d_a_sync = nullptr;
    uint16_t *d_a_async = nullptr;
    uint16_t *d_b_sync = nullptr;
    uint16_t *d_b_async = nullptr;
    uint16_t *d_a_serial = nullptr;
    uint16_t *d_b_serial = nullptr;

    cudaMalloc(&d_a, h_a.size() * sizeof(bf16));
    cudaMalloc(&d_b, h_b.size() * sizeof(bf16));
    cudaMalloc(&d_a_sync, h_a.size() * sizeof(uint16_t));
    cudaMalloc(&d_a_async, h_a.size() * sizeof(uint16_t));
    cudaMalloc(&d_b_sync, h_b.size() * sizeof(uint16_t));
    cudaMalloc(&d_b_async, h_b.size() * sizeof(uint16_t));
    cudaMalloc(&d_a_serial, h_a.size() * sizeof(uint16_t));
    cudaMalloc(&d_b_serial, h_b.size() * sizeof(uint16_t));
    CudaCheckError();

    cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(bf16), cudaMemcpyHostToDevice);
    CudaCheckError();

    global_tile_a a_desc{d_a, nullptr, nullptr, nullptr, nullptr};
    global_tile_b b_desc{d_b, nullptr, nullptr, nullptr, nullptr};
    gemm_stage_compare_kernel<<<1, kBlockThreads>>>(a_desc, b_desc, d_a_sync, d_a_async, d_b_sync, d_b_async);
    gemm_stage_compare_serial_kernel<<<1, kBlockThreads>>>(a_desc, b_desc, d_a_serial, d_b_serial);
    CudaCheckError();

    std::vector<uint16_t> h_a_sync(h_a.size(), 0);
    std::vector<uint16_t> h_a_async(h_a.size(), 0);
    std::vector<uint16_t> h_b_sync(h_b.size(), 0);
    std::vector<uint16_t> h_b_async(h_b.size(), 0);
    std::vector<uint16_t> h_a_serial(h_a.size(), 0);
    std::vector<uint16_t> h_b_serial(h_b.size(), 0);

    cudaMemcpy(h_a_sync.data(), d_a_sync, h_a_sync.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_a_async.data(), d_a_async, h_a_async.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b_sync.data(), d_b_sync, h_b_sync.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b_async.data(), d_b_async, h_b_async.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_a_serial.data(), d_a_serial, h_a_serial.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b_serial.data(), d_b_serial, h_b_serial.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    CudaCheckError();

    test_info a_info{"c500_gemm_async_stage_compare_a", test_result::FAILED};
    compare_buffers(a_info.label, h_a_sync, h_a_async, a_info);
    results.push_back(a_info);

    test_info b_info{"c500_gemm_async_stage_compare_b", test_result::FAILED};
    compare_buffers(b_info.label, h_b_sync, h_b_async, b_info);
    results.push_back(b_info);

    test_info a_serial_info{"c500_gemm_async_stage_compare_a_serialized", test_result::FAILED};
    compare_buffers(a_serial_info.label, h_a_sync, h_a_serial, a_serial_info);
    results.push_back(a_serial_info);

    test_info b_serial_info{"c500_gemm_async_stage_compare_b_serialized", test_result::FAILED};
    compare_buffers(b_serial_info.label, h_b_sync, h_b_serial, b_serial_info);
    results.push_back(b_serial_info);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_a_sync);
    cudaFree(d_a_async);
    cudaFree(d_b_sync);
    cudaFree(d_b_async);
    cudaFree(d_a_serial);
    cudaFree(d_b_serial);
    CudaCheckError();
}

} // namespace c500::memory::gemm_async_stage_compare

#endif
