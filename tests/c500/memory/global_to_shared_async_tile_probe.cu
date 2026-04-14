#include "testing_flags.cuh"

#ifdef TEST_C500_MEMORY

#include <array>
#include <vector>

#include "testing_commons.cuh"
#include "arch/c500/async_primitives.cuh"

namespace c500::memory::global_to_shared_async_tile_probe {

namespace {

using shared_tile = kittens::st_bf<64, 32>;
using global_tile = kittens::gl<kittens::bf16, 1, 1, 128, 32, shared_tile>;
using shared_tile_b = kittens::st_bf<32, 64>;
using global_tile_b = kittens::gl<kittens::bf16, 1, 1, 32, 128, shared_tile_b>;

constexpr int kBlockThreads = 256;
constexpr int kLoadThreads = 128;
constexpr int kWaveThreads = 64;
constexpr int kTileElems = 64 * 32;
constexpr int kTileBElems = 32 * 64;
constexpr int kGlobalBCols = 128;

template<typename ST>
__device__ inline uint16_t logical_read_u16(const ST &tile, int row, int col) {
    const uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(&tile.data[0]));
    kittens::bf16 value;
    kittens::move<kittens::bf16>::lds(value, tile.idx(smem, {row, col}));
    return reinterpret_cast<const uint16_t &>(value);
}

__global__ void async_tile_probe_kernel(const __grid_constant__ global_tile src, uint16_t *dst) {
    __shared__ shared_tile tiles[2][1];

    const int load_id = threadIdx.x / kLoadThreads;
    auto tok = kittens::arch::c500::load_async_tile<kLoadThreads, 2, true>(
        tiles[load_id][0], src, kittens::coord<shared_tile>{load_id, 0});
    kittens::arch::c500::wait(tok);
    __syncthreads();

    for (int idx = threadIdx.x; idx < kTileElems; idx += kBlockThreads) {
        const int row = idx / shared_tile::cols;
        const int col = idx % shared_tile::cols;
        dst[idx] = logical_read_u16(tiles[0][0], row, col);
        dst[kTileElems + idx] = logical_read_u16(tiles[1][0], row, col);
    }
}

__global__ void async_tile_probe_wave_kernel(const __grid_constant__ global_tile src, uint16_t *dst) {
    __shared__ shared_tile tiles[2][1];

    const int warp = threadIdx.x / kWaveThreads;
    if (warp < 2) {
        auto tok = kittens::arch::c500::load_async_tile<kWaveThreads, 2, true>(
            tiles[warp][0], src, kittens::coord<shared_tile>{warp, 0});
        kittens::arch::c500::wait(tok);
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < kTileElems; idx += kBlockThreads) {
        const int row = idx / shared_tile::cols;
        const int col = idx % shared_tile::cols;
        dst[idx] = logical_read_u16(tiles[0][0], row, col);
        dst[kTileElems + idx] = logical_read_u16(tiles[1][0], row, col);
    }
}

__global__ void async_tile_b_probe_kernel(const __grid_constant__ global_tile_b src, uint16_t *dst) {
    __shared__ shared_tile_b tiles[2][1];

    const int warp = threadIdx.x / kWaveThreads;
    if (warp < 2) {
        auto tok = kittens::arch::c500::load_async_tile<kWaveThreads, 2, true>(
            tiles[warp][0], src, kittens::coord<shared_tile_b>{0, warp});
        kittens::arch::c500::wait(tok);
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < kTileBElems; idx += kBlockThreads) {
        const int row = idx / shared_tile_b::cols;
        const int col = idx % shared_tile_b::cols;
        dst[idx] = logical_read_u16(tiles[0][0], row, col);
        dst[kTileBElems + idx] = logical_read_u16(tiles[1][0], row, col);
    }
}

bool run_probe(test_data &results, bool wave_mode) {
    test_info info{wave_mode ? "c500_global_to_shared_async_tile_probe_wave64"
                             : "c500_global_to_shared_async_tile_probe_group128",
                   test_result::FAILED};

    std::vector<kittens::bf16> h_src(2 * kTileElems);
    for (int i = 0; i < 2 * kTileElems; ++i) {
        h_src[i] = __float2bfloat16(static_cast<float>(i + 1));
    }
    std::vector<uint16_t> h_dst(2 * kTileElems, 0);

    kittens::bf16 *d_src = nullptr;
    uint16_t *d_dst = nullptr;
    cudaMalloc(&d_src, h_src.size() * sizeof(kittens::bf16));
    cudaMalloc(&d_dst, h_dst.size() * sizeof(uint16_t));
    CudaCheckError();

    cudaMemcpy(d_src, h_src.data(), h_src.size() * sizeof(kittens::bf16), cudaMemcpyHostToDevice);
    CudaCheckError();

    global_tile src_desc{d_src, nullptr, nullptr, nullptr, nullptr};
    if (wave_mode) {
        async_tile_probe_wave_kernel<<<1, kBlockThreads>>>(src_desc, d_dst);
    } else {
        async_tile_probe_kernel<<<1, kBlockThreads>>>(src_desc, d_dst);
    }
    CudaCheckError();
    cudaMemcpy(h_dst.data(), d_dst, h_dst.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    CudaCheckError();

    bool good = true;
    for (int i = 0; i < 2 * kTileElems; ++i) {
        const uint16_t expected = reinterpret_cast<const uint16_t &>(h_src[i]);
        if (h_dst[i] != expected) {
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
        for (int i = 0; i < 2 * kTileElems; ++i) {
            const uint16_t expected = reinterpret_cast<const uint16_t &>(h_src[i]);
            if (h_dst[i] != expected) {
                std::cout << "first mismatch idx=" << i
                          << " expected=0x" << std::hex << expected
                          << " actual=0x" << h_dst[i]
                          << std::dec << std::endl;
                break;
            }
        }
    }

    results.push_back(info);
    cudaFree(d_src);
    cudaFree(d_dst);
    CudaCheckError();
    return good;
}

bool run_b_probe(test_data &results) {
    test_info info{"c500_global_to_shared_async_tile_probe_b_wave64", test_result::FAILED};

    std::vector<kittens::bf16> h_src(2 * kTileBElems);
    for (int i = 0; i < 2 * kTileBElems; ++i) {
        h_src[i] = __float2bfloat16(static_cast<float>(i + 1));
    }
    std::vector<uint16_t> h_dst(2 * kTileBElems, 0);

    kittens::bf16 *d_src = nullptr;
    uint16_t *d_dst = nullptr;
    cudaMalloc(&d_src, h_src.size() * sizeof(kittens::bf16));
    cudaMalloc(&d_dst, h_dst.size() * sizeof(uint16_t));
    CudaCheckError();

    cudaMemcpy(d_src, h_src.data(), h_src.size() * sizeof(kittens::bf16), cudaMemcpyHostToDevice);
    CudaCheckError();

    global_tile_b src_desc{d_src, nullptr, nullptr, nullptr, nullptr};
    async_tile_b_probe_kernel<<<1, kBlockThreads>>>(src_desc, d_dst);
    CudaCheckError();
    cudaMemcpy(h_dst.data(), d_dst, h_dst.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    CudaCheckError();

    bool good = true;
    for (int tile = 0; tile < 2 && good; ++tile) {
        for (int row = 0; row < shared_tile_b::rows && good; ++row) {
            for (int col = 0; col < shared_tile_b::cols; ++col) {
                const int out_idx = tile * kTileBElems + row * shared_tile_b::cols + col;
                const int src_idx = row * kGlobalBCols + tile * shared_tile_b::cols + col;
                const uint16_t expected = reinterpret_cast<const uint16_t &>(h_src[src_idx]);
                if (h_dst[out_idx] != expected) {
                    good = false;
                    std::cout << "test `" << info.label << "` ----- ALERT! FAILED test `" << info.label << "` -----" << std::endl;
                    std::cout << "first mismatch idx=" << out_idx
                              << " expected=0x" << std::hex << expected
                              << " actual=0x" << h_dst[out_idx]
                              << std::dec << std::endl;
                    break;
                }
            }
        }
    }
    if (good) {
        std::cout << "test `" << info.label << "` -- PASSED" << std::endl;
        info.result = test_result::PASSED;
    }

    results.push_back(info);
    cudaFree(d_src);
    cudaFree(d_dst);
    CudaCheckError();
    return good;
}

} // namespace

void tests(test_data &results) {
    std::cout << " ----- Starting ops/c500/memory/global_to_shared_async_tile_probe tests! -----\n" << std::endl;
    run_probe(results, false);
    run_probe(results, true);
    run_b_probe(results);
}

} // namespace c500::memory::global_to_shared_async_tile_probe

#endif
