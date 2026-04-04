#include "testing_flags.cuh"

#ifdef TEST_C500_GEMM_FRAGMENT_PROBE

#include <bit>
#include <cmath>
#include <type_traits>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "testing_commons.cuh"

#include "arch/c500/fragments.cuh"
#include "arch/c500/layouts/accumulator_export.cuh"
#include "arch/c500/mma.cuh"

namespace c500::mma::fragment_probe {

namespace {

using internal_atom = kittens::arch::c500::mma_bf16_16x16x16_fp32;
using public_atom = kittens::arch::c500::bf16_mma_atom;
using shared_a_tile = kittens::st<typename internal_atom::a_scalar, internal_atom::M, internal_atom::K, false>;
using shared_b_tile = kittens::st<typename internal_atom::b_scalar, internal_atom::K, internal_atom::N, false>;
using export_map = kittens::arch::c500::gemm::accumulator_tile_map;

constexpr int kWaveLanes = kittens::WAVE_THREADS;
constexpr int kARegistersPerLane = internal_atom::a_registers;
constexpr int kBRegistersPerLane = internal_atom::b_registers;
constexpr int kAccumulatorRegisters = public_atom::c_registers;
constexpr float kUniformInput = 1.0f;
constexpr float kAccumulatorSeed = 0.5f;
constexpr float kExpectedAccumulator = 16.5f;

static_assert(std::is_same_v<typename public_atom::a_scalar, kittens::bf16>);
static_assert(std::is_same_v<typename public_atom::b_scalar, kittens::bf16>);
static_assert(std::is_same_v<typename public_atom::c_scalar, float>);
static_assert(sizeof(kittens::arch::c500::fragment_a<public_atom>) == 2 * sizeof(uint32_t));
static_assert(sizeof(kittens::arch::c500::fragment_b<public_atom>) == 2 * sizeof(uint32_t));
static_assert(sizeof(kittens::arch::c500::fragment_c<public_atom>) == 4 * sizeof(float));
static_assert(export_map::wave_base_row(3) == 64);
static_assert(export_map::wave_base_col(3) == 64);
static_assert(export_map::atom_col(63, 3) == 15);

__host__ __device__ inline uint32_t pack_bf16_pair(float x, float y) {
    const kittens::bf16 lo = __float2bfloat16(x);
    const kittens::bf16 hi = __float2bfloat16(y);
    const kittens::bf16_2 pair{lo, hi};
    return *reinterpret_cast<const uint32_t *>(&pair);
}

__host__ __device__ inline float a_value(int row, int col) {
    return static_cast<float>(row * internal_atom::K + col + 1);
}

__host__ __device__ inline float b_value(int row, int col) {
    return static_cast<float>(1000 + row * internal_atom::N + col);
}

__global__ void fragment_feed_probe_kernel(uint32_t *a_out, uint32_t *b_out) {
    __shared__ shared_a_tile a_tile;
    __shared__ shared_b_tile b_tile;

    for (int idx = threadIdx.x; idx < internal_atom::M * internal_atom::K; idx += blockDim.x) {
        const int row = idx / internal_atom::K;
        const int col = idx % internal_atom::K;
        a_tile[{row, col}] = __float2bfloat16(a_value(row, col));
    }

    for (int idx = threadIdx.x; idx < internal_atom::K * internal_atom::N; idx += blockDim.x) {
        const int row = idx / internal_atom::N;
        const int col = idx % internal_atom::N;
        b_tile[{row, col}] = __float2bfloat16(b_value(row, col));
    }
    __syncthreads();

    kittens::arch::c500::fragment_a<internal_atom> raw_a{};
    kittens::arch::c500::fragment_b<internal_atom> raw_b{};
    kittens::arch::c500::load_a<internal_atom>(raw_a, a_tile, 0, 0);
    kittens::arch::c500::load_b<internal_atom>(raw_b, b_tile, 0, 0);

    const int lane = threadIdx.x;
#pragma unroll
    for (int i = 0; i < kARegistersPerLane; ++i) {
        a_out[lane * kARegistersPerLane + i] = raw_a.reg[i];
    }
#pragma unroll
    for (int i = 0; i < kBRegistersPerLane; ++i) {
        b_out[lane * kBRegistersPerLane + i] = raw_b.reg[i];
    }
}

__global__ void fragment_mma_probe_kernel(float *out) {
    kittens::arch::c500::fragment_a<public_atom> a{};
    kittens::arch::c500::fragment_b<public_atom> b{};
    kittens::arch::c500::fragment_c<public_atom> c{};

    a.reg[0] = pack_bf16_pair(kUniformInput, kUniformInput);
    a.reg[1] = pack_bf16_pair(kUniformInput, kUniformInput);
    b.reg[0] = pack_bf16_pair(kUniformInput, kUniformInput);
    b.reg[1] = pack_bf16_pair(kUniformInput, kUniformInput);

#pragma unroll
    for (int i = 0; i < kAccumulatorRegisters; ++i) {
        c.reg[i] = kAccumulatorSeed;
    }

    const auto d = kittens::arch::c500::mma(public_atom{}, a, b, c);
    const int lane = threadIdx.x;

#pragma unroll
    for (int i = 0; i < kAccumulatorRegisters; ++i) {
        out[lane * kAccumulatorRegisters + i] = d.reg[i];
    }
}

bool run_fragment_feed_contract(test_data &results) {
    test_info info{"c500_gemm_bf16_fragment_feed_contract", test_result::FAILED};

    uint32_t *d_a = nullptr;
    uint32_t *d_b = nullptr;
    std::vector<uint32_t> h_a(kWaveLanes * kARegistersPerLane, 0);
    std::vector<uint32_t> h_b(kWaveLanes * kBRegistersPerLane, 0);
    cudaMalloc(&d_a, h_a.size() * sizeof(uint32_t));
    cudaMalloc(&d_b, h_b.size() * sizeof(uint32_t));
    CudaCheckError();

    fragment_feed_probe_kernel<<<1, kWaveLanes>>>(d_a, d_b);
    CudaCheckError();
    cudaMemcpy(h_a.data(), d_a, h_a.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b.data(), d_b, h_b.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    CudaCheckError();

    bool good = true;
    for (int lane = 0; lane < kWaveLanes && good; ++lane) {
        const int row = lane & 0x0f;
        const int group = lane >> 4;
        const int col = lane & 0x0f;
        const uint32_t expect_a0 = pack_bf16_pair(a_value(row, group + 0), a_value(row, group + 4));
        const uint32_t expect_a1 = pack_bf16_pair(a_value(row, group + 8), a_value(row, group + 12));
        const uint32_t expect_b0 = pack_bf16_pair(b_value(group + 0, col), b_value(group + 4, col));
        const uint32_t expect_b1 = pack_bf16_pair(b_value(group + 8, col), b_value(group + 12, col));
        if (h_a[lane * kARegistersPerLane + 0] != expect_a0 ||
            h_a[lane * kARegistersPerLane + 1] != expect_a1 ||
            h_b[lane * kBRegistersPerLane + 0] != expect_b0 ||
            h_b[lane * kBRegistersPerLane + 1] != expect_b1) {
            good = false;
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

bool run_fragment_mma_contract(test_data &results) {
    test_info info{"c500_gemm_bf16_fragment_mma_contract", test_result::FAILED};

    float *d_out = nullptr;
    std::vector<float> h_out(kWaveLanes * kAccumulatorRegisters, 0.0f);
    cudaMalloc(&d_out, h_out.size() * sizeof(float));
    CudaCheckError();

    fragment_mma_probe_kernel<<<1, kWaveLanes>>>(d_out);
    CudaCheckError();
    cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost);
    CudaCheckError();

    bool good = true;
    for (float value : h_out) {
        if (std::abs(value - kExpectedAccumulator) > 1e-3f) {
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
    cudaFree(d_out);
    CudaCheckError();
    return good;
}

bool run_export_map_contract(test_data &results) {
    test_info info{"c500_gemm_bf16_accumulator_export_map", test_result::FAILED};

    const bool good =
        export_map::wave_row(0) == 0 &&
        export_map::wave_col(0) == 0 &&
        export_map::wave_row(3) == 1 &&
        export_map::wave_col(3) == 1 &&
        export_map::wave_base_row(2) == 64 &&
        export_map::wave_base_col(1) == 64 &&
        export_map::atom_row(63) == 15 &&
        export_map::atom_col(0, 0) == 0 &&
        export_map::atom_col(31, 3) == 7 &&
        export_map::atom_col(63, 3) == 15;

    std::cout << "test `" << info.label << "`";
    if (good) {
        std::cout << " -- PASSED" << std::endl;
        info.result = test_result::PASSED;
    } else {
        std::cout << " ----- ALERT! FAILED test `" << info.label << "` -----" << std::endl;
    }

    results.push_back(info);
    return good;
}

} // namespace

void tests(test_data &results) {
    std::cout << " ----- Starting ops/c500/mma/fragment_probe tests! -----\n" << std::endl;
    run_fragment_feed_contract(results);
    run_fragment_mma_contract(results);
    run_export_map_contract(results);
    std::cout << "INFO: C500 fragment coverage now freezes the native shared->fragment feed, public mma contract, and accumulator export map.\n" << std::endl;
}

} // namespace c500::mma::fragment_probe

#endif
