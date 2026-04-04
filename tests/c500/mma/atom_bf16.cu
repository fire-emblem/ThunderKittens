#include "testing_flags.cuh"

#ifdef TEST_C500_MMA_ATOM_BF16

#include "testing_commons.cuh"

namespace c500::mma::atom_bf16 {

namespace {

using atom = kittens::arch::c500::mma_bf16_16x16x16_fp32;
using native_a_fragment = kittens::arch::c500::fragment_a<atom>;
using native_b_fragment = kittens::arch::c500::fragment_b<atom>;
using native_c_fragment = kittens::arch::c500::fragment_c<atom>;

constexpr int kWaveLanes = kittens::WAVE_THREADS;
constexpr int kAccumulatorRegs = 4;
constexpr float kUniformInput = 1.0f;
constexpr float kUniformSeed = 0.5f;
constexpr float kExpectedAccumulator = 16.5f;

__host__ __device__ inline uint32_t pack_bf16_pair(float x, float y) {
    const kittens::bf16 lo = __float2bfloat16(x);
    const kittens::bf16 hi = __float2bfloat16(y);
    const kittens::bf16_2 pair{lo, hi};
    return *reinterpret_cast<const uint32_t *>(&pair);
}

__global__ void atom_uniform_smoke_kernel(float *out) {
    native_a_fragment a;
    native_b_fragment b;
    native_c_fragment c;
    native_c_fragment d;

    a.reg[0] = pack_bf16_pair(kUniformInput, kUniformInput);
    a.reg[1] = pack_bf16_pair(kUniformInput, kUniformInput);
    b.reg[0] = pack_bf16_pair(kUniformInput, kUniformInput);
    b.reg[1] = pack_bf16_pair(kUniformInput, kUniformInput);

#pragma unroll
    for (int i = 0; i < kAccumulatorRegs; ++i) {
        c.reg[i] = kUniformSeed;
    }

    kittens::arch::c500::mma<atom>(d, a, b, c);

    const int lane = threadIdx.x;
#pragma unroll
    for (int i = 0; i < kAccumulatorRegs; ++i) {
        out[lane * kAccumulatorRegs + i] = d.reg[i];
    }
}

bool run_atom_uniform_smoke(test_data &results) {
    test_info info{"c500_mma_atom_bf16_contract_smoke", test_result::FAILED};

    float *d_out = nullptr;
    std::vector<float> h_out(kWaveLanes * kAccumulatorRegs, 0.0f);
    cudaMalloc(&d_out, h_out.size() * sizeof(float));
    CudaCheckError();

    atom_uniform_smoke_kernel<<<1, kWaveLanes>>>(d_out);
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
    }
    else {
        std::cout << " ----- ALERT! FAILED test `" << info.label << "` -----" << std::endl;
    }

    results.push_back(info);
    cudaFree(d_out);
    CudaCheckError();
    return good;
}

} // namespace

namespace atom_bf16_contract {

static_assert(atom::M == 16 && atom::N == 16 && atom::K == 16,
              "The first-wave C500 atom scaffold is fixed to 16x16x16.");
static_assert(atom::wave_size == kittens::WAVE_THREADS,
              "C500 atom scaffolds assume native wave-sized execution.");
static_assert(std::is_same_v<typename atom::a_scalar, kittens::bf16>,
              "The first C500 atom scaffold covers bf16 inputs.");
static_assert(std::is_same_v<typename atom::b_scalar, kittens::bf16>,
              "The first C500 atom scaffold covers bf16 inputs.");
static_assert(std::is_same_v<typename atom::c_scalar, float>,
              "The first C500 atom scaffold covers fp32 accumulation.");
static_assert(sizeof(native_a_fragment) == 2 * sizeof(uint32_t),
              "The first native bf16 atom test expects two packed A registers per lane.");
static_assert(sizeof(native_b_fragment) == 2 * sizeof(uint32_t),
              "The first native bf16 atom test expects two packed B registers per lane.");
static_assert(sizeof(native_c_fragment) == kAccumulatorRegs * sizeof(float),
              "The first native bf16 atom test expects four fp32 accumulator scalars per lane.");
static_assert(requires(
                  native_c_fragment &dst,
                  const native_a_fragment &a,
                  const native_b_fragment &b,
                  const native_c_fragment &src) {
                  kittens::arch::c500::mma<atom>(dst, a, b, src);
              },
              "The atom probe keeps the native mma entrypoint callable for the frozen bf16 contract.");

} // namespace atom_bf16_contract

void tests(test_data &results) {
    std::cout << " ----- Starting ops/c500/mma/atom_bf16 tests! -----\n" << std::endl;
    run_atom_uniform_smoke(results);
    std::cout << "INFO: C500 bf16 atom coverage currently validates one uniform native atom path without freezing lane ownership or per-lane export order.\n" << std::endl;
}

} // namespace c500::mma::atom_bf16

#endif
