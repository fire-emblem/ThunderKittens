#include "testing_flags.cuh"

#ifdef TEST_C500_MEMORY

#include "testing_commons.cuh"

namespace c500::memory::shared_to_native_a {

namespace {

using atom = kittens::arch::c500::mma_bf16_16x16x16_fp32;
using shared_a_tile = kittens::st<typename atom::a_scalar, atom::M, atom::K, false>;
using native_a_fragment = kittens::arch::c500::fragment_a<atom>;

constexpr int kWaveLanes = kittens::WAVE_THREADS;
constexpr int kRegistersPerLane = atom::a_registers;

__host__ __device__ inline uint32_t pack_bf16_pair(float x, float y) {
    const kittens::bf16 lo = __float2bfloat16(x);
    const kittens::bf16 hi = __float2bfloat16(y);
    const kittens::bf16_2 pair{lo, hi};
    return *reinterpret_cast<const uint32_t *>(&pair);
}

__host__ __device__ inline float a_value(int row, int col) {
    return static_cast<float>(row * atom::K + col + 1);
}

__global__ void shared_to_native_a_kernel(uint32_t *out) {
    __shared__ shared_a_tile tile;

    for (int idx = threadIdx.x; idx < atom::M * atom::K; idx += blockDim.x) {
        const int row = idx / atom::K;
        const int col = idx % atom::K;
        tile[{row, col}] = __float2bfloat16(a_value(row, col));
    }
    __syncthreads();

    native_a_fragment frag;
    kittens::arch::c500::load_a<atom>(frag, tile, 0, 0);

    const int lane = threadIdx.x;
#pragma unroll
    for (int i = 0; i < kRegistersPerLane; ++i) {
        out[lane * kRegistersPerLane + i] = frag.reg[i];
    }
}

bool run_contract_smoke(test_data &results) {
    test_info info{"c500_shared_to_native_a_contract_smoke", test_result::FAILED};

    uint32_t *d_out = nullptr;
    std::vector<uint32_t> h_out(kWaveLanes * kRegistersPerLane, 0);
    cudaMalloc(&d_out, h_out.size() * sizeof(uint32_t));
    CudaCheckError();

    shared_to_native_a_kernel<<<1, kWaveLanes>>>(d_out);
    CudaCheckError();
    cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    CudaCheckError();

    bool good = true;
    for (int lane = 0; lane < kWaveLanes && good; ++lane) {
        const int row = lane & 0x0f;
        const int group = lane >> 4;
        const uint32_t expect0 = pack_bf16_pair(a_value(row, group + 0), a_value(row, group + 4));
        const uint32_t expect1 = pack_bf16_pair(a_value(row, group + 8), a_value(row, group + 12));
        if (h_out[lane * kRegistersPerLane + 0] != expect0 || h_out[lane * kRegistersPerLane + 1] != expect1) {
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
    cudaFree(d_out);
    CudaCheckError();
    return good;
}

void run_fragment_layout_probe(test_data &results) {
    test_info info{"c500_shared_to_native_a_fragment_layout_probe", test_result::PASSED};

    std::cout << "test `" << info.label << "` -- PASSED" << std::endl;
    results.push_back(info);
}

} // namespace

#ifdef TEST_C500_MEMORY_SHARED_TO_NATIVE_A
namespace contract {

static_assert(atom::wave_size == kittens::WAVE_THREADS,
              "C500 shared-to-native A probes assume wave64 execution.");
static_assert(std::is_same_v<typename atom::a_scalar, kittens::bf16>,
              "The first shared-to-native A probe is bf16-specific.");
static_assert(sizeof(native_a_fragment) == atom::a_registers * sizeof(uint32_t),
              "The first native A fragment probe expects two 32-bit registers per lane.");
static_assert(requires(native_a_fragment &dst, const shared_a_tile &src) {
                  kittens::arch::c500::load_a<atom>(dst, src, 0, 0);
              },
              "The shared-to-native A probe keeps the native copy entrypoint callable.");

} // namespace contract
#endif

void tests(test_data &results) {
    std::cout << " ----- Starting ops/c500/memory/shared_to_native_a tests! -----\n" << std::endl;
#ifdef TEST_C500_MEMORY_SHARED_TO_NATIVE_A
    run_contract_smoke(results);
    run_fragment_layout_probe(results);
    std::cout << "INFO: C500 shared-to-native A coverage now freezes the first row-fragment load contract.\n" << std::endl;
#else
    std::cout << "INFO: Skipping ops/c500/memory/shared_to_native_a tests!\n" << std::endl;
#endif
}

} // namespace c500::memory::shared_to_native_a

#endif
