#include "testing_flags.cuh"

#ifdef TEST_C500_MEMORY_ASYNC_BSM

#include <array>

#include "testing_commons.cuh"

#include "arch/c500/async_primitives.cuh"

namespace c500::memory::global_to_shared_async_bsm {

namespace {

constexpr int kWaveLanes = kittens::WAVE_THREADS;
constexpr int kWords = 4;

__global__ void global_to_shared_async_bsm_kernel(const uint32_t *src, uint32_t *dst) {
    __shared__ uint32_t smem[kWords];

    if (threadIdx.x < kWords) {
        smem[threadIdx.x] = 0xdead0000u + static_cast<uint32_t>(threadIdx.x);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        kittens::arch::c500::detail::ldg_b128_bsm_no_pred(
            smem,
            src
        );
        kittens::arch::c500::wait(kittens::arch::c500::async_token<1>{});
    }
    __syncthreads();

    if (threadIdx.x < kWords) {
        dst[threadIdx.x] = smem[threadIdx.x];
    }
}

bool run_smoke(test_data &results) {
    test_info info{"c500_global_to_shared_async_bsm_smoke", test_result::FAILED};

    const std::array<uint32_t, kWords> h_src{
        0x01020304u, 0x11121314u, 0x21222324u, 0x31323334u
    };
    std::array<uint32_t, kWords> h_dst{};

    uint32_t *d_src = nullptr;
    uint32_t *d_dst = nullptr;
    cudaMalloc(&d_src, sizeof(h_src));
    cudaMalloc(&d_dst, sizeof(h_dst));
    CudaCheckError();
    cudaMemcpy(d_src, h_src.data(), sizeof(h_src), cudaMemcpyHostToDevice);
    CudaCheckError();

    global_to_shared_async_bsm_kernel<<<1, kWaveLanes>>>(d_src, d_dst);
    CudaCheckError();
    cudaMemcpy(h_dst.data(), d_dst, sizeof(h_dst), cudaMemcpyDeviceToHost);
    CudaCheckError();

    const bool good = (h_dst == h_src);

    std::cout << "test `" << info.label << "`";
    if (good) {
        std::cout << " -- PASSED" << std::endl;
        info.result = test_result::PASSED;
    } else {
        std::cout << " ----- ALERT! FAILED test `" << info.label << "` -----" << std::endl;
        std::cout << "expected:";
        for (uint32_t v : h_src) {
            std::cout << " 0x" << std::hex << v;
        }
        std::cout << "\nactual:  ";
        for (uint32_t v : h_dst) {
            std::cout << " 0x" << std::hex << v;
        }
        std::cout << std::dec << std::endl;
    }

    results.push_back(info);
    cudaFree(d_src);
    cudaFree(d_dst);
    CudaCheckError();
    return good;
}

} // namespace

void tests(test_data &results) {
    std::cout << " ----- Starting ops/c500/memory/global_to_shared_async_bsm tests! -----\n" << std::endl;
    run_smoke(results);
    std::cout << "INFO: C500 async gmem->shared probe currently freezes a single 16B builtin transaction plus explicit wait.\n" << std::endl;
}

} // namespace c500::memory::global_to_shared_async_bsm

#endif
