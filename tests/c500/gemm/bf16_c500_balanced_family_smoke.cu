#include "testing_flags.cuh"

#ifdef TEST_C500_GEMM_BALANCED_FAMILY_SMOKE

#include "testing_commons.cuh"
#include "../../../kernels/gemm/common.cuh"
#include "arch/c500/gemm/bf16_contracts.cuh"
#include "arch/c500/gemm/families/bf16_balanced_128x128x128_stage4.cuh"

namespace c500::mma::balanced_family_smoke {

namespace {

constexpr int kM = 128;
constexpr int kN = 128;
constexpr int kK = 128;

using family = kittens::arch::c500::gemm::families::bf16_balanced_128x128x128_stage4;
using contracts = family::contracts;
using shared_tile_a = family::shared_tile_a;
using shared_tile_b = family::shared_tile_b;
using shared_tile_c = family::shared_tile_c;

template<int M, int K>
using a_gl = kittens::gl<kittens::bf16, 1, 1, M, K, shared_tile_a>;
template<int K, int N>
using b_gl = kittens::gl<kittens::bf16, 1, 1, K, N, shared_tile_b>;
template<int M, int N>
using c_gl = kittens::gl<kittens::bf16, 1, 1, M, N, shared_tile_c>;

struct gemm_globals {
    a_gl<kM, kK> a;
    b_gl<kK, kN> b;
    c_gl<kM, kN> c;
};

__global__ __launch_bounds__(contracts::kThreads)
void balanced_family_smoke_kernel(const __grid_constant__ gemm_globals g) {
    kittens::arch::c500::gemm::families::run_bf16_balanced_128x128x128_stage4<kM, kN, kK>(g);
}

bool run_smoke(test_data &results) {
    test_info info{"c500_gemm_bf16_balanced_family_128x128x128_smoke", test_result::FAILED};

    kittens::bf16 *d_a = nullptr;
    kittens::bf16 *d_b = nullptr;
    kittens::bf16 *d_c = nullptr;
    kittens::bf16 *d_ref = nullptr;

    cudaMalloc(&d_a, kM * kK * sizeof(kittens::bf16));
    cudaMalloc(&d_b, kK * kN * sizeof(kittens::bf16));
    cudaMalloc(&d_c, kM * kN * sizeof(kittens::bf16));
    cudaMalloc(&d_ref, kM * kN * sizeof(kittens::bf16));
    CudaCheckError();

    fill<__nv_bfloat16, FillMode::RANDOM>(reinterpret_cast<__nv_bfloat16 *>(d_a), kM * kK, 2024, -1.0f, 1.0f);
    fill<__nv_bfloat16, FillMode::RANDOM>(reinterpret_cast<__nv_bfloat16 *>(d_b), kK * kN, 2025, -1.0f, 1.0f);
    fill<__nv_bfloat16, FillMode::CONSTANT>(reinterpret_cast<__nv_bfloat16 *>(d_c), kM * kN, 0.0f);
    fill<__nv_bfloat16, FillMode::CONSTANT>(reinterpret_cast<__nv_bfloat16 *>(d_ref), kM * kN, 0.0f);
    CudaCheckError();

    reference_gemm<__nv_bfloat16, __nv_bfloat16, false>(reinterpret_cast<__nv_bfloat16 *>(d_ref),
                                                         reinterpret_cast<__nv_bfloat16 *>(d_a),
                                                         reinterpret_cast<__nv_bfloat16 *>(d_b),
                                                         kM, kN, kK);
    cudaDeviceSynchronize();
    CudaCheckError();

    gemm_globals g{
        a_gl<kM, kK>{d_a, nullptr, nullptr, nullptr, nullptr},
        b_gl<kK, kN>{d_b, nullptr, nullptr, nullptr, nullptr},
        c_gl<kM, kN>{d_c, nullptr, nullptr, nullptr, nullptr},
    };

    balanced_family_smoke_kernel<<<1, contracts::kThreads>>>(g);
    cudaDeviceSynchronize();
    CudaCheckError();

    std::vector<float> empty_input;
    std::vector<float> ref_out(kM * kN, 0.0f);
    std::vector<kittens::bf16> h_ref(kM * kN);
    cudaMemcpy(h_ref.data(), d_ref, kM * kN * sizeof(kittens::bf16), cudaMemcpyDeviceToHost);
    CudaCheckError();
    for (int i = 0; i < kM * kN; ++i) {
        ref_out[i] = __bfloat162float(h_ref[i]);
    }

    info.result = validate(d_a, d_c, empty_input, ref_out, info.label, kN, 0.02f);
    results.push_back(info);

    cudaFree(d_ref);
    cudaFree(d_c);
    cudaFree(d_b);
    cudaFree(d_a);
    CudaCheckError();
    return info.result == test_result::PASSED;
}

} // namespace

void tests(test_data &results) {
    std::cout << " ----- Starting ops/c500/mma/balanced_family_smoke tests! -----\n" << std::endl;
    run_smoke(results);
}

} // namespace c500::mma::balanced_family_smoke

#endif
