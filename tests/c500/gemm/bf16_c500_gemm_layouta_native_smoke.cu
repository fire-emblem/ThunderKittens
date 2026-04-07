#include "testing_flags.cuh"

#ifdef TEST_C500_GEMM_LAYOUTA_NATIVE_SMOKE

#include "testing_commons.cuh"
#include "../../../kernels/gemm/common.cuh"
#include "arch/c500/gemm/bf16_mainloop.cuh"

namespace c500::mma::gemm_layouta_native_smoke {

namespace {

constexpr int kM = 128;
constexpr int kN = 128;

using contracts = kittens::arch::c500::gemm::bf16_mainloop_family::contracts;
using shared_tileA = kittens::arch::c500::gemm::bf16_shared_tile_a;
using shared_tileC = kittens::arch::c500::gemm::bf16_shared_tile_c;

template<int M, int K>
using a_gl = kittens::gl<kittens::bf16, 1, 1, M, K, shared_tileA>;
template<int N, int K>
using b_layouta_gl = kittens::gl<kittens::bf16, 1, 1, N, K>;
template<int M, int N>
using c_gl = kittens::gl<kittens::bf16, 1, 1, M, N, shared_tileC>;

template<int K>
struct gemm_globals {
    a_gl<kM, K> a;
    b_layouta_gl<kN, K> b;
    c_gl<kM, kN> c;
};

template<int K>
__global__ __launch_bounds__(contracts::kThreads)
void gemm_smoke_kernel(const __grid_constant__ gemm_globals<K> g) {
    kittens::arch::c500::gemm::run_bf16_mainloop_layouta<kM, kN, K>(g);
}

template<int K>
bool run_smoke_case(test_data &results, const std::string &label, uint64_t seed_b) {
    test_info info{label, test_result::FAILED};

    kittens::bf16 *d_a = nullptr;
    kittens::bf16 *d_b = nullptr;
    kittens::bf16 *d_b_layouta = nullptr;
    kittens::bf16 *d_c = nullptr;
    kittens::bf16 *d_ref = nullptr;

    cudaMalloc(&d_a, kM * K * sizeof(kittens::bf16));
    cudaMalloc(&d_b, K * kN * sizeof(kittens::bf16));
    cudaMalloc(&d_b_layouta, kN * K * sizeof(kittens::bf16));
    cudaMalloc(&d_c, kM * kN * sizeof(kittens::bf16));
    cudaMalloc(&d_ref, kM * kN * sizeof(kittens::bf16));
    CudaCheckError();

    fill<__nv_bfloat16, FillMode::RANDOM>(reinterpret_cast<__nv_bfloat16 *>(d_a), kM * K, 2024, -1.0f, 1.0f);
    fill<__nv_bfloat16, FillMode::RANDOM>(reinterpret_cast<__nv_bfloat16 *>(d_b), K * kN, seed_b, -1.0f, 1.0f);
    fill<__nv_bfloat16, FillMode::CONSTANT>(reinterpret_cast<__nv_bfloat16 *>(d_c), kM * kN, 0.0f);
    fill<__nv_bfloat16, FillMode::CONSTANT>(reinterpret_cast<__nv_bfloat16 *>(d_ref), kM * kN, 0.0f);
    CudaCheckError();

    std::vector<kittens::bf16> h_b_layouta(kN * K);
    std::vector<kittens::bf16> h_b(K * kN);
    cudaMemcpy(h_b.data(), d_b, K * kN * sizeof(kittens::bf16), cudaMemcpyDeviceToHost);
    CudaCheckError();
    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < kN; ++n) {
            h_b_layouta[n * K + k] = h_b[k * kN + n];
        }
    }
    cudaMemcpy(d_b_layouta, h_b_layouta.data(), kN * K * sizeof(kittens::bf16), cudaMemcpyHostToDevice);
    CudaCheckError();

    reference_gemm<__nv_bfloat16, __nv_bfloat16, false>(reinterpret_cast<__nv_bfloat16 *>(d_ref),
                                                        reinterpret_cast<__nv_bfloat16 *>(d_a),
                                                        reinterpret_cast<__nv_bfloat16 *>(d_b),
                                                        kM, kN, K);
    cudaDeviceSynchronize();
    CudaCheckError();

    gemm_globals<K> g{
        a_gl<kM, K>{d_a, nullptr, nullptr, nullptr, nullptr},
        b_layouta_gl<kN, K>{d_b_layouta, nullptr, nullptr, nullptr, nullptr},
        c_gl<kM, kN>{d_c, nullptr, nullptr, nullptr, nullptr}
    };

    gemm_smoke_kernel<K><<<1, contracts::kThreads>>>(g);
    cudaDeviceSynchronize();
    CudaCheckError();

    std::vector<float> empty_input;
    std::vector<float> ref_out(kM * kN, 0.0f);
    std::vector<kittens::bf16> h_ref(kM * kN);
    std::vector<kittens::bf16> h_out(kM * kN);
    cudaMemcpy(h_ref.data(), d_ref, kM * kN * sizeof(kittens::bf16), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out.data(), d_c, kM * kN * sizeof(kittens::bf16), cudaMemcpyDeviceToHost);
    CudaCheckError();

    for (int i = 0; i < kM * kN; ++i) {
        ref_out[i] = __bfloat162float(h_ref[i]);
    }

    info.result = validate(d_a, d_c, empty_input, ref_out, info.label, kN, 0.02f);
    results.push_back(info);

    cudaFree(d_b_layouta);
    cudaFree(d_b);
    cudaFree(d_ref);
    CudaCheckError();
    return info.result == test_result::PASSED;
}

} // namespace

void tests(test_data &results) {
    std::cout << " ----- Starting ops/c500/mma/gemm_layouta_native_smoke tests! -----\n" << std::endl;
    run_smoke_case<32>(results, "c500_gemm_bf16_128x128x32_layouta_native_smoke", 3025);
    run_smoke_case<128>(results, "c500_gemm_bf16_128x128x128_layouta_native_smoke", 3026);
}

} // namespace c500::mma::gemm_layouta_native_smoke

#endif
