#pragma once

#include "../families/bf16_balanced_128x128x128_stage4.cuh"
#include "../families/bf16_muxi_128x128x128_stage4.cuh"

#if defined(BF16_C500_USE_LAYOUTC_NATIVE) && BF16_C500_USE_LAYOUTC_NATIVE
#include "../families/bf16_c500_layoutc_128x128x128_stage4.cuh"
#endif

namespace kittens::arch::c500::gemm::dispatch {

using bf16_default_family = families::bf16_balanced_128x128x128_stage4;
using bf16_layouta_native_family = families::bf16_muxi_128x128x128_stage4;
#if defined(BF16_C500_USE_LAYOUTC_NATIVE) && BF16_C500_USE_LAYOUTC_NATIVE
using bf16_layoutc_native_family = families::bf16_c500_layoutc_128x128x128_stage4;
#endif

template<int M, int N, int K, bool LayoutANative = false>
struct bf16_family_selector {
    using type = bf16_default_family;
};

template<int M, int N, int K>
struct bf16_family_selector<M, N, K, true> {
    using type = bf16_layouta_native_family;
};

template<int M, int N, int K, typename Globals>
__device__ inline void run_bf16(const Globals &g) {
    using family = typename bf16_family_selector<M, N, K, false>::type;
    family::template run<M, N, K>(g);
}

template<int M, int N, int K, typename Globals>
__device__ inline void run_bf16_layouta(const Globals &g) {
    using family = typename bf16_family_selector<M, N, K, true>::type;
    family::template run_layouta<M, N, K>(g);
}

template<int M, int N, int K, typename Globals>
__device__ inline void run_bf16_layoutc(const Globals &g) {
#if defined(BF16_C500_USE_LAYOUTC_NATIVE) && BF16_C500_USE_LAYOUTC_NATIVE
    using family = bf16_layoutc_native_family;
    family::template run_layoutc<M, N, K>(g);
#else
    static_assert(M == -1, "BF16_C500_USE_LAYOUTC_NATIVE must be enabled to use run_bf16_layoutc.");
#endif
}

} // namespace kittens::arch::c500::gemm::dispatch
