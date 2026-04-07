#include "bench/runner.cuh"
#include "suites/muxi_shape_suite.cuh"

namespace bf16_c500_tk_local {

#ifndef BF16_C500_MUXI_NATIVE_M
#define BF16_C500_MUXI_NATIVE_M 4096
#endif
#ifndef BF16_C500_MUXI_NATIVE_N
#define BF16_C500_MUXI_NATIVE_N 4096
#endif
#ifndef BF16_C500_MUXI_NATIVE_K
#define BF16_C500_MUXI_NATIVE_K 4096
#endif
#ifndef BF16_C500_MUXI_NATIVE_WARMUP_ITERS
#define BF16_C500_MUXI_NATIVE_WARMUP_ITERS 3
#endif
#ifndef BF16_C500_MUXI_NATIVE_PROFILE_ITERS
#define BF16_C500_MUXI_NATIVE_PROFILE_ITERS 10
#endif

struct single_case {
    static constexpr const char *case_name = "single_case";
    static constexpr int m = BF16_C500_MUXI_NATIVE_M;
    static constexpr int n = BF16_C500_MUXI_NATIVE_N;
    static constexpr int k = BF16_C500_MUXI_NATIVE_K;
    static constexpr int warmup_iters = BF16_C500_MUXI_NATIVE_WARMUP_ITERS;
    static constexpr int profile_iters = BF16_C500_MUXI_NATIVE_PROFILE_ITERS;
#ifdef TK_LOCAL_USE_CONTINUOUSC
    using family = bf16_continuousc_family_t<m, n, k>;
#else
    using family = bf16_mainloop_family_t<m, n, k>;
#endif
#ifdef TK_LOCAL_USE_FP16
    using local_t = __half;
    using ref_t = __half;
#else
    using local_t = __maca_bfloat16;
    using ref_t = __nv_bfloat16;
#endif
};

int run() { return bench::run_case<single_case>(); }

} // namespace bf16_c500_tk_local

int main() { return bf16_c500_tk_local::run(); }
