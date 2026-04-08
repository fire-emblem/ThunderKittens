#include <cuda_bf16.h>
#include <maca_bfloat16.h>

#include "bench/runner.cuh"
#include "cute_tk/mainloop.cuh"

namespace bf16_c500_tk_cute_local {

#ifndef BF16_C500_TK_CUTE_LOCAL_M
#ifdef TK_CUTE_LOCAL_USE_CONTINUOUSC
#define BF16_C500_TK_CUTE_LOCAL_M 4608
#else
#define BF16_C500_TK_CUTE_LOCAL_M 4096
#endif
#endif
#ifndef BF16_C500_TK_CUTE_LOCAL_N
#ifdef TK_CUTE_LOCAL_USE_CONTINUOUSC
#define BF16_C500_TK_CUTE_LOCAL_N 128
#else
#define BF16_C500_TK_CUTE_LOCAL_N 4096
#endif
#endif
#ifndef BF16_C500_TK_CUTE_LOCAL_K
#ifdef TK_CUTE_LOCAL_USE_CONTINUOUSC
#define BF16_C500_TK_CUTE_LOCAL_K 3584
#else
#define BF16_C500_TK_CUTE_LOCAL_K 4096
#endif
#endif
#ifndef BF16_C500_TK_CUTE_LOCAL_WARMUP_ITERS
#define BF16_C500_TK_CUTE_LOCAL_WARMUP_ITERS 1
#endif
#ifndef BF16_C500_TK_CUTE_LOCAL_PROFILE_ITERS
#define BF16_C500_TK_CUTE_LOCAL_PROFILE_ITERS 3
#endif

struct layoutc_case {
    static constexpr const char *case_name =
#ifdef TK_CUTE_LOCAL_USE_CONTINUOUSC
#ifdef TK_CUTE_LOCAL_USE_REUSEA_LAYOUTC
        "cute_tk_reusea_layoutc_case";
#else
        "cute_tk_continuousc_case";
#endif
#else
        "cute_tk_layoutc_case";
#endif
    static constexpr int m = BF16_C500_TK_CUTE_LOCAL_M;
    static constexpr int n = BF16_C500_TK_CUTE_LOCAL_N;
    static constexpr int k = BF16_C500_TK_CUTE_LOCAL_K;
    static constexpr int warmup_iters = BF16_C500_TK_CUTE_LOCAL_WARMUP_ITERS;
    static constexpr int profile_iters = BF16_C500_TK_CUTE_LOCAL_PROFILE_ITERS;
#ifdef TK_CUTE_LOCAL_USE_CONTINUOUSC
#ifdef TK_CUTE_LOCAL_USE_REUSEA_LAYOUTC
    using family = cute_tk::continuousc_reusea_layoutc_family<128, 2, 2, 1>;
#elif defined(TK_CUTE_LOCAL_USE_REUSEA)
    using family = cute_tk::continuousc_reusea_family<128, 2, 2, 1>;
#else
    using family = cute_tk::default_continuousc_family;
#endif
#else
    using family = cute_tk::default_layoutc_family;
#endif
    using local_t = __maca_bfloat16;
    using ref_t = __nv_bfloat16;
};

int run() { return ::bf16_c500_tk_local::bench::run_case<layoutc_case>(); }

} // namespace bf16_c500_tk_cute_local

int main() { return bf16_c500_tk_cute_local::run(); }
