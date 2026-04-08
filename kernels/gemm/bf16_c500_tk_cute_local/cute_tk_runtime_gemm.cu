#include <cstdlib>
#include <iostream>

#include "bench/runner.cuh"
#include "cute_tk/mainloop.cuh"

namespace bf16_c500_tk_cute_local {

inline int env_int(const char *name, int fallback) {
    if (const char *value = std::getenv(name)) {
        return std::atoi(value);
    }
    return fallback;
}

template <typename LocalT, typename RefT, typename Family>
int run_family(const char *case_name, int m, int n, int k, int warmup,
               int profile) {
    return ::bf16_c500_tk_local::bench::run_runtime_case<Family, LocalT, RefT>(
        case_name, m, n, k, warmup, profile);
}

template <typename LocalT, typename RefT>
int run_continuousc_dispatch(int m, int n, int k, int warmup, int profile) {
    if (m == 4608 && n == 128 && k == 3584) {
        using family = cute_tk::continuousc_reusea_perf_family<4608, 128, 3584>;
        return run_family<LocalT, RefT, family>("cute_runtime_case_4608x128x3584",
                                                m, n, k, warmup, profile);
    }
    if (m == 4608 && n == 256 && k == 3584) {
        using family = cute_tk::continuousc_reusea_perf_family<4608, 256, 3584>;
        return run_family<LocalT, RefT, family>("cute_runtime_case_4608x256x3584",
                                                m, n, k, warmup, profile);
    }
    if (m == 3584 && n == 128 && k == 3584) {
        using family = cute_tk::continuousc_reusea_perf_family<3584, 128, 3584>;
        return run_family<LocalT, RefT, family>("cute_runtime_case_3584x128x3584",
                                                m, n, k, warmup, profile);
    }
    if (m == 3584 && n == 128 && k == 18944) {
        using family = cute_tk::continuousc_reusea_perf_family<3584, 128, 18944>;
        return run_family<LocalT, RefT, family>("cute_runtime_case_3584x128x18944",
                                                m, n, k, warmup, profile);
    }
    if (m == 3584 && n == 256 && k == 18944) {
        using family = cute_tk::continuousc_reusea_perf_family<3584, 256, 18944>;
        return run_family<LocalT, RefT, family>("cute_runtime_case_3584x256x18944",
                                                m, n, k, warmup, profile);
    }
    if (m == 37888 && n == 256 && k == 3584) {
        using family = cute_tk::continuousc_reusea_perf_family<37888, 256, 3584>;
        return run_family<LocalT, RefT, family>("cute_runtime_case_37888x256x3584",
                                                m, n, k, warmup, profile);
    }

    using family = cute_tk::default_continuousc_family;
    return run_family<LocalT, RefT, family>("cute_runtime_case_default", m, n, k,
                                            warmup, profile);
}

int run() {
    const int m = env_int("TK_CUTE_M", 4096);
    const int n = env_int("TK_CUTE_N", 4096);
    const int k = env_int("TK_CUTE_K", 4096);
    const int warmup = env_int("TK_CUTE_WARMUP", 1);
    const int profile = env_int("TK_CUTE_PROFILE", 3);

#ifdef TK_CUTE_USE_FP16
    return run_continuousc_dispatch<__half, __half>(m, n, k, warmup, profile);
#else
    return run_continuousc_dispatch<__maca_bfloat16, __nv_bfloat16>(
        m, n, k, warmup, profile);
#endif
}

} // namespace bf16_c500_tk_cute_local

int main() { return bf16_c500_tk_cute_local::run(); }
