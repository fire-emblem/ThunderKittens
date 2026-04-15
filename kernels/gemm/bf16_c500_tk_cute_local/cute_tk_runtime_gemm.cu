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

inline bool env_flag(const char *name) { return env_int(name, 0) != 0; }

template <typename LocalT, typename RefT, typename Family>
int run_family(const char *case_name, int m, int n, int k, int warmup,
               int profile) {
    return ::bf16_c500_tk_local::bench::run_runtime_case<Family, LocalT, RefT>(
        case_name, m, n, k, warmup, profile);
}

template <int M, int N, int K, typename LocalT, typename RefT>
int run_best_family_case(int warmup, int profile) {
    using family = cute_tk::best_family_t<M, N, K>;
    return run_family<LocalT, RefT, family>("cute_runtime_case_shape_aware_best",
                                            M, N, K, warmup, profile);
}

template <typename LocalT, typename RefT>
int run_shape_aware_dispatch(int m, int n, int k, int warmup, int profile) {
    if (m == 1664 && n == 1024 && k == 16384) {
        return run_best_family_case<1664, 1024, 16384, LocalT, RefT>(warmup,
                                                                      profile);
    }
    if (m == 2048 && n == 2048 && k == 2048) {
        return run_best_family_case<2048, 2048, 2048, LocalT, RefT>(warmup,
                                                                    profile);
    }
    if (m == 4096 && n == 4096 && k == 4096) {
        return run_best_family_case<4096, 4096, 4096, LocalT, RefT>(warmup,
                                                                    profile);
    }
    if (m == 8192 && n == 8192 && k == 8192) {
        return run_best_family_case<8192, 8192, 8192, LocalT, RefT>(warmup,
                                                                    profile);
    }
    if (m == 4608 && n == 128 && k == 3584) {
        return run_best_family_case<4608, 128, 3584, LocalT, RefT>(warmup,
                                                                    profile);
    }
    if (m == 4608 && n == 256 && k == 3584) {
        return run_best_family_case<4608, 256, 3584, LocalT, RefT>(warmup,
                                                                    profile);
    }
    if (m == 3584 && n == 128 && k == 3584) {
        return run_best_family_case<3584, 128, 3584, LocalT, RefT>(warmup,
                                                                    profile);
    }
    if (m == 3584 && n == 128 && k == 18944) {
        return run_best_family_case<3584, 128, 18944, LocalT, RefT>(warmup,
                                                                     profile);
    }
    if (m == 37888 && n == 256 && k == 3584) {
        return run_best_family_case<37888, 256, 3584, LocalT, RefT>(warmup,
                                                                     profile);
    }
    if (m == 37888 && n == 128 && k == 3584) {
        return run_best_family_case<37888, 128, 3584, LocalT, RefT>(warmup,
                                                                     profile);
    }
    return -1;
}

template <typename LocalT, typename RefT>
int run_layoutc_dispatch(int m, int n, int k, int warmup, int profile) {
    if (env_flag("TK_CUTE_USE_SHAPE_AWARE")) {
        if (int rc =
                run_shape_aware_dispatch<LocalT, RefT>(m, n, k, warmup, profile);
            rc >= 0) {
            return rc;
        }
        return -1;
    }
    if (env_flag("TK_CUTE_USE_TN_EXAMPLE")) {
        using family = cute_tk::tn_example_bf16_stage4_family;
        return run_family<LocalT, RefT, family>(
            "cute_runtime_case_tn_example", m, n, k, warmup, profile);
    }
    if (env_flag("TK_CUTE_USE_TN_CONSERVATIVE")) {
        using family = cute_tk::tn_example_conservative_bf16_stage4_family;
        if (family::supports_runtime_shape(m, n, k)) {
            return run_family<LocalT, RefT, family>(
                "cute_runtime_case_tn_conservative", m, n, k, warmup,
                profile);
        }
        return -1;
    }
    if (env_flag("TK_CUTE_USE_TN_LINEAR_GEOMETRY")) {
        using family = cute_tk::tn_example_linear_geom_bf16_stage4_family;
        if (family::supports_runtime_shape(m, n, k)) {
            return run_family<LocalT, RefT, family>(
                "cute_runtime_case_tn_linear_geometry", m, n, k, warmup,
                profile);
        }
        return -1;
    }
    if (env_flag("TK_CUTE_USE_LAYOUTC_TN_TUNING")) {
        using family = cute_tk::layoutc_tn_tuned_bf16_stage4_family;
        if (family::supports_runtime_shape(m, n, k)) {
            return run_family<LocalT, RefT, family>(
                "cute_runtime_case_layoutc_tn_tuned", m, n, k, warmup,
                profile);
        }
        return -1;
    }
    if (env_flag("TK_CUTE_USE_SQUARE_TT256")) {
        if (m == 256 && n == 256 && k == 64) {
            using family = cute_tk::square_tt_256x256x64_stage4_family;
            return run_family<LocalT, RefT, family>(
                "cute_runtime_case_256tile_square_tt256", m, n, k, warmup,
                profile);
        }
        if (m == 2048 && n == 2048 && k == 2048) {
            using family = cute_tk::square_tt_256x256x64_stage4_family;
            return run_family<LocalT, RefT, family>(
                "cute_runtime_case_2048cube_square_tt256", m, n, k, warmup,
                profile);
        }
    if (m == 4096 && n == 4096 && k == 4096) {
        using family = cute_tk::square_tt_256x256x64_stage4_family;
        return run_family<LocalT, RefT, family>(
            "cute_runtime_case_4096cube_square_tt256", m, n, k, warmup,
            profile);
        }
    }
    if (m == 1664 && n == 1024 && k == 16384) {
        using family = cute_tk::default_layoutc_family;
        return run_family<LocalT, RefT, family>("cute_runtime_case_1664x1024x16384_layoutc",
                                                m, n, k, warmup, profile);
    }
    if (m == 2048 && n == 2048 && k == 2048) {
        using family = cute_tk::layoutc_perf_family<2048, 2048, 2048>;
        return run_family<LocalT, RefT, family>("cute_runtime_case_2048cube_layoutc",
                                                m, n, k, warmup, profile);
    }
    if (m == 4096 && n == 4096 && k == 4096) {
        using family = cute_tk::layoutc_perf_family<4096, 4096, 4096>;
        return run_family<LocalT, RefT, family>("cute_runtime_case_4096cube_layoutc",
                                                m, n, k, warmup, profile);
    }
    if (m == 8192 && n == 8192 && k == 8192) {
        using family = cute_tk::layoutc_perf_family<8192, 8192, 8192>;
        return run_family<LocalT, RefT, family>("cute_runtime_case_8192cube_layoutc",
                                                m, n, k, warmup, profile);
    }
    return -1;
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
    if (int rc = run_layoutc_dispatch<__half, __half>(m, n, k, warmup, profile);
        rc >= 0) {
        return rc;
    }
    return run_continuousc_dispatch<__half, __half>(m, n, k, warmup, profile);
#else
    if (int rc = run_layoutc_dispatch<__maca_bfloat16, __nv_bfloat16>(
            m, n, k, warmup, profile);
        rc >= 0) {
        return rc;
    }
    return run_continuousc_dispatch<__maca_bfloat16, __nv_bfloat16>(
        m, n, k, warmup, profile);
#endif
}

} // namespace bf16_c500_tk_cute_local

int main() { return bf16_c500_tk_cute_local::run(); }
