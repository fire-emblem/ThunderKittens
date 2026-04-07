#include <cstdlib>
#include <iostream>

#include "bench/runner.cuh"

namespace bf16_c500_tk_local {

inline int env_int(const char *name, int fallback) {
    if (const char *value = std::getenv(name)) {
        return std::atoi(value);
    }
    return fallback;
}

int run() {
    const int m = env_int("TK_LOCAL_M", 4096);
    const int n = env_int("TK_LOCAL_N", 4096);
    const int k = env_int("TK_LOCAL_K", 4096);
    const int warmup = env_int("TK_LOCAL_WARMUP", 1);
    const int profile = env_int("TK_LOCAL_PROFILE", 3);

#ifdef TK_LOCAL_USE_CONTINUOUSC
    using family = bf16_continuousc_family_t<128, 128, 128>;
#else
    using family = bf16_mainloop_family_t<128, 128, 128>;
#endif

#ifdef TK_LOCAL_USE_FP16
    return bench::run_runtime_case<family, __half, __half>(
        "runtime_case_fp16", m, n, k, warmup, profile);
#else
    return bench::run_runtime_case<family, __maca_bfloat16, __nv_bfloat16>(
        "runtime_case_bf16", m, n, k, warmup, profile);
#endif
}

} // namespace bf16_c500_tk_local

int main() { return bf16_c500_tk_local::run(); }
