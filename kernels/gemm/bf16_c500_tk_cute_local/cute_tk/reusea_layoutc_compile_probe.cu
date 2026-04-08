#include <cuda_bf16.h>
#include <maca_bfloat16.h>

#include "../bench/runner.cuh"
#include "mainloop.cuh"

namespace {

struct probe_case {
    static constexpr const char *case_name = "cute_tk_reusea_layoutc_probe";
    static constexpr int m = 4608;
    static constexpr int n = 128;
    static constexpr int k = 3584;
    static constexpr int warmup_iters = 1;
    static constexpr int profile_iters = 1;
    using family =
        bf16_c500_tk_cute_local::cute_tk::continuousc_reusea_layoutc_family<
            128, 2, 2, 1>;
    using local_t = __maca_bfloat16;
    using ref_t = __nv_bfloat16;
};

} // namespace

int main() { return ::bf16_c500_tk_local::bench::run_case<probe_case>(); }
