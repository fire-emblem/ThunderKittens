#pragma once

#include "../families/bf16_layoutc_128x128x128_stage4.cuh"

namespace bf16_c500_tk_local::dispatch {

using bf16_default_family = families::bf16_layoutc_128x128x128_stage4;

template<int M, int N, int K>
struct bf16_family_selector {
    using type = bf16_default_family;
};

template<int M, int N, int K>
using bf16_family_t = typename bf16_family_selector<M, N, K>::type;

} // namespace bf16_c500_tk_local::dispatch
