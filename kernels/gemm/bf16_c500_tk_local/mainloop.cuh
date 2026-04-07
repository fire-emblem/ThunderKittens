#pragma once

#include "dispatch/bf16_dispatch.cuh"
#include "families/bf16_continuousc_128x128x128_stage4.cuh"

namespace bf16_c500_tk_local {

template<int M, int N, int K>
using bf16_mainloop_family_t = dispatch::bf16_family_t<M, N, K>;

template<int M, int N, int K>
using bf16_continuousc_family_t = families::bf16_continuousc_128x128x128_stage4;

} // namespace bf16_c500_tk_local
