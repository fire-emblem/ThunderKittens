#pragma once

#include "dispatch/bf16_dispatch.cuh"
#include "families/bf16_continuousc_128x128x128_stage4.cuh"
#include "families/bf16_continuousc_reusea_n128_params.cuh"

namespace bf16_c500_tk_local {

template<int M, int N, int K>
using bf16_mainloop_family_t = dispatch::bf16_family_t<M, N, K>;

template<int M, int N, int K>
using bf16_continuousc_family_t = families::bf16_continuousc_128x128x128_stage4;

template<int NTile, int APerWarp, int SplitN, int SplitK>
using bf16_continuousc_reusea_n_family_t =
    families::bf16_continuousc_reusea_n_params<NTile, APerWarp, SplitN, SplitK>;

} // namespace bf16_c500_tk_local
