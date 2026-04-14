#pragma once

#include "copy_atom.cuh"
#include "continuousc_family.cuh"
#include "continuousc_reusea_family.cuh"
#include "continuousc_reusea_layoutc_family.cuh"
#include "epilogue_atom.cuh"
#include "layoutc_family.cuh"
#include "layoutc_tt_256x256x64_traits.cuh"
#include "layoutc_square_candidates.cuh"
#include "layout_atom.cuh"
#include "mainloop_atom.cuh"
#include "mma_atom.cuh"
#include "square_tt_256x256x64_family.cuh"
#include "tn_example_family.cuh"
#include "sync_atom.cuh"

namespace bf16_c500_tk_cute_local::cute_tk {

using default_layoutc_family =
    ::bf16_c500_tk_cute_local::cute_tk::families::layoutc_128x128x128_stage4;
template <int M, int N, int K>
using layoutc_perf_family =
    ::bf16_c500_tk_cute_local::cute_tk::families::layoutc_family<
        typename layoutc_perf_policy<M, N, K>::tile_shape,
        typename layoutc_perf_policy<M, N, K>::stage_policy>;
using default_continuousc_family =
    ::bf16_c500_tk_cute_local::cute_tk::families::continuousc_128x128x128_stage4;
template <int NTile, int APerWarp, int SplitN, int SplitK>
using continuousc_reusea_family =
    ::bf16_c500_tk_cute_local::cute_tk::families::continuousc_reusea_n_params<
        NTile, APerWarp, SplitN, SplitK>;
template <int M, int N, int K>
using continuousc_reusea_perf_family =
    ::bf16_c500_tk_cute_local::cute_tk::families::continuousc_reusea_family<
        typename continuousc_reusea_perf_policy<M, N, K>::tile_shape,
        typename continuousc_reusea_perf_policy<M, N, K>::stage_policy,
        continuousc_reusea_perf_policy<M, N, K>::a_per_warp,
        continuousc_reusea_perf_policy<M, N, K>::split_n,
        continuousc_reusea_perf_policy<M, N, K>::split_k>;
template <int NTile, int APerWarp, int SplitN, int SplitK>
using continuousc_reusea_layoutc_family =
    ::bf16_c500_tk_cute_local::cute_tk::families::
        continuousc_reusea_layoutc_n_params<NTile, APerWarp, SplitN, SplitK>;

} // namespace bf16_c500_tk_cute_local::cute_tk
