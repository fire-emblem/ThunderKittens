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

template <int M, int N, int K>
struct best_family_policy {
    using type = default_layoutc_family;
};

template <>
struct best_family_policy<1664, 1024, 16384> {
    using type = default_layoutc_family;
};

template <>
struct best_family_policy<2048, 2048, 2048> {
    using type = default_layoutc_family;
};

template <>
struct best_family_policy<4096, 4096, 4096> {
    using type = tn_example_bf16_stage4_family;
};

template <>
struct best_family_policy<4608, 128, 3584> {
    using type = continuousc_reusea_perf_family<4608, 128, 3584>;
};

template <>
struct best_family_policy<4608, 256, 3584> {
    using type = tn_example_bf16_stage4_family;
};

template <>
struct best_family_policy<3584, 128, 3584> {
    using type = continuousc_reusea_perf_family<3584, 128, 3584>;
};

template <>
struct best_family_policy<3584, 128, 18944> {
    using type = continuousc_reusea_perf_family<3584, 128, 18944>;
};

template <>
struct best_family_policy<37888, 256, 3584> {
    using type = tn_example_bf16_stage4_family;
};

template <>
struct best_family_policy<37888, 128, 3584> {
    using type = tn_example_bf16_stage4_family;
};

template <int M, int N, int K>
using best_family_t = typename best_family_policy<M, N, K>::type;

} // namespace bf16_c500_tk_cute_local::cute_tk
