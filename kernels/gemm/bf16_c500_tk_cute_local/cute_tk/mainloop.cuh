#pragma once

#include "copy_atom.cuh"
#include "continuousc_family.cuh"
#include "continuousc_reusea_family.cuh"
#include "continuousc_reusea_layoutc_family.cuh"
#include "epilogue_atom.cuh"
#include "layoutc_family.cuh"
#include "layout_atom.cuh"
#include "mainloop_atom.cuh"
#include "mma_atom.cuh"
#include "sync_atom.cuh"

namespace bf16_c500_tk_cute_local::cute_tk {

using default_layoutc_family =
    ::bf16_c500_tk_cute_local::cute_tk::families::layoutc_128x128x128_stage4;
using default_continuousc_family =
    ::bf16_c500_tk_cute_local::cute_tk::families::continuousc_128x128x128_stage4;
template <int NTile, int APerWarp, int SplitN, int SplitK>
using continuousc_reusea_family =
    ::bf16_c500_tk_cute_local::cute_tk::families::continuousc_reusea_n_params<
        NTile, APerWarp, SplitN, SplitK>;
template <int NTile, int APerWarp, int SplitN, int SplitK>
using continuousc_reusea_layoutc_family =
    ::bf16_c500_tk_cute_local::cute_tk::families::
        continuousc_reusea_layoutc_n_params<NTile, APerWarp, SplitN, SplitK>;

} // namespace bf16_c500_tk_cute_local::cute_tk
