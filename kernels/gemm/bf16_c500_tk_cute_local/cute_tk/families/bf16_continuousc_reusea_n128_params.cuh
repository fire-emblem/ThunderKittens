#pragma once

// Legacy family wrapper - now delegates to the new continuousc_reusea_family
// This file exists for backward compatibility with tk_local entry point

#include "continuousc_reusea_family.cuh"

namespace bf16_c500_tk_local::families {

template <int NTile, int APerWarp, int SplitN, int SplitK>
using bf16_continuousc_reusea_n_params =
    ::bf16_c500_tk_cute_local::cute_tk::families::continuousc_reusea_param_family_t<
        NTile, APerWarp, SplitN, SplitK>;

} // namespace bf16_c500_tk_local::families