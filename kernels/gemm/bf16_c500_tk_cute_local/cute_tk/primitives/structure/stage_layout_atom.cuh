#pragma once

// Backward compatibility layer - delegates to layout/stage_layout.cuh
// This file will be removed after migration is complete

#include "../layout/stage_layout.cuh"

namespace bf16_c500_tk_cute_local::cute_tk {

template <typename StageContractT>
using stage_layout_atom = ::bf16_c500_tk_cute_local::primitives::stage_layout_atom_t<StageContractT>;

using default_stage_layout_atom =
    stage_layout_atom<::bf16_c500_tk_cute_local::contracts::stage_contract>;

} // namespace bf16_c500_tk_cute_local::cute_tk
