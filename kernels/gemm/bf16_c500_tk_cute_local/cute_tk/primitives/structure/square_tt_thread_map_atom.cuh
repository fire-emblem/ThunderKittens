#pragma once

// Backward compatibility layer - delegates to layout/thread_map.cuh
// This file will be removed after migration is complete

#include "../layout/thread_map.cuh"
#include "../../contracts/square_tt_tile_contract.cuh"

namespace bf16_c500_tk_cute_local::cute_tk {

using square_tt_thread_map_atom = ::bf16_c500_tk_cute_local::primitives::square_tile_thread_map_t;

} // namespace bf16_c500_tk_cute_local::cute_tk