#pragma once

// Legacy family wrapper - now delegates to the new layoutc_family
// This file exists for backward compatibility with tk_local entry point

#include "layoutc_family.cuh"

namespace bf16_c500_tk_local::families {

// Legacy alias for backward compatibility
using bf16_layoutc_128x128x128_stage4 =
    ::bf16_c500_tk_cute_local::cute_tk::families::layoutc_tile128x128x128_stage4_family_t;

} // namespace bf16_c500_tk_local::families
