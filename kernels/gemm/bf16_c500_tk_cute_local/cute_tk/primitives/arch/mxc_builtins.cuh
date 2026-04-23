#pragma once

#include <maca.h>
#include <maca_bfloat16.h>
#include <maca_fp16.h>
#include <mc_runtime.h>

namespace bf16_c500_tk_cute_local::arch {

using float4_native = __NATIVE_VECTOR__(4, float);
using uint2_native = __NATIVE_VECTOR__(2, uint);
using half_native = __half;
using bf16_native = __maca_bfloat16;

} // namespace bf16_c500_tk_cute_local::arch
