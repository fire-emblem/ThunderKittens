#include <maca.h>
#include <maca_bfloat16.h>

#include "layoutc_tail.cuh"

using ALdsType = __NATIVE_VECTOR__(4, uint);
using BLdsType = __NATIVE_VECTOR__(4, uint);
using ALdgType = __NATIVE_VECTOR__(4, uint);
using BLdgType = __NATIVE_VECTOR__(4, uint);
using FLOAT4 = __NATIVE_VECTOR__(4, float);

__global__ void layoutc_tail_compile_probe_kernel() {
    constexpr int Stage = 4;
    __shared__ uint8_t wsm[0x10000];
    ALdsType a[Stage][4] = {};
    BLdsType b[Stage][4] = {};
    FLOAT4 c[4][4] = {};
    int a_lds_offset[4] = {};
    int b_lds_offset[4] = {};
    int a_ldg_offset[2][4] = {};
    int b_ldg_offset[2][4] = {};
    uint8_t *gptr = nullptr;

    bf16_c500_tk_local::kernel::run_layoutc_tail_iteration<
        __maca_bfloat16, Stage, FLOAT4, ALdsType, BLdsType, ALdgType, BLdgType>(
        c, a, b, wsm, wsm, a_lds_offset, b_lds_offset, a_ldg_offset,
        b_ldg_offset, gptr, gptr, 0, 0, 0, 0);
    bf16_c500_tk_local::kernel::drain_layoutc_tail<
        __maca_bfloat16, Stage, FLOAT4, ALdsType, BLdsType>(
        c, a, b, wsm, a_lds_offset, b_lds_offset);
}
