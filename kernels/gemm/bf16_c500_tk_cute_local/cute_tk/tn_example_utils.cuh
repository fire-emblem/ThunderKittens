#pragma once

#include <maca_bfloat16.h>
#include <maca_fp16.h>
#include <mc_common.h>
#include <mc_runtime.h>

#include <type_traits>

using tn_example_float4 = __NATIVE_VECTOR__(4, float);

#ifdef arrive_gvmcnt
#undef arrive_gvmcnt
#endif
#define arrive_gvmcnt(num) __builtin_mxc_arrive_gvmcnt(num)
#ifdef arrive_bsmcnt
#undef arrive_bsmcnt
#endif
#define arrive_bsmcnt(num) __builtin_mxc_arrive_bsmcnt(num)

#ifdef LDG_B128_BSM_NO_PREDICATOR
#undef LDG_B128_BSM_NO_PREDICATOR
#endif
#define LDG_B128_BSM_NO_PREDICATOR(saddr, gaddr)                                          \
    __builtin_mxc_ldg_b128_bsm_predicator(saddr, gaddr, 0, true, true, false, true, 1, 1, \
                                          MACA_ICMP_EQ);
#ifdef LDG_B128_BSM_WITH_PREDICATOR
#undef LDG_B128_BSM_WITH_PREDICATOR
#endif
#define LDG_B128_BSM_WITH_PREDICATOR(saddr, gaddr, cmp_op1, cmp_op2, cmp_type)               \
    __builtin_mxc_ldg_b128_bsm_predicator(saddr, gaddr, 0, true, true, false, true, cmp_op1, \
                                          cmp_op2, cmp_type);
#ifdef LDG_B64_BSM_NO_PREDICATOR
#undef LDG_B64_BSM_NO_PREDICATOR
#endif
#define LDG_B64_BSM_NO_PREDICATOR(saddr, gaddr)                                          \
    __builtin_mxc_ldg_b64_bsm_predicator(saddr, gaddr, 0, true, true, false, true, 1, 1, \
                                         MACA_ICMP_EQ);
#ifdef LDG_B64_BSM_WITH_PREDICATOR
#undef LDG_B64_BSM_WITH_PREDICATOR
#endif
#define LDG_B64_BSM_WITH_PREDICATOR(saddr, gaddr, cmp_op1, cmp_op2, cmp_type)               \
    __builtin_mxc_ldg_b64_bsm_predicator(saddr, gaddr, 0, true, true, false, true, cmp_op1, \
                                         cmp_op2, cmp_type);

template <typename T, bool SwapAB = false>
__forceinline__ __device__ tn_example_float4 mma_16x16x16b16(uint a0, uint a1,
                                                             uint b0, uint b1,
                                                             tn_example_float4 C) {
    using UINT2 = __NATIVE_VECTOR__(2, uint);

    UINT2 A;
    UINT2 B;
    if constexpr (SwapAB) {
        A = UINT2{b0, b1};
        B = UINT2{a0, a1};
    } else {
        A = UINT2{a0, a1};
        B = UINT2{b0, b1};
    }

    if constexpr (std::is_same<T, __half>::value) {
        return __builtin_mxc_mma_16x16x16f16(A, B, C);
    } else {
        return __builtin_mxc_mma_16x16x16bf16(A, B, C);
    }
}
