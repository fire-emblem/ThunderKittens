#pragma once

#include <maca.h>

namespace bf16_c500_tk_cute_local::arch {

__device__ __forceinline__ void ldg_b128_bsm_no_predicator(void *saddr, void *gaddr) {
    __builtin_mxc_ldg_b128_bsm_predicator(saddr, gaddr, 0, true, true, false, true, 1, 1, MACA_ICMP_EQ);
}

template <int CmpType>
__device__ __forceinline__ void ldg_b128_bsm_with_predicator(
    void *saddr, void *gaddr, int cmp_op1, int cmp_op2) {
    __builtin_mxc_ldg_b128_bsm_predicator(
        saddr, gaddr, 0, true, true, false, true, cmp_op1, cmp_op2, CmpType);
}

template <int CmpType>
__device__ __forceinline__ void ldg_b128_bsm_with_predicator_noret0(
    void *saddr, void *gaddr, int cmp_op1, int cmp_op2) {
    __builtin_mxc_ldg_b128_bsm_predicator(
        saddr, gaddr, 0, false, true, false, true, cmp_op1, cmp_op2, CmpType);
}

__device__ __forceinline__ void ldg_b64_bsm_no_predicator(void *saddr, void *gaddr) {
    __builtin_mxc_ldg_b64_bsm_predicator(saddr, gaddr, 0, true, true, false, true, 1, 1, MACA_ICMP_EQ);
}

template <int CmpType>
__device__ __forceinline__ void ldg_b64_bsm_with_predicator(
    void *saddr, void *gaddr, int cmp_op1, int cmp_op2) {
    __builtin_mxc_ldg_b64_bsm_predicator(
        saddr, gaddr, 0, true, true, false, true, cmp_op1, cmp_op2, CmpType);
}

template <int CmpType>
__device__ __forceinline__ void ldg_b64_bsm_with_predicator_noret0(
    void *saddr, void *gaddr, int cmp_op1, int cmp_op2) {
    __builtin_mxc_ldg_b64_bsm_predicator(
        saddr, gaddr, 0, false, true, false, true, cmp_op1, cmp_op2, CmpType);
}

} // namespace bf16_c500_tk_cute_local::arch
