#pragma once

#include <maca_bfloat16.h>
#include <maca_fp16.h>
#include <mc_common.h>
#include <mc_runtime.h>

#include <type_traits>

#include "primitives/compute/mma_atom.cuh"
#include "primitives/pipeline/copy_atom.cuh"
#include "primitives/pipeline/sync_atom.cuh"

namespace bf16_c500_tk_cute_local::cute_tk {

using tn_example_float4 = mma_atom::float4_t;

template <typename PtrType>
__device__ __forceinline__ void tn_issue_b128_bsm_no_pred(uint8_t *saddr,
                                                           PtrType *gaddr) {
    copy_atom::issue_b128_bsm_no_pred(saddr, gaddr);
}

template <typename PtrType>
__device__ __forceinline__ void tn_issue_b128_bsm_pred(
    uint8_t *saddr, PtrType *gaddr, int cmp_op1, int cmp_op2, int cmp_type) {
    copy_atom::issue_b128_bsm_pred(saddr, gaddr, cmp_op1, cmp_op2, cmp_type);
}

template <typename PtrType>
__device__ __forceinline__ void tn_issue_b64_bsm_no_pred(uint8_t *saddr,
                                                          PtrType *gaddr) {
    copy_atom::issue_b64_bsm_no_pred(saddr, gaddr);
}

template <typename PtrType>
__device__ __forceinline__ void tn_issue_b64_bsm_pred(
    uint8_t *saddr, PtrType *gaddr, int cmp_op1, int cmp_op2, int cmp_type) {
    copy_atom::issue_b64_bsm_pred(saddr, gaddr, cmp_op1, cmp_op2, cmp_type);
}

template <typename T, bool SwapAB = false>
__forceinline__ __device__ tn_example_float4 mma_16x16x16b16(uint a0, uint a1,
                                                             uint b0, uint b1,
                                                             tn_example_float4 C) {
    if constexpr (SwapAB) {
        return mma_atom::template fma_pair<T>(b0, b1, a0, a1, C);
    } else {
        return mma_atom::template fma_pair<T>(a0, a1, b0, b1, C);
    }
}

} // namespace bf16_c500_tk_cute_local::cute_tk

#ifdef arrive_gvmcnt
#undef arrive_gvmcnt
#endif
#define arrive_gvmcnt(num)                                                        \
    ::bf16_c500_tk_cute_local::cute_tk::sync_atom::template wait_gmem_async<num>()

#ifdef arrive_bsmcnt
#undef arrive_bsmcnt
#endif
#define arrive_bsmcnt(num)                                                        \
    ::bf16_c500_tk_cute_local::cute_tk::sync_atom::template wait_bsm_async<num>()

#ifdef LDG_B128_BSM_NO_PREDICATOR
#undef LDG_B128_BSM_NO_PREDICATOR
#endif
#define LDG_B128_BSM_NO_PREDICATOR(saddr, gaddr)                                  \
    ::bf16_c500_tk_cute_local::cute_tk::tn_issue_b128_bsm_no_pred((saddr),       \
                                                                   (gaddr));

#ifdef LDG_B128_BSM_WITH_PREDICATOR
#undef LDG_B128_BSM_WITH_PREDICATOR
#endif
#define LDG_B128_BSM_WITH_PREDICATOR(saddr, gaddr, cmp_op1, cmp_op2, cmp_type)   \
    ::bf16_c500_tk_cute_local::cute_tk::tn_issue_b128_bsm_pred(                   \
        (saddr), (gaddr), (cmp_op1), (cmp_op2), (cmp_type));

#ifdef LDG_B64_BSM_NO_PREDICATOR
#undef LDG_B64_BSM_NO_PREDICATOR
#endif
#define LDG_B64_BSM_NO_PREDICATOR(saddr, gaddr)                                   \
    ::bf16_c500_tk_cute_local::cute_tk::tn_issue_b64_bsm_no_pred((saddr),        \
                                                                  (gaddr));

#ifdef LDG_B64_BSM_WITH_PREDICATOR
#undef LDG_B64_BSM_WITH_PREDICATOR
#endif
#define LDG_B64_BSM_WITH_PREDICATOR(saddr, gaddr, cmp_op1, cmp_op2, cmp_type)    \
    ::bf16_c500_tk_cute_local::cute_tk::tn_issue_b64_bsm_pred(                    \
        (saddr), (gaddr), (cmp_op1), (cmp_op2), (cmp_type));
