#pragma once

#include "../arch/async_copy.cuh"

namespace bf16_c500_tk_cute_local::primitives {

// Pipeline copy primitive - async copy operations for pipeline stages
struct pipeline_copy_t {
    template <typename PtrType>
    __device__ __forceinline__ static void issue_b128_bsm_no_pred(
        uint8_t *saddr, PtrType *gaddr) {
        ::bf16_c500_tk_cute_local::arch::ldg_b128_bsm_no_predicator(
            reinterpret_cast<void *>(saddr), reinterpret_cast<void *>(gaddr));
    }

    template <int CmpType, typename PtrType>
    __device__ __forceinline__ static void issue_b128_bsm_pred(
        uint8_t *saddr, PtrType *gaddr, int cmp_op1, int cmp_op2) {
        ::bf16_c500_tk_cute_local::arch::ldg_b128_bsm_with_predicator<CmpType>(
            reinterpret_cast<void *>(saddr), reinterpret_cast<void *>(gaddr),
            cmp_op1, cmp_op2);
    }

    template <typename VecType>
    __device__ __forceinline__ static VecType load_gmem_128(const VecType *ptr) {
        using INT128 = __NATIVE_VECTOR__(4, int);
        return __builtin_mxc_load_global_async128(
            reinterpret_cast<INT128 *>(const_cast<VecType *>(ptr)));
    }

    template <typename SaddrType, typename PtrType>
    __device__ __forceinline__ static void load_b_stage_pred_noret0(
        SaddrType saddr,
        PtrType *gaddr,
        int cmp_op1,
        int cmp_op2) {
        ::bf16_c500_tk_cute_local::arch::ldg_b128_bsm_with_predicator_noret0<MACA_ICMP_EQ>(
            reinterpret_cast<void *>(saddr), reinterpret_cast<void *>(gaddr),
            cmp_op1, cmp_op2);
    }
};

} // namespace bf16_c500_tk_cute_local::primitives

// Backward compatibility alias
namespace bf16_c500_tk_cute_local::cute_tk {
using copy_atom = ::bf16_c500_tk_cute_local::primitives::pipeline_copy_t;
}