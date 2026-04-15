#pragma once

#include "../../../kernel/layoutc_prologue.cuh"

namespace bf16_c500_tk_cute_local::cute_tk {

struct copy_atom {
    template <typename PtrType>
    __device__ __forceinline__ static void issue_b128_bsm_no_pred(
        uint8_t *saddr, PtrType *gaddr) {
        __builtin_mxc_ldg_b128_bsm_predicator(saddr, gaddr, 0, true, true,
                                              false, true, 1, 1, MACA_ICMP_EQ);
    }

    template <typename VecType>
    __device__ __forceinline__ static VecType load_gmem_128(const VecType *ptr) {
        using INT128 = __NATIVE_VECTOR__(4, int);
        return __builtin_mxc_load_global_async128(
            reinterpret_cast<INT128 *>(const_cast<VecType *>(ptr)));
    }

    template <typename PtrType>
    __device__ __forceinline__ static void load_b_stage_pred_noret0(
        uint8_t *saddr,
        PtrType *gaddr,
        int cmp_op1,
        int cmp_op2) {
        LDG_B128_BSM_WITH_PREDICATOR_NORET0(saddr, gaddr, cmp_op1, cmp_op2,
                                            MACA_ICMP_EQ);
    }

    template <typename ALdgType, typename BLdgType, typename T,
              typename StageContract =
                  ::bf16_c500_tk_local::contracts::stage_contract>
    __device__ __forceinline__ static void issue_prologue(
        uint8_t *wsm_ldg,
        uint8_t *a_ptr,
        uint8_t *b_ptr,
        const int (&a_ldg_offset)[2][4],
        const int (&b_ldg_offset)[2][4],
        int k,
        int n,
        int start_col) {
        ::bf16_c500_tk_local::kernel::issue_layoutc_prologue<ALdgType, BLdgType, T,
                                                             StageContract>(
            wsm_ldg, a_ptr, b_ptr, a_ldg_offset, b_ldg_offset, k, n, start_col);
    }

    template <typename ALdsType, typename BLdsType,
              typename StageContract =
                  ::bf16_c500_tk_local::contracts::stage_contract>
    __device__ __forceinline__ static void prime_fragments(
        ALdsType (&a)[4][4],
        BLdsType (&b)[4][4],
        uint8_t *wsm_lds,
        const int (&a_lds_offset)[4],
        const int (&b_lds_offset)[4]) {
        ::bf16_c500_tk_local::kernel::prime_layoutc_registers<ALdsType, BLdsType,
                                                              StageContract>(
            a, b, wsm_lds, a_lds_offset, b_lds_offset);
    }

    template <typename ALdsType, typename BLdsType,
              typename StageContract =
                  ::bf16_c500_tk_local::contracts::stage_contract>
    __device__ __forceinline__ static void reload_stage(
        ALdsType (&a)[4][4],
        BLdsType (&b)[4][4],
        int lds_idx,
        uint8_t *wsm_lds,
        const int (&a_lds_offset)[4],
        const int (&b_lds_offset)[4]) {
        ::bf16_c500_tk_local::kernel::
            reload_layoutc_stage_from_shared<ALdsType, BLdsType, StageContract>(
            a, b, lds_idx, wsm_lds, a_lds_offset, b_lds_offset);
    }
};

} // namespace bf16_c500_tk_cute_local::cute_tk
