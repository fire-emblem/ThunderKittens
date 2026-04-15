#pragma once

#include "../kernel/layoutc_support.cuh"

namespace bf16_c500_tk_cute_local::cute_tk {

struct issue_order_atom {
    template <typename StageLayout>
    __device__ __forceinline__ static void issue_a_bank_no_pred(
        uint8_t *wsm_ldg, uint8_t *a_ptr, int a_ldg_offset, int stage_idx,
        int bank_idx) {
        LDG_B128_BSM_NO_PREDICATOR(
            wsm_ldg + StageLayout::a_stage_offset(stage_idx, bank_idx),
            a_ptr + a_ldg_offset);
    }

    template <typename StageLayout>
    __device__ __forceinline__ static void issue_b_bank_no_pred(
        uint8_t *wsm_ldg, uint8_t *b_ptr, int b_ldg_offset, int stage_idx,
        int bank_idx) {
        LDG_B128_BSM_NO_PREDICATOR(
            wsm_ldg + StageLayout::b_stage_offset(stage_idx, bank_idx),
            b_ptr + b_ldg_offset);
    }

    template <typename StageLayout, typename VecType, typename ScalarT>
    __device__ __forceinline__ static void issue_a_bank_pred(
        uint8_t *wsm_ldg, uint8_t *a_ptr, int a_ldg_offset, int stage_idx,
        int bank_idx, int cmp_op1, int cmp_op2) {
        __builtin_mxc_ldg_b128_bsm_predicator(
            wsm_ldg + StageLayout::a_stage_offset(stage_idx, bank_idx),
            a_ptr + a_ldg_offset, 0, true, true, false, true, cmp_op1, cmp_op2,
            MACA_ICMP_SLT);
    }

    template <typename StageLayout, typename VecType, typename ScalarT>
    __device__ __forceinline__ static void issue_b_bank_pred(
        uint8_t *wsm_ldg, uint8_t *b_ptr, int b_ldg_offset, int stage_idx,
        int bank_idx, int cmp_op1, int cmp_op2) {
        __builtin_mxc_ldg_b128_bsm_predicator(
            wsm_ldg + StageLayout::b_stage_offset(stage_idx, bank_idx),
            b_ptr + b_ldg_offset, 0, true, true, false, true, cmp_op1, cmp_op2,
            MACA_ICMP_SLT);
    }
};

} // namespace bf16_c500_tk_cute_local::cute_tk
