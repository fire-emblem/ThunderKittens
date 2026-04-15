#pragma once

#include "../../../kernel/layoutc_support.cuh"

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

    template <typename StageLayout, typename ALdgType, typename BLdgType,
              typename ScalarT>
    __device__ __forceinline__ static void issue_ab_stage_pred(
        uint8_t *wsm_ldg, uint8_t *a_ptr, uint8_t *b_ptr,
        const int (&a_ldg_offset)[2][4], const int (&b_ldg_offset)[2][4],
        int stage_idx, int a_cmp_op1, int a_cmp_op2, int b_cmp_op1_bank0,
        int b_cmp_op2_bank0, int b_cmp_op1_bank1, int b_cmp_op2_bank1) {
        issue_a_bank_pred<StageLayout, ALdgType, ScalarT>(
            wsm_ldg, a_ptr, a_ldg_offset[0][stage_idx], stage_idx, 0,
            a_cmp_op1, a_cmp_op2);
        issue_a_bank_pred<StageLayout, ALdgType, ScalarT>(
            wsm_ldg, a_ptr, a_ldg_offset[1][stage_idx], stage_idx, 1,
            a_cmp_op1, a_cmp_op2);
        issue_b_bank_pred<StageLayout, BLdgType, ScalarT>(
            wsm_ldg, b_ptr, b_ldg_offset[0][stage_idx], stage_idx, 0,
            b_cmp_op1_bank0, b_cmp_op2_bank0);
        issue_b_bank_pred<StageLayout, BLdgType, ScalarT>(
            wsm_ldg, b_ptr, b_ldg_offset[1][stage_idx], stage_idx, 1,
            b_cmp_op1_bank1, b_cmp_op2_bank1);
    }
};

} // namespace bf16_c500_tk_cute_local::cute_tk
