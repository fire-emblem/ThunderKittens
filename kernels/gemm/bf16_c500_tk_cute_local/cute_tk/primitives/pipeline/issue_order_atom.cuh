#pragma once

#include "copy_atom.cuh"

namespace bf16_c500_tk_cute_local::primitives {

// Pipeline issue order primitive - load issue ordering across pipeline stages
struct pipeline_issue_order_t {
    template <typename StageLayout>
    __device__ __forceinline__ static void issue_a_bank_no_pred(
        uint8_t *wsm_ldg, uint8_t *a_ptr, int a_ldg_offset, int stage_idx,
        int bank_idx) {
        pipeline_copy_t::issue_b128_bsm_no_pred(
            wsm_ldg + StageLayout::a_stage_offset(stage_idx, bank_idx),
            a_ptr + a_ldg_offset);
    }

    template <typename StageLayout>
    __device__ __forceinline__ static void issue_b_bank_no_pred(
        uint8_t *wsm_ldg, uint8_t *b_ptr, int b_ldg_offset, int stage_idx,
        int bank_idx) {
        pipeline_copy_t::issue_b128_bsm_no_pred(
            wsm_ldg + StageLayout::b_stage_offset(stage_idx, bank_idx),
            b_ptr + b_ldg_offset);
    }

    template <typename StageLayout, typename VecType, typename ScalarT>
    __device__ __forceinline__ static void issue_a_bank_pred(
        uint8_t *wsm_ldg, uint8_t *a_ptr, int a_ldg_offset, int stage_idx,
        int bank_idx, int cmp_op1, int cmp_op2) {
        pipeline_copy_t::template issue_b128_bsm_pred<MACA_ICMP_SLT>(
            wsm_ldg + StageLayout::a_stage_offset(stage_idx, bank_idx),
            a_ptr + a_ldg_offset, cmp_op1, cmp_op2);
    }

    template <typename StageLayout, typename VecType, typename ScalarT>
    __device__ __forceinline__ static void issue_b_bank_pred(
        uint8_t *wsm_ldg, uint8_t *b_ptr, int b_ldg_offset, int stage_idx,
        int bank_idx, int cmp_op1, int cmp_op2) {
        pipeline_copy_t::template issue_b128_bsm_pred<MACA_ICMP_SLT>(
            wsm_ldg + StageLayout::b_stage_offset(stage_idx, bank_idx),
            b_ptr + b_ldg_offset, cmp_op1, cmp_op2);
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

} // namespace bf16_c500_tk_cute_local::primitives

// Backward compatibility alias
namespace bf16_c500_tk_cute_local::cute_tk {
using issue_order_atom = ::bf16_c500_tk_cute_local::primitives::pipeline_issue_order_t;
}