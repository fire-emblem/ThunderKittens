#pragma once

#include "fragment_atom.cuh"
#include "issue_order_atom.cuh"

namespace bf16_c500_tk_cute_local::cute_tk {

struct prologue_atom {
    template <typename ALdgType, typename BLdgType, typename T,
              typename ALdsType, typename BLdsType,
              typename StageContract =
                  ::bf16_c500_tk_cute_local::contracts::stage_contract>
    __device__ __forceinline__ static void prime_layoutc(
        uint8_t *wsm_ldg, uint8_t *&a_ptr, uint8_t *&b_ptr, int &k, int n,
        int start_col, uint8_t *wsm_lds, ALdsType (&a)[4][4],
        BLdsType (&b)[4][4], const int (&a_ldg_offset)[2][4],
        const int (&b_ldg_offset)[2][4], const int (&a_lds_offset)[4],
        const int (&b_lds_offset)[4]) {
        constexpr int stage_count = StageContract::stage_count;
#pragma unroll
        for (int stage_i = 0; stage_i < stage_count; ++stage_i) {
            issue_order_atom::template issue_ab_stage_pred<StageContract, ALdgType, BLdgType, T>(
                wsm_ldg, a_ptr, b_ptr, a_ldg_offset, b_ldg_offset, stage_i, 0,
                k / (sizeof(ALdgType) / sizeof(T)), start_col + stage_i * 16,
                n, start_col + (4 + stage_i) * 16, n);
        }

        a_ptr += (128 / 8) * 16 * sizeof(ALdgType);
        b_ptr += 16 * n * sizeof(BLdgType);
        k -= 128;

        fragment_atom::prime_registers<ALdsType, BLdsType, StageContract>(
            a, b, wsm_lds, a_lds_offset, b_lds_offset);
    }
};

} // namespace bf16_c500_tk_cute_local::cute_tk
