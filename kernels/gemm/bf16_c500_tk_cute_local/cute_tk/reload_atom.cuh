#pragma once

namespace bf16_c500_tk_cute_local::cute_tk {

struct reload_atom {
    template <typename LdsType>
    __device__ __forceinline__ static LdsType load_fragment(
        uint8_t *wsm_lds, const int (&lds_offset)[4], int idx) {
        return *reinterpret_cast<LdsType *>(wsm_lds + lds_offset[idx]);
    }

    template <typename ALdsType, typename BLdsType>
    __device__ __forceinline__ static void load_pair_stage(
        ALdsType (&a)[4][4], BLdsType (&b)[4][4], int dst_stage_idx,
        uint8_t *wsm_lds, const int (&a_lds_offset)[4],
        const int (&b_lds_offset)[4]) {
        a[dst_stage_idx][0] = load_fragment<ALdsType>(wsm_lds, a_lds_offset, 0);
        a[dst_stage_idx][1] = load_fragment<ALdsType>(wsm_lds, a_lds_offset, 1);
        a[dst_stage_idx][2] = load_fragment<ALdsType>(wsm_lds, a_lds_offset, 2);
        a[dst_stage_idx][3] = load_fragment<ALdsType>(wsm_lds, a_lds_offset, 3);
        b[dst_stage_idx][0] = load_fragment<BLdsType>(wsm_lds, b_lds_offset, 0);
        b[dst_stage_idx][1] = load_fragment<BLdsType>(wsm_lds, b_lds_offset, 1);
        b[dst_stage_idx][2] = load_fragment<BLdsType>(wsm_lds, b_lds_offset, 2);
        b[dst_stage_idx][3] = load_fragment<BLdsType>(wsm_lds, b_lds_offset, 3);
    }
};

} // namespace bf16_c500_tk_cute_local::cute_tk
