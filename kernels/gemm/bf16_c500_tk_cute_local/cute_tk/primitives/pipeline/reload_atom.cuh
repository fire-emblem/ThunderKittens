#pragma once

#include "fragment_atom.cuh"

namespace bf16_c500_tk_cute_local::cute_tk {

// reload_atom is an alias for fragment_atom
// All reload operations are now unified under fragment_atom
// This header provides backward compatibility

struct reload_atom {
    template <typename LdsType>
    __device__ __forceinline__ static LdsType load_fragment(
        uint8_t *wsm_lds, const int (&lds_offset)[4], int idx) {
        return fragment_atom::load_from_shared<LdsType>(wsm_lds, lds_offset, idx);
    }

    template <typename ALdsType, typename BLdsType>
    __device__ __forceinline__ static void load_pair_stage(
        ALdsType (&a)[4][4], BLdsType (&b)[4][4], int dst_stage_idx,
        uint8_t *wsm_lds, const int (&a_lds_offset)[4],
        const int (&b_lds_offset)[4]) {
        fragment_atom::reload_stage<ALdsType, BLdsType>(
            a, b, dst_stage_idx, wsm_lds, a_lds_offset, b_lds_offset);
    }
};

} // namespace bf16_c500_tk_cute_local::cute_tk
