#pragma once

#include "fragment_atom.cuh"

namespace bf16_c500_tk_cute_local::primitives {

// Pipeline reload primitive - reload operations (alias for fragment)
// All reload operations are now unified under pipeline_fragment_t
struct pipeline_reload_t {
    template <typename LdsType>
    __device__ __forceinline__ static LdsType load_fragment(
        uint8_t *wsm_lds, const int (&lds_offset)[4], int idx) {
        return pipeline_fragment_t::load_from_shared<LdsType>(wsm_lds, lds_offset, idx);
    }

    template <typename ALdsType, typename BLdsType>
    __device__ __forceinline__ static void load_pair_stage(
        ALdsType (&a)[4][4], BLdsType (&b)[4][4], int dst_stage_idx,
        uint8_t *wsm_lds, const int (&a_lds_offset)[4],
        const int (&b_lds_offset)[4]) {
        pipeline_fragment_t::reload_stage<ALdsType, BLdsType>(
            a, b, dst_stage_idx, wsm_lds, a_lds_offset, b_lds_offset);
    }
};

} // namespace bf16_c500_tk_cute_local::primitives

// Backward compatibility alias
namespace bf16_c500_tk_cute_local::cute_tk {
using reload_atom = ::bf16_c500_tk_cute_local::primitives::pipeline_reload_t;
}