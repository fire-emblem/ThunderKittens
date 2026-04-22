#pragma once

#include "store_ops_atom.cuh"

namespace bf16_c500_tk_cute_local::cute_tk {

struct bias_atom {
    template <typename CStgType, bool HasOneDimBias>
    __device__ __forceinline__ static void load_layoutc_bias(
        CStgType (&bias_load)[4], const void *bias, int start_row, int slot,
        int lane) {
        primitives::bias_load_atom::load_layoutc_bias<CStgType, HasOneDimBias>(
            bias_load, bias, start_row, slot, lane);
    }
};

} // namespace bf16_c500_tk_cute_local::cute_tk
