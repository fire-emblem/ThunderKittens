#pragma once

#include "store_ops_atom.cuh"

namespace bf16_c500_tk_cute_local::primitives {

// Epilogue bias primitive - bias loading wrapper
struct epilogue_bias_t {
    template <typename CStgType, bool HasOneDimBias>
    __device__ __forceinline__ static void load_layoutc_bias(
        CStgType (&bias_load)[4], const void *bias, int start_row, int slot,
        int lane) {
        epilogue_bias_load_t::load_layoutc_bias<CStgType, HasOneDimBias>(
            bias_load, bias, start_row, slot, lane);
    }
};

} // namespace bf16_c500_tk_cute_local::primitives

// Backward compatibility alias
namespace bf16_c500_tk_cute_local::cute_tk {
using bias_atom = ::bf16_c500_tk_cute_local::primitives::epilogue_bias_t;
}