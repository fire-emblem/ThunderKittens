#pragma once

namespace bf16_c500_tk_cute_local::primitives {

// Square tile fragment primitive - fragment packing for 256x256x64 tiles
struct square_tile_fragment_t {
    template <typename BRegType, typename PackedType>
    __device__ __forceinline__ static void pack_b_quartet(
        BRegType (&dst)[2], PackedType const (&src)[4]) {
        dst[0][0] = __builtin_mxc_byte_perm(src[0], src[1], 0x01000504u);
        dst[1][0] = __builtin_mxc_byte_perm(src[0], src[1], 0x03020706u);
        dst[0][1] = __builtin_mxc_byte_perm(src[2], src[3], 0x01000504u);
        dst[1][1] = __builtin_mxc_byte_perm(src[2], src[3], 0x03020706u);
    }
};

} // namespace bf16_c500_tk_cute_local::primitives

// Backward compatibility alias
namespace bf16_c500_tk_cute_local::cute_tk {
using square_tt_fragment_atom = ::bf16_c500_tk_cute_local::primitives::square_tile_fragment_t;
}