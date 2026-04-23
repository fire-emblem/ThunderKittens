#pragma once

// Stage geometry descriptor for tiled GEMM
// Encapsulates load/store offsets for A and B matrices across pipeline stages
// This is an abstract primitive - kernel-specific instantiations are in kernel/

namespace bf16_c500_tk_cute_local::primitives {

// Base stage geometry for standard GEMM
template <typename ALdgType, typename BLdgType, typename ALdsType,
          typename BLdsType>
struct stage_geometry_t {
    int a_ldg_offset[2][4];  // Global load offsets for A (2 banks x 4 stages)
    int b_ldg_offset[2][4];  // Global load offsets for B (2 banks x 4 stages)
    int a_lds_offset[4];     // Shared load offsets for A (4 fragments)
    int b_lds_offset[4];     // Shared load offsets for B (4 fragments)
};

// Extended stage geometry for swizzled TN layout (includes predicate operands)
template <typename ALdgType, typename BLdgType, typename ALdsType,
          typename BLdsType>
struct swizzled_tn_geometry_t : stage_geometry_t<ALdgType, BLdgType, ALdsType, BLdsType> {
    int a_cmp_op1;  // Predicate operand for A matrix
    int b_cmp_op1;  // Predicate operand for B matrix
};

} // namespace bf16_c500_tk_cute_local::primitives