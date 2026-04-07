#pragma once

#include "fragment_layouts.cuh"

namespace kittens::arch::c500 {

// Transitional note for new GEMM code:
// use arch/c500/primitives/*.cuh as the stable backend entry layer.

template<typename Atom>
struct fragment_a;

template<typename Atom>
struct fragment_b;

template<typename Atom>
struct fragment_c;

template<typename Input>
struct fragment_a<mma_input_16x16x16_fp32<Input>> {
    uint32_t reg[mma_input_16x16x16_fp32<Input>::a_registers];
};

template<typename Input>
struct fragment_b<mma_input_16x16x16_fp32<Input>> {
    uint32_t reg[mma_input_16x16x16_fp32<Input>::b_registers];
};

template<typename Input>
struct fragment_c<mma_input_16x16x16_fp32<Input>> {
    float reg[mma_input_16x16x16_fp32<Input>::c_registers];
};

template<typename Atom>
__device__ inline void mma(fragment_c<Atom>& d,
                           const fragment_a<Atom>& a,
                           const fragment_b<Atom>& b,
                           const fragment_c<Atom>& c);

namespace detail {

using native_ab_vector = __NATIVE_VECTOR__(2, uint32_t);
using native_c_vector = __NATIVE_VECTOR__(4, float);

__device__ inline native_c_vector mma_native_bf16(const fragment_a<mma_bf16_16x16x16_fp32> &a,
                                                  const fragment_b<mma_bf16_16x16x16_fp32> &b,
                                                  const fragment_c<mma_bf16_16x16x16_fp32> &c) {
    const native_ab_vector native_a{a.reg[0], a.reg[1]};
    const native_ab_vector native_b{b.reg[0], b.reg[1]};
    const native_c_vector native_c{c.reg[0], c.reg[1], c.reg[2], c.reg[3]};
    return __builtin_mxc_mma_16x16x16bf16(native_b, native_a, native_c);
}

} // namespace detail

template<>
__device__ inline void mma<mma_bf16_16x16x16_fp32>(fragment_c<mma_bf16_16x16x16_fp32> &d,
                                                   const fragment_a<mma_bf16_16x16x16_fp32> &a,
                                                   const fragment_b<mma_bf16_16x16x16_fp32> &b,
                                                   const fragment_c<mma_bf16_16x16x16_fp32> &c) {
    const auto result = detail::mma_native_bf16(a, b, c);

#pragma unroll
    for (int i = 0; i < mma_bf16_16x16x16_fp32::c_registers; ++i) {
        d.reg[i] = result[i];
    }
}

} // namespace kittens::arch::c500
