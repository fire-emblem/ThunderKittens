#pragma once

#include "fragment_layouts.cuh"

namespace kittens::arch::c500 {

template<typename Atom>
struct fragment_a;

template<typename Atom>
struct fragment_b;

template<typename Atom>
struct fragment_c;

template<typename Input>
struct fragment_a<mma_input_16x16x16_fp32<Input>>;

template<typename Input>
struct fragment_b<mma_input_16x16x16_fp32<Input>>;

template<typename Input>
struct fragment_c<mma_input_16x16x16_fp32<Input>> {
    float reg[4];
};

template<typename Atom>
__device__ inline void mma(fragment_c<Atom>& d,
                           const fragment_a<Atom>& a,
                           const fragment_b<Atom>& b,
                           const fragment_c<Atom>& c);

} // namespace kittens::arch::c500
