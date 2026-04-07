#pragma once

#include "../mma_atoms.cuh"

namespace kittens::arch::c500::primitives {

template<typename Atom>
using mma_fragment_a = kittens::arch::c500::fragment_a<Atom>;

template<typename Atom>
using mma_fragment_b = kittens::arch::c500::fragment_b<Atom>;

template<typename Atom>
using mma_fragment_c = kittens::arch::c500::fragment_c<Atom>;

using bf16_mma_16x16x16_fp32_atom = kittens::arch::c500::mma_bf16_16x16x16_fp32;
using bf16_fragment_a = mma_fragment_a<bf16_mma_16x16x16_fp32_atom>;
using bf16_fragment_b = mma_fragment_b<bf16_mma_16x16x16_fp32_atom>;
using bf16_fragment_c = mma_fragment_c<bf16_mma_16x16x16_fp32_atom>;

template<typename D, typename A, typename B, typename C>
__device__ inline void mma(D &dst, const A &a, const B &b, const C &c) {
    kittens::arch::c500::mma<bf16_mma_16x16x16_fp32_atom>(dst, a, b, c);
}

__device__ inline bf16_fragment_c mma(const bf16_fragment_a &a,
                                      const bf16_fragment_b &b,
                                      const bf16_fragment_c &c) {
    bf16_fragment_c dst{};
    mma(dst, a, b, c);
    return dst;
}

} // namespace kittens::arch::c500::primitives
