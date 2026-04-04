#pragma once

#include "fragments.cuh"

namespace kittens::arch::c500 {

template<typename>
inline constexpr bool unsupported_public_mma_atom_v = false;

namespace contract_detail {

__device__ inline fragment_a<mma_bf16_16x16x16_fp32>
to_internal(const fragment_a<bf16_mma_atom> &src) {
    fragment_a<mma_bf16_16x16x16_fp32> dst{};
    dst.reg[0] = src.reg[0];
    dst.reg[1] = src.reg[1];
    return dst;
}

__device__ inline fragment_b<mma_bf16_16x16x16_fp32>
to_internal(const fragment_b<bf16_mma_atom> &src) {
    fragment_b<mma_bf16_16x16x16_fp32> dst{};
    dst.reg[0] = src.reg[0];
    dst.reg[1] = src.reg[1];
    return dst;
}

__device__ inline fragment_c<mma_bf16_16x16x16_fp32>
to_internal(const fragment_c<bf16_mma_atom> &src) {
    fragment_c<mma_bf16_16x16x16_fp32> dst{};
#pragma unroll
    for (int i = 0; i < bf16_mma_atom::c_registers; ++i) {
        dst.reg[i] = src.reg[i];
    }
    return dst;
}

} // namespace contract_detail

template<typename Atom>
__device__ inline void mma(fragment_c<Atom> &,
                           const fragment_a<Atom> &,
                           const fragment_b<Atom> &,
                           const fragment_c<Atom> &) {
    static_assert(unsupported_public_mma_atom_v<Atom>,
                  "Unsupported C500 public mma atom.");
}

template<>
__device__ inline void mma<bf16_mma_atom>(fragment_c<bf16_mma_atom> &d,
                                          const fragment_a<bf16_mma_atom> &a,
                                          const fragment_b<bf16_mma_atom> &b,
                                          const fragment_c<bf16_mma_atom> &c) {
    const auto internal_a = contract_detail::to_internal(a);
    const auto internal_b = contract_detail::to_internal(b);
    const auto internal_c = contract_detail::to_internal(c);
    const auto result = detail::mma_native_bf16(internal_a, internal_b, internal_c);

#pragma unroll
    for (int i = 0; i < bf16_mma_atom::c_registers; ++i) {
        d.reg[i] = result[i];
    }
}

template<typename Atom>
__device__ inline fragment_c<Atom> mma(Atom,
                                       const fragment_a<Atom> &a,
                                       const fragment_b<Atom> &b,
                                       const fragment_c<Atom> &c) {
    fragment_c<Atom> out{};
    mma<Atom>(out, a, b, c);
    return out;
}

} // namespace kittens::arch::c500
