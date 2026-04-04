#pragma once

#include "fragments.cuh"

namespace kittens::arch::c500 {

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
