#pragma once

#include "mma_atoms.cuh"

namespace kittens::arch::c500 {

template<typename Atom, typename SharedTile>
__device__ inline void load_a(fragment_a<Atom>& dst,
                              const SharedTile& src,
                              int tile_m,
                              int tile_k);

template<typename Atom, typename SharedTile>
__device__ inline void load_b(fragment_b<Atom>& dst,
                              const SharedTile& src,
                              int tile_k,
                              int tile_n);

} // namespace kittens::arch::c500
