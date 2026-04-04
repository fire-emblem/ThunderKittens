#pragma once

#include "mma_atoms.cuh"

namespace kittens::arch::c500 {

template<typename Atom, typename OutputTile>
__device__ inline void store_c(OutputTile& dst,
                               const fragment_c<Atom>& src,
                               int tile_m,
                               int tile_n);

} // namespace kittens::arch::c500
