#pragma once

#include "mma_atoms.cuh"

namespace kittens::arch::c500 {

template<typename Atom, typename OutputTile>
__device__ inline void store_c(OutputTile& dst,
                               const fragment_c<Atom>& src,
                               int tile_m,
                               int tile_n);

namespace detail {

template<typename Atom>
__device__ inline void scatter_native_accumulator(float2 &d0,
                                                  float2 &d1,
                                                  const fragment_c<Atom> &src) {
    const int lane = kittens::laneid();
    const int row = lane & 0x0f;
    const int lane_group = lane >> 4;
    const float r0 = src.reg[0];
    const float r1 = src.reg[1];
    const float r2 = src.reg[2];
    const float r3 = src.reg[3];
    const uint64_t mask = 0xffffffffffffffffull;
    constexpr int width = Atom::wave_size;

    const float src0_r0 = __shfl_sync(mask, r0, row + 0 * 16, width);
    const float src1_r0 = __shfl_sync(mask, r0, row + 1 * 16, width);
    const float src2_r0 = __shfl_sync(mask, r0, row + 2 * 16, width);
    const float src3_r0 = __shfl_sync(mask, r0, row + 3 * 16, width);
    const float src0_r1 = __shfl_sync(mask, r1, row + 0 * 16, width);
    const float src1_r1 = __shfl_sync(mask, r1, row + 1 * 16, width);
    const float src2_r1 = __shfl_sync(mask, r1, row + 2 * 16, width);
    const float src3_r1 = __shfl_sync(mask, r1, row + 3 * 16, width);
    const float src0_r2 = __shfl_sync(mask, r2, row + 0 * 16, width);
    const float src1_r2 = __shfl_sync(mask, r2, row + 1 * 16, width);
    const float src2_r2 = __shfl_sync(mask, r2, row + 2 * 16, width);
    const float src3_r2 = __shfl_sync(mask, r2, row + 3 * 16, width);
    const float src0_r3 = __shfl_sync(mask, r3, row + 0 * 16, width);
    const float src1_r3 = __shfl_sync(mask, r3, row + 1 * 16, width);
    const float src2_r3 = __shfl_sync(mask, r3, row + 2 * 16, width);
    const float src3_r3 = __shfl_sync(mask, r3, row + 3 * 16, width);

    if (lane_group == 0) {
        d0.x = src0_r0;
        d0.y = src1_r0;
        d1.x = src2_r0;
        d1.y = src3_r0;
    } else if (lane_group == 1) {
        d0.x = src0_r1;
        d0.y = src1_r1;
        d1.x = src2_r1;
        d1.y = src3_r1;
    } else if (lane_group == 2) {
        d0.x = src0_r2;
        d0.y = src1_r2;
        d1.x = src2_r2;
        d1.y = src3_r2;
    } else {
        d0.x = src0_r3;
        d0.y = src1_r3;
        d1.x = src2_r3;
        d1.y = src3_r3;
    }
}

} // namespace detail

template<typename Atom, ducks::rt::row_layout OutputTile>
__device__ inline void store_c(OutputTile& dst,
                               const fragment_c<Atom>& src,
                               int tile_m,
                               int tile_n) {
    static_assert(std::is_same_v<typename OutputTile::T, float>,
                  "C500 epilogue atom currently exports to float accumulator tiles only.");
    if constexpr (std::is_same_v<Atom, mma_bf16_16x16x16_fp32> ||
                  std::is_same_v<Atom, mma_f16_16x16x16_fp32>) {
        detail::scatter_native_accumulator<Atom>(dst.tiles[tile_m][tile_n].data[0],
                                                 dst.tiles[tile_m][tile_n].data[1],
                                                 src);
    } else {
        static_assert(sizeof(Atom) == 0, "Unsupported C500 store_c atom.");
    }
}

} // namespace kittens::arch::c500
