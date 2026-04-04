#pragma once

#include "../epilogue_atoms.cuh"
#include "../layouts/accumulator_export.cuh"

namespace kittens::arch::c500::gemm {

template<typename OutputTile>
__device__ inline void store_epilogue(OutputTile &dst,
                                      const fragment_c<mma_bf16_16x16x16_fp32> &src,
                                      int tile_m,
                                      int tile_n) {
    store_c<mma_bf16_16x16x16_fp32>(dst, src, tile_m, tile_n);
}

} // namespace kittens::arch::c500::gemm
