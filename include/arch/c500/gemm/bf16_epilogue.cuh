#pragma once

#include "../layouts/accumulator_export.cuh"

namespace kittens::arch::c500::gemm {

template<typename GlobalC, typename Accumulator>
__device__ inline void store_epilogue(const GlobalC &, const Accumulator &, int, int) {
    // Placeholder Task 3 epilogue contract; writeback lands with the mainloop path.
}

} // namespace kittens::arch::c500::gemm
