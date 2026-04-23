#pragma once

#include "../arch/mma.cuh"
#include "../arch/mxc_builtins.cuh"

namespace bf16_c500_tk_cute_local::primitives {

// MMA atom - abstract matrix multiply-accumulate primitive
// Wraps hardware MMA with a clean interface
struct mma_atom_t {
    // 16x16x16 BF16 MMA operation
    template <typename AccumType>
    __device__ __forceinline__ static AccumType mma_16x16x16_bf16(
        bf16_native a, bf16_native b, AccumType c) {
        return ::bf16_c500_tk_cute_local::arch::mma_16x16x16_bf16(a, b, c);
    }

    // 16x16x16 FP16 MMA operation
    template <typename AccumType>
    __device__ __forceinline__ static AccumType mma_16x16x16_f16(
        half_native a, half_native b, AccumType c) {
        return ::bf16_c500_tk_cute_local::arch::mma_16x16x16_f16(a, b, c);
    }

    // FMA pair - compute two MMA operations
    template <typename AccumType, typename InputType>
    __device__ __forceinline__ static void fma_pair(
        AccumType (&c)[4][4],
        const InputType (&a)[4],
        const InputType (&b)[4]) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                c[i][j] = mma_16x16x16_bf16(a[i], b[j], c[i][j]);
            }
        }
    }

    // Accumulate k-group - triangular accumulation pattern
    template <typename AccumType, typename InputType, int StageCount>
    __device__ __forceinline__ static void accumulate_kgroup(
        AccumType (&c)[4][4],
        InputType (&a)[StageCount][4],
        InputType (&b)[StageCount][4],
        int stage_i) {
        // Stage 0: compute all 4 pairs
        fma_pair(c, a[0], b[0]);
        if constexpr (StageCount > 1) {
            if (stage_i >= 1) {
                fma_pair(c, a[1], b[1]);
            }
            if constexpr (StageCount > 2) {
                if (stage_i >= 2) {
                    fma_pair(c, a[2], b[2]);
                }
                if constexpr (StageCount > 3) {
                    if (stage_i >= 3) {
                        fma_pair(c, a[3], b[3]);
                    }
                }
            }
        }
    }
};

} // namespace bf16_c500_tk_cute_local::primitives