#pragma once

#include "../mma.cuh"

namespace bf16_c500_tk_cute_local::cute_tk {

struct mma_atom {
    using float4_t = ::bf16_c500_tk_local::primitives::float4_native;

    template <typename T>
    __device__ __forceinline__ static float4_t
    fma_pair(uint a0, uint a1, uint b0, uint b1, float4_t c) {
        return ::bf16_c500_tk_local::primitives::mma_16x16x16_b16<T>(
            a0, a1, b0, b1, c);
    }

    template <typename T, int StageCount, int SharedNumCycleB, int APerWarp,
              typename FragType>
    __device__ __forceinline__ static void accumulate_reusea_tile(
        float4_t (&c_f32)[SharedNumCycleB][APerWarp],
        FragType const (&tmpA)[StageCount][APerWarp],
        int stage_idx,
        FragType const (&tmpB)[SharedNumCycleB]) {
        for (int j = 0; j < SharedNumCycleB; ++j) {
            for (int idxA = 0; idxA < APerWarp; ++idxA) {
                c_f32[j][idxA] = fma_pair<T>(
                    tmpA[stage_idx][idxA][0], tmpA[stage_idx][idxA][1],
                    tmpB[j][0], tmpB[j][1], c_f32[j][idxA]);
                c_f32[j][idxA] = fma_pair<T>(
                    tmpA[stage_idx][idxA][2], tmpA[stage_idx][idxA][3],
                    tmpB[j][2], tmpB[j][3], c_f32[j][idxA]);
            }
        }
    }

    template <typename T, typename BFragType, typename AFragType, typename CFragType>
    __device__ __forceinline__ static CFragType
    accumulate_kgroup(const BFragType (&b_frag)[4],
                      const AFragType (&a_frag)[4],
                      CFragType c) {
        c = fma_pair<T>(b_frag[0][0], b_frag[0][1], a_frag[0][0],
                        a_frag[0][1], c);
        c = fma_pair<T>(b_frag[0][2], b_frag[0][3], a_frag[0][2],
                        a_frag[0][3], c);
        c = fma_pair<T>(b_frag[1][0], b_frag[1][1], a_frag[1][0],
                        a_frag[1][1], c);
        c = fma_pair<T>(b_frag[1][2], b_frag[1][3], a_frag[1][2],
                        a_frag[1][3], c);
        c = fma_pair<T>(b_frag[2][0], b_frag[2][1], a_frag[2][0],
                        a_frag[2][1], c);
        c = fma_pair<T>(b_frag[2][2], b_frag[2][3], a_frag[2][2],
                        a_frag[2][3], c);
        c = fma_pair<T>(b_frag[3][0], b_frag[3][1], a_frag[3][0],
                        a_frag[3][1], c);
        c = fma_pair<T>(b_frag[3][2], b_frag[3][3], a_frag[3][2],
                        a_frag[3][3], c);
        return c;
    }
};

} // namespace bf16_c500_tk_cute_local::cute_tk
