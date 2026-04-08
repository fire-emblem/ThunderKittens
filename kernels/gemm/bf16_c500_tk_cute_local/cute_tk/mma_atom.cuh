#pragma once

#include "../kernel/layoutc_support.cuh"

namespace bf16_c500_tk_cute_local::cute_tk {

struct mma_atom {
    using float4_t = ::bf16_c500_tk_local::kernel::FLOAT4;

    template <typename T>
    __device__ __forceinline__ static float4_t
    fma_pair(uint a0, uint a1, uint b0, uint b1, float4_t c) {
        return ::bf16_c500_tk_local::kernel::mma_16x16x16b16<T>(
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
        return ::bf16_c500_tk_local::kernel::accumulate_layoutc_kgroup<T>(
            b_frag, a_frag, c);
    }
};

} // namespace bf16_c500_tk_cute_local::cute_tk
