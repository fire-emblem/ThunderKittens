#pragma once

#include "../traits.cuh"
#include "operand_layouts.cuh"

namespace kittens::arch::c500::gemm {

namespace detail {

constexpr int kLdsVectorBytes = 16;
constexpr int kLdsStepBytes = 0x400;
constexpr int kWavePairStrideBytes = 0x1000;

} // namespace detail

__host__ __device__ inline int lds_offset_a(int lane, int i) {
    const int slot = lane / wave_traits::kWaveSize;
    const int wave_lane = lane & (wave_traits::kWaveSize - 1);
    return bf16_128x128x128_stage_layout::kAStageOffset +
           (wave_lane +
            (slot / 2) * (detail::kWavePairStrideBytes / detail::kLdsVectorBytes) +
            i * (detail::kLdsStepBytes / detail::kLdsVectorBytes)) *
               detail::kLdsVectorBytes;
}

__host__ __device__ inline int lds_offset_b(int lane, int i) {
    const int slot = lane / wave_traits::kWaveSize;
    const int wave_lane = lane & (wave_traits::kWaveSize - 1);
    return bf16_128x128x128_stage_layout::kBStageOffset +
           (wave_lane +
            (slot & 1) * (detail::kWavePairStrideBytes / detail::kLdsVectorBytes) +
            i * (detail::kLdsStepBytes / detail::kLdsVectorBytes)) *
               detail::kLdsVectorBytes;
}

} // namespace kittens::arch::c500::gemm
