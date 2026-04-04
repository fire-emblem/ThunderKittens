#pragma once

#include "../../common/base_types.cuh"

namespace kittens::arch::c500 {

template<typename Atom>
struct fragment_layout_traits;

// First native wave64 atom tag for the C500 hot path. This file keeps only
// atom traits and lane mapping; fragment payloads live in the backend headers.
template<typename Input>
struct mma_input_16x16x16_fp32 {
    using a_scalar = Input;
    using b_scalar = Input;
    using c_scalar = float;

    static constexpr int M = 16;
    static constexpr int N = 16;
    static constexpr int K = 16;
    static constexpr int wave_size = 64;
    static constexpr int a_registers = 2;
    static constexpr int b_registers = 2;
    static constexpr int c_registers = 4;
};

using mma_bf16_16x16x16_fp32 = mma_input_16x16x16_fp32<bf16>;
using mma_f16_16x16x16_fp32 = mma_input_16x16x16_fp32<half>;
struct bf16_mma_atom : mma_input_16x16x16_fp32<bf16> {};

template<typename Input>
struct fragment_layout_traits<mma_input_16x16x16_fp32<Input>> {
    static __device__ inline int lane_row(int lane) { return lane & 0x0f; }
    static __device__ inline int lane_group(int lane) { return lane >> 4; }
};

template<>
struct fragment_layout_traits<bf16_mma_atom> {
    static __device__ inline int lane_row(int lane) { return lane & 0x0f; }
    static __device__ inline int lane_group(int lane) { return lane >> 4; }
};

} // namespace kittens::arch::c500
