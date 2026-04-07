#pragma once

#include "../traits.cuh"

namespace kittens::arch::c500::primitives {

struct balanced_wave_traits {
    static constexpr int kWaveSize = kittens::arch::c500::wave_traits::kWaveSize;
    static constexpr int kLaneMask = kittens::arch::c500::wave_traits::kLaneMask;
    static constexpr int kLaneGroupWidth = kittens::arch::c500::wave_traits::kLaneGroupSize;

    __device__ static inline int lane_id() { return kittens::arch::c500::wave_traits::lane_id(); }
    __device__ static inline int wave_id() { return kittens::arch::c500::wave_traits::wave_id(); }
    __device__ static inline int lane_row(int lane) { return kittens::arch::c500::wave_traits::lane_row(lane); }
    __device__ static inline int lane_group(int lane) {
        return kittens::arch::c500::wave_traits::lane_group(lane);
    }
};

} // namespace kittens::arch::c500::primitives
