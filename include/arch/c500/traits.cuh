#pragma once

namespace kittens::arch::c500 {

struct wave_traits {
    static constexpr int kWaveSize = 64;
    static constexpr int kLaneMask = kWaveSize - 1;
    static constexpr int kLaneGroupSize = 16;

    __device__ static inline int lane_id() { return threadIdx.x & kLaneMask; }
    __device__ static inline int wave_id() { return threadIdx.x / kWaveSize; }
    __device__ static inline int lane_row(int lane) { return lane & 0x0f; }
    __device__ static inline int lane_group(int lane) { return lane >> 4; }
};

} // namespace kittens::arch::c500
