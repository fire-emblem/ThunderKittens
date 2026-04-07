#pragma once

namespace kittens::arch::c500::gemm {

struct bf16_128x128x128_stage_layout {
    static constexpr int kTileM = 128;
    static constexpr int kTileN = 128;
    static constexpr int kTileK = 128;

    static constexpr int kStages = 4;
    static constexpr int kThreads = 256;
    static constexpr int kWaveCount = 4;

    static constexpr int kWaveTileM = 64;
    static constexpr int kWaveTileN = 64;

    static constexpr int kStageBytes = 0x4000;
    static constexpr int kAStageOffset = 0x0000;
    static constexpr int kBStageOffset = 0x2000;
    static constexpr int kOperandStageBytes = kBStageOffset - kAStageOffset;

    __host__ __device__ static constexpr int stage_offset(int stage) {
        return stage * kStageBytes;
    }

    __host__ __device__ static constexpr int a_stage_offset(int stage) {
        return stage_offset(stage) + kAStageOffset;
    }

    __host__ __device__ static constexpr int b_stage_offset(int stage) {
        return stage_offset(stage) + kBStageOffset;
    }

    __host__ __device__ static constexpr int a_group_offset(int stage, int group) {
        return a_stage_offset(stage) + group * 0x1000;
    }

    __host__ __device__ static constexpr int b_group_offset(int stage, int group) {
        return b_stage_offset(stage) + group * 0x1000;
    }
};

} // namespace kittens::arch::c500::gemm
