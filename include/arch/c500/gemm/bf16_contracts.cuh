#pragma once

namespace kittens::arch::c500::gemm {

struct bf16_contracts {
    static constexpr int kBlockM = 128;
    static constexpr int kBlockN = 128;
    static constexpr int kBlockK = 128;
    static constexpr int kWarpM = 64;
    static constexpr int kWarpN = 64;
    static constexpr int kThreads = 256;
    static constexpr int kStages = 4;
    static constexpr int kWaveM = 2;
    static constexpr int kWaveN = 2;
    static constexpr int kLoadGroups = 2;
    static constexpr int kNumWorkers = 4;
};

} // namespace kittens::arch::c500::gemm
