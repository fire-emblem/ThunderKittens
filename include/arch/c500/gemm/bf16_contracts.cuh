#pragma once

namespace kittens::arch::c500::gemm {

struct bf16_contracts {
    static constexpr int kBlockM = 128;
    static constexpr int kBlockN = 128;
    static constexpr int kBlockK = 128;
    static constexpr int kThreads = 256;
    static constexpr int kStages = 4;
    static constexpr int kWaveM = 2;
    static constexpr int kWaveN = 2;
};

} // namespace kittens::arch::c500::gemm
