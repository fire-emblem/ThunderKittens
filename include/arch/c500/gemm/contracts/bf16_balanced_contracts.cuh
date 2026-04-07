#pragma once

#include "bf16_balanced_operand_layout.cuh"
#include "bf16_balanced_stage_layout.cuh"

namespace kittens::arch::c500::gemm::contracts {

struct bf16_balanced_128x128x128_stage4 {
    using stage_layout = bf16_balanced_stage_layout;
    using operand_layout = bf16_balanced_operand_layout;

    static constexpr int kBlockM = stage_layout::kTileM;
    static constexpr int kBlockN = stage_layout::kTileN;
    static constexpr int kBlockK = stage_layout::kTileK;
    static constexpr int kStageK = operand_layout::kGroupTileK * operand_layout::kKGroups;
    static constexpr int kWarpM = stage_layout::kWaveTileM;
    static constexpr int kWarpN = stage_layout::kWaveTileN;
    static constexpr int kThreads = stage_layout::kThreads;
    static constexpr int kStages = stage_layout::kStages;
    static constexpr int kWaveSize = operand_layout::kWaveSize;
    static constexpr int kWaveM = kBlockM / kWarpM;
    static constexpr int kWaveN = kBlockN / kWarpN;
    static constexpr int kLoadGroups = stage_layout::kLoadGroups;
    static constexpr int kNumWorkers = kWaveM * kWaveN;
};

static_assert(bf16_balanced_128x128x128_stage4::kWaveM == 2);
static_assert(bf16_balanced_128x128x128_stage4::kWaveN == 2);
static_assert(bf16_balanced_128x128x128_stage4::kStageK == bf16_balanced_128x128x128_stage4::operand_layout::kAlignedTileK *
                                                           bf16_balanced_128x128x128_stage4::operand_layout::kAlignedMmaSteps);
static_assert(bf16_balanced_128x128x128_stage4::kNumWorkers ==
              bf16_balanced_128x128x128_stage4::stage_layout::kWaveCount);
static_assert(bf16_balanced_128x128x128_stage4::kThreads ==
              bf16_balanced_128x128x128_stage4::kNumWorkers *
                  bf16_balanced_128x128x128_stage4::kWaveSize);

} // namespace kittens::arch::c500::gemm::contracts
