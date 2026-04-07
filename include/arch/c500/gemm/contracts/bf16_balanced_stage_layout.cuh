#pragma once

#include "../../layouts/operand_layouts.cuh"

namespace kittens::arch::c500::gemm::contracts {

struct bf16_balanced_stage_layout : public kittens::arch::c500::gemm::bf16_128x128x128_stage_layout {
    using legacy_layout = kittens::arch::c500::gemm::bf16_128x128x128_stage_layout;
    static constexpr int kLoadGroups = 2;
};

static_assert(bf16_balanced_stage_layout::kTileM == bf16_balanced_stage_layout::legacy_layout::kTileM);
static_assert(bf16_balanced_stage_layout::kTileN == bf16_balanced_stage_layout::legacy_layout::kTileN);
static_assert(bf16_balanced_stage_layout::kTileK == bf16_balanced_stage_layout::legacy_layout::kTileK);
static_assert(bf16_balanced_stage_layout::kStages == bf16_balanced_stage_layout::legacy_layout::kStages);
static_assert(bf16_balanced_stage_layout::kThreads == bf16_balanced_stage_layout::legacy_layout::kThreads);
static_assert(bf16_balanced_stage_layout::kWaveCount == bf16_balanced_stage_layout::legacy_layout::kWaveCount);
static_assert(bf16_balanced_stage_layout::kWaveTileM == bf16_balanced_stage_layout::legacy_layout::kWaveTileM);
static_assert(bf16_balanced_stage_layout::kWaveTileN == bf16_balanced_stage_layout::legacy_layout::kWaveTileN);
static_assert(bf16_balanced_stage_layout::kStageBytes == bf16_balanced_stage_layout::legacy_layout::kStageBytes);
static_assert(bf16_balanced_stage_layout::kAStageOffset == bf16_balanced_stage_layout::legacy_layout::kAStageOffset);
static_assert(bf16_balanced_stage_layout::kBStageOffset == bf16_balanced_stage_layout::legacy_layout::kBStageOffset);
static_assert(bf16_balanced_stage_layout::kOperandStageBytes ==
              bf16_balanced_stage_layout::legacy_layout::kOperandStageBytes);

} // namespace kittens::arch::c500::gemm::contracts
