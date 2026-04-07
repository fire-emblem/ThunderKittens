#include "testing_flags.cuh"

#ifdef TEST_C500_GEMM_BALANCED_CONTRACT_PROBE

#include "arch/c500/gemm/contracts/bf16_balanced_contracts.cuh"
#include "arch/c500/layouts/native_stage_layout.cuh"
#include "arch/c500/layouts/operand_layouts.cuh"

using contracts = kittens::arch::c500::gemm::contracts::bf16_balanced_128x128x128_stage4;
using stage_layout = contracts::stage_layout;
using operand_layout = contracts::operand_layout;
using legacy_stage_layout = kittens::arch::c500::gemm::bf16_128x128x128_stage_layout;
using native_operand_layout = kittens::arch::c500::gemm::bf16_native_stage_layout;

static_assert(contracts::kBlockM == 128);
static_assert(contracts::kBlockN == 128);
static_assert(contracts::kBlockK == 128);
static_assert(contracts::kStages == 4);
static_assert(contracts::kWaveSize == 64);
static_assert(stage_layout::kTileM == contracts::kBlockM);
static_assert(stage_layout::kTileN == contracts::kBlockN);
static_assert(stage_layout::kTileK == contracts::kBlockK);
static_assert(stage_layout::kTileM == legacy_stage_layout::kTileM);
static_assert(stage_layout::kTileN == legacy_stage_layout::kTileN);
static_assert(stage_layout::kTileK == legacy_stage_layout::kTileK);
static_assert(stage_layout::kStages == legacy_stage_layout::kStages);
static_assert(stage_layout::kThreads == legacy_stage_layout::kThreads);
static_assert(stage_layout::kWaveCount == legacy_stage_layout::kWaveCount);
static_assert(stage_layout::kStageBytes == legacy_stage_layout::kStageBytes);
static_assert(stage_layout::kAStageOffset == legacy_stage_layout::kAStageOffset);
static_assert(stage_layout::kBStageOffset == legacy_stage_layout::kBStageOffset);
static_assert(operand_layout::kWaveSize == contracts::kWaveSize);
static_assert(operand_layout::kBlockTileK == contracts::kBlockK);
static_assert(operand_layout::kGroupTileK == 8);
static_assert(operand_layout::kAtomRowStride == 32);
static_assert(operand_layout::kWaveSize == native_operand_layout::kWaveSize);
static_assert(operand_layout::kKGroups == native_operand_layout::kKGroups);
static_assert(operand_layout::kAtomRowStride == native_operand_layout::kLdsRowStride);
static_assert(operand_layout::kAtomColStride == native_operand_layout::kLdsColStride);
static_assert(operand_layout::kPackedHalfCount == 2);
static_assert(operand_layout::kAlignedMmaSteps == 2);
static_assert(operand_layout::kAlignedTileK == 16);
static_assert(operand_layout::kWaveRowGroups == 2);
static_assert(operand_layout::kWaveColGroups == 2);
static_assert(operand_layout::a_row(0, 0, 0) == 0);
static_assert(operand_layout::a_row(2, 0, 0) == 16);
static_assert(operand_layout::b_col(1, 0, 0) == 16);
static_assert(operand_layout::b_col(2, 0, 0) == 0);

#endif
