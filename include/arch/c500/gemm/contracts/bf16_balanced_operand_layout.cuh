#pragma once

#include "bf16_balanced_stage_layout.cuh"
#include "../../layouts/native_stage_layout.cuh"

namespace kittens::arch::c500::gemm::contracts {

struct bf16_balanced_operand_layout : public kittens::arch::c500::gemm::bf16_native_stage_layout {
    using native_layout = kittens::arch::c500::gemm::bf16_native_stage_layout;
    using stage_layout = bf16_balanced_stage_layout;

    static constexpr int kWaveSize = kittens::arch::c500::gemm::bf16_native_stage_layout::kWaveSize;
    static constexpr int kLaneMn = native_layout::kMmaThreadMN;
    static constexpr int kLaneGroups = kWaveSize / kLaneMn;
    static constexpr int kLaneGroupKStride = native_layout::kMmaThreadK;
    static constexpr int kVecElems = native_layout::kLdsNumPerThread;
    static constexpr int kAtomsMN = native_layout::kMAtoms;
    static constexpr int kKGroups = kittens::arch::c500::gemm::bf16_native_stage_layout::kKGroups;
    static constexpr int kAGroupCount = stage_layout::kTileM / stage_layout::kWaveTileM;
    static constexpr int kBGroupCount = stage_layout::kTileN / stage_layout::kWaveTileN;
    static constexpr int kPackedHalfCount = 2;
    static constexpr int kGroupTileK = kVecElems;
    static constexpr int kAtomRowStride = kittens::arch::c500::gemm::bf16_native_stage_layout::kLdsRowStride;
    static constexpr int kAtomColStride = kittens::arch::c500::gemm::bf16_native_stage_layout::kLdsColStride;
    static constexpr int kAlignedKGroupStride = kPackedHalfCount;
    static constexpr int kAlignedMmaSteps = kKGroups / kAlignedKGroupStride;
    static constexpr int kAlignedTileK = kAlignedKGroupStride * kGroupTileK;
    static constexpr int kAsyncTransactionCount = kAtomsMN * kKGroups;
    static constexpr int kWaveTileM = stage_layout::kWaveTileM;
    static constexpr int kWaveTileN = stage_layout::kWaveTileN;
    static constexpr int kBlockTileK = stage_layout::kTileK;
    static constexpr int kWaveRowGroups = stage_layout::kTileM / stage_layout::kWaveTileM;
    static constexpr int kWaveColGroups = stage_layout::kTileN / stage_layout::kWaveTileN;

    __host__ __device__ static constexpr int lane_mn(int lane) { return lane & 0x0f; }
    __host__ __device__ static constexpr int lane_group(int lane) { return lane >> 4; }
    __host__ __device__ static constexpr int k_vec(int k_group, int lane) {
        return ((4 * k_group + lane_group(lane)) ^ lane_mn(lane));
    }
    __host__ __device__ static constexpr int k_elem(int k_group, int lane) {
        return k_vec(k_group, lane) * kVecElems;
    }
    __host__ __device__ static constexpr int a_row(int wave, int m, int lane) {
        return lane_mn(lane) + (wave / kWaveColGroups) * kLaneMn + m * kAtomRowStride;
    }
    __host__ __device__ static constexpr int b_col(int wave, int n, int lane) {
        return lane_mn(lane) + (wave % kWaveColGroups) * kLaneMn + n * kAtomColStride;
    }
};

static_assert(bf16_balanced_operand_layout::kWaveSize == bf16_balanced_operand_layout::native_layout::kWaveSize);
static_assert(bf16_balanced_operand_layout::kKGroups == bf16_balanced_operand_layout::native_layout::kKGroups);
static_assert(bf16_balanced_operand_layout::kAtomRowStride == bf16_balanced_operand_layout::native_layout::kLdsRowStride);
static_assert(bf16_balanced_operand_layout::kAtomColStride == bf16_balanced_operand_layout::native_layout::kLdsColStride);
static_assert(bf16_balanced_operand_layout::kWaveRowGroups * bf16_balanced_operand_layout::kWaveColGroups ==
              bf16_balanced_stage_layout::kWaveCount);
static_assert(bf16_balanced_operand_layout::kWaveRowGroups == bf16_balanced_operand_layout::kAGroupCount);
static_assert(bf16_balanced_operand_layout::kWaveColGroups == bf16_balanced_operand_layout::kBGroupCount);

} // namespace kittens::arch::c500::gemm::contracts
