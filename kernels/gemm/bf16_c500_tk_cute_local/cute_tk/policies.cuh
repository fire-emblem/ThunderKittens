#pragma once

namespace bf16_c500_tk_cute_local::cute_tk {

template <int TileM_, int TileN_, int TileK_>
struct tile_shape_policy {
    static constexpr int tile_m = TileM_;
    static constexpr int tile_n = TileN_;
    static constexpr int tile_k = TileK_;
};

template <int StageCount_>
struct stage_count_policy {
    static constexpr int stage_count = StageCount_;
};

using tile_128x128x128 = tile_shape_policy<128, 128, 128>;
using stage_4 = stage_count_policy<4>;

template <int NTile, int APerWarp, int SplitN, int SplitK, int Stages_>
struct continuousc_reusea_schedule {
    static constexpr int kWaveSize = 64;
    static constexpr int kStages = Stages_;
    static constexpr int kElementsPerAccess = 8;
    static constexpr int kRowThreadsPerMma = 16;
    static constexpr int kColThreadsPerMma = 4;
    static constexpr int kElementsPerThreadPerMma = 4;
    static constexpr int kBlockDimX = 256;
    static constexpr int kWarpPerBlock = kBlockDimX / kWaveSize;
    static constexpr int kNumCycleB = NTile / kRowThreadsPerMma;
    static constexpr int kSharedNumCycleB =
        (kNumCycleB + SplitN - 1) / SplitN;
    static constexpr int kSharedArriveCount =
        (kSharedNumCycleB + kWarpPerBlock - 1) / kWarpPerBlock;

    struct runtime_state {
        int num_warps;
        int warp_id;
        int split_n_id;
        int split_k_id;
        int warp_id_in_block;
        int lane_id;
        int quarter_warp_id;
        int quarter_lane_id;
        int warp_rows_group_begin;
        int num_cycle_a;
        int split_n_start;
        int split_n_end;
        int split_k_start;
        int split_k_end;
        int end;
    };

    __device__ __forceinline__ static runtime_state make(int m, int k) {
        runtime_state state{};
        const int rows_group = m / kRowThreadsPerMma / APerWarp;

        state.num_warps = __builtin_mxc_readfirstlane(
            gridDim.x * blockDim.x / kWaveSize / SplitN / SplitK);
        state.warp_id = __builtin_mxc_readfirstlane(
                            (blockIdx.x * blockDim.x + threadIdx.x) / kWaveSize) %
                        state.num_warps;
        state.split_n_id = __builtin_mxc_readfirstlane(
            (((blockIdx.x * blockDim.x + threadIdx.x) / kWaveSize) /
             state.num_warps) %
            SplitN);
        state.split_k_id = __builtin_mxc_readfirstlane(
            (((blockIdx.x * blockDim.x + threadIdx.x) / kWaveSize) /
             state.num_warps / SplitN) %
            SplitK);
        state.warp_id_in_block = threadIdx.x / kWaveSize;
        state.lane_id = threadIdx.x & (kWaveSize - 1);
        state.quarter_warp_id = state.lane_id / 16;
        state.quarter_lane_id = state.lane_id & 15;

        state.warp_rows_group_begin =
            __builtin_mxc_readfirstlane(state.warp_id * (rows_group / state.num_warps) +
                                        min(state.warp_id, rows_group % state.num_warps)) *
            APerWarp;
        state.num_cycle_a = __builtin_mxc_readfirstlane(
            k / (kColThreadsPerMma * kElementsPerAccess));

        state.split_n_start = __builtin_mxc_readfirstlane(
            state.split_n_id * (kNumCycleB / SplitN) +
            min(state.split_n_id, (kNumCycleB % SplitN)));
        state.split_n_end = __builtin_mxc_readfirstlane(
            (state.split_n_id + 1) * (kNumCycleB / SplitN) +
            min(state.split_n_id + 1, (kNumCycleB % SplitN)));
        state.split_k_start = __builtin_mxc_readfirstlane(
            state.split_k_id * (state.num_cycle_a / kStages / SplitK) +
            min(state.split_k_id, (state.num_cycle_a / kStages) % SplitK));
        state.split_k_end = __builtin_mxc_readfirstlane(
            (state.split_k_id + 1) * (state.num_cycle_a / kStages / SplitK) +
            min(state.split_k_id + 1, (state.num_cycle_a / kStages) % SplitK));
        state.end = __builtin_mxc_readfirstlane(state.split_k_end - state.split_k_start);
        return state;
    }

    __device__ __forceinline__ static int initial_a_offset(runtime_state const &state,
                                                           int k) {
        return ((state.warp_rows_group_begin * kRowThreadsPerMma) *
                (k / kElementsPerAccess)) +
               state.lane_id + state.split_k_start * kStages * 64;
    }

    __device__ __forceinline__ static int initial_b_offset(runtime_state const &state) {
        return state.lane_id + state.split_n_start * 64 +
               state.split_k_start * kStages * kNumCycleB * 64;
    }

    __device__ __forceinline__ static int initial_c_offset(runtime_state const &state,
                                                           int idx_a,
                                                           int m) {
        return (state.quarter_lane_id + state.split_n_start * kRowThreadsPerMma) *
                   (m / kElementsPerThreadPerMma) +
               (state.warp_rows_group_begin + idx_a) * kColThreadsPerMma +
               state.quarter_warp_id;
    }

    __device__ __forceinline__ static int split_num_cycle_b(
        runtime_state const &state) {
        return __builtin_mxc_readfirstlane(state.split_n_end - state.split_n_start);
    }

    __device__ __forceinline__ static int advance_c_offset(int c_offset, int m) {
        return c_offset + kRowThreadsPerMma * (m / kElementsPerThreadPerMma);
    }
};

} // namespace bf16_c500_tk_cute_local::cute_tk
