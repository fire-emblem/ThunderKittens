#pragma once

#include "../compute/mma_atom.cuh"
#include "sync_atom.cuh"

namespace bf16_c500_tk_cute_local::primitives {

// Pipeline mainloop primitive - main loop body composition
struct pipeline_mainloop_t {
    template <int SharedNumCycleB, typename SharedStage, typename BFragArray>
    __device__ __forceinline__ static void load_b_fragments(
        SharedStage const &shared_stage,
        int lane_id,
        BFragArray &tmp_b) {
        for (int j = 0; j < SharedNumCycleB; ++j) {
            tmp_b[j] = shared_stage[j][lane_id];
        }
    }

    template <typename T, int StageCount, int SharedNumCycleB, int APerWarp,
              typename FragType>
    __device__ __forceinline__ static void compute_stage(
        ::bf16_c500_tk_cute_local::cute_tk::mma_atom::float4_t (&c_f32)[SharedNumCycleB][APerWarp],
        FragType const (&tmp_a)[StageCount][APerWarp],
        int stage_idx,
        FragType const (&tmp_b)[SharedNumCycleB]) {
        ::bf16_c500_tk_cute_local::cute_tk::mma_atom::template accumulate_reusea_tile<T, StageCount, SharedNumCycleB,
                                                  APerWarp>(c_f32, tmp_a,
                                                            stage_idx, tmp_b);
    }

    template <int SharedArriveCount, int APerWarp, int Stages>
    __device__ __forceinline__ static void wait_steady() {
        pipeline_sync_t::template wait_gmem_async<(SharedArriveCount + APerWarp) *
                                            (Stages - 1)>();
    }

    template <int SharedArriveCount, int Stages>
    __device__ __forceinline__ static void wait_drain(int stage_idx) {
        switch (stage_idx) {
            case 0:
                if constexpr (Stages >= 1) {
                    pipeline_sync_t::template wait_gmem_async<(SharedArriveCount + 1) *
                                                        (Stages - 1)>();
                }
                break;
            case 1:
                if constexpr (Stages >= 2) {
                    pipeline_sync_t::template wait_gmem_async<(SharedArriveCount + 1) *
                                                        (Stages - 2)>();
                }
                break;
            case 2:
                if constexpr (Stages >= 3) {
                    pipeline_sync_t::template wait_gmem_async<(SharedArriveCount + 1) *
                                                        (Stages - 3)>();
                }
                break;
            case 3:
                if constexpr (Stages >= 4) {
                    pipeline_sync_t::template wait_gmem_async<(SharedArriveCount + 1) *
                                                        (Stages - 4)>();
                }
                break;
            case 4:
                if constexpr (Stages >= 5) {
                    pipeline_sync_t::template wait_gmem_async<(SharedArriveCount + 1) *
                                                        (Stages - 5)>();
                }
                break;
            case 5:
                if constexpr (Stages >= 6) {
                    pipeline_sync_t::template wait_gmem_async<(SharedArriveCount + 1) *
                                                        (Stages - 6)>();
                }
                break;
            case 6:
                if constexpr (Stages >= 7) {
                    pipeline_sync_t::template wait_gmem_async<(SharedArriveCount + 1) *
                                                        (Stages - 7)>();
                }
                break;
            case 7:
                if constexpr (Stages >= 8) {
                    pipeline_sync_t::template wait_gmem_async<(SharedArriveCount + 1) *
                                                        (Stages - 8)>();
                }
                break;
            default:
                break;
        }
    }
};

} // namespace bf16_c500_tk_cute_local::primitives

// Backward compatibility alias
namespace bf16_c500_tk_cute_local::cute_tk {
using mainloop_atom = ::bf16_c500_tk_cute_local::primitives::pipeline_mainloop_t;
}