#pragma once

#include "sync_atom.cuh"

namespace bf16_c500_tk_cute_local::cute_tk {

struct schedule_atom {
    template <typename SchedulePolicy>
    __device__ __forceinline__ static void maybe_sync_each_stage_issue() {
        if constexpr (requires { SchedulePolicy::sync_each_stage_issue; }) {
            if constexpr (SchedulePolicy::sync_each_stage_issue) {
                sync_atom::barrier();
            }
        }
    }

    template <int StageCount>
    __device__ __forceinline__ static void wait_prologue_stage0() {
        sync_atom::template wait_gmem_async<4 * (StageCount - 1)>();
        sync_atom::barrier();
    }

    template <int StageCount>
    __device__ __forceinline__ static void wait_prologue_stage1() {
        sync_atom::template wait_gmem_async<4 * (StageCount - 2)>();
        sync_atom::barrier();
    }

    template <int StageCount>
    __device__ __forceinline__ static void wait_steady_window() {
        sync_atom::template wait_gmem_async<4 * StageCount - 10>();
        sync_atom::barrier();
    }

    template <int SharedArriveCount, int APerWarp, int Stages>
    __device__ __forceinline__ static void wait_reusea_steady() {
        sync_atom::template wait_gmem_async<(SharedArriveCount + APerWarp) *
                                            (Stages - 1)>();
        sync_atom::barrier();
    }

    template <typename SchedulePolicy>
    __device__ __forceinline__ static void maybe_sync_before_tail_drain() {
        if constexpr (requires { SchedulePolicy::sync_before_tail_drain; }) {
            if constexpr (SchedulePolicy::sync_before_tail_drain) {
                sync_atom::barrier();
            }
        }
    }

    template <int StageCount>
    __device__ __forceinline__ static void wait_tail_stage(int stage_idx) {
        switch (stage_idx) {
            case 0:
                if constexpr (StageCount >= 2) {
                    sync_atom::template wait_gmem_async<4 * (StageCount - 2)>();
                    sync_atom::barrier();
                }
                break;
            case 1:
                if constexpr (StageCount >= 3) {
                    sync_atom::template wait_gmem_async<4 * (StageCount - 3)>();
                    sync_atom::barrier();
                }
                break;
            case 2:
                if constexpr (StageCount >= 4) {
                    sync_atom::template wait_gmem_async<4 * (StageCount - 4)>();
                    sync_atom::barrier();
                }
                break;
            default:
                break;
        }
    }

    template <int SharedArriveCount, int Stages>
    __device__ __forceinline__ static void wait_reusea_drain(int stage_idx) {
        switch (stage_idx) {
            case 0:
                if constexpr (Stages >= 1) {
                    sync_atom::template wait_gmem_async<(SharedArriveCount + 1) *
                                                        (Stages - 1)>();
                    sync_atom::barrier();
                }
                break;
            case 1:
                if constexpr (Stages >= 2) {
                    sync_atom::template wait_gmem_async<(SharedArriveCount + 1) *
                                                        (Stages - 2)>();
                    sync_atom::barrier();
                }
                break;
            case 2:
                if constexpr (Stages >= 3) {
                    sync_atom::template wait_gmem_async<(SharedArriveCount + 1) *
                                                        (Stages - 3)>();
                    sync_atom::barrier();
                }
                break;
            case 3:
                if constexpr (Stages >= 4) {
                    sync_atom::template wait_gmem_async<(SharedArriveCount + 1) *
                                                        (Stages - 4)>();
                    sync_atom::barrier();
                }
                break;
            case 4:
                if constexpr (Stages >= 5) {
                    sync_atom::template wait_gmem_async<(SharedArriveCount + 1) *
                                                        (Stages - 5)>();
                    sync_atom::barrier();
                }
                break;
            case 5:
                if constexpr (Stages >= 6) {
                    sync_atom::template wait_gmem_async<(SharedArriveCount + 1) *
                                                        (Stages - 6)>();
                    sync_atom::barrier();
                }
                break;
            case 6:
                if constexpr (Stages >= 7) {
                    sync_atom::template wait_gmem_async<(SharedArriveCount + 1) *
                                                        (Stages - 7)>();
                    sync_atom::barrier();
                }
                break;
            case 7:
                if constexpr (Stages >= 8) {
                    sync_atom::template wait_gmem_async<(SharedArriveCount + 1) *
                                                        (Stages - 8)>();
                    sync_atom::barrier();
                }
                break;
            default:
                break;
        }
    }
};

} // namespace bf16_c500_tk_cute_local::cute_tk
