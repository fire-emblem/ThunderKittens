#pragma once

#include "../async.cuh"

namespace kittens::arch::c500::primitives {

// Stable entry boundary for C500 GEMM pipeline primitives during the transition.
using kittens::arch::c500::async_token;
using kittens::arch::c500::combine;
using kittens::arch::c500::wait;
using kittens::arch::c500::wait_for_async_copies;

__device__ inline void barrier() {
    __builtin_mxc_barrier_inst();
}

template<int RemainingOutstanding>
__device__ inline void arrive_gvmcnt() {
    __builtin_mxc_arrive_gvmcnt(RemainingOutstanding);
}

template<int RemainingOutstanding>
__device__ inline void wait_until() {
    arrive_gvmcnt<RemainingOutstanding>();
    barrier();
}

template<int TransactionsPerStage>
__device__ inline void wait_stage_window(int outstanding_stages) {
    switch (outstanding_stages) {
        case 0:
            wait_until<0>();
            break;
        case 1:
            wait_until<TransactionsPerStage>();
            break;
        case 2:
            wait_until<2 * TransactionsPerStage>();
            break;
        case 3:
            wait_until<3 * TransactionsPerStage>();
            break;
        default:
            wait_until<0>();
            break;
    }
}

template<int TransactionsPerStage>
__device__ inline void wait_stage_prefix(int prefetched_stages) {
    wait_stage_window<TransactionsPerStage>(prefetched_stages - 1);
}

template<int TransactionsPerStage>
__device__ inline void wait_stage_steady_state(int stages, int steps_issued) {
    const int outstanding_stages = max(0, min(stages - 1, steps_issued));
    wait_stage_window<TransactionsPerStage>(outstanding_stages);
}

} // namespace kittens::arch::c500::primitives
