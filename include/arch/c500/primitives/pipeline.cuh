#pragma once

#include "../async.cuh"

namespace kittens::arch::c500::primitives {

// Stable entry boundary for C500 GEMM pipeline primitives during the transition.
using kittens::arch::c500::async_token;
using kittens::arch::c500::combine;
using kittens::arch::c500::wait;
using kittens::arch::c500::wait_for_async_copies;

template<int RemainingOutstanding>
__device__ inline void wait_until() {
    kittens::arch::c500::wait_for_async_copies<RemainingOutstanding>();
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

} // namespace kittens::arch::c500::primitives
