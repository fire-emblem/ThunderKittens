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

} // namespace kittens::arch::c500::primitives
