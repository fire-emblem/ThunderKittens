#pragma once

#include "async_primitives.cuh"

namespace kittens::arch::c500 {

template<int RemainingOutstanding>
__device__ inline void wait_gvmcnt() {
    wait_until<RemainingOutstanding>();
}

template<typename T>
__device__ inline void async_copy_128b(void *dst_shared_ptr,
                                       const T *src,
                                       int cmp_lhs,
                                       int cmp_rhs) {
    detail::ldg_b128_bsm_pred(dst_shared_ptr, src, cmp_lhs, cmp_rhs);
}

} // namespace kittens::arch::c500
