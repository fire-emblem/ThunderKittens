#pragma once

#include <maca.h>

namespace kittens::arch::c500 {

template<int Transactions>
struct async_token {
    static constexpr int transactions = Transactions;
};

template<int A, int B>
__device__ inline async_token<A + B> combine(async_token<A>, async_token<B>) {
    return {};
}

template<int RemainingOutstanding>
__device__ inline void wait_for_async_copies() {
    __builtin_mxc_arrive_gvmcnt(RemainingOutstanding);
    __builtin_mxc_barrier_inst();
}

template<int Transactions>
__device__ inline void wait(async_token<Transactions>) {
    wait_for_async_copies<0>();
}

template<typename T>
__device__ inline async_token<1> async_copy_128b(void *dst_shared_ptr,
                                                  const T *src,
                                                  int cmp_lhs,
                                                  int cmp_rhs) {
    __builtin_mxc_ldg_b128_bsm_predicator(
        dst_shared_ptr,
        const_cast<void *>(reinterpret_cast<const void *>(src)),
        0,
        true,
        true,
        false,
        true,
        cmp_lhs,
        cmp_rhs,
        MACA_ICMP_SLT);
    return {};
}

} // namespace kittens::arch::c500
