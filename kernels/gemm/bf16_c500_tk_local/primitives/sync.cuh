#pragma once

#include <maca.h>

namespace bf16_c500_tk_local::primitives {

template <int Num>
__device__ __forceinline__ void arrive_gvmcnt() { __builtin_mxc_arrive_gvmcnt(Num); }
template <int Num>
__device__ __forceinline__ void arrive_bsmcnt() { __builtin_mxc_arrive_bsmcnt(Num); }
__device__ __forceinline__ void barrier() { __builtin_mxc_barrier_inst(); }

} // namespace bf16_c500_tk_local::primitives
