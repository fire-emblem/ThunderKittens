#pragma once

#include <maca.h>

namespace bf16_c500_tk_cute_local::arch {

template <int Num>
__device__ __forceinline__ void arrive_gvmcnt() { __builtin_mxc_arrive_gvmcnt(Num); }
template <int Num>
__device__ __forceinline__ void arrive_bsmcnt() { __builtin_mxc_arrive_bsmcnt(Num); }
__device__ __forceinline__ void barrier() { __builtin_mxc_barrier_inst(); }

// Aliases for clearer naming
template <int N>
__device__ __forceinline__ void wait_gmem_async() { arrive_gvmcnt<N>(); }
template <int N>
__device__ __forceinline__ void wait_smem_async() { arrive_bsmcnt<N>(); }
template <int N>
__device__ __forceinline__ void arrive_gmem_async() { arrive_gvmcnt<N>(); }
template <int N>
__device__ __forceinline__ void arrive_smem_async() { arrive_bsmcnt<N>(); }

} // namespace bf16_c500_tk_cute_local::arch
