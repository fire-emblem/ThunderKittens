#pragma once

#include "../sync.cuh"

namespace bf16_c500_tk_cute_local::cute_tk {

struct sync_atom {
    // Wait for N global memory async operations to complete
    // Direct mapping to MXC arrive_gvmcnt
    template <int Num>
    __device__ __forceinline__ static void arrive_gvmcnt() {
        ::bf16_c500_tk_local::primitives::arrive_gvmcnt<Num>();
    }

    // Legacy alias - prefer arrive_gvmcnt for clarity
    template <int Num>
    __device__ __forceinline__ static void wait_gmem_async() {
        arrive_gvmcnt<Num>();
    }

    // Block-level barrier
    __device__ __forceinline__ static void barrier() {
        ::bf16_c500_tk_local::primitives::barrier();
    }
};

} // namespace bf16_c500_tk_cute_local::cute_tk
