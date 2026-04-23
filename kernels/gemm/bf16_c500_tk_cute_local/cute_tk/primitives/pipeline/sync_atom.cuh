#pragma once

#include "../arch/sync.cuh"

namespace bf16_c500_tk_cute_local::primitives {

// Pipeline sync primitive - synchronization for pipeline stages
struct pipeline_sync_t {
    // Wait for N global memory async operations to complete
    template <int Num>
    __device__ __forceinline__ static void arrive_gvmcnt() {
        ::bf16_c500_tk_cute_local::arch::arrive_gvmcnt<Num>();
    }

    // Alias for clarity
    template <int Num>
    __device__ __forceinline__ static void wait_gmem_async() {
        arrive_gvmcnt<Num>();
    }

    // Block-level barrier
    __device__ __forceinline__ static void barrier() {
        ::bf16_c500_tk_cute_local::arch::barrier();
    }
};

} // namespace bf16_c500_tk_cute_local::primitives

// Backward compatibility alias
namespace bf16_c500_tk_cute_local::cute_tk {
using sync_atom = ::bf16_c500_tk_cute_local::primitives::pipeline_sync_t;
}