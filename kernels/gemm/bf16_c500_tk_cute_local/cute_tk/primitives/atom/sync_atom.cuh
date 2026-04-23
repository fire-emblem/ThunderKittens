#pragma once

#include "../arch/sync.cuh"

namespace bf16_c500_tk_cute_local::primitives {

// Sync atom - abstract synchronization primitive
// Wraps hardware barrier with a clean interface
struct sync_atom_t {
    // Wait for global memory async operations
    template <int N>
    __device__ __forceinline__ static void wait_gmem_async() {
        ::bf16_c500_tk_cute_local::arch::wait_gmem_async<N>();
    }

    // Wait for shared memory async operations
    template <int N>
    __device__ __forceinline__ static void wait_smem_async() {
        ::bf16_c500_tk_cute_local::arch::wait_smem_async<N>();
    }

    // Block-level barrier
    __device__ __forceinline__ static void barrier() {
        ::bf16_c500_tk_cute_local::arch::barrier();
    }

    // Arrive at global memory counter
    template <int N>
    __device__ __forceinline__ static void arrive_gmem_async() {
        ::bf16_c500_tk_cute_local::arch::arrive_gmem_async<N>();
    }

    // Arrive at shared memory counter
    template <int N>
    __device__ __forceinline__ static void arrive_smem_async() {
        ::bf16_c500_tk_cute_local::arch::arrive_smem_async<N>();
    }
};

} // namespace bf16_c500_tk_cute_local::primitives