#pragma once

#include "../arch/async_copy.cuh"

namespace bf16_c500_tk_cute_local::primitives {

// Copy atom - abstract async copy primitive
// Wraps hardware async copy with a clean interface
struct copy_atom_t {
    // Issue 128-byte async global-to-shared load with predicate
    template <typename PredType = int>
    __device__ __forceinline__ static void issue_b128_g2s(
        void *saddr, const void *gaddr, PredType pred = 1) {
        ::bf16_c500_tk_cute_local::arch::ldg_b128_bsm_predicator(
            saddr, gaddr, pred, pred, pred, pred);
    }

    // Issue 64-byte async global-to-shared load with predicate
    template <typename PredType = int>
    __device__ __forceinline__ static void issue_b64_g2s(
        void *saddr, const void *gaddr, PredType pred = 1) {
        ::bf16_c500_tk_cute_local::arch::ldg_b64_bsm_predicator(
            saddr, gaddr, pred, pred, pred, pred);
    }

    // Issue 128-byte async global-to-register load
    __device__ __forceinline__ static float4_native load_global_b128(const void *ptr) {
        return ::bf16_c500_tk_cute_local::arch::load_global_async128(ptr);
    }
};

} // namespace bf16_c500_tk_cute_local::primitives