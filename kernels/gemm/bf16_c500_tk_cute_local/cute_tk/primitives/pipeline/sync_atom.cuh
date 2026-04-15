#pragma once

#include "../../../primitives/sync.cuh"

namespace bf16_c500_tk_cute_local::cute_tk {

struct sync_atom {
    template <int Num>
    __device__ __forceinline__ static void wait_gmem_async() {
        ::bf16_c500_tk_local::primitives::arrive_gvmcnt<Num>();
    }

    __device__ __forceinline__ static void barrier() {
        ::bf16_c500_tk_local::primitives::barrier();
    }
};

} // namespace bf16_c500_tk_cute_local::cute_tk
