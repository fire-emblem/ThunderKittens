#pragma once

#include "../host/layout_traits.cuh"
#include "../kernel/layoutc_geometry.cuh"

namespace bf16_c500_tk_cute_local::cute_tk {

struct layoutc_layout_atom {
    using host_layout = ::bf16_c500_tk_local::host::layoutc_host_traits;

    template <typename ALdgType, typename BLdgType, typename ALdsType,
              typename BLdsType, typename T>
    __host__ __device__ static auto
    make_stage_geometry(int tid, int lane, int slot, int lda, int n) {
        return ::bf16_c500_tk_local::kernel::make_layoutc_stage_geometry<
            ALdgType, BLdgType, ALdsType, BLdsType, T>(tid, lane, slot, lda, n);
    }
};

struct continuousc_layout_atom {
    using host_layout = ::bf16_c500_tk_local::host::continuousc_host_traits;

    template <typename ALdgType, typename BLdgType, typename ALdsType,
              typename BLdsType, typename T>
    __host__ __device__ static auto
    make_stage_geometry(int tid, int lane, int slot, int lda, int n) {
        return ::bf16_c500_tk_local::kernel::make_layoutc_stage_geometry<
            ALdgType, BLdgType, ALdsType, BLdsType, T>(tid, lane, slot, lda, n);
    }
};

} // namespace bf16_c500_tk_cute_local::cute_tk
