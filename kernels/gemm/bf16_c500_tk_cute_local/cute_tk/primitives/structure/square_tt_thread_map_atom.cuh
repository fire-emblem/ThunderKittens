#pragma once

#include <maca.h>

#include "../../../contracts/square_tt_tile_contract.cuh"

namespace bf16_c500_tk_cute_local::cute_tk {

struct square_tt_thread_map_atom {
    using contract = ::bf16_c500_tk_cute_local::contracts::square_tt_tile_contract;

    using int4_t = __NATIVE_VECTOR__(4, int32_t);
    using ab_type = __NATIVE_VECTOR__(2, uint32_t);
    using a_ldg_type = __NATIVE_VECTOR__(2, uint32_t);
    using b_ldg_type = __NATIVE_VECTOR__(1, uint32_t);
    using sts_type = __NATIVE_VECTOR__(2, uint32_t);
    using lds_type = __NATIVE_VECTOR__(2, uint32_t);
    using stg_type = __NATIVE_VECTOR__(1, uint32_t);

    __device__ __forceinline__ static int sts_offset_bytes(int tidx) {
        return tidx / 256 * 2048 + tidx % 16 * 128 +
               (tidx / 16 + tidx % 16) % 16 * 8;
    }

    __device__ __forceinline__ static int lds_offset_bytes(int tidx, int lane_id,
                                                           int stage_group) {
        return (tidx % 16 + lane_id / 16 + 4 * stage_group) % 16 * 8 +
               (lane_id / 16 + 4 * stage_group) * 128;
    }

    __device__ __forceinline__ static int a_ldg_k(int tidx) {
        return (tidx % 16) * 4;
    }

    __device__ __forceinline__ static int a_ldg_m(int tidx, int row_limit,
                                                  int i) {
        return min(row_limit - 1, tidx / 16 + i * 32);
    }

    __device__ __forceinline__ static int b_ldg_k_base(int lane_id) {
        return lane_id / 16 * 4;
    }

    __device__ __forceinline__ static int b_ldg_n(int tidx, int wave_id,
                                                  int col_limit) {
        return min(col_limit - 2, tidx % 16 * 2 + wave_id * 32);
    }

    __device__ __forceinline__ static int epilogue_col(int tidx, int wave_id) {
        return wave_id * 32 + tidx % 16 * 2;
    }

    __device__ __forceinline__ static int epilogue_row_base(int lane_id) {
        return lane_id / 16 * 4;
    }
};

} // namespace bf16_c500_tk_cute_local::cute_tk
