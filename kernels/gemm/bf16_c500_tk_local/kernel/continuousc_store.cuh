#pragma once

#include <maca.h>
#include <maca_bfloat16.h>
#include <maca_fp16.h>

namespace bf16_c500_tk_local::kernel {

template <typename Tc, typename Tscal, typename CVectorType, typename Float4,
          bool IsBetaZero, bool HasOneDimBias>
__device__ __forceinline__ void store_continuousc_tile(
    Tc *c_ptr,
    const Float4 (&c_f32)[4][4],
    int src_m,
    int src_n,
    int start_row,
    int start_col,
    int slot,
    int lane,
    Tscal alpha,
    Tscal beta,
    const CVectorType (&bias_load)[4]) {
    const int quarter_warp_id = lane >> 4;
    const int quarter_lane_id = lane & 15;
    const int warp_rows_group_begin = start_row / 16 + slot / 2 * 4;
    const int warp_cols_group_begin = start_col / 16 + (slot & 1) * 4;

#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = (warp_cols_group_begin + j) * 16 + quarter_lane_id;
            const int col_base = (warp_rows_group_begin + i) * 16 + quarter_warp_id * 4;
            if (row < src_n && (col_base + 3) < src_m) {
                float c_f32_res[4];
#pragma unroll 4
                for (int t = 0; t < 4; ++t) {
                    c_f32_res[t] = c_f32[i][j][t] * alpha;
                }
                if constexpr (!IsBetaZero) {
#pragma unroll 4
                    for (int t = 0; t < 4; ++t) {
                        c_f32_res[t] += beta * static_cast<Tscal>(
                            c_ptr[static_cast<size_t>(row) * src_m + col_base + t]);
                    }
                }
                Tc c_tc_tmp[4] = {0};
#pragma unroll 4
                for (int t = 0; t < 4; ++t) {
                    c_tc_tmp[t] = static_cast<Tc>(c_f32_res[t]);
                }
                if constexpr (HasOneDimBias) {
                    const Tc *bias_tc =
                        reinterpret_cast<const Tc *>(&bias_load[i]);
#pragma unroll 4
                    for (int t = 0; t < 4; ++t) {
                        c_tc_tmp[t] = __hadd(c_tc_tmp[t], bias_tc[t]);
                    }
                }
#pragma unroll 4
                for (int t = 0; t < 4; ++t) {
                    c_ptr[static_cast<size_t>(row) * src_m + col_base + t] =
                        c_tc_tmp[t];
                }
            }
        }
    }
}

} // namespace bf16_c500_tk_local::kernel
