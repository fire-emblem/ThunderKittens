#pragma once

#include <maca.h>
#include <maca_bfloat16.h>
#include <maca_fp16.h>

namespace bf16_c500_tk_local::kernel {

template <typename Tc, typename CStgType>
__device__ __forceinline__ size_t layoutc_store_offset(int src_n, int start_row, int start_col, int slot, int lane, int i, int j) {
    const int quarter_warp_id = lane >> 4;
    const int quarter_lane_id = lane & 15;
    const int warp_store_offset =
        ((quarter_warp_id > 1 ? (quarter_warp_id + 30) : quarter_warp_id)) +
        quarter_lane_id * 2;
    const int warp_rows_group_begin = start_row / 16 + slot / 2 * 4;
    const int warp_cols_group_begin = start_col / 16 + (slot & 1) * 4;
    return ((warp_rows_group_begin + i) / 2) * (4 * 8 * 16 / 4) * (src_n / 16) +
           warp_store_offset +
           ((warp_rows_group_begin + i) % 2) * 64 +
           (warp_cols_group_begin + j) * 2 * 64;
}

template <typename Tc, typename Tscal, typename CStgType, typename Float4, bool IsBetaZero, bool HasOneDimBias>
__device__ __forceinline__ void store_layoutc_tile(
    CStgType *c_ptr,
    const Float4 (&c_f32)[4][4],
    int src_n,
    int start_row,
    int start_col,
    int slot,
    int lane,
    Tscal alpha,
    Tscal beta,
    const CStgType (&bias_load)[4]) {
#pragma unroll
    for (int j = 0; j < 4; j++) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const size_t c_offset = layoutc_store_offset<Tc, CStgType>(src_n, start_row, start_col, slot, lane, i, j);
            if ((start_col + (slot & 1) * 64 + j * 16) < src_n) {
                float c_f32_res[4];
#pragma unroll 4
                for (int t = 0; t < 4; t++) {
                    c_f32_res[t] = c_f32[i][j][t] * alpha;
                }
                if constexpr (!IsBetaZero) {
                    CStgType c_tmp = c_ptr[c_offset];
                    Tc *c_tmp_ptr = reinterpret_cast<Tc *>(&c_tmp);
                    c_f32_res[0] += beta * static_cast<Tscal>(c_tmp_ptr[0]);
                    c_f32_res[1] += beta * static_cast<Tscal>(c_tmp_ptr[1]);
                    c_f32_res[2] += beta * static_cast<Tscal>(c_tmp_ptr[2]);
                    c_f32_res[3] += beta * static_cast<Tscal>(c_tmp_ptr[3]);
                }
                Tc c_tc_tmp[4] = {0};
#pragma unroll 4
                for (int t = 0; t < 4; t++) {
                    c_tc_tmp[t] = static_cast<Tc>(c_f32_res[t]);
                }
                if constexpr (HasOneDimBias) {
                    Tc *bias_tc = reinterpret_cast<Tc *>(const_cast<CStgType *>(&bias_load[i]));
                    for (int t = 0; t < 4; t++) {
                        c_tc_tmp[t] = __hadd(c_tc_tmp[t], bias_tc[t]);
                    }
                }
                c_ptr[c_offset] = *reinterpret_cast<CStgType *>(&c_tc_tmp[0]);
            }
        }
    }
}

} // namespace bf16_c500_tk_local::kernel
