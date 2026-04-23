#pragma once

#include <maca.h>
#include <maca_bfloat16.h>
#include <maca_fp16.h>

namespace bf16_c500_tk_cute_local::primitives {

// Epilogue bias load primitive
struct epilogue_bias_load_t {
    template <typename CStgType, bool HasOneDimBias>
    __device__ __forceinline__ static void load_layoutc_bias(
        CStgType (&bias_load)[4],
        const void *bias,
        int start_row,
        int slot,
        int lane) {
        if constexpr (HasOneDimBias) {
            for (int i = 0; i < 4; ++i) {
                const int bias_offset =
                    start_row / 16 * 4 + (lane / 16) + slot / 2 * 4 * 4 + i * 4;
                bias_load[i] =
                    (reinterpret_cast<const CStgType *>(bias))[bias_offset];
            }
        }
    }
};

// Layout-C store offset calculation
template <typename Tc, typename CStgType>
__device__ __forceinline__ size_t layoutc_store_offset(
    int src_n, int start_row, int start_col, int slot, int lane, int i, int j) {
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

// Layout-C tile store primitive
struct epilogue_layoutc_store_t {
    template <typename Tc, typename Tscal, typename CStgType, typename Float4,
              bool IsBetaZero, bool HasOneDimBias>
    __device__ __forceinline__ static void store_tile(
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
                const size_t c_offset = layoutc_store_offset<Tc, CStgType>(
                    src_n, start_row, start_col, slot, lane, i, j);
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
};

// Continuous-C tile store primitive
struct epilogue_continuousc_store_t {
    template <typename Tc, typename Tscal, typename CVectorType, typename Float4,
              bool IsBetaZero, bool HasOneDimBias>
    __device__ __forceinline__ static void store_tile(
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
};

} // namespace bf16_c500_tk_cute_local::primitives

// Backward compatibility aliases
namespace bf16_c500_tk_cute_local::cute_tk::primitives {
using bias_load_atom = ::bf16_c500_tk_cute_local::primitives::epilogue_bias_load_t;
using layoutc_store_atom = ::bf16_c500_tk_cute_local::primitives::epilogue_layoutc_store_t;
using continuousc_store_atom = ::bf16_c500_tk_cute_local::primitives::epilogue_continuousc_store_t;
using ::bf16_c500_tk_cute_local::primitives::layoutc_store_offset;
}

namespace bf16_c500_tk_local::kernel {
using ::bf16_c500_tk_cute_local::cute_tk::primitives::bias_load_atom;
using ::bf16_c500_tk_cute_local::cute_tk::primitives::continuousc_store_atom;
using ::bf16_c500_tk_cute_local::cute_tk::primitives::layoutc_store_atom;
using ::bf16_c500_tk_cute_local::cute_tk::primitives::layoutc_store_offset;
}