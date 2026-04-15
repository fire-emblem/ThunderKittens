#pragma once

#include <type_traits>

#include <maca.h>
#include <maca_bfloat16.h>
#include <maca_fp16.h>

namespace bf16_c500_tk_cute_local::cute_tk {

struct store_atom {
    template <typename Tc, typename Tscal, typename Float4, bool IsBetaZero>
    __device__ __forceinline__ static void store_layoutc_fragment(
        Tc *c_ptr, Float4 const &frag, int src_m, int src_n,
        int warp_rows_group_begin, int split_n_start, int idx_a, int j,
        int quarter_warp_id, int quarter_lane_id, Tscal alpha, Tscal beta) {
        (void)src_m;
        using c_store_t = __NATIVE_VECTOR__(sizeof(Tc), uint);
        const int row_n = (split_n_start + j) * 16 + quarter_lane_id;
        const int col_m_base =
            (warp_rows_group_begin + idx_a) * 16 + quarter_warp_id * 4;
        const size_t m_blk = static_cast<size_t>(col_m_base) / 32;
        const size_t n_blk = static_cast<size_t>(row_n) / 16;
        const size_t mma_col = (static_cast<size_t>(col_m_base) % 32) / 8;
        const size_t lane_row = static_cast<size_t>(row_n) % 16;
        const size_t lane_col = static_cast<size_t>(col_m_base) % 8;
        const size_t raw_idx_base =
            ((((m_blk * (src_n / 16) + n_blk) * 4 + mma_col) * 16 + lane_row) *
                 8 +
             lane_col);

        float out_f32[4];
        Tc out_tc[4];
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            out_f32[t] = frag[t] * alpha;
        }
        if constexpr (!IsBetaZero) {
            c_store_t c_old =
                *reinterpret_cast<c_store_t *>(c_ptr + raw_idx_base);
            Tc *c_old_tc = reinterpret_cast<Tc *>(&c_old);
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                out_f32[t] += beta * static_cast<Tscal>(c_old_tc[t]);
            }
        }
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            out_tc[t] = static_cast<Tc>(out_f32[t]);
        }
        *reinterpret_cast<c_store_t *>(c_ptr + raw_idx_base) =
            *reinterpret_cast<c_store_t *>(&out_tc[0]);
    }

    template <typename Tc, typename Tscal, typename Float4, bool IsBetaZero,
              int SplitK>
    __device__ __forceinline__ static void store_continuousc_fragment(
        Tc *c_ptr, Float4 const &frag, int src_m, int src_n, int c_offset,
        int row_n, int alpha_idx, int beta_idx, int quarter_warp_id,
        int quarter_lane_id, Tscal alpha, Tscal beta) {
        (void)src_n;
        (void)alpha_idx;
        (void)beta_idx;
        (void)quarter_warp_id;
        (void)quarter_lane_id;
        using c_store_t = __NATIVE_VECTOR__(sizeof(Tc), uint);

        float out_f32[4];
        Tc out_tc[4];
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            out_f32[t] = frag[t] * alpha;
            if constexpr (!IsBetaZero) {
                out_f32[t] += beta * static_cast<Tscal>(
                    c_ptr[row_n * src_m + (c_offset % (src_m / 4)) * 4 + t]);
            }
            out_tc[t] = static_cast<Tc>(out_f32[t]);
        }

        c_store_t *c_vec = reinterpret_cast<c_store_t *>(c_ptr);
        if constexpr (SplitK > 1) {
            if constexpr (std::is_same_v<Tc, __half>) {
                atomicAdd(reinterpret_cast<__half2 *>(&c_vec[c_offset]),
                          {out_tc[0], out_tc[1]});
                atomicAdd(reinterpret_cast<__half2 *>(&c_vec[c_offset]) + 1,
                          {out_tc[2], out_tc[3]});
            } else if constexpr (std::is_same_v<Tc, __maca_bfloat16>) {
                atomicAdd(reinterpret_cast<__maca_bfloat162 *>(&c_vec[c_offset]),
                          {out_tc[0], out_tc[1]});
                atomicAdd(reinterpret_cast<__maca_bfloat162 *>(&c_vec[c_offset]) + 1,
                          {out_tc[2], out_tc[3]});
            } else {
                c_vec[c_offset] = *reinterpret_cast<c_store_t *>(&out_tc[0]);
            }
        } else {
            c_vec[c_offset] = *reinterpret_cast<c_store_t *>(&out_tc[0]);
        }
    }
};

} // namespace bf16_c500_tk_cute_local::cute_tk
