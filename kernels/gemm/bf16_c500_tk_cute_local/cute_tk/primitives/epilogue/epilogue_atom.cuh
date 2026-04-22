#pragma once

#include "store_ops_atom.cuh"
#include "bias_atom.cuh"
#include "store_atom.cuh"

namespace bf16_c500_tk_cute_local::cute_tk {

struct epilogue_atom {
    template <typename CStgType, bool HasOneDimBias>
    __device__ __forceinline__ static void load_bias(
        CStgType (&bias_load)[4],
        const void *bias,
        int start_row,
        int slot,
        int lane) {
        bias_atom::template load_layoutc_bias<CStgType, HasOneDimBias>(
            bias_load, bias, start_row, slot, lane);
    }

    template <typename Tc, typename Tscal, typename CStgType, typename Float4,
              bool IsBetaZero, bool HasOneDimBias>
    __device__ __forceinline__ static void store_layoutc(
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
        ::bf16_c500_tk_local::kernel::store_layoutc_tile<Tc, Tscal, CStgType, Float4,
                                                         IsBetaZero, HasOneDimBias>(
            c_ptr, c_f32, src_n, start_row, start_col, slot, lane, alpha, beta,
            bias_load);
    }

    template <typename Tc, typename Tscal, typename Float4, bool IsBetaZero>
    __device__ __forceinline__ static void store_layoutc_fragment(
        Tc *c_ptr,
        Float4 const &frag,
        int src_m,
        int src_n,
        int warp_rows_group_begin,
        int split_n_start,
        int idx_a,
        int j,
        int quarter_warp_id,
        int quarter_lane_id,
        Tscal alpha,
        Tscal beta) {
        store_atom::template store_layoutc_fragment<Tc, Tscal, Float4,
                                                    IsBetaZero>(
            c_ptr, frag, src_m, src_n, warp_rows_group_begin, split_n_start,
            idx_a, j, quarter_warp_id, quarter_lane_id, alpha, beta);
    }

    template <typename Tc, typename Tscal, typename Float4, bool IsBetaZero,
              int SplitK>
    __device__ __forceinline__ static void store_continuousc_fragment(
        Tc *c_ptr,
        Float4 const &frag,
        int src_m,
        int src_n,
        int c_offset,
        int row_n,
        int alpha_idx,
        int beta_idx,
        int quarter_warp_id,
        int quarter_lane_id,
        Tscal alpha,
        Tscal beta) {
        store_atom::template store_continuousc_fragment<Tc, Tscal, Float4,
                                                        IsBetaZero, SplitK>(
            c_ptr, frag, src_m, src_n, c_offset, row_n, alpha_idx, beta_idx,
            quarter_warp_id, quarter_lane_id, alpha, beta);
    }

    template <typename Tc, typename Tscal, typename CVecType, typename Float4,
              bool IsBetaZero, bool HasOneDimBias>
    __device__ __forceinline__ static void store_continuousc(
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
        const CVecType (&bias_load)[4]) {
        ::bf16_c500_tk_local::kernel::store_continuousc_tile<Tc, Tscal, CVecType,
                                                             Float4, IsBetaZero,
                                                             HasOneDimBias>(
            c_ptr, c_f32, src_m, src_n, start_row, start_col, slot, lane, alpha,
            beta, bias_load);
    }
};

} // namespace bf16_c500_tk_cute_local::cute_tk
