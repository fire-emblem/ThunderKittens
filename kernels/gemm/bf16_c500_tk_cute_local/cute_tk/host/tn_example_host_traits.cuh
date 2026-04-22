#pragma once

#include "layout_pack.cuh"

namespace bf16_c500_tk_local::host {

struct tn_example_host_traits {
    template <typename NativeT, typename RowT>
    static std::vector<NativeT> pack_a_runtime(
        int m, int k, const std::vector<RowT> &row_major_a) {
        return make_a_rowmajor_runtime<NativeT>(m, k, row_major_a);
    }

    template <typename NativeT, typename RowT>
    static std::vector<NativeT> pack_b_runtime(
        int k, int n, const std::vector<RowT> &row_major_b) {
        std::vector<NativeT> native(static_cast<size_t>(k) * n);
        for (int row_n = 0; row_n < n; ++row_n) {
            for (int col_k = 0; col_k < k; ++col_k) {
                native[static_cast<size_t>(row_n) * k + col_k] =
                    cast_row_to_native<NativeT>(
                        row_major_b[static_cast<size_t>(row_n) * k + col_k]);
            }
        }
        return native;
    }

    template <typename NativeT>
    static float load_c_runtime(const std::vector<NativeT> &raw_c, int m, int,
                                int row_n, int col_m) {
        return load_contiguous_logical_runtime(raw_c, m, row_n, col_m);
    }
};

} // namespace bf16_c500_tk_local::host
