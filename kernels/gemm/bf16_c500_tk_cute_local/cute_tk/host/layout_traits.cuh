#pragma once

#include "layout_pack.cuh"

namespace bf16_c500_tk_local::host {

struct layoutc_host_traits {
    template <typename NativeT, typename RowT, int M, int K>
    static std::vector<NativeT> pack_a_typed(
        const std::vector<RowT> &row_major_a) {
        return make_a_native<M, K, NativeT>(row_major_a);
    }

    template <int M, int K>
    static std::vector<bf16> pack_a(
        const std::vector<__nv_bfloat16> &row_major_a) {
        return pack_a_typed<bf16, __nv_bfloat16, M, K>(row_major_a);
    }

    template <typename NativeT, typename RowT, int K, int N>
    static std::vector<NativeT> pack_b_typed(
        const std::vector<RowT> &row_major_b) {
        return make_b_native<K, N, NativeT>(row_major_b);
    }

    template <int K, int N>
    static std::vector<bf16> pack_b(
        const std::vector<__nv_bfloat16> &row_major_b) {
        return pack_b_typed<bf16, __nv_bfloat16, K, N>(row_major_b);
    }

    template <typename NativeT, int M, int N>
    static float load_c_typed(const std::vector<NativeT> &raw_c, int row_n,
                              int col_m) {
        return load_layoutc_logical<M, N>(raw_c, row_n, col_m);
    }

    template <typename NativeT, typename RowT>
    static std::vector<NativeT> pack_a_runtime(
        int m, int k, const std::vector<RowT> &row_major_a) {
        return make_a_native_runtime<NativeT>(m, k, row_major_a);
    }

    template <typename NativeT, typename RowT>
    static std::vector<NativeT> pack_b_runtime(
        int k, int n, const std::vector<RowT> &row_major_b) {
        return make_b_native_runtime<NativeT>(k, n, row_major_b);
    }

    template <typename NativeT>
    static float load_c_runtime(const std::vector<NativeT> &raw_c, int m, int n,
                                int row_n, int col_m) {
        return load_layoutc_logical_runtime(raw_c, m, n, row_n, col_m);
    }

    template <int M, int N>
    static float load_c(const std::vector<bf16> &raw_c, int row_n, int col_m) {
        return load_c_typed<bf16, M, N>(raw_c, row_n, col_m);
    }
};

struct continuousc_host_traits {
    template <typename NativeT, typename RowT, int M, int K>
    static std::vector<NativeT> pack_a_typed(
        const std::vector<RowT> &row_major_a) {
        return make_a_native<M, K, NativeT>(row_major_a);
    }

    template <int M, int K>
    static std::vector<bf16> pack_a(
        const std::vector<__nv_bfloat16> &row_major_a) {
        return pack_a_typed<bf16, __nv_bfloat16, M, K>(row_major_a);
    }

    template <typename NativeT, typename RowT, int K, int N>
    static std::vector<NativeT> pack_b_typed(
        const std::vector<RowT> &row_major_b) {
        return make_b_native<K, N, NativeT>(row_major_b);
    }

    template <int K, int N>
    static std::vector<bf16> pack_b(
        const std::vector<__nv_bfloat16> &row_major_b) {
        return pack_b_typed<bf16, __nv_bfloat16, K, N>(row_major_b);
    }

    template <typename NativeT, int M, int N>
    static float load_c_typed(const std::vector<NativeT> &raw_c, int row_n,
                              int col_m) {
        return load_contiguous_logical<M, N>(raw_c, row_n, col_m);
    }

    template <typename NativeT, typename RowT>
    static std::vector<NativeT> pack_a_runtime(
        int m, int k, const std::vector<RowT> &row_major_a) {
        return make_a_native_runtime<NativeT>(m, k, row_major_a);
    }

    template <typename NativeT, typename RowT>
    static std::vector<NativeT> pack_b_runtime(
        int k, int n, const std::vector<RowT> &row_major_b) {
        return make_b_native_runtime<NativeT>(k, n, row_major_b);
    }

    template <typename NativeT>
    static float load_c_runtime(const std::vector<NativeT> &raw_c, int m, int n,
                                int row_n, int col_m) {
        return load_contiguous_logical_runtime(raw_c, m, row_n, col_m);
    }

    template <int M, int N>
    static float load_c(const std::vector<bf16> &raw_c, int row_n, int col_m) {
        return load_c_typed<bf16, M, N>(raw_c, row_n, col_m);
    }
};

struct square_tt_host_traits {
    template <typename NativeT, typename RowT, int M, int K>
    static std::vector<NativeT> pack_a_typed(
        const std::vector<RowT> &row_major_a) {
        return make_a_rowmajor_runtime<NativeT>(M, K, row_major_a);
    }

    template <typename NativeT, typename RowT, int K, int N>
    static std::vector<NativeT> pack_b_typed(
        const std::vector<RowT> &row_major_b) {
        return make_b_colmajor_runtime<NativeT>(K, N, row_major_b);
    }

    template <typename NativeT, int M, int N>
    static float load_c_typed(const std::vector<NativeT> &raw_c, int row_n,
                              int col_m) {
        return load_contiguous_logical<M, N>(raw_c, row_n, col_m);
    }

    template <typename NativeT, typename RowT>
    static std::vector<NativeT> pack_a_runtime(
        int m, int k, const std::vector<RowT> &row_major_a) {
        return make_a_rowmajor_runtime<NativeT>(m, k, row_major_a);
    }

    template <typename NativeT, typename RowT>
    static std::vector<NativeT> pack_b_runtime(
        int k, int n, const std::vector<RowT> &row_major_b) {
        return make_b_colmajor_runtime<NativeT>(k, n, row_major_b);
    }

    template <typename NativeT>
    static float load_c_runtime(const std::vector<NativeT> &raw_c, int m, int n,
                                int row_n, int col_m) {
        return load_contiguous_logical_runtime(raw_c, m, row_n, col_m);
    }
};

} // namespace bf16_c500_tk_local::host
