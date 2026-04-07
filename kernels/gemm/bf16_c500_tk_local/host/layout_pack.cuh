#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <maca_bfloat16.h>

#include <type_traits>
#include <vector>

namespace bf16_c500_tk_local::host {

using bf16 = __maca_bfloat16;

template <typename NativeT, typename RowT>
__host__ inline NativeT cast_row_to_native(RowT value) {
    if constexpr (std::is_same_v<NativeT, bf16> &&
                  std::is_same_v<RowT, __nv_bfloat16>) {
        return static_cast<bf16>(value);
    } else if constexpr (std::is_same_v<NativeT, __half> &&
                         std::is_same_v<RowT, __half>) {
        return value;
    } else {
        return static_cast<NativeT>(value);
    }
}

template <typename NativeT>
__host__ inline float cast_native_to_float(NativeT value) {
    if constexpr (std::is_same_v<NativeT, bf16>) {
        return static_cast<float>(value);
    } else if constexpr (std::is_same_v<NativeT, __half>) {
        return __half2float(value);
    } else {
        return static_cast<float>(value);
    }
}

template<int M, int K, typename NativeT, typename RowT>
std::vector<NativeT> make_a_native(const std::vector<RowT> &row_major_a) {
    static_assert(M % 16 == 0 && K % 8 == 0);
    std::vector<NativeT> native(static_cast<size_t>(M) * K);
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            const size_t dst =
                ((((static_cast<size_t>(m) / 16) * (K / 8) + (k / 8)) * 16 + (m % 16)) * 8 + (k % 8));
            native[dst] = cast_row_to_native<NativeT>(row_major_a[static_cast<size_t>(m) * K + k]);
        }
    }
    return native;
}

template <typename NativeT, typename RowT>
std::vector<NativeT> make_a_native_runtime(int m, int k,
                                           const std::vector<RowT> &row_major_a) {
    std::vector<NativeT> native(static_cast<size_t>(m) * k);
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < k; ++col) {
            const size_t dst =
                ((((static_cast<size_t>(row) / 16) * (k / 8) + (col / 8)) * 16 +
                  (row % 16)) *
                     8 +
                 (col % 8));
            native[dst] =
                cast_row_to_native<NativeT>(row_major_a[static_cast<size_t>(row) * k + col]);
        }
    }
    return native;
}

template<int K, int N, typename NativeT, typename RowT>
std::vector<NativeT> make_b_native(const std::vector<RowT> &row_major_b) {
    static_assert(N % 16 == 0 && K % 32 == 0);
    std::vector<NativeT> native(static_cast<size_t>(K) * N);
    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            const size_t dst =
                (((((static_cast<size_t>(k) / 32) * (N / 16) + (n / 16)) * 4 + ((k % 32) / 8)) * 16 + (n % 16)) * 8 +
                 (k % 8));
            native[dst] = cast_row_to_native<NativeT>(row_major_b[static_cast<size_t>(n) * K + k]);
        }
    }
    return native;
}

template <typename NativeT, typename RowT>
std::vector<NativeT> make_b_native_runtime(int k, int n,
                                           const std::vector<RowT> &row_major_b) {
    std::vector<NativeT> native(static_cast<size_t>(k) * n);
    for (int row_k = 0; row_k < k; ++row_k) {
        for (int col_n = 0; col_n < n; ++col_n) {
            const size_t dst =
                (((((static_cast<size_t>(row_k) / 32) * (n / 16) + (col_n / 16)) * 4 +
                   ((row_k % 32) / 8)) *
                      16 +
                  (col_n % 16)) *
                     8 +
                 (row_k % 8));
            native[dst] = cast_row_to_native<NativeT>(
                row_major_b[static_cast<size_t>(col_n) * k + row_k]);
        }
    }
    return native;
}

template<int M, int N, typename NativeT>
float load_layoutc_logical(const std::vector<NativeT> &raw_c, int row_n, int col_m) {
    static_assert(M % 32 == 0 && N % 16 == 0);
    const size_t m_blk = static_cast<size_t>(col_m) / 32;
    const size_t n_blk = static_cast<size_t>(row_n) / 16;
    const size_t mma_col = (static_cast<size_t>(col_m) % 32) / 8;
    const size_t lane_row = static_cast<size_t>(row_n) % 16;
    const size_t lane_col = static_cast<size_t>(col_m) % 8;
    const size_t raw_idx =
        ((((m_blk * (N / 16) + n_blk) * 4 + mma_col) * 16 + lane_row) * 8 + lane_col);
    return cast_native_to_float(raw_c[raw_idx]);
}

template <typename NativeT>
float load_layoutc_logical_runtime(const std::vector<NativeT> &raw_c, int m,
                                   int n, int row_n, int col_m) {
    const size_t m_blk = static_cast<size_t>(col_m) / 32;
    const size_t n_blk = static_cast<size_t>(row_n) / 16;
    const size_t mma_col = (static_cast<size_t>(col_m) % 32) / 8;
    const size_t lane_row = static_cast<size_t>(row_n) % 16;
    const size_t lane_col = static_cast<size_t>(col_m) % 8;
    const size_t raw_idx =
        ((((m_blk * (n / 16) + n_blk) * 4 + mma_col) * 16 + lane_row) * 8 + lane_col);
    return cast_native_to_float(raw_c[raw_idx]);
}

template<int M, int N, typename NativeT>
float load_contiguous_logical(const std::vector<NativeT> &raw_c, int row_n, int col_m) {
    return cast_native_to_float(raw_c[static_cast<size_t>(row_n) * M + col_m]);
}

template <typename NativeT>
float load_contiguous_logical_runtime(const std::vector<NativeT> &raw_c, int m,
                                      int row_n, int col_m) {
    return cast_native_to_float(raw_c[static_cast<size_t>(row_n) * m + col_m]);
}

} // namespace bf16_c500_tk_local::host
