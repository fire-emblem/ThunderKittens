#pragma once

#include <cuda_bf16.h>
#include <maca_bfloat16.h>

#include <vector>

namespace bf16_c500_tk_local::host {

using bf16 = __maca_bfloat16;

template<int M, int K>
std::vector<bf16> make_a_native(const std::vector<__nv_bfloat16> &row_major_a) {
    static_assert(M % 16 == 0 && K % 8 == 0);
    std::vector<bf16> native(static_cast<size_t>(M) * K);
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            const size_t dst =
                ((((static_cast<size_t>(m) / 16) * (K / 8) + (k / 8)) * 16 + (m % 16)) * 8 + (k % 8));
            native[dst] = static_cast<bf16>(row_major_a[static_cast<size_t>(m) * K + k]);
        }
    }
    return native;
}

template<int K, int N>
std::vector<bf16> make_b_native(const std::vector<__nv_bfloat16> &row_major_b) {
    static_assert(N % 16 == 0 && K % 32 == 0);
    std::vector<bf16> native(static_cast<size_t>(K) * N);
    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            const size_t dst =
                (((((static_cast<size_t>(k) / 32) * (N / 16) + (n / 16)) * 4 + ((k % 32) / 8)) * 16 + (n % 16)) * 8 +
                 (k % 8));
            native[dst] = static_cast<bf16>(row_major_b[static_cast<size_t>(n) * K + k]);
        }
    }
    return native;
}

template<int M, int N>
float load_layoutc_logical(const std::vector<bf16> &raw_c, int row_n, int col_m) {
    static_assert(M % 32 == 0 && N % 16 == 0);
    const size_t m_blk = static_cast<size_t>(col_m) / 32;
    const size_t n_blk = static_cast<size_t>(row_n) / 16;
    const size_t mma_col = (static_cast<size_t>(col_m) % 32) / 8;
    const size_t lane_row = static_cast<size_t>(row_n) % 16;
    const size_t lane_col = static_cast<size_t>(col_m) % 8;
    const size_t raw_idx =
        ((((m_blk * (N / 16) + n_blk) * 4 + mma_col) * 16 + lane_row) * 8 + lane_col);
    return static_cast<float>(raw_c[raw_idx]);
}

} // namespace bf16_c500_tk_local::host
