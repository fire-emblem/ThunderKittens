#pragma once

namespace bf16_c500_tk_cute_local::cute_tk::kernel {

template <typename ALdgType, typename BLdgType, typename ALdsType,
          typename BLdsType>
struct tn_example_stage_geometry {
    int a_ldg_offset[2][4];
    int b_ldg_offset[2][4];
    int a_lds_offset[4];
    int b_lds_offset[4];
    int a_cmp_op1;
    int b_cmp_op1;
};

struct tn_example_swizzled_geometry {
    template <typename ALdgType, typename BLdgType, typename ALdsType,
              typename BLdsType>
    __device__ __forceinline__ static tn_example_stage_geometry<ALdgType, BLdgType,
                                                                ALdsType, BLdsType>
    make(int tid, int lane, int slot, int lda, int ldb, int m_a, int n_b) {
        tn_example_stage_geometry<ALdgType, BLdgType, ALdsType, BLdsType> g{};
        const int a_row = (tid / 16) * 4;
        const int a_col = (tid & 15) ^ (tid / 16);
        const int b_col = a_row;
        const int b_row = a_col;
        g.a_cmp_op1 = a_col;
        g.b_cmp_op1 = b_row;

        g.a_ldg_offset[0][0] =
            (a_col + lda * (a_row + 0 < m_a ? a_row + 0 : m_a - 1)) * sizeof(ALdgType);
        g.a_ldg_offset[0][1] =
            (a_col + lda * (a_row + 1 < m_a ? a_row + 1 : m_a - 1)) * sizeof(ALdgType);
        g.a_ldg_offset[0][2] =
            (a_col + lda * (a_row + 2 < m_a ? a_row + 2 : m_a - 1)) * sizeof(ALdgType);
        g.a_ldg_offset[0][3] =
            (a_col + lda * (a_row + 3 < m_a ? a_row + 3 : m_a - 1)) * sizeof(ALdgType);
        g.a_ldg_offset[1][0] =
            (a_col + lda * (a_row + 64 + 0 < m_a ? a_row + 64 + 0 : m_a - 1)) * sizeof(ALdgType);
        g.a_ldg_offset[1][1] =
            (a_col + lda * (a_row + 64 + 1 < m_a ? a_row + 64 + 1 : m_a - 1)) * sizeof(ALdgType);
        g.a_ldg_offset[1][2] =
            (a_col + lda * (a_row + 64 + 2 < m_a ? a_row + 64 + 2 : m_a - 1)) * sizeof(ALdgType);
        g.a_ldg_offset[1][3] =
            (a_col + lda * (a_row + 64 + 3 < m_a ? a_row + 64 + 3 : m_a - 1)) * sizeof(ALdgType);

        g.b_ldg_offset[0][0] =
            (b_row + ldb * (b_col + 0 < n_b ? b_col + 0 : n_b - 1)) * sizeof(BLdgType);
        g.b_ldg_offset[0][1] =
            (b_row + ldb * (b_col + 1 < n_b ? b_col + 1 : n_b - 1)) * sizeof(BLdgType);
        g.b_ldg_offset[0][2] =
            (b_row + ldb * (b_col + 2 < n_b ? b_col + 2 : n_b - 1)) * sizeof(BLdgType);
        g.b_ldg_offset[0][3] =
            (b_row + ldb * (b_col + 3 < n_b ? b_col + 3 : n_b - 1)) * sizeof(BLdgType);
        g.b_ldg_offset[1][0] =
            (b_row + ldb * (b_col + 64 + 0 < n_b ? b_col + 64 + 0 : n_b - 1)) * sizeof(BLdgType);
        g.b_ldg_offset[1][1] =
            (b_row + ldb * (b_col + 64 + 1 < n_b ? b_col + 64 + 1 : n_b - 1)) * sizeof(BLdgType);
        g.b_ldg_offset[1][2] =
            (b_row + ldb * (b_col + 64 + 2 < n_b ? b_col + 64 + 2 : n_b - 1)) * sizeof(BLdgType);
        g.b_ldg_offset[1][3] =
            (b_row + ldb * (b_col + 64 + 3 < n_b ? b_col + 64 + 3 : n_b - 1)) * sizeof(BLdgType);

        const int lds_row = tid & 15;
        int lds_col[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            lds_col[i] = (4 * i + lane / 16) ^ lds_row;
        }
        const int a_lds_offset_tmp = (slot / 2) * 256;
        const int b_lds_offset_tmp = (slot & 1) * 256 +
                                     (0x2000 / static_cast<int>(sizeof(ALdsType)));
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int tmp = lds_row * 16 + lds_col[i];
            g.b_lds_offset[i] = (tmp + b_lds_offset_tmp) * sizeof(ALdsType);
            g.a_lds_offset[i] = (tmp + a_lds_offset_tmp) * sizeof(ALdsType);
        }
        return g;
    }
};

struct tn_example_linear_geometry {
    template <typename ALdgType, typename BLdgType, typename ALdsType,
              typename BLdsType>
    __device__ __forceinline__ static tn_example_stage_geometry<ALdgType, BLdgType,
                                                                ALdsType, BLdsType>
    make(int tid, int lane, int slot, int lda, int ldb, int m_a, int n_b) {
        tn_example_stage_geometry<ALdgType, BLdgType, ALdsType, BLdsType> g{};
        const int a_row = (tid / 16) * 4;
        const int a_col = tid & 15;
        const int b_col = a_row;
        const int b_row = a_col;
        g.a_cmp_op1 = a_col;
        g.b_cmp_op1 = b_row;

        g.a_ldg_offset[0][0] =
            (a_col + lda * (a_row + 0 < m_a ? a_row + 0 : m_a - 1)) * sizeof(ALdgType);
        g.a_ldg_offset[0][1] =
            (a_col + lda * (a_row + 1 < m_a ? a_row + 1 : m_a - 1)) * sizeof(ALdgType);
        g.a_ldg_offset[0][2] =
            (a_col + lda * (a_row + 2 < m_a ? a_row + 2 : m_a - 1)) * sizeof(ALdgType);
        g.a_ldg_offset[0][3] =
            (a_col + lda * (a_row + 3 < m_a ? a_row + 3 : m_a - 1)) * sizeof(ALdgType);
        g.a_ldg_offset[1][0] =
            (a_col + lda * (a_row + 64 + 0 < m_a ? a_row + 64 + 0 : m_a - 1)) * sizeof(ALdgType);
        g.a_ldg_offset[1][1] =
            (a_col + lda * (a_row + 64 + 1 < m_a ? a_row + 64 + 1 : m_a - 1)) * sizeof(ALdgType);
        g.a_ldg_offset[1][2] =
            (a_col + lda * (a_row + 64 + 2 < m_a ? a_row + 64 + 2 : m_a - 1)) * sizeof(ALdgType);
        g.a_ldg_offset[1][3] =
            (a_col + lda * (a_row + 64 + 3 < m_a ? a_row + 64 + 3 : m_a - 1)) * sizeof(ALdgType);

        g.b_ldg_offset[0][0] =
            (b_row + ldb * (b_col + 0 < n_b ? b_col + 0 : n_b - 1)) * sizeof(BLdgType);
        g.b_ldg_offset[0][1] =
            (b_row + ldb * (b_col + 1 < n_b ? b_col + 1 : n_b - 1)) * sizeof(BLdgType);
        g.b_ldg_offset[0][2] =
            (b_row + ldb * (b_col + 2 < n_b ? b_col + 2 : n_b - 1)) * sizeof(BLdgType);
        g.b_ldg_offset[0][3] =
            (b_row + ldb * (b_col + 3 < n_b ? b_col + 3 : n_b - 1)) * sizeof(BLdgType);
        g.b_ldg_offset[1][0] =
            (b_row + ldb * (b_col + 64 + 0 < n_b ? b_col + 64 + 0 : n_b - 1)) * sizeof(BLdgType);
        g.b_ldg_offset[1][1] =
            (b_row + ldb * (b_col + 64 + 1 < n_b ? b_col + 64 + 1 : n_b - 1)) * sizeof(BLdgType);
        g.b_ldg_offset[1][2] =
            (b_row + ldb * (b_col + 64 + 2 < n_b ? b_col + 64 + 2 : n_b - 1)) * sizeof(BLdgType);
        g.b_ldg_offset[1][3] =
            (b_row + ldb * (b_col + 64 + 3 < n_b ? b_col + 64 + 3 : n_b - 1)) * sizeof(BLdgType);

        const int lds_row = tid & 15;
        const int lds_col_base = lane / 16;
        const int a_lds_offset_tmp = (slot / 2) * 256;
        const int b_lds_offset_tmp = (slot & 1) * 256 +
                                     (0x2000 / static_cast<int>(sizeof(ALdsType)));
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int tmp = lds_row * 16 + (4 * i + lds_col_base);
            g.b_lds_offset[i] = (tmp + b_lds_offset_tmp) * sizeof(ALdsType);
            g.a_lds_offset[i] = (tmp + a_lds_offset_tmp) * sizeof(ALdsType);
        }
        return g;
    }
};

} // namespace bf16_c500_tk_cute_local::cute_tk::kernel
