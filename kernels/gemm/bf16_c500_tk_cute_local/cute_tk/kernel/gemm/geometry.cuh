#pragma once

// Kernel-specific geometry providers for GEMM
// These are concrete implementations that use the abstract primitives

#include "../primitives/layout/stage_geometry.cuh"
#include "../primitives/arch/mxc_builtins.cuh"

namespace bf16_c500_tk_cute_local::kernel::gemm {

// Column-major C layout geometry provider
// Used for standard GEMM with C stored in column-major order
struct column_major_c_geometry_t {
    template <typename ALdgType, typename BLdgType, typename ALdsType,
              typename BLdsType>
    __device__ __forceinline__ static auto make(int tid, int lane, int slot,
                                                 int lda, int n) {
        primitives::stage_geometry_t<ALdgType, BLdgType, ALdsType, BLdsType> g{};

        // A matrix: each thread loads 8 elements along K dimension
        // Offsets are strided by 16*lda bytes per row
        g.a_ldg_offset[0][0] = (tid + 16 * lda * 0) * sizeof(ALdgType);
        g.a_ldg_offset[0][1] = (tid + 16 * lda * 1) * sizeof(ALdgType);
        g.a_ldg_offset[0][2] = (tid + 16 * lda * 2) * sizeof(ALdgType);
        g.a_ldg_offset[0][3] = (tid + 16 * lda * 3) * sizeof(ALdgType);
        g.a_ldg_offset[1][0] = (tid + 16 * lda * 4) * sizeof(ALdgType);
        g.a_ldg_offset[1][1] = (tid + 16 * lda * 5) * sizeof(ALdgType);
        g.a_ldg_offset[1][2] = (tid + 16 * lda * 6) * sizeof(ALdgType);
        g.a_ldg_offset[1][3] = (tid + 16 * lda * 7) * sizeof(ALdgType);

        // B matrix: column-major access pattern
        // Each warp quadrant handles different column groups
        const int b_row_offset = lane + slot * 64 * (n / 16);
        g.b_ldg_offset[0][0] = (b_row_offset + 64 * 0) * sizeof(BLdgType);
        g.b_ldg_offset[0][1] = (b_row_offset + 64 * 1) * sizeof(BLdgType);
        g.b_ldg_offset[0][2] = (b_row_offset + 64 * 2) * sizeof(BLdgType);
        g.b_ldg_offset[0][3] = (b_row_offset + 64 * 3) * sizeof(BLdgType);
        g.b_ldg_offset[1][0] = (b_row_offset + 64 * 4) * sizeof(BLdgType);
        g.b_ldg_offset[1][1] = (b_row_offset + 64 * 5) * sizeof(BLdgType);
        g.b_ldg_offset[1][2] = (b_row_offset + 64 * 6) * sizeof(BLdgType);
        g.b_ldg_offset[1][3] = (b_row_offset + 64 * 7) * sizeof(BLdgType);

        // Shared memory offsets with bank conflict avoidance
#pragma unroll
        for (int i = 0; i < 4; i++) {
            g.a_lds_offset[i] = (lane + slot / 2 * 0x1000 / sizeof(ALdsType) +
                                 i * 0x400 / sizeof(ALdsType)) *
                                sizeof(ALdsType);
            g.b_lds_offset[i] = (lane + 0x2000 / sizeof(BLdsType) +
                                 (slot & 1) * 0x1000 / sizeof(BLdsType) +
                                 i * 0x400 / sizeof(BLdsType)) *
                                sizeof(BLdgType);
        }
        return g;
    }
};

// Continuous C layout geometry provider
// Used for GEMM where C is stored contiguously
using continuous_c_geometry_t = column_major_c_geometry_t;

// Swizzled TN geometry provider
// Used for transposed B matrix with swizzling
struct swizzled_tn_geometry_t {
    template <typename ALdgType, typename BLdgType, typename ALdsType,
              typename BLdsType>
    __device__ __forceinline__ static auto make(int tid, int lane, int slot,
                                                 int lda, int ldb, int m_a,
                                                 int n_b) {
        primitives::stage_geometry_t<ALdgType, BLdgType, ALdsType, BLdsType> g{};

        const int a_row = (tid / 16) * 4;
        const int a_col = (tid & 15) ^ (tid / 16);
        const int b_col = a_row;
        const int b_row = a_col;

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

} // namespace bf16_c500_tk_cute_local::kernel::gemm