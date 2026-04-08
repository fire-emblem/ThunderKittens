#pragma once

#include <maca.h>

namespace bf16_c500_tk_local::kernel {

template <typename ALdgType, typename BLdgType, typename ALdsType, typename BLdsType, typename T>
struct layoutc_stage_geometry {
    int a_ldg_offset[2][4];
    int b_ldg_offset[2][4];
    int a_lds_offset[4];
    int b_lds_offset[4];
};

template <typename ALdgType, typename BLdgType, typename ALdsType, typename BLdsType, typename T>
__device__ __forceinline__ layoutc_stage_geometry<ALdgType, BLdgType, ALdsType, BLdsType, T>
make_layoutc_stage_geometry(int tid, int lane, int slot, int lda, int n) {
    layoutc_stage_geometry<ALdgType, BLdgType, ALdsType, BLdsType, T> g{};

    g.a_ldg_offset[0][0] = (tid + 16 * lda * 0) * sizeof(ALdgType);
    g.a_ldg_offset[0][1] = (tid + 16 * lda * 1) * sizeof(ALdgType);
    g.a_ldg_offset[0][2] = (tid + 16 * lda * 2) * sizeof(ALdgType);
    g.a_ldg_offset[0][3] = (tid + 16 * lda * 3) * sizeof(ALdgType);
    g.a_ldg_offset[1][0] = (tid + 16 * lda * 4) * sizeof(ALdgType);
    g.a_ldg_offset[1][1] = (tid + 16 * lda * 5) * sizeof(ALdgType);
    g.a_ldg_offset[1][2] = (tid + 16 * lda * 6) * sizeof(ALdgType);
    g.a_ldg_offset[1][3] = (tid + 16 * lda * 7) * sizeof(ALdgType);

    const int b_row_offset = lane + slot * 64 * (n / 16);
    g.b_ldg_offset[0][0] = (b_row_offset + 64 * 0) * sizeof(BLdgType);
    g.b_ldg_offset[0][1] = (b_row_offset + 64 * 1) * sizeof(BLdgType);
    g.b_ldg_offset[0][2] = (b_row_offset + 64 * 2) * sizeof(BLdgType);
    g.b_ldg_offset[0][3] = (b_row_offset + 64 * 3) * sizeof(BLdgType);
    g.b_ldg_offset[1][0] = (b_row_offset + 64 * 4) * sizeof(BLdgType);
    g.b_ldg_offset[1][1] = (b_row_offset + 64 * 5) * sizeof(BLdgType);
    g.b_ldg_offset[1][2] = (b_row_offset + 64 * 6) * sizeof(BLdgType);
    g.b_ldg_offset[1][3] = (b_row_offset + 64 * 7) * sizeof(BLdgType);

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

} // namespace bf16_c500_tk_local::kernel
