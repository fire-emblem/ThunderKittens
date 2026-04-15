#pragma once

#include <cmath>
#include <type_traits>
#include "composition/family_pattern.cuh"
#include "primitives/pipeline/issue_order_atom.cuh"
#include "policies.cuh"
#include "primitives/pipeline/reload_atom.cuh"
#include "primitives/pipeline/schedule_atom.cuh"
#include "primitives/structure/stage_layout_atom.cuh"
#include "tn_example_utils.cuh"
#include "tn_example_geometry.cuh"

namespace bf16_c500_tk_cute_local::cute_tk::kernel {

template <typename T, typename Tc, typename Tscal, bool IsBetaZero,
          typename Pattern = ::bf16_c500_tk_cute_local::cute_tk::family_pattern<
              ::bf16_c500_tk_cute_local::cute_tk::swizzled_tn_semantic_tag,
              ::bf16_c500_tk_cute_local::cute_tk::tile_128x128x128,
              ::bf16_c500_tk_cute_local::cute_tk::swizzled_tn_layout_atom,
              ::bf16_c500_tk_cute_local::cute_tk::tn_example_stage4_schedule,
              ::bf16_c500_tk_cute_local::cute_tk::default_stage_layout_atom>>
__forceinline__ __device__ void hgemm_tn_128x128x128_4m1n8k_256t_device(const void *A,
                                                                        const void *B,
                                                                        void *C,
                                                                        int M,
                                                                        int N,
                                                                        int K,
                                                                        int lda,
                                                                        int ldb,
                                                                        int ldc,
                                                                        Tscal alpha,
                                                                        Tscal beta,
                                                                        int bidx,
                                                                        int bidy) {
    using tile_shape = typename Pattern::tile_shape;
    using geometry_policy = typename Pattern::geometry_provider;
    using schedule_policy = typename Pattern::schedule_policy;
    using stage_layout = typename Pattern::stage_layout_atom;
    constexpr int TileM = tile_shape::tile_m;
    constexpr int TileN = tile_shape::tile_n;
    constexpr int Stage = schedule_policy::stage_count;
    static_assert(tile_shape::tile_m == 128 && tile_shape::tile_n == 128 &&
                      tile_shape::tile_k == 128,
                  "tn_example tile-shape abstraction is in place, but only 128x128x128 is implemented today");
    static_assert(Stage == 4,
                  "tn_example schedule abstraction is in place, but only stage4 is implemented today");
    static_assert(stage_layout::stage_count == Stage,
                  "tn_example stage layout must agree with schedule stage count");

    using ALdgType = __NATIVE_VECTOR__(4, uint);
    using BLdgType = __NATIVE_VECTOR__(4, uint);
    using CStgType = __NATIVE_VECTOR__(sizeof(Tc), uint);  // 4 FP32, or 4 FP16, or 4 BF16
    using ALdsType = ALdgType;
    using BLdsType = BLdgType;
    using FLOAT4 = __NATIVE_VECTOR__(4, float);

    uint8_t *APtr = const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(A));
    uint8_t *BPtr = const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(B));
    CStgType *CPtr = reinterpret_cast<CStgType *>(C);

    lda /= sizeof(ALdgType) / sizeof(T);   // lda needs to be align to 8
    ldb /= sizeof(BLdgType) / sizeof(T);   // ldb needs to be align to 8
    ldc /= sizeof(CStgType) / sizeof(Tc);  // ldc needs to be align to 4

    const int startRow = bidx * TileM;
    const int startCol = bidy * TileN;

    M -= startRow;
    N -= startCol;

    const int M_A = M;
    const int N_B = N;
    const int M_C = M / (sizeof(CStgType) / sizeof(Tc));
    const int N_C = N;

    APtr += startRow * lda * sizeof(ALdgType);
    // BPtr += startCol * ldb;
    BPtr += startCol * ldb * sizeof(BLdgType);
    CPtr += startCol * ldc + startRow / (sizeof(CStgType) / sizeof(Tc));

    const int tid = threadIdx.x;
    // __builtin_mxc_readfirstlane可强制让slot为streg
    const int slot = __builtin_mxc_readfirstlane(tid / 64);
    const int lane = tid & 63;

    const auto geometry = geometry_policy::template make<ALdgType, BLdgType, ALdsType, BLdsType>(
        tid, lane, slot, lda, ldb, M_A, N_B);
    const auto &ALdgOffset = geometry.a_ldg_offset;
    const auto &BLdgOffset = geometry.b_ldg_offset;
    const auto &ALdsOffset = geometry.a_lds_offset;
    const auto &BLdsOffset = geometry.b_lds_offset;
    const int A_col = geometry.a_cmp_op1;
    const int B_row = geometry.b_cmp_op1;

    __shared__ uint8_t WSM[stage_layout::stage_bytes * stage_layout::stage_count];

    FLOAT4 C_f32[4][4] = {0};
    ALdsType a[4][4];
    BLdsType b[4][4];

    uint8_t *WSM_Ldg = WSM + slot * 0x400;

    for (int stage_i = 0; stage_i < Stage; ++stage_i) {
        issue_order_atom::template issue_ab_stage_pred<stage_layout, ALdgType,
                                                       BLdgType, T>(
            WSM_Ldg, APtr, BPtr, ALdgOffset, BLdgOffset, stage_i, A_col,
            K / (sizeof(ALdgType) / sizeof(T)), B_row,
            K / (sizeof(BLdgType) / sizeof(T)), B_row,
            K / (sizeof(BLdgType) / sizeof(T)));
        schedule_atom::template maybe_sync_each_stage_issue<schedule_policy>();
    }
    APtr += 128 * sizeof(T);
    BPtr += 128 * sizeof(T);
    K -= 128;

    schedule_atom::template wait_prologue_stage0<Stage>();

    // WSM_lds如果使用ALdsType*指针，会使得部分lds指令要重新计算 new_offset=offset*16
    uint8_t *WSM_lds = reinterpret_cast<uint8_t *>(&WSM[0]);

    reload_atom::load_pair_stage(a, b, 0, WSM_lds, ALdsOffset, BLdsOffset);

    schedule_atom::template wait_prologue_stage1<Stage>();

    reload_atom::load_pair_stage(a, b, 1, WSM_lds + stage_layout::stage_base_offset(1), ALdsOffset, BLdsOffset);

    // LDG not consider mask of K
    for (; K >= 128; K -= 128) {
        {
            C_f32[0][0] =
                mma_16x16x16b16<T>(b[0][0][0], b[0][0][1], a[0][0][0], a[0][0][1], C_f32[0][0]);
            LDG_B128_BSM_NO_PREDICATOR(WSM_Ldg + stage_layout::a_stage_offset(0, 0), APtr + ALdgOffset[0][0])
            C_f32[0][0] =
                mma_16x16x16b16<T>(b[0][0][2], b[0][0][3], a[0][0][2], a[0][0][3], C_f32[0][0]);
            C_f32[0][0] =
                mma_16x16x16b16<T>(b[0][1][0], b[0][1][1], a[0][1][0], a[0][1][1], C_f32[0][0]);
            C_f32[0][0] =
                mma_16x16x16b16<T>(b[0][1][2], b[0][1][3], a[0][1][2], a[0][1][3], C_f32[0][0]);
            C_f32[0][0] =
                mma_16x16x16b16<T>(b[0][2][0], b[0][2][1], a[0][2][0], a[0][2][1], C_f32[0][0]);
            C_f32[0][0] =
                mma_16x16x16b16<T>(b[0][2][2], b[0][2][3], a[0][2][2], a[0][2][3], C_f32[0][0]);
            C_f32[0][0] =
                mma_16x16x16b16<T>(b[0][3][0], b[0][3][1], a[0][3][0], a[0][3][1], C_f32[0][0]);
            C_f32[0][0] =
                mma_16x16x16b16<T>(b[0][3][2], b[0][3][3], a[0][3][2], a[0][3][3], C_f32[0][0]);

            C_f32[1][0] =
                mma_16x16x16b16<T>(b[0][0][0], b[0][0][1], a[1][0][0], a[1][0][1], C_f32[1][0]);
            LDG_B128_BSM_NO_PREDICATOR(WSM_Ldg + stage_layout::a_stage_offset(0, 1), APtr + ALdgOffset[1][0])
            C_f32[1][0] =
                mma_16x16x16b16<T>(b[0][0][2], b[0][0][3], a[1][0][2], a[1][0][3], C_f32[1][0]);
            C_f32[1][0] =
                mma_16x16x16b16<T>(b[0][1][0], b[0][1][1], a[1][1][0], a[1][1][1], C_f32[1][0]);
            C_f32[1][0] =
                mma_16x16x16b16<T>(b[0][1][2], b[0][1][3], a[1][1][2], a[1][1][3], C_f32[1][0]);
            C_f32[1][0] =
                mma_16x16x16b16<T>(b[0][2][0], b[0][2][1], a[1][2][0], a[1][2][1], C_f32[1][0]);
            C_f32[1][0] =
                mma_16x16x16b16<T>(b[0][2][2], b[0][2][3], a[1][2][2], a[1][2][3], C_f32[1][0]);
            C_f32[1][0] =
                mma_16x16x16b16<T>(b[0][3][0], b[0][3][1], a[1][3][0], a[1][3][1], C_f32[1][0]);
            schedule_atom::template wait_steady_window<Stage>();

            C_f32[1][0] =
                mma_16x16x16b16<T>(b[0][3][2], b[0][3][3], a[1][3][2], a[1][3][3], C_f32[1][0]);
            a[2][0] = reload_atom::load_fragment<ALdsType>(WSM_lds + stage_layout::stage_base_offset(2), ALdsOffset, 0);
            C_f32[0][1] =
                mma_16x16x16b16<T>(b[1][0][0], b[1][0][1], a[0][0][0], a[0][0][1], C_f32[0][1]);
            LDG_B128_BSM_NO_PREDICATOR(WSM_Ldg + stage_layout::b_stage_offset(0, 0), BPtr + BLdgOffset[0][0])
            C_f32[0][1] =
                mma_16x16x16b16<T>(b[1][0][2], b[1][0][3], a[0][0][2], a[0][0][3], C_f32[0][1]);
            a[2][1] = reload_atom::load_fragment<ALdsType>(WSM_lds + stage_layout::stage_base_offset(2), ALdsOffset, 1);
            C_f32[0][1] =
                mma_16x16x16b16<T>(b[1][1][0], b[1][1][1], a[0][1][0], a[0][1][1], C_f32[0][1]);
            C_f32[0][1] =
                mma_16x16x16b16<T>(b[1][1][2], b[1][1][3], a[0][1][2], a[0][1][3], C_f32[0][1]);
            a[2][2] = reload_atom::load_fragment<ALdsType>(WSM_lds + stage_layout::stage_base_offset(2), ALdsOffset, 2);
            C_f32[0][1] =
                mma_16x16x16b16<T>(b[1][2][0], b[1][2][1], a[0][2][0], a[0][2][1], C_f32[0][1]);
            C_f32[0][1] =
                mma_16x16x16b16<T>(b[1][2][2], b[1][2][3], a[0][2][2], a[0][2][3], C_f32[0][1]);
            a[2][3] = reload_atom::load_fragment<ALdsType>(WSM_lds + stage_layout::stage_base_offset(2), ALdsOffset, 3);
            C_f32[0][1] =
                mma_16x16x16b16<T>(b[1][3][0], b[1][3][1], a[0][3][0], a[0][3][1], C_f32[0][1]);
            C_f32[0][1] =
                mma_16x16x16b16<T>(b[1][3][2], b[1][3][3], a[0][3][2], a[0][3][3], C_f32[0][1]);
            b[2][0] = reload_atom::load_fragment<BLdsType>(WSM_lds + stage_layout::stage_base_offset(2), BLdsOffset, 0);

            C_f32[1][1] =
                mma_16x16x16b16<T>(b[1][0][0], b[1][0][1], a[1][0][0], a[1][0][1], C_f32[1][1]);
            LDG_B128_BSM_NO_PREDICATOR(WSM_Ldg + stage_layout::b_stage_offset(0, 1), BPtr + BLdgOffset[1][0])
            C_f32[1][1] =
                mma_16x16x16b16<T>(b[1][0][2], b[1][0][3], a[1][0][2], a[1][0][3], C_f32[1][1]);
            b[2][1] = reload_atom::load_fragment<BLdsType>(WSM_lds + stage_layout::stage_base_offset(2), BLdsOffset, 1);
            C_f32[1][1] =
                mma_16x16x16b16<T>(b[1][1][0], b[1][1][1], a[1][1][0], a[1][1][1], C_f32[1][1]);
            C_f32[1][1] =
                mma_16x16x16b16<T>(b[1][1][2], b[1][1][3], a[1][1][2], a[1][1][3], C_f32[1][1]);
            b[2][2] = reload_atom::load_fragment<BLdsType>(WSM_lds + stage_layout::stage_base_offset(2), BLdsOffset, 2);
            C_f32[1][1] =
                mma_16x16x16b16<T>(b[1][2][0], b[1][2][1], a[1][2][0], a[1][2][1], C_f32[1][1]);
            C_f32[1][1] =
                mma_16x16x16b16<T>(b[1][2][2], b[1][2][3], a[1][2][2], a[1][2][3], C_f32[1][1]);
            b[2][3] = reload_atom::load_fragment<BLdsType>(WSM_lds + stage_layout::stage_base_offset(2), BLdsOffset, 3);
            C_f32[1][1] =
                mma_16x16x16b16<T>(b[1][3][0], b[1][3][1], a[1][3][0], a[1][3][1], C_f32[1][1]);
            C_f32[1][1] =
                mma_16x16x16b16<T>(b[1][3][2], b[1][3][3], a[1][3][2], a[1][3][3], C_f32[1][1]);

            C_f32[2][0] =
                mma_16x16x16b16<T>(b[0][0][0], b[0][0][1], a[2][0][0], a[2][0][1], C_f32[2][0]);
            LDG_B128_BSM_NO_PREDICATOR(WSM_Ldg + stage_layout::a_stage_offset(1, 0), APtr + ALdgOffset[0][1])
            C_f32[2][0] =
                mma_16x16x16b16<T>(b[0][0][2], b[0][0][3], a[2][0][2], a[2][0][3], C_f32[2][0]);
            C_f32[2][0] =
                mma_16x16x16b16<T>(b[0][1][0], b[0][1][1], a[2][1][0], a[2][1][1], C_f32[2][0]);
            C_f32[2][0] =
                mma_16x16x16b16<T>(b[0][1][2], b[0][1][3], a[2][1][2], a[2][1][3], C_f32[2][0]);
            C_f32[2][0] =
                mma_16x16x16b16<T>(b[0][2][0], b[0][2][1], a[2][2][0], a[2][2][1], C_f32[2][0]);
            C_f32[2][0] =
                mma_16x16x16b16<T>(b[0][2][2], b[0][2][3], a[2][2][2], a[2][2][3], C_f32[2][0]);
            C_f32[2][0] =
                mma_16x16x16b16<T>(b[0][3][0], b[0][3][1], a[2][3][0], a[2][3][1], C_f32[2][0]);
            C_f32[2][0] =
                mma_16x16x16b16<T>(b[0][3][2], b[0][3][3], a[2][3][2], a[2][3][3], C_f32[2][0]);

            C_f32[2][1] =
                mma_16x16x16b16<T>(b[1][0][0], b[1][0][1], a[2][0][0], a[2][0][1], C_f32[2][1]);
            LDG_B128_BSM_NO_PREDICATOR(WSM_Ldg + stage_layout::a_stage_offset(1, 1), APtr + ALdgOffset[1][1])
            C_f32[2][1] =
                mma_16x16x16b16<T>(b[1][0][2], b[1][0][3], a[2][0][2], a[2][0][3], C_f32[2][1]);
            C_f32[2][1] =
                mma_16x16x16b16<T>(b[1][1][0], b[1][1][1], a[2][1][0], a[2][1][1], C_f32[2][1]);
            schedule_atom::template wait_steady_window<Stage>();
            C_f32[2][1] =
                mma_16x16x16b16<T>(b[1][1][2], b[1][1][3], a[2][1][2], a[2][1][3], C_f32[2][1]);
            a[3][0] = reload_atom::load_fragment<ALdsType>(WSM_lds + stage_layout::stage_base_offset(3), ALdsOffset, 0);
            C_f32[2][1] =
                mma_16x16x16b16<T>(b[1][2][0], b[1][2][1], a[2][2][0], a[2][2][1], C_f32[2][1]);
            C_f32[2][1] =
                mma_16x16x16b16<T>(b[1][2][2], b[1][2][3], a[2][2][2], a[2][2][3], C_f32[2][1]);
            a[3][1] = reload_atom::load_fragment<ALdsType>(WSM_lds + stage_layout::stage_base_offset(3), ALdsOffset, 1);
            C_f32[2][1] =
                mma_16x16x16b16<T>(b[1][3][0], b[1][3][1], a[2][3][0], a[2][3][1], C_f32[2][1]);
            C_f32[2][1] =
                mma_16x16x16b16<T>(b[1][3][2], b[1][3][3], a[2][3][2], a[2][3][3], C_f32[2][1]);
            a[3][2] = reload_atom::load_fragment<ALdsType>(WSM_lds + stage_layout::stage_base_offset(3), ALdsOffset, 2);

            C_f32[0][2] =
                mma_16x16x16b16<T>(b[2][0][0], b[2][0][1], a[0][0][0], a[0][0][1], C_f32[0][2]);
            LDG_B128_BSM_NO_PREDICATOR(WSM_Ldg + stage_layout::b_stage_offset(1, 0), BPtr + BLdgOffset[0][1])
            C_f32[0][2] =
                mma_16x16x16b16<T>(b[2][0][2], b[2][0][3], a[0][0][2], a[0][0][3], C_f32[0][2]);
            a[3][3] = reload_atom::load_fragment<ALdsType>(WSM_lds + stage_layout::stage_base_offset(3), ALdsOffset, 3);
            C_f32[0][2] =
                mma_16x16x16b16<T>(b[2][1][0], b[2][1][1], a[0][1][0], a[0][1][1], C_f32[0][2]);
            C_f32[0][2] =
                mma_16x16x16b16<T>(b[2][1][2], b[2][1][3], a[0][1][2], a[0][1][3], C_f32[0][2]);
            b[3][0] = reload_atom::load_fragment<BLdsType>(WSM_lds + stage_layout::stage_base_offset(3), BLdsOffset, 0);
            C_f32[0][2] =
                mma_16x16x16b16<T>(b[2][2][0], b[2][2][1], a[0][2][0], a[0][2][1], C_f32[0][2]);
            C_f32[0][2] =
                mma_16x16x16b16<T>(b[2][2][2], b[2][2][3], a[0][2][2], a[0][2][3], C_f32[0][2]);
            b[3][1] = reload_atom::load_fragment<BLdsType>(WSM_lds + stage_layout::stage_base_offset(3), BLdsOffset, 1);
            C_f32[0][2] =
                mma_16x16x16b16<T>(b[2][3][0], b[2][3][1], a[0][3][0], a[0][3][1], C_f32[0][2]);
            C_f32[0][2] =
                mma_16x16x16b16<T>(b[2][3][2], b[2][3][3], a[0][3][2], a[0][3][3], C_f32[0][2]);
            b[3][2] = reload_atom::load_fragment<BLdsType>(WSM_lds + stage_layout::stage_base_offset(3), BLdsOffset, 2);

            C_f32[1][2] =
                mma_16x16x16b16<T>(b[2][0][0], b[2][0][1], a[1][0][0], a[1][0][1], C_f32[1][2]);
            LDG_B128_BSM_NO_PREDICATOR(WSM_Ldg + stage_layout::b_stage_offset(1, 1), BPtr + BLdgOffset[1][1])
            C_f32[1][2] =
                mma_16x16x16b16<T>(b[2][0][2], b[2][0][3], a[1][0][2], a[1][0][3], C_f32[1][2]);
            b[3][3] = reload_atom::load_fragment<BLdsType>(WSM_lds + stage_layout::stage_base_offset(3), BLdsOffset, 3);
            C_f32[1][2] =
                mma_16x16x16b16<T>(b[2][1][0], b[2][1][1], a[1][1][0], a[1][1][1], C_f32[1][2]);
            C_f32[1][2] =
                mma_16x16x16b16<T>(b[2][1][2], b[2][1][3], a[1][1][2], a[1][1][3], C_f32[1][2]);
            C_f32[1][2] =
                mma_16x16x16b16<T>(b[2][2][0], b[2][2][1], a[1][2][0], a[1][2][1], C_f32[1][2]);
            C_f32[1][2] =
                mma_16x16x16b16<T>(b[2][2][2], b[2][2][3], a[1][2][2], a[1][2][3], C_f32[1][2]);
            C_f32[1][2] =
                mma_16x16x16b16<T>(b[2][3][0], b[2][3][1], a[1][3][0], a[1][3][1], C_f32[1][2]);
            C_f32[1][2] =
                mma_16x16x16b16<T>(b[2][3][2], b[2][3][3], a[1][3][2], a[1][3][3], C_f32[1][2]);

            C_f32[2][2] =
                mma_16x16x16b16<T>(b[2][0][0], b[2][0][1], a[2][0][0], a[2][0][1], C_f32[2][2]);
            LDG_B128_BSM_NO_PREDICATOR(WSM_Ldg + stage_layout::a_stage_offset(2, 0), APtr + ALdgOffset[0][2])
            C_f32[2][2] =
                mma_16x16x16b16<T>(b[2][0][2], b[2][0][3], a[2][0][2], a[2][0][3], C_f32[2][2]);
            C_f32[2][2] =
                mma_16x16x16b16<T>(b[2][1][0], b[2][1][1], a[2][1][0], a[2][1][1], C_f32[2][2]);
            C_f32[2][2] =
                mma_16x16x16b16<T>(b[2][1][2], b[2][1][3], a[2][1][2], a[2][1][3], C_f32[2][2]);
            C_f32[2][2] =
                mma_16x16x16b16<T>(b[2][2][0], b[2][2][1], a[2][2][0], a[2][2][1], C_f32[2][2]);
            C_f32[2][2] =
                mma_16x16x16b16<T>(b[2][2][2], b[2][2][3], a[2][2][2], a[2][2][3], C_f32[2][2]);
            C_f32[2][2] =
                mma_16x16x16b16<T>(b[2][3][0], b[2][3][1], a[2][3][0], a[2][3][1], C_f32[2][2]);
            C_f32[2][2] =
                mma_16x16x16b16<T>(b[2][3][2], b[2][3][3], a[2][3][2], a[2][3][3], C_f32[2][2]);

            C_f32[3][0] =
                mma_16x16x16b16<T>(b[0][0][0], b[0][0][1], a[3][0][0], a[3][0][1], C_f32[3][0]);
            LDG_B128_BSM_NO_PREDICATOR(WSM_Ldg + stage_layout::a_stage_offset(2, 1), APtr + ALdgOffset[1][2])
            C_f32[3][0] =
                mma_16x16x16b16<T>(b[0][0][2], b[0][0][3], a[3][0][2], a[3][0][3], C_f32[3][0]);
            C_f32[3][0] =
                mma_16x16x16b16<T>(b[0][1][0], b[0][1][1], a[3][1][0], a[3][1][1], C_f32[3][0]);
            C_f32[3][0] =
                mma_16x16x16b16<T>(b[0][1][2], b[0][1][3], a[3][1][2], a[3][1][3], C_f32[3][0]);
            C_f32[3][0] =
                mma_16x16x16b16<T>(b[0][2][0], b[0][2][1], a[3][2][0], a[3][2][1], C_f32[3][0]);
            C_f32[3][0] =
                mma_16x16x16b16<T>(b[0][2][2], b[0][2][3], a[3][2][2], a[3][2][3], C_f32[3][0]);
            C_f32[3][0] =
                mma_16x16x16b16<T>(b[0][3][0], b[0][3][1], a[3][3][0], a[3][3][1], C_f32[3][0]);
            C_f32[3][0] =
                mma_16x16x16b16<T>(b[0][3][2], b[0][3][3], a[3][3][2], a[3][3][3], C_f32[3][0]);
            schedule_atom::template wait_steady_window<Stage>();

            C_f32[0][3] =
                mma_16x16x16b16<T>(b[3][0][0], b[3][0][1], a[0][0][0], a[0][0][1], C_f32[0][3]);
            LDG_B128_BSM_NO_PREDICATOR(WSM_Ldg + stage_layout::b_stage_offset(2, 0), BPtr + BLdgOffset[0][2])
            C_f32[0][3] =
                mma_16x16x16b16<T>(b[3][0][2], b[3][0][3], a[0][0][2], a[0][0][3], C_f32[0][3]);
            b[0][0] = *reinterpret_cast<ALdsType *>(WSM_lds + 0 + BLdsOffset[0]);
            C_f32[0][3] =
                mma_16x16x16b16<T>(b[3][1][0], b[3][1][1], a[0][1][0], a[0][1][1], C_f32[0][3]);
            C_f32[0][3] =
                mma_16x16x16b16<T>(b[3][1][2], b[3][1][3], a[0][1][2], a[0][1][3], C_f32[0][3]);
            b[0][1] = *reinterpret_cast<ALdsType *>(WSM_lds + 0 + BLdsOffset[1]);
            C_f32[0][3] =
                mma_16x16x16b16<T>(b[3][2][0], b[3][2][1], a[0][2][0], a[0][2][1], C_f32[0][3]);
            C_f32[0][3] =
                mma_16x16x16b16<T>(b[3][2][2], b[3][2][3], a[0][2][2], a[0][2][3], C_f32[0][3]);
            b[0][2] = *reinterpret_cast<ALdsType *>(WSM_lds + 0 + BLdsOffset[2]);
            C_f32[0][3] =
                mma_16x16x16b16<T>(b[3][3][0], b[3][3][1], a[0][3][0], a[0][3][1], C_f32[0][3]);
            C_f32[0][3] =
                mma_16x16x16b16<T>(b[3][3][2], b[3][3][3], a[0][3][2], a[0][3][3], C_f32[0][3]);
            b[0][3] = *reinterpret_cast<ALdsType *>(WSM_lds + 0 + BLdsOffset[3]);

            C_f32[3][1] =
                mma_16x16x16b16<T>(b[1][0][0], b[1][0][1], a[3][0][0], a[3][0][1], C_f32[3][1]);
            LDG_B128_BSM_NO_PREDICATOR(WSM_Ldg + stage_layout::b_stage_offset(2, 1), BPtr + BLdgOffset[1][2])
            C_f32[3][1] =
                mma_16x16x16b16<T>(b[1][0][2], b[1][0][3], a[3][0][2], a[3][0][3], C_f32[3][1]);
            a[0][0] = *reinterpret_cast<ALdsType *>(WSM_lds + 0 + ALdsOffset[0]);
            C_f32[3][1] =
                mma_16x16x16b16<T>(b[1][1][0], b[1][1][1], a[3][1][0], a[3][1][1], C_f32[3][1]);
            C_f32[3][1] =
                mma_16x16x16b16<T>(b[1][1][2], b[1][1][3], a[3][1][2], a[3][1][3], C_f32[3][1]);
            a[0][1] = *reinterpret_cast<ALdsType *>(WSM_lds + 0 + ALdsOffset[1]);
            C_f32[3][1] =
                mma_16x16x16b16<T>(b[1][2][0], b[1][2][1], a[3][2][0], a[3][2][1], C_f32[3][1]);
            C_f32[3][1] =
                mma_16x16x16b16<T>(b[1][2][2], b[1][2][3], a[3][2][2], a[3][2][3], C_f32[3][1]);
            a[0][2] = *reinterpret_cast<ALdsType *>(WSM_lds + 0 + ALdsOffset[2]);
            C_f32[3][1] =
                mma_16x16x16b16<T>(b[1][3][0], b[1][3][1], a[3][3][0], a[3][3][1], C_f32[3][1]);
            C_f32[3][1] =
                mma_16x16x16b16<T>(b[1][3][2], b[1][3][3], a[3][3][2], a[3][3][3], C_f32[3][1]);
            a[0][3] = *reinterpret_cast<ALdsType *>(WSM_lds + 0 + ALdsOffset[3]);

            C_f32[1][3] =
                mma_16x16x16b16<T>(b[3][0][0], b[3][0][1], a[1][0][0], a[1][0][1], C_f32[1][3]);
            LDG_B128_BSM_NO_PREDICATOR(WSM_Ldg + stage_layout::a_stage_offset(3, 0), APtr + ALdgOffset[0][3])
            C_f32[1][3] =
                mma_16x16x16b16<T>(b[3][0][2], b[3][0][3], a[1][0][2], a[1][0][3], C_f32[1][3]);
            C_f32[1][3] =
                mma_16x16x16b16<T>(b[3][1][0], b[3][1][1], a[1][1][0], a[1][1][1], C_f32[1][3]);
            C_f32[1][3] =
                mma_16x16x16b16<T>(b[3][1][2], b[3][1][3], a[1][1][2], a[1][1][3], C_f32[1][3]);
            C_f32[1][3] =
                mma_16x16x16b16<T>(b[3][2][0], b[3][2][1], a[1][2][0], a[1][2][1], C_f32[1][3]);
            C_f32[1][3] =
                mma_16x16x16b16<T>(b[3][2][2], b[3][2][3], a[1][2][2], a[1][2][3], C_f32[1][3]);
            C_f32[1][3] =
                mma_16x16x16b16<T>(b[3][3][0], b[3][3][1], a[1][3][0], a[1][3][1], C_f32[1][3]);
            C_f32[1][3] =
                mma_16x16x16b16<T>(b[3][3][2], b[3][3][3], a[1][3][2], a[1][3][3], C_f32[1][3]);

            C_f32[3][2] =
                mma_16x16x16b16<T>(b[2][0][0], b[2][0][1], a[3][0][0], a[3][0][1], C_f32[3][2]);
            LDG_B128_BSM_NO_PREDICATOR(WSM_Ldg + stage_layout::a_stage_offset(3, 1), APtr + ALdgOffset[1][3])
            C_f32[3][2] =
                mma_16x16x16b16<T>(b[2][0][2], b[2][0][3], a[3][0][2], a[3][0][3], C_f32[3][2]);
            C_f32[3][2] =
                mma_16x16x16b16<T>(b[2][1][0], b[2][1][1], a[3][1][0], a[3][1][1], C_f32[3][2]);
            C_f32[3][2] =
                mma_16x16x16b16<T>(b[2][1][2], b[2][1][3], a[3][1][2], a[3][1][3], C_f32[3][2]);
            C_f32[3][2] =
                mma_16x16x16b16<T>(b[2][2][0], b[2][2][1], a[3][2][0], a[3][2][1], C_f32[3][2]);
            schedule_atom::template wait_steady_window<Stage>();
            C_f32[3][2] =
                mma_16x16x16b16<T>(b[2][2][2], b[2][2][3], a[3][2][2], a[3][2][3], C_f32[3][2]);
            a[1][0] = reload_atom::load_fragment<ALdsType>(WSM_lds + stage_layout::stage_base_offset(1), ALdsOffset, 0);
            C_f32[3][2] =
                mma_16x16x16b16<T>(b[2][3][0], b[2][3][1], a[3][3][0], a[3][3][1], C_f32[3][2]);
            C_f32[3][2] =
                mma_16x16x16b16<T>(b[2][3][2], b[2][3][3], a[3][3][2], a[3][3][3], C_f32[3][2]);
            a[1][1] = reload_atom::load_fragment<ALdsType>(WSM_lds + stage_layout::stage_base_offset(1), ALdsOffset, 1);

            C_f32[2][3] =
                mma_16x16x16b16<T>(b[3][0][0], b[3][0][1], a[2][0][0], a[2][0][1], C_f32[2][3]);
            LDG_B128_BSM_NO_PREDICATOR(WSM_Ldg + stage_layout::b_stage_offset(3, 0), BPtr + BLdgOffset[0][3])
            C_f32[2][3] =
                mma_16x16x16b16<T>(b[3][0][2], b[3][0][3], a[2][0][2], a[2][0][3], C_f32[2][3]);
            a[1][2] = reload_atom::load_fragment<ALdsType>(WSM_lds + stage_layout::stage_base_offset(1), ALdsOffset, 2);
            C_f32[2][3] =
                mma_16x16x16b16<T>(b[3][1][0], b[3][1][1], a[2][1][0], a[2][1][1], C_f32[2][3]);
            C_f32[2][3] =
                mma_16x16x16b16<T>(b[3][1][2], b[3][1][3], a[2][1][2], a[2][1][3], C_f32[2][3]);
            a[1][3] = reload_atom::load_fragment<ALdsType>(WSM_lds + stage_layout::stage_base_offset(1), ALdsOffset, 3);
            C_f32[2][3] =
                mma_16x16x16b16<T>(b[3][2][0], b[3][2][1], a[2][2][0], a[2][2][1], C_f32[2][3]);
            C_f32[2][3] =
                mma_16x16x16b16<T>(b[3][2][2], b[3][2][3], a[2][2][2], a[2][2][3], C_f32[2][3]);
            b[1][0] = reload_atom::load_fragment<BLdsType>(WSM_lds + stage_layout::stage_base_offset(1), BLdsOffset, 0);
            C_f32[2][3] =
                mma_16x16x16b16<T>(b[3][3][0], b[3][3][1], a[2][3][0], a[2][3][1], C_f32[2][3]);
            C_f32[2][3] =
                mma_16x16x16b16<T>(b[3][3][2], b[3][3][3], a[2][3][2], a[2][3][3], C_f32[2][3]);
            b[1][1] = reload_atom::load_fragment<BLdsType>(WSM_lds + stage_layout::stage_base_offset(1), BLdsOffset, 1);

            C_f32[3][3] =
                mma_16x16x16b16<T>(b[3][0][0], b[3][0][1], a[3][0][0], a[3][0][1], C_f32[3][3]);
            LDG_B128_BSM_NO_PREDICATOR(WSM_Ldg + stage_layout::b_stage_offset(3, 1), BPtr + BLdgOffset[1][3])
            C_f32[3][3] =
                mma_16x16x16b16<T>(b[3][0][2], b[3][0][3], a[3][0][2], a[3][0][3], C_f32[3][3]);
            b[1][2] = reload_atom::load_fragment<BLdsType>(WSM_lds + stage_layout::stage_base_offset(1), BLdsOffset, 2);
            C_f32[3][3] =
                mma_16x16x16b16<T>(b[3][1][0], b[3][1][1], a[3][1][0], a[3][1][1], C_f32[3][3]);
            C_f32[3][3] =
                mma_16x16x16b16<T>(b[3][1][2], b[3][1][3], a[3][1][2], a[3][1][3], C_f32[3][3]);
            b[1][3] = reload_atom::load_fragment<BLdsType>(WSM_lds + stage_layout::stage_base_offset(1), BLdsOffset, 3);
            C_f32[3][3] =
                mma_16x16x16b16<T>(b[3][2][0], b[3][2][1], a[3][2][0], a[3][2][1], C_f32[3][3]);
            C_f32[3][3] =
                mma_16x16x16b16<T>(b[3][2][2], b[3][2][3], a[3][2][2], a[3][2][3], C_f32[3][3]);
            C_f32[3][3] =
                mma_16x16x16b16<T>(b[3][3][0], b[3][3][1], a[3][3][0], a[3][3][1], C_f32[3][3]);
            C_f32[3][3] =
                mma_16x16x16b16<T>(b[3][3][2], b[3][3][3], a[3][3][2], a[3][3][3], C_f32[3][3]);
        }
        APtr += 128 * sizeof(T);
        BPtr += 128 * sizeof(T) * ldb / ldb;
        // add 256 will cause using private memory in this kernel, optmize in the future
        // BPtr += 256;
    }

    schedule_atom::template maybe_sync_before_tail_drain<schedule_policy>();

    // LDG consider mask of K
    // 这里其实有lds冗余，但是不展开不太好调整lds指令，要确保a[0]和b[0]用完才能lds(a[0]/b[0])
    if (K > 0) {
#pragma unroll
        for (int stage_i = 0; stage_i < Stage; ++stage_i) {
            const int ldsIdx = (stage_i + 1) % Stage;
            uint8_t *WSM_lds2 = WSM_lds + stage_layout::stage_base_offset(ldsIdx);

            for (int i = 0; i < stage_i; ++i) {
                C_f32[stage_i][i + 0] = mma_16x16x16b16<T>(b[i][0][0], b[i][0][1], a[stage_i][0][0],
                                                           a[stage_i][0][1], C_f32[stage_i][i + 0]);
                C_f32[stage_i][i + 0] = mma_16x16x16b16<T>(b[i][0][2], b[i][0][3], a[stage_i][0][2],
                                                           a[stage_i][0][3], C_f32[stage_i][i + 0]);
                C_f32[stage_i][i + 0] = mma_16x16x16b16<T>(b[i][1][0], b[i][1][1], a[stage_i][1][0],
                                                           a[stage_i][1][1], C_f32[stage_i][i + 0]);
                C_f32[stage_i][i + 0] = mma_16x16x16b16<T>(b[i][1][2], b[i][1][3], a[stage_i][1][2],
                                                           a[stage_i][1][3], C_f32[stage_i][i + 0]);
                C_f32[stage_i][i + 0] = mma_16x16x16b16<T>(b[i][2][0], b[i][2][1], a[stage_i][2][0],
                                                           a[stage_i][2][1], C_f32[stage_i][i + 0]);
                C_f32[stage_i][i + 0] = mma_16x16x16b16<T>(b[i][2][2], b[i][2][3], a[stage_i][2][2],
                                                           a[stage_i][2][3], C_f32[stage_i][i + 0]);
                C_f32[stage_i][i + 0] = mma_16x16x16b16<T>(b[i][3][0], b[i][3][1], a[stage_i][3][0],
                                                           a[stage_i][3][1], C_f32[stage_i][i + 0]);
                C_f32[stage_i][i + 0] = mma_16x16x16b16<T>(b[i][3][2], b[i][3][3], a[stage_i][3][2],
                                                           a[stage_i][3][3], C_f32[stage_i][i + 0]);
            }
            for (int i = 0; i <= stage_i; ++i) {
                C_f32[i][stage_i + 0] =
                    mma_16x16x16b16<T>(b[stage_i][0][0], b[stage_i][0][1], a[i][0][0], a[i][0][1],
                                       C_f32[i][stage_i + 0]);
                C_f32[i][stage_i + 0] =
                    mma_16x16x16b16<T>(b[stage_i][0][2], b[stage_i][0][3], a[i][0][2], a[i][0][3],
                                       C_f32[i][stage_i + 0]);
                C_f32[i][stage_i + 0] =
                    mma_16x16x16b16<T>(b[stage_i][1][0], b[stage_i][1][1], a[i][1][0], a[i][1][1],
                                       C_f32[i][stage_i + 0]);
                C_f32[i][stage_i + 0] =
                    mma_16x16x16b16<T>(b[stage_i][1][2], b[stage_i][1][3], a[i][1][2], a[i][1][3],
                                       C_f32[i][stage_i + 0]);
                C_f32[i][stage_i + 0] =
                    mma_16x16x16b16<T>(b[stage_i][2][0], b[stage_i][2][1], a[i][2][0], a[i][2][1],
                                       C_f32[i][stage_i + 0]);
                C_f32[i][stage_i + 0] =
                    mma_16x16x16b16<T>(b[stage_i][2][2], b[stage_i][2][3], a[i][2][2], a[i][2][3],
                                       C_f32[i][stage_i + 0]);
                C_f32[i][stage_i + 0] =
                    mma_16x16x16b16<T>(b[stage_i][3][0], b[stage_i][3][1], a[i][3][0], a[i][3][1],
                                       C_f32[i][stage_i + 0]);
                C_f32[i][stage_i + 0] =
                    mma_16x16x16b16<T>(b[stage_i][3][2], b[stage_i][3][3], a[i][3][2], a[i][3][3],
                                       C_f32[i][stage_i + 0]);
            }

            schedule_atom::template wait_prologue_stage1<Stage>();

            issue_order_atom::template issue_ab_stage_pred<stage_layout,
                                                           ALdgType, BLdgType,
                                                           T>(
                WSM_Ldg, APtr, BPtr, ALdgOffset, BLdgOffset, stage_i, A_col,
                K / (sizeof(ALdgType) / sizeof(T)), B_row,
                K / (sizeof(BLdgType) / sizeof(T)), B_row,
                K / (sizeof(BLdgType) / sizeof(T)));

            reload_atom::load_pair_stage(a, b, ldsIdx, WSM_lds2, ALdsOffset, BLdsOffset);
        }
    }

    // only do LDA and MMA, no LDGs
    {
#pragma unroll
        for (int stage_i = 0; stage_i < Stage; ++stage_i) {
            const int ldsIdx = (stage_i + 1) % Stage;
            uint8_t *WSM_lds2 = WSM_lds + stage_layout::stage_base_offset(ldsIdx);

            for (int i = 0; i < stage_i; ++i) {
                C_f32[stage_i][i + 0] = mma_16x16x16b16<T>(b[i][0][0], b[i][0][1], a[stage_i][0][0],
                                                           a[stage_i][0][1], C_f32[stage_i][i + 0]);
                C_f32[stage_i][i + 0] = mma_16x16x16b16<T>(b[i][0][2], b[i][0][3], a[stage_i][0][2],
                                                           a[stage_i][0][3], C_f32[stage_i][i + 0]);
                C_f32[stage_i][i + 0] = mma_16x16x16b16<T>(b[i][1][0], b[i][1][1], a[stage_i][1][0],
                                                           a[stage_i][1][1], C_f32[stage_i][i + 0]);
                C_f32[stage_i][i + 0] = mma_16x16x16b16<T>(b[i][1][2], b[i][1][3], a[stage_i][1][2],
                                                           a[stage_i][1][3], C_f32[stage_i][i + 0]);
                C_f32[stage_i][i + 0] = mma_16x16x16b16<T>(b[i][2][0], b[i][2][1], a[stage_i][2][0],
                                                           a[stage_i][2][1], C_f32[stage_i][i + 0]);
                C_f32[stage_i][i + 0] = mma_16x16x16b16<T>(b[i][2][2], b[i][2][3], a[stage_i][2][2],
                                                           a[stage_i][2][3], C_f32[stage_i][i + 0]);
                C_f32[stage_i][i + 0] = mma_16x16x16b16<T>(b[i][3][0], b[i][3][1], a[stage_i][3][0],
                                                           a[stage_i][3][1], C_f32[stage_i][i + 0]);
                C_f32[stage_i][i + 0] = mma_16x16x16b16<T>(b[i][3][2], b[i][3][3], a[stage_i][3][2],
                                                           a[stage_i][3][3], C_f32[stage_i][i + 0]);
            }
            for (int i = 0; i <= stage_i; ++i) {
                C_f32[i][stage_i + 0] =
                    mma_16x16x16b16<T>(b[stage_i][0][0], b[stage_i][0][1], a[i][0][0], a[i][0][1],
                                       C_f32[i][stage_i + 0]);
                C_f32[i][stage_i + 0] =
                    mma_16x16x16b16<T>(b[stage_i][0][2], b[stage_i][0][3], a[i][0][2], a[i][0][3],
                                       C_f32[i][stage_i + 0]);
                C_f32[i][stage_i + 0] =
                    mma_16x16x16b16<T>(b[stage_i][1][0], b[stage_i][1][1], a[i][1][0], a[i][1][1],
                                       C_f32[i][stage_i + 0]);
                C_f32[i][stage_i + 0] =
                    mma_16x16x16b16<T>(b[stage_i][1][2], b[stage_i][1][3], a[i][1][2], a[i][1][3],
                                       C_f32[i][stage_i + 0]);
                C_f32[i][stage_i + 0] =
                    mma_16x16x16b16<T>(b[stage_i][2][0], b[stage_i][2][1], a[i][2][0], a[i][2][1],
                                       C_f32[i][stage_i + 0]);
                C_f32[i][stage_i + 0] =
                    mma_16x16x16b16<T>(b[stage_i][2][2], b[stage_i][2][3], a[i][2][2], a[i][2][3],
                                       C_f32[i][stage_i + 0]);
                C_f32[i][stage_i + 0] =
                    mma_16x16x16b16<T>(b[stage_i][3][0], b[stage_i][3][1], a[i][3][0], a[i][3][1],
                                       C_f32[i][stage_i + 0]);
                C_f32[i][stage_i + 0] =
                    mma_16x16x16b16<T>(b[stage_i][3][2], b[stage_i][3][3], a[i][3][2], a[i][3][3],
                                       C_f32[i][stage_i + 0]);
            }

            schedule_atom::template wait_tail_stage<Stage>(stage_i);

            if (stage_i < Stage - 1) {
                reload_atom::load_pair_stage(a, b, ldsIdx, WSM_lds2, ALdsOffset, BLdsOffset);
            }
        }
    }

    /*
     * layout of C:
     *       col0       col1       col2       col3       col4
     * row0 C[0][0]+0  C[0][1]+0  C[0][2]+0  C[0][3]+0  C[0][0]+1
     * row1 C[1][0]+0
     * row2 C[2][0]+0
     * row3 C[3][0]+0
     */

    const int C_row = (tid & 15) + (slot / 2) * 16;
    const int C_col = (lane / 16 * 16) + (slot & 1) * 64;
    const int CStgOffset = C_row + C_col * ldc;
    const bool C_row_flag = C_row < M_C;

    for (int i = 0; i < 4; ++i) {
        for (int j2 = 0; j2 < 4; ++j2) {
            const bool C_col_flag = i * 4 + j2 < N_C;
            const bool stgFlag = C_row_flag && C_col_flag;

            if (stgFlag) {
                float C_f32_res[4];
                for (int j = 0; j < 4; ++j) {
                    C_f32_res[j] = C_f32[j][j2][i] * static_cast<float>(alpha);
                }

                if constexpr (!IsBetaZero) {
                    CStgType C_tmp5 = CPtr[CStgOffset];
                    Tc *C_tmp6 = reinterpret_cast<Tc *>(&C_tmp5);
                    for (int j = 0; j < 4; ++j) {
                        C_f32_res[j] += static_cast<float>(C_tmp6[j]) * static_cast<float>(beta);
                    }
                }

                if constexpr (std::is_same<Tc, __half>::value ||
                              std::is_same<Tc, maca_bfloat16>::value) {
                    /*
                     *  inplace cvt will cost private memory, following is not good
                     *  Tc *C_half_tmp = reinterpret_cast<Tc*>(&C_tmp[0]);
                     */
                    Tc C_half_tmp[4] = {0};
                    for (int j = 0; j < 4; ++j) {
                        C_half_tmp[j] = static_cast<Tc>(C_f32_res[j]);
                    }

                    uint *C_half2_tmp = reinterpret_cast<uint *>(&C_f32_res[0]);
                    uint16_t *C_halfToB16_tmp = reinterpret_cast<uint16_t *>(&C_half_tmp[0]);
                    C_half2_tmp[0] = static_cast<uint>(C_halfToB16_tmp[0]) +
                                     (static_cast<uint>(C_halfToB16_tmp[1]) << 16);
                    C_half2_tmp[1] = static_cast<uint>(C_halfToB16_tmp[2]) +
                                     (static_cast<uint>(C_halfToB16_tmp[3]) << 16);
                }

                CPtr[CStgOffset] = *reinterpret_cast<CStgType *>(&C_f32_res[0]);
            }
            CPtr += ldc;
        }
    }
}

struct swizzled_tn_stage4_body {
    template <typename T, typename Tc, typename Tscal, bool IsBetaZero,
              bool HasOneDimBias, typename Pattern>
    __device__ __forceinline__ static void run(
        const void *A, const void *B, void *C, int M, int N, int K, int lda,
        int ldb, int ldc, Tscal alpha, Tscal beta, const void *bias, int bidx,
        int bidy) {
        (void)bias;
        static_assert(!HasOneDimBias,
                      "swizzled_tn body does not support one-dim bias");
        hgemm_tn_128x128x128_4m1n8k_256t_device<T, Tc, Tscal, IsBetaZero,
                                               Pattern>(
            A, B, C, M, N, K, lda, ldb, ldc, alpha, beta, bidx, bidy);
    }
};

} // namespace bf16_c500_tk_cute_local::cute_tk::kernel
