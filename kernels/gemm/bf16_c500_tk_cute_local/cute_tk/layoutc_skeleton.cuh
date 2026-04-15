// The development of this file is kindly supported by MetaX.

#pragma once
#include <maca.h>
#include <maca_bfloat16.h>
#include <maca_fp16.h>
#include <mc_runtime.h>

#include "../kernel/layoutc_prologue.cuh"
#include "../kernel/layoutc_support.cuh"
#include "../kernel/layoutc_tail.cuh"
#include "copy_atom.cuh"
#include "epilogue_atom.cuh"
#include "layout_atom.cuh"
#include "mma_atom.cuh"

namespace bf16_c500_tk_cute_local::cute_tk::kernel {

using ::bf16_c500_tk_local::kernel::drain_layoutc_tail;
using ::bf16_c500_tk_local::kernel::load_layoutc_fragment_from_shared;
using ::bf16_c500_tk_local::kernel::mma_16x16x16b16;
using ::bf16_c500_tk_local::kernel::run_layoutc_tail_iteration;

template <typename T, typename Tc, typename Tscal, bool IsBetaZero,
          bool HasOneDimBias, bool OutputContinuousC = false,
          typename LayoutAtom = layoutc_layout_atom>
__forceinline__ __device__ void
layoutc_stage4_device(
    const void *A, const void *B, void *C, int M, int N, int K, int lda,
    int ldb, int ldc, Tscal alpha, Tscal beta, const void *bias, int bidx,
    int bidy) {
    constexpr int TileM = 128;
    constexpr int TileN = 128;
    constexpr int Stage = 4;
    const int src_N = N;
    using ALdgType = __NATIVE_VECTOR__(4, uint);
    using BLdgType = __NATIVE_VECTOR__(4, uint);
    using CStgType = __NATIVE_VECTOR__(sizeof(Tc), uint);
    using ALdsType = ALdgType;
    using BLdsType = BLdgType;
    using FLOAT4 = __NATIVE_VECTOR__(4, float);

    uint8_t *APtr = const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(A));
    uint8_t *BPtr = const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(B));
    // CStgType *CPtr = reinterpret_cast<CStgType *>(C);

    lda /= sizeof(ALdgType) / sizeof(T);
    ldb /= sizeof(BLdgType) / sizeof(T);
    // ldc /= sizeof(CStgType) / sizeof(Tc);

    const int startRow = bidx * TileM;
    const int startCol = bidy * TileN;

    APtr += startRow * lda * sizeof(ALdgType);
    BPtr += startCol / 128 * 64 * (128 / 16) * sizeof(BLdgType);
    // CPtr += startCol * ldc + startRow / (sizeof(CStgType) / sizeof(Tc));

    const int tid = threadIdx.x;
    const int slot = __builtin_mxc_readfirstlane(tid / 64);
    const int lane = tid & 63;

    const auto geometry =
        LayoutAtom::template make_stage_geometry<ALdgType, BLdgType,
                                                 ALdsType, BLdsType, T>(
            tid, lane, slot, lda, N);
    const auto &ALdgOffset = geometry.a_ldg_offset;
    const auto &BLdgOffset = geometry.b_ldg_offset;
    const auto &ALdsOffset = geometry.a_lds_offset;
    const auto &BLdsOffset = geometry.b_lds_offset;

    __shared__ uint8_t WSM[0x10000]; // 64KB

    FLOAT4 C_f32[4][4] = {}; // = {} means all zeros
    ALdsType a[4][4];
    BLdsType b[4][4];

    uint8_t *WSM_Ldg = WSM + slot * 0x400;

    copy_atom::template issue_prologue<ALdgType, BLdgType, T>(
        WSM_Ldg, APtr, BPtr, geometry.a_ldg_offset, geometry.b_ldg_offset, K, N,
        startCol);

    APtr += (128 / 8) * 16 * sizeof(ALdgType);
    BPtr += 16 * N * sizeof(BLdgType);
    K -= 128;

    uint8_t *WSM_lds = reinterpret_cast<uint8_t *>(&WSM[0]);
    copy_atom::prime_fragments(a, b, WSM_lds, geometry.a_lds_offset,
                               geometry.b_lds_offset);

    for (; K >= 128; K -= 128) {
        {
            C_f32[0][0] = mma_16x16x16b16<T, true>(
                b[0][0][0], b[0][0][1], a[0][0][0], a[0][0][1], C_f32[0][0]);
            LDG_B128_BSM_NO_PREDICATOR(WSM_Ldg + 0x4000 * 0 + 0x0000,
                                       APtr + ALdgOffset[0][0])
            C_f32[0][0] = mma_16x16x16b16<T, true>(
                b[0][0][2], b[0][0][3], a[0][0][2], a[0][0][3], C_f32[0][0]);
            C_f32[0][0] = mma_16x16x16b16<T, true>(
                b[0][1][0], b[0][1][1], a[0][1][0], a[0][1][1], C_f32[0][0]);
            C_f32[0][0] = mma_16x16x16b16<T, true>(
                b[0][1][2], b[0][1][3], a[0][1][2], a[0][1][3], C_f32[0][0]);
            C_f32[0][0] = mma_16x16x16b16<T, true>(
                b[0][2][0], b[0][2][1], a[0][2][0], a[0][2][1], C_f32[0][0]);
            C_f32[0][0] = mma_16x16x16b16<T, true>(
                b[0][2][2], b[0][2][3], a[0][2][2], a[0][2][3], C_f32[0][0]);
            C_f32[0][0] = mma_16x16x16b16<T, true>(
                b[0][3][0], b[0][3][1], a[0][3][0], a[0][3][1], C_f32[0][0]);
            C_f32[0][0] = mma_16x16x16b16<T, true>(
                b[0][3][2], b[0][3][3], a[0][3][2], a[0][3][3], C_f32[0][0]);

            C_f32[1][0] = mma_16x16x16b16<T, true>(
                b[0][0][0], b[0][0][1], a[1][0][0], a[1][0][1], C_f32[1][0]);
            LDG_B128_BSM_NO_PREDICATOR(WSM_Ldg + 0x4000 * 0 + 0x1000,
                                       APtr + ALdgOffset[1][0])
            C_f32[1][0] = mma_16x16x16b16<T, true>(
                b[0][0][2], b[0][0][3], a[1][0][2], a[1][0][3], C_f32[1][0]);
            C_f32[1][0] = mma_16x16x16b16<T, true>(
                b[0][1][0], b[0][1][1], a[1][1][0], a[1][1][1], C_f32[1][0]);
            C_f32[1][0] = mma_16x16x16b16<T, true>(
                b[0][1][2], b[0][1][3], a[1][1][2], a[1][1][3], C_f32[1][0]);
            C_f32[1][0] = mma_16x16x16b16<T, true>(
                b[0][2][0], b[0][2][1], a[1][2][0], a[1][2][1], C_f32[1][0]);
            C_f32[1][0] = mma_16x16x16b16<T, true>(
                b[0][2][2], b[0][2][3], a[1][2][2], a[1][2][3], C_f32[1][0]);
            C_f32[1][0] = mma_16x16x16b16<T, true>(
                b[0][3][0], b[0][3][1], a[1][3][0], a[1][3][1], C_f32[1][0]);
            arrive_gvmcnt(4 * (Stage - 3) + 2);
            __builtin_mxc_barrier_inst();
            C_f32[1][0] = mma_16x16x16b16<T, true>(
                b[0][3][2], b[0][3][3], a[1][3][2], a[1][3][3], C_f32[1][0]);

            a[2][0] = load_layoutc_fragment_from_shared<ALdsType>(
                WSM_lds + 0x8000, ALdsOffset, 0);
            C_f32[0][1] = mma_16x16x16b16<T, true>(
                b[1][0][0], b[1][0][1], a[0][0][0], a[0][0][1], C_f32[0][1]);
            __builtin_mxc_ldg_b128_bsm_predicator(
                WSM_Ldg + 0x4000 * 0 + 0x2000, BPtr + BLdgOffset[0][0], 0, true,
                true, false, true, startCol + 0, N, MACA_ICMP_SLT);
            C_f32[0][1] = mma_16x16x16b16<T, true>(
                b[1][0][2], b[1][0][3], a[0][0][2], a[0][0][3], C_f32[0][1]);
            a[2][1] = load_layoutc_fragment_from_shared<ALdsType>(
                WSM_lds + 0x8000, ALdsOffset, 1);
            C_f32[0][1] = mma_16x16x16b16<T, true>(
                b[1][1][0], b[1][1][1], a[0][1][0], a[0][1][1], C_f32[0][1]);
            C_f32[0][1] = mma_16x16x16b16<T, true>(
                b[1][1][2], b[1][1][3], a[0][1][2], a[0][1][3], C_f32[0][1]);
            a[2][2] = load_layoutc_fragment_from_shared<ALdsType>(
                WSM_lds + 0x8000, ALdsOffset, 2);
            C_f32[0][1] = mma_16x16x16b16<T, true>(
                b[1][2][0], b[1][2][1], a[0][2][0], a[0][2][1], C_f32[0][1]);
            C_f32[0][1] = mma_16x16x16b16<T, true>(
                b[1][2][2], b[1][2][3], a[0][2][2], a[0][2][3], C_f32[0][1]);
            a[2][3] = load_layoutc_fragment_from_shared<ALdsType>(
                WSM_lds + 0x8000, ALdsOffset, 3);
            C_f32[0][1] = mma_16x16x16b16<T, true>(
                b[1][3][0], b[1][3][1], a[0][3][0], a[0][3][1], C_f32[0][1]);
            C_f32[0][1] = mma_16x16x16b16<T, true>(
                b[1][3][2], b[1][3][3], a[0][3][2], a[0][3][3], C_f32[0][1]);
            b[2][0] = load_layoutc_fragment_from_shared<BLdsType>(
                WSM_lds + 0x8000, BLdsOffset, 0);

            C_f32[1][1] = mma_16x16x16b16<T, true>(
                b[1][0][0], b[1][0][1], a[1][0][0], a[1][0][1], C_f32[1][1]);
            __builtin_mxc_ldg_b128_bsm_predicator(
                WSM_Ldg + 0x4000 * 0 + 0x3000, BPtr + BLdgOffset[1][0], 0, true,
                true, false, true, startCol + 64, N, MACA_ICMP_SLT);
            C_f32[1][1] = mma_16x16x16b16<T, true>(
                b[1][0][2], b[1][0][3], a[1][0][2], a[1][0][3], C_f32[1][1]);
            b[2][1] = load_layoutc_fragment_from_shared<BLdsType>(
                WSM_lds + 0x8000, BLdsOffset, 1);
            C_f32[1][1] = mma_16x16x16b16<T, true>(
                b[1][1][0], b[1][1][1], a[1][1][0], a[1][1][1], C_f32[1][1]);
            C_f32[1][1] = mma_16x16x16b16<T, true>(
                b[1][1][2], b[1][1][3], a[1][1][2], a[1][1][3], C_f32[1][1]);
            b[2][2] = load_layoutc_fragment_from_shared<BLdsType>(
                WSM_lds + 0x8000, BLdsOffset, 2);
            C_f32[1][1] = mma_16x16x16b16<T, true>(
                b[1][2][0], b[1][2][1], a[1][2][0], a[1][2][1], C_f32[1][1]);
            C_f32[1][1] = mma_16x16x16b16<T, true>(
                b[1][2][2], b[1][2][3], a[1][2][2], a[1][2][3], C_f32[1][1]);
            b[2][3] = load_layoutc_fragment_from_shared<BLdsType>(
                WSM_lds + 0x8000, BLdsOffset, 3);
            C_f32[1][1] = mma_16x16x16b16<T, true>(
                b[1][3][0], b[1][3][1], a[1][3][0], a[1][3][1], C_f32[1][1]);
            C_f32[1][1] = mma_16x16x16b16<T, true>(
                b[1][3][2], b[1][3][3], a[1][3][2], a[1][3][3], C_f32[1][1]);

            C_f32[2][0] = mma_16x16x16b16<T, true>(
                b[0][0][0], b[0][0][1], a[2][0][0], a[2][0][1], C_f32[2][0]);
            LDG_B128_BSM_NO_PREDICATOR(WSM_Ldg + 0x4000 * 1 + 0x0000,
                                       APtr + ALdgOffset[0][1])
            C_f32[2][0] = mma_16x16x16b16<T, true>(
                b[0][0][2], b[0][0][3], a[2][0][2], a[2][0][3], C_f32[2][0]);
            C_f32[2][0] = mma_16x16x16b16<T, true>(
                b[0][1][0], b[0][1][1], a[2][1][0], a[2][1][1], C_f32[2][0]);
            C_f32[2][0] = mma_16x16x16b16<T, true>(
                b[0][1][2], b[0][1][3], a[2][1][2], a[2][1][3], C_f32[2][0]);
            C_f32[2][0] = mma_16x16x16b16<T, true>(
                b[0][2][0], b[0][2][1], a[2][2][0], a[2][2][1], C_f32[2][0]);
            C_f32[2][0] = mma_16x16x16b16<T, true>(
                b[0][2][2], b[0][2][3], a[2][2][2], a[2][2][3], C_f32[2][0]);
            C_f32[2][0] = mma_16x16x16b16<T, true>(
                b[0][3][0], b[0][3][1], a[2][3][0], a[2][3][1], C_f32[2][0]);
            C_f32[2][0] = mma_16x16x16b16<T, true>(
                b[0][3][2], b[0][3][3], a[2][3][2], a[2][3][3], C_f32[2][0]);

            C_f32[2][1] = mma_16x16x16b16<T, true>(
                b[1][0][0], b[1][0][1], a[2][0][0], a[2][0][1], C_f32[2][1]);
            LDG_B128_BSM_NO_PREDICATOR(WSM_Ldg + 0x4000 * 1 + 0x1000,
                                       APtr + ALdgOffset[1][1])
            C_f32[2][1] = mma_16x16x16b16<T, true>(
                b[1][0][2], b[1][0][3], a[2][0][2], a[2][0][3], C_f32[2][1]);
            C_f32[2][1] = mma_16x16x16b16<T, true>(
                b[1][1][0], b[1][1][1], a[2][1][0], a[2][1][1], C_f32[2][1]);
            arrive_gvmcnt(4 * (Stage - 4) + 6);
            __builtin_mxc_barrier_inst();
            C_f32[2][1] = mma_16x16x16b16<T, true>(
                b[1][1][2], b[1][1][3], a[2][1][2], a[2][1][3], C_f32[2][1]);
            a[3][0] = load_layoutc_fragment_from_shared<ALdsType>(
                WSM_lds + 0xC000, ALdsOffset, 0);
            C_f32[2][1] = mma_16x16x16b16<T, true>(
                b[1][2][0], b[1][2][1], a[2][2][0], a[2][2][1], C_f32[2][1]);
            C_f32[2][1] = mma_16x16x16b16<T, true>(
                b[1][2][2], b[1][2][3], a[2][2][2], a[2][2][3], C_f32[2][1]);
            a[3][1] = load_layoutc_fragment_from_shared<ALdsType>(
                WSM_lds + 0xC000, ALdsOffset, 1);
            C_f32[2][1] = mma_16x16x16b16<T, true>(
                b[1][3][0], b[1][3][1], a[2][3][0], a[2][3][1], C_f32[2][1]);
            C_f32[2][1] = mma_16x16x16b16<T, true>(
                b[1][3][2], b[1][3][3], a[2][3][2], a[2][3][3], C_f32[2][1]);
            a[3][2] = load_layoutc_fragment_from_shared<ALdsType>(
                WSM_lds + 0xC000, ALdsOffset, 2);

            C_f32[0][2] = mma_16x16x16b16<T, true>(
                b[2][0][0], b[2][0][1], a[0][0][0], a[0][0][1], C_f32[0][2]);
            __builtin_mxc_ldg_b128_bsm_predicator(
                WSM_Ldg + 0x4000 * 1 + 0x2000, BPtr + BLdgOffset[0][1], 0, true,
                true, false, true, startCol + 16, N, MACA_ICMP_SLT);
            C_f32[0][2] = mma_16x16x16b16<T, true>(
                b[2][0][2], b[2][0][3], a[0][0][2], a[0][0][3], C_f32[0][2]);
            a[3][3] = load_layoutc_fragment_from_shared<ALdsType>(
                WSM_lds + 0xC000, ALdsOffset, 3);
            C_f32[0][2] = mma_16x16x16b16<T, true>(
                b[2][1][0], b[2][1][1], a[0][1][0], a[0][1][1], C_f32[0][2]);
            C_f32[0][2] = mma_16x16x16b16<T, true>(
                b[2][1][2], b[2][1][3], a[0][1][2], a[0][1][3], C_f32[0][2]);
            b[3][0] = load_layoutc_fragment_from_shared<BLdsType>(
                WSM_lds + 0xC000, BLdsOffset, 0);
            C_f32[0][2] = mma_16x16x16b16<T, true>(
                b[2][2][0], b[2][2][1], a[0][2][0], a[0][2][1], C_f32[0][2]);
            C_f32[0][2] = mma_16x16x16b16<T, true>(
                b[2][2][2], b[2][2][3], a[0][2][2], a[0][2][3], C_f32[0][2]);
            b[3][1] = load_layoutc_fragment_from_shared<BLdsType>(
                WSM_lds + 0xC000, BLdsOffset, 1);
            C_f32[0][2] = mma_16x16x16b16<T, true>(
                b[2][3][0], b[2][3][1], a[0][3][0], a[0][3][1], C_f32[0][2]);
            C_f32[0][2] = mma_16x16x16b16<T, true>(
                b[2][3][2], b[2][3][3], a[0][3][2], a[0][3][3], C_f32[0][2]);
            b[3][2] = load_layoutc_fragment_from_shared<BLdsType>(
                WSM_lds + 0xC000, BLdsOffset, 2);

            C_f32[1][2] = mma_16x16x16b16<T, true>(
                b[2][0][0], b[2][0][1], a[1][0][0], a[1][0][1], C_f32[1][2]);
            __builtin_mxc_ldg_b128_bsm_predicator(
                WSM_Ldg + 0x4000 * 1 + 0x3000, BPtr + BLdgOffset[1][1], 0, true,
                true, false, true, startCol + 80, N, MACA_ICMP_SLT);
            C_f32[1][2] = mma_16x16x16b16<T, true>(
                b[2][0][2], b[2][0][3], a[1][0][2], a[1][0][3], C_f32[1][2]);
            b[3][3] = load_layoutc_fragment_from_shared<BLdsType>(
                WSM_lds + 0xC000, BLdsOffset, 3);
            C_f32[1][2] = mma_16x16x16b16<T, true>(
                b[2][1][0], b[2][1][1], a[1][1][0], a[1][1][1], C_f32[1][2]);
            C_f32[1][2] = mma_16x16x16b16<T, true>(
                b[2][1][2], b[2][1][3], a[1][1][2], a[1][1][3], C_f32[1][2]);
            C_f32[1][2] = mma_16x16x16b16<T, true>(
                b[2][2][0], b[2][2][1], a[1][2][0], a[1][2][1], C_f32[1][2]);
            C_f32[1][2] = mma_16x16x16b16<T, true>(
                b[2][2][2], b[2][2][3], a[1][2][2], a[1][2][3], C_f32[1][2]);
            C_f32[1][2] = mma_16x16x16b16<T, true>(
                b[2][3][0], b[2][3][1], a[1][3][0], a[1][3][1], C_f32[1][2]);
            C_f32[1][2] = mma_16x16x16b16<T, true>(
                b[2][3][2], b[2][3][3], a[1][3][2], a[1][3][3], C_f32[1][2]);

            C_f32[2][2] = mma_16x16x16b16<T, true>(
                b[2][0][0], b[2][0][1], a[2][0][0], a[2][0][1], C_f32[2][2]);
            LDG_B128_BSM_NO_PREDICATOR(WSM_Ldg + 0x4000 * 2 + 0x0000,
                                       APtr + ALdgOffset[0][2])
            C_f32[2][2] = mma_16x16x16b16<T, true>(
                b[2][0][2], b[2][0][3], a[2][0][2], a[2][0][3], C_f32[2][2]);
            C_f32[2][2] = mma_16x16x16b16<T, true>(
                b[2][1][0], b[2][1][1], a[2][1][0], a[2][1][1], C_f32[2][2]);
            C_f32[2][2] = mma_16x16x16b16<T, true>(
                b[2][1][2], b[2][1][3], a[2][1][2], a[2][1][3], C_f32[2][2]);
            C_f32[2][2] = mma_16x16x16b16<T, true>(
                b[2][2][0], b[2][2][1], a[2][2][0], a[2][2][1], C_f32[2][2]);
            C_f32[2][2] = mma_16x16x16b16<T, true>(
                b[2][2][2], b[2][2][3], a[2][2][2], a[2][2][3], C_f32[2][2]);
            C_f32[2][2] = mma_16x16x16b16<T, true>(
                b[2][3][0], b[2][3][1], a[2][3][0], a[2][3][1], C_f32[2][2]);
            C_f32[2][2] = mma_16x16x16b16<T, true>(
                b[2][3][2], b[2][3][3], a[2][3][2], a[2][3][3], C_f32[2][2]);

            C_f32[3][0] = mma_16x16x16b16<T, true>(
                b[0][0][0], b[0][0][1], a[3][0][0], a[3][0][1], C_f32[3][0]);
            LDG_B128_BSM_NO_PREDICATOR(WSM_Ldg + 0x4000 * 2 + 0x1000,
                                       APtr + ALdgOffset[1][2])
            C_f32[3][0] = mma_16x16x16b16<T, true>(
                b[0][0][2], b[0][0][3], a[3][0][2], a[3][0][3], C_f32[3][0]);
            C_f32[3][0] = mma_16x16x16b16<T, true>(
                b[0][1][0], b[0][1][1], a[3][1][0], a[3][1][1], C_f32[3][0]);
            C_f32[3][0] = mma_16x16x16b16<T, true>(
                b[0][1][2], b[0][1][3], a[3][1][2], a[3][1][3], C_f32[3][0]);
            C_f32[3][0] = mma_16x16x16b16<T, true>(
                b[0][2][0], b[0][2][1], a[3][2][0], a[3][2][1], C_f32[3][0]);
            C_f32[3][0] = mma_16x16x16b16<T, true>(
                b[0][2][2], b[0][2][3], a[3][2][2], a[3][2][3], C_f32[3][0]);
            C_f32[3][0] = mma_16x16x16b16<T, true>(
                b[0][3][0], b[0][3][1], a[3][3][0], a[3][3][1], C_f32[3][0]);
            C_f32[3][0] = mma_16x16x16b16<T, true>(
                b[0][3][2], b[0][3][3], a[3][3][2], a[3][3][3], C_f32[3][0]);
            arrive_gvmcnt(4 * (Stage - 5) + 10);
            __builtin_mxc_barrier_inst();

            C_f32[0][3] = mma_16x16x16b16<T, true>(
                b[3][0][0], b[3][0][1], a[0][0][0], a[0][0][1], C_f32[0][3]);
            __builtin_mxc_ldg_b128_bsm_predicator(
                WSM_Ldg + 0x4000 * 2 + 0x2000, BPtr + BLdgOffset[0][2], 0, true,
                true, false, true, startCol + 32, N, MACA_ICMP_SLT);
            C_f32[0][3] = mma_16x16x16b16<T, true>(
                b[3][0][2], b[3][0][3], a[0][0][2], a[0][0][3], C_f32[0][3]);
            b[0][0] = load_layoutc_fragment_from_shared<BLdsType>(
                WSM_lds, BLdsOffset, 0);
            C_f32[0][3] = mma_16x16x16b16<T, true>(
                b[3][1][0], b[3][1][1], a[0][1][0], a[0][1][1], C_f32[0][3]);
            C_f32[0][3] = mma_16x16x16b16<T, true>(
                b[3][1][2], b[3][1][3], a[0][1][2], a[0][1][3], C_f32[0][3]);
            b[0][1] = load_layoutc_fragment_from_shared<BLdsType>(
                WSM_lds, BLdsOffset, 1);
            C_f32[0][3] = mma_16x16x16b16<T, true>(
                b[3][2][0], b[3][2][1], a[0][2][0], a[0][2][1], C_f32[0][3]);
            C_f32[0][3] = mma_16x16x16b16<T, true>(
                b[3][2][2], b[3][2][3], a[0][2][2], a[0][2][3], C_f32[0][3]);
            b[0][2] = load_layoutc_fragment_from_shared<BLdsType>(
                WSM_lds, BLdsOffset, 2);
            C_f32[0][3] = mma_16x16x16b16<T, true>(
                b[3][3][0], b[3][3][1], a[0][3][0], a[0][3][1], C_f32[0][3]);
            C_f32[0][3] = mma_16x16x16b16<T, true>(
                b[3][3][2], b[3][3][3], a[0][3][2], a[0][3][3], C_f32[0][3]);
            b[0][3] = load_layoutc_fragment_from_shared<BLdsType>(
                WSM_lds, BLdsOffset, 3);

            C_f32[3][1] = mma_16x16x16b16<T, true>(
                b[1][0][0], b[1][0][1], a[3][0][0], a[3][0][1], C_f32[3][1]);
            __builtin_mxc_ldg_b128_bsm_predicator(
                WSM_Ldg + 0x4000 * 2 + 0x3000, BPtr + BLdgOffset[1][2], 0, true,
                true, false, true, startCol + 96, N, MACA_ICMP_SLT);
            C_f32[3][1] = mma_16x16x16b16<T, true>(
                b[1][0][2], b[1][0][3], a[3][0][2], a[3][0][3], C_f32[3][1]);
            a[0][0] = load_layoutc_fragment_from_shared<ALdsType>(
                WSM_lds, ALdsOffset, 0);
            C_f32[3][1] = mma_16x16x16b16<T, true>(
                b[1][1][0], b[1][1][1], a[3][1][0], a[3][1][1], C_f32[3][1]);
            C_f32[3][1] = mma_16x16x16b16<T, true>(
                b[1][1][2], b[1][1][3], a[3][1][2], a[3][1][3], C_f32[3][1]);
            a[0][1] = load_layoutc_fragment_from_shared<ALdsType>(
                WSM_lds, ALdsOffset, 1);
            C_f32[3][1] = mma_16x16x16b16<T, true>(
                b[1][2][0], b[1][2][1], a[3][2][0], a[3][2][1], C_f32[3][1]);
            C_f32[3][1] = mma_16x16x16b16<T, true>(
                b[1][2][2], b[1][2][3], a[3][2][2], a[3][2][3], C_f32[3][1]);
            a[0][2] = load_layoutc_fragment_from_shared<ALdsType>(
                WSM_lds, ALdsOffset, 2);
            C_f32[3][1] = mma_16x16x16b16<T, true>(
                b[1][3][0], b[1][3][1], a[3][3][0], a[3][3][1], C_f32[3][1]);
            C_f32[3][1] = mma_16x16x16b16<T, true>(
                b[1][3][2], b[1][3][3], a[3][3][2], a[3][3][3], C_f32[3][1]);
            a[0][3] = load_layoutc_fragment_from_shared<ALdsType>(
                WSM_lds, ALdsOffset, 3);

            C_f32[1][3] = mma_16x16x16b16<T, true>(
                b[3][0][0], b[3][0][1], a[1][0][0], a[1][0][1], C_f32[1][3]);
            LDG_B128_BSM_NO_PREDICATOR(WSM_Ldg + 0x4000 * 3 + 0x0000,
                                       APtr + ALdgOffset[0][3])
            C_f32[1][3] = mma_16x16x16b16<T, true>(
                b[3][0][2], b[3][0][3], a[1][0][2], a[1][0][3], C_f32[1][3]);
            C_f32[1][3] = mma_16x16x16b16<T, true>(
                b[3][1][0], b[3][1][1], a[1][1][0], a[1][1][1], C_f32[1][3]);
            C_f32[1][3] = mma_16x16x16b16<T, true>(
                b[3][1][2], b[3][1][3], a[1][1][2], a[1][1][3], C_f32[1][3]);
            C_f32[1][3] = mma_16x16x16b16<T, true>(
                b[3][2][0], b[3][2][1], a[1][2][0], a[1][2][1], C_f32[1][3]);
            C_f32[1][3] = mma_16x16x16b16<T, true>(
                b[3][2][2], b[3][2][3], a[1][2][2], a[1][2][3], C_f32[1][3]);
            C_f32[1][3] = mma_16x16x16b16<T, true>(
                b[3][3][0], b[3][3][1], a[1][3][0], a[1][3][1], C_f32[1][3]);
            C_f32[1][3] = mma_16x16x16b16<T, true>(
                b[3][3][2], b[3][3][3], a[1][3][2], a[1][3][3], C_f32[1][3]);

            C_f32[3][2] = mma_16x16x16b16<T, true>(
                b[2][0][0], b[2][0][1], a[3][0][0], a[3][0][1], C_f32[3][2]);
            LDG_B128_BSM_NO_PREDICATOR(WSM_Ldg + 0x4000 * 3 + 0x1000,
                                       APtr + ALdgOffset[1][3])
            C_f32[3][2] = mma_16x16x16b16<T, true>(
                b[2][0][2], b[2][0][3], a[3][0][2], a[3][0][3], C_f32[3][2]);
            C_f32[3][2] = mma_16x16x16b16<T, true>(
                b[2][1][0], b[2][1][1], a[3][1][0], a[3][1][1], C_f32[3][2]);
            C_f32[3][2] = mma_16x16x16b16<T, true>(
                b[2][1][2], b[2][1][3], a[3][1][2], a[3][1][3], C_f32[3][2]);
            C_f32[3][2] = mma_16x16x16b16<T, true>(
                b[2][2][0], b[2][2][1], a[3][2][0], a[3][2][1], C_f32[3][2]);
            arrive_gvmcnt(4 * (Stage - 6) + 14);
            __builtin_mxc_barrier_inst();
            C_f32[3][2] = mma_16x16x16b16<T, true>(
                b[2][2][2], b[2][2][3], a[3][2][2], a[3][2][3], C_f32[3][2]);
            a[1][0] = load_layoutc_fragment_from_shared<ALdsType>(
                WSM_lds + 0x4000, ALdsOffset, 0);
            C_f32[3][2] = mma_16x16x16b16<T, true>(
                b[2][3][0], b[2][3][1], a[3][3][0], a[3][3][1], C_f32[3][2]);
            C_f32[3][2] = mma_16x16x16b16<T, true>(
                b[2][3][2], b[2][3][3], a[3][3][2], a[3][3][3], C_f32[3][2]);
            a[1][1] = load_layoutc_fragment_from_shared<ALdsType>(
                WSM_lds + 0x4000, ALdsOffset, 1);

            C_f32[2][3] = mma_16x16x16b16<T, true>(
                b[3][0][0], b[3][0][1], a[2][0][0], a[2][0][1], C_f32[2][3]);
            __builtin_mxc_ldg_b128_bsm_predicator(
                WSM_Ldg + 0x4000 * 3 + 0x2000, BPtr + BLdgOffset[0][3], 0, true,
                true, false, true, startCol + 48, N, MACA_ICMP_SLT);
            C_f32[2][3] = mma_16x16x16b16<T, true>(
                b[3][0][2], b[3][0][3], a[2][0][2], a[2][0][3], C_f32[2][3]);
            a[1][2] = load_layoutc_fragment_from_shared<ALdsType>(
                WSM_lds + 0x4000, ALdsOffset, 2);
            C_f32[2][3] = mma_16x16x16b16<T, true>(
                b[3][1][0], b[3][1][1], a[2][1][0], a[2][1][1], C_f32[2][3]);
            C_f32[2][3] = mma_16x16x16b16<T, true>(
                b[3][1][2], b[3][1][3], a[2][1][2], a[2][1][3], C_f32[2][3]);
            a[1][3] = load_layoutc_fragment_from_shared<ALdsType>(
                WSM_lds + 0x4000, ALdsOffset, 3);
            C_f32[2][3] = mma_16x16x16b16<T, true>(
                b[3][2][0], b[3][2][1], a[2][2][0], a[2][2][1], C_f32[2][3]);
            C_f32[2][3] = mma_16x16x16b16<T, true>(
                b[3][2][2], b[3][2][3], a[2][2][2], a[2][2][3], C_f32[2][3]);
            b[1][0] = load_layoutc_fragment_from_shared<BLdsType>(
                WSM_lds + 0x4000, BLdsOffset, 0);
            C_f32[2][3] = mma_16x16x16b16<T, true>(
                b[3][3][0], b[3][3][1], a[2][3][0], a[2][3][1], C_f32[2][3]);
            C_f32[2][3] = mma_16x16x16b16<T, true>(
                b[3][3][2], b[3][3][3], a[2][3][2], a[2][3][3], C_f32[2][3]);
            b[1][1] = load_layoutc_fragment_from_shared<BLdsType>(
                WSM_lds + 0x4000, BLdsOffset, 1);

            C_f32[3][3] = mma_16x16x16b16<T, true>(
                b[3][0][0], b[3][0][1], a[3][0][0], a[3][0][1], C_f32[3][3]);
            __builtin_mxc_ldg_b128_bsm_predicator(
                WSM_Ldg + 0x4000 * 3 + 0x3000, BPtr + BLdgOffset[1][3], 0, true,
                true, false, true, startCol + 112, N, MACA_ICMP_SLT);
            C_f32[3][3] = mma_16x16x16b16<T, true>(
                b[3][0][2], b[3][0][3], a[3][0][2], a[3][0][3], C_f32[3][3]);
            b[1][2] = load_layoutc_fragment_from_shared<BLdsType>(
                WSM_lds + 0x4000, BLdsOffset, 2);
            C_f32[3][3] = mma_16x16x16b16<T, true>(
                b[3][1][0], b[3][1][1], a[3][1][0], a[3][1][1], C_f32[3][3]);
            C_f32[3][3] = mma_16x16x16b16<T, true>(
                b[3][1][2], b[3][1][3], a[3][1][2], a[3][1][3], C_f32[3][3]);
            b[1][3] = load_layoutc_fragment_from_shared<BLdsType>(
                WSM_lds + 0x4000, BLdsOffset, 3);
            C_f32[3][3] = mma_16x16x16b16<T, true>(
                b[3][2][0], b[3][2][1], a[3][2][0], a[3][2][1], C_f32[3][3]);
            C_f32[3][3] = mma_16x16x16b16<T, true>(
                b[3][2][2], b[3][2][3], a[3][2][2], a[3][2][3], C_f32[3][3]);
            C_f32[3][3] = mma_16x16x16b16<T, true>(
                b[3][3][0], b[3][3][1], a[3][3][0], a[3][3][1], C_f32[3][3]);
            C_f32[3][3] = mma_16x16x16b16<T, true>(
                b[3][3][2], b[3][3][3], a[3][3][2], a[3][3][3], C_f32[3][3]);
        }

        APtr += (128 / 8) * 16 * sizeof(ALdgType);
        BPtr += 16 * N * sizeof(BLdgType);
    }

    if (K > 0) {
        for (int stage_i = 0; stage_i < Stage; ++stage_i) {
            run_layoutc_tail_iteration<T, Stage, FLOAT4, ALdsType, BLdsType,
                                       ALdgType, BLdgType>(
                C_f32, a, b, WSM_Ldg, WSM_lds, ALdsOffset, BLdsOffset,
                ALdgOffset, BLdgOffset, APtr, BPtr, stage_i, K, N, startCol);
        }
    }

    drain_layoutc_tail<T, Stage>(C_f32, a, b, WSM_lds, ALdsOffset, BLdsOffset);

    CStgType bias_load[4];
    epilogue_atom::template load_bias<CStgType, HasOneDimBias>(
        bias_load, bias, startRow, slot, lane);

    if constexpr (OutputContinuousC) {
        Tc *C_ptr = reinterpret_cast<Tc *>(C);
        epilogue_atom::template store_continuousc<Tc, Tscal, CStgType, FLOAT4,
                                                  IsBetaZero, HasOneDimBias>(
            C_ptr, C_f32, M, N, startRow, startCol, slot, lane, alpha, beta,
            bias_load);
    } else {
        CStgType *C_ptr = reinterpret_cast<CStgType *>(C);
        epilogue_atom::template store_layoutc<Tc, Tscal, CStgType, FLOAT4,
                                              IsBetaZero, HasOneDimBias>(
            C_ptr, C_f32, src_N, startRow, startCol, slot, lane, alpha, beta,
            bias_load);
    }
}

template <typename T, typename Tc, typename Tscal, bool IsBetaZero,
          bool HasOneDimBias, typename LayoutAtom = layoutc_layout_atom>
__global__ void cute_tk_bf16_layoutc_128x128x128_stage4(
    const void *A, const void *B, void *C, int M, int N, int K, int lda,
    int ldb, int ldc, Tscal alpha, Tscal beta, const void *bias = nullptr) {
    layoutc_stage4_device<T, Tc, Tscal, IsBetaZero, HasOneDimBias, false,
                          LayoutAtom>(
        A, B, C, M, N, K, lda, ldb, ldc, alpha, beta, bias, blockIdx.x,
        blockIdx.y);
}

} // namespace bf16_c500_tk_cute_local::cute_tk::kernel
