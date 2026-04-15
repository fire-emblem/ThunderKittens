#pragma once

#include <maca.h>
#include <maca_bfloat16.h>
#include <mc_runtime.h>

#include <bit>
#include <cute/tensor.hpp>

#include "../contracts/square_tt_tile_contract.cuh"
#include "primitives/pipeline/square_tt_fragment_atom.cuh"
#include "primitives/pipeline/square_tt_stage_io_atom.cuh"
#include "primitives/structure/square_tt_thread_map_atom.cuh"

namespace bf16_c500_tk_cute_local::cute_tk::kernel {

template <typename T, typename Tc, typename Tscal, bool IsBetaZero,
          bool HasOneDimBias>
__global__ void __launch_bounds__(512) cute_tk_bf16_square_tt_tile256x256x64_stage4(
    const void *A, const void *B, void *C, int M, int N, int K, int lda,
    int ldb, int ldc, Tscal alpha, Tscal beta, const void *bias) {
    (void)lda;
    (void)ldb;
    (void)ldc;
    (void)beta;
    (void)bias;
    static_assert(IsBetaZero, "square_tt_256x256x64 assumes beta == 0");
    static_assert(!HasOneDimBias, "square_tt_256x256x64 does not support bias");
    static_assert(std::is_same_v<T, __maca_bfloat16>,
                  "square_tt_256x256x64 is currently BF16-only");
    static_assert(std::is_same_v<Tc, __maca_bfloat16>,
                  "square_tt_256x256x64 output is currently BF16-only");

    using namespace cute;
    using contract =
        ::bf16_c500_tk_cute_local::contracts::square_tt_tile_contract;
    using thread_map =
        ::bf16_c500_tk_cute_local::cute_tk::square_tt_thread_map_atom;
    using int4_t = typename thread_map::int4_t;
    using ab_type = typename thread_map::ab_type;
    using a_ldg_type = typename thread_map::a_ldg_type;
    using b_ldg_type = typename thread_map::b_ldg_type;
    using sts_type = typename thread_map::sts_type;
    using lds_type = typename thread_map::lds_type;

    constexpr int kTileM = contract::tile_m;
    constexpr int kTileN = contract::tile_n;
    constexpr int kTileK = contract::tile_k;
    constexpr int kASmemSize = contract::a_smem_bytes;

    extern __shared__ uint8_t smem[];

    int tidx = threadIdx.x;
    int bidx = blockIdx.y;
    int bidy = blockIdx.x;
    int wave_id = threadIdx.x / contract::wave_size;
    int lane_id = threadIdx.x % contract::wave_size;

    Tensor tA = make_tensor(make_gmem_ptr((T *)A), make_shape(M, K),
                            make_stride(K, Int<1>{}));
    Tensor tB = make_tensor(make_gmem_ptr((T *)B), make_shape(N, K),
                            make_stride(Int<1>{}, N));

    Tensor gA =
        local_tile(tA, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(bidy, _));
    Tensor gB =
        local_tile(tB, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(bidx, _));

    const int row_limit = min(kTileM, M - bidy * kTileM);
    const int col_limit = min(kTileN, N - bidx * kTileN);
    const int k_head = (K - 1) % kTileK + 1;

    a_ldg_type A_frag[contract::ldg_a_count];
    int ldg_kA = thread_map::a_ldg_k(tidx);
    int ldg_m[contract::ldg_a_count];
    for (int i = 0; i < contract::ldg_a_count; ++i) {
        ldg_m[i] = thread_map::a_ldg_m(tidx, row_limit, i);
        A_frag[i] = __builtin_mxc_ldg_b64_predicator(
            &(gA(ldg_m[i], ldg_kA, 0)), 0, true, true, false, false, ldg_kA,
            k_head, MACA_ICMP_SLT);
    }

    uint32_t B_frag[contract::ldg_b_rows][contract::ldg_b_cols];
    int ldg_kB_base = thread_map::b_ldg_k_base(lane_id);
    int ldg_n = thread_map::b_ldg_n(tidx, wave_id, col_limit);
    for (int i = 0; i < contract::ldg_b_rows; ++i) {
        for (int j = 0; j < contract::ldg_b_cols; ++j) {
            *reinterpret_cast<b_ldg_type *>(&(B_frag[i][j])) =
                __builtin_mxc_ldg_b32_predicator(
                    &(gB(ldg_n, ldg_kB_base + j + i * 16, 0)), 0, true, true,
                    false, false, ldg_kB_base + j + i * 16, k_head,
                    MACA_ICMP_SLT);
        }
    }

#define STS_B64X2_I(i)                                                           \
    square_tt_stage_io_atom::template store_pair<sts_type>(                      \
        smem, sts_off, contract::sts_stride_bytes, i, A_frag[i], A_frag[i + 1])
#define LDS_B64X2_STAGE_I(stage, i)                                              \
    square_tt_stage_io_atom::template load_pair<lds_type>(                       \
        a[i], a[i + 1], smem, lds_off[stage], contract::lds_stride_bytes, i)
#define LDG_A(tileK, i)                                                          \
    asm(";--------------");                                                                   \
    A_frag[i] = __builtin_mxc_ldg_b64(&(gA(ldg_m[i], ldg_kA, tileK)), 0, -1,    \
                                      true, true, false, false);                 \
    asm(";--------------");
#define LDG_B(tileK, i, j)                                                       \
    asm(";--------------");                                                                   \
    *reinterpret_cast<b_ldg_type *>(&(B_frag[i][j])) =                           \
        __builtin_mxc_ldg_b32(&(gB(ldg_n, ldg_kB_base + j + i * 16, tileK)), 0,  \
                              -1, true, true, false, false);                     \
    asm(";--------------");
#define MMA_MNK(m, n, k)                                                         \
    accum[m][n] = __builtin_mxc_mma_16x16x16bf16(a[m], b[k][n], accum[m][n]);

    int sts_off = thread_map::sts_offset_bytes(tidx);
    square_tt_stage_io_atom::template store_pair<sts_type>(
        smem, sts_off, contract::sts_stride_bytes, 0, A_frag[0], A_frag[1]);
    square_tt_stage_io_atom::template store_pair<sts_type>(
        smem, sts_off, contract::sts_stride_bytes, 2, A_frag[2], A_frag[3]);
    square_tt_stage_io_atom::template store_pair<sts_type>(
        smem, sts_off, contract::sts_stride_bytes, 4, A_frag[4], A_frag[5]);
    square_tt_stage_io_atom::template store_pair<sts_type>(
        smem, sts_off, contract::sts_stride_bytes, 6, A_frag[6], A_frag[7]);

    ab_type a[contract::accum_m];
    int lds_off[4];
    for (int i = 0; i < 4; ++i) {
        lds_off[i] = thread_map::lds_offset_bytes(tidx, lane_id, i);
    }
    int4_t accum[contract::accum_m][contract::accum_n] = {0};
    __syncthreadshared();
    square_tt_stage_io_atom::template load_pair<lds_type>(
        a[0], a[1], smem, lds_off[0], contract::lds_stride_bytes, 0);
    square_tt_stage_io_atom::template load_pair<lds_type>(
        a[2], a[3], smem, lds_off[0], contract::lds_stride_bytes, 2);
    square_tt_stage_io_atom::template load_pair<lds_type>(
        a[4], a[5], smem, lds_off[0], contract::lds_stride_bytes, 4);
    square_tt_stage_io_atom::template load_pair<lds_type>(
        a[6], a[7], smem, lds_off[0], contract::lds_stride_bytes, 6);

    ab_type b[2][2];
    square_tt_fragment_atom::pack_b_quartet(b[0], B_frag[0]);

    int num_tile_k = size<2>(gA);
    uint32_t tileK = 1;
    for (; tileK < num_tile_k; ++tileK) {
        LDG_A(tileK, 0); LDG_A(tileK, 1); MMA_MNK(0, 0, 0); LDG_A(tileK, 2);
        MMA_MNK(0, 1, 0);
        square_tt_fragment_atom::pack_b_quartet(b[1], B_frag[1]);
        MMA_MNK(1, 0, 0); LDG_A(tileK, 3); LDS_B64X2_STAGE_I(0, 8); MMA_MNK(1, 1, 0);
        LDS_B64X2_STAGE_I(0, 10); MMA_MNK(2, 0, 0); LDG_A(tileK, 4);
        LDS_B64X2_STAGE_I(0, 12);
        MMA_MNK(2, 1, 0); LDS_B64X2_STAGE_I(0, 14); MMA_MNK(3, 0, 0); LDG_A(tileK, 5);
        MMA_MNK(3, 1, 0); MMA_MNK(4, 0, 0); LDG_A(tileK, 6); MMA_MNK(4, 1, 0);
        MMA_MNK(5, 0, 0); LDG_A(tileK, 7); MMA_MNK(5, 1, 0); MMA_MNK(6, 0, 0);
        MMA_MNK(6, 1, 0); MMA_MNK(7, 0, 0); MMA_MNK(7, 1, 0);

        LDS_B64X2_STAGE_I(1, 0); MMA_MNK(8, 0, 0); LDS_B64X2_STAGE_I(1, 2);
        MMA_MNK(8, 1, 0); LDS_B64X2_STAGE_I(1, 4); MMA_MNK(9, 0, 0);
        LDS_B64X2_STAGE_I(1, 6); MMA_MNK(9, 1, 0); MMA_MNK(10, 0, 0);
        MMA_MNK(10, 1, 0); MMA_MNK(11, 0, 0); LDG_B(tileK, 0, 0); MMA_MNK(11, 1, 0);
        MMA_MNK(12, 0, 0); LDG_B(tileK, 0, 1); MMA_MNK(12, 1, 0); MMA_MNK(13, 0, 0);
        MMA_MNK(13, 1, 0); MMA_MNK(14, 0, 0); LDG_B(tileK, 0, 2); MMA_MNK(14, 1, 0);
        MMA_MNK(15, 0, 0); MMA_MNK(15, 1, 0);

        LDG_B(tileK, 0, 3); MMA_MNK(0, 0, 1); LDS_B64X2_STAGE_I(1, 8); MMA_MNK(0, 1, 1);
        LDS_B64X2_STAGE_I(1, 10); MMA_MNK(1, 0, 1); LDS_B64X2_STAGE_I(1, 12);
        MMA_MNK(1, 1, 1); LDS_B64X2_STAGE_I(1, 14); MMA_MNK(2, 0, 1); MMA_MNK(2, 1, 1);
        MMA_MNK(3, 0, 1); MMA_MNK(3, 1, 1); LDG_B(tileK, 1, 0); MMA_MNK(4, 0, 1);
        MMA_MNK(4, 1, 1); LDG_B(tileK, 1, 1); MMA_MNK(5, 0, 1); MMA_MNK(5, 1, 1);
        LDG_B(tileK, 1, 2); MMA_MNK(6, 0, 1); MMA_MNK(6, 1, 1); LDG_B(tileK, 1, 3);
        MMA_MNK(7, 0, 1); MMA_MNK(7, 1, 1);

        LDS_B64X2_STAGE_I(2, 0); MMA_MNK(8, 0, 1); LDS_B64X2_STAGE_I(2, 2); MMA_MNK(8, 1, 1);
        LDS_B64X2_STAGE_I(2, 4); MMA_MNK(9, 0, 1); LDS_B64X2_STAGE_I(2, 6); MMA_MNK(9, 1, 1);
        square_tt_fragment_atom::pack_b_quartet(b[0], B_frag[2]);
        MMA_MNK(10, 0, 1);
        MMA_MNK(10, 1, 1); MMA_MNK(11, 0, 1); MMA_MNK(11, 1, 1); LDG_B(tileK, 2, 0);
        MMA_MNK(12, 0, 1); MMA_MNK(12, 1, 1); LDG_B(tileK, 2, 1); MMA_MNK(13, 0, 1);
        MMA_MNK(13, 1, 1); LDG_B(tileK, 2, 2); MMA_MNK(14, 0, 1); MMA_MNK(14, 1, 1);
        LDG_B(tileK, 2, 3); MMA_MNK(15, 0, 1); MMA_MNK(15, 1, 1);

        LDS_B64X2_STAGE_I(2, 8); MMA_MNK(0, 0, 0); LDS_B64X2_STAGE_I(2, 10); MMA_MNK(0, 1, 0);
        LDS_B64X2_STAGE_I(2, 12); MMA_MNK(1, 0, 0); LDS_B64X2_STAGE_I(2, 14); MMA_MNK(1, 1, 0);
        MMA_MNK(2, 0, 0); MMA_MNK(2, 1, 0); MMA_MNK(3, 0, 0); MMA_MNK(3, 1, 0);
        MMA_MNK(4, 0, 0); MMA_MNK(4, 1, 0); MMA_MNK(5, 0, 0); MMA_MNK(5, 1, 0);
        MMA_MNK(6, 0, 0); MMA_MNK(6, 1, 0); MMA_MNK(7, 0, 0); MMA_MNK(7, 1, 0);

        LDS_B64X2_STAGE_I(3, 0); MMA_MNK(8, 0, 0); LDS_B64X2_STAGE_I(3, 2);
        square_tt_fragment_atom::pack_b_quartet(b[1], B_frag[3]);
        MMA_MNK(8, 1, 0); LDS_B64X2_STAGE_I(3, 4);
        MMA_MNK(9, 0, 0); LDS_B64X2_STAGE_I(3, 6); MMA_MNK(9, 1, 0); LDG_B(tileK, 3, 0);
        MMA_MNK(10, 0, 0); MMA_MNK(10, 1, 0); LDG_B(tileK, 3, 1); MMA_MNK(11, 0, 0);
        MMA_MNK(11, 1, 0); LDG_B(tileK, 3, 2); MMA_MNK(12, 0, 0); MMA_MNK(12, 1, 0);
        LDG_B(tileK, 3, 3); MMA_MNK(13, 0, 0); MMA_MNK(13, 1, 0); MMA_MNK(14, 0, 0);
        MMA_MNK(14, 1, 0); MMA_MNK(15, 0, 0); MMA_MNK(15, 1, 0);

        LDS_B64X2_STAGE_I(3, 8); LDS_B64X2_STAGE_I(3, 10); MMA_MNK(0, 0, 1);
        LDS_B64X2_STAGE_I(3, 12); LDS_B64X2_STAGE_I(3, 14); MMA_MNK(0, 1, 1);
        MMA_MNK(1, 0, 1); MMA_MNK(1, 1, 1); MMA_MNK(2, 0, 1); MMA_MNK(2, 1, 1);
        MMA_MNK(3, 0, 1);
        sts_off ^= kASmemSize;
        lds_off[0] ^= kASmemSize;
        MMA_MNK(3, 1, 1);
        lds_off[1] ^= kASmemSize;
        lds_off[2] ^= kASmemSize;
        lds_off[3] ^= kASmemSize;
        MMA_MNK(4, 0, 1); MMA_MNK(4, 1, 1); MMA_MNK(5, 0, 1); MMA_MNK(5, 1, 1);
        MMA_MNK(6, 0, 1); MMA_MNK(6, 1, 1); MMA_MNK(7, 0, 1); MMA_MNK(7, 1, 1);

        STS_B64X2_I(0); MMA_MNK(8, 0, 1); STS_B64X2_I(2); MMA_MNK(8, 1, 1);
        MMA_MNK(9, 0, 1); STS_B64X2_I(4); MMA_MNK(9, 1, 1); STS_B64X2_I(6);
        MMA_MNK(10, 0, 1); MMA_MNK(10, 1, 1); MMA_MNK(11, 0, 1); MMA_MNK(11, 1, 1);
        __syncthreadshared();
        LDS_B64X2_STAGE_I(0, 0); MMA_MNK(12, 0, 1); LDS_B64X2_STAGE_I(0, 2);
        LDS_B64X2_STAGE_I(0, 4); LDS_B64X2_STAGE_I(0, 6); MMA_MNK(12, 1, 1);
        MMA_MNK(13, 0, 1); MMA_MNK(13, 1, 1); MMA_MNK(14, 0, 1); MMA_MNK(14, 1, 1);
        MMA_MNK(15, 0, 1); MMA_MNK(15, 1, 1);
    }

    MMA_MNK(0, 0, 0); LDS_B64X2_STAGE_I(0, 8); MMA_MNK(0, 1, 0); LDS_B64X2_STAGE_I(0, 10);
    MMA_MNK(1, 0, 0); LDS_B64X2_STAGE_I(0, 12); MMA_MNK(1, 1, 0); LDS_B64X2_STAGE_I(0, 14);
    MMA_MNK(2, 0, 0); MMA_MNK(2, 1, 0); MMA_MNK(3, 0, 0); MMA_MNK(3, 1, 0);
    MMA_MNK(4, 0, 0); MMA_MNK(4, 1, 0); MMA_MNK(5, 0, 0); MMA_MNK(5, 1, 0);
    MMA_MNK(6, 0, 0); MMA_MNK(6, 1, 0); MMA_MNK(7, 0, 0); MMA_MNK(7, 1, 0);

    LDS_B64X2_STAGE_I(1, 0); MMA_MNK(8, 0, 0); LDS_B64X2_STAGE_I(1, 2); MMA_MNK(8, 1, 0);
    LDS_B64X2_STAGE_I(1, 4); MMA_MNK(9, 0, 0); LDS_B64X2_STAGE_I(1, 6); MMA_MNK(9, 1, 0);
    MMA_MNK(10, 0, 0); MMA_MNK(10, 1, 0); MMA_MNK(11, 0, 0); MMA_MNK(11, 1, 0);
    MMA_MNK(12, 0, 0); MMA_MNK(12, 1, 0); MMA_MNK(13, 0, 0); MMA_MNK(13, 1, 0);
    MMA_MNK(14, 0, 0); MMA_MNK(14, 1, 0);
    square_tt_fragment_atom::pack_b_quartet(b[1], B_frag[1]);
    MMA_MNK(15, 0, 0);
    MMA_MNK(15, 1, 0);

    LDS_B64X2_STAGE_I(1, 8); MMA_MNK(0, 0, 1); LDS_B64X2_STAGE_I(1, 10); MMA_MNK(0, 1, 1);
    LDS_B64X2_STAGE_I(1, 12); MMA_MNK(1, 0, 1); LDS_B64X2_STAGE_I(1, 14); MMA_MNK(1, 1, 1);
    MMA_MNK(2, 0, 1); MMA_MNK(2, 1, 1); MMA_MNK(3, 0, 1); MMA_MNK(3, 1, 1);
    MMA_MNK(4, 0, 1); MMA_MNK(4, 1, 1); MMA_MNK(5, 0, 1); MMA_MNK(5, 1, 1);
    MMA_MNK(6, 0, 1); MMA_MNK(6, 1, 1); MMA_MNK(7, 0, 1); MMA_MNK(7, 1, 1);

    LDS_B64X2_STAGE_I(2, 0); MMA_MNK(8, 0, 1); LDS_B64X2_STAGE_I(2, 2); MMA_MNK(8, 1, 1);
    LDS_B64X2_STAGE_I(2, 4); MMA_MNK(9, 0, 1); LDS_B64X2_STAGE_I(2, 6); MMA_MNK(9, 1, 1);
    MMA_MNK(10, 0, 1); MMA_MNK(10, 1, 1); MMA_MNK(11, 0, 1); MMA_MNK(11, 1, 1);
    MMA_MNK(12, 0, 1); MMA_MNK(12, 1, 1); MMA_MNK(13, 0, 1); MMA_MNK(13, 1, 1);
    MMA_MNK(14, 0, 1); MMA_MNK(14, 1, 1);
    square_tt_fragment_atom::pack_b_quartet(b[0], B_frag[2]);
    MMA_MNK(15, 0, 1);
    MMA_MNK(15, 1, 1);

    LDS_B64X2_STAGE_I(2, 8); MMA_MNK(0, 0, 0); LDS_B64X2_STAGE_I(2, 10); MMA_MNK(0, 1, 0);
    LDS_B64X2_STAGE_I(2, 12); MMA_MNK(1, 0, 0); LDS_B64X2_STAGE_I(2, 14); MMA_MNK(1, 1, 0);
    MMA_MNK(2, 0, 0); MMA_MNK(2, 1, 0); MMA_MNK(3, 0, 0); MMA_MNK(3, 1, 0);
    MMA_MNK(4, 0, 0); MMA_MNK(4, 1, 0); MMA_MNK(5, 0, 0); MMA_MNK(5, 1, 0);
    MMA_MNK(6, 0, 0); MMA_MNK(6, 1, 0); MMA_MNK(7, 0, 0); MMA_MNK(7, 1, 0);

    LDS_B64X2_STAGE_I(3, 0); MMA_MNK(8, 0, 0); LDS_B64X2_STAGE_I(3, 2); MMA_MNK(8, 1, 0);
    LDS_B64X2_STAGE_I(3, 4); MMA_MNK(9, 0, 0); LDS_B64X2_STAGE_I(3, 6); MMA_MNK(9, 1, 0);
    MMA_MNK(10, 0, 0); MMA_MNK(10, 1, 0); MMA_MNK(11, 0, 0); MMA_MNK(11, 1, 0);
    MMA_MNK(12, 0, 0); MMA_MNK(12, 1, 0); MMA_MNK(13, 0, 0); MMA_MNK(13, 1, 0);
    MMA_MNK(14, 0, 0); MMA_MNK(14, 1, 0);
    square_tt_fragment_atom::pack_b_quartet(b[1], B_frag[3]);
    MMA_MNK(15, 0, 0);
    MMA_MNK(15, 1, 0);

    LDS_B64X2_STAGE_I(3, 8); MMA_MNK(0, 0, 1); MMA_MNK(0, 1, 1); LDS_B64X2_STAGE_I(3, 10);
    MMA_MNK(1, 0, 1); LDS_B64X2_STAGE_I(3, 12); MMA_MNK(1, 1, 1); LDS_B64X2_STAGE_I(3, 14);
    MMA_MNK(2, 0, 1); MMA_MNK(2, 1, 1); MMA_MNK(3, 0, 1); MMA_MNK(3, 1, 1);
    MMA_MNK(4, 0, 1); MMA_MNK(4, 1, 1); MMA_MNK(5, 0, 1); MMA_MNK(5, 1, 1);
    MMA_MNK(6, 0, 1); MMA_MNK(6, 1, 1); MMA_MNK(7, 0, 1); MMA_MNK(7, 1, 1);
    MMA_MNK(8, 0, 1); MMA_MNK(8, 1, 1); MMA_MNK(9, 0, 1); MMA_MNK(9, 1, 1);
    MMA_MNK(10, 0, 1); MMA_MNK(10, 1, 1); MMA_MNK(11, 0, 1); MMA_MNK(11, 1, 1);
    MMA_MNK(12, 0, 1); MMA_MNK(12, 1, 1); MMA_MNK(13, 0, 1); MMA_MNK(13, 1, 1);
    MMA_MNK(14, 0, 1); MMA_MNK(14, 1, 1); MMA_MNK(15, 0, 1); MMA_MNK(15, 1, 1);

    Tensor tC = make_tensor(make_gmem_ptr((Tc *)C), make_shape(M, N),
                            make_stride(N, Int<1>{}));
    Tensor gC =
        local_tile(tC, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(bidy, bidx));

    int colC = thread_map::epilogue_col(tidx, wave_id);
    int rowC_base = thread_map::epilogue_row_base(lane_id);
    bool colC_mask = colC < col_limit;
    for (uint32_t i = 0; i < 16; ++i) {
        for (uint32_t j = 0; j < 4; ++j) {
            int rowC = rowC_base + i * 16 + j;
            uint32_t raw0 = reinterpret_cast<uint32_t *>(&(accum[i][0]))[j];
            uint32_t raw1 = reinterpret_cast<uint32_t *>(&(accum[i][1]))[j];
            float f0 = std::bit_cast<float>(raw0);
            float f1 = std::bit_cast<float>(raw1);
            int out_row0 = colC;
            int out_row1 = colC + 1;
            int out_col = rowC;
            if (out_row0 < col_limit && out_col < row_limit) {
                gC(out_row0, out_col) = static_cast<Tc>(f0 * alpha);
            }
            if (out_row1 < col_limit && out_col < row_limit) {
                gC(out_row1, out_col) = static_cast<Tc>(f1 * alpha);
            }
        }
    }

#undef MMA_MNK
#undef LDG_B
#undef LDG_A
#undef LDS_B64X2_STAGE_I
#undef STS_B64X2_I
#undef LDS_B64X2_STAGE_I
#undef LDSx2
#undef STS_B64X2_I
#undef STSx2
#undef TK_FENC
}

} // namespace bf16_c500_tk_cute_local::cute_tk::kernel
