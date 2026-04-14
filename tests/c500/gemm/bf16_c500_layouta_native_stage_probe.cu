#include "testing_flags.cuh"

#ifdef TEST_C500_GEMM_LAYOUTA_NATIVE_STAGE_PROBE

#include "testing_commons.cuh"
#include "../../../kernels/gemm/common.cuh"

#include "arch/c500/gemm/families/bf16_muxi_128x128x128_stage4.cuh"
#include "arch/c500/gemm/primitives/bf16_layouta_native_stage.cuh"

namespace c500::mma::layouta_native_stage_probe {

namespace {

using family = kittens::arch::c500::gemm::families::bf16_muxi_128x128x128_stage4;
using fallback_family = family::fallback_family;
using contracts = family::contracts;
using shared_tile_a = family::shared_tile_a;
using shared_tile_c = family::shared_tile_c;
using atom = family::atom;
using frag_c = family::frag_c;
using reg_tile_c = family::reg_tile_c;

constexpr int kM = 128;
constexpr int kN = 128;
constexpr int kK = 128;
constexpr int kPairMaps = 6;

template<int M, int K>
using a_gl = kittens::gl<kittens::bf16, 1, 1, M, K, shared_tile_a>;
template<int N, int K>
using b_layouta_gl = kittens::gl<kittens::bf16, 1, 1, N, K>;
template<int M, int N>
using c_gl = kittens::gl<kittens::bf16, 1, 1, M, N, shared_tile_c>;

struct gemm_globals {
    a_gl<kM, kK> a;
    b_layouta_gl<kN, kK> b;
    c_gl<kM, kN> c;
};

__global__ __launch_bounds__(contracts::kThreads)
void native_stage_probe_kernel(const __grid_constant__ gemm_globals g) {
    using native_vec = kittens::arch::c500::gemm::primitives::bf16_layouta_native_vec;
    using native_acc = kittens::arch::c500::gemm::primitives::bf16_layouta_native_acc;
    using native_ops = kittens::arch::c500::gemm::primitives::bf16_layouta_native_stage_operands;

    const int tid = threadIdx.x;
    const int slot = kittens::warpid();
    const int lane = kittens::laneid();
    const int quarter_lane = lane & 15;
    const int quarter_warp = lane >> 4;
    const int row_group = slot / contracts::kWaveN;
    const int col_group = slot % contracts::kWaveN;

    constexpr int kVecElems = sizeof(native_vec) / sizeof(kittens::bf16);
    const int lda_vec = g.a.template stride<2>() / kVecElems;

    auto *a_ptr = reinterpret_cast<uint8_t *>(g.a.raw_ptr);
    auto *b_ptr = reinterpret_cast<uint8_t *>(g.b.raw_ptr);

    int a_ldg_offset[2][4];
    a_ldg_offset[0][0] = (tid + 16 * lda_vec * 0) * static_cast<int>(sizeof(native_vec));
    a_ldg_offset[0][1] = (tid + 16 * lda_vec * 1) * static_cast<int>(sizeof(native_vec));
    a_ldg_offset[0][2] = (tid + 16 * lda_vec * 2) * static_cast<int>(sizeof(native_vec));
    a_ldg_offset[0][3] = (tid + 16 * lda_vec * 3) * static_cast<int>(sizeof(native_vec));
    a_ldg_offset[1][0] = (tid + 16 * lda_vec * 4) * static_cast<int>(sizeof(native_vec));
    a_ldg_offset[1][1] = (tid + 16 * lda_vec * 5) * static_cast<int>(sizeof(native_vec));
    a_ldg_offset[1][2] = (tid + 16 * lda_vec * 6) * static_cast<int>(sizeof(native_vec));
    a_ldg_offset[1][3] = (tid + 16 * lda_vec * 7) * static_cast<int>(sizeof(native_vec));

    const int b_row_offset = (quarter_lane * g.b.template stride<2>() / kVecElems) + slot * 4 + quarter_warp;

    int b_ldg_offset[2][4];
    b_ldg_offset[0][0] = b_row_offset * static_cast<int>(sizeof(native_vec)) + 0 * 16 * g.b.template stride<2>() * sizeof(kittens::bf16);
    b_ldg_offset[0][1] = b_row_offset * static_cast<int>(sizeof(native_vec)) + 1 * 16 * g.b.template stride<2>() * sizeof(kittens::bf16);
    b_ldg_offset[0][2] = b_row_offset * static_cast<int>(sizeof(native_vec)) + 2 * 16 * g.b.template stride<2>() * sizeof(kittens::bf16);
    b_ldg_offset[0][3] = b_row_offset * static_cast<int>(sizeof(native_vec)) + 3 * 16 * g.b.template stride<2>() * sizeof(kittens::bf16);
    b_ldg_offset[1][0] = b_row_offset * static_cast<int>(sizeof(native_vec)) + 4 * 16 * g.b.template stride<2>() * sizeof(kittens::bf16);
    b_ldg_offset[1][1] = b_row_offset * static_cast<int>(sizeof(native_vec)) + 5 * 16 * g.b.template stride<2>() * sizeof(kittens::bf16);
    b_ldg_offset[1][2] = b_row_offset * static_cast<int>(sizeof(native_vec)) + 6 * 16 * g.b.template stride<2>() * sizeof(kittens::bf16);
    b_ldg_offset[1][3] = b_row_offset * static_cast<int>(sizeof(native_vec)) + 7 * 16 * g.b.template stride<2>() * sizeof(kittens::bf16);

    int a_lds_offset[4];
    int b_lds_offset[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        a_lds_offset[i] = (lane + (slot / 2) * (0x1000 / static_cast<int>(sizeof(native_vec))) +
                           i * (0x400 / static_cast<int>(sizeof(native_vec)))) *
                          static_cast<int>(sizeof(native_vec));
        b_lds_offset[i] = (lane + (0x2000 / static_cast<int>(sizeof(native_vec))) +
                           (slot & 1) * (0x1000 / static_cast<int>(sizeof(native_vec))) +
                           i * (0x400 / static_cast<int>(sizeof(native_vec)))) *
                          static_cast<int>(sizeof(native_vec));
    }

    __shared__ KITTENS_ALIGN_AS(16) uint8_t wsm[0x10000];
    auto *wsm_ldg = wsm + slot * 0x400;
    auto *wsm_lds = wsm;

    auto issue_native_stage = [&](int stage_slot, int global_stage) {
        const int stage_off = 0x4000 * stage_slot;
        kittens::arch::c500::detail::ldg_b128_bsm_pred(
            wsm_ldg + stage_off + 0x0000,
            a_ptr + a_ldg_offset[0][global_stage],
            0,
            kK / kVecElems);
        kittens::arch::c500::detail::ldg_b128_bsm_pred(
            wsm_ldg + stage_off + 0x1000,
            a_ptr + a_ldg_offset[1][global_stage],
            0,
            kK / kVecElems);
        kittens::arch::c500::detail::ldg_b128_bsm_pred(
            wsm_ldg + stage_off + 0x2000,
            b_ptr + b_ldg_offset[0][global_stage],
            global_stage * 16,
            kN);
        kittens::arch::c500::detail::ldg_b128_bsm_pred(
            wsm_ldg + stage_off + 0x3000,
            b_ptr + b_ldg_offset[1][global_stage],
            (4 + global_stage) * 16,
            kN);
    };

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        issue_native_stage(i, i);
    }
    kittens::arch::c500::primitives::wait_stage_prefix<4>(4);

    const int stage_offsets[4] = {0x0000, 0x4000, 0x8000, 0xC000};
    native_ops ops{};
    kittens::arch::c500::gemm::primitives::load_native_stage_operands_for_rows(
        ops, wsm_lds, stage_offsets, a_lds_offset, b_lds_offset);

    native_acc acc_native[4][4];
    kittens::arch::c500::gemm::primitives::zero_native_accumulators(acc_native);
    kittens::arch::c500::gemm::primitives::consume_native_stage_full(acc_native, ops);

    frag_c acc[4][4];
#pragma unroll
    for (int m = 0; m < 4; ++m) {
#pragma unroll
        for (int n = 0; n < 4; ++n) {
#pragma unroll
            for (int r = 0; r < atom::c_registers; ++r) {
                acc[m][n].reg[r] = acc_native[m][n][r];
            }
        }
    }

    family::store_accumulators_layouta(g.c, acc, 0, 0, row_group, col_group);
}

__device__ inline kittens::arch::c500::gemm::primitives::bf16_layouta_native_pair
select_native_pair(const kittens::arch::c500::gemm::primitives::bf16_layouta_native_vec &vec,
                   int map_id,
                   int half_id) {
    using native_pair = kittens::arch::c500::gemm::primitives::bf16_layouta_native_pair;
    switch (map_id) {
        case 0:
            return half_id == 0 ? native_pair{vec[0], vec[1]} : native_pair{vec[2], vec[3]};
        case 1:
            return half_id == 0 ? native_pair{vec[1], vec[0]} : native_pair{vec[3], vec[2]};
        case 2:
            return half_id == 0 ? native_pair{vec[0], vec[2]} : native_pair{vec[1], vec[3]};
        case 3:
            return half_id == 0 ? native_pair{vec[2], vec[0]} : native_pair{vec[3], vec[1]};
        case 4:
            return half_id == 0 ? native_pair{vec[0], vec[3]} : native_pair{vec[1], vec[2]};
        default:
            return half_id == 0 ? native_pair{vec[3], vec[0]} : native_pair{vec[2], vec[1]};
    }
}

__device__ inline void consume_native_stage_variant(
    kittens::arch::c500::gemm::primitives::bf16_layouta_native_acc (&acc)[4][4],
    const kittens::arch::c500::gemm::primitives::bf16_layouta_native_stage_operands &ops,
    int a_map,
    int b_map) {
#pragma unroll
    for (int m = 0; m < 4; ++m) {
#pragma unroll
        for (int n = 0; n < 4; ++n) {
#pragma unroll
            for (int kg = 0; kg < 4; ++kg) {
                acc[m][n] = __builtin_mxc_mma_16x16x16bf16(
                    select_native_pair(ops.b[n][kg], b_map, 0),
                    select_native_pair(ops.a[m][kg], a_map, 0),
                    acc[m][n]);
                acc[m][n] = __builtin_mxc_mma_16x16x16bf16(
                    select_native_pair(ops.b[n][kg], b_map, 1),
                    select_native_pair(ops.a[m][kg], a_map, 1),
                    acc[m][n]);
            }
        }
    }
}

__global__ __launch_bounds__(contracts::kThreads)
void native_stage_probe_kernel_export(const __grid_constant__ gemm_globals g) {
    using native_vec = kittens::arch::c500::gemm::primitives::bf16_layouta_native_vec;
    using native_acc = kittens::arch::c500::gemm::primitives::bf16_layouta_native_acc;
    using native_ops = kittens::arch::c500::gemm::primitives::bf16_layouta_native_stage_operands;

    const int tid = threadIdx.x;
    const int slot = kittens::warpid();
    const int lane = kittens::laneid();
    const int quarter_lane = lane & 15;
    const int quarter_warp = lane >> 4;
    const int row_group = slot / contracts::kWaveN;
    const int col_group = slot % contracts::kWaveN;

    constexpr int kVecElems = sizeof(native_vec) / sizeof(kittens::bf16);
    const int lda_vec = g.a.template stride<2>() / kVecElems;

    auto *a_ptr = reinterpret_cast<uint8_t *>(g.a.raw_ptr);
    auto *b_ptr = reinterpret_cast<uint8_t *>(g.b.raw_ptr);

    int a_ldg_offset[2][4];
    a_ldg_offset[0][0] = (tid + 16 * lda_vec * 0) * static_cast<int>(sizeof(native_vec));
    a_ldg_offset[0][1] = (tid + 16 * lda_vec * 1) * static_cast<int>(sizeof(native_vec));
    a_ldg_offset[0][2] = (tid + 16 * lda_vec * 2) * static_cast<int>(sizeof(native_vec));
    a_ldg_offset[0][3] = (tid + 16 * lda_vec * 3) * static_cast<int>(sizeof(native_vec));
    a_ldg_offset[1][0] = (tid + 16 * lda_vec * 4) * static_cast<int>(sizeof(native_vec));
    a_ldg_offset[1][1] = (tid + 16 * lda_vec * 5) * static_cast<int>(sizeof(native_vec));
    a_ldg_offset[1][2] = (tid + 16 * lda_vec * 6) * static_cast<int>(sizeof(native_vec));
    a_ldg_offset[1][3] = (tid + 16 * lda_vec * 7) * static_cast<int>(sizeof(native_vec));

    const int b_row_offset = (quarter_lane * g.b.template stride<2>() / kVecElems) + slot * 4 + quarter_warp;

    int b_ldg_offset[2][4];
    b_ldg_offset[0][0] = b_row_offset * static_cast<int>(sizeof(native_vec)) + 0 * 16 * g.b.template stride<2>() * sizeof(kittens::bf16);
    b_ldg_offset[0][1] = b_row_offset * static_cast<int>(sizeof(native_vec)) + 1 * 16 * g.b.template stride<2>() * sizeof(kittens::bf16);
    b_ldg_offset[0][2] = b_row_offset * static_cast<int>(sizeof(native_vec)) + 2 * 16 * g.b.template stride<2>() * sizeof(kittens::bf16);
    b_ldg_offset[0][3] = b_row_offset * static_cast<int>(sizeof(native_vec)) + 3 * 16 * g.b.template stride<2>() * sizeof(kittens::bf16);
    b_ldg_offset[1][0] = b_row_offset * static_cast<int>(sizeof(native_vec)) + 4 * 16 * g.b.template stride<2>() * sizeof(kittens::bf16);
    b_ldg_offset[1][1] = b_row_offset * static_cast<int>(sizeof(native_vec)) + 5 * 16 * g.b.template stride<2>() * sizeof(kittens::bf16);
    b_ldg_offset[1][2] = b_row_offset * static_cast<int>(sizeof(native_vec)) + 6 * 16 * g.b.template stride<2>() * sizeof(kittens::bf16);
    b_ldg_offset[1][3] = b_row_offset * static_cast<int>(sizeof(native_vec)) + 7 * 16 * g.b.template stride<2>() * sizeof(kittens::bf16);

    int a_lds_offset[4];
    int b_lds_offset[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        a_lds_offset[i] = (lane + (slot / 2) * (0x1000 / static_cast<int>(sizeof(native_vec))) +
                           i * (0x400 / static_cast<int>(sizeof(native_vec)))) *
                          static_cast<int>(sizeof(native_vec));
        b_lds_offset[i] = (lane + (0x2000 / static_cast<int>(sizeof(native_vec))) +
                           (slot & 1) * (0x1000 / static_cast<int>(sizeof(native_vec))) +
                           i * (0x400 / static_cast<int>(sizeof(native_vec)))) *
                          static_cast<int>(sizeof(native_vec));
    }

    __shared__ KITTENS_ALIGN_AS(16) uint8_t wsm[0x10000];
    auto *wsm_ldg = wsm + slot * 0x400;
    auto *wsm_lds = wsm;

    auto issue_native_stage = [&](int stage_slot, int global_stage) {
        const int stage_off = 0x4000 * stage_slot;
        kittens::arch::c500::detail::ldg_b128_bsm_pred(
            wsm_ldg + stage_off + 0x0000,
            a_ptr + a_ldg_offset[0][global_stage],
            0,
            kK / kVecElems);
        kittens::arch::c500::detail::ldg_b128_bsm_pred(
            wsm_ldg + stage_off + 0x1000,
            a_ptr + a_ldg_offset[1][global_stage],
            0,
            kK / kVecElems);
        kittens::arch::c500::detail::ldg_b128_bsm_pred(
            wsm_ldg + stage_off + 0x2000,
            b_ptr + b_ldg_offset[0][global_stage],
            global_stage * 16,
            kN);
        kittens::arch::c500::detail::ldg_b128_bsm_pred(
            wsm_ldg + stage_off + 0x3000,
            b_ptr + b_ldg_offset[1][global_stage],
            (4 + global_stage) * 16,
            kN);
    };

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        issue_native_stage(i, i);
    }
    kittens::arch::c500::primitives::wait_stage_prefix<4>(4);

    const int stage_offsets[4] = {0x0000, 0x4000, 0x8000, 0xC000};
    native_ops ops{};
    kittens::arch::c500::gemm::primitives::load_native_stage_operands_for_rows(
        ops, wsm_lds, stage_offsets, a_lds_offset, b_lds_offset);

    native_acc acc_native[4][4];
    kittens::arch::c500::gemm::primitives::zero_native_accumulators(acc_native);
    kittens::arch::c500::gemm::primitives::consume_native_stage_full(acc_native, ops);

    frag_c acc[4][4];
#pragma unroll
    for (int m = 0; m < 4; ++m) {
#pragma unroll
        for (int n = 0; n < 4; ++n) {
#pragma unroll
            for (int r = 0; r < atom::c_registers; ++r) {
                acc[m][n].reg[r] = acc_native[m][n][r];
            }
        }
    }

    reg_tile_c out;
    fallback_family::export_accumulators(out, acc);
    kittens::warp::store(g.c, out, {0, 0, row_group, col_group});
}

__global__ __launch_bounds__(contracts::kThreads)
void native_stage_probe_kernel_variant(const __grid_constant__ gemm_globals g,
                                       int a_map,
                                       int b_map) {
    using native_vec = kittens::arch::c500::gemm::primitives::bf16_layouta_native_vec;
    using native_acc = kittens::arch::c500::gemm::primitives::bf16_layouta_native_acc;
    using native_ops = kittens::arch::c500::gemm::primitives::bf16_layouta_native_stage_operands;

    const int tid = threadIdx.x;
    const int slot = kittens::warpid();
    const int lane = kittens::laneid();
    const int quarter_lane = lane & 15;
    const int quarter_warp = lane >> 4;
    const int row_group = slot / contracts::kWaveN;
    const int col_group = slot % contracts::kWaveN;

    constexpr int kVecElems = sizeof(native_vec) / sizeof(kittens::bf16);
    const int lda_vec = g.a.template stride<2>() / kVecElems;

    auto *a_ptr = reinterpret_cast<uint8_t *>(g.a.raw_ptr);
    auto *b_ptr = reinterpret_cast<uint8_t *>(g.b.raw_ptr);

    int a_ldg_offset[2][4];
    a_ldg_offset[0][0] = (tid + 16 * lda_vec * 0) * static_cast<int>(sizeof(native_vec));
    a_ldg_offset[0][1] = (tid + 16 * lda_vec * 1) * static_cast<int>(sizeof(native_vec));
    a_ldg_offset[0][2] = (tid + 16 * lda_vec * 2) * static_cast<int>(sizeof(native_vec));
    a_ldg_offset[0][3] = (tid + 16 * lda_vec * 3) * static_cast<int>(sizeof(native_vec));
    a_ldg_offset[1][0] = (tid + 16 * lda_vec * 4) * static_cast<int>(sizeof(native_vec));
    a_ldg_offset[1][1] = (tid + 16 * lda_vec * 5) * static_cast<int>(sizeof(native_vec));
    a_ldg_offset[1][2] = (tid + 16 * lda_vec * 6) * static_cast<int>(sizeof(native_vec));
    a_ldg_offset[1][3] = (tid + 16 * lda_vec * 7) * static_cast<int>(sizeof(native_vec));

    const int b_row_offset = (quarter_lane * g.b.template stride<2>() / kVecElems) + slot * 4 + quarter_warp;

    int b_ldg_offset[2][4];
    b_ldg_offset[0][0] = b_row_offset * static_cast<int>(sizeof(native_vec)) + 0 * 16 * g.b.template stride<2>() * sizeof(kittens::bf16);
    b_ldg_offset[0][1] = b_row_offset * static_cast<int>(sizeof(native_vec)) + 1 * 16 * g.b.template stride<2>() * sizeof(kittens::bf16);
    b_ldg_offset[0][2] = b_row_offset * static_cast<int>(sizeof(native_vec)) + 2 * 16 * g.b.template stride<2>() * sizeof(kittens::bf16);
    b_ldg_offset[0][3] = b_row_offset * static_cast<int>(sizeof(native_vec)) + 3 * 16 * g.b.template stride<2>() * sizeof(kittens::bf16);
    b_ldg_offset[1][0] = b_row_offset * static_cast<int>(sizeof(native_vec)) + 4 * 16 * g.b.template stride<2>() * sizeof(kittens::bf16);
    b_ldg_offset[1][1] = b_row_offset * static_cast<int>(sizeof(native_vec)) + 5 * 16 * g.b.template stride<2>() * sizeof(kittens::bf16);
    b_ldg_offset[1][2] = b_row_offset * static_cast<int>(sizeof(native_vec)) + 6 * 16 * g.b.template stride<2>() * sizeof(kittens::bf16);
    b_ldg_offset[1][3] = b_row_offset * static_cast<int>(sizeof(native_vec)) + 7 * 16 * g.b.template stride<2>() * sizeof(kittens::bf16);

    int a_lds_offset[4];
    int b_lds_offset[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        a_lds_offset[i] = (lane + (slot / 2) * (0x1000 / static_cast<int>(sizeof(native_vec))) +
                           i * (0x400 / static_cast<int>(sizeof(native_vec)))) *
                          static_cast<int>(sizeof(native_vec));
        b_lds_offset[i] = (lane + (0x2000 / static_cast<int>(sizeof(native_vec))) +
                           (slot & 1) * (0x1000 / static_cast<int>(sizeof(native_vec))) +
                           i * (0x400 / static_cast<int>(sizeof(native_vec)))) *
                          static_cast<int>(sizeof(native_vec));
    }

    __shared__ KITTENS_ALIGN_AS(16) uint8_t wsm[0x10000];
    auto *wsm_ldg = wsm + slot * 0x400;
    auto *wsm_lds = wsm;

    auto issue_native_stage = [&](int stage_slot, int global_stage) {
        const int stage_off = 0x4000 * stage_slot;
        kittens::arch::c500::detail::ldg_b128_bsm_pred(
            wsm_ldg + stage_off + 0x0000,
            a_ptr + a_ldg_offset[0][global_stage],
            0,
            kK / kVecElems);
        kittens::arch::c500::detail::ldg_b128_bsm_pred(
            wsm_ldg + stage_off + 0x1000,
            a_ptr + a_ldg_offset[1][global_stage],
            0,
            kK / kVecElems);
        kittens::arch::c500::detail::ldg_b128_bsm_pred(
            wsm_ldg + stage_off + 0x2000,
            b_ptr + b_ldg_offset[0][global_stage],
            global_stage * 16,
            kN);
        kittens::arch::c500::detail::ldg_b128_bsm_pred(
            wsm_ldg + stage_off + 0x3000,
            b_ptr + b_ldg_offset[1][global_stage],
            (4 + global_stage) * 16,
            kN);
    };

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        issue_native_stage(i, i);
    }
    kittens::arch::c500::primitives::wait_stage_prefix<4>(4);

    const int stage_offsets[4] = {0x0000, 0x4000, 0x8000, 0xC000};
    native_ops ops{};
    kittens::arch::c500::gemm::primitives::load_native_stage_operands_for_rows(
        ops, wsm_lds, stage_offsets, a_lds_offset, b_lds_offset);

    native_acc acc_native[4][4];
    kittens::arch::c500::gemm::primitives::zero_native_accumulators(acc_native);
    consume_native_stage_variant(acc_native, ops, a_map, b_map);

    frag_c acc[4][4];
#pragma unroll
    for (int m = 0; m < 4; ++m) {
#pragma unroll
        for (int n = 0; n < 4; ++n) {
#pragma unroll
            for (int r = 0; r < atom::c_registers; ++r) {
                acc[m][n].reg[r] = acc_native[m][n][r];
            }
        }
    }

    reg_tile_c out;
    fallback_family::export_accumulators(out, acc);
    kittens::warp::store(g.c, out, {0, 0, row_group, col_group});
}

__global__ __launch_bounds__(contracts::kThreads)
void native_stage_probe_kernel_debug(const __grid_constant__ gemm_globals g,
                                     uint32_t *a_words_out,
                                     uint32_t *b_words_out) {
    using native_vec = kittens::arch::c500::gemm::primitives::bf16_layouta_native_vec;
    using native_ops = kittens::arch::c500::gemm::primitives::bf16_layouta_native_stage_operands;

    const int tid = threadIdx.x;
    const int slot = kittens::warpid();
    const int lane = kittens::laneid();
    const int quarter_lane = lane & 15;
    const int quarter_warp = lane >> 4;

    constexpr int kVecElems = sizeof(native_vec) / sizeof(kittens::bf16);
    const int lda_vec = g.a.template stride<2>() / kVecElems;

    auto *a_ptr = reinterpret_cast<uint8_t *>(g.a.raw_ptr);
    auto *b_ptr = reinterpret_cast<uint8_t *>(g.b.raw_ptr);

    int a_ldg_offset[2][4];
    a_ldg_offset[0][0] = (tid + 16 * lda_vec * 0) * static_cast<int>(sizeof(native_vec));
    a_ldg_offset[0][1] = (tid + 16 * lda_vec * 1) * static_cast<int>(sizeof(native_vec));
    a_ldg_offset[0][2] = (tid + 16 * lda_vec * 2) * static_cast<int>(sizeof(native_vec));
    a_ldg_offset[0][3] = (tid + 16 * lda_vec * 3) * static_cast<int>(sizeof(native_vec));
    a_ldg_offset[1][0] = (tid + 16 * lda_vec * 4) * static_cast<int>(sizeof(native_vec));
    a_ldg_offset[1][1] = (tid + 16 * lda_vec * 5) * static_cast<int>(sizeof(native_vec));
    a_ldg_offset[1][2] = (tid + 16 * lda_vec * 6) * static_cast<int>(sizeof(native_vec));
    a_ldg_offset[1][3] = (tid + 16 * lda_vec * 7) * static_cast<int>(sizeof(native_vec));

    const int b_row_offset = (quarter_lane * g.b.template stride<2>() / kVecElems) + slot * 4 + quarter_warp;

    int b_ldg_offset[2][4];
    b_ldg_offset[0][0] = b_row_offset * static_cast<int>(sizeof(native_vec)) + 0 * 16 * g.b.template stride<2>() * sizeof(kittens::bf16);
    b_ldg_offset[0][1] = b_row_offset * static_cast<int>(sizeof(native_vec)) + 1 * 16 * g.b.template stride<2>() * sizeof(kittens::bf16);
    b_ldg_offset[0][2] = b_row_offset * static_cast<int>(sizeof(native_vec)) + 2 * 16 * g.b.template stride<2>() * sizeof(kittens::bf16);
    b_ldg_offset[0][3] = b_row_offset * static_cast<int>(sizeof(native_vec)) + 3 * 16 * g.b.template stride<2>() * sizeof(kittens::bf16);
    b_ldg_offset[1][0] = b_row_offset * static_cast<int>(sizeof(native_vec)) + 4 * 16 * g.b.template stride<2>() * sizeof(kittens::bf16);
    b_ldg_offset[1][1] = b_row_offset * static_cast<int>(sizeof(native_vec)) + 5 * 16 * g.b.template stride<2>() * sizeof(kittens::bf16);
    b_ldg_offset[1][2] = b_row_offset * static_cast<int>(sizeof(native_vec)) + 6 * 16 * g.b.template stride<2>() * sizeof(kittens::bf16);
    b_ldg_offset[1][3] = b_row_offset * static_cast<int>(sizeof(native_vec)) + 7 * 16 * g.b.template stride<2>() * sizeof(kittens::bf16);

    int a_lds_offset[4];
    int b_lds_offset[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        a_lds_offset[i] = (lane + (slot / 2) * (0x1000 / static_cast<int>(sizeof(native_vec))) +
                           i * (0x400 / static_cast<int>(sizeof(native_vec)))) *
                          static_cast<int>(sizeof(native_vec));
        b_lds_offset[i] = (lane + (0x2000 / static_cast<int>(sizeof(native_vec))) +
                           (slot & 1) * (0x1000 / static_cast<int>(sizeof(native_vec))) +
                           i * (0x400 / static_cast<int>(sizeof(native_vec)))) *
                          static_cast<int>(sizeof(native_vec));
    }

    __shared__ KITTENS_ALIGN_AS(16) uint8_t wsm[0x10000];
    auto *wsm_ldg = wsm + slot * 0x400;
    auto *wsm_lds = wsm;

    auto issue_native_stage = [&](int stage_slot, int global_stage) {
        const int stage_off = 0x4000 * stage_slot;
        kittens::arch::c500::detail::ldg_b128_bsm_pred(
            wsm_ldg + stage_off + 0x0000,
            a_ptr + a_ldg_offset[0][global_stage],
            0,
            kK / kVecElems);
        kittens::arch::c500::detail::ldg_b128_bsm_pred(
            wsm_ldg + stage_off + 0x1000,
            a_ptr + a_ldg_offset[1][global_stage],
            0,
            kK / kVecElems);
        kittens::arch::c500::detail::ldg_b128_bsm_pred(
            wsm_ldg + stage_off + 0x2000,
            b_ptr + b_ldg_offset[0][global_stage],
            global_stage * 16,
            kN);
        kittens::arch::c500::detail::ldg_b128_bsm_pred(
            wsm_ldg + stage_off + 0x3000,
            b_ptr + b_ldg_offset[1][global_stage],
            (4 + global_stage) * 16,
            kN);
    };

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        issue_native_stage(i, i);
    }
    kittens::arch::c500::primitives::wait_stage_prefix<4>(4);

    if (threadIdx.x == 0) {
        const int stage_offsets[4] = {0x0000, 0x4000, 0x8000, 0xC000};
        native_ops ops{};
        kittens::arch::c500::gemm::primitives::load_native_stage_operands_for_rows(
            ops, wsm_lds, stage_offsets, a_lds_offset, b_lds_offset);

        for (int m = 0; m < 4; ++m) {
            for (int kg = 0; kg < 4; ++kg) {
                for (int w = 0; w < 4; ++w) {
                    a_words_out[(m * 4 + kg) * 4 + w] = ops.a[m][kg][w];
                    b_words_out[(m * 4 + kg) * 4 + w] = ops.b[m][kg][w];
                }
            }
        }
    }
}

bool run_probe(test_data &results) {
    test_info info{"c500_gemm_bf16_layouta_native_stage_contract", test_result::FAILED};

    kittens::bf16 *d_a = nullptr;
    kittens::bf16 *d_b = nullptr;
    kittens::bf16 *d_b_layouta = nullptr;
    kittens::bf16 *d_c_layouta = nullptr;
    kittens::bf16 *d_c_export = nullptr;
    kittens::bf16 *d_c_variant = nullptr;
    kittens::bf16 *d_ref = nullptr;
    uint32_t *d_a_words = nullptr;
    uint32_t *d_b_words = nullptr;

    cudaMalloc(&d_a, kM * kK * sizeof(kittens::bf16));
    cudaMalloc(&d_b, kK * kN * sizeof(kittens::bf16));
    cudaMalloc(&d_b_layouta, kN * kK * sizeof(kittens::bf16));
    cudaMalloc(&d_c_layouta, kM * kN * sizeof(kittens::bf16));
    cudaMalloc(&d_c_export, kM * kN * sizeof(kittens::bf16));
    cudaMalloc(&d_c_variant, kM * kN * sizeof(kittens::bf16));
    cudaMalloc(&d_ref, kM * kN * sizeof(kittens::bf16));
    cudaMalloc(&d_a_words, 4 * 4 * 4 * sizeof(uint32_t));
    cudaMalloc(&d_b_words, 4 * 4 * 4 * sizeof(uint32_t));
    CudaCheckError();

    fill<__nv_bfloat16, FillMode::RANDOM>(reinterpret_cast<__nv_bfloat16 *>(d_a), kM * kK, 2024, -1.0f, 1.0f);
    fill<__nv_bfloat16, FillMode::RANDOM>(reinterpret_cast<__nv_bfloat16 *>(d_b), kK * kN, 3026, -1.0f, 1.0f);
    fill<__nv_bfloat16, FillMode::CONSTANT>(reinterpret_cast<__nv_bfloat16 *>(d_c_layouta), kM * kN, 0.0f);
    fill<__nv_bfloat16, FillMode::CONSTANT>(reinterpret_cast<__nv_bfloat16 *>(d_c_export), kM * kN, 0.0f);
    fill<__nv_bfloat16, FillMode::CONSTANT>(reinterpret_cast<__nv_bfloat16 *>(d_c_variant), kM * kN, 0.0f);
    fill<__nv_bfloat16, FillMode::CONSTANT>(reinterpret_cast<__nv_bfloat16 *>(d_ref), kM * kN, 0.0f);
    CudaCheckError();

    std::vector<kittens::bf16> h_b_layouta(kN * kK);
    std::vector<kittens::bf16> h_b(kK * kN);
    cudaMemcpy(h_b.data(), d_b, kK * kN * sizeof(kittens::bf16), cudaMemcpyDeviceToHost);
    CudaCheckError();
    for (int k = 0; k < kK; ++k) {
        for (int n = 0; n < kN; ++n) {
            h_b_layouta[n * kK + k] = h_b[k * kN + n];
        }
    }
    cudaMemcpy(d_b_layouta, h_b_layouta.data(), kN * kK * sizeof(kittens::bf16), cudaMemcpyHostToDevice);
    CudaCheckError();

    reference_gemm<__nv_bfloat16, __nv_bfloat16, false>(reinterpret_cast<__nv_bfloat16 *>(d_ref),
                                                        reinterpret_cast<__nv_bfloat16 *>(d_a),
                                                        reinterpret_cast<__nv_bfloat16 *>(d_b),
                                                        kM, kN, kK);
    cudaDeviceSynchronize();
    CudaCheckError();

    gemm_globals g{
        a_gl<kM, kK>{d_a, nullptr, nullptr, nullptr, nullptr},
        b_layouta_gl<kN, kK>{d_b_layouta, nullptr, nullptr, nullptr, nullptr},
        c_gl<kM, kN>{d_c_layouta, nullptr, nullptr, nullptr, nullptr}
    };

    native_stage_probe_kernel<<<1, contracts::kThreads>>>(g);
    cudaDeviceSynchronize();
    CudaCheckError();

    gemm_globals g_export{
        a_gl<kM, kK>{d_a, nullptr, nullptr, nullptr, nullptr},
        b_layouta_gl<kN, kK>{d_b_layouta, nullptr, nullptr, nullptr, nullptr},
        c_gl<kM, kN>{d_c_export, nullptr, nullptr, nullptr, nullptr}
    };

    native_stage_probe_kernel_export<<<1, contracts::kThreads>>>(g_export);
    cudaDeviceSynchronize();
    CudaCheckError();

    native_stage_probe_kernel_debug<<<1, contracts::kThreads>>>(g, d_a_words, d_b_words);
    cudaDeviceSynchronize();
    CudaCheckError();

    std::vector<float> empty_input;
    std::vector<float> ref_out(kM * kN, 0.0f);
    std::vector<float> out_layouta(kM * kN, 0.0f);
    std::vector<float> out_export(kM * kN, 0.0f);
    std::vector<kittens::bf16> h_ref(kM * kN);
    std::vector<kittens::bf16> h_out_layouta(kM * kN);
    std::vector<kittens::bf16> h_out_export(kM * kN);
    std::vector<uint32_t> h_a_words(4 * 4 * 4);
    std::vector<uint32_t> h_b_words(4 * 4 * 4);
    cudaMemcpy(h_ref.data(), d_ref, kM * kN * sizeof(kittens::bf16), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_layouta.data(), d_c_layouta, kM * kN * sizeof(kittens::bf16), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_export.data(), d_c_export, kM * kN * sizeof(kittens::bf16), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_a_words.data(), d_a_words, h_a_words.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b_words.data(), d_b_words, h_b_words.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    CudaCheckError();
    for (int i = 0; i < kM * kN; ++i) {
        ref_out[i] = __bfloat162float(h_ref[i]);
        out_layouta[i] = __bfloat162float(h_out_layouta[i]);
        out_export[i] = __bfloat162float(h_out_export[i]);
    }

    info.result = validate(d_a, d_c_layouta, empty_input, ref_out, info.label, kN, 0.02f);
    if (info.result != test_result::PASSED) {
        int first_layouta = -1;
        for (int i = 0; i < kM * kN; ++i) {
            if (fabsf(ref_out[i] - out_layouta[i]) > 0.02f) {
                first_layouta = i;
                break;
            }
        }
        if (first_layouta >= 0) {
            const int row = first_layouta / kN;
            const int col = first_layouta % kN;
            std::cout << "layouta store mismatch row=" << row
                      << " col=" << col
                      << " ref=" << ref_out[first_layouta]
                      << " got=" << out_layouta[first_layouta] << std::endl;
        }
        int first_export = -1;
        for (int i = 0; i < kM * kN; ++i) {
            if (fabsf(ref_out[i] - out_export[i]) > 0.02f) {
                first_export = i;
                break;
            }
        }
        if (first_export >= 0) {
            const int row = first_export / kN;
            const int col = first_export % kN;
            std::cout << "standard export mismatch row=" << row
                      << " col=" << col
                      << " ref=" << ref_out[first_export]
                      << " got=" << out_export[first_export] << std::endl;
        }
        int best_a_map = -1;
        int best_b_map = -1;
        int best_mismatches = kM * kN + 1;
        float best_max_abs = 1.0e30f;
        std::vector<kittens::bf16> h_variant(kM * kN);
        std::vector<float> out_variant(kM * kN, 0.0f);
        for (int a_map = 0; a_map < kPairMaps; ++a_map) {
            for (int b_map = 0; b_map < kPairMaps; ++b_map) {
                cudaMemset(d_c_variant, 0, kM * kN * sizeof(kittens::bf16));
                CudaCheckError();
                gemm_globals g_variant{
                    a_gl<kM, kK>{d_a, nullptr, nullptr, nullptr, nullptr},
                    b_layouta_gl<kN, kK>{d_b_layouta, nullptr, nullptr, nullptr, nullptr},
                    c_gl<kM, kN>{d_c_variant, nullptr, nullptr, nullptr, nullptr}
                };
                native_stage_probe_kernel_variant<<<1, contracts::kThreads>>>(g_variant, a_map, b_map);
                cudaDeviceSynchronize();
                CudaCheckError();
                cudaMemcpy(h_variant.data(), d_c_variant, kM * kN * sizeof(kittens::bf16), cudaMemcpyDeviceToHost);
                CudaCheckError();

                int mismatches = 0;
                float max_abs = 0.0f;
                for (int i = 0; i < kM * kN; ++i) {
                    out_variant[i] = __bfloat162float(h_variant[i]);
                    const float diff = fabsf(ref_out[i] - out_variant[i]);
                    if (diff > 0.02f) {
                        ++mismatches;
                    }
                    if (diff > max_abs) {
                        max_abs = diff;
                    }
                }
                if (mismatches < best_mismatches ||
                    (mismatches == best_mismatches && max_abs < best_max_abs)) {
                    best_mismatches = mismatches;
                    best_max_abs = max_abs;
                    best_a_map = a_map;
                    best_b_map = b_map;
                }
            }
        }
        std::cout << "best pair-map candidate a_map=" << best_a_map
                  << " b_map=" << best_b_map
                  << " mismatches=" << best_mismatches
                  << " max_abs=" << best_max_abs << std::endl;
        std::cout << "lane0 slot0 a_words:" << std::endl;
        for (int m = 0; m < 4; ++m) {
            for (int kg = 0; kg < 4; ++kg) {
                const int base = (m * 4 + kg) * 4;
                std::cout << "  a[" << m << "][" << kg << "] = {"
                          << h_a_words[base + 0] << ", "
                          << h_a_words[base + 1] << ", "
                          << h_a_words[base + 2] << ", "
                          << h_a_words[base + 3] << "}" << std::endl;
            }
        }
        std::cout << "lane0 slot0 b_words:" << std::endl;
        for (int n = 0; n < 4; ++n) {
            for (int kg = 0; kg < 4; ++kg) {
                const int base = (n * 4 + kg) * 4;
                std::cout << "  b[" << n << "][" << kg << "] = {"
                          << h_b_words[base + 0] << ", "
                          << h_b_words[base + 1] << ", "
                          << h_b_words[base + 2] << ", "
                          << h_b_words[base + 3] << "}" << std::endl;
            }
        }
    }
    results.push_back(info);

    cudaFree(d_b_words);
    cudaFree(d_a_words);
    cudaFree(d_ref);
    cudaFree(d_c_variant);
    cudaFree(d_c_export);
    cudaFree(d_c_layouta);
    cudaFree(d_b_layouta);
    cudaFree(d_b);
    cudaFree(d_a);
    return info.result == test_result::PASSED;
}

} // namespace

void tests(test_data &results) {
    std::cout << " ----- Starting ops/c500/mma/layouta_native_stage_probe tests! -----\n" << std::endl;
    run_probe(results);
}

} // namespace c500::mma::layouta_native_stage_probe

#endif
