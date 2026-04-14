#pragma once

#include <stdint.h>

namespace kittens::arch::c500::gemm::primitives {

using bf16_layouta_native_vec = __NATIVE_VECTOR__(4, uint32_t);
using bf16_layouta_native_pair = __NATIVE_VECTOR__(2, uint32_t);
using bf16_layouta_native_acc = __NATIVE_VECTOR__(4, float);

struct bf16_layouta_native_stage_operands {
    bf16_layouta_native_vec a[4][4];
    bf16_layouta_native_vec b[4][4];
};

__device__ inline void zero_native_accumulators(bf16_layouta_native_acc (&acc)[4][4]) {
#pragma unroll
    for (int m = 0; m < 4; ++m) {
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            acc[m][n] = bf16_layouta_native_acc{0.f, 0.f, 0.f, 0.f};
        }
    }
}

__device__ inline void load_native_stage_operands(bf16_layouta_native_stage_operands &dst,
                                                  uint8_t *wsm_lds,
                                                  int stage_off,
                                                  const int (&a_lds_offset)[4],
                                                  const int (&b_lds_offset)[4]) {
#pragma unroll
    for (int kg = 0; kg < 4; ++kg) {
        dst.a[0][kg] = *reinterpret_cast<bf16_layouta_native_vec *>(wsm_lds + stage_off + a_lds_offset[kg]);
        dst.b[0][kg] = *reinterpret_cast<bf16_layouta_native_vec *>(wsm_lds + stage_off + b_lds_offset[kg]);
    }
}

__device__ inline void load_native_stage_operands_for_rows(
    bf16_layouta_native_stage_operands &dst,
    uint8_t *wsm_lds,
    const int (&stage_offsets)[4],
    const int (&a_lds_offset)[4],
    const int (&b_lds_offset)[4]) {
#pragma unroll
    for (int m = 0; m < 4; ++m) {
#pragma unroll
        for (int kg = 0; kg < 4; ++kg) {
            dst.a[m][kg] =
                *reinterpret_cast<bf16_layouta_native_vec *>(wsm_lds + stage_offsets[m] + a_lds_offset[kg]);
        }
    }
#pragma unroll
    for (int n = 0; n < 4; ++n) {
#pragma unroll
        for (int kg = 0; kg < 4; ++kg) {
            dst.b[n][kg] =
                *reinterpret_cast<bf16_layouta_native_vec *>(wsm_lds + stage_offsets[n] + b_lds_offset[kg]);
        }
    }
}

__device__ inline void consume_native_stage_full(bf16_layouta_native_acc (&acc)[4][4],
                                                 const bf16_layouta_native_stage_operands &ops) {
#pragma unroll
    for (int m = 0; m < 4; ++m) {
#pragma unroll
        for (int n = 0; n < 4; ++n) {
#pragma unroll
            for (int kg = 0; kg < 4; ++kg) {
                acc[m][n] = __builtin_mxc_mma_16x16x16bf16(
                    bf16_layouta_native_pair{ops.b[n][kg][0], ops.b[n][kg][1]},
                    bf16_layouta_native_pair{ops.a[m][kg][0], ops.a[m][kg][1]},
                    acc[m][n]);
                acc[m][n] = __builtin_mxc_mma_16x16x16bf16(
                    bf16_layouta_native_pair{ops.b[n][kg][2], ops.b[n][kg][3]},
                    bf16_layouta_native_pair{ops.a[m][kg][2], ops.a[m][kg][3]},
                    acc[m][n]);
            }
        }
    }
}

} // namespace kittens::arch::c500::gemm::primitives
