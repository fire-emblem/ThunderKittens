#pragma once

#include <cuda_runtime.h>

namespace bf16_c500_tk_local::contracts {

struct stage_contract {
    static constexpr int stage_count = 4;
    static constexpr int stage_bytes = 0x4000;
    static constexpr int a_bank0_offset = 0x0000;
    static constexpr int a_bank1_offset = 0x1000;
    static constexpr int b_bank0_offset = 0x2000;
    static constexpr int b_bank1_offset = 0x3000;

    __host__ __device__ static constexpr int stage_base_offset(int stage_idx) {
        return stage_idx * stage_bytes;
    }

    __host__ __device__ static constexpr int a_bank_offset(int bank_idx) {
        return bank_idx == 0 ? a_bank0_offset : a_bank1_offset;
    }

    __host__ __device__ static constexpr int b_bank_offset(int bank_idx) {
        return bank_idx == 0 ? b_bank0_offset : b_bank1_offset;
    }

    __host__ __device__ static constexpr int a_stage_offset(int stage_idx,
                                                            int bank_idx) {
        return stage_base_offset(stage_idx) + a_bank_offset(bank_idx);
    }

    __host__ __device__ static constexpr int b_stage_offset(int stage_idx,
                                                            int bank_idx) {
        return stage_base_offset(stage_idx) + b_bank_offset(bank_idx);
    }
};

} // namespace bf16_c500_tk_local::contracts
