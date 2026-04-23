#pragma once

#include "../layout/stage_geometry.cuh"

namespace bf16_c500_tk_cute_local::primitives {

// Stage layout atom - abstract memory layout for pipeline stages
// Encapsulates bank offsets and stage base addresses
template <typename StageContractT>
struct stage_layout_atom_t {
    using contract = StageContractT;

    static constexpr int stage_count = contract::stage_count;
    static constexpr int stage_bytes = contract::stage_bytes;
    static constexpr int bank0_a_offset = contract::a_bank0_offset;
    static constexpr int bank1_a_offset = contract::a_bank1_offset;
    static constexpr int bank0_b_offset = contract::b_bank0_offset;
    static constexpr int bank1_b_offset = contract::b_bank1_offset;

    __host__ __device__ static constexpr int stage_base_offset(int stage_idx) {
        return contract::stage_base_offset(stage_idx);
    }

    __host__ __device__ static constexpr int bank_a_offset(int bank_idx) {
        return bank_idx == 0 ? bank0_a_offset : bank1_a_offset;
    }

    __host__ __device__ static constexpr int bank_b_offset(int bank_idx) {
        return bank_idx == 0 ? bank0_b_offset : bank1_b_offset;
    }

    __host__ __device__ static constexpr int a_stage_offset(int stage_idx,
                                                            int bank_idx) {
        return stage_base_offset(stage_idx) + bank_a_offset(bank_idx);
    }

    __host__ __device__ static constexpr int b_stage_offset(int stage_idx,
                                                            int bank_idx) {
        return stage_base_offset(stage_idx) + bank_b_offset(bank_idx);
    }
};

} // namespace bf16_c500_tk_cute_local::primitives