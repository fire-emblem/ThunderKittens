#pragma once

#include "../../contracts/stage_contract.cuh"

namespace bf16_c500_tk_cute_local::cute_tk {

template <typename StageContractT>
struct stage_layout_atom {
    using contract = StageContractT;

    static constexpr int stage_count = contract::stage_count;
    static constexpr int stage_bytes = contract::stage_bytes;
    static constexpr int a_bank0_offset = contract::a_bank0_offset;
    static constexpr int a_bank1_offset = contract::a_bank1_offset;
    static constexpr int b_bank0_offset = contract::b_bank0_offset;
    static constexpr int b_bank1_offset = contract::b_bank1_offset;

    __host__ __device__ static constexpr int stage_base_offset(int stage_idx) {
        return contract::stage_base_offset(stage_idx);
    }

    __host__ __device__ static constexpr int a_stage_offset(int stage_idx,
                                                            int bank_idx) {
        return contract::a_stage_offset(stage_idx, bank_idx);
    }

    __host__ __device__ static constexpr int b_stage_offset(int stage_idx,
                                                            int bank_idx) {
        return contract::b_stage_offset(stage_idx, bank_idx);
    }
};

using default_stage_layout_atom =
    stage_layout_atom<::bf16_c500_tk_cute_local::contracts::stage_contract>;

} // namespace bf16_c500_tk_cute_local::cute_tk
