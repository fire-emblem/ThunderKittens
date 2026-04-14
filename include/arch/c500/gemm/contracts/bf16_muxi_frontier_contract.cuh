#pragma once

#include <stdint.h>

namespace kittens::arch::c500::gemm::contracts {

struct bf16_muxi_frontier_contract {
    using mask_type = uint32_t;
    using bank_mask_type = uint32_t;

    enum class frontier_state : int {
        f0 = 0,
        f1 = 1,
        f2 = 2,
    };

    static constexpr int kAccRows = 4;
    static constexpr int kAccCols = 4;

    // Phase-A skeleton masks:
    // F0: top-left 2x2 band
    // F1: top-left 3x3 band
    // F2: full 4x4 tile
    static constexpr mask_type kFrontierF0 = 0x0033u;
    static constexpr mask_type kFrontierF1 = 0x0777u;
    static constexpr mask_type kFrontierF2 = 0xffffu;

    __host__ __device__ static constexpr frontier_state frontier_state_for_bank_mask(bank_mask_type bank_mask) {
        return bank_mask == 0x3u ? frontier_state::f0 :
               bank_mask == 0x7u ? frontier_state::f1 :
                                   frontier_state::f2;
    }

    __host__ __device__ static constexpr mask_type mask_for_frontier_state(frontier_state state) {
        return state == frontier_state::f0 ? kFrontierF0 :
               state == frontier_state::f1 ? kFrontierF1 :
                                             kFrontierF2;
    }

    __host__ __device__ static constexpr mask_type steady_state_mask_for_bank_mask(bank_mask_type bank_mask) {
        return mask_for_frontier_state(frontier_state_for_bank_mask(bank_mask));
    }

    __host__ __device__ static constexpr mask_type steady_state_mask(int residency_state) {
        return residency_state <= 0 ? kFrontierF0 :
               residency_state == 1 ? kFrontierF1 :
                                      kFrontierF2;
    }

    __host__ __device__ static constexpr mask_type drain_mask(int drain_stage) {
        return drain_stage <= 0 ? kFrontierF2 :
               drain_stage == 1 ? kFrontierF1 :
                                  kFrontierF0;
    }

    __host__ __device__ static constexpr bool cell_active(mask_type mask, int m, int n) {
        return (mask & (mask_type{1} << (m * kAccCols + n))) != 0;
    }
};

static_assert(bf16_muxi_frontier_contract::kAccRows == 4);
static_assert(bf16_muxi_frontier_contract::kAccCols == 4);
static_assert(bf16_muxi_frontier_contract::frontier_state_for_bank_mask(0x3u) ==
              bf16_muxi_frontier_contract::frontier_state::f0);
static_assert(bf16_muxi_frontier_contract::frontier_state_for_bank_mask(0x7u) ==
              bf16_muxi_frontier_contract::frontier_state::f1);
static_assert(bf16_muxi_frontier_contract::frontier_state_for_bank_mask(0xfu) ==
              bf16_muxi_frontier_contract::frontier_state::f2);
static_assert(bf16_muxi_frontier_contract::cell_active(bf16_muxi_frontier_contract::kFrontierF0, 0, 0));
static_assert(!bf16_muxi_frontier_contract::cell_active(bf16_muxi_frontier_contract::kFrontierF0, 3, 3));
static_assert(bf16_muxi_frontier_contract::cell_active(bf16_muxi_frontier_contract::kFrontierF2, 3, 3));

} // namespace kittens::arch::c500::gemm::contracts
