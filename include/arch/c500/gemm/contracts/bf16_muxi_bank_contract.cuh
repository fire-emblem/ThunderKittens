#pragma once

#include <stdint.h>

namespace kittens::arch::c500::gemm::contracts {

struct bf16_muxi_bank_contract {
    using bank_mask_type = uint32_t;

    static constexpr int kResidentBanks = 4;
    static constexpr int kBankKGroups = 4;
    static constexpr int kBankVectorsPerOperand = 4;
    static constexpr int kInitialValidBanks = 2;
    static constexpr int kInvalidStageSlot = -1;
    static constexpr bank_mask_type kInitialResidencyMask = 0x3u;

    __host__ __device__ static constexpr int initial_bank_source(int bank_slot) {
        return bank_slot < kInitialValidBanks ? bank_slot : kInvalidStageSlot;
    }

    __host__ __device__ static constexpr bool initial_bank_valid(int bank_slot) {
        return bank_slot >= 0 && bank_slot < kInitialValidBanks;
    }

    __host__ __device__ static constexpr bank_mask_type bank_bit(int bank_slot) {
        return bank_slot >= 0 && bank_slot < kResidentBanks ? (bank_mask_type{1} << bank_slot) : bank_mask_type{0};
    }

    __host__ __device__ static constexpr bool bank_valid(bank_mask_type residency_mask, int bank_slot) {
        return (residency_mask & bank_bit(bank_slot)) != 0;
    }

    __host__ __device__ static constexpr bank_mask_type bank_mask_after_reload(bank_mask_type residency_mask,
                                                                                int bank_slot) {
        return residency_mask | bank_bit(bank_slot);
    }

    __host__ __device__ static constexpr int reload_target_slot(int step_id) {
        return step_id < 0 ? 0 : (step_id & (kResidentBanks - 1));
    }

    __host__ __device__ static constexpr int reload_source_stage(int step_id) {
        return reload_target_slot(step_id);
    }
};

static_assert(bf16_muxi_bank_contract::kResidentBanks == 4);
static_assert(bf16_muxi_bank_contract::kInitialValidBanks == 2);
static_assert(bf16_muxi_bank_contract::kInitialResidencyMask == 0x3u);
static_assert(bf16_muxi_bank_contract::initial_bank_source(0) == 0);
static_assert(bf16_muxi_bank_contract::initial_bank_source(1) == 1);
static_assert(bf16_muxi_bank_contract::initial_bank_source(2) == bf16_muxi_bank_contract::kInvalidStageSlot);
static_assert(bf16_muxi_bank_contract::bank_valid(0x3u, 0));
static_assert(!bf16_muxi_bank_contract::bank_valid(0x3u, 2));
static_assert(bf16_muxi_bank_contract::bank_mask_after_reload(0x3u, 2) == 0x7u);

} // namespace kittens::arch::c500::gemm::contracts
