#pragma once

#include "../contracts/bf16_muxi_bank_contract.cuh"
#include "../contracts/bf16_muxi_frontier_contract.cuh"

namespace kittens::arch::c500::gemm::schedulers {

struct bf16_muxi_layouta_stage4_scheduler {
    using bank_contract = contracts::bf16_muxi_bank_contract;
    using frontier_contract = contracts::bf16_muxi_frontier_contract;

    enum class phase : int {
        prologue = 0,
        steady_state = 1,
        drain = 2,
    };

    struct state {
        int next_global_tile = 0;
        int resident_stage_slot[bank_contract::kResidentBanks] = {
            bank_contract::initial_bank_source(0),
            bank_contract::initial_bank_source(1),
            bank_contract::initial_bank_source(2),
            bank_contract::initial_bank_source(3),
        };
        typename bank_contract::bank_mask_type residency_mask = bank_contract::kInitialResidencyMask;
        int outstanding_transactions = 0;
        int drain_stage = -1;
        phase current_phase = phase::prologue;
    };

    __host__ __device__ static constexpr int steady_state_frontier_index(const state &scheduler_state) {
        return scheduler_state.residency_mask == 0x3u ? 0 :
               scheduler_state.residency_mask == 0x7u ? 1 :
                                                        2;
    }

    __host__ __device__ static constexpr typename frontier_contract::frontier_state
    current_frontier_state(const state &scheduler_state) {
        return frontier_contract::frontier_state_for_bank_mask(scheduler_state.residency_mask);
    }

    __host__ __device__ static constexpr bool resident_bank_valid(const state &scheduler_state, int bank_slot) {
        return bank_contract::bank_valid(scheduler_state.residency_mask, bank_slot);
    }

    __host__ __device__ static constexpr int reload_bank_slot(const state &scheduler_state) {
        return bank_contract::reload_target_slot(scheduler_state.next_global_tile);
    }

    __host__ __device__ static constexpr int reload_stage_slot(const state &scheduler_state) {
        return bank_contract::reload_source_stage(scheduler_state.next_global_tile);
    }

    __host__ __device__ static constexpr typename frontier_contract::mask_type
    active_frontier(const state &scheduler_state) {
        return scheduler_state.current_phase == phase::drain
                   ? frontier_contract::drain_mask(scheduler_state.drain_stage)
                   : frontier_contract::steady_state_mask_for_bank_mask(scheduler_state.residency_mask);
    }

    __host__ __device__ static constexpr void mark_bank_reloaded(state &scheduler_state, int bank_slot) {
        scheduler_state.residency_mask =
            bank_contract::bank_mask_after_reload(scheduler_state.residency_mask, bank_slot);
    }

    template<typename Context>
    __device__ static inline void prologue(Context &, state &scheduler_state) {
        scheduler_state.current_phase = phase::steady_state;
    }

    template<typename Context>
    __device__ static inline bool step(Context &, state &scheduler_state) {
        if (scheduler_state.current_phase == phase::drain) {
            return false;
        }
        scheduler_state.current_phase = phase::steady_state;
        const int reload_bank = reload_bank_slot(scheduler_state);
        mark_bank_reloaded(scheduler_state, reload_bank);
        ++scheduler_state.next_global_tile;
        return true;
    }

    template<typename Context>
    __device__ static inline void drain(Context &, state &scheduler_state) {
        scheduler_state.current_phase = phase::drain;
        scheduler_state.drain_stage = steady_state_frontier_index(scheduler_state);
    }
};

static_assert(bf16_muxi_layouta_stage4_scheduler::bank_contract::kResidentBanks == 4);
static_assert(bf16_muxi_layouta_stage4_scheduler::frontier_contract::kAccRows == 4);

} // namespace kittens::arch::c500::gemm::schedulers
