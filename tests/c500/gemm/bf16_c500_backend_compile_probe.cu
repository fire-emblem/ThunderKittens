#include "testing_flags.cuh"

#ifdef TEST_C500_GEMM_BACKEND_COMPILE_PROBE

#include <bit>
#include <concepts>
#include <type_traits>

#include "kittens.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "arch/c500/async.cuh"
#include "arch/c500/traits.cuh"
#include "arch/c500/fragments.cuh"
#include "arch/c500/mma.cuh"
#include "arch/c500/primitives/copy.cuh"
#include "arch/c500/primitives/mma.cuh"
#include "arch/c500/primitives/pipeline.cuh"
#include "arch/c500/primitives/layout.cuh"
#include "arch/c500/gemm/contracts/bf16_muxi_bank_contract.cuh"
#include "arch/c500/gemm/contracts/bf16_muxi_frontier_contract.cuh"
#include "arch/c500/gemm/families/bf16_muxi_128x128x128_stage4.cuh"
#include "arch/c500/gemm/schedulers/bf16_muxi_layouta_stage4_scheduler.cuh"
#include "testing_commons.cuh"

namespace c500::mma::backend_compile_probe {

namespace {

using atom = kittens::arch::c500::bf16_mma_atom;
using fragment_a = kittens::arch::c500::fragment_a<atom>;
using fragment_b = kittens::arch::c500::fragment_b<atom>;
using fragment_c = kittens::arch::c500::fragment_c<atom>;
using primitive_atom = kittens::arch::c500::primitives::bf16_mma_16x16x16_fp32_atom;
using primitive_fragment_a = kittens::arch::c500::primitives::bf16_fragment_a;
using primitive_fragment_b = kittens::arch::c500::primitives::bf16_fragment_b;
using primitive_fragment_c = kittens::arch::c500::primitives::bf16_fragment_c;
using muxi_bank_contract = kittens::arch::c500::gemm::contracts::bf16_muxi_bank_contract;
using muxi_frontier_contract = kittens::arch::c500::gemm::contracts::bf16_muxi_frontier_contract;
using muxi_scheduler = kittens::arch::c500::gemm::schedulers::bf16_muxi_layouta_stage4_scheduler;
using muxi_family = kittens::arch::c500::gemm::families::bf16_muxi_128x128x128_stage4;

static_assert(kittens::arch::c500::wave_traits::kWaveSize == 64);
static_assert(primitive_atom::M == 16);
static_assert(kittens::arch::c500::primitives::balanced_wave_traits::kWaveSize == 64);
static_assert(muxi_bank_contract::kResidentBanks == 4);
static_assert(muxi_bank_contract::kInitialValidBanks == 2);
static_assert(muxi_bank_contract::kInitialResidencyMask == 0x3u);
static_assert(muxi_bank_contract::bank_valid(0x3u, 0));
static_assert(muxi_bank_contract::bank_valid(0x3u, 1));
static_assert(!muxi_bank_contract::bank_valid(0x3u, 2));
static_assert(muxi_bank_contract::bank_mask_after_reload(0x3u, 2) == 0x7u);
static_assert(muxi_bank_contract::bank_mask_after_reload(0x7u, 3) == 0xfu);
static_assert(muxi_frontier_contract::kAccRows == 4 && muxi_frontier_contract::kAccCols == 4);
static_assert(muxi_frontier_contract::frontier_state_for_bank_mask(0x3u) ==
              muxi_frontier_contract::frontier_state::f0);
static_assert(muxi_frontier_contract::frontier_state_for_bank_mask(0x7u) ==
              muxi_frontier_contract::frontier_state::f1);
static_assert(muxi_frontier_contract::frontier_state_for_bank_mask(0xfu) ==
              muxi_frontier_contract::frontier_state::f2);
static_assert(muxi_frontier_contract::steady_state_mask_for_bank_mask(0x3u) ==
              muxi_frontier_contract::kFrontierF0);
static_assert(muxi_scheduler::bank_contract::kResidentBanks == 4);
static_assert(muxi_scheduler::frontier_contract::frontier_state_for_bank_mask(0x3u) ==
              muxi_scheduler::frontier_contract::frontier_state::f0);
static_assert(muxi_scheduler::current_frontier_state(muxi_scheduler::state{}) ==
              muxi_scheduler::frontier_contract::frontier_state::f0);
static_assert(muxi_scheduler::resident_bank_valid(muxi_scheduler::state{}, 0));
static_assert(!muxi_scheduler::resident_bank_valid(muxi_scheduler::state{}, 2));
static_assert(muxi_scheduler::reload_bank_slot(muxi_scheduler::state{}) == 0);
static_assert(muxi_scheduler::reload_stage_slot(muxi_scheduler::state{}) == 0);
static_assert(std::is_same_v<typename muxi_family::fallback_family,
                             kittens::arch::c500::gemm::families::bf16_balanced_128x128x128_stage4>);
static_assert(std::is_same_v<typename muxi_family::bank_contract, muxi_bank_contract>);
static_assert(std::is_same_v<typename muxi_family::frontier_contract, muxi_frontier_contract>);
static_assert(std::is_same_v<typename muxi_family::scheduler, muxi_scheduler>);
static_assert(muxi_family::kAtomsM == 4);
static_assert(muxi_family::kAtomsN == 4);
static_assert(atom::M == 16 && atom::N == 16 && atom::K == 16);
static_assert(std::is_same_v<typename atom::a_scalar, kittens::bf16>);
static_assert(kittens::arch::c500::async_token<2>::transactions == 2);
static_assert(kittens::arch::c500::primitives::async_token<2>::transactions == 2);
static_assert(requires(const fragment_a &a, const fragment_b &b, const fragment_c &c) {
    { kittens::arch::c500::mma(atom{}, a, b, c) } -> std::same_as<fragment_c>;
});
static_assert(requires(const primitive_fragment_a &a,
                       const primitive_fragment_b &b,
                       const primitive_fragment_c &c) {
    { kittens::arch::c500::primitives::mma(a, b, c) } -> std::same_as<primitive_fragment_c>;
});

__global__ void bf16_c500_backend_compile_probe_kernel() {
#ifdef KITTENS_C500
    primitive_fragment_a a{};
    primitive_fragment_b b{};
    primitive_fragment_c c{};
    const auto copy_token =
        kittens::arch::c500::primitives::async_copy_128b(static_cast<void *>(nullptr),
                                                         static_cast<const kittens::bf16 *>(nullptr),
                                                         0,
                                                         1);
    kittens::arch::c500::primitives::wait(copy_token);
    kittens::arch::c500::primitives::wait_until<0>();
    const auto lane_group =
        kittens::arch::c500::primitives::balanced_wave_traits::lane_group(threadIdx.x);
    const auto d = kittens::arch::c500::primitives::mma(a, b, c);
    muxi_scheduler::state scheduler_state{};
    const auto frontier = muxi_scheduler::active_frontier(scheduler_state);
    (void)lane_group;
    (void)d;
    (void)frontier;
#endif
}

} // namespace

void tests(test_data &results) {
    test_info info{"c500_gemm_bf16_backend_compile_probe", test_result::PASSED};
    std::cout << "test `" << info.label << "` -- PASSED" << std::endl;
    results.push_back(info);
}

} // namespace c500::mma::backend_compile_probe

#endif
