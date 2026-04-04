#include "testing_flags.cuh"

#ifdef TEST_C500_MEMORY_SHARED_TO_NATIVE_B

#include "testing_commons.cuh"

namespace c500::memory::shared_to_native_b {

namespace {

void append_placeholder(test_data &results, const std::string &label) {
    results.push_back({label, test_result::INVALID});
}

} // namespace

namespace contract {

using atom = kittens::arch::c500::mma_bf16_16x16x16_fp32;
using layout_traits = kittens::arch::c500::fragment_layout_traits<atom>;
using shared_b_tile = kittens::st<typename atom::b_scalar, atom::K, atom::N>;
using native_b_fragment = kittens::arch::c500::fragment_b<atom>;

static_assert(atom::wave_size == kittens::WAVE_THREADS,
              "C500 shared-to-native B probes assume wave64 execution.");
static_assert(std::is_same_v<typename atom::b_scalar, kittens::bf16>,
              "The first shared-to-native B probe is bf16-specific.");
static_assert(std::is_same_v<decltype(layout_traits::lane_group(0)), int>,
              "The shared-to-native B probe expects the lane-group surface to stay queryable.");
static_assert(requires(native_b_fragment &dst, const shared_b_tile &src) {
                  kittens::arch::c500::load_b<atom>(dst, src, 0, 0);
              },
              "The shared-to-native B probe keeps the native copy entrypoint callable.");

} // namespace contract

void tests(test_data &results) {
    std::cout << " ----- Starting ops/c500/memory/shared_to_native_b tests! -----\n" << std::endl;
    append_placeholder(results, "c500_shared_to_native_b_contract_smoke_[pending_backend]");
    append_placeholder(results, "c500_shared_to_native_b_fragment_layout_probe_[pending_backend]");
    std::cout << "INFO: C500 shared-to-native B coverage is a probe surface until native fragment export exists.\n" << std::endl;
}

} // namespace c500::memory::shared_to_native_b

#endif
