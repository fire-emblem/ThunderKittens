#include "testing_flags.cuh"

#ifdef TEST_C500_MEMORY

#include "testing_commons.cuh"

namespace c500::memory::shared_to_native_a {

namespace {

void append_placeholder(test_data &results, const std::string &label) {
    results.push_back({label, test_result::INVALID});
}

} // namespace

#ifdef TEST_C500_MEMORY_SHARED_TO_NATIVE_A
namespace contract {

using atom = kittens::arch::c500::mma_bf16_16x16x16_fp32;
using layout_traits = kittens::arch::c500::fragment_layout_traits<atom>;
using shared_a_tile = kittens::st<typename atom::a_scalar, atom::M, atom::K>;
using native_a_fragment = kittens::arch::c500::fragment_a<atom>;

static_assert(atom::wave_size == kittens::WAVE_THREADS,
              "C500 shared-to-native A probes assume wave64 execution.");
static_assert(std::is_same_v<typename atom::a_scalar, kittens::bf16>,
              "The first shared-to-native A probe is bf16-specific.");
static_assert(std::is_same_v<decltype(layout_traits::lane_row(0)), int>,
              "The shared-to-native A probe expects the lane layout surface to stay queryable.");
static_assert(requires(native_a_fragment &dst, const shared_a_tile &src) {
                  kittens::arch::c500::load_a<atom>(dst, src, 0, 0);
              },
              "The shared-to-native A probe keeps the native copy entrypoint callable.");

} // namespace contract
#endif

void tests(test_data &results) {
    std::cout << " ----- Starting ops/c500/memory/shared_to_native_a tests! -----\n" << std::endl;
#ifdef TEST_C500_MEMORY_SHARED_TO_NATIVE_A
    append_placeholder(results, "c500_shared_to_native_a_contract_smoke_[pending_backend]");
    append_placeholder(results, "c500_shared_to_native_a_fragment_layout_probe_[pending_backend]");
    std::cout << "INFO: C500 shared-to-native A coverage is a probe surface until native fragment export exists.\n" << std::endl;
#else
    std::cout << "INFO: Skipping ops/c500/memory/shared_to_native_a tests!\n" << std::endl;
#endif
}

} // namespace c500::memory::shared_to_native_a

#endif
