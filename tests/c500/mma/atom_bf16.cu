#include "testing_flags.cuh"

#ifdef TEST_C500_MMA_ATOM_BF16

#include "testing_commons.cuh"

namespace c500::mma::atom_bf16 {

namespace {

void append_placeholder(test_data &results, const std::string &label) {
    results.push_back({label, test_result::INVALID});
}

} // namespace

namespace atom_bf16_contract {

using atom = kittens::arch::c500::mma_bf16_16x16x16_fp32;
using layout_traits = kittens::arch::c500::fragment_layout_traits<atom>;

static_assert(atom::M == 16 && atom::N == 16 && atom::K == 16,
              "The first-wave C500 atom scaffold is fixed to 16x16x16.");
static_assert(atom::wave_size == kittens::WAVE_THREADS,
              "C500 atom scaffolds assume native wave-sized execution.");
static_assert(std::is_same_v<typename atom::a_scalar, kittens::bf16>,
              "The first C500 atom scaffold covers bf16 inputs.");
static_assert(std::is_same_v<typename atom::b_scalar, kittens::bf16>,
              "The first C500 atom scaffold covers bf16 inputs.");
static_assert(std::is_same_v<typename atom::c_scalar, float>,
              "The first C500 atom scaffold covers fp32 accumulation.");
static_assert(std::is_same_v<decltype(layout_traits::lane_row(0)), int>,
              "The atom probe expects an integer lane->row mapping surface.");
static_assert(std::is_same_v<decltype(layout_traits::lane_group(0)), int>,
              "The atom probe expects an integer lane->group mapping surface.");
static_assert(requires(
                  kittens::arch::c500::fragment_c<atom> &dst,
                  const kittens::arch::c500::fragment_a<atom> &a,
                  const kittens::arch::c500::fragment_b<atom> &b,
                  const kittens::arch::c500::fragment_c<atom> &src) {
                  kittens::arch::c500::mma<atom>(dst, a, b, src);
              },
              "The atom probe keeps the native mma entrypoint callable for the frozen bf16 contract.");

} // namespace atom_bf16_contract

void tests(test_data &results) {
    std::cout << " ----- Starting ops/c500/mma/atom_bf16 tests! -----\n" << std::endl;
    append_placeholder(results, "c500_mma_atom_bf16_contract_smoke_[pending_backend]");
    append_placeholder(results, "c500_mma_atom_bf16_lane_layout_probe_[pending_backend]");
    std::cout << "INFO: C500 bf16 atom coverage is scaffold-only until native load/mma/store atoms land.\n" << std::endl;
}

} // namespace c500::mma::atom_bf16

#endif
