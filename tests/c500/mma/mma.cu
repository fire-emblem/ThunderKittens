#include "testing_flags.cuh"

#ifdef TEST_C500_MMA

#include "testing_commons.cuh"

namespace c500::mma {
namespace atom_bf16 {
void tests(test_data &results);
}
namespace gemm_bf16 {
void tests(test_data &results);
}

void tests(test_data &results) {
    std::cout << " -------------------- Starting ops/c500/mma tests! --------------------\n" << std::endl;
#ifdef TEST_C500_MMA_ATOM_BF16
    atom_bf16::tests(results);
#else
    std::cout << "INFO: Skipping ops/c500/mma/atom_bf16 tests!\n" << std::endl;
#endif
#ifdef TEST_C500_MMA_GEMM_BF16
    gemm_bf16::tests(results);
#else
    std::cout << "INFO: Skipping ops/c500/mma/gemm_bf16 tests!\n" << std::endl;
#endif
    std::cout << std::endl;
}

} // namespace c500::mma

#endif
