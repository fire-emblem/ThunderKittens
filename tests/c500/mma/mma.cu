#include "testing_flags.cuh"

#ifdef TEST_C500_MMA

#include "testing_commons.cuh"

namespace c500::mma {
namespace backend_compile_probe {
void tests(test_data &results);
}
namespace layout_probe {
void tests(test_data &results);
}
namespace fragment_probe {
void tests(test_data &results);
}
namespace gemm_smoke {
void tests(test_data &results);
}
namespace atom_bf16 {
void tests(test_data &results);
}
namespace gemm_bf16 {
void tests(test_data &results);
}

void tests(test_data &results) {
    std::cout << " -------------------- Starting ops/c500/mma tests! --------------------\n" << std::endl;
#ifdef TEST_C500_GEMM_BACKEND_COMPILE_PROBE
    backend_compile_probe::tests(results);
#else
    std::cout << "INFO: Skipping ops/c500/mma/backend_compile_probe tests!\n" << std::endl;
#endif
#ifdef TEST_C500_GEMM_LAYOUT_PROBE
    layout_probe::tests(results);
#else
    std::cout << "INFO: Skipping ops/c500/mma/layout_probe tests!\n" << std::endl;
#endif
#ifdef TEST_C500_GEMM_FRAGMENT_PROBE
    fragment_probe::tests(results);
#else
    std::cout << "INFO: Skipping ops/c500/mma/fragment_probe tests!\n" << std::endl;
#endif
#ifdef TEST_C500_GEMM_SMOKE
    gemm_smoke::tests(results);
#else
    std::cout << "INFO: Skipping ops/c500/mma/gemm_smoke tests!\n" << std::endl;
#endif
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
