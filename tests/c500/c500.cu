#include "c500.cuh"

#ifdef TEST_C500

namespace c500 {
namespace mma {
void tests(test_data &results);
}
namespace memory {
void tests(test_data &results);
}

void tests(test_data &results) {
    std::cout << " -------------------- Starting ops/c500 tests! --------------------\n" << std::endl;
#ifdef TEST_C500_MMA
    mma::tests(results);
#else
    std::cout << "INFO: Skipping ops/c500/mma tests!\n" << std::endl;
#endif
#ifdef TEST_C500_MEMORY
    memory::tests(results);
#else
    std::cout << "INFO: Skipping ops/c500/memory tests!\n" << std::endl;
#endif
    std::cout << std::endl;
}

} // namespace c500

#endif
