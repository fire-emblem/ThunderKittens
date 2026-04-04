#include "testing_flags.cuh"

#ifdef TEST_C500_MEMORY

#include "testing_commons.cuh"

namespace c500::memory {
namespace shared_to_native_a {
void tests(test_data &results);
}
namespace shared_to_native_b {
void tests(test_data &results);
}

void tests(test_data &results) {
    std::cout << " -------------------- Starting ops/c500/memory tests! --------------------\n" << std::endl;
#ifdef TEST_C500_MEMORY_SHARED_TO_NATIVE_A
    shared_to_native_a::tests(results);
#else
    std::cout << "INFO: Skipping ops/c500/memory/shared_to_native_a tests!\n" << std::endl;
#endif
#ifdef TEST_C500_MEMORY_SHARED_TO_NATIVE_B
    shared_to_native_b::tests(results);
#else
    std::cout << "INFO: Skipping ops/c500/memory/shared_to_native_b tests!\n" << std::endl;
#endif
    std::cout << std::endl;
}

} // namespace c500::memory

#endif
