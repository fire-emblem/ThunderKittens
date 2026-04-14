#include "testing_flags.cuh"

#ifdef TEST_C500_MEMORY

#include "testing_commons.cuh"

namespace c500::memory {
namespace global_to_shared_async_tile_probe {
void tests(test_data &results);
}
namespace raw_gemm_async {
void tests(test_data &results);
}
namespace global_to_shared_async_bsm {
void tests(test_data &results);
}
namespace gemm_async_stage_compare {
void tests(test_data &results);
}
namespace shared_to_native_a {
void tests(test_data &results);
}
namespace shared_to_native_b {
void tests(test_data &results);
}

void tests(test_data &results) {
    std::cout << " -------------------- Starting ops/c500/memory tests! --------------------\n" << std::endl;
#ifdef TEST_C500_MEMORY_SWIZZLED_ASYNC_DIAGNOSTIC
    global_to_shared_async_tile_probe::tests(results);
#endif
#ifdef TEST_C500_MEMORY_SWIZZLED_ASYNC_DIAGNOSTIC
    gemm_async_stage_compare::tests(results);
#endif
#ifdef TEST_C500_MEMORY_RAW_GEMM_ASYNC
    raw_gemm_async::tests(results);
#else
    std::cout << "INFO: Skipping ops/c500/memory/raw_gemm_async tests!\n" << std::endl;
#endif
#ifdef TEST_C500_MEMORY_ASYNC_BSM
    global_to_shared_async_bsm::tests(results);
#else
    std::cout << "INFO: Skipping ops/c500/memory/global_to_shared_async_bsm tests!\n" << std::endl;
#endif
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
