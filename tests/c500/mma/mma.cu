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
namespace stage_probe {
void tests(test_data &results);
}
namespace stage_async_probe {
void tests(test_data &results);
}
namespace raw_vector_gemm_probe {
void tests(test_data &results);
}
namespace operand_stage_probe {
void tests(test_data &results);
}
namespace native_coord_probe {
void tests(test_data &results);
}
namespace operand_layout_probe {
void tests(test_data &results);
}
namespace operand_bridge_probe {
void tests(test_data &results);
}
namespace operand_direct_async_probe {
void tests(test_data &results);
}
namespace operand_b_layouta_direct_async_probe {
void tests(test_data &results);
}
namespace operand_layouta_atom_probe {
void tests(test_data &results);
}
namespace gemm_smoke {
void tests(test_data &results);
}
namespace balanced_family_smoke {
void tests(test_data &results);
}
namespace gemm_layouta_native_smoke {
void tests(test_data &results);
}
namespace layouta_native_stage_probe {
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
#ifdef TEST_C500_GEMM_STAGE_PROBE
    stage_probe::tests(results);
#else
    std::cout << "INFO: Skipping ops/c500/mma/stage_probe tests!\n" << std::endl;
#endif
#ifdef TEST_C500_GEMM_STAGE_ASYNC_PROBE
    stage_async_probe::tests(results);
#else
    std::cout << "INFO: Skipping ops/c500/mma/stage_async_probe tests!\n" << std::endl;
#endif
#ifdef TEST_C500_GEMM_RAW_VECTOR_GEMM_PROBE
    raw_vector_gemm_probe::tests(results);
#else
    std::cout << "INFO: Skipping ops/c500/mma/raw_vector_gemm_probe tests!\n" << std::endl;
#endif
#ifdef TEST_C500_GEMM_OPERAND_STAGE_PROBE
    operand_stage_probe::tests(results);
#else
    std::cout << "INFO: Skipping ops/c500/mma/operand_stage_probe tests!\n" << std::endl;
#endif
#ifdef TEST_C500_GEMM_NATIVE_COORD_PROBE
    native_coord_probe::tests(results);
#else
    std::cout << "INFO: Skipping ops/c500/mma/native_coord_probe tests!\n" << std::endl;
#endif
#ifdef TEST_C500_GEMM_OPERAND_LAYOUT_PROBE
    operand_layout_probe::tests(results);
#else
    std::cout << "INFO: Skipping ops/c500/mma/operand_layout_probe tests!\n" << std::endl;
#endif
#ifdef TEST_C500_GEMM_OPERAND_BRIDGE_PROBE
    operand_bridge_probe::tests(results);
#else
    std::cout << "INFO: Skipping ops/c500/mma/operand_bridge_probe tests!\n" << std::endl;
#endif
#ifdef TEST_C500_GEMM_OPERAND_DIRECT_ASYNC_PROBE
    operand_direct_async_probe::tests(results);
#else
    std::cout << "INFO: Skipping ops/c500/mma/operand_direct_async_probe tests!\n" << std::endl;
#endif
#ifdef TEST_C500_GEMM_OPERAND_B_LAYOUTA_DIRECT_ASYNC_PROBE
    operand_b_layouta_direct_async_probe::tests(results);
#else
    std::cout << "INFO: Skipping ops/c500/mma/operand_b_layouta_direct_async_probe tests!\n" << std::endl;
#endif
#ifdef TEST_C500_GEMM_OPERAND_LAYOUTA_ATOM_PROBE
    operand_layouta_atom_probe::tests(results);
#else
    std::cout << "INFO: Skipping ops/c500/mma/operand_layouta_atom_probe tests!\n" << std::endl;
#endif
#ifdef TEST_C500_GEMM_SMOKE
    gemm_smoke::tests(results);
#else
    std::cout << "INFO: Skipping ops/c500/mma/gemm_smoke tests!\n" << std::endl;
#endif
#ifdef TEST_C500_GEMM_BALANCED_FAMILY_SMOKE
    balanced_family_smoke::tests(results);
#else
    std::cout << "INFO: Skipping ops/c500/mma/balanced_family_smoke tests!\n" << std::endl;
#endif
#ifdef TEST_C500_GEMM_LAYOUTA_NATIVE_SMOKE
    gemm_layouta_native_smoke::tests(results);
#else
    std::cout << "INFO: Skipping ops/c500/mma/gemm_layouta_native_smoke tests!\n" << std::endl;
#endif
#ifdef TEST_C500_GEMM_LAYOUTA_NATIVE_STAGE_PROBE
    layouta_native_stage_probe::tests(results);
#else
    std::cout << "INFO: Skipping ops/c500/mma/layouta_native_stage_probe tests!\n" << std::endl;
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
