#include "testing_flags.cuh"

#ifdef TEST_C500_GEMM_LAYOUT_PROBE

#include "arch/c500/layouts/accumulator_export.cuh"
#include "arch/c500/layouts/lds_offsets.cuh"
#include "arch/c500/layouts/operand_layouts.cuh"
#include "testing_commons.cuh"

namespace c500::mma::layout_probe {

namespace {

using layout = kittens::arch::c500::gemm::bf16_128x128x128_stage_layout;
using export_map = kittens::arch::c500::gemm::accumulator_tile_map;

static_assert(layout::kStages == 4);
static_assert(layout::kTileM == 128);
static_assert(layout::kTileN == 128);
static_assert(layout::kTileK == 128);
static_assert(layout::kThreads == 256);
static_assert(layout::kWaveCount == 4);
static_assert(layout::kStageBytes == 0x4000);
static_assert(layout::kAStageOffset == 0x0000);
static_assert(layout::kBStageOffset == 0x2000);
static_assert(layout::stage_offset(3) == 0xc000);
static_assert(layout::a_stage_offset(2) == 0x8000);
static_assert(layout::b_stage_offset(1) == 0x6000);
static_assert(export_map::kWaveM == 2);
static_assert(export_map::kWaveN == 2);

bool run_stage_layout_contract(test_data &results) {
    test_info info{"c500_gemm_bf16_layout_stage_contract", test_result::FAILED};

    const bool good = layout::kStages == 4 &&
                      layout::kTileM == 128 &&
                      layout::kTileN == 128 &&
                      layout::kTileK == 128 &&
                      layout::kThreads == 256 &&
                      layout::kWaveCount == 4 &&
                      layout::stage_offset(0) == 0x0000 &&
                      layout::stage_offset(3) == 0xc000 &&
                      layout::a_stage_offset(2) == 0x8000 &&
                      layout::b_stage_offset(1) == 0x6000;

    std::cout << "test `" << info.label << "`";
    if (good) {
        std::cout << " -- PASSED" << std::endl;
        info.result = test_result::PASSED;
    } else {
        std::cout << " ----- ALERT! FAILED test `" << info.label << "` -----" << std::endl;
    }

    results.push_back(info);
    return good;
}

bool run_lds_offset_contract(test_data &results) {
    test_info info{"c500_gemm_bf16_layout_lds_offsets", test_result::FAILED};

    const bool good =
        kittens::arch::c500::gemm::lds_offset_a(0, 0) == 0x0000 &&
        kittens::arch::c500::gemm::lds_offset_a(63, 0) == 0x03f0 &&
        kittens::arch::c500::gemm::lds_offset_a(64, 0) == 0x0000 &&
        kittens::arch::c500::gemm::lds_offset_a(128, 0) == 0x1000 &&
        kittens::arch::c500::gemm::lds_offset_a(191, 3) == 0x1ff0 &&
        kittens::arch::c500::gemm::lds_offset_b(0, 0) == 0x2000 &&
        kittens::arch::c500::gemm::lds_offset_b(63, 0) == 0x23f0 &&
        kittens::arch::c500::gemm::lds_offset_b(64, 0) == 0x3000 &&
        kittens::arch::c500::gemm::lds_offset_b(128, 0) == 0x2000 &&
        kittens::arch::c500::gemm::lds_offset_b(255, 3) == 0x3ff0;

    std::cout << "test `" << info.label << "`";
    if (good) {
        std::cout << " -- PASSED" << std::endl;
        info.result = test_result::PASSED;
    } else {
        std::cout << " ----- ALERT! FAILED test `" << info.label << "` -----" << std::endl;
    }

    results.push_back(info);
    return good;
}

} // namespace

void tests(test_data &results) {
    std::cout << " ----- Starting ops/c500/mma/layout_probe tests! -----\n" << std::endl;
    run_stage_layout_contract(results);
    run_lds_offset_contract(results);
    std::cout << "INFO: C500 layout coverage now freezes the first 128x128x128 stage layout and native LDS offset contracts.\n" << std::endl;
}

} // namespace c500::mma::layout_probe

#endif
