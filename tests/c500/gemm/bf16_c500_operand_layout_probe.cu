#include "testing_flags.cuh"

#ifdef TEST_C500_GEMM_OPERAND_LAYOUT_PROBE

#include <vector>

#include "testing_commons.cuh"

#include "arch/c500/gemm/bf16_operand_stage.cuh"

namespace c500::mma::operand_layout_probe {

namespace {

using cta_ring = kittens::arch::c500::gemm::bf16_operand_cta_stage_ring_1;

bool run_probe(test_data &results) {
    test_info info{"c500_gemm_bf16_operand_layout_contract", test_result::FAILED};

    const int stage_bytes = cta_ring::kStageBytes;
    std::vector<int> seen(stage_bytes / cta_ring::kVecBytes, 0);

    bool good = true;

    for (int row_group = 0; row_group < cta_ring::kAGroupCount && good; ++row_group) {
        for (int m = 0; m < cta_ring::kAtomsMN && good; ++m) {
            for (int kg = 0; kg < cta_ring::kKGroups && good; ++kg) {
                for (int lane = 0; lane < cta_ring::kWaveSize && good; ++lane) {
                    const int off = kittens::arch::c500::gemm::operand_cta_a_offset<1>(0, row_group, m, kg, lane);
                    if ((off % cta_ring::kVecBytes) != 0 || off < 0 || off >= stage_bytes) {
                        std::cout << "bad A offset row_group=" << row_group
                                  << " m=" << m
                                  << " kg=" << kg
                                  << " lane=" << lane
                                  << " off=" << off
                                  << std::endl;
                        good = false;
                        break;
                    }
                    seen[off / cta_ring::kVecBytes] += 1;
                }
            }
        }
    }

    for (int col_group = 0; col_group < cta_ring::kBGroupCount && good; ++col_group) {
        for (int n = 0; n < cta_ring::kAtomsMN && good; ++n) {
            for (int kg = 0; kg < cta_ring::kKGroups && good; ++kg) {
                for (int lane = 0; lane < cta_ring::kWaveSize && good; ++lane) {
                    const int off = kittens::arch::c500::gemm::operand_cta_b_offset<1>(0, col_group, n, kg, lane);
                    if ((off % cta_ring::kVecBytes) != 0 || off < 0 || off >= stage_bytes) {
                        std::cout << "bad B offset col_group=" << col_group
                                  << " n=" << n
                                  << " kg=" << kg
                                  << " lane=" << lane
                                  << " off=" << off
                                  << std::endl;
                        good = false;
                        break;
                    }
                    seen[off / cta_ring::kVecBytes] += 1;
                }
            }
        }
    }

    for (size_t i = 0; i < seen.size() && good; ++i) {
        if (seen[i] != 1) {
            std::cout << "coverage mismatch vec_index=" << i << " hits=" << seen[i] << std::endl;
            good = false;
            break;
        }
    }

    if (good) {
        good = cta_ring::kGroupBytes == 0x4000 &&
               cta_ring::kBStageOffset == 0x8000 &&
               cta_ring::kStageBytes == 0x10000;
    }

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
    std::cout << " ----- Starting ops/c500/mma/operand_layout_probe tests! -----\n" << std::endl;
    run_probe(results);
}

} // namespace c500::mma::operand_layout_probe

#endif
