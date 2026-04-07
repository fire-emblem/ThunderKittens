#include "testing_flags.cuh"

#ifdef TEST_C500_GEMM_NATIVE_COORD_PROBE

#include <array>
#include <vector>

#include "testing_commons.cuh"

#include "arch/c500/gemm/bf16_operand_stage.cuh"

namespace c500::mma::native_coord_probe {

namespace {

using coords = kittens::arch::c500::gemm::bf16_balanced_operand_coords;

constexpr int kThreads = 256;
constexpr int kWaves = 4;
constexpr int kAtomsMN = coords::kAtomsMN;
constexpr int kKGroups = coords::kKGroups;

bool run_probe(test_data &results) {
    test_info info{"c500_gemm_bf16_native_coord_contract", test_result::FAILED};

    bool good = true;

    for (int wave = 0; wave < kWaves && good; ++wave) {
        for (int m = 0; m < kAtomsMN && good; ++m) {
            std::array<int, coords::kLaneMn * (coords::kBlockTileK / coords::kVecElems)> seen{};
            const int row_base = (wave / 2) * coords::kLaneMn + m * 32;
            for (int lane = 0; lane < coords::kWaveSize && good; ++lane) {
                const int row = coords::a_row(wave, m, lane);
                if (row < row_base || row >= row_base + coords::kLaneMn) {
                    std::cout << "A row OOB wave=" << wave << " m=" << m << " lane=" << lane << " row=" << row << std::endl;
                    good = false;
                    break;
                }
                for (int kg = 0; kg < kKGroups; ++kg) {
                    const int kvec = coords::k_vec(kg, lane);
                    if (kvec < 0 || kvec >= (coords::kBlockTileK / coords::kVecElems)) {
                        std::cout << "A kvec OOB wave=" << wave << " m=" << m << " lane=" << lane << " kg=" << kg << " kvec=" << kvec << std::endl;
                        good = false;
                        break;
                    }
                    seen[(row - row_base) * (coords::kBlockTileK / coords::kVecElems) + kvec] += 1;
                }
            }
            if (!good) break;
            for (int row = 0; row < coords::kLaneMn && good; ++row) {
                for (int kvec = 0; kvec < (coords::kBlockTileK / coords::kVecElems); ++kvec) {
                    const int hits = seen[row * (coords::kBlockTileK / coords::kVecElems) + kvec];
                    if (hits != 1) {
                        std::cout << "A coverage mismatch wave=" << wave
                                  << " m=" << m
                                  << " row=" << (row_base + row)
                                  << " kvec=" << kvec
                                  << " hits=" << hits
                                  << std::endl;
                        good = false;
                        break;
                    }
                }
            }
        }
    }

    for (int wave = 0; wave < kWaves && good; ++wave) {
        for (int n = 0; n < kAtomsMN && good; ++n) {
            std::array<int, (coords::kBlockTileK / coords::kVecElems) * coords::kLaneMn> seen{};
            const int col_base = (wave % 2) * coords::kLaneMn + n * 32;
            for (int lane = 0; lane < coords::kWaveSize && good; ++lane) {
                const int col = coords::b_col(wave, n, lane);
                if (col < col_base || col >= col_base + coords::kLaneMn) {
                    std::cout << "B col OOB wave=" << wave << " n=" << n << " lane=" << lane << " col=" << col << std::endl;
                    good = false;
                    break;
                }
                for (int kg = 0; kg < kKGroups; ++kg) {
                    const int kvec = coords::k_vec(kg, lane);
                    if (kvec < 0 || kvec >= (coords::kBlockTileK / coords::kVecElems)) {
                        std::cout << "B kvec OOB wave=" << wave << " n=" << n << " lane=" << lane << " kg=" << kg << " kvec=" << kvec << std::endl;
                        good = false;
                        break;
                    }
                    seen[kvec * coords::kLaneMn + (col - col_base)] += 1;
                }
            }
            if (!good) break;
            for (int kvec = 0; kvec < (coords::kBlockTileK / coords::kVecElems) && good; ++kvec) {
                for (int col = 0; col < coords::kLaneMn; ++col) {
                    const int hits = seen[kvec * coords::kLaneMn + col];
                    if (hits != 1) {
                        std::cout << "B coverage mismatch wave=" << wave
                                  << " n=" << n
                                  << " kvec=" << kvec
                                  << " col=" << (col_base + col)
                                  << " hits=" << hits
                                  << std::endl;
                        good = false;
                        break;
                    }
                }
            }
        }
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
    std::cout << " ----- Starting ops/c500/mma/native_coord_probe tests! -----\n" << std::endl;
    run_probe(results);
}

} // namespace c500::mma::native_coord_probe

#endif
