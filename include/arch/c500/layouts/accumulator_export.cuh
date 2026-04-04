#pragma once

namespace kittens::arch::c500::gemm {

struct accumulator_tile_map {
    static constexpr int kWaveM = 2;
    static constexpr int kWaveN = 2;
    static constexpr int kWaveTileM = 64;
    static constexpr int kWaveTileN = 64;
    static constexpr int kAtomM = 16;
    static constexpr int kAtomN = 16;
    static constexpr int kAccumulatorRegisters = 4;

    __host__ __device__ static constexpr int wave_row(int wave) {
        return wave / kWaveN;
    }

    __host__ __device__ static constexpr int wave_col(int wave) {
        return wave % kWaveN;
    }

    __host__ __device__ static constexpr int wave_base_row(int wave) {
        return wave_row(wave) * kWaveTileM;
    }

    __host__ __device__ static constexpr int wave_base_col(int wave) {
        return wave_col(wave) * kWaveTileN;
    }

    __host__ __device__ static constexpr int atom_row(int lane) {
        return lane & 0x0f;
    }

    __host__ __device__ static constexpr int atom_col(int lane, int reg_index) {
        return ((lane >> 4) * kAccumulatorRegisters) + reg_index;
    }
};

} // namespace kittens::arch::c500::gemm
