#pragma once

namespace kittens::arch::c500::gemm {

struct bf16_native_stage_layout {
    static constexpr int kWaveSize = 64;
    static constexpr int kMmaThreadMN = 16;
    static constexpr int kMmaThreadK = 4;
    static constexpr int kLdsNumPerThread = 8;
    static constexpr int kKGroups = 4;
    static constexpr int kMAtoms = 4;
    static constexpr int kNAtoms = 4;
    static constexpr int kLdsRowStride = 32;
    static constexpr int kLdsColStride = 32;

    __host__ __device__ static constexpr int lane_mn(int lane) { return lane & 0x0f; }
    __host__ __device__ static constexpr int lane_group(int lane) { return (lane & 0x3f) >> 4; }
    __host__ __device__ static constexpr int wave_id(int lane) { return lane / kWaveSize; }

    __host__ __device__ static constexpr int lds_k(int lane, int k_group) {
        return ((kMmaThreadK * k_group + lane_group(lane)) ^ lane_mn(lane)) * kLdsNumPerThread;
    }

    __host__ __device__ static constexpr int lds_m_base(int wave, int lane) {
        return lane_mn(lane) + (wave / 2) * kMmaThreadMN;
    }

    __host__ __device__ static constexpr int lds_n_base(int wave, int lane) {
        return lane_mn(lane) + (wave % 2) * kMmaThreadMN;
    }

    __host__ __device__ static constexpr int lds_m(int wave, int m_atom, int lane) {
        return lds_m_base(wave, lane) + m_atom * kLdsRowStride;
    }

    __host__ __device__ static constexpr int lds_n(int wave, int n_atom, int lane) {
        return lds_n_base(wave, lane) + n_atom * kLdsColStride;
    }
};

} // namespace kittens::arch::c500::gemm
