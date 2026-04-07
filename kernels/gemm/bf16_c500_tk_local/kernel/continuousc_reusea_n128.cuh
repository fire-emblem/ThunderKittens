#pragma once

#include <maca.h>
#include <maca_bfloat16.h>
#include <maca_fp16.h>
#include <mc_runtime.h>

#include "layoutc_support.cuh"

namespace bf16_c500_tk_local::kernel {

template <typename T, typename Tc, typename Taccum, int NTile, int APerWarp,
          int SplitN, int SplitK, bool IsBetaZero, bool HasOneDimBias>
__global__ void __launch_bounds__(256) tk_local_continuousc_reusea_n(
    T *A, T *B, Tc *C, int m, int n, int k, Taccum alpha, Taccum beta,
    Tc *bias) {
    static_assert(IsBetaZero, "tk local benchmark path assumes beta == 0");
    static_assert(!HasOneDimBias, "tk local benchmark path does not support bias yet");

    constexpr int Stages = 4;
    constexpr int ElementsPerAccess = 8;
    constexpr int RowThreadsPerMma = 16;
    constexpr int ColThreadsPerMma = 4;
    constexpr int ElementsPerThreadPerMma = 4;
    constexpr int BlockDimX = 256;
    constexpr int WarpPerBlock = BlockDimX / 64;
    constexpr int NumCycleB = NTile / RowThreadsPerMma;
    constexpr int SharedNumCycleB = (NumCycleB + SplitN - 1) / SplitN;
    constexpr int SharedArriveCount =
        (SharedNumCycleB + WarpPerBlock - 1) / WarpPerBlock;

    using UINT4 = __NATIVE_VECTOR__(4, uint);
    using INT128 = __NATIVE_VECTOR__(4, int);
    using CStgType = __NATIVE_VECTOR__(sizeof(Tc), uint);

    const int rowsGroup = m / RowThreadsPerMma / APerWarp;

    const int numWarps = __builtin_mxc_readfirstlane(
        gridDim.x * blockDim.x / 64 / SplitN / SplitK);
    const int warpId = __builtin_mxc_readfirstlane(
                           (blockIdx.x * blockDim.x + threadIdx.x) / 64) %
                       numWarps;
    const int splitNId = __builtin_mxc_readfirstlane(
        (((blockIdx.x * blockDim.x + threadIdx.x) / 64) / numWarps) % SplitN);
    const int splitKId = __builtin_mxc_readfirstlane(
        (((blockIdx.x * blockDim.x + threadIdx.x) / 64) / numWarps / SplitN) %
        SplitK);
    const int warpIdInBlock = threadIdx.x / 64;
    const int laneId = threadIdx.x & 63;
    const int quarterWarpId = laneId / 16;
    const int quarterLaneId = laneId & 15;

    const int warpRowsGroupBegin =
        __builtin_mxc_readfirstlane(
            warpId * (rowsGroup / numWarps) + min(warpId, rowsGroup % numWarps)) *
        APerWarp;
    const int numCycleA =
        __builtin_mxc_readfirstlane(k / (ColThreadsPerMma * ElementsPerAccess));

    const int splitNStart = __builtin_mxc_readfirstlane(
        splitNId * (NumCycleB / SplitN) + min(splitNId, (NumCycleB % SplitN)));
    const int splitNEnd = __builtin_mxc_readfirstlane(
        (splitNId + 1) * (NumCycleB / SplitN) +
        min(splitNId + 1, (NumCycleB % SplitN)));
    const int splitKStart = __builtin_mxc_readfirstlane(
        splitKId * (numCycleA / Stages / SplitK) +
        min(splitKId, (numCycleA / Stages) % SplitK));
    const int splitKEnd = __builtin_mxc_readfirstlane(
        (splitKId + 1) * (numCycleA / Stages / SplitK) +
        min(splitKId + 1, (numCycleA / Stages) % SplitK));
    const int end = __builtin_mxc_readfirstlane(splitKEnd - splitKStart);

    int A_offset =
        ((warpRowsGroupBegin * RowThreadsPerMma) * (k / ElementsPerAccess)) +
        laneId + splitKStart * Stages * 64;
    int B_offset =
        laneId + splitNStart * 64 + splitKStart * Stages * NumCycleB * 64;
    int C_offset[APerWarp];
    for (int i = 0; i < APerWarp; ++i) {
        C_offset[i] =
            (quarterLaneId + splitNStart * RowThreadsPerMma) *
                (m / ElementsPerThreadPerMma) +
            (warpRowsGroupBegin + i) * ColThreadsPerMma + quarterWarpId;
    }

    UINT4 *A_ptr[APerWarp];
    for (int i = 0; i < APerWarp; ++i) {
        A_ptr[i] = reinterpret_cast<UINT4 *>(A) + A_offset +
                   i * (RowThreadsPerMma * (k / ElementsPerAccess));
    }
    UINT4 *B_ptr = reinterpret_cast<UINT4 *>(B) + B_offset;
    CStgType *C_ptr = reinterpret_cast<CStgType *>(C);

    __shared__ UINT4 sharedTmpB[Stages][SharedNumCycleB][64];
    UINT4 *shared_ptr =
        reinterpret_cast<UINT4 *>(&sharedTmpB[0][0][0]) + laneId;

    const int splitNumCycleB =
        __builtin_mxc_readfirstlane(splitNEnd - splitNStart);

    UINT4 tmpA[Stages][APerWarp];
    UINT4 tmpB[SharedNumCycleB];
    FLOAT4 C_f32[SharedNumCycleB][APerWarp];

    for (int i = 0; i < SharedNumCycleB; ++i) {
        for (int idxA = 0; idxA < APerWarp; ++idxA) {
            C_f32[i][idxA] = {0.0f, 0.0f, 0.0f, 0.0f};
        }
    }

    if (end == 0) {
        for (int idxA = 0; idxA < splitNumCycleB; ++idxA) {
            A_ptr[idxA] = reinterpret_cast<UINT4 *>(A);
        }
    }

    int A_ptr_offset = 0;
    int B_ptr_offset = 0;
    int shared_ptr_offset = 0;

#pragma unroll
    for (int t = 0; t < Stages; ++t) {
        for (int idxA = 0; idxA < APerWarp; ++idxA) {
            tmpA[t][idxA] = __builtin_mxc_load_global_async128(
                reinterpret_cast<INT128 *>(A_ptr[idxA] + A_ptr_offset));
        }
        A_ptr_offset += 64;
#pragma unroll
        for (int j = 0; j < SharedNumCycleB; j += WarpPerBlock) {
            const int pred = (j + warpIdInBlock) < splitNumCycleB ? 1 : 0;
            LDG_B128_BSM_WITH_PREDICATOR(shared_ptr + shared_ptr_offset +
                                             (j + warpIdInBlock) * 64,
                                         B_ptr + B_ptr_offset +
                                             (j + warpIdInBlock) * 64,
                                         pred, 1, MACA_ICMP_EQ);
        }
        shared_ptr_offset += SharedNumCycleB * 64;
        B_ptr_offset += NumCycleB * 64;
    }

    for (int iter = 0; iter < end - 1; ++iter) {
        shared_ptr_offset = 0;
#pragma unroll
        for (int t = 0; t < Stages; ++t) {
            __builtin_mxc_arrive_gvmcnt((SharedArriveCount + APerWarp) *
                                        (Stages - 1));
            __builtin_mxc_barrier_inst();

            for (int j = 0; j < SharedNumCycleB; ++j) {
                tmpB[j] = sharedTmpB[t][j][laneId];
            }
            for (int j = 0; j < SharedNumCycleB; ++j) {
                for (int idxA = 0; idxA < APerWarp; ++idxA) {
                    C_f32[j][idxA] = mma_16x16x16b16<T>(
                        tmpA[t][idxA][0], tmpA[t][idxA][1], tmpB[j][0],
                        tmpB[j][1], C_f32[j][idxA]);
                    C_f32[j][idxA] = mma_16x16x16b16<T>(
                        tmpA[t][idxA][2], tmpA[t][idxA][3], tmpB[j][2],
                        tmpB[j][3], C_f32[j][idxA]);
                }
            }
            __builtin_mxc_barrier_inst();

            for (int idxA = 0; idxA < APerWarp; ++idxA) {
                tmpA[t][idxA] = __builtin_mxc_load_global_async128(
                    reinterpret_cast<INT128 *>(A_ptr[idxA] + A_ptr_offset));
            }
            A_ptr_offset += 64;
#pragma unroll
            for (int j = 0; j < SharedNumCycleB; j += WarpPerBlock) {
                const int pred = (j + warpIdInBlock) < splitNumCycleB ? 1 : 0;
                LDG_B128_BSM_WITH_PREDICATOR(shared_ptr + shared_ptr_offset +
                                                 (j + warpIdInBlock) * 64,
                                             B_ptr + B_ptr_offset +
                                                 (j + warpIdInBlock) * 64,
                                             pred, 1, MACA_ICMP_EQ);
            }
            shared_ptr_offset += SharedNumCycleB * 64;
            B_ptr_offset += NumCycleB * 64;
        }
    }

#pragma unroll
    for (int t = 0; t < Stages; ++t) {
        if (t == 0) {
            __builtin_mxc_arrive_gvmcnt((SharedArriveCount + 1) * (Stages - 1));
        } else if (t == 1) {
            __builtin_mxc_arrive_gvmcnt((SharedArriveCount + 1) * (Stages - 2));
        } else if (t == 2) {
            __builtin_mxc_arrive_gvmcnt((SharedArriveCount + 1) * (Stages - 3));
        } else {
            __builtin_mxc_arrive_gvmcnt((SharedArriveCount + 1) * (Stages - 4));
        }
        __builtin_mxc_barrier_inst();

        for (int j = 0; j < SharedNumCycleB; ++j) {
            tmpB[j] = sharedTmpB[t][j][laneId];
        }
        for (int j = 0; j < SharedNumCycleB; ++j) {
            for (int idxA = 0; idxA < APerWarp; ++idxA) {
                C_f32[j][idxA] = mma_16x16x16b16<T>(
                    tmpA[t][idxA][0], tmpA[t][idxA][1], tmpB[j][0], tmpB[j][1],
                    C_f32[j][idxA]);
                C_f32[j][idxA] = mma_16x16x16b16<T>(
                    tmpA[t][idxA][2], tmpA[t][idxA][3], tmpB[j][2], tmpB[j][3],
                    C_f32[j][idxA]);
            }
        }
    }

    if (end == 0) return;

    for (int j = 0; j < SharedNumCycleB && j < splitNumCycleB; ++j) {
        for (int idxA = 0; idxA < APerWarp; ++idxA) {
            float c_f32_res[4];
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                c_f32_res[t] = C_f32[j][idxA][t] * alpha;
            }
            Tc c_tc_tmp[4];
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                c_tc_tmp[t] = static_cast<Tc>(c_f32_res[t]);
            }
            if constexpr (SplitK > 1) {
                if constexpr (std::is_same_v<Tc, __half>) {
                    atomicAdd(reinterpret_cast<__half2 *>(&C_ptr[C_offset[idxA]]),
                              {c_tc_tmp[0], c_tc_tmp[1]});
                    atomicAdd(reinterpret_cast<__half2 *>(&C_ptr[C_offset[idxA]]) + 1,
                              {c_tc_tmp[2], c_tc_tmp[3]});
                } else if constexpr (std::is_same_v<Tc, __maca_bfloat16>) {
                    atomicAdd(reinterpret_cast<__maca_bfloat162 *>(
                                  &C_ptr[C_offset[idxA]]),
                              {c_tc_tmp[0], c_tc_tmp[1]});
                    atomicAdd(reinterpret_cast<__maca_bfloat162 *>(
                                  &C_ptr[C_offset[idxA]]) + 1,
                              {c_tc_tmp[2], c_tc_tmp[3]});
                } else {
                    C_ptr[C_offset[idxA]] =
                        *reinterpret_cast<CStgType *>(&c_tc_tmp[0]);
                }
            } else {
                C_ptr[C_offset[idxA]] =
                    *reinterpret_cast<CStgType *>(&c_tc_tmp[0]);
            }
            C_offset[idxA] += RowThreadsPerMma * (m / ElementsPerThreadPerMma);
        }
    }
}

} // namespace bf16_c500_tk_local::kernel
