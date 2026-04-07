#pragma once

#include <maca.h>
#include <mcr/mc_runtime.h>

#include "async.cuh"
#include "../../ops/thread/util/util.cuh"

namespace kittens::arch::c500 {

// Transitional note for new GEMM code:
// use arch/c500/primitives/*.cuh as the stable backend entry layer.

namespace detail {

template<typename Shared>
__device__ inline void *shared_tile_addr(Shared &dst, int row, int col) {
    const uint32_t shared_base = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));
    const uint32_t shared_addr = dst.idx(shared_base, {row, col});
    return __cvta_shared_to_generic(shared_addr);
}

template<typename SharedVec>
__device__ inline void *shared_vec_addr(SharedVec &dst, int idx) {
    const uint32_t shared_base = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));
    return __cvta_shared_to_generic(shared_base + idx * sizeof(typename SharedVec::dtype));
}

template<typename T>
__device__ inline void ldg_b128_bsm_no_pred(void *dst_shared_ptr, const T *src) {
    __builtin_mxc_ldg_b128_bsm_predicator(
        dst_shared_ptr,
        const_cast<void *>(reinterpret_cast<const void *>(src)),
        0,
        true,
        true,
        false,
        true,
        1,
        1,
        MACA_ICMP_EQ
    );
}

template<typename T>
__device__ inline void ldg_b128_bsm_pred(void *dst_shared_ptr,
                                         const T *src,
                                         int cmp_op1,
                                         int cmp_op2) {
    __builtin_mxc_ldg_b128_bsm_predicator(
        dst_shared_ptr,
        const_cast<void *>(reinterpret_cast<const void *>(src)),
        0,
        true,
        true,
        false,
        true,
        cmp_op1,
        cmp_op2,
        MACA_ICMP_SLT
    );
}

template<typename T>
__device__ inline void ldg_b64_bsm_no_pred(void *dst_shared_ptr, const T *src) {
    __builtin_mxc_ldg_b64_bsm_predicator(
        dst_shared_ptr,
        const_cast<void *>(reinterpret_cast<const void *>(src)),
        0,
        true,
        true,
        false,
        true,
        1,
        1,
        MACA_ICMP_EQ
    );
}

template<typename T>
__device__ inline void ldg_b64_bsm_pred(void *dst_shared_ptr,
                                        const T *src,
                                        int cmp_op1,
                                        int cmp_op2) {
    __builtin_mxc_ldg_b64_bsm_predicator(
        dst_shared_ptr,
        const_cast<void *>(reinterpret_cast<const void *>(src)),
        0,
        true,
        true,
        false,
        true,
        cmp_op1,
        cmp_op2,
        MACA_ICMP_SLT
    );
}

} // namespace detail

template<int RemainingOutstanding>
__device__ inline void wait_until() {
    wait_for_async_copies<RemainingOutstanding>();
}

template<int GroupThreads, int axis, bool assume_aligned, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ inline async_token<
    (ST::rows * ST::cols + GroupThreads * (sizeof(float4) / sizeof(typename ST::dtype)) - 1) /
    (GroupThreads * (sizeof(float4) / sizeof(typename ST::dtype)))
> load_async_tile(ST &dst, const GL &src, const COORD &idx) {
    const int row_stride = src.template stride<axis>();
    constexpr int elem_per_memcpy = sizeof(float4) / sizeof(typename ST::dtype);
    constexpr int memcpy_per_row = ST::cols / elem_per_memcpy;
    constexpr int total_calls =
        (ST::rows * ST::cols + GroupThreads * elem_per_memcpy - 1) /
        (GroupThreads * elem_per_memcpy);

    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    typename GL::dtype *src_ptr = (typename GL::dtype *)&src[unit_coord];
    const int lane = threadIdx.x % GroupThreads;
    const int tile_origin = unit_coord.template dim<axis>();
    const bool full_tile_in_bounds = tile_origin + ST::rows <= src.template shape<axis>();

#pragma unroll
    for (int i = 0; i < total_calls; ++i) {
        const int load_idx = i * GroupThreads + lane;
        const int row = load_idx / memcpy_per_row;
        const int col = (load_idx * elem_per_memcpy) % ST::cols;
        if (row >= ST::rows) {
            continue;
        }
        void *smem_ptr = detail::shared_tile_addr(dst, row, col);

        if constexpr (assume_aligned) {
            detail::ldg_b128_bsm_no_pred(smem_ptr, &src_ptr[row * row_stride + col]);
        } else {
            if (full_tile_in_bounds || row + tile_origin < src.template shape<axis>()) {
                detail::ldg_b128_bsm_no_pred(smem_ptr, &src_ptr[row * row_stride + col]);
            } else {
                *reinterpret_cast<float4 *>(smem_ptr) = float4{0.f, 0.f, 0.f, 0.f};
            }
        }
    }

    return {};
}

template<int GroupThreads, ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
__device__ inline async_token<SV::length / (sizeof(float4) / sizeof(typename SV::dtype))>
load_async_vec(SV &dst, const GL &src, const COORD &idx) {
    constexpr uint32_t elem_per_transfer = sizeof(float4) / sizeof(typename SV::dtype);
    constexpr uint32_t total_calls = SV::length / elem_per_transfer;
    typename GL::dtype *src_ptr = (typename GL::dtype *)&src[(idx.template unit_coord<-1, 3>())];

#pragma unroll
    for (uint32_t i = threadIdx.x % GroupThreads; i < total_calls; i += GroupThreads) {
        detail::ldg_b128_bsm_no_pred(detail::shared_vec_addr(dst, i * elem_per_transfer),
                                     &src_ptr[i * elem_per_transfer]);
    }

    return {};
}

template<int GroupThreads, ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
__device__ inline async_token<SV::length / (sizeof(uint64_t) / sizeof(typename SV::dtype))>
load_async_vec_small(SV &dst, const GL &src, const COORD &idx) {
    constexpr uint32_t elem_per_transfer = sizeof(uint64_t) / sizeof(typename SV::dtype);
    static_assert(sizeof(uint64_t) % sizeof(typename SV::dtype) == 0,
                  "C500 ldg_b64 requires a whole-number element grouping.");
    static_assert(SV::length % elem_per_transfer == 0,
                  "C500 ldg_b64 vector copies require an exact 64-bit transfer count.");
    constexpr uint32_t total_calls = SV::length / elem_per_transfer;
    typename GL::dtype *src_ptr = (typename GL::dtype *)&src[(idx.template unit_coord<-1, 3>())];

#pragma unroll
    for (uint32_t i = threadIdx.x % GroupThreads; i < total_calls; i += GroupThreads) {
        detail::ldg_b64_bsm_no_pred(detail::shared_vec_addr(dst, i * elem_per_transfer),
                                    &src_ptr[i * elem_per_transfer]);
    }

    return {};
}

} // namespace kittens::arch::c500
