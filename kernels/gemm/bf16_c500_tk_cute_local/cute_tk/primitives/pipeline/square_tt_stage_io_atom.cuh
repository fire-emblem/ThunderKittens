#pragma once

namespace bf16_c500_tk_cute_local::cute_tk {

struct square_tt_stage_io_atom {
    template <typename StoreType, typename FragType>
    __device__ __forceinline__ static void store_pair(
        uint8_t *smem, int base_offset, int stride_bytes, int pair_idx,
        FragType const &src0, FragType const &src1) {
        asm(";--------------");
        *reinterpret_cast<StoreType *>(smem + base_offset +
                                       stride_bytes * pair_idx) =
            *reinterpret_cast<StoreType const *>(&src0);
        *reinterpret_cast<StoreType *>(smem + base_offset +
                                       stride_bytes * (pair_idx + 1)) =
            *reinterpret_cast<StoreType const *>(&src1);
        asm(";--------------");
    }

    template <typename LoadType, typename FragType>
    __device__ __forceinline__ static void load_pair(
        FragType &dst0, FragType &dst1, uint8_t *smem, int stage_offset,
        int stride_bytes, int pair_idx) {
        asm(";--------------");
        *reinterpret_cast<LoadType *>(&dst0) =
            *reinterpret_cast<LoadType *>(smem + stage_offset +
                                          stride_bytes * pair_idx);
        *reinterpret_cast<LoadType *>(&dst1) =
            *reinterpret_cast<LoadType *>(smem + stage_offset +
                                          stride_bytes * (pair_idx + 1));
        asm(";--------------");
    }
};

} // namespace bf16_c500_tk_cute_local::cute_tk
