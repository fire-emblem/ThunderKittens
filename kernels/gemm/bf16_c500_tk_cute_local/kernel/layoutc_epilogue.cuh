#pragma once

namespace bf16_c500_tk_local::kernel {

template <typename CStgType, bool HasOneDimBias>
__device__ __forceinline__ void load_layoutc_bias_fragment(
    CStgType (&bias_load)[4],
    const void *bias,
    int start_row,
    int slot,
    int lane) {
    if constexpr (HasOneDimBias) {
        for (int i = 0; i < 4; ++i) {
            const int bias_offset =
                start_row / 16 * 4 + (lane / 16) + slot / 2 * 4 * 4 + i * 4;
            bias_load[i] =
                (reinterpret_cast<const CStgType *>(bias))[bias_offset];
        }
    }
}

} // namespace bf16_c500_tk_local::kernel
