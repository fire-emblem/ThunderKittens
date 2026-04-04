/**
 * @file
 * @brief Warp-level matrix multiply-accumulate operations for tiles stored in registers.
 */

#ifdef KITTENS_C500
struct detail {

__device__ static inline constexpr uint64_t c500_full_mask() {
    return 0xffffffffffffffffull;
}

__device__ static inline constexpr int c500_shuffle_width() {
    return 64;
}

__device__ static inline int c500_native_row(int lane_id) {
    return lane_id & 0xf;
}

__device__ static inline int c500_native_col_group(int lane_id) {
    return lane_id >> 4;
}

__device__ static inline int c500_native_col(int lane_id, int vec_idx) {
    return c500_native_col_group(lane_id) * 4 + vec_idx;
}

__device__ static inline void cute_shuffle_a(uint32_t (&aa)[2],
                                             uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                             int lane_id) {
    int delta = (lane_id % 32 % 8 * 4 + lane_id % 32 / 16 * 2) - lane_id;

    uint32_t tmp0 = __shfl_down_sync(c500_full_mask(), a0, delta);
    uint32_t tmp0_1 = __shfl_down_sync(c500_full_mask(), a0, delta + 1);
    uint32_t tmp1 = __shfl_down_sync(c500_full_mask(), a1, delta);
    uint32_t tmp1_1 = __shfl_down_sync(c500_full_mask(), a1, delta + 1);
    uint32_t tmp2 = __shfl_down_sync(c500_full_mask(), a2, delta);
    uint32_t tmp2_1 = __shfl_down_sync(c500_full_mask(), a2, delta + 1);
    uint32_t tmp3 = __shfl_down_sync(c500_full_mask(), a3, delta);
    uint32_t tmp3_1 = __shfl_down_sync(c500_full_mask(), a3, delta + 1);

    bool upper_half = lane_id % 16 / 8;
    if (lane_id < 32) {
        aa[0] = upper_half ? tmp1 : tmp0;
        aa[1] = upper_half ? tmp1_1 : tmp0_1;
    } else {
        aa[0] = upper_half ? tmp3 : tmp2;
        aa[1] = upper_half ? tmp3_1 : tmp2_1;
    }
}

__device__ static inline void cute_shuffle_b(uint32_t (&bb)[2],
                                             uint32_t b0, uint32_t b1,
                                             int lane_id) {
    int delta = (lane_id % 32 % 8 * 4 + lane_id % 32 / 16 * 2) - lane_id;

    uint32_t tmp0 = __shfl_down_sync(c500_full_mask(), b0, delta);
    uint32_t tmp0_1 = __shfl_down_sync(c500_full_mask(), b0, delta + 1);
    uint32_t tmp1 = __shfl_down_sync(c500_full_mask(), b1, delta);
    uint32_t tmp1_1 = __shfl_down_sync(c500_full_mask(), b1, delta + 1);

    if (lane_id < 32) {
        bb[0] = tmp0;
        bb[1] = tmp0_1;
    } else {
        bb[0] = tmp1;
        bb[1] = tmp1_1;
    }
}

__device__ static inline void cute_shuffle_c(float (&cc)[4],
                                             float c0, float c1, float c2, float c3,
                                             int lane_id) {
    int delta = (lane_id % 8 / 2 + lane_id % 32 / 16 * 16) - lane_id;

    float tmp_c0 = __shfl_down_sync(c500_full_mask(), c0, delta);
    float tmp_c0_4 = __shfl_down_sync(c500_full_mask(), c0, delta + 4);
    float tmp_c0_8 = __shfl_down_sync(c500_full_mask(), c0, delta + 8);
    float tmp_c0_12 = __shfl_down_sync(c500_full_mask(), c0, delta + 12);
    float tmp_c1 = __shfl_down_sync(c500_full_mask(), c1, delta);
    float tmp_c1_4 = __shfl_down_sync(c500_full_mask(), c1, delta + 4);
    float tmp_c1_8 = __shfl_down_sync(c500_full_mask(), c1, delta + 8);
    float tmp_c1_12 = __shfl_down_sync(c500_full_mask(), c1, delta + 12);
    float tmp_c2 = __shfl_down_sync(c500_full_mask(), c2, delta);
    float tmp_c2_4 = __shfl_down_sync(c500_full_mask(), c2, delta + 4);
    float tmp_c2_8 = __shfl_down_sync(c500_full_mask(), c2, delta + 8);
    float tmp_c2_12 = __shfl_down_sync(c500_full_mask(), c2, delta + 12);
    float tmp_c3 = __shfl_down_sync(c500_full_mask(), c3, delta);
    float tmp_c3_4 = __shfl_down_sync(c500_full_mask(), c3, delta + 4);
    float tmp_c3_8 = __shfl_down_sync(c500_full_mask(), c3, delta + 8);
    float tmp_c3_12 = __shfl_down_sync(c500_full_mask(), c3, delta + 12);

    if (lane_id < 32) {
        if (lane_id % 2) {
            cc[0] = tmp_c1;
            cc[1] = tmp_c1_4;
            cc[2] = tmp_c1_8;
            cc[3] = tmp_c1_12;
        } else {
            cc[0] = tmp_c0;
            cc[1] = tmp_c0_4;
            cc[2] = tmp_c0_8;
            cc[3] = tmp_c0_12;
        }
    } else {
        if (lane_id % 2) {
            cc[0] = tmp_c3;
            cc[1] = tmp_c3_4;
            cc[2] = tmp_c3_8;
            cc[3] = tmp_c3_12;
        } else {
            cc[0] = tmp_c2;
            cc[1] = tmp_c2_4;
            cc[2] = tmp_c2_8;
            cc[3] = tmp_c2_12;
        }
    }
}

__device__ static inline void c500_shuffle_tk_c_to_native(float (&cc)[4],
                                                          const float2 &c0,
                                                          const float2 &c1,
                                                          int lane_id) {
    const int row = lane_id & 0xf;
    const int lane_group = lane_id >> 4;
    const float src0_c0x = __shfl_sync(c500_full_mask(), c0.x, row + 0 * 16, c500_shuffle_width());
    const float src1_c0x = __shfl_sync(c500_full_mask(), c0.x, row + 1 * 16, c500_shuffle_width());
    const float src2_c0x = __shfl_sync(c500_full_mask(), c0.x, row + 2 * 16, c500_shuffle_width());
    const float src3_c0x = __shfl_sync(c500_full_mask(), c0.x, row + 3 * 16, c500_shuffle_width());
    const float src0_c0y = __shfl_sync(c500_full_mask(), c0.y, row + 0 * 16, c500_shuffle_width());
    const float src1_c0y = __shfl_sync(c500_full_mask(), c0.y, row + 1 * 16, c500_shuffle_width());
    const float src2_c0y = __shfl_sync(c500_full_mask(), c0.y, row + 2 * 16, c500_shuffle_width());
    const float src3_c0y = __shfl_sync(c500_full_mask(), c0.y, row + 3 * 16, c500_shuffle_width());
    const float src0_c1x = __shfl_sync(c500_full_mask(), c1.x, row + 0 * 16, c500_shuffle_width());
    const float src1_c1x = __shfl_sync(c500_full_mask(), c1.x, row + 1 * 16, c500_shuffle_width());
    const float src2_c1x = __shfl_sync(c500_full_mask(), c1.x, row + 2 * 16, c500_shuffle_width());
    const float src3_c1x = __shfl_sync(c500_full_mask(), c1.x, row + 3 * 16, c500_shuffle_width());
    const float src0_c1y = __shfl_sync(c500_full_mask(), c1.y, row + 0 * 16, c500_shuffle_width());
    const float src1_c1y = __shfl_sync(c500_full_mask(), c1.y, row + 1 * 16, c500_shuffle_width());
    const float src2_c1y = __shfl_sync(c500_full_mask(), c1.y, row + 2 * 16, c500_shuffle_width());
    const float src3_c1y = __shfl_sync(c500_full_mask(), c1.y, row + 3 * 16, c500_shuffle_width());

    if (lane_group == 0) {
        cc[0] = src0_c0x;
        cc[1] = src1_c0x;
        cc[2] = src2_c0x;
        cc[3] = src3_c0x;
    } else if (lane_group == 1) {
        cc[0] = src0_c0y;
        cc[1] = src1_c0y;
        cc[2] = src2_c0y;
        cc[3] = src3_c0y;
    } else if (lane_group == 2) {
        cc[0] = src0_c1x;
        cc[1] = src1_c1x;
        cc[2] = src2_c1x;
        cc[3] = src3_c1x;
    } else {
        cc[0] = src0_c1y;
        cc[1] = src1_c1y;
        cc[2] = src2_c1y;
        cc[3] = src3_c1y;
    }
}

template<typename Packed2>
__device__ static inline void c500_shuffle_tk_ab_to_native(_Float16 (&native)[4],
                                                           const Packed2 &p0,
                                                           const Packed2 &p1,
                                                           int lane_id) {
    const uint32_t lane_bits[4] = {
        static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(&(p0.x))),
        static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(&(p0.y))),
        static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(&(p1.x))),
        static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(&(p1.y)))
    };
    const int row = lane_id & 0xf;
    const int lane_group = lane_id >> 4;
    auto unpack = [](uint32_t bits) {
        uint16_t raw = static_cast<uint16_t>(bits);
        return *reinterpret_cast<_Float16 *>(&raw);
    };

    if (lane_group == 0) {
        native[0] = unpack(__shfl_sync(c500_full_mask(), lane_bits[0], row + 0 * 16, c500_shuffle_width()));
        native[1] = unpack(__shfl_sync(c500_full_mask(), lane_bits[0], row + 1 * 16, c500_shuffle_width()));
        native[2] = unpack(__shfl_sync(c500_full_mask(), lane_bits[0], row + 2 * 16, c500_shuffle_width()));
        native[3] = unpack(__shfl_sync(c500_full_mask(), lane_bits[0], row + 3 * 16, c500_shuffle_width()));
    } else if (lane_group == 1) {
        native[0] = unpack(__shfl_sync(c500_full_mask(), lane_bits[1], row + 0 * 16, c500_shuffle_width()));
        native[1] = unpack(__shfl_sync(c500_full_mask(), lane_bits[1], row + 1 * 16, c500_shuffle_width()));
        native[2] = unpack(__shfl_sync(c500_full_mask(), lane_bits[1], row + 2 * 16, c500_shuffle_width()));
        native[3] = unpack(__shfl_sync(c500_full_mask(), lane_bits[1], row + 3 * 16, c500_shuffle_width()));
    } else if (lane_group == 2) {
        native[0] = unpack(__shfl_sync(c500_full_mask(), lane_bits[2], row + 0 * 16, c500_shuffle_width()));
        native[1] = unpack(__shfl_sync(c500_full_mask(), lane_bits[2], row + 1 * 16, c500_shuffle_width()));
        native[2] = unpack(__shfl_sync(c500_full_mask(), lane_bits[2], row + 2 * 16, c500_shuffle_width()));
        native[3] = unpack(__shfl_sync(c500_full_mask(), lane_bits[2], row + 3 * 16, c500_shuffle_width()));
    } else {
        native[0] = unpack(__shfl_sync(c500_full_mask(), lane_bits[3], row + 0 * 16, c500_shuffle_width()));
        native[1] = unpack(__shfl_sync(c500_full_mask(), lane_bits[3], row + 1 * 16, c500_shuffle_width()));
        native[2] = unpack(__shfl_sync(c500_full_mask(), lane_bits[3], row + 2 * 16, c500_shuffle_width()));
        native[3] = unpack(__shfl_sync(c500_full_mask(), lane_bits[3], row + 3 * 16, c500_shuffle_width()));
    }
}

template<typename Result>
__device__ static inline void cute_scatter_d(float2 &d0, float2 &d1,
                                             const Result &result,
                                             int lane_id) {
    int delta = (lane_id % 4 * 2 + lane_id / 16 * 16) - lane_id;

    float tmp_d0 = __shfl_down_sync(c500_full_mask(), result[0], delta);
    float tmp_d1 = __shfl_down_sync(c500_full_mask(), result[1], delta);
    float tmp_d2 = __shfl_down_sync(c500_full_mask(), result[2], delta);
    float tmp_d3 = __shfl_down_sync(c500_full_mask(), result[3], delta);
    float tmp_d0_1 = __shfl_down_sync(c500_full_mask(), result[0], delta + 1);
    float tmp_d1_1 = __shfl_down_sync(c500_full_mask(), result[1], delta + 1);
    float tmp_d2_1 = __shfl_down_sync(c500_full_mask(), result[2], delta + 1);
    float tmp_d3_1 = __shfl_down_sync(c500_full_mask(), result[3], delta + 1);
    float tmp_d0_32 = __shfl_down_sync(c500_full_mask(), result[0], delta + 32);
    float tmp_d1_32 = __shfl_down_sync(c500_full_mask(), result[1], delta + 32);
    float tmp_d2_32 = __shfl_down_sync(c500_full_mask(), result[2], delta + 32);
    float tmp_d3_32 = __shfl_down_sync(c500_full_mask(), result[3], delta + 32);
    float tmp_d0_32_1 = __shfl_down_sync(c500_full_mask(), result[0], delta + 33);
    float tmp_d1_32_1 = __shfl_down_sync(c500_full_mask(), result[1], delta + 33);
    float tmp_d2_32_1 = __shfl_down_sync(c500_full_mask(), result[2], delta + 33);
    float tmp_d3_32_1 = __shfl_down_sync(c500_full_mask(), result[3], delta + 33);

    if (lane_id < 32) {
        if (lane_id % 16 / 4 == 0) {
            d0 = {tmp_d0, tmp_d0_1};
            d1 = {tmp_d0_32, tmp_d0_32_1};
        } else if (lane_id % 16 / 4 == 1) {
            d0 = {tmp_d1, tmp_d1_1};
            d1 = {tmp_d1_32, tmp_d1_32_1};
        } else if (lane_id % 16 / 4 == 2) {
            d0 = {tmp_d2, tmp_d2_1};
            d1 = {tmp_d2_32, tmp_d2_32_1};
        } else {
            d0 = {tmp_d3, tmp_d3_1};
            d1 = {tmp_d3_32, tmp_d3_32_1};
        }
    }
}

template<typename Result>
__device__ static inline void c500_scatter_native_d_to_tk(float2 &d0, float2 &d1,
                                                          const Result &result,
                                                          int lane_id) {
    const int row = lane_id & 0xf;
    const int lane_group = lane_id >> 4;
    const float r0 = result[0];
    const float r1 = result[1];
    const float r2 = result[2];
    const float r3 = result[3];
    const float src0_r0 = __shfl_sync(c500_full_mask(), r0, row + 0 * 16, c500_shuffle_width());
    const float src1_r0 = __shfl_sync(c500_full_mask(), r0, row + 1 * 16, c500_shuffle_width());
    const float src2_r0 = __shfl_sync(c500_full_mask(), r0, row + 2 * 16, c500_shuffle_width());
    const float src3_r0 = __shfl_sync(c500_full_mask(), r0, row + 3 * 16, c500_shuffle_width());
    const float src0_r1 = __shfl_sync(c500_full_mask(), r1, row + 0 * 16, c500_shuffle_width());
    const float src1_r1 = __shfl_sync(c500_full_mask(), r1, row + 1 * 16, c500_shuffle_width());
    const float src2_r1 = __shfl_sync(c500_full_mask(), r1, row + 2 * 16, c500_shuffle_width());
    const float src3_r1 = __shfl_sync(c500_full_mask(), r1, row + 3 * 16, c500_shuffle_width());
    const float src0_r2 = __shfl_sync(c500_full_mask(), r2, row + 0 * 16, c500_shuffle_width());
    const float src1_r2 = __shfl_sync(c500_full_mask(), r2, row + 1 * 16, c500_shuffle_width());
    const float src2_r2 = __shfl_sync(c500_full_mask(), r2, row + 2 * 16, c500_shuffle_width());
    const float src3_r2 = __shfl_sync(c500_full_mask(), r2, row + 3 * 16, c500_shuffle_width());
    const float src0_r3 = __shfl_sync(c500_full_mask(), r3, row + 0 * 16, c500_shuffle_width());
    const float src1_r3 = __shfl_sync(c500_full_mask(), r3, row + 1 * 16, c500_shuffle_width());
    const float src2_r3 = __shfl_sync(c500_full_mask(), r3, row + 2 * 16, c500_shuffle_width());
    const float src3_r3 = __shfl_sync(c500_full_mask(), r3, row + 3 * 16, c500_shuffle_width());

    if (lane_group == 0) {
        d0.x = src0_r0;
        d0.y = src1_r0;
        d1.x = src2_r0;
        d1.y = src3_r0;
    } else if (lane_group == 1) {
        d0.x = src0_r1;
        d0.y = src1_r1;
        d1.x = src2_r1;
        d1.y = src3_r1;
    } else if (lane_group == 2) {
        d0.x = src0_r2;
        d0.y = src1_r2;
        d1.x = src2_r2;
        d1.y = src3_r2;
    } else {
        d0.x = src0_r3;
        d0.y = src1_r3;
        d1.x = src2_r3;
        d1.y = src3_r3;
    }
}

template<typename FragA, ducks::rt_base::all ABase>
__device__ static inline void c500_pack_tk_a_to_native(FragA &dst, const ABase &src) {
    static_assert(std::is_same_v<typename ABase::T, bf16> || std::is_same_v<typename ABase::T, half>,
                  "C500 native A packing expects bf16/half TK source tiles.");
    static_assert(sizeof(src.data[0]) == sizeof(uint32_t) && sizeof(src.data[1]) == sizeof(uint32_t),
                  "C500 native A packing expects two 32-bit packed TK registers.");
    dst.reg[0] = *reinterpret_cast<const uint32_t *>(&src.data[0]);
    dst.reg[1] = *reinterpret_cast<const uint32_t *>(&src.data[1]);
}

template<typename FragB, ducks::rt_base::all BBase>
__device__ static inline void c500_pack_tk_b_to_native(FragB &dst, const BBase &src) {
    static_assert(std::is_same_v<typename BBase::T, bf16> || std::is_same_v<typename BBase::T, half>,
                  "C500 native B packing expects bf16/half TK source tiles.");
    static_assert(sizeof(src.data[0]) == sizeof(uint32_t) && sizeof(src.data[1]) == sizeof(uint32_t),
                  "C500 native B packing expects two 32-bit packed TK registers.");
    dst.reg[0] = *reinterpret_cast<const uint32_t *>(&src.data[0]);
    dst.reg[1] = *reinterpret_cast<const uint32_t *>(&src.data[1]);
}

template<typename FragC, ducks::rt_base::all CBase>
__device__ static inline void c500_pack_tk_c_to_native(FragC &dst, const CBase &src) {
    static_assert(std::is_same_v<typename CBase::T, float> &&
                  std::is_same_v<typename CBase::layout, ducks::rt_layout::row>,
                  "C500 native accumulator packing expects row-major float TK tiles.");
    const int lane_id = __lane_id();
    c500_shuffle_tk_c_to_native(dst.reg, src.data[0], src.data[1], lane_id);
}

template<ducks::rt_base::all DBase, typename FragC>
__device__ static inline void c500_unpack_native_d_to_tk(DBase &dst, const FragC &src) {
    static_assert(std::is_same_v<typename DBase::T, float> &&
                  std::is_same_v<typename DBase::layout, ducks::rt_layout::row>,
                  "C500 native accumulator export expects row-major float TK tiles.");
    const int lane_id = __lane_id();
    c500_scatter_native_d_to_tk(dst.data[0], dst.data[1], src.reg, lane_id);
}

__device__ static inline void cute_hmma16816(float2 &d0, float2 &d1,
                                             uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                             uint32_t b0, uint32_t b1,
                                             const float2 &c0, const float2 &c1,
                                             std::true_type) {
    uint32_t aa[2];
    uint32_t bb[2];
    float cc[4];
    const int lane_id = __lane_id();

    cute_shuffle_a(aa, a0, a1, a2, a3, lane_id);
    cute_shuffle_b(bb, b0, b1, lane_id);
    c500_shuffle_tk_c_to_native(cc, c0, c1, lane_id);

    const half_2 *haa0 = reinterpret_cast<const half_2 *>(&aa[0]);
    const half_2 *haa1 = reinterpret_cast<const half_2 *>(&aa[1]);
    const half_2 *hbb0 = reinterpret_cast<const half_2 *>(&bb[0]);
    const half_2 *hbb1 = reinterpret_cast<const half_2 *>(&bb[1]);

    auto result = __builtin_mxc_mma_16x16x16f16(
        {static_cast<__fp16>(float(haa0->x)), static_cast<__fp16>(float(haa0->y)),
         static_cast<__fp16>(float(haa1->x)), static_cast<__fp16>(float(haa1->y))},
        {static_cast<__fp16>(float(hbb0->x)), static_cast<__fp16>(float(hbb0->y)),
         static_cast<__fp16>(float(hbb1->x)), static_cast<__fp16>(float(hbb1->y))},
        {cc[0], cc[1], cc[2], cc[3]});

    c500_scatter_native_d_to_tk(d0, d1, result, lane_id);
}

__device__ static inline void cute_hmma16816(float2 &d0, float2 &d1,
                                             uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                             uint32_t b0, uint32_t b1,
                                             const float2 &c0, const float2 &c1,
                                             std::false_type) {
    uint32_t aa[2];
    uint32_t bb[2];
    float cc[4];
    const int lane_id = __lane_id();

    cute_shuffle_a(aa, a0, a1, a2, a3, lane_id);
    cute_shuffle_b(bb, b0, b1, lane_id);
    c500_shuffle_tk_c_to_native(cc, c0, c1, lane_id);

    const bf16_2 *haa0 = reinterpret_cast<const bf16_2 *>(&aa[0]);
    const bf16_2 *haa1 = reinterpret_cast<const bf16_2 *>(&aa[1]);
    const bf16_2 *hbb0 = reinterpret_cast<const bf16_2 *>(&bb[0]);
    const bf16_2 *hbb1 = reinterpret_cast<const bf16_2 *>(&bb[1]);

    auto result = __builtin_mxc_mma_16x16x16bf16(
        {*reinterpret_cast<const _Float16 *>(&(haa0->x)), *reinterpret_cast<const _Float16 *>(&(haa0->y)),
         *reinterpret_cast<const _Float16 *>(&(haa1->x)), *reinterpret_cast<const _Float16 *>(&(haa1->y))},
        {*reinterpret_cast<const _Float16 *>(&(hbb0->x)), *reinterpret_cast<const _Float16 *>(&(hbb0->y)),
         *reinterpret_cast<const _Float16 *>(&(hbb1->x)), *reinterpret_cast<const _Float16 *>(&(hbb1->y))},
        {cc[0], cc[1], cc[2], cc[3]});

    c500_scatter_native_d_to_tk(d0, d1, result, lane_id);
}

};
#endif

/**
 * @brief Perform the HMMA.16816 operation.
 *
 * This function performs the half-precision matrix multiply-accumulate operation
 * using the `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` instruction.
 *
 * @param[out] d0 The first half of the output float2 accumulator.
 * @param[out] d1 The second half of the output float2 accumulator.
 * @param[in] a0 The first half of the first input bf16_2 matrix.
 * @param[in] a1 The second half of the first input bf16_2 matrix.
 * @param[in] a2 The first half of the second input bf16_2 matrix.
 * @param[in] a3 The second half of the second input bf16_2 matrix.
 * @param[in] b0 The first half of the bf16_2 matrix B.
 * @param[in] b1 The second half of the bf16_2 matrix B.
 * @param[in] c0 The first half of the float2 accumulator matrix C.
 * @param[in] c1 The second half of the float2 accumulator matrix C.
 */
__device__ static inline void hmma16816(      float2 &d0,       float2 &d1,
                                        const bf16_2 &a0, const bf16_2 &a1, const bf16_2 &a2, const bf16_2 &a3,
                                        const bf16_2 &b0, const bf16_2 &b1,
                                        const float2 &c0, const float2 &c1                                    ) {
#ifdef KITTENS_C500
    d0 = c0;
    d1 = c1;
    detail::cute_hmma16816(
        d0, d1,
        *reinterpret_cast<const uint32_t*>(&a0), *reinterpret_cast<const uint32_t*>(&a1),
        *reinterpret_cast<const uint32_t*>(&a2), *reinterpret_cast<const uint32_t*>(&a3),
        *reinterpret_cast<const uint32_t*>(&b0), *reinterpret_cast<const uint32_t*>(&b1),
        c0, c1,
        std::false_type{}
    );
#else
    asm volatile(
        // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#multiply-and-accumulate-instruction-mma
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
        "{%0, %1, %2, %3}, " \
        "{%4, %5, %6, %7}, " \
        "{%8, %9}, " \
        "{%10, %11, %12, %13};"

        // D matrix
    :   "+f"(d0.x), "+f"(d0.y),
        "+f"(d1.x), "+f"(d1.y)

        // A matrix
    :   "r"(*(uint32_t*)(&a0)), "r"(*(uint32_t*)(&a1)),
        "r"(*(uint32_t*)(&a2)), "r"(*(uint32_t*)(&a3)),

        // B matrix
        "r"(*(uint32_t*)(&b0)), "r"(*(uint32_t*)(&b1)),

        // C matrix
        "f"(c0.x), "f"(c0.y),
        "f"(c1.x), "f"(c1.y)
    );
#endif
}
/**
 * @brief Perform the HMMA.16816 operation with inputs as fp16 and fp32 accumulators
 *
 * This function performs the half-precision matrix multiply-accumulate operation
 * using the `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` instruction.
 *
 * @param[out] d0 The first half of the output float2 accumulator.
 * @param[out] d1 The second half of the output float2 accumulator.
 * @param[in] a0 The first half of the first input half_2 matrix.
 * @param[in] a1 The second half of the first input half_2 matrix.
 * @param[in] a2 The first half of the second input half_2 matrix.
 * @param[in] a3 The second half of the second input half_2 matrix.
 * @param[in] b0 The first half of the half_2 matrix B.
 * @param[in] b1 The second half of the half_2 matrix B.
 * @param[in] c0 The first half of the float2 accumulator matrix C.
 * @param[in] c1 The second half of the float2 accumulator matrix C.
 */
__device__ static inline void hmma16816(      float2 &d0,       float2 &d1,
                                        const half_2 &a0, const half_2 &a1, const half_2 &a2, const half_2 &a3,
                                        const half_2 &b0, const half_2 &b1,
                                        const float2 &c0, const float2 &c1                                    ) {
#ifdef KITTENS_C500
    d0 = c0;
    d1 = c1;
    detail::cute_hmma16816(
        d0, d1,
        *reinterpret_cast<const uint32_t*>(&a0), *reinterpret_cast<const uint32_t*>(&a1),
        *reinterpret_cast<const uint32_t*>(&a2), *reinterpret_cast<const uint32_t*>(&a3),
        *reinterpret_cast<const uint32_t*>(&b0), *reinterpret_cast<const uint32_t*>(&b1),
        c0, c1,
        std::true_type{}
    );
#else
    asm volatile(
        // https://docs.nvidia.com/cuda/parallel-thread-execution/#multiply-and-accumulate-instruction-mma
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 " \
        "{%0, %1, %2, %3}, " \
        "{%4, %5, %6, %7}, " \
        "{%8, %9}, " \
        "{%10, %11, %12, %13};"

        // D matrix
    :   "+f"(d0.x), "+f"(d0.y),
        "+f"(d1.x), "+f"(d1.y)

        // A matrix
    :   "r"(*(uint32_t*)(&a0)), "r"(*(uint32_t*)(&a1)),
        "r"(*(uint32_t*)(&a2)), "r"(*(uint32_t*)(&a3)),

        // B matrix
        "r"(*(uint32_t*)(&b0)), "r"(*(uint32_t*)(&b1)),

        // C matrix
        "f"(c0.x), "f"(c0.y),
        "f"(c1.x), "f"(c1.y)
    );
#endif
}
/**
 * @brief Perform the HMMA.16816 operation.
 *
 * This function performs the half-precision matrix multiply-accumulate operation
 * using the `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16` instruction.
 *
 * @param[out] d0 The first half of the output half_2 accumulator.
 * @param[out] d1 The second half of the output half_2 accumulator.
 * @param[in] a0 The first half of the first input half_2 matrix.
 * @param[in] a1 The second half of the first input half_2 matrix.
 * @param[in] a2 The first half of the second input half_2 matrix.
 * @param[in] a3 The second half of the second input half_2 matrix.
 * @param[in] b0 The first half of the half_2 matrix B.
 * @param[in] b1 The second half of the half_2 matrix B.
 * @param[in] c0 The first half of the half_2 accumulator matrix C.
 * @param[in] c1 The second half of the half_2 accumulator matrix C.
 */
__device__ static inline void hmma16816(      half_2 &d0,       half_2 &d1,
                                        const half_2 &a0, const half_2 &a1, const half_2 &a2, const half_2 &a3,
                                        const half_2 &b0, const half_2 &b1,
                                        const half_2 &c0, const half_2 &c1                                    ) {
#ifdef KITTENS_C500
    d0 = c0;
    d1 = c1;
#else
    asm volatile(
        // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#multiply-and-accumulate-instruction-mma
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 " \
        "{%0, %1}, " \
        "{%2, %3, %4, %5}, " \
        "{%6, %7}, " \
        "{%8, %9};"

        // D matrix
    :   "=r"(*(uint32_t*)(&d0)), "=r"(*(uint32_t*)(&d1))

        // A matrix
    :   "r"(*(uint32_t*)(&a0)), "r"(*(uint32_t*)(&a1)),
        "r"(*(uint32_t*)(&a2)), "r"(*(uint32_t*)(&a3)),

        // B matrix
        "r"(*(uint32_t*)(&b0)), "r"(*(uint32_t*)(&b1)),

        // C matrix
        "r"(*(uint32_t*)(&c0)), "r"(*(uint32_t*)(&c1))
    );
#endif
}

#if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)
/**
* @brief Perform the HMMA.16816 operation for FP8 using fp8e4m3_2.
*
* Using mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 instruction
* but with fp8e4m3_2 (2 FP8 values) instead of fp8e4m3_4
*/
/**
 * @brief Perform the HMMA.16816 operation for FP8.
 *
 * This function performs the fp8-precision matrix multiply-accumulate operation
 * using the `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` instruction.
 *
 * @param[out] d0 The first half of the output float2 accumulator.
 * @param[out] d1 The second half of the output float2 accumulator.
 * @param[in] a0,a1,a2,a3 Input FP8 matrix A values
 * @param[in] b0,b1 Input FP8 matrix B values
 * @param[in] c0,c1 Input float2 accumulator matrix C values
 */
__device__ static inline void hmma16816(      float2 &d0,       float2 &d1,
                                       const fp8e4m3_4 &a0, const fp8e4m3_4 &a1, 
                                       const fp8e4m3_4 &a2, const fp8e4m3_4 &a3,
                                       const fp8e4m3_4 &b0, const fp8e4m3_4 &b1,
                                       const float2 &c0, const float2 &c1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};"
        
        // D matrix (output)
        : "+f"(d0.x), "+f"(d0.y),
          "+f"(d1.x), "+f"(d1.y)
        
        // A matrix
        : "r"(*(uint32_t*)(&a0)), "r"(*(uint32_t*)(&a1)),
          "r"(*(uint32_t*)(&a2)), "r"(*(uint32_t*)(&a3)),
        
        // B matrix
        "r"(*(uint32_t*)(&b0)), "r"(*(uint32_t*)(&b1)),
        
        // C matrix
        "f"(c0.x), "f"(c0.y),
        "f"(c1.x), "f"(c1.y)
    );
}
#endif

#ifdef KITTENS_C500
template<typename Acc, typename ABase, typename BBase>
__device__ static inline void c500_native_mma_base(Acc &d, const ABase &a, const BBase &b, const Acc &c) {
    const int lane_id = __lane_id();
    float native_c[4];

    detail::c500_shuffle_tk_c_to_native(native_c, c.data[0], c.data[1], lane_id);

    if constexpr (std::is_same_v<typename ABase::dtype, bf16_2>) {
        auto result = __builtin_mxc_mma_16x16x16bf16(
            {*reinterpret_cast<const _Float16 *>(&(b.data[0].x)), *reinterpret_cast<const _Float16 *>(&(b.data[0].y)),
             *reinterpret_cast<const _Float16 *>(&(b.data[1].x)), *reinterpret_cast<const _Float16 *>(&(b.data[1].y))},
            {*reinterpret_cast<const _Float16 *>(&(a.data[0].x)), *reinterpret_cast<const _Float16 *>(&(a.data[0].y)),
             *reinterpret_cast<const _Float16 *>(&(a.data[1].x)), *reinterpret_cast<const _Float16 *>(&(a.data[1].y))},
            {native_c[0], native_c[1], native_c[2], native_c[3]});
        detail::c500_scatter_native_d_to_tk(d.data[0], d.data[1], result, lane_id);
    } else {
        auto result = __builtin_mxc_mma_16x16x16f16(
            {static_cast<__fp16>(float(b.data[0].x)), static_cast<__fp16>(float(b.data[0].y)),
             static_cast<__fp16>(float(b.data[1].x)), static_cast<__fp16>(float(b.data[1].y))},
            {static_cast<__fp16>(float(a.data[0].x)), static_cast<__fp16>(float(a.data[0].y)),
             static_cast<__fp16>(float(a.data[1].x)), static_cast<__fp16>(float(a.data[1].y))},
            {native_c[0], native_c[1], native_c[2], native_c[3]});
        detail::c500_scatter_native_d_to_tk(d.data[0], d.data[1], result, lane_id);
    }
}
#endif

/**
 * @brief Base matrix multiply-accumulate operation for row layout.
 *
 * This function performs the base matrix multiply-accumulate operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<bf16_2, row_layout> matrix.
 * @param[in] b The second input rt_base<bf16_2, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
__device__ static inline void mma_AB_base(rt_base<float, ducks::rt_layout::row> &d,
                                    const rt_base<bf16,  ducks::rt_layout::row> &a,
                                    const rt_base<bf16,  ducks::rt_layout::col> &b, // in col-major mode
                                    const rt_base<float, ducks::rt_layout::row> &c) {
#ifdef KITTENS_C500
    c500_native_mma_base(d, a, b, c);
#else
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2],
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3],
        c.data[2], c.data[3]
    );
#endif
}
/**
 * @brief Base matrix multiply-accumulate operation for row layout
 * with fp16 inputs and fp32 accumulators.
 *
 * This function performs the base matrix multiply-accumulate operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<half_2, row_layout> matrix.
 * @param[in] b The second input rt_base<half_2, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
__device__ static inline void mma_AB_base(rt_base<float, ducks::rt_layout::row> &d,
                                    const rt_base<half,  ducks::rt_layout::row> &a,
                                    const rt_base<half,  ducks::rt_layout::col> &b, // in col-major mode
                                    const rt_base<float, ducks::rt_layout::row> &c) {
#ifdef KITTENS_C500
    c500_native_mma_base(d, a, b, c);
#else
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2],
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3],
        c.data[2], c.data[3]
    );
#endif
}
#if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)
/**
 * @brief Base matrix multiply-accumulate operation for row layout.
 *
 * This function performs the base matrix multiply-accumulate operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<fp8e4m3, row_layout> matrix.
 * @param[in] b The second input rt_base<fp8e4m3, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
__device__ static inline void mma_AB_base(rt_base<float, ducks::rt_layout::row> &d,
                                    const rt_base<fp8e4m3,  ducks::rt_layout::row> &a,
                                    const rt_base<fp8e4m3,  ducks::rt_layout::col> &b, // in col-major mode
                                    const rt_base<float, ducks::rt_layout::row> &c) {
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2],
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3],
        c.data[2], c.data[3]
    );
}
#endif
/**
 * @brief Base matrix multiply-accumulate operation for row layout.
 *
 * This function performs the base matrix multiply-accumulate operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<half_2, row_layout> accumulator.
 * @param[in] a The first input rt_base<half_2, row_layout> matrix.
 * @param[in] b The second input rt_base<half_2, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_base<half_2, row_layout> accumulator matrix.
 */
__device__ static inline void mma_AB_base(rt_base<half, ducks::rt_layout::row> &d,
                                    const rt_base<half, ducks::rt_layout::row> &a,
                                    const rt_base<half, ducks::rt_layout::col> &b, // in col-major mode
                                    const rt_base<half, ducks::rt_layout::row> &c) {
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2],
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3],
        c.data[2], c.data[3]
    );
}
/**
 * @brief Base dot product operation for row layout.
 *
 * This function performs the base dot product operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<bf16_2, row_layout> matrix.
 * @param[in] b The second input rt_base<bf16_2, row_layout> matrix in row-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
__device__ static inline void mma_ABt_base(rt_base<float, ducks::rt_layout::row> &d,
                                     const rt_base<bf16,  ducks::rt_layout::row> &a,
                                     const rt_base<bf16,  ducks::rt_layout::row> &b, // in row-major mode
                                     const rt_base<float, ducks::rt_layout::row> &c) {
#ifdef KITTENS_C500
    c500_native_mma_base(d, a, b, c);
#else
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2], // for some reason this one seems to need to be backwards
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3], // for some reason this one seems to need to be backwards
        c.data[2], c.data[3]
    );
#endif
}
/**
 * @brief Base dot product operation for row layout
 * with fp16 inputs and fp32 accumulators.
 *
 * This function performs the base dot product operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<half_2, row_layout> matrix.
 * @param[in] b The second input rt_base<half_2, row_layout> matrix in row-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
__device__ static inline void mma_ABt_base(rt_base<float, ducks::rt_layout::row> &d,
                                     const rt_base<half,  ducks::rt_layout::row> &a,
                                     const rt_base<half,  ducks::rt_layout::row> &b, // in row-major mode
                                     const rt_base<float, ducks::rt_layout::row> &c) {
#ifdef KITTENS_C500
    c500_native_mma_base(d, a, b, c);
#else
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2], // for some reason this one seems to need to be backwards
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3], // for some reason this one seems to need to be backwards
        c.data[2], c.data[3]
    );
#endif
}
#if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)
/**
 * @brief Base dot product operation for row layout.
 *
 * This function performs the base dot product operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<fp8e4m3x4, row_layout> matrix.
 * @param[in] b The second input rt_base<fp8e4m3x4, row_layout> matrix in row-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
__device__ static inline void mma_ABt_base(rt_base<float, ducks::rt_layout::row> &d,
                                     const rt_base<fp8e4m3,  ducks::rt_layout::row> &a,
                                     const rt_base<fp8e4m3,  ducks::rt_layout::row> &b, // in row-major mode
                                     const rt_base<float, ducks::rt_layout::row> &c) {
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2], // for some reason this one seems to need to be backwards
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3], // for some reason this one seems to need to be backwards
        c.data[2], c.data[3]
    );
}
#endif


/**
 * @brief Base matrix multiply-accumulate operation for row layout with transposed A.
 *
 * This function performs the base matrix multiply-accumulate operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<bf16_2, col_layout> matrix.
 * @param[in] b The second input rt_base<bf16_2, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
__device__ static inline void mma_AtB_base(rt_base<float, ducks::rt_layout::row> &d,
                                     const rt_base<bf16,  ducks::rt_layout::col> &a,
                                     const rt_base<bf16,  ducks::rt_layout::col> &b, // in col-major mode
                                     const rt_base<float, ducks::rt_layout::row> &c) {
#ifdef KITTENS_C500
    c500_native_mma_base(d, a, b, c);
#else
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2],
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3],
        c.data[2], c.data[3]
    );
#endif
}
/**
 * @brief Base matrix multiply-accumulate operation for row layout with transposed A
 * with fp16 inputs and fp32 accumulators.
 *
 * This function performs the base matrix multiply-accumulate operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<half_2, col_layout> matrix.
 * @param[in] b The second input rt_base<half_2, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
__device__ static inline void mma_AtB_base(rt_base<float, ducks::rt_layout::row> &d,
                                     const rt_base<half,  ducks::rt_layout::col> &a,
                                     const rt_base<half,  ducks::rt_layout::col> &b, // in col-major mode
                                     const rt_base<float, ducks::rt_layout::row> &c) {
#ifdef KITTENS_C500
    c500_native_mma_base(d, a, b, c);
#else
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2],
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3],
        c.data[2], c.data[3]
    );
#endif
}
#if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)
/**
 * @brief Base matrix multiply-accumulate operation for row layout with transposed A.
 *
 * This function performs the base matrix multiply-accumulate operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<fp8e4m3x4, col_layout> matrix.
 * @param[in] b The second input rt_base<fp8e4m3x4, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
__device__ static inline void mma_AtB_base(rt_base<float, ducks::rt_layout::row> &d,
                                     const rt_base<fp8e4m3,  ducks::rt_layout::col> &a,
                                     const rt_base<fp8e4m3,  ducks::rt_layout::col> &b, // in col-major mode
                                     const rt_base<float, ducks::rt_layout::row> &c) {
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2],
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3],
        c.data[2], c.data[3]
    );
}
#endif

/**
 * @brief Base matrix multiply-accumulate operation for row layout with transposed A and B.
 *
 * This function performs the base matrix multiply-accumulate operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<bf16_2, col_layout> matrix.
 * @param[in] b The second input rt_base<bf16_2, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
__device__ static inline void mma_AtBt_base(rt_base<float, ducks::rt_layout::row> &d,
                                      const rt_base<bf16,  ducks::rt_layout::col> &a,
                                      const rt_base<bf16,  ducks::rt_layout::row> &b, // in col-major mode
                                      const rt_base<float, ducks::rt_layout::row> &c) {
#ifdef KITTENS_C500
    c500_native_mma_base(d, a, b, c);
#else
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2],
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3],
        c.data[2], c.data[3]
    );
#endif
}
/**
 * @brief Base matrix multiply-accumulate operation for row layout with transposed A and B
 * with fp16 inputs and fp32 accumulators.
 *
 * This function performs the base matrix multiply-accumulate operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<half_2, col_layout> matrix.
 * @param[in] b The second input rt_base<half_2, row_layout> matrix in row-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
__device__ static inline void mma_AtBt_base(rt_base<float, ducks::rt_layout::row> &d,
                                      const rt_base<half,  ducks::rt_layout::col> &a,
                                      const rt_base<half,  ducks::rt_layout::row> &b, // in row-major mode
                                      const rt_base<float, ducks::rt_layout::row> &c) {
#ifdef KITTENS_C500
    c500_native_mma_base(d, a, b, c);
#else
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2],
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3],
        c.data[2], c.data[3]
    );
#endif
}
#if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)
/**
 * @brief Base matrix multiply-accumulate operation for row layout with transposed A and B.
 *
 * This function performs the base matrix multiply-accumulate operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<fp8e4m3x4, col_layout> matrix.
 * @param[in] b The second input rt_base<fp8e4m3x4, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
__device__ static inline void mma_AtBt_base(rt_base<float, ducks::rt_layout::row> &d,
                                      const rt_base<fp8e4m3,  ducks::rt_layout::col> &a,
                                      const rt_base<fp8e4m3,  ducks::rt_layout::row> &b, // in col-major mode
                                      const rt_base<float, ducks::rt_layout::row> &c) {
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2],
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3],
        c.data[2], c.data[3]
    );
}
#endif

/**
 * @brief Matrix multiply-accumulate operation.
 *
 * This function performs the matrix multiply-accumulate operation
 * using the `hmma16816` function.
 *
 * @tparam N The number of row tiles.
 * @tparam K The number of column tiles for the A matrix and row tiles for the B matrix.
 * @tparam M The number of column tiles for the B matrix.
 * @param[out] d The output rt_hf<N, M, row_layout> accumulator.
 * @param[in] a The first input rt_hf<N, K, row_layout> matrix.
 * @param[in] b The second input rt_hf<K, M, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_hf<N, M, row_layout> accumulator matrix.
 */
template<ducks::rt::row_layout D, ducks::rt::row_layout A, ducks::rt::col_layout B, ducks::rt::row_layout C>
__device__ static inline void mma_AB(D &d,
                               const A &a,
                               const B &b,
                               const C &c) {
    KITTENS_CHECK_WARP
    static_assert(D::rows == A::rows && D::cols == B::cols); // Check D matches A, B
    static_assert(A::cols == B::rows); // Check reduction dim is same
    static_assert(D::rows == C::rows && D::cols == C::cols); // Check D matches C
    #if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)
    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)   ||
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, fp8e4m3> &&
            std::is_same_v<typename B::T, fp8e4m3> && std::is_same_v<typename C::T, float>)
    );
    #else
    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)
    );
    #endif
#ifdef KITTENS_C500
    if constexpr (std::is_same_v<typename D::T, float> &&
                  std::is_same_v<typename A::T, bf16> &&
                  std::is_same_v<typename B::T, bf16> &&
                  std::is_same_v<typename C::T, float>) {
        using atom = kittens::arch::c500::mma_bf16_16x16x16_fp32;
        using native_a = kittens::arch::c500::fragment_a<atom>;
        using native_b = kittens::arch::c500::fragment_b<atom>;
        using native_c = kittens::arch::c500::fragment_c<atom>;

        #pragma unroll
        for (int n = 0; n < D::height; ++n) {
            #pragma unroll
            for (int m = 0; m < D::width; ++m) {
                native_c acc_native;
                detail::c500_pack_tk_c_to_native(acc_native, c.tiles[n][m]);

                #pragma unroll
                for (int k = 0; k < A::width; ++k) {
                    native_a a_native;
                    native_b b_native;
                    native_c next_native;
                    detail::c500_pack_tk_a_to_native(a_native, a.tiles[n][k]);
                    detail::c500_pack_tk_b_to_native(b_native, b.tiles[k][m]);
                    kittens::arch::c500::mma<atom>(next_native, a_native, b_native, acc_native);
                    acc_native = next_native;
                }

                detail::c500_unpack_native_d_to_tk(d.tiles[n][m], acc_native);
            }
        }
        return;
    }
#endif
    #pragma unroll
    for(int n = 0; n < D::height; n++) {
        #pragma unroll
        for(int m = 0; m < D::width; m++) {
            mma_AB_base(
                d.tiles[n][m],
                a.tiles[n][0],
                b.tiles[0][m],
                c.tiles[n][m]
            );
            #pragma unroll
            for(int k = 1; k < A::width; k++) {
                mma_AB_base(
                    d.tiles[n][m],
                    a.tiles[n][k],
                    b.tiles[k][m],
                    d.tiles[n][m]
                );
            }
        }
    }
}
/**
 * @brief Dot product operation for row layout.
 *
 * This function performs the dot product operation
 * using the `hmma16816` function.
 *
 * @tparam N The number of row tiles.
 * @tparam K The number of column tiles for the A matrix and row tiles for the B matrix.
 * @tparam M The number of column tiles for the B matrix.
 * @param[out] d The output rt_fl<N, M, row_layout> accumulator.
 * @param[in] a The first input rt_bf<N, K, row_layout> matrix.
 * @param[in] b The second input rt_bf<M, K, row_layout> matrix in row-major mode.
 * @param[in] c The input rt_fl<N, M, row_layout> accumulator matrix.
 */
template<ducks::rt::row_layout D, ducks::rt::row_layout A, ducks::rt::row_layout B, ducks::rt::row_layout C>
__device__ static inline void mma_ABt(D &d,
                                const A &a,
                                const B &b, // notice row and (M, K) instead of col and (K, M)
                                const C &c) {
    KITTENS_CHECK_WARP
    static_assert(D::rows == A::rows && D::cols == B::rows); // Check D matches A, B
    static_assert(A::cols == B::cols); // Check reduction dim is same
    static_assert(D::rows == C::rows && D::cols == C::cols); // Check D matches C
    #if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)
    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)  ||
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, fp8e4m3> &&
            std::is_same_v<typename B::T, fp8e4m3> && std::is_same_v<typename C::T, float>)
    );
    #else
    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)
    );
    #endif
    #pragma unroll
    for(int n = 0; n < D::height; n++) {
        #pragma unroll
        for(int m = 0; m < D::width; m++) {
            mma_ABt_base(
                d.tiles[n][m],
                a.tiles[n][0],
                b.tiles[m][0],
                c.tiles[n][m]
            );
            #pragma unroll
            for(int k = 1; k < A::width; k++) {
                mma_ABt_base(
                    d.tiles[n][m],
                    a.tiles[n][k],
                    b.tiles[m][k],
                    d.tiles[n][m]
                );
            }
        }
    }
}
/**
 * @brief Matrix multiply-accumulate operation with transposed A.
 *
 * This function performs the matrix multiply-accumulate operation
 * using the `hmma16816` instruction.
 *
 * @tparam N The number of row tiles.
 * @tparam K The number of column tiles for the A matrix and row tiles for the B matrix.
 * @tparam M The number of column tiles for the B matrix.
 * @param[out] d The output rt_fl<N, M, row_layout> accumulator.
 * @param[in] a The first input rt_bf<K, N, row_layout> matrix.
 * @param[in] b The second input rt_bf<K, M, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_fl<N, M, row_layout> accumulator matrix.
 */
template<ducks::rt::row_layout D, ducks::rt::col_layout A, ducks::rt::col_layout B, ducks::rt::row_layout C>
__device__ static inline void mma_AtB(D &d,
                                const A &a,
                                const B &b,
                                const C &c) {
    KITTENS_CHECK_WARP
    static_assert(D::rows == A::cols && D::cols == B::cols); // Check D matches A, B
    static_assert(A::rows == B::rows); // Check reduction dim is same
    static_assert(D::rows == C::rows && D::cols == C::cols); // Check D matches C
    #if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)
    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)   ||
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, fp8e4m3> &&
            std::is_same_v<typename B::T, fp8e4m3> && std::is_same_v<typename C::T, float>)
    );
    #else
    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)
    );
    #endif
    #pragma unroll
    for(int n = 0; n < D::height; n++) {
        #pragma unroll
        for(int m = 0; m < D::width; m++) {
            mma_AtB_base(
                d.tiles[n][m],
                a.tiles[0][n],
                b.tiles[0][m],
                c.tiles[n][m]
            );
            #pragma unroll
            for(int k = 1; k < A::height; k++) {
                mma_AtB_base(
                    d.tiles[n][m],
                    a.tiles[k][n],
                    b.tiles[k][m],
                    d.tiles[n][m]
                );
            }
        }
    }
}
/**
 * @brief Matrix multiply-accumulate operation with transposed A and B.
 *
 * This function performs the matrix multiply-accumulate operation
 * using the `hmma16816` instruction.
 *
 * @tparam N The number of row tiles.
 * @tparam K The number of column tiles for the A matrix and row tiles for the B matrix.
 * @tparam M The number of column tiles for the B matrix.
 * @param[out] d The output rt_fl<N, M, row_layout> accumulator.
 * @param[in] a The first input rt_bf<K, N, col_layout> matrix.
 * @param[in] b The second input rt_bf<M, K, row_layout> matrix in column-major mode.
 * @param[in] c The input rt_fl<N, M, row_layout> accumulator matrix.
 */
template<ducks::rt::row_layout D, ducks::rt::col_layout A, ducks::rt::row_layout B, ducks::rt::row_layout C>
__device__ static inline void mma_AtBt(D &d,
                                 const A &a,
                                 const B &b,
                                 const C &c) {
    KITTENS_CHECK_WARP
    static_assert(D::rows == A::cols && D::cols == B::rows); // Check D matches A, B
    static_assert(A::rows == B::cols); // Check reduction dim is same
    static_assert(D::rows == C::rows && D::cols == C::cols); // Check D matches C
    #if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)
    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)   ||
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, fp8e4m3> &&
            std::is_same_v<typename B::T, fp8e4m3> && std::is_same_v<typename C::T, float>)
    );
    #else
    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)
    );
    #endif
    #pragma unroll
    for(int n = 0; n < D::height; n++) {
        #pragma unroll
        for(int m = 0; m < D::width; m++) {
            mma_AtBt_base(
                d.tiles[n][m],
                a.tiles[0][n],
                b.tiles[m][0],
                c.tiles[n][m]
            );
            #pragma unroll
            for(int k = 1; k < A::height; k++) {
                mma_AtBt_base(
                    d.tiles[n][m],
                    a.tiles[k][n],
                    b.tiles[m][k],
                    d.tiles[n][m]
                );
            }
        }
    }
}

template<int trans_A, int trans_B, ducks::rt::all D, ducks::rt::all A, ducks::rt::all B, ducks::rt::all C>
__device__ static inline void mma(D &d,
                                  const A &a,
                                  const B &b,
                                  const C &c) {
    KITTENS_CHECK_WARP
    if constexpr(trans_A == transpose::T) {
        if constexpr(trans_B == transpose::T) {
            mma_AtBt(d, a, b, c);
        } else {
            mma_AtB(d, a, b, c);
        }
    } else {
        if constexpr(trans_B == transpose::T) {
            mma_ABt(d, a, b, c);
        } else {
            mma_AB(d, a, b, c);
        }
    }
}
template<int trans_A, int trans_B, ducks::rt::all A, ducks::rt::all B, ducks::rt::all C>
__device__ static inline C mma(const A &a,
                               const B &b,
                               const C &c) {
    KITTENS_CHECK_WARP
    C d;
    if constexpr(trans_A == transpose::T) {
        if constexpr(trans_B == transpose::T) {
            mma_AtBt(d, a, b, c);
        } else {
            mma_AtB(d, a, b, c);
        }
    } else {
        if constexpr(trans_B == transpose::T) {
            mma_ABt(d, a, b, c);
        } else {
            mma_AB(d, a, b, c);
        }
    }
    return d;
}


//  --------------------------------------------------------------------------------------------------------------------
//  --------------------------------------------------------------------------------------------------------------------
//  -------------------------------------------------- COMPLEX INPUTS --------------------------------------------------
//  --------------------------------------------------------------------------------------------------------------------
//  --------------------------------------------------------------------------------------------------------------------



/**
 * @brief Matrix multiply-accumulate operation for complex tiles
 *
 * This function calls mma_AB with hf arguments
 *
 * @tparam N The number of row tiles.
 * @tparam K The number of column tiles for the A matrix and row tiles for the B matrix.
 * @tparam M The number of column tiles for the B matrix.
 * @param[out] d The output rt_cmplx_hf<N, M, row_layout> accumulator.
 * @param[in] a The first input rt_cmplx_hf<N, K, row_layout> matrix.
 * @param[in] b The second input rt_cmplx_hf<K, M, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_cmplx_hf<N, M, row_layout> accumulator matrix.
 */
template<int N, int K, int M>
__device__ static inline void mma_AB(crt_hf<N, M, ducks::rt_layout::row> &d,
                               const crt_hf<N, K, ducks::rt_layout::row> &a,
                               const crt_hf<K, M, ducks::rt_layout::col> &b,
                               const crt_hf<N, M, ducks::rt_layout::row> &c) {
    KITTENS_CHECK_WARP
    
    // Copy data from input accumulate register into output
    ::kittens::group<1>::copy(d.real, c.real);
    ::kittens::group<1>::copy(d.imag, c.imag);

    // Negative on B matrix so we can use single accum register
    rt_hf<N, K, ducks::rt_layout::row> tmp;
    // Hex value for -1 in float16
    constexpr half factor = std::bit_cast<__half>(uint16_t(0xFB80));
    ::kittens::group<1>::mul(tmp, a.imag, factor);
    mma_AB(d.real, a.real, b.real, d.real);
    mma_AB(d.real, tmp, b.imag, d.real);

    mma_AB(d.imag, a.real, b.imag, d.imag);
    mma_AB(d.imag, a.imag, b.real, d.imag);
}
/**
 * @brief Matrix multiply-accumulate operation for complex tiles
 *
 * This function calls mma_AB with bf16 arguments
 *
 * @tparam N The number of row tiles.
 * @tparam K The number of column tiles for the A matrix and row tiles for the B matrix.
 * @tparam M The number of column tiles for the B matrix.
 * @param[out] d The output rt_cmplx_fl<N, M, row_layout> accumulator.
 * @param[in] a The first input rt_cmplx_bf<N, K, row_layout> matrix.
 * @param[in] b The second input rt_cmplx_bf<K, M, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_cmplx_fl<N, M, row_layout> accumulator matrix.
 */

template<int N, int K, int M>
__device__ static inline void mma_AB(crt_fl<N, M, ducks::rt_layout::row> &d,
                               const crt_bf<N, K, ducks::rt_layout::row> &a,
                               const crt_bf<K, M, ducks::rt_layout::col> &b,
                               const crt_fl<N, M, ducks::rt_layout::row> &c) {
    KITTENS_CHECK_WARP
    
    // Copy data from input accumulate register into output
    ::kittens::group<1>::copy(d.real, c.real);
    ::kittens::group<1>::copy(d.imag, c.imag);

    // Negative on B matrix so we can use single accum register
    kittens::rt_bf<N, K, ducks::rt_layout::row> tmp;
    // Hex value for -1 in bf16
    constexpr bf16 factor = std::bit_cast<__nv_bfloat16>(uint16_t(0xBF80));
    ::kittens::group<1>::mul(tmp, a.imag, factor);
    mma_AB(d.real, a.real, b.real, d.real);
    mma_AB(d.real, tmp, b.imag, d.real);

    mma_AB(d.imag, a.real, b.imag, d.imag);
    mma_AB(d.imag, a.imag, b.real, d.imag);
}
