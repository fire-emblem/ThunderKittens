#pragma once

#include "mma_atoms.cuh"
#include "../../ops/thread/util/util.cuh"

namespace kittens::arch::c500 {

namespace detail {

template<typename Scalar>
__device__ inline uint32_t pack_pair(const Scalar &lo, const Scalar &hi) {
    using packed_type = typename base_types::packing<Scalar>::packed_type;
    const packed_type pair{lo, hi};
    return *reinterpret_cast<const uint32_t *>(&pair);
}

template<typename SharedTile>
__device__ inline uint32_t shared_addr(const SharedTile &src) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(&src.data[0]));
}

} // namespace detail

template<typename Atom, typename SharedTile>
__device__ inline void load_a(fragment_a<Atom>& dst,
                              const SharedTile& src,
                              int tile_m,
                              int tile_k) {
    if constexpr (std::is_same_v<Atom, mma_bf16_16x16x16_fp32>) {
        static_assert(std::is_same_v<typename SharedTile::dtype, typename Atom::a_scalar>,
                      "C500 bf16 A copy atom expects bf16 shared staging.");

        // A staging uses the existing C500 row-fragment pattern:
        // one row per lane, four K positions gathered with stride 4.
        const int lane = kittens::laneid();
        const int row = tile_m + (lane & 0x0f);
        const int col_group = lane >> 4;
        const uint32_t smem = detail::shared_addr(src);

        bf16 v0, v1, v2, v3;
        move<bf16>::lds(v0, src.idx(smem, {row, tile_k + col_group + 0}));
        move<bf16>::lds(v1, src.idx(smem, {row, tile_k + col_group + 4}));
        move<bf16>::lds(v2, src.idx(smem, {row, tile_k + col_group + 8}));
        move<bf16>::lds(v3, src.idx(smem, {row, tile_k + col_group + 12}));

        dst.reg[0] = detail::pack_pair(v0, v1);
        dst.reg[1] = detail::pack_pair(v2, v3);
    } else {
        static_assert(sizeof(Atom) == 0, "Unsupported C500 load_a atom.");
    }
}

template<typename Atom, typename SharedTile>
__device__ inline void load_b(fragment_b<Atom>& dst,
                              const SharedTile& src,
                              int tile_k,
                              int tile_n) {
    if constexpr (std::is_same_v<Atom, mma_bf16_16x16x16_fp32>) {
        static_assert(std::is_same_v<typename SharedTile::dtype, typename Atom::b_scalar>,
                      "C500 bf16 B copy atom expects bf16 shared staging.");

        // B staging is allowed to differ from A. The first native atom follows the
        // existing C500 column-fragment pattern to avoid hot-path repair shuffles.
        const int lane = kittens::laneid();
        const int row_group = lane >> 4;
        const int col = tile_n + (lane & 0x0f);
        const uint32_t smem = detail::shared_addr(src);

        bf16 v0, v1, v2, v3;
        move<bf16>::lds(v0, src.idx(smem, {tile_k + row_group + 0, col}));
        move<bf16>::lds(v1, src.idx(smem, {tile_k + row_group + 4, col}));
        move<bf16>::lds(v2, src.idx(smem, {tile_k + row_group + 8, col}));
        move<bf16>::lds(v3, src.idx(smem, {tile_k + row_group + 12, col}));

        dst.reg[0] = detail::pack_pair(v0, v1);
        dst.reg[1] = detail::pack_pair(v2, v3);
    } else {
        static_assert(sizeof(Atom) == 0, "Unsupported C500 load_b atom.");
    }
}

} // namespace kittens::arch::c500
