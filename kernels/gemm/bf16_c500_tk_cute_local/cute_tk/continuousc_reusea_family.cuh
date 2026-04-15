#pragma once

#include <cstdlib>
#include <iostream>

#include <cuda_runtime.h>

#include "../host/layout_traits.cuh"
#include "composition/family_pattern.cuh"
#include "primitives/structure/geometry_atom.cuh"
#include "continuousc_reusea_skeleton.cuh"
#include "policies.cuh"
#include "primitives/structure/stage_layout_atom.cuh"

namespace bf16_c500_tk_cute_local::cute_tk::families {

template <typename TileShape, typename StagePolicy, int APerWarp, int SplitN,
          int SplitK,
          typename GeometryAtom = ::bf16_c500_tk_cute_local::cute_tk::continuousc_layout_atom,
          typename SchedulePolicy =
              ::bf16_c500_tk_cute_local::cute_tk::continuousc_reusea_schedule_policy<
                  StagePolicy, APerWarp, SplitN, SplitK>,
          typename StageLayoutAtom = ::bf16_c500_tk_cute_local::cute_tk::default_stage_layout_atom>
struct continuousc_reusea_family
    : ::bf16_c500_tk_cute_local::cute_tk::family_pattern<
          ::bf16_c500_tk_cute_local::cute_tk::continuousc_reusea_semantic_tag,
          TileShape, GeometryAtom, SchedulePolicy, StageLayoutAtom> {
    using pattern =
        ::bf16_c500_tk_cute_local::cute_tk::family_pattern<
            ::bf16_c500_tk_cute_local::cute_tk::continuousc_reusea_semantic_tag,
            TileShape, GeometryAtom, SchedulePolicy, StageLayoutAtom>;
    using tile_shape = typename pattern::tile_shape;
    using geometry_atom = typename pattern::geometry_atom;
    using schedule_policy = typename pattern::schedule_policy;
    using stage_layout_atom = typename pattern::stage_layout_atom;
    using host_layout = typename pattern::host_layout;
    static constexpr const char *family_name =
        "cute_tk_continuousc_reusea_n_params";
    static constexpr float alpha = 1.0f;
    static constexpr float beta = 0.0f;
    static constexpr bool requires_zero_init = (SplitK > 1);

    static_assert(TileShape::tile_k == 128,
                  "cute_tk continuousc_reusea currently assumes tile_k == 128");
    static constexpr int NTile = tile_shape::tile_n;
    static constexpr int StageCount = StagePolicy::stage_count;
    static_assert(schedule_policy::stage_count == StageCount,
                  "continuousc_reusea_family schedule policy must match stage count");
    static_assert(schedule_policy::a_per_warp == APerWarp,
                  "continuousc_reusea_family schedule policy must match APerWarp");
    static_assert(schedule_policy::split_n == SplitN,
                  "continuousc_reusea_family schedule policy must match SplitN");
    static_assert(schedule_policy::split_k == SplitK,
                  "continuousc_reusea_family schedule policy must match SplitK");

    static inline dim3 grid(int m, int n) {
        (void)n;
        return dim3((m / 16 / (256 / 64 * APerWarp)) * SplitN * SplitK);
    }

    template <typename T, typename Tc, typename Tscal, bool IsBetaZero,
              bool HasOneDimBias>
    static inline void launch(dim3 grid_dim, const void *a, const void *b,
                              void *c, int m, int n, int k, int lda, int ldb,
                              int ldc, Tscal alpha_value, Tscal beta_value,
                              const void *bias = nullptr) {
        (void)lda;
        (void)ldb;
        (void)ldc;
        using schedule_t =
            ::bf16_c500_tk_cute_local::cute_tk::continuousc_reusea_schedule<
                NTile, APerWarp, SplitN, SplitK, StageCount>;
        if (!schedule_t::valid_k_partition(k)) {
            std::cerr << "cute_tk continuousc_reusea invalid stage configuration: "
                      << "k=" << k << " stage_count=" << StageCount
                      << " split_k=" << SplitK << std::endl;
            std::exit(1);
        }
        ::bf16_c500_tk_cute_local::cute_tk::kernel::
            cute_tk_continuousc_reusea_n<T, Tc, Tscal, StageCount, NTile, APerWarp,
                                         SplitN, SplitK, IsBetaZero,
                                         HasOneDimBias>
            <<<grid_dim, 256>>>(
                reinterpret_cast<T *>(const_cast<void *>(a)),
                reinterpret_cast<T *>(const_cast<void *>(b)),
                reinterpret_cast<Tc *>(c), m, n, k, alpha_value, beta_value,
                reinterpret_cast<Tc *>(const_cast<void *>(bias)));
    }
};

template <int NTile, int APerWarp, int SplitN, int SplitK>
using continuousc_reusea_n_params =
    continuousc_reusea_family<tile_shape_policy<128, NTile, 128>, stage_4,
                              APerWarp, SplitN, SplitK>;

} // namespace bf16_c500_tk_cute_local::cute_tk::families
