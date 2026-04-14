#pragma once

#include <cstdlib>
#include <iostream>

#include <cuda_runtime.h>

#include "../host/layout_traits.cuh"
#include "policies.cuh"

namespace bf16_c500_tk_cute_local::cute_tk {

enum class layoutc_square_candidate_kind {
    tt_256x256x64,
    nt_128x256x64,
};

template <typename TileShape, typename StagePolicy,
          layoutc_square_candidate_kind Kind_>
struct layoutc_square_candidate_family {
    using tile_shape = TileShape;
    using stage_policy = StagePolicy;
    static constexpr layoutc_square_candidate_kind kind = Kind_;
    static constexpr bool implemented = false;
};

using layoutc_tt_256x256x64_stage4_candidate =
    layoutc_square_candidate_family<tile_256x256x64, stage_4,
                                    layoutc_square_candidate_kind::tt_256x256x64>;

using layoutc_nt_128x256x64_stage4_candidate =
    layoutc_square_candidate_family<tile_128x256x64, stage_4,
                                    layoutc_square_candidate_kind::nt_128x256x64>;

template <typename TileShape, typename StagePolicy,
          layoutc_square_candidate_kind Kind_>
struct layoutc_square_family {
    using tile_shape = TileShape;
    using stage_policy = StagePolicy;
    using host_layout = ::bf16_c500_tk_local::host::layoutc_host_traits;
    static constexpr layoutc_square_candidate_kind kind = Kind_;
    static constexpr bool implemented = false;
    static constexpr float alpha = 1.0f;
    static constexpr float beta = 0.0f;
    static constexpr bool requires_zero_init = false;

    static inline dim3 grid(int m, int n) {
        return dim3(m / TileShape::tile_m, n / TileShape::tile_n);
    }

    template <typename T, typename Tc, typename Tscal, bool IsBetaZero,
              bool HasOneDimBias>
    static inline void launch(dim3, const void *, const void *, void *, int, int,
                              int, int, int, int, Tscal, Tscal,
                              const void * = nullptr) {
        std::cerr << "cute_tk square candidate family not implemented yet" << std::endl;
        std::exit(1);
    }
};

using layoutc_tt_256x256x64_stage4_family =
    layoutc_square_family<tile_256x256x64, stage_4,
                          layoutc_square_candidate_kind::tt_256x256x64>;

using layoutc_nt_128x256x64_stage4_family =
    layoutc_square_family<tile_128x256x64, stage_4,
                          layoutc_square_candidate_kind::nt_128x256x64>;

} // namespace bf16_c500_tk_cute_local::cute_tk
