#include "cute_tk/mainloop.cuh"
#include "cute_tk/layoutc_tt_256x256x64_traits.cuh"

using layoutc_2048_policy =
    bf16_c500_tk_cute_local::cute_tk::layoutc_perf_policy<2048, 2048, 2048>;
using layoutc_4096_family =
    bf16_c500_tk_cute_local::cute_tk::layoutc_perf_family<4096, 4096, 4096>;
using layoutc_candidate_tt =
    bf16_c500_tk_cute_local::cute_tk::tile_256x256x64;
using layoutc_candidate_nt =
    bf16_c500_tk_cute_local::cute_tk::tile_128x256x64;
using layoutc_candidate_tt_family =
    bf16_c500_tk_cute_local::cute_tk::layoutc_tt_256x256x64_stage4_candidate;
using layoutc_candidate_nt_family =
    bf16_c500_tk_cute_local::cute_tk::layoutc_nt_128x256x64_stage4_candidate;
using layoutc_square_tt_family =
    bf16_c500_tk_cute_local::cute_tk::layoutc_tt_256x256x64_stage4_family;
using layoutc_square_nt_family =
    bf16_c500_tk_cute_local::cute_tk::layoutc_nt_128x256x64_stage4_family;
using layoutc_tt_traits =
    bf16_c500_tk_cute_local::cute_tk::layoutc_tt_256x256x64_traits;
using square_tt_family =
    bf16_c500_tk_cute_local::cute_tk::square_tt_256x256x64_stage4_family;

static_assert(layoutc_2048_policy::tile_shape::tile_m == 128);
static_assert(layoutc_2048_policy::stage_policy::stage_count == 4);
static_assert(!layoutc_4096_family::requires_zero_init);
static_assert(layoutc_candidate_tt::tile_m == 256 && layoutc_candidate_tt::tile_n == 256 &&
              layoutc_candidate_tt::tile_k == 64);
static_assert(layoutc_candidate_nt::tile_m == 128 && layoutc_candidate_nt::tile_n == 256 &&
              layoutc_candidate_nt::tile_k == 64);
static_assert(!layoutc_candidate_tt_family::implemented);
static_assert(!layoutc_candidate_nt_family::implemented);
static_assert(layoutc_square_tt_family::tile_shape::tile_m == 256);
static_assert(layoutc_square_nt_family::tile_shape::tile_n == 256);
static_assert(layoutc_tt_traits::tile_m == 256);
static_assert(layoutc_tt_traits::tile_n == 256);
static_assert(layoutc_tt_traits::tile_k == 64);
static_assert(layoutc_tt_traits::waves == 4);
static_assert(layoutc_tt_traits::accum_m == 16);
static_assert(layoutc_tt_traits::accum_n == 2);
static_assert(square_tt_family::implemented);

int main() { return 0; }
