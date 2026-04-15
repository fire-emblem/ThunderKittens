#include <maca_bfloat16.h>

#include "continuousc_reusea_family.cuh"
#include "primitives/pipeline/mainloop_atom.cuh"
#include "continuousc_reusea_skeleton.cuh"

using schedule =
    bf16_c500_tk_cute_local::cute_tk::continuousc_reusea_schedule<128, 2, 2, 1, 4>;

static_assert(schedule::kWaveSize == 64);
static_assert(schedule::kRowThreadsPerMma == 16);

using mainloop = bf16_c500_tk_cute_local::cute_tk::mainloop_atom;
using stage3_family =
    bf16_c500_tk_cute_local::cute_tk::families::continuousc_reusea_family<
        bf16_c500_tk_cute_local::cute_tk::tile_shape_policy<128, 128, 128>,
        bf16_c500_tk_cute_local::cute_tk::stage_count_policy<3>, 2, 2, 1>;

static_assert(!stage3_family::requires_zero_init);

__global__ void probe_store(__maca_bfloat16 *c) {
    using float4_t = bf16_c500_tk_cute_local::cute_tk::mma_atom::float4_t;
    float4_t frag = {0.0f, 0.0f, 0.0f, 0.0f};
    mainloop::wait_drain<1, 4>(0);
    bf16_c500_tk_cute_local::cute_tk::epilogue_atom::
        store_continuousc_fragment<__maca_bfloat16, float, float4_t, true, 1>(
            c, frag, 128, 128, 0, 0, 0, 0, 0, 0, 1.0f, 0.0f);
}

int main() { return 0; }
