#pragma once

// Thread mapping atom - abstract thread-to-element mapping for tiled operations
// This is an abstract primitive - kernel-specific mappings are in kernel/

namespace bf16_c500_tk_cute_local::primitives {

// Thread map for square tile (256x256x64)
struct square_tile_thread_map_t {
    static constexpr int tile_m = 256;
    static constexpr int tile_n = 256;
    static constexpr int tile_k = 64;
    static constexpr int threads = 512;
    static constexpr int wave_size = 64;
    static constexpr int waves = threads / wave_size;

    static constexpr int ldg_a_count = 8;
    static constexpr int ldg_b_rows = 4;
    static constexpr int ldg_b_cols = 4;
    static constexpr int accum_m = 16;
    static constexpr int accum_n = 2;

    static constexpr int a_smem_bytes = tile_m * tile_k * 2;
    static constexpr int a_smem_double_buffer_bytes = a_smem_bytes * 2;

    static constexpr int sts_stride_bytes = 4096;
    static constexpr int lds_stride_bytes = 2048;
};

} // namespace bf16_c500_tk_cute_local::primitives