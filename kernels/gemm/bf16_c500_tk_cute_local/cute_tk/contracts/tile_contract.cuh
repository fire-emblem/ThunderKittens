#pragma once

namespace bf16_c500_tk_local::contracts {

struct tile_contract {
    static constexpr int tile_m = 128;
    static constexpr int tile_n = 128;
    static constexpr int tile_k = 128;
    static constexpr int threads = 256;
    static constexpr int wave_size = 64;
    static constexpr int stage_count = 4;
};

} // namespace bf16_c500_tk_local::contracts
