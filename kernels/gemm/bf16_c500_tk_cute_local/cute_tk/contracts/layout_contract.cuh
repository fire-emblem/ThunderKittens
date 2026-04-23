#pragma once

namespace bf16_c500_tk_cute_local::contracts {

struct layout_contract {
    static constexpr int a_pack_row = 16;
    static constexpr int a_pack_col = 8;
    static constexpr int b_pack_k = 32;
    static constexpr int b_pack_n = 16;
    static constexpr int c_pack_m = 32;
    static constexpr int c_pack_n = 16;
};

} // namespace bf16_c500_tk_cute_local::contracts
