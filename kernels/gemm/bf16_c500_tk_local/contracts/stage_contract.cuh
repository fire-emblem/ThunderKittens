#pragma once

namespace bf16_c500_tk_local::contracts {

struct stage_contract {
    static constexpr int stage_bytes = 0x4000;
    static constexpr int a_bank0_offset = 0x0000;
    static constexpr int a_bank1_offset = 0x1000;
    static constexpr int b_bank0_offset = 0x2000;
    static constexpr int b_bank1_offset = 0x3000;
};

} // namespace bf16_c500_tk_local::contracts
