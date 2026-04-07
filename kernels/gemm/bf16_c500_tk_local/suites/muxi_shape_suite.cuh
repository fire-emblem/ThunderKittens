#pragma once

namespace bf16_c500_tk_local::suites {

#define TK_LOCAL_BF16_CASE_MEMBERS                                            \
    using family = bf16_mainloop_family_t<m, n, k>;                           \
    using local_t = __maca_bfloat16;                                          \
    using ref_t = __nv_bfloat16;

struct baseline_4096_cube_case {
    static constexpr const char *case_name = "baseline_4096_cube";
    static constexpr int m = 4096;
    static constexpr int n = 4096;
    static constexpr int k = 4096;
    static constexpr int warmup_iters = 3;
    static constexpr int profile_iters = 10;
    TK_LOCAL_BF16_CASE_MEMBERS
};

struct qwen2_4608x128x3584_case {
    static constexpr const char *case_name = "qwen2_4608x128x3584";
    static constexpr int m = 4608;
    static constexpr int n = 128;
    static constexpr int k = 3584;
    static constexpr int warmup_iters = 3;
    static constexpr int profile_iters = 10;
    TK_LOCAL_BF16_CASE_MEMBERS
};

struct qwen2_3584x128x3584_case {
    static constexpr const char *case_name = "qwen2_3584x128x3584";
    static constexpr int m = 3584;
    static constexpr int n = 128;
    static constexpr int k = 3584;
    static constexpr int warmup_iters = 3;
    static constexpr int profile_iters = 10;
    TK_LOCAL_BF16_CASE_MEMBERS
};

struct qwen2_3584x128x18944_case {
    static constexpr const char *case_name = "qwen2_3584x128x18944";
    static constexpr int m = 3584;
    static constexpr int n = 128;
    static constexpr int k = 18944;
    static constexpr int warmup_iters = 3;
    static constexpr int profile_iters = 10;
    TK_LOCAL_BF16_CASE_MEMBERS
};

struct llama_7168x128x2048_case {
    static constexpr const char *case_name = "llama_7168x128x2048";
    static constexpr int m = 7168;
    static constexpr int n = 128;
    static constexpr int k = 2048;
    static constexpr int warmup_iters = 3;
    static constexpr int profile_iters = 10;
    TK_LOCAL_BF16_CASE_MEMBERS
};

struct llama_7168x256x4096_case {
    static constexpr const char *case_name = "llama_7168x256x4096";
    static constexpr int m = 7168;
    static constexpr int n = 256;
    static constexpr int k = 4096;
    static constexpr int warmup_iters = 3;
    static constexpr int profile_iters = 10;
    TK_LOCAL_BF16_CASE_MEMBERS
};

struct layoutabc_512x128x7168_case {
    static constexpr const char *case_name = "layoutabc_512x128x7168";
    static constexpr int m = 512;
    static constexpr int n = 128;
    static constexpr int k = 7168;
    static constexpr int warmup_iters = 3;
    static constexpr int profile_iters = 10;
    TK_LOCAL_BF16_CASE_MEMBERS
};

struct layouta_7168x128x256_case {
    static constexpr const char *case_name = "layouta_7168x128x256";
    static constexpr int m = 7168;
    static constexpr int n = 128;
    static constexpr int k = 256;
    static constexpr int warmup_iters = 3;
    static constexpr int profile_iters = 10;
    TK_LOCAL_BF16_CASE_MEMBERS
};

#undef TK_LOCAL_BF16_CASE_MEMBERS

} // namespace bf16_c500_tk_local::suites
