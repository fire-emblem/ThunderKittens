#include <iostream>

#include "bench/runner.cuh"
#include "suites/muxi_shape_suite.cuh"

int main() {
    using namespace bf16_c500_tk_local;
    bench::run_case<suites::baseline_4096_cube_case>();
    bench::run_case<suites::qwen2_4608x128x3584_case>();
    bench::run_case<suites::qwen2_3584x128x3584_case>();
    bench::run_case<suites::qwen2_3584x128x18944_case>();
    bench::run_case<suites::llama_7168x128x2048_case>();
    bench::run_case<suites::llama_7168x256x4096_case>();
    bench::run_case<suites::layoutabc_512x128x7168_case>();
    bench::run_case<suites::layouta_7168x128x256_case>();
    return 0;
}
