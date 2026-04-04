# BF16 C500 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable `c500` backend layer plus a new `bf16_c500` GEMM example that uses native C500 layout, async staging, and BF16 MMA semantics inside ThunderKittens.

**Architecture:** Add a minimal reusable `include/arch/c500/` backend contract first, prove its layout and fragment semantics with focused tests, then build `bf16_c500` as the first high-performance consumer. Avoid routing the hot path through `st_bf`, generic tile `load_async`, or generic `warp::mma_AB`.

**Tech Stack:** C++20, cucc/MACA builtins, ThunderKittens internal abstractions, C500-native BF16 MMA, project Makefiles, C500 unit tests.

---

## File Structure

### New files

- `include/arch/c500/traits.cuh`
  Machine-level wave64 traits and helper constants.

- `include/arch/c500/async.cuh`
  Async copy wrappers, token types, and wait helpers for C500 builtins.

- `include/arch/c500/fragments.cuh`
  Native fragment payload definitions and atom aliases reused by kernels.

- `include/arch/c500/mma.cuh`
  Builtin-backed BF16/F16 MMA wrappers exposed as backend primitives.

- `include/arch/c500/layouts/operand_layouts.cuh`
  Native shared-memory stage layout descriptors for GEMM operands.

- `include/arch/c500/layouts/lds_offsets.cuh`
  Lane-to-offset tables/helpers for A and B operand feed.

- `include/arch/c500/layouts/accumulator_export.cuh`
  Native accumulator scatter/export helpers.

- `include/arch/c500/gemm/bf16_contracts.cuh`
  CTA shape, stage count, wave ownership, and GEMM-specific constants.

- `include/arch/c500/gemm/bf16_mainloop.cuh`
  BF16 GEMM mainloop using the backend contracts.

- `include/arch/c500/gemm/bf16_epilogue.cuh`
  BF16 GEMM export/writeback helpers.

- `kernels/gemm/bf16_c500/bf16_c500_gemm.cu`
  Host shell, kernel launch, benchmark, and correctness check.

- `tests/c500/gemm/bf16_c500_layout_probe.cu`
  Validates shared layout and LDS offsets.

- `tests/c500/gemm/bf16_c500_fragment_probe.cu`
  Validates fragment feed and MMA atom semantics.

- `tests/c500/gemm/bf16_c500_gemm_smoke.cu`
  Validates the end-to-end kernel on small and large shapes.

### Existing files to modify

- `include/arch/c500/fragment_layouts.cuh`
  Either keep as thin compatibility shim or redirect to the new fragment definitions.

- `kernels/common.mk`
  Add build wiring for `bf16_c500`.

- `tests/Makefile`
  Register new C500 GEMM tests.

- `tests/c500/c500.cu`
  Include the new GEMM test unit if needed by the existing aggregation pattern.

- `tests/c500/memory/memory.cu`
  Only if the repository’s C500 test organization requires adding declarations/dispatch.

## Task 1: Land The Minimal `c500` Machine Layer

**Files:**
- Create: `include/arch/c500/traits.cuh`
- Create: `include/arch/c500/async.cuh`
- Create: `include/arch/c500/fragments.cuh`
- Create: `include/arch/c500/mma.cuh`
- Modify: `include/arch/c500/fragment_layouts.cuh`

- [ ] **Step 1: Write the failing compile-only contract check**

Create `tests/c500/gemm/bf16_c500_backend_compile_probe.cu` with:

```cpp
#include "kittens.cuh"
#include "arch/c500/traits.cuh"
#include "arch/c500/fragments.cuh"
#include "arch/c500/mma.cuh"

using namespace kittens;

__global__ void bf16_c500_backend_compile_probe() {
#ifdef KITTENS_C500
    static_assert(kittens::arch::c500::wave_traits::kWaveSize == 64);
    using atom = kittens::arch::c500::bf16_mma_atom;
    static_assert(atom::M == 16 && atom::N == 16 && atom::K == 16);
    kittens::arch::c500::fragment_a<atom> a{};
    kittens::arch::c500::fragment_b<atom> b{};
    kittens::arch::c500::fragment_c<atom> c{};
    auto d = kittens::arch::c500::mma(atom{}, a, b, c);
    (void)d;
#endif
}
```

- [ ] **Step 2: Run the compile probe to verify it fails**

Run:

```bash
cd /data/ThunderKittens/tests && make clean GPU_TARGET=C500 >/dev/null && \
make all -j8 GPU_TARGET=C500 COMP_LEVEL=fast NVCC=cucc TEST_INTENSITY=1 \
TEST_MACROS='-DNUM_GPUS=1 -DTEST_C500_GEMM_BACKEND_COMPILE_PROBE' \
TESTS_SRC='unit_tests.cu c500/c500.cu c500/gemm/bf16_c500_backend_compile_probe.cu testing_commons/testing_utils.cu'
```

Expected: FAIL with missing `include/arch/c500/{traits,fragments,mma}.cuh` or undefined symbols.

- [ ] **Step 3: Write the minimal machine-layer headers**

Add `include/arch/c500/traits.cuh`:

```cpp
#pragma once

namespace kittens::arch::c500 {

struct wave_traits {
    static constexpr int kWaveSize = 64;
    static constexpr int kLaneMask = 63;
    static constexpr int kLaneGroupSize = 16;

    __device__ static inline int lane_id() { return threadIdx.x & kLaneMask; }
    __device__ static inline int wave_id() { return threadIdx.x / kWaveSize; }
    __device__ static inline int lane_row(int lane) { return lane & 0x0f; }
    __device__ static inline int lane_group(int lane) { return lane >> 4; }
};

} // namespace kittens::arch::c500
```

Add `include/arch/c500/fragments.cuh`:

```cpp
#pragma once

#include "fragment_layouts.cuh"

namespace kittens::arch::c500 {

using bf16_mma_atom = mma_bf16_16x16x16_fp32;
using f16_mma_atom = mma_f16_16x16x16_fp32;

template<typename Atom>
struct fragment_a { uint32_t reg[Atom::a_registers] = {}; };

template<typename Atom>
struct fragment_b { uint32_t reg[Atom::b_registers] = {}; };

template<typename Atom>
struct fragment_c { float reg[Atom::c_registers] = {}; };

} // namespace kittens::arch::c500
```

Add `include/arch/c500/mma.cuh`:

```cpp
#pragma once

#include "fragments.cuh"

namespace kittens::arch::c500 {

template<typename Atom>
__device__ inline fragment_c<Atom> mma(Atom, fragment_a<Atom> const& a, fragment_b<Atom> const& b, fragment_c<Atom> const& c);

template<>
__device__ inline fragment_c<bf16_mma_atom> mma(bf16_mma_atom,
                                                fragment_a<bf16_mma_atom> const& a,
                                                fragment_b<bf16_mma_atom> const& b,
                                                fragment_c<bf16_mma_atom> const& c) {
    fragment_c<bf16_mma_atom> out;
    auto result = __builtin_mxc_mma_16x16x16bf16(
        {a.reg[0], a.reg[1], 0, 0},
        {b.reg[0], b.reg[1], 0, 0},
        {c.reg[0], c.reg[1], c.reg[2], c.reg[3]});
    out.reg[0] = result[0];
    out.reg[1] = result[1];
    out.reg[2] = result[2];
    out.reg[3] = result[3];
    return out;
}

} // namespace kittens::arch::c500
```

Add `include/arch/c500/async.cuh`:

```cpp
#pragma once

#include <maca.h>

namespace kittens::arch::c500 {

template<int Transactions>
struct async_token {
    static constexpr int transactions = Transactions;
};

template<int RemainingOutstanding>
__device__ inline void wait_gvmcnt() {
    __builtin_mxc_arrive_gvmcnt(RemainingOutstanding);
    __builtin_mxc_barrier_inst();
}

template<typename T>
__device__ inline void async_copy_128b(void* dst_shared_ptr,
                                       T const* src,
                                       int cmp_lhs,
                                       int cmp_rhs) {
    __builtin_mxc_ldg_b128_bsm_predicator(
        dst_shared_ptr,
        const_cast<void*>(reinterpret_cast<void const*>(src)),
        0,
        true,
        true,
        false,
        true,
        cmp_lhs,
        cmp_rhs,
        MACA_ICMP_SLT);
}

} // namespace kittens::arch::c500
```

Update `include/arch/c500/fragment_layouts.cuh` to keep atom traits only and avoid duplicate payload definitions.

- [ ] **Step 4: Run the compile probe to verify it passes**

Run the same command from Step 2.

Expected: build succeeds.

- [ ] **Step 5: Commit**

```bash
git add include/arch/c500/traits.cuh include/arch/c500/async.cuh include/arch/c500/fragments.cuh include/arch/c500/mma.cuh include/arch/c500/fragment_layouts.cuh tests/c500/gemm/bf16_c500_backend_compile_probe.cu
git commit -m "feat: add c500 machine layer contracts"
```

## Task 2: Add Native Layout Contracts And Probes

**Files:**
- Create: `include/arch/c500/layouts/operand_layouts.cuh`
- Create: `include/arch/c500/layouts/lds_offsets.cuh`
- Create: `include/arch/c500/layouts/accumulator_export.cuh`
- Create: `tests/c500/gemm/bf16_c500_layout_probe.cu`
- Create: `tests/c500/gemm/bf16_c500_fragment_probe.cu`
- Modify: `tests/Makefile`
- Modify: `tests/c500/c500.cu`

- [ ] **Step 1: Write the failing layout probe**

Create `tests/c500/gemm/bf16_c500_layout_probe.cu` with:

```cpp
#include "kittens.cuh"
#include "arch/c500/traits.cuh"
#include "arch/c500/layouts/operand_layouts.cuh"
#include "arch/c500/layouts/lds_offsets.cuh"

TEST(c500_gemm, bf16_c500_layout_probe) {
#ifdef KITTENS_C500
    using layout = kittens::arch::c500::gemm::bf16_128x128x128_stage_layout;
    ASSERT_EQ(layout::kStages, 4);
    ASSERT_EQ(layout::kTileM, 128);
    ASSERT_EQ(layout::kTileN, 128);
    ASSERT_EQ(layout::kTileK, 128);
    ASSERT_EQ(layout::kWaveCount, 4);
    ASSERT_EQ(kittens::arch::c500::gemm::lds_offset_a(0, 0), 0);
#endif
}
```

- [ ] **Step 2: Run the layout probe to verify it fails**

Run:

```bash
cd /data/ThunderKittens/tests && make clean GPU_TARGET=C500 >/dev/null && \
make all -j8 GPU_TARGET=C500 COMP_LEVEL=fast NVCC=cucc TEST_INTENSITY=1 \
TEST_MACROS='-DNUM_GPUS=1 -DTEST_C500_GEMM_LAYOUT_PROBE' \
TESTS_SRC='unit_tests.cu c500/c500.cu c500/gemm/bf16_c500_layout_probe.cu testing_commons/testing_utils.cu'
```

Expected: FAIL due to missing layout contracts.

- [ ] **Step 3: Add the native layout contract headers**

Add `include/arch/c500/layouts/operand_layouts.cuh`:

```cpp
#pragma once

namespace kittens::arch::c500::gemm {

struct bf16_128x128x128_stage_layout {
    static constexpr int kTileM = 128;
    static constexpr int kTileN = 128;
    static constexpr int kTileK = 128;
    static constexpr int kStages = 4;
    static constexpr int kThreads = 256;
    static constexpr int kWaveCount = 4;
    static constexpr int kStageBytes = 0x4000;
    static constexpr int kAStageOffset = 0x0000;
    static constexpr int kBStageOffset = 0x2000;
};

} // namespace kittens::arch::c500::gemm
```

Add `include/arch/c500/layouts/lds_offsets.cuh`:

```cpp
#pragma once

#include "../traits.cuh"
#include "operand_layouts.cuh"

namespace kittens::arch::c500::gemm {

__host__ __device__ inline int lds_offset_a(int lane, int i) {
    int slot = lane / wave_traits::kWaveSize;
    int wave_lane = lane & (wave_traits::kWaveSize - 1);
    return (wave_lane + (slot / 2) * 0x1000 / 16 + i * 0x400 / 16) * 16;
}

__host__ __device__ inline int lds_offset_b(int lane, int i) {
    int slot = lane / wave_traits::kWaveSize;
    int wave_lane = lane & (wave_traits::kWaveSize - 1);
    return (wave_lane + 0x2000 / 16 + (slot & 1) * 0x1000 / 16 + i * 0x400 / 16) * 16;
}

} // namespace kittens::arch::c500::gemm
```

Add `include/arch/c500/layouts/accumulator_export.cuh` as a stub contract:

```cpp
#pragma once

namespace kittens::arch::c500::gemm {

struct accumulator_tile_map {
    static constexpr int kWaveM = 2;
    static constexpr int kWaveN = 2;
};

} // namespace kittens::arch::c500::gemm
```

- [ ] **Step 4: Add a fragment probe that exercises the atom contract**

Create `tests/c500/gemm/bf16_c500_fragment_probe.cu`:

```cpp
#include "kittens.cuh"
#include "arch/c500/fragments.cuh"
#include "arch/c500/mma.cuh"

TEST(c500_gemm, bf16_c500_fragment_probe) {
#ifdef KITTENS_C500
    using atom = kittens::arch::c500::bf16_mma_atom;
    kittens::arch::c500::fragment_a<atom> a{};
    kittens::arch::c500::fragment_b<atom> b{};
    kittens::arch::c500::fragment_c<atom> c{};
    auto d = kittens::arch::c500::mma(atom{}, a, b, c);
    ASSERT_EQ(sizeof(d.reg) / sizeof(d.reg[0]), 4);
#endif
}
```

- [ ] **Step 5: Run both probes and verify they pass**

Run:

```bash
cd /data/ThunderKittens/tests && make clean GPU_TARGET=C500 >/dev/null && \
make all -j8 GPU_TARGET=C500 COMP_LEVEL=fast NVCC=cucc TEST_INTENSITY=1 \
TEST_MACROS='-DNUM_GPUS=1 -DTEST_C500_GEMM_LAYOUT_PROBE -DTEST_C500_GEMM_FRAGMENT_PROBE' \
TESTS_SRC='unit_tests.cu c500/c500.cu c500/gemm/bf16_c500_layout_probe.cu c500/gemm/bf16_c500_fragment_probe.cu testing_commons/testing_utils.cu' && \
./unit_tests
```

Expected: both tests PASS.

- [ ] **Step 6: Commit**

```bash
git add include/arch/c500/layouts/operand_layouts.cuh include/arch/c500/layouts/lds_offsets.cuh include/arch/c500/layouts/accumulator_export.cuh tests/c500/gemm/bf16_c500_layout_probe.cu tests/c500/gemm/bf16_c500_fragment_probe.cu tests/Makefile tests/c500/c500.cu
git commit -m "test: add c500 native layout and fragment probes"
```

## Task 3: Create The `bf16_c500` Example Skeleton

**Files:**
- Create: `include/arch/c500/gemm/bf16_contracts.cuh`
- Create: `include/arch/c500/gemm/bf16_epilogue.cuh`
- Create: `kernels/gemm/bf16_c500/bf16_c500_gemm.cu`
- Modify: `kernels/common.mk`

- [ ] **Step 1: Write the failing example build target**

Run:

```bash
make -C /data/ThunderKittens/kernels/gemm/bf16_c500 clean all GPU=C500
```

Expected: FAIL because the directory and files do not exist.

- [ ] **Step 2: Add BF16 GEMM contracts and example skeleton**

Add `include/arch/c500/gemm/bf16_contracts.cuh`:

```cpp
#pragma once

namespace kittens::arch::c500::gemm {

struct bf16_contracts {
    static constexpr int kBlockM = 128;
    static constexpr int kBlockN = 128;
    static constexpr int kBlockK = 128;
    static constexpr int kThreads = 256;
    static constexpr int kStages = 4;
    static constexpr int kWaveM = 2;
    static constexpr int kWaveN = 2;
};

} // namespace kittens::arch::c500::gemm
```

Add `include/arch/c500/gemm/bf16_epilogue.cuh`:

```cpp
#pragma once

#include "../layouts/accumulator_export.cuh"

namespace kittens::arch::c500::gemm {

template<typename GlobalC, typename Accumulator>
__device__ inline void store_epilogue(GlobalC const&, Accumulator const&, int, int) {
    // Placeholder phase-1 epilogue contract; implementation arrives with the kernel.
}

} // namespace kittens::arch::c500::gemm
```

Add `kernels/gemm/bf16_c500/bf16_c500_gemm.cu`:

```cpp
#include <cuda_runtime.h>
#include <iostream>

#include "../common.cuh"
#include "kittens.cuh"
#include "arch/c500/gemm/bf16_contracts.cuh"

using namespace kittens;

namespace bf16_c500 {

template<int M, int N, int K>
__global__ void gemm_kernel(const __grid_constant__ int) {}

int run() {
    std::cout << "bf16_c500 TK GEMM" << std::endl;
    return 0;
}

} // namespace bf16_c500

int main() { return bf16_c500::run(); }
```

- [ ] **Step 3: Add the build wiring**

Update `kernels/common.mk` to recognize `kernels/gemm/bf16_c500` the same way existing GEMM example directories are wired.

- [ ] **Step 4: Build the example and verify the skeleton links**

Run:

```bash
make -C /data/ThunderKittens/kernels/gemm/bf16_c500 clean all GPU=C500
./kernels/gemm/bf16_c500/bf16_c500_gemm.out
```

Expected: build succeeds and prints `bf16_c500 TK GEMM`.

- [ ] **Step 5: Commit**

```bash
git add include/arch/c500/gemm/bf16_contracts.cuh include/arch/c500/gemm/bf16_epilogue.cuh kernels/gemm/bf16_c500/bf16_c500_gemm.cu kernels/common.mk
git commit -m "feat: add bf16 c500 gemm skeleton"
```

## Task 4: Implement A Minimal Native Mainloop And Smoke Test

**Files:**
- Create: `include/arch/c500/gemm/bf16_mainloop.cuh`
- Create: `tests/c500/gemm/bf16_c500_gemm_smoke.cu`
- Modify: `kernels/gemm/bf16_c500/bf16_c500_gemm.cu`
- Modify: `include/arch/c500/gemm/bf16_epilogue.cuh`

- [ ] **Step 1: Write the failing GEMM smoke test**

Create `tests/c500/gemm/bf16_c500_gemm_smoke.cu`:

```cpp
#include "kittens.cuh"

TEST(c500_gemm, bf16_c500_gemm_smoke) {
#ifdef KITTENS_C500
    ASSERT_EQ(0, system("/data/ThunderKittens/kernels/gemm/bf16_c500/bf16_c500_gemm.out"));
#endif
}
```

- [ ] **Step 2: Run the smoke test to verify it fails functionally**

Run:

```bash
make -C /data/ThunderKittens/kernels/gemm/bf16_c500 clean all GPU=C500 && \
cd /data/ThunderKittens/tests && make clean GPU_TARGET=C500 >/dev/null && \
make all -j8 GPU_TARGET=C500 COMP_LEVEL=fast NVCC=cucc TEST_INTENSITY=1 \
TEST_MACROS='-DNUM_GPUS=1 -DTEST_C500_GEMM_SMOKE' \
TESTS_SRC='unit_tests.cu c500/c500.cu c500/gemm/bf16_c500_gemm_smoke.cu testing_commons/testing_utils.cu' && \
./unit_tests
```

Expected: FAIL because the kernel does not compute or validate GEMM yet.

- [ ] **Step 3: Implement the minimal one-stage native mainloop**

Add `include/arch/c500/gemm/bf16_mainloop.cuh`:

```cpp
#pragma once

#include "../async.cuh"
#include "../fragments.cuh"
#include "../mma.cuh"
#include "../layouts/operand_layouts.cuh"
#include "../layouts/lds_offsets.cuh"

namespace kittens::arch::c500::gemm {

template<typename Globals>
__device__ inline void run_bf16_mainloop(Globals const&) {
    // Phase-1 mainloop: one-stage native path for correctness before multistage overlap.
}

} // namespace kittens::arch::c500::gemm
```

Then replace the skeleton kernel in `kernels/gemm/bf16_c500/bf16_c500_gemm.cu` with:

```cpp
template <int M, int N, int K>
__global__ void gemm_kernel(const __grid_constant__ gemm_globals<M, N, K> g) {
    kittens::arch::c500::gemm::run_bf16_mainloop(g);
}
```

Implement the host shell to:

- allocate A/B/C/device buffers
- fill A/B randomly
- launch the kernel
- compute a reference GEMM
- print mean/max error

Reuse the host-side pattern from `kernels/gemm/bf16_ampere/bf16_ampere_gemm.cu` rather than inventing a new benchmark harness.

- [ ] **Step 4: Run the smoke test and direct example build**

Run:

```bash
make -C /data/ThunderKittens/kernels/gemm/bf16_c500 clean all GPU=C500
./kernels/gemm/bf16_c500/bf16_c500_gemm.out
```

Expected: the example runs, prints numerical error stats, and the smoke test passes on at least one small configuration.

- [ ] **Step 5: Commit**

```bash
git add include/arch/c500/gemm/bf16_mainloop.cuh include/arch/c500/gemm/bf16_epilogue.cuh kernels/gemm/bf16_c500/bf16_c500_gemm.cu tests/c500/gemm/bf16_c500_gemm_smoke.cu
git commit -m "feat: add minimal native bf16 c500 gemm path"
```

## Task 5: Upgrade To The Full 4-Stage Native Pipeline

**Files:**
- Modify: `include/arch/c500/async.cuh`
- Modify: `include/arch/c500/layouts/operand_layouts.cuh`
- Modify: `include/arch/c500/gemm/bf16_mainloop.cuh`
- Modify: `kernels/gemm/bf16_c500/bf16_c500_gemm.cu`
- Modify: `tests/c500/gemm/bf16_c500_gemm_smoke.cu`

- [ ] **Step 1: Extend the smoke test to include multiple shapes**

Update `tests/c500/gemm/bf16_c500_gemm_smoke.cu` to cover:

```cpp
TEST(c500_gemm, bf16_c500_gemm_smoke_small) { /* 128^3 */ }
TEST(c500_gemm, bf16_c500_gemm_smoke_large_k32) { /* 4096x4096x32 */ }
TEST(c500_gemm, bf16_c500_gemm_smoke_full) { /* 4096^3 */ }
```

- [ ] **Step 2: Run the smoke tests to verify the one-stage version does not meet the full target**

Run the test binary and record correctness/performance.

Expected: correctness may pass, but the implementation does not yet use the full 4-stage overlap path.

- [ ] **Step 3: Implement the multistage pipeline**

Update `include/arch/c500/gemm/bf16_mainloop.cuh` to:

- add 4-stage prologue
- use stage-indexed native shared layout
- use `async_copy_128b(...)` for A and B operand staging
- use `wait_gvmcnt<...>()` at prologue and steady-state boundaries
- interleave future-stage copy launch with current-stage MMA consumption

Use the stage geometry from `operand_layouts.cuh` and the CTA constants from `bf16_contracts.cuh`.

- [ ] **Step 4: Verify correctness on the three smoke configurations**

Run:

```bash
make -C /data/ThunderKittens/kernels/gemm/bf16_c500 clean all GPU=C500
./kernels/gemm/bf16_c500/bf16_c500_gemm.out
cd /data/ThunderKittens/tests && ./unit_tests
```

Expected: all three smoke tests PASS; direct example reports numerically acceptable BF16 error.

- [ ] **Step 5: Commit**

```bash
git add include/arch/c500/async.cuh include/arch/c500/layouts/operand_layouts.cuh include/arch/c500/gemm/bf16_mainloop.cuh kernels/gemm/bf16_c500/bf16_c500_gemm.cu tests/c500/gemm/bf16_c500_gemm_smoke.cu
git commit -m "perf: add multistage native pipeline for bf16 c500 gemm"
```

## Task 6: Benchmark Against Existing C500 Baselines

**Files:**
- Modify: `kernels/gemm/bf16_c500/bf16_c500_gemm.cu`
- Modify: `docs/analysis/2026-04-04-c500-bf16-gemm-raw-async-findings.md`

- [ ] **Step 1: Add benchmark knobs matching the existing GEMM examples**

Update `kernels/gemm/bf16_c500/bf16_c500_gemm.cu` to support:

```cpp
#ifndef BF16_C500_PROBLEM_M
#define BF16_C500_PROBLEM_M 4096
#endif
#ifndef BF16_C500_PROBLEM_N
#define BF16_C500_PROBLEM_N 4096
#endif
#ifndef BF16_C500_PROBLEM_K
#define BF16_C500_PROBLEM_K 4096
#endif
```

and matching warmup/profile iteration macros.

- [ ] **Step 2: Run baseline benchmarks**

Run:

```bash
make -C /data/ThunderKittens/kernels/gemm/bf16_c500 clean all GPU=C500
./kernels/gemm/bf16_c500/bf16_c500_gemm.out
make -C /data/ThunderKittens/kernels/gemm/bf16_ampere clean all GPU=C500
./kernels/gemm/bf16_ampere/bf16_ampere_gemm.out
```

Expected: collect runtime, TFLOP/s, and error for the new path and the old path.

- [ ] **Step 3: Record the comparison in the analysis doc**

Append a section to `docs/analysis/2026-04-04-c500-bf16-gemm-raw-async-findings.md` summarizing:

- old synchronous baseline
- old raw-async baseline
- new `bf16_c500`
- mcBLAS baseline

- [ ] **Step 4: Commit**

```bash
git add kernels/gemm/bf16_c500/bf16_c500_gemm.cu docs/analysis/2026-04-04-c500-bf16-gemm-raw-async-findings.md
git commit -m "bench: compare bf16 c500 gemm against existing baselines"
```

## Self-Review

- Spec coverage:
  - `c500` reusable backend layer is covered by Tasks 1 and 2.
  - `bf16_c500` example creation is covered by Tasks 3 and 4.
  - full native pipeline is covered by Task 5.
  - correctness and performance validation are covered by Tasks 2, 4, 5, and 6.
- Placeholder scan:
  - No `TODO`/`TBD` placeholders remain in task steps.
  - The only intentionally open-ended parts are tuning decisions inside Task 5, but the required files, commands, and acceptance criteria are explicit.
- Type consistency:
  - `bf16_mma_atom`, `wave_traits`, `operand_layouts`, and `bf16_contracts` names are consistent across tasks.

