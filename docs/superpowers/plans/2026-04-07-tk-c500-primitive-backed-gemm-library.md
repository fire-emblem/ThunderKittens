# TK C500 Primitive-Backed GEMM Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first performance-first C500 native GEMM library path inside ThunderKittens, centered on a balanced BF16 `4096^3`-class family backed by explicit primitives and contracts rather than the transitional `bf16_ampere` adaptation path.

**Architecture:** Introduce a three-layer C500 stack inside ThunderKittens: a primitive layer that exposes hardware truth, a contract layer that defines native GEMM building blocks, and a first kernel-family layer that targets one balanced high-performance BF16 family. Keep ThunderKittens as the framework shell while letting the C500 backend own physical layout, fragments, pipeline semantics, and steady-state scheduling.

**Tech Stack:** ThunderKittens C++/CUDA-style kernel code, MXMACA/C500 builtins, local probe tests under `tests/c500`, existing BF16 GEMM examples under `kernels/gemm`, local mcBLAS baseline kernels, git for frequent commits.

---

## File Map

### Existing files to modify

- `include/arch/c500/async_primitives.cuh`
  - Current low-level async copy wrappers; should become part of the formal primitive layer.
- `include/arch/c500/mma_atoms.cuh`
  - Current C500 MMA atom wrappers; should be kept as the seed of the native MMA primitive layer.
- `include/arch/c500/gemm/bf16_contracts.cuh`
  - Transitional balanced-family constants; should be split into clearer contract roles.
- `include/arch/c500/gemm/bf16_mainloop.cuh`
  - Transitional mainloop implementation; should stop being the single place that mixes primitives, contracts, and family behavior.
- `include/arch/c500/gemm/bf16_operand_stage.cuh`
  - Current bridge-style operand path; should either become a formal native operand contract or be demoted to compatibility/diagnostic support.
- `kernels/gemm/bf16_c500/bf16_c500_gemm.cu`
  - Current C500 GEMM entry point; should become the first balanced-family launch surface.
- `kernels/gemm/bf16_ampere/bf16_ampere_gemm.cu`
  - Keep only as a transitional compatibility target if still needed by tests; do not let it remain the architecture anchor.

### New files to create

- `include/arch/c500/primitives/copy.cuh`
  - Formal copy primitive API for global-to-shared and shared-to-register operations.
- `include/arch/c500/primitives/mma.cuh`
  - Formal native MMA primitive API for BF16/F16 -> FP32 atoms.
- `include/arch/c500/primitives/pipeline.cuh`
  - Formal wait/barrier/arrival primitives for steady-state scheduling.
- `include/arch/c500/primitives/layout.cuh`
  - Shared traits for lane ownership, native fragment semantics, and stage residency.
- `include/arch/c500/gemm/contracts/bf16_balanced_contracts.cuh`
  - The first balanced-family tile/stage/operand/export contracts.
- `include/arch/c500/gemm/contracts/bf16_balanced_stage_layout.cuh`
  - Physical shared-memory stage layout for the first balanced family.
- `include/arch/c500/gemm/contracts/bf16_balanced_operand_layout.cuh`
  - Native operand ownership and fragment materialization rules.
- `include/arch/c500/gemm/families/bf16_balanced_128x128x128_stage4.cuh`
  - The first performance anchor family mainloop and epilogue wiring.
- `include/arch/c500/gemm/dispatch/bf16_dispatch.cuh`
  - A minimal dispatch layer that selects the first balanced family for supported shapes.
- `tests/c500/gemm/bf16_c500_balanced_contract_probe.cu`
  - Probe for balanced-family stage/operand/export claims.
- `tests/c500/gemm/bf16_c500_balanced_family_smoke.cu`
  - Smoke test for the new family path.
- `tests/c500/gemm/bf16_c500_balanced_family_perf.cu`
  - Performance benchmark harness for `4096^3`.

### Existing tests to keep running

- `tests/c500/gemm/bf16_c500_backend_compile_probe.cu`
- `tests/c500/gemm/bf16_c500_operand_stage_probe.cu`
- `tests/c500/gemm/bf16_c500_raw_vector_gemm_probe.cu`
- `tests/c500/memory/raw_gemm_async.cu`
- `kernels/gemm/baselines/bf16_mcblas/bf16_mcblas_gemm.cu`

---

### Task 1: Freeze the C500 primitive layer boundary

**Files:**
- Create: `include/arch/c500/primitives/copy.cuh`
- Create: `include/arch/c500/primitives/mma.cuh`
- Create: `include/arch/c500/primitives/pipeline.cuh`
- Create: `include/arch/c500/primitives/layout.cuh`
- Modify: `include/arch/c500/async_primitives.cuh`
- Modify: `include/arch/c500/mma_atoms.cuh`
- Test: `tests/c500/gemm/bf16_c500_backend_compile_probe.cu`

- [ ] **Step 1: Write the failing compile probe update**

Add a compile-only test include block to `tests/c500/gemm/bf16_c500_backend_compile_probe.cu` so the new primitive headers become required:

```cpp
#include "arch/c500/primitives/copy.cuh"
#include "arch/c500/primitives/mma.cuh"
#include "arch/c500/primitives/pipeline.cuh"
#include "arch/c500/primitives/layout.cuh"

static_assert(kittens::arch::c500::primitives::bf16_mma_atom::M == 16);
static_assert(kittens::arch::c500::primitives::balanced_wave_traits::kWaveSize == 64);
```

- [ ] **Step 2: Run the compile probe to verify it fails**

Run: `make -C tests/c500/gemm bf16_c500_backend_compile_probe`

Expected: FAIL with missing header or missing symbol errors for the new primitive namespace.

- [ ] **Step 3: Create the minimal primitive-layer headers**

Seed the new files with the formal namespace and re-export the existing transitional primitives under a stable boundary:

```cpp
// include/arch/c500/primitives/mma.cuh
#pragma once

#include "../mma_atoms.cuh"

namespace kittens::arch::c500::primitives {

using bf16_mma_atom = kittens::arch::c500::mma_bf16_16x16x16_fp32;
using bf16_fragment_a = kittens::arch::c500::fragment_a<bf16_mma_atom>;
using bf16_fragment_b = kittens::arch::c500::fragment_b<bf16_mma_atom>;
using bf16_fragment_c = kittens::arch::c500::fragment_c<bf16_mma_atom>;

template<typename D, typename A, typename B, typename C>
__device__ inline void mma(D &dst, const A &a, const B &b, const C &c) {
    kittens::arch::c500::mma<bf16_mma_atom>(dst, a, b, c);
}

} // namespace kittens::arch::c500::primitives
```

```cpp
// include/arch/c500/primitives/layout.cuh
#pragma once

namespace kittens::arch::c500::primitives {

struct balanced_wave_traits {
    static constexpr int kWaveSize = 64;
    static constexpr int kLaneGroupWidth = 16;
};

} // namespace kittens::arch::c500::primitives
```

- [ ] **Step 4: Rewire the old headers to include the new layer**

Update `include/arch/c500/async_primitives.cuh` and `include/arch/c500/mma_atoms.cuh` to remain source-compatible but explicitly comment that the stable entry point for new GEMM code is `arch/c500/primitives/*`.

```cpp
// Transitional note for new GEMM code:
// use arch/c500/primitives/*.cuh as the stable backend entry layer.
```

- [ ] **Step 5: Run the compile probe to verify it passes**

Run: `make -C tests/c500/gemm bf16_c500_backend_compile_probe`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add include/arch/c500/primitives/copy.cuh \
        include/arch/c500/primitives/mma.cuh \
        include/arch/c500/primitives/pipeline.cuh \
        include/arch/c500/primitives/layout.cuh \
        include/arch/c500/async_primitives.cuh \
        include/arch/c500/mma_atoms.cuh \
        tests/c500/gemm/bf16_c500_backend_compile_probe.cu
git commit -m "feat: add formal c500 primitive layer"
```

### Task 2: Split the first balanced-family contracts out of the transitional mainloop

**Files:**
- Create: `include/arch/c500/gemm/contracts/bf16_balanced_contracts.cuh`
- Create: `include/arch/c500/gemm/contracts/bf16_balanced_stage_layout.cuh`
- Create: `include/arch/c500/gemm/contracts/bf16_balanced_operand_layout.cuh`
- Modify: `include/arch/c500/gemm/bf16_contracts.cuh`
- Modify: `include/arch/c500/gemm/bf16_operand_stage.cuh`
- Test: `tests/c500/gemm/bf16_c500_balanced_contract_probe.cu`

- [ ] **Step 1: Write the failing contract probe**

Create `tests/c500/gemm/bf16_c500_balanced_contract_probe.cu` with assertions that the first family contract is explicit and self-contained:

```cpp
#include "arch/c500/gemm/contracts/bf16_balanced_contracts.cuh"

using contracts = kittens::arch::c500::gemm::contracts::bf16_balanced_128x128x128_stage4;

static_assert(contracts::kBlockM == 128);
static_assert(contracts::kBlockN == 128);
static_assert(contracts::kBlockK == 128);
static_assert(contracts::kStages == 4);
static_assert(contracts::kWaveSize == 64);
```

- [ ] **Step 2: Run the probe to verify it fails**

Run: `make -C tests/c500/gemm bf16_c500_balanced_contract_probe`

Expected: FAIL because the new contract headers do not yet exist.

- [ ] **Step 3: Create the balanced-family contract headers**

Move the fixed constants out of `bf16_contracts.cuh` into a family-specific namespace:

```cpp
// include/arch/c500/gemm/contracts/bf16_balanced_contracts.cuh
#pragma once

namespace kittens::arch::c500::gemm::contracts {

struct bf16_balanced_128x128x128_stage4 {
    static constexpr int kBlockM = 128;
    static constexpr int kBlockN = 128;
    static constexpr int kBlockK = 128;
    static constexpr int kStageK = 32;
    static constexpr int kStages = 4;
    static constexpr int kThreads = 256;
    static constexpr int kWaveSize = 64;
    static constexpr int kWaveM = 2;
    static constexpr int kWaveN = 2;
};

} // namespace kittens::arch::c500::gemm::contracts
```

- [ ] **Step 4: Demote the old `bf16_contracts` header to a compatibility alias**

Replace the old constants with a type alias or forwarding struct so existing code still compiles while new code uses the family-specific contract.

```cpp
using bf16_contracts = contracts::bf16_balanced_128x128x128_stage4;
```

- [ ] **Step 5: Refactor operand-stage helpers to depend on the new contract namespace**

Update `include/arch/c500/gemm/bf16_operand_stage.cuh` to include the new contract headers and stop owning baked-in family constants outside that namespace.

- [ ] **Step 6: Run the probe to verify it passes**

Run: `make -C tests/c500/gemm bf16_c500_balanced_contract_probe`

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add include/arch/c500/gemm/contracts/bf16_balanced_contracts.cuh \
        include/arch/c500/gemm/contracts/bf16_balanced_stage_layout.cuh \
        include/arch/c500/gemm/contracts/bf16_balanced_operand_layout.cuh \
        include/arch/c500/gemm/bf16_contracts.cuh \
        include/arch/c500/gemm/bf16_operand_stage.cuh \
        tests/c500/gemm/bf16_c500_balanced_contract_probe.cu
git commit -m "refactor: split balanced c500 gemm contracts"
```

### Task 3: Build the first native balanced-family mainloop skeleton

**Files:**
- Create: `include/arch/c500/gemm/families/bf16_balanced_128x128x128_stage4.cuh`
- Modify: `include/arch/c500/gemm/bf16_mainloop.cuh`
- Test: `tests/c500/gemm/bf16_c500_balanced_family_smoke.cu`

- [ ] **Step 1: Write the failing smoke test**

Create `tests/c500/gemm/bf16_c500_balanced_family_smoke.cu` to exercise one CTA tile path for a simple supported shape:

```cpp
TEST(c500_gemm, bf16_balanced_family_smoke) {
    constexpr int M = 128, N = 128, K = 128;
    // Allocate BF16 buffers, launch the new family entry, compare against a host or mcBLAS reference.
}
```

- [ ] **Step 2: Run the smoke test to verify it fails**

Run: `make -C tests/c500/gemm bf16_c500_balanced_family_smoke`

Expected: FAIL because the new family header and launch path do not exist.

- [ ] **Step 3: Create the family skeleton**

Create a new family file that owns:

- prologue
- steady-state loop
- epilogue callout

with the balanced contract as a template anchor:

```cpp
template<typename Contracts, typename Globals>
__device__ inline void run_family(const Globals &g) {
    // prologue
    // steady-state
    // epilogue
}
```

At this stage, it is acceptable to delegate internal implementation to the current transitional helpers while preserving the new family boundary. The purpose of this task is to move ownership, not yet to finalize the optimized schedule.

- [ ] **Step 4: Rewire `bf16_mainloop.cuh` into a compatibility forwarding layer**

Update `include/arch/c500/gemm/bf16_mainloop.cuh` so new code includes the family file and legacy entry points call into the family layer where possible.

- [ ] **Step 5: Run the smoke test to verify it passes**

Run: `make -C tests/c500/gemm bf16_c500_balanced_family_smoke`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add include/arch/c500/gemm/families/bf16_balanced_128x128x128_stage4.cuh \
        include/arch/c500/gemm/bf16_mainloop.cuh \
        tests/c500/gemm/bf16_c500_balanced_family_smoke.cu
git commit -m "feat: add first c500 balanced gemm family skeleton"
```

### Task 4: Move the public C500 BF16 GEMM entry point to the family/dispatch model

**Files:**
- Create: `include/arch/c500/gemm/dispatch/bf16_dispatch.cuh`
- Modify: `kernels/gemm/bf16_c500/bf16_c500_gemm.cu`
- Modify: `kernels/gemm/bf16_ampere/bf16_ampere_gemm.cu`
- Test: `tests/c500/gemm/bf16_c500_gemm_smoke.cu`

- [ ] **Step 1: Write the failing dispatch expectation in the smoke test**

Augment `tests/c500/gemm/bf16_c500_gemm_smoke.cu` so it includes the new dispatch header and calls the public C500 BF16 GEMM path through the dispatch surface.

```cpp
#include "arch/c500/gemm/dispatch/bf16_dispatch.cuh"
```

- [ ] **Step 2: Run the smoke test to verify it fails**

Run: `make -C tests/c500/gemm bf16_c500_gemm_smoke`

Expected: FAIL due to the missing dispatch layer or outdated public entry path.

- [ ] **Step 3: Add the minimal dispatch layer**

Implement a dispatch function that currently recognizes one supported family:

```cpp
namespace kittens::arch::c500::gemm::dispatch {

template<int M, int N, int K, typename Globals>
__device__ inline void run_bf16(const Globals &g) {
    families::run_bf16_balanced_128x128x128_stage4(g);
}

} // namespace kittens::arch::c500::gemm::dispatch
```

- [ ] **Step 4: Update the public kernel entry points**

Use the dispatch layer from `kernels/gemm/bf16_c500/bf16_c500_gemm.cu`.

For `kernels/gemm/bf16_ampere/bf16_ampere_gemm.cu`, keep only the minimum compatibility bridge needed to preserve current scope and tests; do not let it remain the authoritative C500 design surface.

- [ ] **Step 5: Run the smoke test to verify it passes**

Run: `make -C tests/c500/gemm bf16_c500_gemm_smoke`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add include/arch/c500/gemm/dispatch/bf16_dispatch.cuh \
        kernels/gemm/bf16_c500/bf16_c500_gemm.cu \
        kernels/gemm/bf16_ampere/bf16_ampere_gemm.cu \
        tests/c500/gemm/bf16_c500_gemm_smoke.cu
git commit -m "refactor: route c500 bf16 gemm through dispatch layer"
```

### Task 5: Replace the coarse steady-state loop with a family-owned native schedule

**Files:**
- Modify: `include/arch/c500/gemm/families/bf16_balanced_128x128x128_stage4.cuh`
- Modify: `include/arch/c500/primitives/pipeline.cuh`
- Modify: `include/arch/c500/primitives/copy.cuh`
- Test: `tests/c500/gemm/bf16_c500_stage_async_probe.cu`
- Test: `tests/c500/gemm/bf16_c500_balanced_family_smoke.cu`

- [ ] **Step 1: Strengthen the stage-async probe**

Extend `tests/c500/gemm/bf16_c500_stage_async_probe.cu` to assert the new family-owned schedule can still issue and observe async stage windows correctly.

- [ ] **Step 2: Run the probe to establish the current baseline**

Run: `make -C tests/c500/gemm bf16_c500_stage_async_probe`

Expected: PASS before refactor, establishing a guardrail.

- [ ] **Step 3: Rewrite the family steady-state to own scheduling**

Update `include/arch/c500/gemm/families/bf16_balanced_128x128x128_stage4.cuh` so the steady-state loop no longer delegates to the old coarse whole-stage structure. Express:

- prologue stage fill
- explicit future-stage issue
- partial wait windows
- consumer-side stage stepping

The implementation may still be less interleaved than the final target, but the family file must become the single owner of the schedule.

- [ ] **Step 4: Consolidate wait/count helpers in `primitives/pipeline.cuh`**

Move the wait-window helpers out of transitional GEMM files and into a reusable primitive API:

```cpp
template<int Outstanding>
__device__ inline void wait_until_transactions();
```

- [ ] **Step 5: Run probes and smoke tests**

Run:

```bash
make -C tests/c500/gemm bf16_c500_stage_async_probe
make -C tests/c500/gemm bf16_c500_balanced_family_smoke
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add include/arch/c500/gemm/families/bf16_balanced_128x128x128_stage4.cuh \
        include/arch/c500/primitives/pipeline.cuh \
        include/arch/c500/primitives/copy.cuh \
        tests/c500/gemm/bf16_c500_stage_async_probe.cu \
        tests/c500/gemm/bf16_c500_balanced_family_smoke.cu
git commit -m "feat: move c500 balanced family to native schedule ownership"
```

### Task 6: Establish the first balanced-family performance benchmark and mcBLAS comparison

**Files:**
- Create: `tests/c500/gemm/bf16_c500_balanced_family_perf.cu`
- Modify: `kernels/gemm/baselines/bf16_mcblas/bf16_mcblas_gemm.cu`
- Test: `tests/c500/gemm/bf16_c500_balanced_family_perf.cu`

- [ ] **Step 1: Write the benchmark harness**

Create `tests/c500/gemm/bf16_c500_balanced_family_perf.cu` to benchmark:

- ThunderKittens balanced family
- mcBLAS baseline
- shape `4096 x 4096 x 4096`
- device-event timing
- warmup and repeated iterations

```cpp
// Measure avg_ms and TFLOP/s for both implementations.
```

- [ ] **Step 2: Run the harness to verify it fails before implementation**

Run: `make -C tests/c500/gemm bf16_c500_balanced_family_perf`

Expected: FAIL because the new harness is not wired up yet.

- [ ] **Step 3: Add the benchmark implementation**

Use the existing mcBLAS baseline kernel as the comparison path and emit a concise report:

```cpp
printf("tk_balanced_ms=%.3f tk_tflops=%.2f mcblas_ms=%.3f mcblas_tflops=%.2f\n",
       tk_ms, tk_tflops, mcblas_ms, mcblas_tflops);
```

- [ ] **Step 4: Run the benchmark**

Run: `make -C tests/c500/gemm bf16_c500_balanced_family_perf && tests/c500/gemm/bf16_c500_balanced_family_perf`

Expected: program exits successfully and reports device-event timing for both paths.

- [ ] **Step 5: Commit**

```bash
git add tests/c500/gemm/bf16_c500_balanced_family_perf.cu \
        kernels/gemm/baselines/bf16_mcblas/bf16_mcblas_gemm.cu
git commit -m "test: add c500 balanced family perf benchmark"
```

### Task 7: Document the first family boundary and preserve compatibility paths explicitly

**Files:**
- Modify: `docs/superpowers/specs/2026-04-07-tk-c500-primitive-backed-gemm-library-design.md`
- Modify: `docs/analysis/2026-04-04-c500-bf16-gemm-raw-async-findings.md`
- Test: n/a

- [ ] **Step 1: Update the design doc with implementation status**

Add a short section documenting that the first implementation target is the balanced family and that `bf16_ampere` is no longer the architectural anchor.

- [ ] **Step 2: Update the findings doc**

Append a short note summarizing which transitional paths remain compatibility/diagnostic paths after the first family lands.

- [ ] **Step 3: Review the docs for consistency**

Run:

```bash
rg -n "bf16_ampere|balanced family|compatibility" docs/superpowers/specs docs/analysis
```

Expected: the docs show one clear story about the new architecture.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-04-07-tk-c500-primitive-backed-gemm-library-design.md \
        docs/analysis/2026-04-04-c500-bf16-gemm-raw-async-findings.md
git commit -m "docs: clarify c500 balanced family architecture"
```

---

## Spec Coverage Check

- Primitive layer: covered by Task 1.
- Contract layer: covered by Task 2.
- First balanced family: covered by Tasks 3 and 5.
- Dispatch surface: covered by Task 4.
- `4096^3` anchor benchmarking against mcBLAS: covered by Task 6.
- Documentation and compatibility-path positioning: covered by Task 7.

No spec section requiring immediate first-wave implementation is left without a task. Additional shape families are intentionally deferred until the first balanced family is proven.

## Placeholder Scan

- No `TODO` or `TBD` markers remain in task steps.
- Commands, file paths, and expected outcomes are explicitly listed.
- Later tasks rely only on symbols and files introduced in earlier tasks.

## Type and Naming Consistency

- The balanced family is consistently referred to as `bf16_balanced_128x128x128_stage4`.
- The architectural layering is consistently `primitives -> contracts -> families -> dispatch`.
- Transitional `bf16_ampere` remains a compatibility surface rather than the design anchor.

