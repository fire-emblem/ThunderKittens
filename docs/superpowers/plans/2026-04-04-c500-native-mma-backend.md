# C500 Native MMA Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild ThunderKittens' C500 path around native wave64 MMA fragments and copy atoms so `kernels/gemm/bf16_ampere/bf16_ampere_gemm.cu` runs correctly and can be tuned toward `mcBLAS`-class performance on `4096^3`.

**Architecture:** Add a new `arch/c500` backend layer that owns fragment layout traits, copy atoms, MMA atoms, and epilogue atoms. Keep ThunderKittens' high-level tile/group abstraction style, but let C500 use backend-specific native fragments and staging layouts in the hot path instead of forcing Ampere-style logical `rt/rv` layouts through the K-loop.

**Tech Stack:** ThunderKittens C++ templates, MXMACA/CUCC, C500 builtin MMA intrinsics, existing TK tests, C500-specific unit/perf tests, `mx-smi` and event-timed GEMM benchmarking.

---

### Task 1: Freeze The C500 Backend Contract

**Files:**
- Create: `docs/superpowers/specs/2026-04-04-c500-native-mma-backend-design.md`
- Modify: `docs/analysis/2026-04-04-c500-mma-builtins-survey.md`
- Test: `docs/superpowers/specs/2026-04-04-c500-native-mma-backend-design.md`

- [ ] **Step 1: Write the backend contract spec**

```md
# C500 Native MMA Backend Design

## Contract
- Native wave size is 64.
- The performance path is `global -> shared staging -> native fragment -> builtin mma -> native accumulator -> epilogue/store`.
- Logical `rt/rv` are not the hot-path fragment contract on C500.

## First atom
- Start with bf16/f16 inputs and fp32 accumulate.
- Freeze one native atom shape first and treat it as the source of truth for copy/mma/store semantics.

## Compatibility
- Preserve ThunderKittens' abstract style.
- Do not preserve Nvidia-specific physical fragment semantics.
- Prefer performance over full interface parity.
```

- [ ] **Step 2: Record the concrete builtins and fragment assumptions**

Add a new section to `docs/analysis/2026-04-04-c500-mma-builtins-survey.md` summarizing:

```md
## Frozen first-wave backend contract

- Primary atom: bf16/f16 input, fp32 accumulate
- Backend owns:
  - native A/B/C fragment layout
  - shared-to-native copy atoms
  - builtin invocation wrapper
  - native-accumulator export path
- No hot-path round-trip through logical `rt/rv`
```

- [ ] **Step 3: Review the spec for ambiguity**

Run: `sed -n '1,220p' docs/superpowers/specs/2026-04-04-c500-native-mma-backend-design.md`

Expected: The document clearly distinguishes logical TK abstractions from C500 native fragment contracts and explicitly states that performance takes priority over full physical-layout compatibility.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-04-04-c500-native-mma-backend-design.md docs/analysis/2026-04-04-c500-mma-builtins-survey.md
git commit -m "docs: define c500 native mma backend contract"
```

### Task 2: Add C500 Backend Skeleton Files

**Files:**
- Create: `include/arch/c500/mma_atoms.cuh`
- Create: `include/arch/c500/fragment_layouts.cuh`
- Create: `include/arch/c500/copy_atoms.cuh`
- Create: `include/arch/c500/epilogue_atoms.cuh`
- Modify: `include/common/common.cuh`
- Test: `include/arch/c500/mma_atoms.cuh`

- [ ] **Step 1: Create the fragment layout traits header**

```cpp
#pragma once

namespace kittens::arch::c500 {

template<typename Atom>
struct fragment_layout_traits;

struct mma_bf16_16x16x16_fp32 {
    using a_scalar = bf16;
    using b_scalar = bf16;
    using c_scalar = float;
    static constexpr int M = 16;
    static constexpr int N = 16;
    static constexpr int K = 16;
    static constexpr int wave_size = 64;
};

template<>
struct fragment_layout_traits<mma_bf16_16x16x16_fp32> {
    static __device__ inline int lane_row(int lane) { return lane & 0xf; }
    static __device__ inline int lane_group(int lane) { return lane >> 4; }
};

} // namespace kittens::arch::c500
```

- [ ] **Step 2: Create the native MMA atom header**

```cpp
#pragma once

#include "fragment_layouts.cuh"

namespace kittens::arch::c500 {

template<typename Atom>
struct fragment_a;
template<typename Atom>
struct fragment_b;
template<typename Atom>
struct fragment_c;

template<>
struct fragment_c<mma_bf16_16x16x16_fp32> {
    float reg[4];
};

template<typename Atom>
__device__ inline void mma(fragment_c<Atom>& d,
                           const fragment_a<Atom>& a,
                           const fragment_b<Atom>& b,
                           const fragment_c<Atom>& c);

} // namespace kittens::arch::c500
```

- [ ] **Step 3: Create copy and epilogue stubs**

```cpp
#pragma once

namespace kittens::arch::c500 {

template<typename Atom, typename SharedTile>
__device__ inline void load_a(fragment_a<Atom>& dst, const SharedTile& src, int tile_m, int tile_k);

template<typename Atom, typename SharedTile>
__device__ inline void load_b(fragment_b<Atom>& dst, const SharedTile& src, int tile_k, int tile_n);

template<typename Atom, typename SharedTile>
__device__ inline void store_c(SharedTile& dst, const fragment_c<Atom>& src, int tile_m, int tile_n);

} // namespace kittens::arch::c500
```

- [ ] **Step 4: Wire the new backend headers into common include flow**

Add guarded includes in `include/common/common.cuh`:

```cpp
#ifdef KITTENS_C500
#include "../arch/c500/fragment_layouts.cuh"
#include "../arch/c500/mma_atoms.cuh"
#include "../arch/c500/copy_atoms.cuh"
#include "../arch/c500/epilogue_atoms.cuh"
#endif
```

- [ ] **Step 5: Run a compile-only sanity check**

Run: `make -C tests clean GPU_TARGET=C500 && make -C tests all -j8 GPU_TARGET=C500 COMP_LEVEL=fast NVCC=cucc TEST_INTENSITY=1 TEST_MACROS='-DTEST_GROUP_MMA_WARP -DNUM_GPUS=1'`

Expected: The tree still compiles with the new backend headers present, even though the new C500 backend functions are still stubs.

- [ ] **Step 6: Commit**

```bash
git add include/common/common.cuh include/arch/c500/mma_atoms.cuh include/arch/c500/fragment_layouts.cuh include/arch/c500/copy_atoms.cuh include/arch/c500/epilogue_atoms.cuh
git commit -m "refactor: add c500 backend skeleton"
```

### Task 3: Add Atom-Level C500 Unit Tests

**Files:**
- Create: `tests/c500/mma/atom_bf16.cu`
- Create: `tests/c500/memory/shared_to_native_a.cu`
- Create: `tests/c500/memory/shared_to_native_b.cu`
- Modify: `tests/Makefile`
- Modify: `tests/unit_tests.cu`
- Test: `tests/c500/mma/atom_bf16.cu`

- [ ] **Step 1: Add a focused atom math test**

```cpp
// tests/c500/mma/atom_bf16.cu
#ifdef KITTENS_C500
TEST(c500_mma_atom_bf16, smoke) {
    // Fill one logical 16x16x16 multiply case with deterministic host reference.
    // Launch one-wave kernel that:
    //   1. loads native A/B fragments
    //   2. runs one builtin MMA atom
    //   3. writes native accumulator back to global memory
    // Compare every output element against host reference.
}
#endif
```

- [ ] **Step 2: Add shared-to-native copy tests**

```cpp
// tests/c500/memory/shared_to_native_a.cu
#ifdef KITTENS_C500
TEST(c500_copy_atom_a, smoke) {
    // Stage a known 16x16 A tile in shared memory, load native A fragment,
    // export through a debug bridge, compare against reference.
}
#endif
```

Duplicate the same pattern for `shared_to_native_b.cu`.

- [ ] **Step 3: Register the new tests**

Update `tests/Makefile` and `tests/unit_tests.cu` to compile and dispatch the new `tests/c500/*` files under `GPU_TARGET=C500`.

- [ ] **Step 4: Verify the tests fail for the right reason before implementation**

Run: `make -C tests clean GPU_TARGET=C500 && make -C tests all -j8 GPU_TARGET=C500 COMP_LEVEL=fast NVCC=cucc TEST_INTENSITY=1 TEST_MACROS='-DTEST_C500_NATIVE_ATOMS -DNUM_GPUS=1'`

Expected: Build or runtime fails because the new backend stubs are not implemented yet.

- [ ] **Step 5: Commit**

```bash
git add tests/Makefile tests/unit_tests.cu tests/c500/mma/atom_bf16.cu tests/c500/memory/shared_to_native_a.cu tests/c500/memory/shared_to_native_b.cu
git commit -m "test: add c500 atom and copy contract coverage"
```

### Task 4: Implement The First Native bf16 MMA Atom

**Files:**
- Modify: `include/arch/c500/fragment_layouts.cuh`
- Modify: `include/arch/c500/mma_atoms.cuh`
- Test: `tests/c500/mma/atom_bf16.cu`

- [ ] **Step 1: Define the A/B/C native fragment payloads**

Add concrete storage for the first atom:

```cpp
template<>
struct fragment_a<mma_bf16_16x16x16_fp32> {
    uint32_t reg[2];
};

template<>
struct fragment_b<mma_bf16_16x16x16_fp32> {
    uint32_t reg[2];
};

template<>
struct fragment_c<mma_bf16_16x16x16_fp32> {
    float reg[4];
};
```

- [ ] **Step 2: Implement the builtin wrapper**

```cpp
template<>
__device__ inline void mma(fragment_c<mma_bf16_16x16x16_fp32>& d,
                           const fragment_a<mma_bf16_16x16x16_fp32>& a,
                           const fragment_b<mma_bf16_16x16x16_fp32>& b,
                           const fragment_c<mma_bf16_16x16x16_fp32>& c) {
    // Call the chosen __builtin_mxc_mma_* primitive here.
    // Preserve C500-native register ordering instead of round-tripping through logical rt.
}
```

- [ ] **Step 3: Make the atom test pass**

Run: `make -C tests clean GPU_TARGET=C500 && make -C tests all -j8 GPU_TARGET=C500 COMP_LEVEL=fast NVCC=cucc TEST_INTENSITY=1 TEST_MACROS='-DTEST_C500_NATIVE_ATOMS -DNUM_GPUS=1' && ./tests/unit_tests dump`

Expected: The atom math test passes, even if copy-atom tests still fail.

- [ ] **Step 4: Commit**

```bash
git add include/arch/c500/fragment_layouts.cuh include/arch/c500/mma_atoms.cuh
git commit -m "feat: implement first c500 bf16 mma atom"
```

### Task 5: Implement Shared-To-Native Copy Atoms

**Files:**
- Modify: `include/arch/c500/copy_atoms.cuh`
- Modify: `include/arch/c500/fragment_layouts.cuh`
- Test: `tests/c500/memory/shared_to_native_a.cu`
- Test: `tests/c500/memory/shared_to_native_b.cu`

- [ ] **Step 1: Freeze A and B shared staging assumptions**

Document in code comments:

```cpp
// A shared layout is chosen to minimize wave64 load/shuffle cost for native A fragments.
// B shared layout may differ; do not force A/B to share a staging layout if that increases hot-path repair.
```

- [ ] **Step 2: Implement `load_a`**

```cpp
template<>
__device__ inline void load_a<mma_bf16_16x16x16_fp32>(fragment_a<mma_bf16_16x16x16_fp32>& dst,
                                                       const SharedTile& src,
                                                       int tile_m,
                                                       int tile_k) {
    // Use lds and the frozen native fragment mapping to fill dst.reg[] directly.
}
```

- [ ] **Step 3: Implement `load_b`**

```cpp
template<>
__device__ inline void load_b<mma_bf16_16x16x16_fp32>(fragment_b<mma_bf16_16x16x16_fp32>& dst,
                                                       const SharedTile& src,
                                                       int tile_k,
                                                       int tile_n) {
    // Match the native B fragment contract directly.
}
```

- [ ] **Step 4: Run the copy contract tests**

Run: `make -C tests clean GPU_TARGET=C500 && make -C tests all -j8 GPU_TARGET=C500 COMP_LEVEL=fast NVCC=cucc TEST_INTENSITY=1 TEST_MACROS='-DTEST_C500_NATIVE_ATOMS -DNUM_GPUS=1' && ./tests/unit_tests dump`

Expected: The new copy-atom tests pass, proving that shared staging can feed native fragments without logical `rt` mediation.

- [ ] **Step 5: Commit**

```bash
git add include/arch/c500/copy_atoms.cuh include/arch/c500/fragment_layouts.cuh
git commit -m "feat: add c500 shared to native copy atoms"
```

### Task 6: Dispatch Warp MMA Through The C500 Backend

**Files:**
- Modify: `include/ops/group/mma/warp.cuh`
- Modify: `include/arch/c500/mma_atoms.cuh`
- Test: `tests/group/mma/warp/mma.cu`

- [ ] **Step 1: Add a C500 backend dispatch path in `warp.cuh`**

```cpp
#ifdef KITTENS_C500
// Route bf16/f16 warp MMA through arch::c500::mma atom wrappers.
#else
// Existing non-C500 path unchanged.
#endif
```

- [ ] **Step 2: Keep the public warp-level entrypoint stable where it still helps**

```cpp
template<typename D, typename A, typename B, typename C>
__device__ inline void mma_AB(D& d, const A& a, const B& b, const C& c) {
    // Dispatch to C500 native atom path instead of forcing legacy TK fragment semantics.
}
```

- [ ] **Step 3: Re-run the focused warp MMA tests**

Run: `make -C tests clean GPU_TARGET=C500 && make -C tests all -j8 GPU_TARGET=C500 COMP_LEVEL=fast NVCC=cucc TEST_INTENSITY=1 TEST_MACROS='-DTEST_GROUP_MMA_WARP -DNUM_GPUS=1' && ./tests/unit_tests dump`

Expected: Existing warp MMA coverage still passes or narrows down remaining bridge-only failures.

- [ ] **Step 4: Commit**

```bash
git add include/ops/group/mma/warp.cuh include/arch/c500/mma_atoms.cuh
git commit -m "refactor: dispatch warp mma through c500 backend"
```

### Task 7: Rebuild `bf16_ampere_gemm` Around Native Fragments

**Files:**
- Modify: `kernels/gemm/bf16_ampere/bf16_ampere_gemm.cu`
- Modify: `include/arch/c500/copy_atoms.cuh`
- Modify: `include/arch/c500/epilogue_atoms.cuh`
- Test: `kernels/gemm/bf16_ampere/bf16_ampere_gemm.cu`

- [ ] **Step 1: Replace logical-`rt` hot-path loads with native fragment loads**

```cpp
using atom = kittens::arch::c500::mma_bf16_16x16x16_fp32;
using a_frag = kittens::arch::c500::fragment_a<atom>;
using b_frag = kittens::arch::c500::fragment_b<atom>;
using c_frag = kittens::arch::c500::fragment_c<atom>;

// In the K-loop:
a_frag a_native;
b_frag b_native;
load_a<atom>(a_native, ...);
load_b<atom>(b_native, ...);
mma<atom>(c_native, a_native, b_native, c_native);
```

- [ ] **Step 2: Keep logical layout bridges out of the K-loop**

Add a short comment in the mainloop:

```cpp
// C500 hot path stays in native fragments; logical tile bridges are limited to setup/debug/epilogue boundaries.
```

- [ ] **Step 3: Implement a minimal epilogue atom**

```cpp
template<typename OutputTile>
__device__ inline void store_c(OutputTile& out, const c_frag& acc, int tile_m, int tile_n) {
    // Export native accumulator values to the output tile without reintroducing the legacy fragment path.
}
```

- [ ] **Step 4: Verify correctness on the standalone kernel**

Run: `timeout 60s make -C kernels/gemm/bf16_ampere clean all run GPU=C500`

Expected: The standalone GEMM binary runs and prints a finite error summary against the reference.

- [ ] **Step 5: Commit**

```bash
git add kernels/gemm/bf16_ampere/bf16_ampere_gemm.cu include/arch/c500/copy_atoms.cuh include/arch/c500/epilogue_atoms.cuh
git commit -m "feat: move bf16 ampere gemm onto c500 native fragments"
```

### Task 8: Add C500-Specific Benchmark And Baseline Collection

**Files:**
- Create: `tests/c500/gemm/bf16_gemm_perf.cu`
- Modify: `kernels/gemm/bf16_ampere/bf16_ampere_gemm.cu`
- Modify: `docs/analysis/2026-04-04-c500-mma-builtins-survey.md`
- Test: `tests/c500/gemm/bf16_gemm_perf.cu`

- [ ] **Step 1: Add a perf harness for fixed-shape bf16 GEMM**

```cpp
// tests/c500/gemm/bf16_gemm_perf.cu
// Time 4096^3 with CUDA events, record min/avg/max, and print TFLOPS.
```

- [ ] **Step 2: Record the benchmark protocol**

Append to the analysis doc:

```md
## Benchmark protocol
- Check device health with `mx-smi`
- Use event timing, not host wall-clock
- Fix shape at 4096^3
- Report warmup count, iteration count, min/avg/max, TFLOPS, and ratio vs mcBLAS
```

- [ ] **Step 3: Capture the first native baseline**

Run:

```bash
mx-smi
timeout 120s make -C kernels/gemm/bf16_ampere clean all GPU=C500
timeout 120s make -C kernels/gemm/bf16_ampere run GPU=C500
```

Expected: A stable baseline exists for the native-fragment version before tuning.

- [ ] **Step 4: Commit**

```bash
git add tests/c500/gemm/bf16_gemm_perf.cu kernels/gemm/bf16_ampere/bf16_ampere_gemm.cu docs/analysis/2026-04-04-c500-mma-builtins-survey.md
git commit -m "test: add c500 bf16 gemm performance baseline"
```

### Task 9: Tune Tile Shapes, Pipeline Depth, And Worker Decomposition

**Files:**
- Modify: `kernels/gemm/bf16_ampere/bf16_ampere_gemm.cu`
- Modify: `include/arch/c500/copy_atoms.cuh`
- Modify: `tests/c500/gemm/bf16_gemm_perf.cu`
- Test: `tests/c500/gemm/bf16_gemm_perf.cu`

- [ ] **Step 1: Expose tunable backend policy constants**

```cpp
constexpr int BLOCK_M = /* candidate */;
constexpr int BLOCK_N = /* candidate */;
constexpr int BLOCK_K = /* candidate */;
constexpr int PIPE_STAGES = /* candidate */;
constexpr int NUM_WORKERS = /* candidate */;
```

- [ ] **Step 2: Search a bounded policy grid**

Test at least:

```txt
BLOCK_M/BLOCK_N in {64x64, 128x64, 64x128, 128x128}
BLOCK_K in {16, 32, 64}
PIPE_STAGES in {1, 2, 3}
```

- [ ] **Step 3: Keep the best candidate and record the result**

Run:

```bash
timeout 120s make -C kernels/gemm/bf16_ampere clean all GPU=C500
timeout 120s make -C kernels/gemm/bf16_ampere run GPU=C500
```

Expected: The chosen policy measurably improves TFLOPS over the initial native baseline.

- [ ] **Step 4: Compare against mcBLAS**

Run the repository-local or system-provided `mcBLAS` benchmark for the same `4096^3 bf16` case and record the ratio.

Expected: A quantitative gap is known and the current backend lands in one of these buckets:
- `< 60% mcBLAS`: structure still wrong
- `60%-80% mcBLAS`: native path is viable but not yet mature
- `80%+ mcBLAS`: backend is in the right optimization regime

- [ ] **Step 5: Commit**

```bash
git add kernels/gemm/bf16_ampere/bf16_ampere_gemm.cu include/arch/c500/copy_atoms.cuh tests/c500/gemm/bf16_gemm_perf.cu
git commit -m "perf: tune c500 bf16 gemm backend policy"
```

### Task 10: Reconcile Logical Bridges And Broader TK Tests

**Files:**
- Modify: `include/ops/group/register/vec/conversions.cuh`
- Modify: `include/ops/group/register/vec/reductions.cuh`
- Modify: `include/ops/group/register/vec/maps.cuh`
- Modify: `include/ops/group/register/tile/maps.cuh`
- Modify: `include/ops/group/register/tile/reductions.cuh`
- Test: `tests/unit_tests`

- [ ] **Step 1: Replace ad hoc C500 lane formulas with backend traits usage**

Example direction:

```cpp
// Instead of local wave32-derived shuffle math:
// derive C500 logical/native mapping from arch::c500::fragment_layout_traits<...>.
```

- [ ] **Step 2: Re-run the full test binary**

Run:

```bash
make -C tests clean GPU_TARGET=C500
make -C tests all -j8 GPU_TARGET=C500 COMP_LEVEL=fast NVCC=cucc TEST_INTENSITY=1 TEST_MACROS='-DTEST_ALL -DNUM_GPUS=1'
./tests/unit_tests dump
```

Expected: The broader TK test suite converges because logical bridge layers now consume one unified C500 backend truth.

- [ ] **Step 3: Commit**

```bash
git add include/ops/group/register/vec/conversions.cuh include/ops/group/register/vec/reductions.cuh include/ops/group/register/vec/maps.cuh include/ops/group/register/tile/maps.cuh include/ops/group/register/tile/reductions.cuh
git commit -m "refactor: align tk logical bridges with c500 backend traits"
```

### Task 11: Final Validation And Handoff

**Files:**
- Modify: `docs/analysis/2026-04-04-c500-mma-builtins-survey.md`
- Modify: `README.md`
- Test: `kernels/gemm/bf16_ampere/bf16_ampere_gemm.cu`

- [ ] **Step 1: Capture final validation commands and outcomes**

Run:

```bash
mx-smi
make -C tests clean GPU_TARGET=C500
make -C tests all -j8 GPU_TARGET=C500 COMP_LEVEL=fast NVCC=cucc TEST_INTENSITY=1 TEST_MACROS='-DTEST_ALL -DNUM_GPUS=1'
./tests/unit_tests dump
timeout 120s make -C kernels/gemm/bf16_ampere clean all run GPU=C500
```

Expected: Final logs show the status of correctness coverage and the tuned bf16 GEMM performance path.

- [ ] **Step 2: Document how to use the new backend**

Add a short README or analysis note section:

```md
## C500 native bf16 GEMM
- Uses C500 native MMA backend
- Performance path is backend-specific and does not preserve legacy physical fragment semantics
- High-level ThunderKittens style is preserved
```

- [ ] **Step 3: Commit**

```bash
git add docs/analysis/2026-04-04-c500-mma-builtins-survey.md README.md
git commit -m "docs: record c500 native backend usage and validation"
```

## Self-Review

- Spec coverage: This plan covers the backend contract, new C500 native files, atom/copy tests, native GEMM conversion, performance tuning, and broader TK logical bridge reconciliation.
- Placeholder scan: The plan intentionally leaves the exact builtin symbol and exact best tile policy open to discovery, but every phase states where that decision must be encoded and which command verifies it.
- Type consistency: The same `mma_bf16_16x16x16_fp32` atom, `fragment_a/b/c`, `load_a/load_b/store_c`, and native-first GEMM path names are used throughout the plan.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-04-c500-native-mma-backend.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
