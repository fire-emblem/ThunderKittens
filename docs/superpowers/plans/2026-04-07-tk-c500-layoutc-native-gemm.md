# TK C500 LayoutC Native GEMM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land a C500-native `layoutC` BF16 GEMM family in ThunderKittens and wire it into the `bf16_c500` example for `4096^3` benchmarking.

**Architecture:** Port the effective muxi `layoutC` kernel hierarchy into a TK-owned C500 family, keep native `A/B` layout preprocessing in the example path, and dispatch the benchmark through this family without trying to generalize the whole GEMM framework first.

**Tech Stack:** ThunderKittens C++ kernel code, MXMACA/C500 builtins, local BF16 GEMM example harness, C500 runtime.

---

### Task 1: Add the dedicated C500 layoutC family

**Files:**
- Create: `include/arch/c500/gemm/families/bf16_c500_layoutc_128x128x128_stage4.cuh`
- Modify: `include/arch/c500/gemm/dispatch/bf16_dispatch.cuh`

- [ ] Add the family file with the fixed `128x128x128`, `Stage=4`, `256-thread` kernel body.
- [ ] Expose one dispatch entry for the new family without disturbing the existing transitional families.

### Task 2: Wire native-layout preprocessing into the bf16_c500 example

**Files:**
- Modify: `kernels/gemm/bf16_c500/bf16_c500_gemm.cu`

- [ ] Add host-side `A` native layout repack.
- [ ] Add host-side `B` native layout repack matching muxi `layoutB`.
- [ ] Add a launch path for the new family and preserve the reference GEMM check.

### Task 3: Build and benchmark the example path

**Files:**
- Modify: `kernels/gemm/bf16_c500/bf16_c500_gemm.cu`

- [ ] Compile `kernels/gemm/bf16_c500`
- [ ] Run the default BF16 benchmark on C500
- [ ] Use the measured result as the next optimization baseline
