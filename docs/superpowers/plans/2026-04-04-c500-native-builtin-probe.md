# C500 Native Builtin Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone C500 BF16 native-builtin performance probe that mimics a pipeline-style kernel using raw `mxc` builtins and measures device-event throughput without requiring correctness.

**Architecture:** Create a new independent benchmark under `kernels/gemm/` that does not depend on ThunderKittens GEMM abstractions. Use raw shared memory, `__builtin_mxc_ldg_b128_bsm_predicator`, `__builtin_mxc_mma_16x16x16bf16`, and `__builtin_mxc_arrive_gvmcnt/__builtin_mxc_barrier_inst` to approximate a pipeline-style producer/consumer loop and report BF16 TFLOP/s.

**Tech Stack:** C++20, MXMACA `cucc`, CUDA runtime events, C500 native builtins.

---

### Task 1: Add the standalone target

**Files:**
- Create: `kernels/gemm/bf16_c500_native_builtin_probe/Makefile`
- Create: `kernels/gemm/bf16_c500_native_builtin_probe/bf16_c500_native_builtin_probe.cu`

- [ ] **Step 1: Verify the target does not exist**

Run: `make -C kernels/gemm/bf16_c500_native_builtin_probe`
Expected: FAIL with `No such file or directory`

- [ ] **Step 2: Add the standalone build target**

Create a Makefile that builds `bf16_c500_native_builtin_probe.out` with `GPU ?= C500`.

- [ ] **Step 3: Add the native builtin benchmark**

Implement a pipeline-style BF16 kernel using:
- raw global buffers for A/B
- shared-memory stage ring
- `__builtin_mxc_ldg_b128_bsm_predicator`
- `__builtin_mxc_mma_16x16x16bf16`
- `__builtin_mxc_arrive_gvmcnt` and `__builtin_mxc_barrier_inst`

The benchmark only needs performance output and a sink to prevent dead-code elimination.

- [ ] **Step 4: Build the target**

Run: `make -C kernels/gemm/bf16_c500_native_builtin_probe clean all GPU=C500 NVCC=cucc`
Expected: PASS and produce `bf16_c500_native_builtin_probe.out`

### Task 2: Run and summarize results

**Files:**
- Modify: `kernels/gemm/bf16_c500_native_builtin_probe/bf16_c500_native_builtin_probe.cu`

- [ ] **Step 1: Run the native probe**

Run: `./kernels/gemm/bf16_c500_native_builtin_probe/bf16_c500_native_builtin_probe.out`
Expected: PASS and print device-event runtime and BF16 TFLOP/s

- [ ] **Step 2: Fix any compile/runtime issues minimally**

Keep fixes scoped to the native probe.

- [ ] **Step 3: Report the measured upper bound**

Summarize the benchmark environment, timing boundary, and measured native-builtin throughput.
