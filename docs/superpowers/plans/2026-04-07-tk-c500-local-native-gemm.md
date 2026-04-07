# TK C500 Local Native GEMM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a local `kernels/gemm/bf16_c500_tk_local/` prototype that preserves the validated C500 muxi-native BF16 GEMM dataflow, correctness, and `~150 TFLOP/s` performance while restructuring it into TK-aligned local layers without depending on external muxi kernel source includes.

**Architecture:** Start by cloning the proven standalone path into a new local prototype directory, then progressively split host layout logic, native contracts, builtin wrappers, and the kernel body into focused local headers. Preserve behavior after every move and only localize one concern at a time so performance regressions can be attributed cleanly.

**Tech Stack:** C++20, MXMACA `cucc`, C500 native builtins, CUDA runtime events, project `kernels/common.mk`, local standalone benchmark code.

---

### Task 1: Create The Local Prototype Skeleton

**Files:**
- Create: `kernels/gemm/bf16_c500_tk_local/Makefile`
- Create: `kernels/gemm/bf16_c500_tk_local/bf16_c500_tk_local_gemm.cu`
- Test: `kernels/gemm/bf16_c500_tk_local/bf16_c500_tk_local_gemm.cu`

- [ ] **Step 1: Copy the current standalone target into the new local directory**

Create `kernels/gemm/bf16_c500_tk_local/Makefile` by mirroring the working target:

```make
GPU ?= C500
SRC := bf16_c500_tk_local_gemm.cu
OUT := bf16_c500_tk_local_gemm.out
CMD := ./bf16_c500_tk_local_gemm.out
CONFIG := standalone
include ../../common.mk

ifeq ($(GPU),C500)
NVCCFLAGS += -mllvm -metaxgpu-disable-bsm-offset=0
NVCCFLAGS += -mllvm -metaxgpu-force-global-saddr=1
endif
```

Create `kernels/gemm/bf16_c500_tk_local/bf16_c500_tk_local_gemm.cu` as a direct behavioral copy of the validated `bf16_c500_muxi_native_gemm.cu`, changing only the namespace, banner text, and local target name.

- [ ] **Step 2: Build the copied target and verify it compiles**

Run:

```bash
make -C kernels/gemm/bf16_c500_tk_local clean
make -C kernels/gemm/bf16_c500_tk_local GPU=C500 -j4 \
  EXTRA_NVCCFLAGS='-DBF16_C500_MUXI_NATIVE_M=256 -DBF16_C500_MUXI_NATIVE_N=256 -DBF16_C500_MUXI_NATIVE_K=256 -DBF16_C500_MUXI_NATIVE_WARMUP_ITERS=1 -DBF16_C500_MUXI_NATIVE_PROFILE_ITERS=1'
```

Expected: successful `cucc` build producing `bf16_c500_tk_local_gemm.out`

- [ ] **Step 3: Run the copied target on a small shape and verify correctness**

Run:

```bash
cd kernels/gemm/bf16_c500_tk_local && ./bf16_c500_tk_local_gemm.out
```

Expected output includes:

```text
abs mean:      0.000000
abs max:       0.000000
err mean:      0.000000
err max:       0.000000
```

- [ ] **Step 4: Commit the skeleton**

```bash
git add kernels/gemm/bf16_c500_tk_local
git commit -m "feat: add local c500 native gemm skeleton"
```

### Task 2: Split Host Layout And Reference Logic

**Files:**
- Create: `kernels/gemm/bf16_c500_tk_local/host/layout_pack.cuh`
- Create: `kernels/gemm/bf16_c500_tk_local/host/reference.cuh`
- Modify: `kernels/gemm/bf16_c500_tk_local/bf16_c500_tk_local_gemm.cu`
- Test: `kernels/gemm/bf16_c500_tk_local/bf16_c500_tk_local_gemm.cu`

- [ ] **Step 1: Move host-side packing helpers into `host/layout_pack.cuh`**

Add:

```c++
#pragma once

#include <maca_bfloat16.h>
#include <cuda_bf16.h>
#include <vector>

namespace bf16_c500_tk_local::host {

using bf16 = __maca_bfloat16;

template<int M, int K>
std::vector<bf16> make_a_native(const std::vector<__nv_bfloat16> &row_major_a);

template<int K, int N>
std::vector<bf16> make_b_native(const std::vector<__nv_bfloat16> &row_major_b);

template<int M, int N>
float load_layoutc_logical(const std::vector<bf16> &raw_c, int row_n, int col_m);

} // namespace bf16_c500_tk_local::host
```

Move the existing validated implementations unchanged.

- [ ] **Step 2: Move reference helpers into `host/reference.cuh`**

Add:

```c++
#pragma once

#include "../common.cuh"

namespace bf16_c500_tk_local::host {

using ::FillMode;
using ::fill;
using ::reference_gemm;

} // namespace bf16_c500_tk_local::host
```

Keep this layer thin in phase 1. The goal is file ownership, not a rewritten reference GEMM.

- [ ] **Step 3: Update the entry translation unit to include and use the host headers**

Replace local helper definitions in `bf16_c500_tk_local_gemm.cu` with:

```c++
#include "host/layout_pack.cuh"
#include "host/reference.cuh"
```

And update call sites to use `bf16_c500_tk_local::host::...`.

- [ ] **Step 4: Rebuild and rerun the small correctness check**

Run:

```bash
make -C kernels/gemm/bf16_c500_tk_local clean
make -C kernels/gemm/bf16_c500_tk_local GPU=C500 -j4 \
  EXTRA_NVCCFLAGS='-DBF16_C500_MUXI_NATIVE_M=256 -DBF16_C500_MUXI_NATIVE_N=256 -DBF16_C500_MUXI_NATIVE_K=256 -DBF16_C500_MUXI_NATIVE_WARMUP_ITERS=1 -DBF16_C500_MUXI_NATIVE_PROFILE_ITERS=1'
cd kernels/gemm/bf16_c500_tk_local && ./bf16_c500_tk_local_gemm.out
```

Expected: build success and zero-error output.

- [ ] **Step 5: Commit the host split**

```bash
git add kernels/gemm/bf16_c500_tk_local
git commit -m "refactor: split local c500 host layout helpers"
```

### Task 3: Add Local Primitive And Contract Headers

**Files:**
- Create: `kernels/gemm/bf16_c500_tk_local/primitives/mxc_builtins.cuh`
- Create: `kernels/gemm/bf16_c500_tk_local/primitives/async_copy.cuh`
- Create: `kernels/gemm/bf16_c500_tk_local/primitives/mma.cuh`
- Create: `kernels/gemm/bf16_c500_tk_local/primitives/sync.cuh`
- Create: `kernels/gemm/bf16_c500_tk_local/contracts/tile_contract.cuh`
- Create: `kernels/gemm/bf16_c500_tk_local/contracts/stage_contract.cuh`
- Create: `kernels/gemm/bf16_c500_tk_local/contracts/layout_contract.cuh`
- Modify: `kernels/gemm/bf16_c500_tk_local/bf16_c500_tk_local_gemm.cu`
- Test: `kernels/gemm/bf16_c500_tk_local/bf16_c500_tk_local_gemm.cu`

- [ ] **Step 1: Create thin primitive wrappers**

Add `primitives/mma.cuh`:

```c++
#pragma once

#include <maca.h>
#include <maca_bfloat16.h>

namespace bf16_c500_tk_local::primitives {

using float4_native = __NATIVE_VECTOR__(4, float);
using uint2_native = __NATIVE_VECTOR__(2, uint);

__device__ __forceinline__ float4_native mma_16x16x16_bf16(uint a0, uint a1, uint b0, uint b1, float4_native c) {
    return __builtin_mxc_mma_16x16x16bf16(uint2_native{a0, a1}, uint2_native{b0, b1}, c);
}

} // namespace bf16_c500_tk_local::primitives
```

Add `primitives/sync.cuh` with thin wrappers for `arrive_gvmcnt` and `__builtin_mxc_barrier_inst`.

Add `primitives/async_copy.cuh` with thin wrappers around the required `__builtin_mxc_ldg_b128_bsm_predicator` forms already used by the validated kernel.

- [ ] **Step 2: Create family contract headers**

Add `contracts/tile_contract.cuh`:

```c++
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
```

Add `contracts/stage_contract.cuh` containing:

```c++
static constexpr int stage_bytes = 0x4000;
static constexpr int a_bank0 = 0x0000;
static constexpr int a_bank1 = 0x1000;
static constexpr int b_bank0 = 0x2000;
static constexpr int b_bank1 = 0x3000;
```

Add `contracts/layout_contract.cuh` containing named comments and constants documenting the validated A/B/C pack-unpack mappings.

- [ ] **Step 3: Include the new headers without changing behavior**

In `bf16_c500_tk_local_gemm.cu`, include the new primitive and contract headers and use constants from them for block size and tile dimensions where this is mechanically safe.

- [ ] **Step 4: Rebuild and rerun `4096^3`**

Run:

```bash
make -C kernels/gemm/bf16_c500_tk_local clean
make -C kernels/gemm/bf16_c500_tk_local GPU=C500 -j4 \
  EXTRA_NVCCFLAGS='-DBF16_C500_MUXI_NATIVE_M=4096 -DBF16_C500_MUXI_NATIVE_N=4096 -DBF16_C500_MUXI_NATIVE_K=4096 -DBF16_C500_MUXI_NATIVE_WARMUP_ITERS=3 -DBF16_C500_MUXI_NATIVE_PROFILE_ITERS=10'
cd kernels/gemm/bf16_c500_tk_local && ./bf16_c500_tk_local_gemm.out
```

Expected: zero-error output and performance in the same band as the known standalone baseline, approximately `149 TFLOP/s` or better.

- [ ] **Step 5: Commit primitive and contract localization**

```bash
git add kernels/gemm/bf16_c500_tk_local
git commit -m "refactor: add local c500 primitive and contract headers"
```

### Task 4: Localize The Kernel Body

**Files:**
- Create: `kernels/gemm/bf16_c500_tk_local/kernel/layoutc_mainloop.cuh`
- Create: `kernels/gemm/bf16_c500_tk_local/kernel/layoutc_epilogue.cuh`
- Modify: `kernels/gemm/bf16_c500_tk_local/bf16_c500_tk_local_gemm.cu`
- Test: `kernels/gemm/bf16_c500_tk_local/bf16_c500_tk_local_gemm.cu`

- [ ] **Step 1: Copy the currently used muxi `layoutC` kernel body into a local kernel header**

Create `kernel/layoutc_mainloop.cuh` and move in the actual kernel body that is currently being included indirectly from the external muxi source. Preserve issue order, register arrays, stage offsets, and builtin usage.

The local file should expose one entry:

```c++
namespace bf16_c500_tk_local::kernel {

template <typename T, typename Tc, typename Tscal, bool IsBetaZero, bool HasOneDimBias>
__global__ void layout_hgemm_tn_128x128x128_4m1n8k_256t_layoutc(...);

} // namespace bf16_c500_tk_local::kernel
```

- [ ] **Step 2: Split epilogue-only helpers into `layoutc_epilogue.cuh` if needed**

If store/export logic is separable without changing order, move only the export helper code into:

```c++
namespace bf16_c500_tk_local::kernel {

template <typename T, typename Tc, typename Tscal, bool IsBetaZero, bool HasOneDimBias>
__device__ __forceinline__ void layoutc_store_epilogue(...);

} // namespace bf16_c500_tk_local::kernel
```

If the split complicates the initial localization, keep the entire kernel in `layoutc_mainloop.cuh` and leave `layoutc_epilogue.cuh` as a small header reserved for the next refactor.

- [ ] **Step 3: Remove direct inclusion of external muxi kernel source**

Replace:

```c++
#include "/data/muxi_native_layout_kernels/csrc/muxi_hgemm_layoutC.cuh"
```

with:

```c++
#include "kernel/layoutc_mainloop.cuh"
```

And update the launch site to call the local namespace entry.

- [ ] **Step 4: Rebuild and rerun both correctness and performance checks**

Run:

```bash
make -C kernels/gemm/bf16_c500_tk_local clean
make -C kernels/gemm/bf16_c500_tk_local GPU=C500 -j4 \
  EXTRA_NVCCFLAGS='-DBF16_C500_MUXI_NATIVE_M=256 -DBF16_C500_MUXI_NATIVE_N=256 -DBF16_C500_MUXI_NATIVE_K=256 -DBF16_C500_MUXI_NATIVE_WARMUP_ITERS=1 -DBF16_C500_MUXI_NATIVE_PROFILE_ITERS=1'
cd kernels/gemm/bf16_c500_tk_local && ./bf16_c500_tk_local_gemm.out
make -C /data/ThunderKittens/kernels/gemm/bf16_c500_tk_local clean
make -C /data/ThunderKittens/kernels/gemm/bf16_c500_tk_local GPU=C500 -j4 \
  EXTRA_NVCCFLAGS='-DBF16_C500_MUXI_NATIVE_M=4096 -DBF16_C500_MUXI_NATIVE_N=4096 -DBF16_C500_MUXI_NATIVE_K=4096 -DBF16_C500_MUXI_NATIVE_WARMUP_ITERS=3 -DBF16_C500_MUXI_NATIVE_PROFILE_ITERS=10'
cd /data/ThunderKittens/kernels/gemm/bf16_c500_tk_local && ./bf16_c500_tk_local_gemm.out
```

Expected: zero errors and performance staying in the validated native range.

- [ ] **Step 5: Commit kernel localization**

```bash
git add kernels/gemm/bf16_c500_tk_local
git commit -m "refactor: localize c500 layoutc native kernel"
```

### Task 5: Clean The Entry File Into A Thin Wiring Layer

**Files:**
- Modify: `kernels/gemm/bf16_c500_tk_local/bf16_c500_tk_local_gemm.cu`
- Test: `kernels/gemm/bf16_c500_tk_local/bf16_c500_tk_local_gemm.cu`

- [ ] **Step 1: Reduce the `.cu` entry file to wiring only**

Keep in `bf16_c500_tk_local_gemm.cu` only:

- includes
- benchmark constants
- device allocation and copies
- launch lambda
- timing
- output checking
- `main()`

Remove any remaining duplicated helper bodies now owned by `host/`, `contracts/`, `primitives/`, or `kernel/`.

- [ ] **Step 2: Rebuild and rerun the final `4096^3` benchmark**

Run:

```bash
make -C kernels/gemm/bf16_c500_tk_local clean
make -C kernels/gemm/bf16_c500_tk_local GPU=C500 -j4 \
  EXTRA_NVCCFLAGS='-DBF16_C500_MUXI_NATIVE_M=4096 -DBF16_C500_MUXI_NATIVE_N=4096 -DBF16_C500_MUXI_NATIVE_K=4096 -DBF16_C500_MUXI_NATIVE_WARMUP_ITERS=3 -DBF16_C500_MUXI_NATIVE_PROFILE_ITERS=10'
cd kernels/gemm/bf16_c500_tk_local && ./bf16_c500_tk_local_gemm.out
```

Expected:

```text
abs mean:      0.000000
abs max:       0.000000
err mean:      0.000000
err max:       0.000000
Performance:   ~150 TFLOP/s
```

- [ ] **Step 3: Commit the local prototype phase**

```bash
git add kernels/gemm/bf16_c500_tk_local docs/superpowers/specs/2026-04-07-tk-c500-local-native-gemm-design.md docs/superpowers/plans/2026-04-07-tk-c500-local-native-gemm.md
git commit -m "feat: add local c500 native gemm prototype"
```
