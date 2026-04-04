# C500 MMA Builtins Survey for ThunderKittens

## Scope

This note summarizes the MMA-related `__builtin_mxc_*` usage that matters for the current C500 bring-up of ThunderKittens, with the immediate target still being `kernels/gemm/bf16_ampere/bf16_ampere_gemm.cu`.

This document follows the project rule that C500 adaptation must stay inside ThunderKittens abstractions and internals:

- do not replace ThunderKittens kernels with external GEMM APIs
- do not add helper fallback GEMM kernels
- when CUDA/PTX primitives are unavailable, replace them with C500-compatible ThunderKittens-internal implementations built from supported language features, compiler builtins, and runtime functionality

## Source Set

Primary local sources used for this survey:

- `/opt/maca/include/cute/arch/mma_sm80.hpp`
- `/opt/maca/include/mctlass/arch/mma_sm75.h`
- `/opt/maca/include/mctlass/arch/mma_sm80.h`
- `/opt/maca/include/mcflashinfer/mma.cuh`
- `/data/muxi_native_layout_kernels/csrc/utils.cuh`
- `/data/muxi_native_layout_kernels/csrc/muxi_hgemm_utils.cuh`
- `/data/muxi_native_layout_kernels/csrc/muxi_hgemm_layoutC.cuh`

Referenced public repos for cross-checking intent and provenance:

- <https://github.com/MetaX-MACA/McFlashInfer>
- <https://github.com/MetaX-MACA/mcoplib>

## Executive Summary

There are two distinct builtin usage styles in the MetaX/MACA ecosystem, and ThunderKittens adaptation must not mix them accidentally:

1. `native-layout direct builtin`
   - Used in `mctlass`, `mcflashinfer` fallback paths, and `muxi_native_layout_kernels`.
   - The caller already stores A/B/C fragments in the builtin's native wave64-friendly layout.
   - The code can directly call `__builtin_mxc_mma_16x16x16f16(...)` or `__builtin_mxc_mma_16x16x16bf16(...)`.

2. `SM80-compat wrapper over wave64`
   - Used in `cute/arch/mma_sm80.hpp` under `__MACA_ARCH__`.
   - The caller still speaks CUDA SM80 logical warp-fragment layout, so the implementation must:
     - gather/repack operands across a full wave64 with `__shfl_down_sync`
     - call the builtin
     - scatter results back to the SM80 logical fragment layout

ThunderKittens today is much closer to case 2 for `warp mma`: its register tiles and `hmma16816` interfaces are modeled after CUDA SM80 warp-level semantics, not the native MetaX fragment layout used by `mctlass`/`mcflashinfer` direct-builtin code.

For the current bring-up, that mismatch is evidence of where ThunderKittens still carries SM80-era assumptions, not a recommendation to freeze the backend around the compatibility wrapper. The performance-first backend direction is to keep ThunderKittens abstractions at the API boundary while moving the hot path toward the native-layout direct-builtin model evidenced by `mctlass`, `mcflashinfer`, and `muxi_native_layout_kernels`; `cute/arch/mma_sm80.hpp` remains a valuable reference for understanding the cost and mechanics of bridging from SM80 logical fragments when a diagnostic or temporary translation layer is unavoidable.

## Builtin Inventory

### Core MMA builtins

These are the primary MMA builtins observed across the surveyed code:

| Builtin | Observed meaning | Operand shape seen in code | Accumulator shape | Current TK relevance |
| --- | --- | --- | --- | --- |
| `__builtin_mxc_mma_16x16x16f16` | fp16 MMA | 4 half values for A, 4 half values for B | 4 floats or 4 halfs | Required |
| `__builtin_mxc_mma_16x16x16bf16` | bf16 MMA | 4 bf16-like values for A, 4 bf16-like values for B | 4 floats or 4 bf16-like values | Required |
| `__builtin_mxc_mma_16x16x16i8` | int8 MMA | native packed integer fragments | 4 ints | Not needed now |
| `__builtin_mxc_mma_16x16x8tf32` | tf32 MMA | tf32-packed fragments | 4 floats | Not needed now |
| `__builtin_mxc_mma_16x16x4f32` | fp32 MMA | fp32 packed fragments | 4 floats | Not needed now |
| `__builtin_mxc_mma_16x16x4f64` | fp64 MMA | fp64 packed fragments | 4 doubles | Not needed now |

Observed call patterns:

- `mctlass` direct style:
  - `__builtin_mxc_mma_16x16x16f16(b.to_macahalf4(), a.to_macahalf4(), c.to_macafloat4())`
  - `__builtin_mxc_mma_16x16x16bf16(b.to_macahalf4(), a.to_macahalf4(), c.to_macafloat4())`
- `mcflashinfer` fallback style:
  - `__builtin_mxc_mma_16x16x16f16(b, a, {C[0], C[1], C[2], C[3]})`
  - `__builtin_mxc_mma_16x16x16bf16(b, a, {C[0], C[1], C[2], C[3]})`
- `muxi_native_layout_kernels` style:
  - `__builtin_mxc_mma_16x16x16f16(A, B, C)`
  - `__builtin_mxc_mma_16x16x16bf16(A, B, C)`

Important observation:

- The apparent A/B order is not uniform at the wrapper call site.
- Some wrappers pass `(b, a, c)`, some pass `(A, B, C)`, and some expose `SwapAB`.
- Therefore, ThunderKittens should not infer builtin operand semantics from variable names alone.
- The correct interpretation must come from the surrounding fragment layout contract.

### Adjacent builtins and intrinsics relevant to bring-up

These are not all MMA builtins, but they repeatedly appear around native MetaX GEMM kernels and are likely to matter for later stages of ThunderKittens adaptation:

| Primitive | Observed role | Current TK relevance |
| --- | --- | --- |
| `__lane_id()` | hardware lane id in wave64 | Required for wave64 wrapper code |
| `__shfl_down_sync(...)` | cross-lane fragment gather/scatter | Required for SM80-compatible wrapper path |
| `__builtin_mxc_readfirstlane(...)` | broadcast lane value / derive warp-wide constants | Likely needed later |
| `__builtin_mxc_arrive_gvmcnt(...)` | async global-memory completion tracking | Possible later |
| `__builtin_mxc_arrive_bsmcnt(...)` | async shared-memory or BSM completion tracking | Possible later |
| `__builtin_mxc_ldg_b128_bsm_predicator(...)` | predicated async/global load path | Possible later |
| `__builtin_mxc_ldg_b64_bsm_predicator(...)` | narrower predicated async/global load path | Possible later |
| `__builtin_mxc_load_global_async64(...)` | async 64-bit global load | Possible later |

For the current `bf16_ampere_gemm` bring-up, the first wave of work should stay focused on:

- `__builtin_mxc_mma_16x16x16bf16`
- `__builtin_mxc_mma_16x16x16f16`
- `__lane_id()`
- `__shfl_down_sync(...)`

## What the SDK shows about wave64 MMA layout

### 1. `cute` SM80 compatibility path is explicitly wave64-aware

In `/opt/maca/include/cute/arch/mma_sm80.hpp`, the `__MACA_ARCH__` implementation of:

- `SM80_16x8x16_F32F16F16F32_TN`
- `SM80_16x8x16_F32BF16BF16F32_TN`

does not directly feed the caller's fragments into the builtin.

Instead it:

1. reads the hardware lane id with `__lane_id()`
2. computes shuffle deltas from that lane id
3. repacks A and B across 64 lanes
4. repacks C across 64 lanes
5. calls the builtin
6. shuffles D back
7. only materializes the logical SM80 outputs on lanes `< 32`

This is the clearest evidence that:

- C500 hardware executes this builtin with wave64 semantics
- but a CUDA SM80-like `m16n8k16` warp API still expects a different logical fragment layout

### 2. The `cute` shuffle formulas are the best evidence for SM80-to-wave64 translation mechanics

The A/B repack pattern used by `cute` is:

```cpp
delta = (lane_id % 32 % 8 * 4 + lane_id % 32 / 16 * 2) - lane_id;
```

The C repack pattern is:

```cpp
delta = (lane_id % 8 / 2 + lane_id % 32 / 16 * 16) - lane_id;
```

The D writeback repack pattern is:

```cpp
delta = (lane_id % 4 * 2 + lane_id / 16 * 16) - lane_id;
```

These formulas are the clearest evidence for how an SM80-style logical fragment wrapper repacks data across a wave64. They should inform probe design and any temporary translation layer, but they are not the native backend contract.

### 3. Only lower 32 lanes own the logical SM80 result

In the `cute` SM80 wrapper, the final result assignment is guarded by:

```cpp
if (lane_id < 32) { ... }
```

This implies:

- wave64 participates in the builtin execution
- but the exposed logical SM80 fragment still behaves like a 32-thread warp fragment

This distinction is easy to lose if we globally equate:

- hardware `wave = 64`
- ThunderKittens logical `warp fragment = 64`

Those are not automatically the same abstraction.

## What `mctlass` shows

### 1. `mctlass` exposes native direct-builtin contracts

In `/opt/maca/include/mctlass/arch/mma_sm80.h`, the `GemmShape<16,16,16>` specializations directly call:

- `__builtin_mxc_mma_16x16x16f16(...)`
- `__builtin_mxc_mma_16x16x16bf16(...)`

with `FragmentA = Array<half_t, 4>` or `Array<bfloat16_t, 4>`, `FragmentB = Array<... ,4>`, and `FragmentC = Array<float, 4>` or half/bfloat variants.

This is useful because it tells us the native builtin-facing fragment contract is naturally expressed as:

- 4 scalar 16-bit A values
- 4 scalar 16-bit B values
- 4 accumulator scalars

### 2. `mctlass` also shows shape lifting

In `/opt/maca/include/mctlass/arch/mma_sm75.h`, a `GemmShape<16,8,8>` fp16 path is lowered to:

```cpp
__builtin_mxc_mma_16x16x16f16(
  {a[0], a[1], 0, 0},
  {b[0], b[1], 0, 0},
  {c[0], c[1], c[2], c[3]}
)
```

So at least one higher-level abstraction on MetaX is implemented by embedding a smaller logical MMA shape into the native `16x16x16` builtin contract.

That is relevant to ThunderKittens because it confirms that:

- the builtin itself is the primitive
- higher-level CUDA-like shapes may need explicit packing or padding logic

## What `mcflashinfer` shows

### 1. Direct builtin use appears in fallback/native paths

In `/opt/maca/include/mcflashinfer/mma.cuh`, fallback paths use:

- `__builtin_mxc_mma_16x16x16f16(b, a, {C[0], C[1], C[2], C[3]})`
- `__builtin_mxc_mma_16x16x16bf16(b, a, {C[0], C[1], C[2], C[3]})`

without the `cute`-style wave64 shuffle bridge.

That strongly suggests those code paths already hold fragments in the native builtin layout, not the CUDA SM80 logical layout.

### 2. This is why copying `mcflashinfer` directly into ThunderKittens would be incorrect

`mcflashinfer` is valuable as a builtin usage inventory source, but not as a drop-in semantic template for ThunderKittens `warp.cuh`.

Reason:

- ThunderKittens `hmma16816` API shape models CUDA warp-fragment semantics
- `mcflashinfer` direct-builtin code models native builtin-facing fragment semantics

So the two layers sit at different abstraction levels.

## What `muxi_native_layout_kernels` shows

### 1. The project is explicitly native-layout and wave64-centric

In `/data/muxi_native_layout_kernels/csrc/utils.cuh`:

- `WARP_SIZE` is defined as `64`
- wrapper code directly returns `__builtin_mxc_mma_16x16x16f16(A, B, C)` or `__builtin_mxc_mma_16x16x16bf16(A, B, C)`

In `/data/muxi_native_layout_kernels/csrc/muxi_hgemm_utils.cuh`:

- `mma_16x16x16b16<T, SwapAB>` exposes direct builtin access
- `SwapAB` exists because operand ordering must match the surrounding native layout, not a universal symbolic A/B convention

### 2. Layout kernels are useful as native-layout references, not as SM80 wrappers

Files such as:

- `/data/muxi_native_layout_kernels/csrc/muxi_hgemm_layoutC.cuh`

show explicit wave64-style indexing:

- `slot = tid / 64`
- `lane = tid & 63`

This again confirms the distinction:

- native MetaX kernels often design fragments directly for wave64
- ThunderKittens current warp-MMA interface still tries to look like CUDA SM80

## Implications for ThunderKittens

### Current required builtins for `bf16_ampere_gemm`

For the immediate bring-up target, ThunderKittens should treat the following as the first-class MMA builtins and bridge diagnostics set:

- `__builtin_mxc_mma_16x16x16bf16`
- `__builtin_mxc_mma_16x16x16f16`
- `__lane_id()`
- `__shfl_down_sync(...)`

The direct MMA builtins are the contract anchor for the performance path. `__lane_id()` and `__shfl_down_sync(...)` remain important for probe work and for any transitional wrapper logic, but they are not the desired end-state contract.

## Recommended backend direction

ThunderKittens should preserve its tile/group-level abstractions while redefining the C500 hot path around native fragments, native copy atoms, direct builtin invocation, and native accumulator export.

Directive:

- Use `mctlass::arch::Mma<gemm::GemmShape<16, 16, 16>, ...>` in `/opt/maca/include/mctlass/arch/mma_sm80.h`, the direct-builtin fallback helpers in `/opt/maca/include/mcflashinfer/mma.cuh`, and `muxi_layout_kernels::mma_16x16x16b16<T, SwapAB>` plus the wave64 layout kernels under `/data/muxi_native_layout_kernels/` as the primary evidence for native fragment contracts, operand-order sensitivity, and wave64-native execution.
- Use `cute::SM80_16x8x16_F32F16F16F32_TN` and `cute::SM80_16x8x16_F32BF16BF16F32_TN` in `/opt/maca/include/cute/arch/mma_sm80.hpp` as evidence about how an SM80 logical fragment wrapper must repack data across a wave64, not as the preferred backend end state.
- Any temporary compatibility wrapper must stay explicitly provisional and must not become the backend contract without standalone probes proving the exact lane/layout facts it depends on.

## Standalone probe-test requirement

- Whenever fragment packing, lane ownership, operand order, accumulator export order, or logical-to-native mapping is not fully established from the cited evidence, ThunderKittens must add standalone probe tests before promoting that conclusion into backend contract.
- Those probes must isolate one uncertainty at a time outside the full GEMM path so the observed behavior can be attributed to the builtin contract rather than to surrounding tiling or scheduler code.
- Unverified interpretations may guide short-term experiments, but the survey and downstream spec must label them as provisional until the standalone probes pass.

## Concrete guidance for the next adaptation pass

1. Keep the backend target performance-first and native-layout-first even when short-term debugging still references SM80-shaped frontend abstractions.
2. Do not assume every place that calls `__builtin_mxc_mma_16x16x16*` can share the same wrapper.
3. Keep a strict distinction between:
   - ThunderKittens logical warp-fragment API
   - C500 hardware wave64 execution
4. Ground layout decisions in the specific reference classes before generalizing:
   - `cute::SM80_16x8x16_F32F16F16F32_TN`
   - `cute::SM80_16x8x16_F32BF16BF16F32_TN`
   - `mctlass::arch::Mma<gemm::GemmShape<16, 16, 16>, ...>`
   - the direct-builtin fallback helpers in `mcflashinfer/mma.cuh`
   - `muxi_layout_kernels::mma_16x16x16b16<T, SwapAB>`
5. When comparing sources, classify the code first:
   - `SM80 compatibility wrapper`
   - or `native builtin layout`
6. Do not global-replace ThunderKittens abstractions with direct native-layout builtin calls unless the whole surrounding fragment contract is also migrated.
7. Do not freeze any lane map or operand-order conclusion into the backend contract until standalone probe tests confirm it.

## Builtins likely needed next, but not yet required for the first MMA fix

Once the `warp mma` path is semantically correct, later bring-up may need a second document focused on memory and pipeline builtins:

- `__builtin_mxc_readfirstlane`
- `__builtin_mxc_arrive_gvmcnt`
- `__builtin_mxc_arrive_bsmcnt`
- `__builtin_mxc_ldg_b128_bsm_predicator`
- `__builtin_mxc_ldg_b64_bsm_predicator`
- `__builtin_mxc_load_global_async64`

Those appear repeatedly in native MetaX GEMM implementations, but they are not the first blocker for ThunderKittens `bf16_ampere_gemm`.

## Bottom Line

For ThunderKittens C500 bring-up, `__builtin_mxc_mma_16x16x16f16` and `__builtin_mxc_mma_16x16x16bf16` should currently be understood in two layers:

- as native wave64-facing tensor-core-like primitives
- and, separately, as backends underneath a CUDA-SM80-compatible logical fragment wrapper

The backend contract should be written around the first interpretation. The second interpretation is diagnostic or transitional only unless standalone probes prove that a specific wrapper mapping is both correct and still required at the ThunderKittens abstraction boundary.

## Transient debugging context from ThunderKittens (2026-04-04)

This section records a dated debugging snapshot from the bring-up effort. It is useful for explaining why the current implementation still fails, but it is not source-backed stable guidance and must not override the evidence- and probe-gated backend direction above.

After restructuring `include/ops/group/mma/warp.cuh` toward a more `cute`-like internal helper layout, the minimal C500 warp-MMA unit test still fails uniformly, but with a very informative pattern:

- `warp_mma_AB_[1x1]x1` writes only the upper half of the `16x16` output tile
- the lower half of the output is all zeros
- the nonzero upper half is also numerically wrong, but it is clearly not random garbage

This strongly suggests the immediate blocker is not "the shuffle formulas are missing", but rather:

- the `cute` SM80-compat wrapper assumes full wave64 execution participation
- while ThunderKittens register tile ownership and warp-scope execution model are still organized around `32` logical lanes

Relevant local evidence:

- `cute/arch/mma_sm80.hpp` explicitly gathers/scatters across 64 lanes
- ThunderKittens `rt_base` still hardcodes:

```cpp
static constexpr int elements_per_thread = num_elements / 32;
```

Implication:

- a `cute`-style wrapper can be structurally correct in isolation
- but it still cannot become semantically correct until ThunderKittens cleanly separates:
  - execution wave width on C500
  - logical fragment ownership of the SM80-style warp API

This is the next core gap to close.

## Frozen first-wave backend contract

- Logical `rt/rv` are not the C500 hot-path fragment contract. They may survive as compatibility-facing abstractions, but the backend hot path must not round-trip through logical `rt/rv` fragments.
- The frozen performance path is `global -> shared staging -> native fragment -> builtin mma -> native accumulator -> epilogue/store`.
- The backend owns native fragment layout, shared-to-native copy atoms, builtin invocation, and native-accumulator export so that ThunderKittens frontend code continues to see ThunderKittens-style tiles rather than hardware-specific fragment details.
- Primary atom: bf16/f16 inputs with fp32 accumulation, treated as the source of truth for first-wave layout and scalar packing.
- Interface evolution is allowed, and expected, when it improves performance while preserving ThunderKittens-style abstractions at the tile/group boundary and avoiding leakage of low-level hardware minutiae.
- This contract freezes the performance path and ownership boundaries only. Any unresolved per-lane packing, operand-order rule, or accumulator export map remains provisional until standalone probe tests validate it against the evidence sources above.
