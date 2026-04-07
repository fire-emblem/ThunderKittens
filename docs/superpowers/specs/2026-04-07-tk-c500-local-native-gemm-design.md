# TK C500 Local Native GEMM Design

## Goal

Add a new local prototype directory under `kernels/gemm/` that hosts a C500-native BF16 GEMM implementation in a ThunderKittens-aligned structure, while remaining independent from the current TK backend, dispatch, and family integration layers.

The immediate target is a standalone `4096^3` BF16 GEMM path that:

- preserves the proven muxi-style native dataflow
- preserves correctness
- preserves the current `~150 TFLOP/s` performance level on C500
- removes direct dependence on external muxi kernel source includes

This prototype is intended to become the source of truth for later reintegration into ThunderKittens proper.

## Non-Goals

This design does not attempt to:

- integrate with existing TK `include/arch/c500` dispatch or family selection
- preserve Ampere or warp32 compatibility semantics
- generalize across arbitrary tile shapes in the first phase
- introduce an external library GEMM call as the hot path
- add a fallback CUDA GEMM implementation

## Constraints

The design follows project priority rules:

- C500 adaptation must remain inside project-owned abstractions and internals
- unsupported CUDA/PTX-specific pieces must be replaced by project-owned C500-compatible implementations
- no external GEMM API substitution in the final TK path
- no separate fallback GEMM path as the adaptation answer

For this local prototype phase, the implementation may live outside current TK backend headers, but it must still be structured so it can be migrated into project-owned backend layers later.

## Scope

Phase 1 scope is intentionally narrow:

- datatype: BF16
- architecture: C500
- tile family: native `layoutC`, `128x128x128`, `stage=4`, `256 threads`
- benchmark focus: `M=N=K=4096`
- standalone benchmark executable under `kernels/gemm/`

The implementation should also preserve small-shape correctness testing for layout contracts so that future refactors do not silently break packing or unpacking.

## Directory Structure

Create a new directory:

- `kernels/gemm/bf16_c500_tk_local/`

Initial file structure:

- `kernels/gemm/bf16_c500_tk_local/Makefile`
- `kernels/gemm/bf16_c500_tk_local/bf16_c500_tk_local_gemm.cu`
- `kernels/gemm/bf16_c500_tk_local/primitives/mxc_builtins.cuh`
- `kernels/gemm/bf16_c500_tk_local/primitives/async_copy.cuh`
- `kernels/gemm/bf16_c500_tk_local/primitives/mma.cuh`
- `kernels/gemm/bf16_c500_tk_local/primitives/sync.cuh`
- `kernels/gemm/bf16_c500_tk_local/contracts/tile_contract.cuh`
- `kernels/gemm/bf16_c500_tk_local/contracts/stage_contract.cuh`
- `kernels/gemm/bf16_c500_tk_local/contracts/layout_contract.cuh`
- `kernels/gemm/bf16_c500_tk_local/kernel/layoutc_mainloop.cuh`
- `kernels/gemm/bf16_c500_tk_local/kernel/layoutc_epilogue.cuh`
- `kernels/gemm/bf16_c500_tk_local/host/layout_pack.cuh`
- `kernels/gemm/bf16_c500_tk_local/host/reference.cuh`

## Architectural Model

The prototype should mirror the native C500 execution model, not Ampere compatibility assumptions.

### 1. Primitive Layer

The primitive layer owns only thin wrappers around C500-native compiler builtins and synchronization semantics.

Responsibilities:

- async `global -> shared` issue wrappers
- predicated async copy wrappers
- barrier and outstanding-count helpers
- BF16 MMA wrappers over `__builtin_mxc_mma_*`
- small native vector helper aliases

Rules:

- no tile semantics
- no scheduler policy
- no host-side layout logic

This layer is intended to become the later `include/arch/c500/primitives/*` source.

### 2. Contract Layer

The contract layer owns physical facts of the chosen GEMM family.

Responsibilities:

- CTA tile shape
- wave64 ownership model
- `stage=4` ring geometry
- shared-memory bank offsets
- A native layout contract
- B native layout contract
- C layout/export contract

This layer must encode the exact contracts already validated in the standalone muxi-native path:

- A logical row-major `A[M,K]` is packed into the native A layout
- B logical row-major `B[N,K]` is packed into the native B layout used by `layoutB`
- raw C output is interpreted through the muxi `layoutC` logical mapping, not through a simple row-major or transpose assumption

### 3. Kernel Layer

The kernel layer owns the actual device-side GEMM body for this one family.

Responsibilities:

- prologue preload
- stage-ring setup
- bank-resident register loads
- steady-state MMA issue sequence
- epilogue/export sequence

The first implementation should remain very close to the proven muxi-style schedule so that the local prototype preserves performance. This phase should avoid abstraction that changes issue order or register-bank semantics.

The split between `layoutc_mainloop.cuh` and `layoutc_epilogue.cuh` is organizational, not a requirement to create a generic framework.

### 4. Host Layer

The host layer owns:

- input generation
- A native packing
- B native packing
- raw C logical decode for correctness checking
- reference GEMM
- benchmark timing and reporting

This layer must retain the current corrected contracts:

- B is logically `[N,K]`
- reference output is `[N,M]`
- raw `layoutC` output is decoded via the validated 5D view/permute equivalent mapping

## Build Configuration

The local prototype must build through the existing `kernels/common.mk` standalone flow, but its target-local `Makefile` must also append the C500 options required to match muxi performance:

- `-mllvm -metaxgpu-disable-bsm-offset=0`
- `-mllvm -metaxgpu-force-global-saddr=1`

These flags are part of the validated native path contract and must not depend on callers remembering to pass them manually.

## Validation Requirements

The prototype is complete for phase 1 only if all of the following hold:

1. It compiles as a standalone target under `kernels/gemm/`.
2. Small-shape correctness passes with zero observed error for the current deterministic inputs.
3. `4096^3` BF16 runs correctly.
4. The reported performance remains in the same band as the proven standalone baseline, approximately `150 TFLOP/s`.
5. The code no longer includes the external muxi kernel source file directly.

## Implementation Strategy

The migration should happen in small steps:

1. Copy the existing validated standalone target into the new local directory.
2. Preserve behavior while moving host-side packing and checking into `host/`.
3. Localize the kernel code and required helper code into `kernel/` and `primitives/`.
4. Introduce explicit `contracts/` files for the native layout and stage-ring constants.
5. Rebuild and revalidate after each structural move.

This ordering avoids mixing “new abstraction” and “new behavior” in the same step.

## Risks

### 1. Performance loss during refactor

If the mainloop is abstracted too early, issue order can drift and throughput can collapse even while correctness remains intact.

Mitigation:

- keep first local kernel very close to the proven schedule
- treat abstraction as file ownership and naming first, not control-flow redesign

### 2. Layout contract regressions

The native path relies on exact A/B/C pack-unpack contracts. Small indexing changes can preserve build success while completely invalidating correctness.

Mitigation:

- keep dedicated host-side layout helpers
- preserve a small correctness run before large performance runs

### 3. Premature generalization

Trying to support multiple shapes or multiple families immediately will obscure the native backend contract before the first family is stabilized.

Mitigation:

- keep phase 1 locked to one family and one benchmark target

## Follow-On Path

Once the local prototype is stable, later work should migrate layers inward in this order:

1. primitive wrappers into project C500 primitive headers
2. contracts into `include/arch/c500/gemm/contracts/*`
3. kernel family into `include/arch/c500/gemm/families/*`
4. dispatch integration
5. TK-facing user entry and broader test coverage

That later migration is intentionally excluded from this phase so that the local prototype can serve as a clean performance reference.
