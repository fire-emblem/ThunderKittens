# C500 BF16 Ampere GEMM Design

## Goal

Adapt ThunderKittens so that `kernels/gemm/bf16_ampere/bf16_ampere_gemm.cu` can be built and run with `GPU=C500`, assuming `c500` is architecture-compatible with the existing Ampere execution path.

## Scope

This change only covers the standalone GEMM entrypoint under `kernels/gemm/bf16_ampere/`.

In scope:
- Add a `C500` GPU target to the shared kernel build system.
- Route `C500` through the existing Ampere ThunderKittens code path.
- Keep the GEMM kernel source unchanged unless compilation or launch constraints require a minimal compatibility fix.
- Verify that the kernel can build and execute through the existing standalone workflow.

Out of scope:
- Adapting the broader test suite.
- Adapting other kernels, demos, or Python/PyTorch entrypoints.
- Introducing a full C500 backend abstraction across the repository.
- Performance tuning beyond making the kernel run correctly.

## Context

The current repository already has an RTX 4080 adaptation that reuses the Ampere feature path through `KITTENS_AMPERE`. The target GEMM kernel is also already an Ampere-oriented implementation: it depends on Ampere-era primitives such as `cp.async`, `ldmatrix`, and `mma.sync` rather than Hopper/Blackwell-only features such as WGMMA or TMA.

That makes the main integration point the shared build configuration rather than the kernel algorithm itself.

## Proposed Approach

### Option A: Minimal alias in `kernels/common.mk`

Add `GPU=C500` as a new architecture choice and map it directly onto the same ThunderKittens feature macro set as Ampere.

Pros:
- Smallest patch.
- Lowest implementation risk.
- Matches the short-term goal of running one kernel.

Cons:
- Leaves no room for future C500-specific quirks without editing the same branch later.

### Option B: Dedicated `C500` target reusing Ampere behavior

Add a distinct `GPU=C500` branch in `kernels/common.mk`, keep it on the Ampere ThunderKittens code path for now, and reserve the branch for future C500-specific flags if needed.

Pros:
- Still small.
- Better long-term maintainability.
- Makes the build surface explicit for users.

Cons:
- Slightly more configuration than a pure alias.

### Option C: Per-kernel local override

Handle C500 only inside `kernels/gemm/bf16_ampere/Makefile` or by adding local compile flags for this one target.

Pros:
- Fastest one-off hack.

Cons:
- Does not scale to the next kernel.
- Splits architecture logic across files.

## Decision

Use Option B.

`GPU=C500` will be added to `kernels/common.mk` as a first-class build target, but it will initially reuse the same ThunderKittens feature path as Ampere. This preserves the minimum scope while keeping a clean extension point for later C500-specific adjustments.

## Build Behavior

The `C500` build branch will:
- Define `KITTENS_AMPERE` so existing Ampere-specialized code paths remain active.
- Use a C500-specific compile target if the active CUDA toolchain supports it.
- Otherwise fall back to an Ampere-compatible `gencode` target so the kernel can still be compiled and tested in environments where C500-specific `sm_` support is not yet wired into nvcc.

The fallback behavior is intentional: the immediate goal is operational compatibility for this kernel, not a final packaging story for every toolchain variant.

## Kernel Changes

The preferred implementation is to leave `kernels/gemm/bf16_ampere/bf16_ampere_gemm.cu` untouched.

Only the following kinds of minimal edits are allowed if validation proves they are required:
- Adjusting launch-time shared-memory attributes.
- Adding a guard or compatibility tweak around a build-time assumption.
- Updating the local Makefile comment or defaults so `GPU=C500` is discoverable.

No algorithm rewrite, tiling change, or new kernel variant is planned in this task.

## Validation Plan

Validation will happen in this order:

1. Build the standalone kernel with `make GPU=C500`.
2. If the toolchain rejects a C500-specific `gencode`, switch to the documented compatibility fallback in `kernels/common.mk`.
3. Run the produced binary.
4. Confirm that the program reaches correctness checking without launch or illegal-instruction failures.

Success means:
- The kernel builds from the existing Makefile entrypoint.
- The binary runs to completion.
- The correctness check passes.

## Risks

### Toolchain support risk

The active nvcc version may not recognize a C500-specific architecture code. The design handles this with an Ampere-compatible fallback target.

### ISA mismatch risk

Even if C500 is described as Ampere-like, some instructions or occupancy assumptions may still differ. If that happens, the first response is a minimal compatibility tweak, not a repository-wide abstraction.

### Shared memory limit risk

Current Ampere branches already encode target-specific shared-memory assumptions for some GPUs such as RTX 4080. If C500 has a different practical limit, the dedicated `GPU=C500` branch gives us a safe place to encode it without changing the Ampere path globally.

## Files Expected To Change

- `kernels/common.mk`
- Possibly `kernels/gemm/bf16_ampere/Makefile`
- Only if required by validation: `kernels/gemm/bf16_ampere/bf16_ampere_gemm.cu`
