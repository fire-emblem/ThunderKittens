# C500 Native MMA Backend Design

## Purpose
- Establish a performance-first path for C500 that can be tuned toward mcBLAS-class throughput without pretending the hardware speaks the same logical fragments as Ampere.
- Keep ThunderKittens' tile/group abstractions intact, but let the backend freely choose native wave64 fragments, staging layouts, and invocation patterns that avoid needless data movement.

## Evidence Base for Layout Conclusions
- Layout and operand-order conclusions for the first-wave backend must be grounded in the previously surveyed reference classes, not in naming conventions or CUDA-era assumptions.
- `cute::SM80_16x8x16_F32F16F16F32_TN` and `cute::SM80_16x8x16_F32BF16BF16F32_TN` in `/opt/maca/include/cute/arch/mma_sm80.hpp` are evidence for how a wave64 builtin can be wrapped behind an SM80-style logical fragment API, including the need for explicit gather/scatter and the fact that only part of the wave maps back to the logical result.
- `mctlass::arch::Mma<gemm::GemmShape<16, 16, 16>, ...>` specializations in `/opt/maca/include/mctlass/arch/mma_sm80.h` are evidence for the native direct-builtin fragment contract: four 16-bit A scalars, four 16-bit B scalars, and four accumulator scalars flowing directly into `__builtin_mxc_mma_16x16x16*`.
- The fallback direct-builtin helpers in `/opt/maca/include/mcflashinfer/mma.cuh` are evidence that native-layout call sites can invoke the same builtins without the `cute` compatibility shuffle path once operands already match the native contract.
- `muxi_layout_kernels::mma_16x16x16b16<T, SwapAB>` in `/data/muxi_native_layout_kernels/csrc/muxi_hgemm_utils.cuh` and the wave64 indexing in `/data/muxi_native_layout_kernels/csrc/muxi_hgemm_layoutC.cuh` are evidence that native kernels treat fragment layout and operand order as backend-owned details and may require explicit operand swapping.

## Frozen Contract (first-wave backend)
- Native wave size is 64.
- The performance hot path is strictly `global -> shared staging -> native fragment -> builtin mma -> native accumulator -> epilogue/store`.
- Logical `rt/rv` are not the hot-path fragment contract on C500; they remain a compatibility veneer, while the backend owns the native fragment semantics under the hood.
- The backend owns both the shared-to-native copy atoms and the native-to-epilogue accumulator export so that the frontend only sees ThunderKittens-style tiles.
- The backend must not reintroduce Nvidia-specific physical fragment layouts (per-lane orderly fragments, constant multiplication by 32, etc.); native fragments may have a different packing as long as the internal layout traits stay self-consistent.
- Performance goals outweigh full parity with existing Ampere APIs, but any interface changes must continue to read as ThunderKittens abstractions and not expose hardware-specific minutiae.
- This frozen section only locks evidence-backed backend responsibilities and dataflow. It does not freeze any unproven per-lane fragment packing, operand-order assumption, accumulator export map, or exact scalar/register placement before standalone probe tests confirm them.

## First-wave Atom Shape
- Start with bf16/f16 inputs and fp32 accumulation as the source-of-truth atom when defining copy, MMA, and store semantics.
- Freeze one atomic shape for the first-wave backend so downstream layers can target a single MMA problem shape and a single backend entry path during bring-up.
- That shape freeze does not authorize downstream code to assume exact fragment packing for A, B, C, or the fp32 accumulator until standalone probe tests validate those details.
- Treat this native atom as the contract anchor for future builtins, copy atoms, and any tuning knobs; more shapes may follow only after the first-wave path is stable and any new layout claims are confirmed by standalone probes.

## Backend Responsibilities
- Express native fragment layout traits so shared tiles know how to materialize/sweep fragments without relying on `rt/rv` semantics.
- Provide shared-to-native copy routines for A and B, including any wave64 reorganization, so the hot path never round-trips through the logical frontend fragment path.
- Wrap the builtin `__builtin_mxc_mma_*` calls so that the backend owns operand ordering, lane reorganization, and accumulator handling.
- Export the native accumulator through a controlled path back into the ThunderKittens C tile, keeping the epilogue/store logic on the framework side.
- Ensure `global -> shared -> fragment -> mma -> accumulator -> store` is a single, measurable performance path; non-critical helpers such as debug layers may deviate but must not bleed into the hot path.

## Interface Evolution Guidance
- Interface changes are allowed (and expected) when they improve performance, provided they continue to read as ThunderKittens abstractions (tile dimensions, group-level tiling, etc.) and do not leak low-level hardware details.
- New frontend hooks should describe what the backend provides rather than how it does so; documentation should name the data flow, not the register math.
- Any future support for additional atoms or datatype mixes should update this spec only after the existing contract (bf16/f16 -> fp32) remains stable and the new fragment/lane details are backed by evidence plus standalone probes.

## Standalone Probe-Test Gate
- Any unresolved C500 question about fragment packing, lane ownership, operand order, accumulator export order, or logical-to-native mapping must be settled with a standalone probe test before it is promoted into backend contract language.
- Those probe tests must isolate one uncertainty at a time outside the full GEMM path so that the observed lane/layout behavior is attributable to the builtin contract rather than to surrounding scheduler or tiling code.
- Implementation work may proceed with documented provisional hypotheses, but the spec must not present those hypotheses as frozen backend facts until the standalone probes pass and the result is cross-checked against the evidence sources listed above.

## Validation
- Reviewers must verify the spec clearly separates ThunderKittens abstractions from C500 native fragments, explicitly names the performance path, and reminds contributors not to revert to `rt/rv` as the hot path.
- Reviewers must also verify that any layout conclusion is either tied to the cited evidence classes or explicitly marked as awaiting standalone probe validation.
- Self-review should focus on spotting any ambiguity (e.g., unspecified ordering or missing responsibilities) and tightening language before moving on to implementation.
