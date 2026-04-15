# TK Cute Phase-0 Ownership and Legacy Dependency Inventory

## Scope
This inventory freezes the initial migration state for the approved primitive-kernel-unification plan.

Audited mainline roots:
- `cute_tk/*family.cuh`
- `cute_tk/*skeleton.cuh`
- `cute_tk/composition/*`
- `cute_tk/mainloop.cuh`
- `cute_tk_runtime_gemm.cu`

## Lane → Body ownership map

| Lane | Current body protocol | Current steady-state/body owner | Target body template |
| --- | --- | --- | --- |
| `layoutc` | `gemm_kernel_template` + `tile128_stage4_body_template` | `layoutc_stage4_impl` in `cute_tk/layoutc_skeleton.cuh`, but still consumes `kernel/layoutc_*` helpers | `tile128_stage4_body_template` |
| `swizzled_tn` | `gemm_kernel_template` + `tile128_stage4_body_template` | `swizzled_tn_stage4_impl` in `cute_tk/tn_example_skeleton.cuh` | `tile128_stage4_body_template` |
| `continuousc` | `gemm_kernel_template` + `tile128_stage4_body_template` | `continuousc_stage4_impl` in `cute_tk/continuousc_skeleton.cuh`, but delegates to `::bf16_c500_tk_local::kernel::tk_local_b16_128x128x128_stage4_device` | `tile128_stage4_body_template` |
| `continuousc_reusea` | family registry only | `cute_tk/continuousc_reusea_skeleton.cuh` plus legacy support dependencies | `reusea_body_template` |
| `square_tt` | family registry only | `cute_tk/square_tt_tile256x256x64_skeleton.cuh` | `square_tile256_stage4_body_template` |

## Current legacy dependency inventory

### Direct include / delegation edges from mainline roots

| File | Legacy edge | Category | Likely elimination path |
| --- | --- | --- | --- |
| `cute_tk/continuousc_skeleton.cuh` | includes `../kernel/layoutc_mainloop.cuh` and delegates to `::bf16_c500_tk_local::kernel::tk_local_b16_128x128x128_stage4_device` | steady-state body delegation | rewrite `continuousc_stage4_impl` onto cute-side tile128 primitives |
| `cute_tk/continuousc_reusea_skeleton.cuh` | includes `../kernel/layoutc_support.cuh` | helper macro / primitive leakage | split needed helpers into cute_tk primitives and delete include |
| `cute_tk/primitives/structure/geometry_atom.cuh` | includes `../../../kernel/layoutc_geometry.cuh`; calls `make_layoutc_stage_geometry` | structure helper still owned by legacy kernel layer | move stage geometry contract/provider into cute_tk contracts/primitives |
| `cute_tk/primitives/pipeline/issue_order_atom.cuh` | includes `../../../kernel/layoutc_support.cuh` | primitive depending on legacy helper layer | absorb helper surface into cute_tk primitives and remove include |
| `cute_tk/primitives/pipeline/copy_atom.cuh` | includes `../../../kernel/layoutc_prologue.cuh`; delegates to `issue_layoutc_prologue`, `prime_layoutc_registers`, `reload_layoutc_stage_from_shared` | primitive depending on legacy prologue/reload helpers | move prologue/reload stable actions into cute_tk primitives/body protocol |
| `cute_tk/primitives/epilogue/bias_atom.cuh` | includes `../../../kernel/layoutc_epilogue.cuh`; delegates to `load_layoutc_bias_fragment` | primitive depending on legacy epilogue helper | move bias-load action into cute_tk epilogue primitive |
| `cute_tk/primitives/epilogue/epilogue_atom.cuh` | includes `../../../kernel/continuousc_store.cuh`, `layoutc_epilogue.cuh`, `layoutc_store.cuh` | primitive façade backed by legacy store layer | move store/epilogue semantics into cute_tk epilogue layer |
| `cute_tk/primitives/compute/mma_atom.cuh` | includes `../../../kernel/layoutc_support.cuh`; uses legacy `FLOAT4`, `mma_16x16x16b16`, `accumulate_layoutc_kgroup` | compute primitive still depends on legacy compute helper | move float4 + mma + kgroup accumulation fully under cute_tk compute primitive |
| `cute_tk/layoutc_skeleton.cuh` | includes `../kernel/layoutc_prologue.cuh`, `../kernel/layoutc_support.cuh`, `../kernel/layoutc_tail.cuh`; uses `drain_layoutc_tail`, `run_layoutc_tail_iteration`, legacy mma helper | direct legacy include + body/helper dependency | absorb prologue/tail/kgroup logic into cute_tk body/primitives |

### Transitive / indirect legacy dependence
- `layoutc_family` and `swizzled_tn_family` appear normalized at family/launch level, but their body implementations still depend on legacy helper files through `layoutc_skeleton.cuh`, `mma_atom.cuh`, `copy_atom.cuh`, and `issue_order_atom.cuh`.
- `continuousc_family` appears normalized at family/body-protocol level, but its body still jumps directly into a tk-local implementation.
- `mainloop.cuh` and `cute_tk_runtime_gemm.cu` do not directly include legacy code, but they transitively rely on it through the family/skeleton graph above.

## Phase-0 covered shape set

### tile128 family
- `1664x1024x16384`
- `2048x2048x2048`
- `4096x4096x4096`
- `8192x8192x8192`

### reusea family
- `4608x128x3584`
- `3584x128x3584`
- `3584x128x18944`
- `4608x256x3584`
- `37888x256x3584`
- `37888x128x3584`

### square family
- `256x256x64`
- `2048x2048x2048`
- `4096x4096x4096`

## Phase-0 exit condition checklist
- Baseline artifact generated under `.omx/baselines/primitive-kernel-unification/`
- Covered shape set frozen
- Lane/body ownership map frozen
- Legacy dependency inventory frozen
