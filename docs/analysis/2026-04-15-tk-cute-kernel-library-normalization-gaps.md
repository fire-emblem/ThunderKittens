# TK Cute Kernel Library Normalization Gaps

## Purpose

This note records what still prevents the current BF16 GEMM lanes from looking
like a proper primitive-library-driven kernel set in the style of
ThunderKittens and CuTe.

The standard is not "some helpers exist". The standard is:

- primitives model stable actions and contracts
- kernels call those primitives directly
- family shells are thin semantic wrappers
- new kernel lanes can reuse existing contracts before inventing local helpers

## Target library model

A normalized GEMM lane should look like:

```c++
family = semantic_tag
       + family_pattern
       + geometry_provider
       + stage_layout_atom
       + pipeline atoms
       + epilogue facade
       + minimal family-specific choreography
```

Where the family-specific part is only the irreducible hot-path ordering.

## Current lanes

- `layoutc`
- `swizzled_tn`
- `continuousc`
- `continuousc_reusea`
- `square_tt`

## Gap summary

| Lane | Library conformance today | Main gaps |
| --- | --- | --- |
| `layoutc` | medium-high | still owns a large hand-written mainloop body; many direct `LDG_B128_BSM...` calls remain in skeleton/prologue |
| `swizzled_tn` | medium-high | same 128-family hot loop is still open-coded instead of expressed through a reusable kernel body |
| `continuousc` | medium | shares the template kernel-entry surface now, but still delegates its real body to tk-local logic |
| `continuousc_reusea` | medium | schedule/store sharing exists, but hot refill/seed and work partition remain mostly private |
| `square_tt` | low-medium | only stage-I/O and fragment-pack atoms exist; most execution still depends on macro-heavy local choreography |

## What is still non-conformant

### 1. Shared kernel entry exists, but not a shared 128-family body yet

`layoutc` and `swizzled_tn` now share a single template kernel-entry surface
through `composition/gemm_kernel_template.cuh`.

Current issue:
- each lane still keeps its own large stage4 device-body function
- primitives are called inside those skeletons, but the steady-state body
  itself is not yet a reusable 128-family template body

Consequence:
- launch structure is normalized
- hot-loop structure is not yet normalized

### 2. `continuousc` still delegates its body to tk-local logic

Current issue:
- shell naming, dispatch, and launch entry are unified
- deeper geometry/stage/pipeline usage is still not on par with `layoutc` and
  `swizzled_tn`
- the lane still forwards its real stage4 body into tk-local code instead of
  expressing it through the cute-side primitive vocabulary

Consequence:
- adding future continuous-C variants would still encourage local ad-hoc logic
- the lane looks normalized at the surface, but not yet in its body

### 3. `continuousc_reusea` still hides too much family-private pipeline logic

Current issue:
- hot refill / seed / work partition are still deeply local
- this is correct for performance today, but it means the lane is not yet a
  clean library caller

Consequence:
- the lane is hard to extend systematically
- future variants risk cloning private choreography instead of composing from
  primitives

### 4. `square_tt` is only partially translated into primitive language

Current issue:
- stage I/O and fragment pack now have dedicated atoms
- but the lane still depends on macro-style choreography for most of its data
  movement and compute sequencing
- representative square-shape runs still show non-zero numerical error on
  larger problems, so the lane is not yet ready to be promoted into the same
  correctness-confidence tier as the best 128-family paths

Consequence:
- it is a peer high-performance lane, but not yet a normalized kernel in the
  library style

### 5. Primitive layers are not yet cleanly split between cross-family and
family-local atoms

Current issue:
- the codebase has both truly shared atoms and square/reusea-specific ones
- but the distinction is implicit rather than codified

Consequence:
- contributors can easily over-generalize or under-generalize the next atom

## What already matches the target style

These parts are already aligned with a real primitive-library design:

- `geometry_atom`
- `stage_layout_atom`
- `schedule_atom`
- `issue_order_atom`
- `reload_atom`
- `store_atom`
- `bias_atom`
- `mma_atom`
- `family_pattern`
- shape-aware dispatch

These are not wrappers anymore; they encode reusable actions or contracts.

## Normalization rules going forward

### Rule 1: prefer semantic actions over instruction wrappers

Good:
- `issue_ab_stage_pred(...)`
- `load_pair_stage(...)`
- `pack_b_quartet(...)`

Bad:
- one wrapper per builtin with no higher-level meaning

### Rule 2: distinguish shared primitives from family-local primitives

Shared primitives:
- geometry
- stage layout
- wait / issue / reload / store / bias / mma

Family-local but library-worthy primitives:
- `square_tt_stage_io_atom`
- `square_tt_fragment_atom`
- future `reusea_refill_atom` only if it proves stable and safe

### Rule 3: share actions first, kernel bodies second

Do not jump from hand-written kernels to a fake universal body.
First extract:
- stable actions
- stable contracts
- repeated stage-level units

Then build template kernel bodies where lanes are already demonstrably close.

## Recommended next normalization steps

### Priority 1: build a shared 128-family template kernel body

Target lanes:
- `layoutc`
- `swizzled_tn`

Why:
- they already share the most primitives
- they are the best candidates for the first true template kernel body

### Priority 2: lift `continuousc` into the same primitive vocabulary depth

Why:
- right now it lags behind the best-normalized lanes
- bringing it up closes a real library gap

### Priority 3: continue square-tt decomposition without forcing fake symmetry

Next likely square-tt units:
- fragment arrangement
- additional stage movement units
- epilogue/store boundary checks

## Bottom line

The codebase already has a real primitive library, but it does not yet have a
fully normalized set of template GEMM kernels.

The remaining work is not to invent more wrappers. It is to:

- convert close lanes into shared template bodies
- keep extracting stable semantic actions
- leave irreducible hot-path choreography family-specific until the evidence
  supports the next lift
