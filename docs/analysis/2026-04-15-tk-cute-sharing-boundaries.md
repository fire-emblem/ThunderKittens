# TK Cute Sharing Boundaries Across Current Kernel Families

## Purpose

This note records which parts of the current Cute-side BF16 GEMM kernels are
already suitable for shared primitive-library treatment, which parts are only
partially shareable, and which parts should remain family-specific for now.

The goal is to keep the abstraction aligned with both CuTe and ThunderKittens
style:

- share stable, repeated actions
- preserve family-specific hot-path choreography
- use evidence rather than cosmetic symmetry to decide boundaries

## Kernel families considered

- `layoutc`
- `swizzled_tn` (formerly `tn_example` on the public surface)
- `continuousc`
- `continuousc_reusea`
- `square_tt`

## Summary table

| Layer / concern | Shareability now | Why | Current vehicle |
| --- | --- | --- | --- |
| Semantic contract | partial | Semantics differ (`layoutc`, `continuousc`, `swizzled_tn`, `reusea`, `square_tt`) | `family_pattern`, family shells |
| Geometry / thread map | high | Strong repeated role, decisive performance axis | `geometry_atom`, geometry providers |
| Stage layout / residency | high | Shared-memory stage placement is a reusable contract | `stage_layout_atom` |
| Stage issue action | high | Stage-level A/B issue is a stable repeated action | `issue_order_atom` |
| Wait / barrier window | high | Wait semantics repeat across families | `schedule_atom` |
| Prologue preparation | medium | Stable in some lanes, but not all hot paths | `prologue_atom` |
| Shared -> register fragment reload | high | Repeated exact movement action | `reload_atom` |
| MMA primitive | high | Compute atom itself is stable and reused | `mma_atom` |
| Fragment store | high | Stable low-level writeback action | `store_atom` |
| Bias fragment load | high | Stable low-level epilogue action | `bias_atom` |
| Full epilogue semantic layer | partial | Semantic contract differs by output family | `epilogue_atom` façade |
| Full steady-state mainloop | low | Family-specific performance choreography still differs materially | family skeletons |
| ReuseA hot refill / seed path | low | Proven sensitive to abstraction; can regress hard | kept in `continuousc_reusea_skeleton` |
| Shape selection / dispatch | high | Mature cross-family concern | `best_shape_selected_family_t` |

## Detailed boundaries

### 1. Semantic contract

Shareability: **partial**

Why:
- `layoutc`, `continuousc`, `swizzled_tn`, `continuousc_reusea`, and
  `square_tt` do not mean the same thing semantically.
- They differ in input/output layout expectations and intended shape family.

Recommendation:
- keep semantic tags explicit in `family_pattern`
- do not collapse these into a fake universal semantic layer

### 2. Geometry / thread map

Shareability: **high**

Why:
- thread ownership
- A/B load offsets
- LDS offsets
- predicate compare operands

all form a coherent contract that is reused across multiple families.

Evidence:
- swizzled geometry versus linear geometry was measured as a decisive
  performance factor.

Recommendation:
- continue treating geometry as a first-class primitive boundary
- future kernel families should attach to geometry providers rather than hardcode
  thread/data mapping locally when possible

### 3. Stage layout / residency

Shareability: **high**

Why:
- per-stage byte span
- A/B bank placement
- stage base offsets

are structural contracts, not family-specific business logic.

Recommendation:
- keep `stage_layout_atom` separate from geometry
- new families should explicitly opt into a stage-layout contract instead of
  scattering raw offsets

### 4. Stage issue action

Shareability: **high**

Why:
- issuing one A/B stage is a repeated semantic action across `layoutc` and
  `swizzled_tn`
- this is more valuable than just wrapping one instruction at a time

Recommendation:
- keep stage-level issue primitives (`issue_ab_stage_pred`) as the preferred
  granularity
- avoid stopping at bank-level wrappers if a stage-level action is repeated

### 5. Wait / barrier window

Shareability: **high**

Why:
- prologue waits
- steady-state wait windows
- tail waits

repeat across leading lanes and are meaningful on their own.

Recommendation:
- keep `schedule_atom` focused on stable wait/sync primitives
- do not prematurely turn it into a giant reorderable scheduler

### 6. Prologue preparation

Shareability: **medium**

Why:
- layoutc prologue + prime is stable enough to share
- reusea seed/refill looked abstractable, but measurement showed severe
  performance regression when over-abstracted

Recommendation:
- only keep prologue actions in the shared layer if the action is both stable
  and measured safe
- hot family-specific preparation should remain direct until proven otherwise

### 7. Shared -> register reload

Shareability: **high**

Why:
- `load_pair_stage(...)` is a meaningful repeated action that appears in more
  than one family
- this is exactly the sort of stable movement primitive a scalable library
  should share

Recommendation:
- continue replacing repeated exact eight-load blocks with stage-pair reload
  primitives where the semantic unit matches

### 8. MMA primitive

Shareability: **high**

Why:
- compute atom itself is stable
- family differences are mostly around feeding and ordering, not around the MMA
  op abstraction itself

Recommendation:
- keep MMA at the primitive-library layer
- avoid embedding MMA details back into family-local helpers

### 9. Fragment store / bias load

Shareability: **high** at the low-level action layer, **partial** at the full
semantic epilogue layer

Why:
- fragment writeback math repeats
- bias fragment load repeats
- full epilogue semantics still differ across layout/output families

Recommendation:
- keep `store_atom` and `bias_atom` low-level
- keep `epilogue_atom` as a semantic façade
- do not flatten semantic epilogues into a fake universal writeback object

### 10. Full mainloop choreography

Shareability: **low** for now

Why:
- `layoutc`, `swizzled_tn`, `continuousc_reusea`, and `square_tt` still differ
  materially in hot-path ordering and work decomposition

Recommendation:
- do not force one universal mainloop body yet
- share actions, not full choreography

### 10.5. Square-TT stage I/O

Shareability: **medium-high** inside the square-tt family, **partial** across
the wider family set

Why:
- square-tt has repeated stage-store and stage-load actions (`STSx2` /
  `LDSx2`-style pairs)
- these actions are meaningful semantic units inside that family even though
  the full choreography is still square-tt-specific

Recommendation:
- keep square-tt-specific stage I/O in the primitive library as its own atom
- treat it as a family-local primitive that may later inform more general
  stage-I/O abstraction if another lane converges on the same unit

### 11. ReuseA hot refill / seed path

Shareability: **low** for now

Why:
- a previous attempt to abstract this path preserved correctness but badly hurt
  performance

Recommendation:
- keep it family-specific until a clearly safe shared action is identified

### 12. Shape selection / dispatch

Shareability: **high**

Why:
- dispatch is naturally cross-family
- shape-aware family selection is already part of the mature public surface

Recommendation:
- continue to add new high-performance lanes by plugging them into the shared
  dispatch vocabulary rather than inventing one-off runtime entrypoints

## Bottom line

The current families are **not** "the same kernel with different names".
Their hot loops still differ materially.

But they are also **not** unrelated worlds.
A large part of their data movement and structural logic already fits a shared
primitive library.

The correct long-term model is:

- share structural contracts and repeated actions
- keep family-specific choreography where the evidence says it still matters
- grow the primitive library upward only when a repeated semantic action is
  clearly present and performance-safe
