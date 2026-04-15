# TK Cute C500 Shape-Aware Results and Example Abstraction Plan

## Purpose

This note records two things:

1. the current `shape-aware best` benchmark results across the supported
   BF16 C500 tk-cute shapes;
2. a concrete abstraction model for reconciling the imported hand-written
   example style with the existing tk-cute family stack so we can scale to
   more high-performance kernels without collapsing everything into one
   monolithic kernel.

## Shape-aware benchmark results

Source file:

- `kernels/gemm/bf16_c500_tk_cute_local/results/current/cute_tk_vs_tk_local.csv`

Current `shape-aware best` selections:

| Shape | Selected family | TFLOP/s | vs mcBLAS |
| --- | --- | ---: | ---: |
| 1664x1024x16384 | `cute_tk_layoutc_128x128x128_stage4` | 152.165 | 0.684x |
| 2048x2048x2048 | `cute_tk_layoutc_128x128x128_stage4` | 98.496 | 0.595x |
| 4096x4096x4096 | `cute_tk_tn_example_bf16_128x128x128_stage4` | 149.089 | 0.755x |
| 4608x128x3584 | `cute_tk_continuousc_reusea_n_params` | 45.454 | 0.744x |
| 3584x128x3584 | `cute_tk_continuousc_reusea_n_params` | 41.705 | 0.753x |
| 4608x256x3584 | `cute_tk_tn_example_bf16_128x128x128_stage4` | 76.400 | 0.756x |
| 37888x256x3584 | `cute_tk_tn_example_bf16_128x128x128_stage4` | 135.182 | 0.914x |
| 3584x128x18944 | `cute_tk_continuousc_reusea_n_params` | 64.580 | 0.604x |
| 37888x128x3584 | `cute_tk_tn_example_bf16_128x128x128_stage4` | 124.655 | 1.628x |

### Immediate conclusion

The project already needs multiple family lanes:

- `layoutc` remains best on some square / layout-native shapes
- `tn_example` is best on several large-throughput shapes
- `continuousc_reusea` remains best on some narrow-N reuseA-heavy shapes

This means the right abstraction target is not a universal kernel. It is a
composable family-selection system.

## What we learned from the imported example

The strongest imported idea is not the whole kernel body. It is the
**geometry contract**:

- thread-to-fragment ownership
- global-load indexing
- shared-memory landing pattern
- LDS fragment reload offsets
- the MMA consumption order that matches that geometry

Experiments showed that replacing the swizzled geometry with a linearized
version in the imported TN kernel drops performance dramatically. So the
geometry must be treated as a first-class optimization component, not as a
small helper.

## Current abstraction state

The codebase now contains the beginnings of a scalable decomposition:

### 1. Host layout traits

Examples:

- `host::layoutc_host_traits`
- `host::continuousc_host_traits`
- `host::tn_example_host_traits`

This layer defines semantic boundaries for:

- A/B packing
- C readback interpretation
- reference alignment

### 2. Geometry providers

Examples:

- `current_layoutc_geometry_provider`
- `tn_example_swizzled_geometry_provider`
- `tn_example_linear_geometry_provider`

This layer owns:

- load offsets
- LDS offsets
- thread-map details
- predicate compare operands

### 3. Geometry atoms

`geometry_atom<HostLayout, GeometryProvider>` is the bridge that binds host
semantics to a geometry implementation.

Current aliases include:

- `layoutc_layout_atom`
- `continuousc_layout_atom`
- `tn_example_swizzled_layout_atom`
- `tn_example_linear_layout_atom`

### 4. Family skeletons

The imported TN path is now modeled as:

- `tn_example_family<GeometryAtom>`

This lets one kernel skeleton spawn multiple family variants by swapping only
geometry.

### 5. Shape legality and family dispatch

Families now declare `supports_runtime_shape(...)`, and runtime dispatch can
select different winners by shape.

## How example and tk-cute should unify going forward

The cleanest scalable model is:

### Layer A: Semantic family boundary

Examples:

- `layoutc`
- `continuousc`
- `reusea`
- `tn_example`

A family answers:

- what layout semantics it expects
- what output semantics it writes
- which shapes it is designed for

### Layer B: Geometry provider

A geometry provider answers:

- how threads map to data
- how A/B land in shared memory
- how LDS offsets are formed
- which compare operands control predicated loads

This is where the most reusable value from the imported example lives.

### Layer C: Pipeline / schedule policy

A schedule policy answers:

- stage count
- issue order
- wait-window strategy
- overlap between load / LDS / MMA

The imported example proved that whole-schedule grafting is not always a win,
so this layer should be swappable independently from geometry.

### Layer D: Tile-shape policy

A tile policy answers:

- CTA tile M/N/K
- thread count expectations
- fragment count expectations

### Layer E: Epilogue / store policy

An epilogue policy answers:

- output layout
- alpha / beta handling
- accumulator export
- bias / reduction handling

## Recommended next technical direction

To support more kernels across more shapes, we should evolve toward:

`family semantic x geometry provider x schedule policy x tile policy x epilogue policy`

That does **not** mean creating a giant generic kernel immediately.
It means making each axis explicit enough that new families can reuse proven
pieces without copy-pasting whole kernels.

## Practical roadmap

### Short term

- keep `tn_example` as a family candidate
- keep shape-aware dispatch based on measured winners
- continue benchmarking with the current result table

### Medium term

- lift current `layoutc` and `continuousc_reusea` further toward explicit
  geometry-provider and schedule-policy composition
- avoid another whole-schedule graft experiment unless a new family semantic
  needs it

### Long term

- build multiple high-performance kernels by composing proven geometry,
  schedule, and tile policies per shape family rather than trying to force one
  kernel to dominate all shapes

## Bottom line

The imported example and current tk-cute implementation unify best if we treat
example-style code as a source of **components**, especially geometry, rather
than a monolithic replacement kernel.

The current project is now in a good position to scale because the critical
abstraction seam — geometry provider selection — has been established.


## Schedule policy seam status

The imported TN example path now also exposes an explicit schedule/stage
policy seam.

Current status:

- `tn_example_stage4_schedule` is the first formal schedule policy
- `tn_example_family` is parameterized by both `GeometryAtom` and
  `SchedulePolicy`
- the current implementation still executes only the existing stage4
  schedule, but the family/skeleton boundary no longer hardcodes schedule as
  an invisible implementation detail

This is an intentional intermediate state:

- geometry has already been proven to be a decisive optimization axis
- schedule is now positioned as the next composable axis
- future work can add alternative stage-count or issue-order policies without
  cloning the full family shell again


## Layoutc schedule seam status

The current `layoutc` mainline family now also accepts an explicit
schedule-policy template parameter in addition to its geometry atom.

Current status:

- `layoutc_family<TileShape, StagePolicy, GeometryAtom, SchedulePolicy>` is
  now the canonical form
- `layoutc_stage4_schedule` is the initial formal schedule policy
- `layoutc_skeleton` accepts `SchedulePolicy` explicitly, even though only the
  current stage4 schedule is implemented today

This means both imported-example families and current layoutc families now
share the same high-level composition direction:

- host traits
- geometry provider / geometry atom
- schedule policy
- family shell
- shape legality / dispatch

That is the minimum viable abstraction set required to scale toward more
shape-specific high-performance kernels without cloning full implementations.


## Schedule variant experiment

We also ran a same-geometry / different-schedule comparison on the imported
TN example path by adding a more conservative stage4 schedule variant.

Shapes tested:

- 1664x1024x16384
- 2048x2048x2048
- 4096x4096x4096

Representative averages:

| Shape | conservative schedule | default TN schedule | Winner |
| --- | ---: | ---: | --- |
| 1664x1024x16384 | 151.310 | 151.298 | conservative (effectively tied) |
| 2048x2048x2048 | 99.111 | 100.017 | default TN schedule |
| 4096x4096x4096 | 148.672 | 149.160 | default TN schedule |

Interpretation:

- exposing schedule as an axis is correct
- a more conservative synchronization-heavy stage4 variant does not beat the
  existing imported TN schedule on the tested shapes
- future schedule work should explore more meaningful issue-order / wait-window
  variants, not just extra synchronization


## Continuousc / reusea composition status

The current `continuousc`, `continuousc_reusea`, and
`continuousc_reusea_layoutc` family shells now also expose explicit
`geometry_atom` and `schedule_policy` seams at the family level.

This does not yet rewrite their kernel internals around alternate geometry or
schedule implementations, but it aligns their family-shell abstraction model
with the direction already established for `tn_example` and `layoutc`.

The practical result is that all major tk-cute family lanes now speak a shared
composition vocabulary:

- host layout traits
- geometry atom
- stage-layout atom
- schedule policy
- family shell
- shape legality / dispatch

That is the foundation required for future multi-family scaling work.


## Unified family-pattern status

Major tk-cute family shells now also inherit a common composition pattern:

- `layoutc`
- `tn_example`
- `continuousc`
- `continuousc_reusea`
- `continuousc_reusea_layoutc`

The shared pattern captures:

- semantic tag
- tile shape
- geometry atom
- schedule policy
- stage-layout atom
- host layout (derived from geometry atom)

This is intentionally a **family-shell unification**, not a forced kernel-body
unification. It gives the codebase a single abstract mode of describing family
composition without pretending that all mainloops are already interchangeable.


## Tile-shape seam status

The imported TN example family now also accepts an explicit tile-shape policy
parameter in addition to geometry and schedule. Current behavior remains locked
to `128x128x128`, but tile shape is no longer buried as an invisible family
constant.

Current status:

- `tn_example_family<GeometryAtom, SchedulePolicy, TileShape>` is now the
  canonical form
- `tile_128x128x128` is the only implemented TN example tile today
- the family grid logic and shape legality now flow through the tile policy

This means the imported example path has explicit seams for all three major
kernel-shaping axes we currently care about:

- geometry
- schedule / stage
- tile shape


## Stage-layout seam status

The next reusable primitive extracted from the imported example is the shared
stage-layout contract: the physical per-stage shared-memory byte span plus the
fixed A/B bank offsets inside each stage.

Current status:

- `cute_tk/stage_layout_atom.cuh` now exposes a reusable
  `stage_layout_atom<StageContract>`
- both the imported `tn_example` path and the current `layoutc` path thread a
  stage-layout atom through their family/skeleton boundary
- the legacy `stage_contract` now provides named helpers such as
  `stage_base_offset`, `a_stage_offset`, and `b_stage_offset`
- layoutc prologue/register-prime helpers and the TN example skeleton now use
  these named stage-layout helpers instead of repeating raw `0x4000/0x1000`
  slot math inline

Why this matters:

- geometry answers *which thread owns which data*
- stage layout answers *where that data physically lives inside each shared
  stage*

Keeping those as separate seams is what lets example-style low-level code turn
into reusable Cute/TK building blocks rather than another monolithic family.
