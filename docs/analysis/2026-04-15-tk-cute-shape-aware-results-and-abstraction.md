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
