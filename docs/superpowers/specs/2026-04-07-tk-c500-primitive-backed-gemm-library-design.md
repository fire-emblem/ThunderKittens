# TK C500 Primitive-Backed High-Performance GEMM Library Design

## Purpose

- Define how ThunderKittens can evolve from a collection of architecture-specific GEMM examples into a host framework for a C500-native, primitive-backed high-performance GEMM library.
- Lock the performance-first design direction before more implementation work accumulates around the current transitional `bf16_ampere` bring-up path.
- Clarify which parts of ThunderKittens can remain stable, which parts must become backend-owned, and how different GEMM shape families should be represented without collapsing into one least-common-denominator kernel.

## Problem Statement

The current C500 bring-up work has already proven two things:

- ThunderKittens can host C500-specific internals and builtins without violating project rules.
- Functionality alone is not enough; a path that compiles and produces correct BF16 GEMM results can still be structurally incapable of approaching mcBLAS-class performance.

Evidence from local high-performance references shows that C500 performance comes from a tightly matched native contract:

- `global -> shared -> native fragment -> builtin mma -> native accumulator -> store`
- wave64-native lane ownership
- shared-memory staging layouts designed for the consumer, not just the producer
- explicit async-copy accounting and barrier cadence
- interleaved issue scheduling inside the steady-state mainloop

The open design question is therefore not "can ThunderKittens call the right builtins?" but "can ThunderKittens host a performance-first GEMM library whose hot path is backend-owned while the user-facing framework remains recognizably ThunderKittens?"

## Executive Decision

- ThunderKittens is a viable host framework for a C500-native high-performance GEMM library.
- The existing framework abstractions are sufficient at the outer boundary: tile/group composition, kernel organization, type taxonomy, tests, benchmarks, and operator-facing APIs remain useful.
- The current GEMM hot-path abstractions are not sufficient as-is. They must be restructured so the C500 backend owns physical layout, fragment semantics, stage semantics, and mainloop scheduling.
- The correct target is not "one universal GEMM kernel" but "one primitive library plus multiple GEMM kernel families with a unified dispatch and frontend style."

In short:

- ThunderKittens can remain the shell.
- C500-native GEMM must become its own backend-controlled library inside that shell.

## Evidence Base

The design below is grounded in the following local references and observations:

- `muxi_native_layout_kernels`
  - `/data/muxi_native_layout_kernels/csrc/muxi_hgemm_layoutA.cuh`
  - `/data/muxi_native_layout_kernels/csrc/muxi_hgemm_layoutC.cuh`
  - `/data/muxi_native_layout_kernels/csrc/muxi_hgemm.cu`
  - `/data/muxi_native_layout_kernels/csrc/gemm_layout_A.cu`
- system `mctlass`
  - `/opt/maca/include/mctlass/gemm/threadblock/maca_group_mma_multistage/maca_group_mma_multistage_bf16_and_fp16_tn_128x128x128.h`
  - `/opt/maca/include/mctlass/arch/mma_sm80.h`
- system `cute`
  - `/opt/maca/include/cute/arch/mma_sm80.hpp`
  - `/opt/maca/include/cute/arch/copy_sm80.hpp`
- system `mcflashinfer`
  - `/opt/maca-3.3.0/include/mcflashinfer/cp_async.cuh`
  - `/opt/maca-3.3.0/include/mcflashinfer/mma.cuh`
  - `/opt/maca-3.3.0/include/mcflashinfer/permuted_smem.cuh`
  - `/opt/maca-3.3.0/include/mcflashinfer/attention/mla_utils_64b.cuh`
- ThunderKittens local C500 design/docs
  - `/data/ThunderKittens/docs/superpowers/specs/2026-04-04-c500-native-mma-backend-design.md`
  - `/data/ThunderKittens/docs/analysis/2026-04-04-c500-mma-builtins-survey.md`
  - `/data/ThunderKittens/docs/analysis/2026-04-04-c500-bf16-gemm-raw-async-findings.md`

This spec does not freeze any unproven per-lane fragment packing or exact operand order beyond what has already been validated or explicitly isolated by probe tests. When a statement below depends on a still-evolving implementation detail, it is described as a requirement on the design structure rather than a frozen micro-layout fact.

## First Principles

### 1. Primitive libraries exist to preserve hardware truth

A primitive library is not just a header collection of helper wrappers. It is the layer that exposes the real execution model of the hardware in a reusable, composable form. For C500 GEMM, that truth includes:

- wave64 participation
- native MMA fragment contracts
- async global-to-shared staging
- shared-memory consumer layouts
- pipeline wait and barrier semantics
- accumulator export rules

If the primitive layer hides these semantics behind a logical contract copied from another architecture, the resulting GEMM library will inherit the wrong cost model.

### 2. A high-performance GEMM library is not one kernel

High-performance GEMM libraries are built from:

- a primitive layer
- reusable layout and pipeline contracts
- multiple kernel families
- shape-aware dispatch and tuning

The goal is not to make every GEMM shape share the same mainloop body. The goal is to make every high-performance GEMM family share the same architectural vocabulary.

### 3. Frontend uniformity and backend physical uniformity are different goals

ThunderKittens should preserve frontend consistency:

- tile-oriented APIs
- group-level composition
- recognizable kernel organization
- reusable testing and benchmarking patterns

ThunderKittens must not require backend physical uniformity:

- identical shared layouts across architectures
- identical register fragments across architectures
- identical stage semantics across architectures
- identical mainloop structure across architectures

The backend must be free to choose the physically correct implementation as long as the frontend contract remains coherent.

### 4. Performance comes from matched contracts, not isolated builtins

Using the right builtins is necessary but not sufficient. The local references consistently show that performance comes from a fully matched chain:

- global tile partition
- async copy primitive
- consumer-ready shared layout
- low-overhead shared-to-register path
- native builtin-facing fragments
- fine-grained steady-state issue schedule
- efficient accumulator export and store

Breaking any link in that chain typically lowers the performance ceiling more than micro-tuning later can recover.

## What ThunderKittens Can Keep

The following ThunderKittens framework roles remain valid and useful:

- Tile/group/kernel organization
  - ThunderKittens is still a good host for expressing operator structure and collaboration patterns.
- Global/shared/register type systems
  - These remain useful as frontend-facing concepts, even if C500 hot paths reinterpret their backend realization.
- Architecture namespaces and backend partitioning
  - `arch/c500` is the correct place to host a native backend stack.
- Kernel examples, tests, probes, and benchmarks
  - Existing project structure is a strong advantage for iterative bring-up and verification.
- Operator-facing style
  - User-facing APIs can remain ThunderKittens-like even if the backend does not mirror Ampere/H100 internals.

In other words, the project does not need a new framework. It needs a stronger backend layering model.

## What ThunderKittens Must Change

The following assumptions cannot remain in the C500 GEMM hot path:

- Shared tiles as generic logical containers first, consumer layouts second
- Register fragments that primarily encode CUDA SM80-era logical MMA views
- A single generalized mainloop structure reused across architectures
- Copy/load/mma/store interfaces that force every backend through the same intermediate semantics
- The assumption that unsupported CUDA/PTX primitives can be replaced one-for-one without rethinking the surrounding dataflow

These assumptions are acceptable for compatibility paths, debug paths, and transitional paths. They are not acceptable for the performance hot path.

## Target Architecture

The recommended C500 GEMM library structure inside ThunderKittens has three layers:

### Layer 1: Primitive Layer

This layer maps directly to hardware-relevant operations. It should be organized by capability, not by old frontend semantics.

#### Copy primitives

- Global-to-shared async copy
- Predicated global-to-shared copy
- Shared-to-register vector load
- Shared transpose / permuted shared load
- Vectorized global store where relevant

These primitives must be expressed in terms of C500-native destinations and ownership. The primitive API should describe where data lands in the native stage contract, not merely that a copy occurred.

#### MMA primitives

- BF16/F16 input, FP32 accumulate atomic MMA contracts
- Future extension points for other types and atom shapes

This layer owns:

- builtin invocation
- operand order
- native fragment register packing
- accumulator register handling

#### Pipeline/synchronization primitives

- async outstanding count control
- barrier control
- optional issue/scheduler hints

This layer must expose the semantics needed to build `muxi`-style steady-state schedules rather than only whole-stage wait patterns.

#### Layout primitives

- lane ownership traits
- fragment layout traits
- stage residency traits
- accumulator export traits

These traits define where data belongs physically. They are not secondary metadata; they are part of the primitive contract.

### Layer 2: Native GEMM Contract Layer

This layer composes primitives into reusable GEMM building blocks.

Recommended contract categories:

- `Atom contract`
  - one MMA atom shape and datatype path, such as BF16/F16 to FP32
- `Operand contract`
  - how A/B fragments map across lanes and registers
- `Stage contract`
  - how one stage of A/B lives in shared memory
- `Tile contract`
  - how a CTA tile decomposes into waves and subtiles
- `Pipeline contract`
  - stage count, prefetch depth, producer/consumer cadence, partial wait policy
- `Epilogue/export contract`
  - how native accumulators become the output layout

This layer is the reusable "GEMM design language" for C500.

### Layer 3: Kernel Family and Dispatch Layer

This layer implements actual GEMM families and chooses among them.

Recommended responsibilities:

- Maintain multiple specialized GEMM families
- Map shapes and dtypes to candidate families
- Support offline or static tuning data
- Keep the operator-facing API uniform

The kernel-family layer should be the first place where shape specialization becomes visible. It should not leak into the primitive layer.

## Recommended Kernel Families

The first C500 GEMM library should not aim for exhaustive coverage. It should define clear family boundaries.

### Family A: Large balanced GEMM

Representative workload:

- `4096 x 4096 x 4096`
- similar large square or near-square GEMMs

Primary goals:

- maximize tensor-core utilization
- maximize overlap between copy and compute
- use large CTA tiles and deeper pipeline

This should be the first performance anchor family because:

- it exercises the architecture cleanly
- it provides a meaningful roofline target
- it exposes whether the native pipeline is structurally sound

### Family B: Narrow-N or narrow-M GEMM

Representative workload:

- very large `M`, small `N`
- very large `N`, small `M`

Primary goals:

- keep wave and CTA decomposition efficient when one output dimension is small
- avoid reusing a balanced-kernel shape that wastes computation or occupancy

This family is important because local `muxi` evidence shows narrow GEMM is a meaningful, distinct optimization target rather than a degenerate leftover case.

### Family C: Small-K GEMM

Representative workload:

- `K` in the low hundreds
- medium to large `M/N`

Primary goals:

- avoid overpaying pipeline overhead
- retune tile depth and buffering strategy

This family often needs different stage and tile choices from large balanced GEMM.

### Family D: Edge and predicated variants

Representative workload:

- non-multiple dimensions
- partial tiles

Primary goals:

- preserve correctness
- reuse nearby family structure where possible
- keep boundary handling from contaminating the hot-path steady state

These variants should exist, but they should not define the base family contracts.

## What "Shape-General" Should Mean

The phrase "support different GEMM shapes" must be interpreted carefully.

The design target is:

- one primitive library
- one contract system
- multiple kernel families
- one uniform dispatch surface

The design target is not:

- one universally optimal mainloop body
- one physically identical stage layout for every shape
- one single set of tile constants that scales across all aspect ratios

The correct level of generality is:

- common architectural vocabulary
- family-specific implementation

That is how high-performance GEMM libraries remain both reusable and fast.

## Required C500 Backend Ownership

The following areas must be backend-owned for the performance path:

### Physical shared layout

The backend must decide:

- how stage buffers are arranged
- how A/B tiles are partitioned per wave
- how staging aligns with consumer access

Frontend logical tiles may still exist, but they cannot dictate the physical layout of the hot path.

### Native fragment semantics

The backend must own:

- lane-to-fragment mapping
- fragment register packing
- builtin operand ordering
- accumulator ownership

Frontend register-tile compatibility layers may still exist for debugging or transitional code, but they must not define the hot path.

### Mainloop issue schedule

The backend must own the steady-state schedule:

- prefetch strategy
- interleaving of copy/load/mma
- partial wait cadence
- barrier placement

The framework must allow a backend mainloop to look less "uniform" if that is what the hardware demands.

### Epilogue/export mapping

The backend must define how native accumulators are exported to the framework-visible output form. This is necessary because the accumulator ownership model is tied to native fragment semantics.

## Constraints on the ThunderKittens API Surface

The user-facing API does not need to remain identical to the current Ampere path. It must satisfy these constraints instead:

- It still reads like ThunderKittens rather than raw builtin code.
- It describes tile and operator intent, not low-level register math.
- It allows backend selection or family selection without exposing hardware minutiae.
- It does not force every backend to inherit Ampere-era physical assumptions.

This means interface evolution is acceptable if it improves performance and preserves the project's abstraction style.

## Why a Pure "Primitive Substitution" Port Is Not Enough

A primitive-substitution port keeps the original execution structure and swaps unsupported operations for supported ones. That approach is insufficient here because the local evidence shows the dominant differences are structural:

- wave64 instead of warp32 assumptions
- native fragment ownership instead of logical SM80 fragments
- consumer-ready stage layout instead of generic staging
- fine-grained interleaving instead of whole-stage loops

Therefore:

- unsupported CUDA/PTX primitives should be replaced internally, per project rules
- but the surrounding kernel structure must also change where the performance model requires it

This is compatible with project constraints because the redesign remains inside ThunderKittens internals rather than delegating to external GEMM libraries.

## Recommended Development Order

### Phase 1: Stabilize the primitive library

- Freeze the first C500 MMA atom path
- Freeze the first copy primitives needed for GEMM
- Freeze the first stage-layout trait family
- Continue probe tests until fragment and layout claims are solid

Deliverable:

- a dependable C500 primitive substrate that does not assume Ampere layout semantics

### Phase 2: Build the first native contract family

- Target BF16/F16 to FP32
- Target a `128x128x128`-class large balanced family
- Encode wave ownership, stage layout, and export layout explicitly

Deliverable:

- one complete native contract stack suitable for a roofline-relevant GEMM family

### Phase 3: Build the first high-performance kernel family

- Implement a balanced large-GEMM family that targets `4096^3`-class workloads
- Use it as the performance anchor
- Compare directly against mcBLAS and local high-performance references

Deliverable:

- one family with strong evidence that the architecture is sound

### Phase 4: Expand to additional shape families

- narrow GEMM
- small-K GEMM
- edge/predicated variants

Deliverable:

- library coverage beyond the anchor family without sacrificing the first family's peak performance

### Phase 5: Add dispatch and tuning

- static heuristics first
- offline tuning tables where justified
- keep dispatch visible at the family level, not at the primitive level

Deliverable:

- a true GEMM library rather than a set of manually selected kernels

## Acceptance Criteria

This design is successful if all of the following become true:

- ThunderKittens still presents a coherent framework-level interface for GEMM development.
- The C500 backend owns all hot-path physical contracts.
- The first balanced BF16 GEMM family reaches a performance tier meaningfully competitive with local high-performance references.
- Adding new shape families does not require redesigning the primitive layer.
- Compatibility/debug layers remain possible without contaminating the hot path.

## Non-Goals

- Forcing the C500 path to preserve every existing Ampere/H100 internal abstraction
- Producing one single mainloop implementation for all architectures and shapes
- Claiming that all future datatypes or MMA atoms are already covered by this first-wave design
- Freezing any unproven micro-layout detail that still requires targeted probe validation

## Risks

### Risk: over-preserving current abstractions

If too many current GEMM assumptions are preserved, the resulting backend may compile and pass tests while remaining structurally below the performance ceiling.

### Risk: overspecializing too early

If the first implementation hardcodes too many details directly into one kernel without building the contract layer, later shape-family expansion will become expensive and error-prone.

### Risk: confusing frontend consistency with backend sameness

A desire for elegant uniform APIs can accidentally force the backend through an expensive compatibility path.

### Risk: premature family expansion

Trying to solve all shape classes before proving the anchor family will dilute effort and blur whether the primitive and contract layers are actually correct.

## Validation Plan

Reviewers of this design should verify that:

- the spec distinguishes clearly between primitive layer, contract layer, and kernel-family layer
- the spec treats ThunderKittens as a host framework, not as a fixed hot-path implementation
- shape support is described as family-based rather than single-kernel universalism
- the design remains performance-first while staying within the project's internal-only adaptation rules
- no section quietly reintroduces SM80/Ampere physical assumptions as mandatory C500 contracts

## Immediate Next Step

The next planning and implementation work should treat the first balanced BF16 C500 GEMM family as the anchor target and build the primitive and contract stack around that family before broadening shape coverage.
