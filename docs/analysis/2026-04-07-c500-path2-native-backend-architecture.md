# C500 Path 2: Native Backend Architecture Analysis

## Scope

This document records the architecture-first analysis path for C500 in ThunderKittens.

Path 2 means:

- stop treating C500 as an Ampere-compatibility adaptation problem
- stop growing the current transitional bridge paths into the final backend
- model the C500 GEMM backend around the native execution contract evidenced by muxi-style kernels
- then place that model inside ThunderKittens as a backend-owned implementation stack

This is a design analysis document, not an implementation claim.

## Executive Decision

ThunderKittens should keep its outer framework style, but the C500 hot path should be rebuilt as an independent native backend stack.

The design target is:

- ThunderKittens at the user-facing shell
- C500-native contracts and schedulers in the hot path
- family-based dispatch instead of one generalized compatibility mainloop

The current historical adaptation burden should be treated as transitional only.

## Why The Old Direction Should Be Cut Back

The current repository already proves several things:

- C500-specific builtins can be hosted inside ThunderKittens
- the project can carry C500-only primitives and tests
- correctness can be recovered through operand-stage and bridge-style paths

However, those successes are not enough to justify the current hot-path structure.

The main architectural problem is that the current transition stack still assumes one or more of the following:

- a generalized mainloop can remain the center of the design
- native C500 semantics can be represented as a small variation of the existing Ampere-shaped contracts
- stage data can be flattened into logical operands before compute without changing the performance model
- a single shared abstraction can serve both compatibility and peak-performance paths

For C500 native GEMM, these assumptions are too expensive.

They create the wrong ownership boundaries:

- layout ownership stays too high in the stack
- scheduling ownership stays too generic
- register-bank semantics are hidden inside ad hoc helpers
- high-performance family structure gets mixed with compatibility code

As a result, the code can become correct while remaining structurally incapable of matching muxi or mcBLAS-class throughput.

## What Muxi Actually Models

The local muxi `layoutA` implementation should be read as a native execution model, not just as a source of builtin call snippets.

### 1. Physical stage ring

Muxi uses a fixed physical ring:

- 4 stages
- each stage is `0x4000` bytes
- each stage is subdivided into:
  - `+0x0000` A low bank
  - `+0x1000` A high bank
  - `+0x2000` B low bank
  - `+0x3000` B high bank

This layout is not a generic tile container. It is already consumer-oriented physical storage.

### 2. Wave-owned bank residency

The arrays `a[4][4]` and `b[4][4]` in muxi are easy to misread.

The important interpretation is:

- the first index is a bank-residency slot inside the active 4-stage ring
- the second index is the bank-local K-group vector index
- those arrays are not a convenient logical view of a whole K tile
- they are the register-resident working set that the scheduler updates incrementally

This means the correct abstraction is not "stage operands".
It is "bank-resident register state".

### 3. Steady-state scheduling is interleaved, not phased

Muxi does not follow:

- load one whole stage
- decode one whole stage
- compute one whole stage
- refill one whole stage

Instead it performs:

- partial MMA issue
- partial `ldg -> smem` issue
- barrier / outstanding-count control
- partial `smem -> reg` refill into a specific bank slot
- more MMA issue on the active frontier

So the scheduler is a first-class object.
It is not an implementation detail under a generic mainloop.

### 4. Tail follows the same model

Muxi tail handling is not a separate algorithm.

It keeps the same objects:

- stage ring
- bank-resident register state
- accumulator frontier

and only reduces the active frontier as the ring drains.

Therefore the correct backend structure must share one scheduler model between:

- prologue
- steady state
- drain / tail

If ThunderKittens uses different abstractions for those three phases, it will keep fighting the hardware contract.

## Architectural Root Cause In Current TK C500 Work

The main remaining issue is not a missing builtin wrapper or a local word-order bug.

The deeper root cause is:

- ThunderKittens does not yet have a backend layer whose primary object is a native wave64 GEMM scheduler with bank-resident register state

Instead, the current stack still centers around these transitional objects:

- generalized mainloop wrappers
- bridge-style operand stages
- flattened stage-consume helpers
- family code that partially owns both compatibility behavior and native behavior

That is the wrong center of gravity for a C500-native backend.

## Recommended C500 Backend Layering

The recommended design is a four-layer stack.

### Layer 1: Arch primitives

Location:

- `include/arch/c500/primitives/*`

Responsibilities:

- async copy issue
- predicated async copy issue
- barrier and outstanding-count control
- native MMA invocation
- native accumulator export

Rules:

- no Ampere-shaped logical MMA vocabulary here
- no generalized tile semantics here
- no family policy here

This layer exposes hardware truth only.

### Layer 2: GEMM physical contracts

Recommended location:

- `include/arch/c500/gemm/contracts/*`

Responsibilities:

- CTA tile geometry
- stage ring physical layout
- wave ownership
- bank slot semantics
- register-bank semantics
- native accumulator ownership
- epilogue/export mapping

This is where C500-specific physical facts become reusable contracts.

The missing contract today is the most important one:

- bank-register contract

Without this layer, register-bank meaning gets hidden inside helpers and cannot be scheduled correctly.

### Layer 3: Native schedulers

Recommended location:

- `include/arch/c500/gemm/schedulers/*`

Responsibilities:

- prologue issue policy
- steady-state issue order
- barrier cadence
- register-bank refill points
- tail drain policy

This layer should own the muxi-like execution ordering directly.

It should not be buried inside a generic family file or folded into one helper called once per stage.

For the first family, the scheduler should explicitly model:

- 4-stage ring
- 4 resident bank slots
- interleaved MMA / `ldg -> smem` / `smem -> reg` issue
- tail drain frontier

### Layer 4: Families and dispatch

Location can remain:

- `include/arch/c500/gemm/families/*`
- `include/arch/c500/gemm/dispatch/*`

Responsibilities:

- choose a contract
- choose a scheduler
- bind epilogue/export
- expose a clean family entrypoint
- dispatch by problem shape and layout

The family layer should stop owning low-level native semantics.
It should compose the layers below it.

## What Should Be Treated As Historical Burden

The following components should no longer be allowed to define the final C500 design.

### 1. `bf16_mainloop.cuh` as the center of backend structure

File:

- `include/arch/c500/gemm/bf16_mainloop.cuh`

Problem:

- it assumes one generalized mainloop entry shape should remain central
- this is too high-level for a scheduler-dominated native backend

Recommended status:

- compatibility facade only
- not the place where native scheduling logic lives

### 2. Operand-stage bridge path as the main correctness path

File:

- `include/arch/c500/gemm/bf16_operand_stage.cuh`

Problem:

- useful for bringing up correctness
- useful for probes and fallback paths
- but it re-expresses the problem in a bridge contract rather than the native bank-resident contract

Recommended status:

- keep as compatibility / diagnostic infrastructure
- do not evolve it into the primary high-performance backend

### 3. Flattened native stage consume helpers

File:

- `include/arch/c500/gemm/primitives/bf16_layouta_native_stage.cuh`

Problem:

- it encodes the wrong object model
- it assumes stage data can be loaded into one flattened `a[4][4] / b[4][4]` object and then consumed with a regular nested loop
- muxi evidence says the correct object is bank-resident register state updated under scheduler control

Recommended status:

- keep only as a probe helper if still useful
- do not treat it as the primitive contract for the final backend

### 4. Family files that mix compatibility and native models

Files:

- `include/arch/c500/gemm/families/bf16_balanced_128x128x128_stage4.cuh`
- `include/arch/c500/gemm/families/bf16_muxi_128x128x128_stage4.cuh`

Problem:

- too much ownership lives directly in family files
- native semantics, scheduling, transitional fallback, and epilogue concerns are mixed together

Recommended status:

- keep family files, but strip them down into compositional frontends
- move real native schedule logic into scheduler files

## Target Dataflow For The Final C500 Path

The final high-performance C500 path in ThunderKittens should look like this:

- global partitioning
- async `global -> stage-ring` issue
- barrier / outstanding-count control
- bank-selective `stage-ring -> native registers`
- native builtin MMA on bank-resident register state
- native accumulator export
- TK-style store / launch surface

Crucially, the semantic center is:

- bank-resident register state under scheduler control

not:

- logical operand fragments reconstructed every step

## Interface Consequence

The user-facing interface does not need to match the old Ampere path exactly.

The right goal is:

- preserve ThunderKittens style at the outer API
- permit C500-native internals to diverge completely from Ampere assumptions

This means:

- frontend tile/group abstractions may remain familiar
- backend fragment and stage semantics may be entirely C500-specific
- dispatch can select a family that has no internal structural similarity to Ampere

That is acceptable and preferable for performance.

## Recommended Refactor Direction

The next implementation wave should follow this order.

### Step 1

Freeze C500-native design boundaries:

- primitive layer
- contract layer
- scheduler layer
- family/dispatch layer

### Step 2

Demote the old bridge path:

- keep it compiling
- keep tests around it
- stop treating it as the mainline design

### Step 3

Introduce a first explicit native scheduler:

- `muxi_layouta_128x128x128_stage4`

with direct ownership of:

- prologue
- steady state
- tail

### Step 4

Define a bank-register contract header that names:

- bank slot meaning
- reload points
- accumulator frontier ownership

### Step 5

Make family code a thin composition layer on top of contract + scheduler.

## Decision Table

### Keep

- outer ThunderKittens framework style
- C500 primitive namespace
- family/dispatch structure
- tests, probes, benchmarks

### Downgrade

- operand-stage bridge path
- generalized mainloop center
- flattened native-stage consume helper

### Build New

- bank-register contract layer
- native scheduler layer
- muxi-style family implemented through those layers

## Final Position

Path 2 is the correct architecture direction.

The repository should explicitly choose:

- performance-first
- C500-native
- scheduler-owned hot path

and stop trying to preserve historical Ampere-shaped internal semantics in the final backend.

ThunderKittens remains the shell.
C500 GEMM becomes a native backend library inside that shell.
