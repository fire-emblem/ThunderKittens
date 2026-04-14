# C500 Path 2 Refactor Sequence

## Purpose

This document turns the path-2 design analysis into a practical refactor sequence.

The goal is not to describe every implementation detail.
The goal is to define the safest order of structural changes so that:

- the new C500-native backend can be introduced cleanly
- current compatibility and diagnostic paths remain available during migration
- the repository does not get trapped in a half-old, half-new design

## Design Inputs

This sequence is based on:

- `2026-04-07-c500-path2-native-backend-architecture.md`
- `2026-04-07-c500-path2-module-map.md`
- `2026-04-07-c500-path2-scheduler-model.md`
- `2026-04-07-c500-path2-bank-and-frontier-contracts.md`

## Guiding Rules

### 1. Build new layers before deleting old ones

The migration should add the missing path-2 layers first:

- scheduler layer
- bank/frontier contracts

Only after those exist should the old family-centered logic be thinned out.

### 2. Keep compatibility paths compiling during migration

The following paths should remain available while native refactor proceeds:

- current balanced fallback family
- current operand-stage bridge path
- existing compile/smoke/probe tests

They are not the target design, but they are useful guard rails.

### 3. Do not evolve provisional helpers into permanent architecture

Files like:

- `bf16_mainloop.cuh`
- `bf16_operand_stage.cuh`
- `primitives/bf16_layouta_native_stage.cuh`

should not receive more ownership as part of the migration.

### 4. Migrate ownership, not just code

Each step should move one responsibility to its final home.

The main migration is:

- from family-owned logic
- to contract-owned and scheduler-owned logic

## Phase A: Freeze New Layer Boundaries

### Objective

Create the final architectural slots before moving behavior.

### Add

- `include/arch/c500/gemm/schedulers/`
- `include/arch/c500/gemm/contracts/bf16_muxi_bank_contract.cuh`
- `include/arch/c500/gemm/contracts/bf16_muxi_frontier_contract.cuh`

### Keep existing files unchanged in role for now

- `families/bf16_balanced_128x128x128_stage4.cuh`
- `families/bf16_muxi_128x128x128_stage4.cuh`
- `dispatch/bf16_dispatch.cuh`

### Deliverable

At the end of phase A:

- the repository has the target directory structure
- the new layers exist as explicit placeholders
- no main behavior is moved yet

## Phase B: Move Contracts Out Of Families

### Objective

Remove scheduler-visible semantics from family files and place them into contract headers.

### Move

From family-owned logic into contract-owned logic:

- resident bank slot meaning
- initial valid bank state
- reload target-slot semantics
- frontier state names and masks

### Keep where they are

- physical stage layout
- operand coordinate layout

because those already mostly live in contract-like files.

### Deliverable

At the end of phase B:

- the muxi family can refer to:
  - physical stage contract
  - operand coordinate contract
  - bank contract
  - frontier contract

without embedding those semantics directly.

## Phase C: Introduce The Scheduler Skeleton

### Objective

Make scheduler the owner of execution order before trying to match the full muxi sequence.

### Add

One first scheduler:

- `bf16_muxi_layouta_stage4_scheduler`

with three explicit entrypoints:

- `prologue`
- `step`
- `drain`

### Keep scheduler shallow at first

The first version does not need the full final ordering.
It only needs to own:

- scheduler state object
- phase identity
- event API boundary

### Deliverable

At the end of phase C:

- family no longer owns the concept of "mainloop order"
- scheduler exists as the single place where native issue order is meant to live

## Phase D: Extract Low-Level Helpers To Their Final Homes

### Objective

Separate reusable low-level pieces from transitional bridge logic.

### Keep and relocate conceptually

From `bf16_stage_primitives.cuh`:

- physical stage-ring async issue helpers
- raw shared-memory read helpers

These should remain available to the new scheduler.

### Downgrade

From `bf16_stage_primitives.cuh`:

- `mma_raw_stage_*bridge*`

These should be explicitly treated as:

- compatibility
- diagnostics
- probe support

### Leave untouched as compatibility

- `bf16_operand_stage.cuh`

### Deliverable

At the end of phase D:

- scheduler depends only on primitives and contracts
- not on bridge-style stage-to-operand rematerialization

## Phase E: Rebuild The Muxi Family As A Thin Composition Layer

### Objective

Turn the muxi family file into a wrapper over contracts and scheduler.

### Family should bind

- physical stage contract
- operand layout contract
- bank contract
- frontier contract
- muxi scheduler
- epilogue/export path

### Family should stop owning

- bank refill choreography
- wait/barrier cadence
- prologue sequencing
- tail sequencing

### Deliverable

At the end of phase E:

- `families/bf16_muxi_128x128x128_stage4.cuh` becomes thin
- the actual native model lives below it

## Phase F: Demote Historical Center Files

### Objective

Make the old architectural center obviously non-central.

### Downgrade

- `bf16_mainloop.cuh` to forwarding/compatibility only
- `bf16_contracts.cuh` to alias/header convenience only
- `primitives/bf16_layouta_native_stage.cuh` to probe-only or removable support

### Keep

- `dispatch/bf16_dispatch.cuh`

but update it to select the new composition-based family.

### Deliverable

At the end of phase F:

- old files may still exist
- but they no longer determine the structure of the backend

## Phase G: Reconnect Product Paths

### Objective

Only after the structure is correct, reconnect the example and tests to the new native path.

### Reconnect

- C500 BF16 GEMM example
- focused stage/scheduler probes
- family smoke tests
- later performance benchmarks

### Rule

Do not reconnect product paths earlier.
Otherwise old and new designs get mixed during migration.

## Minimal Risk Ordering

The safest order is:

1. add new contract/scheduler files
2. move semantics into contracts
3. create scheduler skeleton
4. split low-level helpers from bridge helpers
5. thin the muxi family into composition
6. demote old center files
7. reconnect tests and examples to the new native path

This order preserves a working fallback path for as long as possible.

## What Must Not Happen

### 1. No direct family rewrite before contracts exist

If the family is rewritten first, hidden semantics just move around without becoming explicit.

### 2. No continued growth of `bf16_layouta_native_stage.cuh`

That would keep reinforcing the wrong object model.

### 3. No "temporary" re-centralization in `bf16_mainloop.cuh`

That recreates the old ownership pattern.

### 4. No early deletion of compatibility paths

The old paths are still useful for diagnosis and regression isolation while the new backend is built.

## Practical End State

After the refactor, the intended structure is:

- primitives
- contracts
- schedulers
- families
- dispatch

with the old bridge path still present, but clearly outside the performance-first design.

## Final Position

The correct next implementation move is not another local fix in the current muxi family.

The correct next implementation move is phase A:

- add the missing contract and scheduler files
- freeze the final ownership boundaries

Then proceed through the sequence above without giving new central responsibilities to the historical compatibility files.
