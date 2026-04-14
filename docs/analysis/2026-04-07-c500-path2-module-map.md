# C500 Path 2 Module Map

## Purpose

This document turns the path-2 architecture decision into a concrete module map for the current `include/arch/c500/gemm` tree.

Goal:

- decide which existing files remain part of the final design
- decide which files are compatibility-only
- decide which responsibilities must move into new layers
- avoid growing the current transitional structure by inertia

This is a design and ownership map, not an implementation patch.

## Layer Model

For path 2, the target C500 stack is:

1. `arch primitives`
2. `gemm physical contracts`
3. `gemm schedulers`
4. `families + dispatch`

The current tree does not yet have a proper scheduler layer.
That missing layer is the main structural gap.

## Current Tree

Current files under `include/arch/c500/gemm`:

- `bf16_contracts.cuh`
- `bf16_epilogue.cuh`
- `bf16_mainloop.cuh`
- `bf16_operand_stage.cuh`
- `bf16_stage_primitives.cuh`
- `contracts/bf16_balanced_contracts.cuh`
- `contracts/bf16_balanced_operand_layout.cuh`
- `contracts/bf16_balanced_stage_layout.cuh`
- `dispatch/bf16_dispatch.cuh`
- `families/bf16_balanced_128x128x128_stage4.cuh`
- `families/bf16_muxi_128x128x128_stage4.cuh`
- `primitives/bf16_layouta_native_stage.cuh`

## Module Classification

### Keep As Final-Layer Building Blocks

These files are structurally compatible with the path-2 direction.

#### `dispatch/bf16_dispatch.cuh`

Status:

- keep

Reason:

- dispatch belongs at the top of the stack
- it already acts as a family selector
- its current structure is simple and correct in principle

Needed changes:

- dispatch should select among native C500 families
- it should stop selecting between "default balanced" and "experimental native fallback"
- selection logic should become family-centric rather than bridge-centric

Future role:

- permanent dispatch layer

#### `contracts/bf16_balanced_contracts.cuh`

Status:

- keep and evolve

Reason:

- this is already moving in the right direction
- it expresses family-level physical geometry
- it is a suitable contract-layer file

Needed changes:

- add explicit scheduler-facing contract categories
- split bank/register semantics out instead of letting families own them

Future role:

- permanent physical contract layer

#### `contracts/bf16_balanced_stage_layout.cuh`

Status:

- keep and evolve

Reason:

- stage layout is a physical contract
- the file is in the correct conceptual layer

Current limitation:

- it is still mostly an alias/wrapper around legacy layout definitions

Needed changes:

- stop inheriting identity from legacy compatibility layout
- let this file become the canonical owner of the physical stage-ring definition for the family

Future role:

- permanent stage-layout contract

#### `contracts/bf16_balanced_operand_layout.cuh`

Status:

- keep and evolve, but narrow its meaning

Reason:

- it already expresses wave64-native coordinate logic
- this belongs in the contract layer

Current limitation:

- it still conflates operand layout with part of what should become bank-register semantics

Needed changes:

- keep lane and K-group coordinate formulas here
- move scheduler-visible bank-slot semantics into a separate contract file

Future role:

- permanent operand coordinate contract

#### `bf16_epilogue.cuh`

Status:

- keep, with light restructuring

Reason:

- epilogue/export is a real backend layer
- the current file is small and does not carry major historical burden

Needed changes:

- potentially rename or relocate if multiple C500 family-specific epilogues appear
- keep the actual accumulator export policy family-bound, not mainloop-bound

Future role:

- permanent export/epilogue adapter

## Keep But Downgrade To Compatibility Or Transitional Support

These files are useful, but they should no longer define the center of the design.

### `bf16_contracts.cuh`

Status:

- keep as compatibility alias only

Reason:

- it currently just aliases `contracts::bf16_balanced_128x128x128_stage4`
- this is acceptable as a shallow compatibility surface

Rule:

- do not add new design logic here
- new code should include explicit contract headers instead

Future role:

- compatibility include

### `bf16_mainloop.cuh`

Status:

- downgrade

Reason:

- path 2 rejects the idea that one generalized mainloop file should remain the center of C500 GEMM
- native C500 scheduling must move below family/dispatch into scheduler-owned files

Current problem:

- it still presents the illusion that the backend has one mainloop identity
- it forwards both balanced and muxi family behavior through one shared surface

Future role:

- compatibility facade
- optional forwarding header for legacy callers

Do not:

- add new native scheduling logic here

### `bf16_operand_stage.cuh`

Status:

- downgrade hard

Reason:

- this file is useful for bridge-style correctness paths
- it is not the right abstraction for the final performance path

Current problem:

- it encodes a re-materialized operand-stage representation
- that is a bridge contract, not the muxi-native execution contract

Future role:

- compatibility path
- diagnostic path
- probe support

Do not:

- make this the basis of the final C500 family design

### `bf16_stage_primitives.cuh`

Status:

- split and downgrade

Reason:

- the file contains multiple kinds of logic mixed together:
  - async issue into physical stage ring
  - raw shared-load helpers
  - bridge paths into logical operand fragments
  - stage-tile MMA bridge helpers

The first category is valuable.
The bridge-oriented categories are transitional.

Recommended split:

- keep physical stage-ring copy and low-level raw shared access in a permanent layer
- demote `mma_raw_stage_*bridge*` helpers to compatibility/diagnostic support

Future role:

- partially retained after extraction

## Replace Or Retire As Final-Path Primitives

These files encode the wrong final object model.

### `primitives/bf16_layouta_native_stage.cuh`

Status:

- retire from final design

Reason:

- it models a flattened `stage operands` object:
  - `a[4][4]`
  - `b[4][4]`
  - regular nested-loop consume

That is not the muxi execution model.

Probe evidence already indicates:

- the remaining bug is not a small pair-order mistake
- the missing semantics are at the bank/index ownership level

Therefore:

- this file is a useful probe vehicle
- it is not the correct primitive contract for the final backend

Future role:

- probe-only helper
- or removable once the scheduler path is established

## Restructure Heavily: Families

Families should remain, but with much less ownership.

### `families/bf16_balanced_128x128x128_stage4.cuh`

Status:

- keep file category, reduce responsibility

Current problem:

- the file owns too much:
  - contracts
  - zero/init logic
  - stage issue
  - operand-stage issue
  - stage consume
  - layoutA fallback behavior
  - partial native logic
  - export/store

This is too much ownership for a family file.

Future role:

- thin composition layer:
  - choose contract
  - choose scheduler
  - choose export/epilogue
  - expose family entrypoint

Move out:

- low-level scheduling
- bank-register refill ordering
- native mainloop issue order

### `families/bf16_muxi_128x128x128_stage4.cuh`

Status:

- rewrite conceptually

Current problem:

- it currently tries to compress muxi semantics into:
  - a flattened stage load
  - one per-tile consume helper
  - simple refill/wait progression

That is not the right model.

Future role:

- first true native family wrapper around:
  - a muxi-style contract set
  - a muxi-style scheduler

Rule:

- do not continue extending the current flattened helper-based implementation

## Missing Files That Should Be Added

Path 2 needs new files more than it needs more edits to the old ones.

### New contract files

Recommended:

- `include/arch/c500/gemm/contracts/bf16_muxi_bank_contract.cuh`
- `include/arch/c500/gemm/contracts/bf16_muxi_register_contract.cuh`
- `include/arch/c500/gemm/contracts/bf16_muxi_accumulator_frontier.cuh`

Responsibilities:

- define bank slot meaning
- define which reload updates which resident register bank
- define which accumulator frontier is valid after each scheduler phase

This is the main missing abstraction today.

### New scheduler files

Recommended:

- `include/arch/c500/gemm/schedulers/bf16_muxi_layouta_stage4_scheduler.cuh`
- optionally:
  - `.../bf16_muxi_layouta_stage4_prologue.cuh`
  - `.../bf16_muxi_layouta_stage4_tail.cuh`

Responsibilities:

- prologue sequencing
- steady-state interleave
- barrier cadence
- register-bank refill points
- tail drain

This is where muxi should actually be modeled.

## Responsibility Migration Map

### From `bf16_mainloop.cuh`

Move out:

- all real execution policy

Keep:

- optional compatibility forwarding

### From `bf16_operand_stage.cuh`

Keep:

- bridge correctness infrastructure

Do not promote:

- operand-stage representation into final native path

### From `bf16_stage_primitives.cuh`

Keep:

- physical stage-ring issue helpers
- raw shared-memory access helpers

Move out:

- bridge MMA tile helpers into compatibility support

### From `bf16_layouta_native_stage.cuh`

Discard as final-path design:

- flattened stage-operands object model

### From family files

Move out:

- scheduler logic
- detailed refill ordering
- bank reload choreography

Keep:

- composition and public family entrypoints

## Minimal Stable Topology After Refactor

After path-2 refactor, the stable topology should be:

- `arch/c500/primitives/*`
- `arch/c500/gemm/contracts/*`
- `arch/c500/gemm/schedulers/*`
- `arch/c500/gemm/families/*`
- `arch/c500/gemm/dispatch/*`

and legacy compatibility files should clearly sit off to the side rather than in the center.

## Decision Summary

### Keep

- `dispatch/bf16_dispatch.cuh`
- `contracts/bf16_balanced_contracts.cuh`
- `contracts/bf16_balanced_stage_layout.cuh`
- `contracts/bf16_balanced_operand_layout.cuh`
- `bf16_epilogue.cuh`

### Downgrade

- `bf16_contracts.cuh`
- `bf16_mainloop.cuh`
- `bf16_operand_stage.cuh`
- bridge parts of `bf16_stage_primitives.cuh`

### Rewrite Or Replace

- `families/bf16_balanced_128x128x128_stage4.cuh`
- `families/bf16_muxi_128x128x128_stage4.cuh`
- `primitives/bf16_layouta_native_stage.cuh`

### Add

- `gemm/contracts/*` for bank/register/frontier semantics
- `gemm/schedulers/*` for muxi-style native execution order

## Final Recommendation

The next design-valid step is not another local fix in the current muxi family file.

The next design-valid step is:

- create the missing scheduler layer
- create the missing bank-register contract layer
- re-home existing responsibilities into those layers
- then rebuild the muxi family as a thin composition wrapper

That is the cleanest way to remove the historical C500 adaptation burden while preserving ThunderKittens as the outer framework.
