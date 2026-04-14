# C500 Path 2 Scheduler Model

## Purpose

This document converts the muxi `layoutA` kernel body into a scheduler-oriented model that can be hosted cleanly inside ThunderKittens.

The intent is to stop reasoning in terms of:

- scattered code snippets
- isolated builtin calls
- flattened stage helpers

and instead define:

- scheduler state
- state transitions
- events
- ownership boundaries

This is the missing design layer between physical contracts and family entrypoints.

## Core Observation

Muxi is not built around a "mainloop over stages".

It is built around a scheduler over three kinds of events:

1. `LDG -> SMEM` issue into one physical stage slot
2. `SMEM -> REG` refill into one resident bank slot
3. `MMA` issue on one currently valid accumulator frontier

The scheduler keeps all three moving in one interleaved order.

That means the C500 backend needs a first-class scheduler layer.

## The Objects The Scheduler Owns

The scheduler should explicitly own the following objects.

### 1. Physical stage ring

This is the shared-memory ring:

- 4 stage slots
- each slot is `0x4000`
- each slot contains:
  - A low half at `+0x0000`
  - A high half at `+0x1000`
  - B low half at `+0x2000`
  - B high half at `+0x3000`

The scheduler does not reinterpret this as a generic tile.

### 2. Resident bank state

The scheduler owns a register-resident state:

- `A_bank[4][4]`
- `B_bank[4][4]`

but these names should not be exposed as "stage operands".

The correct semantic is:

- 4 resident bank slots
- each slot carries 4 bank-local K-group vectors
- each slot may be valid, pending refill, or newly loaded

So a better internal model is:

```cpp
struct resident_bank_slot {
    native_vec a[4];
    native_vec b[4];
    bool valid;
};
```

and then:

```cpp
resident_bank_slot banks[4];
```

This is much closer to what muxi is doing than a flattened `stage_operands` struct.

### 3. Accumulator frontier

The scheduler owns a `4 x 4` accumulator tile:

- `C[4][4]`

but not every point on that grid is equally schedulable at every moment.

Muxi advances an active frontier through this grid.

The meaningful abstraction is:

- a subset of accumulator cells is currently legal to update from the available resident banks

This should be expressed explicitly.

### 4. Outstanding async state

The scheduler also owns:

- how many async groups are still outstanding
- when the next barrier is required
- which stage slot can be refilled next

This is not just pipeline bookkeeping.
It is part of the legal transition system.

## Scheduler State

The scheduler can be modeled with a compact state structure.

Suggested shape:

```cpp
struct layouta_stage4_scheduler_state {
    int next_global_tile;          // next K=128 tile not yet issued
    int resident_stage_slot[4];    // physical stage slot backing each bank slot
    bool bank_valid[4];            // whether bank slot is populated
    int outstanding_transactions;  // async copies not yet retired
    int drain_stage;               // drain progress in tail, -1 in steady state
};
```

This state is not tied to one specific code layout, but it makes the ownership explicit:

- which bank slot maps to which physical stage slot
- what can still be issued
- whether the scheduler is in prologue, steady state, or drain

## Phases

Muxi can be modeled as one scheduler with three phases.

## Phase 1: Prologue

### Purpose

- fill enough physical stage slots to start compute
- materialize the first resident banks

### Muxi evidence

At the start:

- all 4 stage slots receive `LDG -> SMEM`
- then barrier
- bank 0 is loaded from stage slot 0
- then barrier
- bank 1 is loaded from stage slot 1

So prologue does not fully materialize all 4 resident banks immediately.

It produces a partial resident state:

- bank 0 valid
- bank 1 valid
- bank 2 invalid
- bank 3 invalid

and then compute begins while future banks are still being retired and loaded.

### Scheduler interpretation

Prologue should be modeled as:

1. issue up to `min(4, num_k_tiles)` stage copies
2. retire enough async work for bank 0
3. load bank 0
4. retire enough async work for bank 1
5. load bank 1
6. enter steady state

### Why this matters

This immediately shows why the current flattened helper model is wrong:

- the legal resident state at steady-state entry is not "all 4 banks loaded"
- it is a partial state that the scheduler completes later while already computing

## Phase 2: Steady State

### Purpose

- maintain a moving resident-bank window
- keep MMA fed
- keep the stage ring refilled

### Structural pattern in muxi

The body from line ~157 onward is not random.
It follows a repeated motif:

1. update some accumulator cells using already-valid banks
2. issue one or two `LDG -> SMEM` operations into a stage slot that is about to become free
3. wait/barrier at a specific retirement threshold
4. load one new A or B bank fragment from a future stage slot into one resident bank slot
5. continue MMA on an expanded or shifted frontier

This repeats until all four resident bank slots become active and then start rotating.

### Frontier view

The accumulator frontier grows and rotates through the 4x4 grid.

A useful abstract description is:

- the scheduler advances along anti-diagonals of the accumulator tile
- each new resident bank enables one new row-band or column-band of legal updates
- full 4x4 coverage is achieved only after enough bank refill events have completed

This is why a scheduler abstraction is necessary:

- the legal MMA set depends on bank validity history
- it is not just a rectangular nested loop

### Event classes in steady state

The scheduler needs these event types:

- `mma_row_bank(bank_b, bank_a_row, acc_cell)`
- `mma_col_bank(bank_b_col, bank_a, acc_cell)`
- `issue_a_half(stage_slot, global_tile, half)`
- `issue_b_half(stage_slot, global_tile, half)`
- `retire_to_threshold(outstanding_target)`
- `reload_a_bank(bank_slot, stage_slot)`
- `reload_b_bank(bank_slot, stage_slot)`

In code, these can later be grouped into coarser helpers, but the scheduler model should treat them distinctly.

## Phase 3: Drain / Tail

### Purpose

- finish all legal accumulator updates after no more new global tiles remain

### Muxi evidence

The code from line ~550 onward shows:

- for each `stage_i`
  - consume all earlier bank interactions on one frontier
  - consume the current diagonal frontier
  - retire and barrier
  - if more resident banks remain, reload the next one

Then a final block repeats the same pattern with fewer reloads.

### Scheduler interpretation

Drain is not a special kernel.

It is:

- steady state with `next_global_tile == end`
- bank rotation still allowed while resident stages remain
- legal frontier shrinking as no new stage slots are issued

So the same scheduler object can support drain by flipping one condition:

- no more issue events are legal

Everything else remains the same model.

## Recommended Scheduler API

The scheduler layer should expose a small number of explicit operations.

Suggested interface shape:

```cpp
struct bf16_muxi_layouta_stage4_scheduler {
    using state = bf16_muxi_layouta_stage4_scheduler_state;

    template<typename Context>
    __device__ static void prologue(Context&, state&);

    template<typename Context>
    __device__ static bool step(Context&, state&);

    template<typename Context>
    __device__ static void drain(Context&, state&);
};
```

Where `Context` owns:

- pointers / tiles for A/B/C
- stage ring base
- resident bank storage
- accumulators
- lane/wave metadata

The key point is that `step()` means:

- execute one legal scheduling macro-step

not:

- consume one logical stage

## Recommended Contract Split

To support the scheduler cleanly, contracts should be split more explicitly.

### 1. Physical stage contract

Already partly present.
Owns:

- stage slot count
- stage slot offsets
- physical A/B halves

### 2. Operand coordinate contract

Already partly present.
Owns:

- lane coordinates
- K-group coordinate formulas
- row/column ownership formulas

### 3. Bank-register contract

Missing today.
Should own:

- meaning of each resident bank slot
- which refill populates which slot
- which A/B vectors belong to a bank slot
- when a bank slot becomes valid

### 4. Frontier contract

Missing today.
Should own:

- which accumulator cells are legal after each scheduler macro-step
- how the frontier expands or shrinks

## Mapping To Current TK Files

### Should stay below scheduler

- `include/arch/c500/primitives/pipeline.cuh`
- `include/arch/c500/mma_atoms.cuh`
- physical copy helpers extracted from `bf16_stage_primitives.cuh`

### Should move beside scheduler as contracts

- `contracts/bf16_balanced_stage_layout.cuh`
- `contracts/bf16_balanced_operand_layout.cuh`
- new bank/frontier contract files

### Should become thin wrappers over scheduler

- `families/bf16_muxi_128x128x128_stage4.cuh`

### Should not define the scheduler

- `bf16_mainloop.cuh`
- `bf16_operand_stage.cuh`
- `primitives/bf16_layouta_native_stage.cuh`

## What Not To Do

The scheduler design should explicitly avoid these traps.

### 1. Do not model "one stage consume" as the primary step

That is not muxi semantics.

### 2. Do not flatten resident banks into one temporary stage-operands object

That erases scheduler-visible validity and refill structure.

### 3. Do not separate tail into a fundamentally different path

Drain is the same scheduler with issue disabled.

### 4. Do not let family files hardcode the whole execution sequence

That prevents reuse and makes future shape families harder.

## Minimal Viable Scheduler Design For First Implementation

The first implementation does not need to solve every future family.

It only needs to formalize one scheduler:

- `bf16_muxi_layouta_128x128x128_stage4`

and freeze these properties:

- 4 stage slots
- 4 resident bank slots
- prologue with partial bank residency
- interleaved steady-state macro-steps
- drain using the same resident-bank model

If that scheduler is explicit and compositional, later families can reuse the same architecture even if their exact schedule differs.

## Final Position

The correct next design layer for C500 in ThunderKittens is a scheduler layer.

Without it:

- contracts remain too passive
- families remain too overloaded
- native semantics remain trapped inside ad hoc helpers

With it:

- muxi can be modeled cleanly
- historical compatibility burden can be isolated
- ThunderKittens can host a true C500-native backend without pretending it still has Ampere-shaped internals
