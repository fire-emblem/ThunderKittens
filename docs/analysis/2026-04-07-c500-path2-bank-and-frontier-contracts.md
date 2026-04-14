# C500 Path 2 Bank And Frontier Contracts

## Purpose

This document defines the two missing contract categories required by the path-2 C500 design:

- bank-register contract
- accumulator-frontier contract

These contracts sit between:

- physical layout contracts
- scheduler implementation

Without them, the scheduler has no stable semantics to target and family code is forced to encode hidden assumptions directly.

## Why These Contracts Are Missing Today

The current C500 tree has:

- physical stage layout
- wave/lane coordinate formulas
- primitive wrappers

But it does not yet have explicit answers to these questions:

- what does resident bank slot `0/1/2/3` mean at any given moment?
- when a bank slot is refilled, which accumulator cells become legal to update?
- what subset of the `4x4` accumulator tile is valid after each refill event?
- how does the active frontier change during drain?

Today, those answers are implicit inside muxi code order or hidden inside provisional helpers.

That is not a stable backend contract.

## Bank-Register Contract

## Role

The bank-register contract defines the semantic meaning of the resident register banks independently of how they are physically loaded.

It should answer:

- how many resident bank slots exist
- what data each slot contains
- how slot identity changes over time
- what scheduler events can populate or invalidate a slot

## Proposed Object Model

The scheduler should not think in terms of:

- `a[4][4]`
- `b[4][4]`

as anonymous arrays.

It should think in terms of four resident bank slots:

```cpp
struct bf16_muxi_bank_slot {
    native_vec a[4];
    native_vec b[4];
    int source_stage_slot;
    bool valid;
};
```

and then:

```cpp
bf16_muxi_bank_slot resident[4];
```

This does three useful things:

1. it gives bank state explicit identity
2. it decouples slot identity from shared stage offsets
3. it lets the scheduler describe reloads as state transitions instead of anonymous array writes

## Contract Fields

The bank-register contract should define at least:

- `kResidentBanks = 4`
- `kBankKGroups = 4`
- `kBankVectorsPerOperand = 4`
- `kInitialValidBanks = 2`

And semantic helpers like:

- `initial_bank_source(bank_slot)`
- `reload_target_slot(step_id)`
- `reload_source_stage(step_id)`
- `bank_is_valid(state, bank_slot)`

The important point is not the exact helper names.
It is that bank identity becomes a formal contract object.

## Initial Residency Contract

From muxi prologue:

- bank 0 is first loaded from stage slot 0
- bank 1 is next loaded from stage slot 1
- bank 2 and 3 become valid only after steady-state refill begins

So the initial bank contract is:

- `resident[0].valid = true`
- `resident[1].valid = true`
- `resident[2].valid = false`
- `resident[3].valid = false`

This matters because it disproves any design that assumes:

- all resident bank slots are populated before compute begins

## Reload Contract

The muxi scheduler performs reloads in bank-slot order, but not all at once.

The contract should capture that reload is:

- selective
- interleaved with MMA
- visible to scheduler state

The contract should therefore not be phrased as:

- "load one stage"

It should be phrased as:

- "reload bank slot `k` from stage slot `s`"

The physical stage contract tells us where the data is.
The bank contract tells us what logical resident state changes.

## Slot Identity Versus Stage Identity

These two identities must be separate:

- physical stage slot identity
- resident bank slot identity

If they are merged, the design collapses back into a flattened stage model.

Recommended invariant:

- resident bank slots are scheduler-owned logical resources
- stage slots are producer/consumer ring resources
- the scheduler may map one to the other differently over time

Even if the first implementation uses a simple one-to-one mapping, the contract should keep them conceptually separate.

## Accumulator-Frontier Contract

## Role

The frontier contract defines which accumulator cells are legal to update for a given resident-bank state.

It should answer:

- which `(m, n)` cells in `C[4][4]` are enabled after a given refill progress
- how the frontier expands in steady state
- how the frontier shrinks in drain

## Why A Frontier Contract Is Needed

Muxi does not update the full `4x4` accumulator tile uniformly from the beginning.

Instead, the valid update region evolves.

Evidence from the steady-state ordering:

- early updates involve cells such as `C[0][0]`, `C[1][0]`
- then cells using bank 1 become valid
- then bank 2 and 3 refill events unlock additional cells
- tail drains remaining diagonals in an ordered way

This is not just loop scheduling noise.

It is a correctness-relevant contract:

- some cells are not yet legal because the needed resident bank slot is not valid

## Proposed Frontier Model

A minimal abstract representation is:

```cpp
struct bf16_muxi_frontier {
    bool active[4][4];
};
```

But the contract should expose higher-level semantics:

- `frontier_for_residency(mask)`
- `drain_frontier(stage_i)`
- `cell_is_legal(mask, m, n)`

where `mask` represents which resident banks are valid.

## Recommended Semantic Interpretation

The useful abstraction is:

- the frontier advances along anti-diagonals of the `4x4` accumulator grid

This matches the qualitative structure of muxi:

- new bank validity unlocks new row/column bands
- steady-state fills toward full coverage
- drain collapses coverage in the reverse direction

The contract should be phrased in those terms instead of raw line-by-line code order.

## Frontier States

For design purposes, the first implementation can think in these abstract frontier states:

### Frontier F0

Valid when only banks `0` and `1` are resident.

Meaning:

- only the initial diagonal band is legal
- enough work exists to start compute
- full tile coverage is not yet available

### Frontier F1

Valid after bank `2` begins to populate.

Meaning:

- new row/column bands become legal
- scheduler can expand the active accumulator set

### Frontier F2

Valid after bank `3` becomes resident.

Meaning:

- full `4x4` coverage is available in steady state

### Frontier D(stage_i)

Drain frontier after no more new global tiles remain.

Meaning:

- legal coverage shrinks by stage progression
- same bank model, fewer legal cells

The exact per-cell tables can be frozen later in code or generated as constexpr masks.
The design requirement is that this contract exists as an explicit layer.

## Contract API Shape

Recommended location:

- `include/arch/c500/gemm/contracts/bf16_muxi_bank_contract.cuh`
- `include/arch/c500/gemm/contracts/bf16_muxi_frontier_contract.cuh`

Recommended shape:

```cpp
struct bf16_muxi_bank_contract {
    static constexpr int kResidentBanks = 4;
    static constexpr int kBankKGroups = 4;

    __host__ __device__ static constexpr int initial_valid_banks();
    __host__ __device__ static constexpr int reload_target_slot(int step);
};
```

```cpp
struct bf16_muxi_frontier_contract {
    using mask_type = uint32_t;

    __host__ __device__ static constexpr mask_type steady_state_mask(int residency_state);
    __host__ __device__ static constexpr mask_type drain_mask(int drain_stage);
    __host__ __device__ static constexpr bool cell_active(mask_type mask, int m, int n);
};
```

The exact signatures can differ, but the logical split should remain.

## Interaction With Scheduler

The scheduler should consume these contracts like this:

- bank contract:
  - tells the scheduler what slot reload means
  - tells the scheduler which banks are valid
- frontier contract:
  - tells the scheduler which accumulator cells may be updated at this moment

This gives a clean separation:

- contracts define legality
- scheduler defines order

That is the right boundary.

## Interaction With Families

Families should not encode bank or frontier semantics directly.

A family should only bind:

- physical contract
- bank contract
- frontier contract
- scheduler
- export/epilogue

This is exactly how path 2 removes the historical burden from family files.

## Interaction With Existing Transitional Files

### `bf16_operand_stage.cuh`

Does not own these semantics.
It remains a bridge path.

### `bf16_layouta_native_stage.cuh`

Should not own these semantics.
It currently tries to collapse them into one stage-operands object, which is precisely what the new contracts avoid.

### `bf16_mainloop.cuh`

Should not own these semantics.
It can remain a shallow forwarding layer if needed.

## Design Consequence

Once bank and frontier contracts exist, the implementation path becomes much cleaner:

1. create bank/frontier contracts
2. write scheduler against those contracts
3. bind scheduler into muxi family
4. leave compatibility paths untouched

Without these contracts, every implementation step would keep mixing:

- legality
- scheduling
- storage

inside one file.

That is exactly what path 2 is trying to avoid.

## Final Position

The missing backend contract set for C500 is:

- physical stage layout
- operand coordinate layout
- bank-register contract
- accumulator-frontier contract

The repository already has the first two in partial form.
Path 2 should add the second two before any major native-family rewrite begins.
