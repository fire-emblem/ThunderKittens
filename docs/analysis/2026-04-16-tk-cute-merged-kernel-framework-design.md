# TK Cute Merged Kernel Framework Design

## Goal

Turn the current set of high-performance GEMM lanes into a real primitive-
library-driven framework in the style of ThunderKittens and CuTe:

- primitives model stable actions and contracts
- families become registration surfaces, not implementation centers
- kernel bodies are grouped by structural class
- new lanes plug into the same framework vocabulary instead of inventing new
  one-off paths

The target is **not** one universal GEMM kernel. The target is a small set of
body templates backed by one primitive library.

## Design principles

1. **Share stable contracts first**
   - geometry
   - stage layout
   - schedule / issue / reload
   - mma
   - store / bias

2. **Share body protocols before body implementations**
   - first normalize launch and body interfaces
   - only later merge hot steady-state implementations where evidence supports
     it

3. **Preserve irreducible lane-specific choreography**
   - `continuousc_reusea`
   - `square_tt`
   still have real differences in hot-path structure

4. **Distinguish shared primitives from family-local primitives**
   - shared primitives belong to the common library
   - family-local primitives still live in the library, but remain explicitly
     scoped to that lane

## Framework layers

### 1. Family registry layer

A family should register:

- semantic tag
- pattern
- body template
- implementation hooks
- launch geometry (threads / grid)

A family should not own the core kernel wrapper anymore.

Current direction already in place for leading lanes:
- `layoutc_family`
- `swizzled_tn_family`
- `continuousc_family`

### 2. Pattern layer

`family_pattern` remains the canonical structural contract:

- semantic tag
- tile shape
- geometry atom
- schedule policy
- stage-layout atom

This is the cross-lane configuration object.

### 3. Kernel-entry layer

`composition/gemm_kernel_template.cuh` provides the shared global-entry
surface:

- one template kernel entry
- one template launch helper

This normalizes launch shape without pretending all bodies are the same.

### 4. Body-template layer

Body templates group lanes by structural class.

#### 4.1 `tile128_stage4_body_template`

Current users:
- `layoutc`
- `swizzled_tn`
- `continuousc`

Role:
- enforce tile=`128x128x128`
- enforce stage=`4`
- normalize body protocol
- delegate inner work to lane-specific `run_stage4(...)`

This is the first real shared body contract.

#### 4.2 Future `reusea_body_template`

Target user:
- `continuousc_reusea`

Reason:
- refill / seed / work partition are too different from the 128-family body
- but the lane still deserves its own normalized body template

#### 4.3 Future `square_tile256_stage4_body_template`

Target user:
- `square_tt`
- future square-derived lanes

Reason:
- tile `256x256x64`
- distinct fragment arrangement
- distinct stage-I/O structure

### 5. Primitive library layer

#### Shared structure primitives
- `geometry_atom`
- `stage_layout_atom`

#### Shared pipeline primitives
- `schedule_atom`
- `issue_order_atom`
- `prologue_atom`
- `reload_atom`

#### Shared compute primitive
- `mma_atom`

#### Shared epilogue primitives
- `store_atom`
- `bias_atom`
- `epilogue_atom` as semantic façade

#### Family-local but library-worthy primitives
- `square_tt_stage_io_atom`
- `square_tt_fragment_atom`
- future `reusea_refill_atom` if a stable action boundary appears

## Current lane classification

### Near-normalized lanes
- `layoutc`
- `swizzled_tn`

These already share:
- pattern vocabulary
- kernel-entry surface
- tile128/stage4 body protocol
- most primitive-library calls

They are the best candidates for the first shared inner compute body.

### Surface-normalized but body-not-yet-normalized
- `continuousc`

This lane now shares the family/pattern/entry/body protocol, but still delegates
its actual stage4 work to tk-local code.

### Partially normalized, still hot-path private
- `continuousc_reusea`

This lane shares family vocabulary and part of the primitive layer, but its hot
path is still private enough that it should move to its own body template rather
than being forced into the tile128 body.

### Primitive-decomposition phase
- `square_tt`

This lane now participates in the same naming and registry surface, and has
family-local primitives for:
- stage-I/O
- fragment arrangement
- thread-map / offset calculation

Its former monolithic traits bundle has been split into:
- `contracts/square_tt_tile_contract.cuh`
- `primitives/structure/square_tt_thread_map_atom.cuh`

It still needs its own body template before it can be considered structurally
normalized.

## Immediate next steps

### Step 1
Merge `layoutc` and `swizzled_tn` further by extracting the first shared inner
compute-body helpers inside the `tile128_stage4_body_template` family.

### Step 2
Create `reusea_body_template` so `continuousc_reusea` stops living as a giant
special case.

### Step 3
Create `square_tile256_stage4_body_template` and move `square_tt` onto the same
family registration discipline as the 128-family lanes.

### Step 4
Replace `continuousc`'s tk-local body with a cute-side primitive-driven body.

## Bottom line

The merged framework should be:

```cpp
family = semantic_tag
       + pattern
       + body_template
       + impl_hooks
```

backed by one primitive library.

That gives the codebase a scalable path to add new high-performance GEMM lanes
without falling back to one-off kernel files or fake universal-kernel designs.
