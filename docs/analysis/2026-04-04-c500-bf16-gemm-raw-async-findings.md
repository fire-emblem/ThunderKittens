# C500 BF16 GEMM Raw Async Findings

## Current status

- Target kernel: `kernels/gemm/bf16_ampere/bf16_ampere_gemm.cu`
- Constraint: adaptation must stay inside ThunderKittens internals
- C500 does not support PTX, and `ldmatrix` is unavailable
- Current functional C500 path uses:
  - native `__builtin_mxc_ldg_b128_bsm_predicator` global-to-shared async staging
  - raw shared-memory staging buffers for A/B
  - existing TK/C500 MMA atom wrappers for compute

## What is proven

### 1. Swizzled TK shared tiles are not a valid BSM async destination

Directly targeting `st_bf<...>` / swizzled shared-tile layouts with C500 BSM async loads fails logical readback.

This was isolated with diagnostic probes and is why the current functional path uses raw shared buffers instead of `st_bf` as the async destination.

### 2. Raw shared async staging is functionally correct

`tests/c500/memory/raw_gemm_async.cu` validates raw async staging for the current GEMM-sized A/B tiles.

### 3. Current coarse-grained raw async path is correct but not yet fast enough

Observed functional kernel state:

- `4096^3` correctness is good (`err mean ~= 3.6e-05`)
- Throughput is around `37 TFLOP/s`
- This is well below the previous synchronous C500 baseline (`~54.6 TFLOP/s`)
- It is also far from the mcBLAS reference level

### 4. The current raw-vector consumer hypothesis is false

The latest targeted probes establish that the producer side is not the next blocker.

- `tests/c500/gemm/bf16_c500_stage_async_probe.cu` proves:
  - `issue_ab_stage_async(...)`
  - `load_stage_a_words(...)`
  - `load_stage_b_words(...)`
  return the expected raw 16B vectors from the current contiguous stage buffers.
- `tests/c500/gemm/bf16_c500_raw_vector_gemm_probe.cu` proves:
  - directly interpreting those vectors as MMA operands fails numerically
  - the failure is not just a full-GEMM artifact
  - even the first scalar-reference fragment for `A` and `B` has no exact match among the current
    `load_stage_*_words(...) + make_*_fragment(...)` candidates

This means:

- `make_a_fragment(...)` / `make_b_fragment(...)` are not enough to describe the real C500 operand mapping
- the missing semantic is not only `mma_k`
- the current fast-path abstraction is missing at least the explicit `m/n` selection dimension

In short:

- raw contiguous async producer: validated
- raw vector consumer contract: disproven
- next root cause: native operand-consumer mapping

### 5. Explicit operand semantics can already be represented inside TK

A new internal operand-layer abstraction now exists in:

- `include/arch/c500/gemm/bf16_operand_stage.cuh`

And a targeted probe now passes:

- `tests/c500/gemm/bf16_c500_operand_stage_probe.cu`
- `c500_gemm_bf16_operand_stage_contract`

What this proves:

- the current scalar-gather hot path can be lifted into an explicit `m/n/kg/half` operand representation
- that representation is sufficient to reconstruct the same per-lane A/B MMA fragments as the current correct path

This is important because it decouples two problems:

1. operand semantics
2. producer/storage layout

The project no longer needs to discover both at once.

### 6. Naive hot-path helper insertion regressed performance and was reverted

An attempted “equivalent refactor” that routed the mainloop through operand helpers caused the `4096^3`
BF16 GEMM example to regress from roughly `31 TFLOP/s` to about `27 TFLOP/s`.

That experiment was reverted immediately.

This establishes another useful constraint:

- even semantics-preserving abstraction changes can hurt the generated code shape on C500
- therefore the next migration must keep the validated operand abstraction off the hot path until the
  producer/consumer layout is ready to use it natively

## Async wait-semantics finding

For the current helper shape:

- `A stage`: `64x32 bf16`
- `B stage`: `32x64 bf16`
- loadgroup threads: `128`
- each thread issues:
  - `2` async ops for A
  - `2` async ops for B
  - total `4` async ops per thread per stage

So, for a true multistage pipeline, the reference-consistent steady-state threshold is:

- `wait_until<4>()` when leaving one future stage in flight
- `wait_until<0>()` only for prologue-without-next-stage and tail drain

This matches muxi / mctlass counting.

## Why `wait_until<4>()` alone did not fix performance

The failed experiments were not evidence that the threshold was wrong.

The real issue is stage granularity:

- the current helper stages a **whole K tile** (`32`) at once
- with only whole-stage buffers, there is no genuinely free slot to overlap a future whole-stage copy with the currently consumed stage unless staging is interleaved at finer granularity
- reference kernels achieve overlap by interleaving copy / consume at a more fine-grained schedule than the current ThunderKittens helper does

In other words:

- the current raw async path is **coarse-grained async**
- reference kernels are **fine-grained interleaved async**

## Most likely next optimization direction

The highest-probability next step is **not** more `gvmcnt` guesswork and **not** more tuning on the current
scalar-gather hot path.

It is one of:

1. Introduce a C500-native operand-consumer abstraction that carries `m/n/kg/half` semantics explicitly
2. Rework producer layout so shared-memory staging matches the real native consumer contract
3. Then rebuild the mainloop as a fine-grained interleaved multistage pipeline on top of that native layout

Reference comparison against muxi / mcTlass indicates the desired high-performance shape is:

- wave64-native consumer coordinates
- vector LDS payloads
- direct MMA operand feed
- interleaved async copy and MMA issue

That is a different architecture from the current:

- raw row-major stage
- scalar LDS gather
- `pack_pair(...)`
- whole-stage wait/consume cadence

## Practical conclusion

The current repository now has:

- a correct raw async staging path
- a passing raw async UT
- explicit proof that the old swizzled async destination path is invalid
- explicit proof that the current raw-vector fragment assumption is invalid

The remaining gap is now more specific than “performance structure”:

- first architectural gap: no correct native operand-consumer mapping yet
- second architectural gap: no native interleaved multistage pipeline yet

So the next serious implementation step should be:

- design and validate a C500-native stage/operand layer inside ThunderKittens
- then migrate the BF16 GEMM hot path to that layer
