# C500 Muxi LayoutA Native Path Root Cause

## Summary

The current ThunderKittens C500 `layoutA` native path is not merely under-optimized. It is mathematically incomplete for `K >= 256`, so schedule tuning on top of it is not a valid next step.

## Evidence

### 1. Minimal no-refill case already fails

Command:

```bash
make -C kernels/gemm/bf16_c500 clean
make -C kernels/gemm/bf16_c500 run GPU=C500 \
  EXTRA_NVCCFLAGS='-DBF16_C500_USE_LAYOUTA_NATIVE=1 \
                   -DBF16_C500_EXPERIMENTAL_LAYOUTA_NATIVE=1 \
                   -DBF16_C500_PROBLEM_M=128 \
                   -DBF16_C500_PROBLEM_N=128 \
                   -DBF16_C500_PROBLEM_K=256 \
                   -DBF16_C500_WARMUP_ITERS=1 \
                   -DBF16_C500_PROFILE_ITERS=1'
```

Observed:

- kernel runs
- result is wrong
- this case has `num_k_tiles = 2` and `prefetch_tiles = 2`
- therefore there is no steady-state refill traffic at all

This isolates the bug to the native stage-consume path itself, not the refill schedule.

### 2. Experimental native path can trap on large `K`

For `4096^3`, enabling the experimental native path can trigger a device memory violation. This is a secondary failure mode that appears after the primary correctness issue.

### 3. Fallback layoutA path remains correct

Disabling the experimental native path restores correctness, but performance drops sharply. This is expected because the fallback path uses the operand-stage bridge path rather than a true muxi-native consumer layout.

### 4. Single-stage `K=128` probe still fails

The focused test `tests/c500/gemm/bf16_c500_layouta_native_stage_probe.cu` constructs exactly one native `128x128x128` stage and consumes it without any steady-state refill logic.

Observed:

- `layouta store` path fails
- `standard export` path also fails with the same first mismatch
- therefore the failure is not caused by the `layoutA`-specific accumulator store path

This isolates the remaining bug to one of:

- native shared-bank interpretation
- register-bank assembly from shared into `a[4][4]` / `b[4][4]`
- higher-level bank/index semantics around the muxi stage contract

### 5. Simple `vec[4]` pair regrouping does not rescue correctness

The stage probe was extended to search several high-value ways of splitting each loaded `native_vec[4]` into the two MMA calls required per `K-group`, including:

- `[0,1]` + `[2,3]`
- reversed adjacent pairs
- crossed pairs such as `[0,2]` + `[1,3]`
- crossed pairs such as `[0,3]` + `[1,2]`

Observed:

- no tested pair regrouping comes close to correctness
- the best candidate still leaves essentially the entire `128x128` tile incorrect

Implication:

- the remaining bug is not a small local word-order mistake inside one 16-byte vector
- the bug is at a larger semantic level than simple pair slicing

## Root Cause

The current native `layoutA` implementation in TK is a simplified approximation of the muxi kernel:

- it loads one `native_vec[4]` set for A and one `native_vec[4]` set for B per stage
- it performs only two MMA calls per accumulator per stage

That is not enough to consume a full `128`-wide K tile.

For a `16x16x16` BF16 MMA atom:

- each output atom needs `128 / 16 = 8` MMA contributions per `K=128` stage
- the current simplified path only applies `2`

So the current path is not just scheduled differently from muxi; it is computing a different dataflow.

After correcting that first issue and adding a full-stage consumer, the probe evidence now shows a second-order root cause:

- TK still does not yet match muxi's native bank/index contract for `layoutA`
- the mismatch survives both store-path replacement and local `vec[4]` pair regrouping
- therefore the remaining work is to reproduce muxi's higher-level stage-bank interpretation, not to keep trying local word shuffles

## Implication

Do not keep tuning:

- wait windows
- segmented refill order
- barrier placement
- launch bounds

on top of the current simplified native path.

Those can matter later, but only after the native path is rewritten to consume the full muxi stage contract.

## Correct Next Step

The next valid implementation step is to port the muxi `layoutA` mainloop structure more faithfully into TK internals:

- preserve the `4 x 16KB` stage ring physical layout
- preserve the muxi per-stage A/B register sets, not a flattened approximation
- preserve the full per-stage MMA consumption count for `K=128`
- preserve muxi's stage-bank ownership semantics across the four waves, not just its per-thread load formulas
- only after that, reintroduce interleaved refill scheduling

In short:

- first restore full native mathematical semantics
- then match muxi's bank/index contract exactly
- only then optimize pipeline overlap
