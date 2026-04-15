# TK Cute C500: Example-Derived Geometry Findings

## Purpose

This note captures what we learned by importing the hand-written MetaX
GEMM example into `kernels/gemm/bf16_c500_tk_cute_local` and testing
which parts of that implementation materially help the current tk-cute
family stack.

The focus is intentionally narrow:

- What from the example gives real speedups?
- What does not help when directly grafted into current layoutc?
- What should be abstracted next inside tk-cute?

## Tested ideas

We evaluated three levels of import.

### 1. Whole-family import: `tn_example`

We imported the example-style TN BF16 kernel as an independent family:

- `cute_tk_tn_example_bf16_128x128x128_stage4`

This path preserves the example's:

- thread map
- shared-memory swizzle
- software pipeline schedule
- edge predication structure

### 2. Whole-schedule graft into layoutc

We wrapped the same imported schedule as a layoutc-tuned candidate:

- `cute_tk_layoutc_tn_tuned_bf16_128x128x128_stage4`

The goal was to test whether the example's whole schedule could simply
replace current layoutc logic.

### 3. Geometry-only experiment

We parameterized the imported TN example kernel by geometry policy and
compared:

- swizzled geometry (original example mapping)
- linear geometry (same kernel, same pipeline, no swizzle)

This isolates the value of the example's thread/LDS mapping from the
rest of its schedule.

## Key result 1: Whole schedule import is useful as an independent family

The imported `tn_example` family is competitive on several benchmark
shapes and wins on some of them.

Representative results:

| Shape | `tn_example` TFLOP/s | Current family TFLOP/s | Winner |
| --- | ---: | ---: | --- |
| 1664x1024x16384 | 151.72 | 152.92 | current layoutc |
| 2048x2048x2048 | 98.77 | 100.18 | current layoutc |
| 4096x4096x4096 | 149.66 | 148.49 | tn_example |
| 4608x256x3584 | 76.70 | 62.87 | tn_example |
| 37888x256x3584 | 133.96 | 119.11 | tn_example |
| 37888x128x3584 | 126.71 | 101.72 | tn_example |
| 4608x128x3584 | 39.99 | 48.20 | current reusea |
| 3584x128x3584 | 30.15 | 39.73 | current reusea |
| 3584x128x18944 | 42.34 | 64.17 | current reusea |

Conclusion:

- The example is worth keeping as a family candidate.
- It is not universally dominant.
- It is strongest on several large-throughput shapes.
- Current specialized reusea families still win on some narrow-N cases.

## Key result 2: Whole-schedule graft into layoutc is not a win

We compared current layoutc against a layoutc-tuned candidate that
reused the imported TN-example schedule.

Results:

| Shape | `layoutc_tuned` TFLOP/s | `layoutc` TFLOP/s | Difference |
| --- | ---: | ---: | ---: |
| 1664x1024x16384 | 152.45 | 153.20 | -0.75 |
| 2048x2048x2048 | 96.84 | 101.22 | -4.38 |
| 4096x4096x4096 | 150.02 | 150.22 | -0.20 |

Conclusion:

Directly replacing current layoutc scheduling with the imported
example schedule is not a profitable direction.

This suggests current layoutc already captures many of the right ideas
for its own layout semantics, and that the example's value is not in
serving as a drop-in layoutc replacement.

## Key result 3: Geometry / swizzle is a decisive optimization

This was the clearest result.

We ran the imported TN example kernel with two geometry modes:

- original swizzled example geometry
- linearized geometry with the same schedule and host path

Results:

| Shape | Swizzled geometry TFLOP/s | Linear geometry TFLOP/s | Gain |
| --- | ---: | ---: | ---: |
| 1664x1024x16384 | 152.00 | 54.36 | +97.64 |
| 2048x2048x2048 | 97.27 | 40.06 | +57.21 |
| 4096x4096x4096 | 147.95 | 52.98 | +94.97 |

This is not a small effect. It is a factor-of-2-to-3 difference.

Conclusion:

The example's thread map / LDS swizzle / geometry contract is the most
clearly valuable optimization component we tested.

## What is actually worth borrowing from the example

### 1. Geometry contract, not just a clever offset formula

The example's value is not one XOR expression by itself.
The value is the full contract formed by:

- thread-to-fragment ownership
- global-load indexing
- shared-memory landing layout
- LDS fragment reload offsets
- the MMA consumption order that matches those offsets

This must be treated as a first-class geometry provider.

### 2. Family-owned shape legality

The example handles shapes that are not simple global `% 128` cases.
The runner change to let families declare runtime shape support is a
real improvement and should remain.

### 3. Host-traits as a semantic boundary

The imported family was only practical to benchmark cleanly after giving
it explicit host traits. That pattern should be preserved.

## What is *not* worth borrowing directly

### 1. Copying the whole hand-written schedule into current layoutc

Measured and not useful.

### 2. Treating one imported kernel as the universal answer

The data already shows shape-dependent winners. We should keep multiple
families and route by measured behavior.

## Recommended next abstraction step

The next scaling move should be:

### Introduce geometry providers as first-class tk-cute components

For example:

- current layoutc geometry provider
- example-style swizzled geometry provider
- future narrow-N / reusea geometry providers

Families should be able to choose geometry providers independently from:

- software pipeline policy
- epilogue/store policy
- host layout policy

That is the cleanest way to make the imported insight reusable.

## Recommended product direction

Short term:

- keep `tn_example` as an independent family candidate
- route shapes by measured performance instead of ideology

Medium term:

- abstract geometry providers
- avoid another whole-schedule graft experiment
- test geometry reuse inside existing families only after the provider
  abstraction exists

## Bottom line

The example's biggest concrete gift to tk-cute is not its full kernel
body. It is its geometry contract.

The experiments strongly indicate:

- whole-schedule import is useful as a *family*
- whole-schedule graft into layoutc is not useful
- geometry/swizzle import is a decisive optimization ingredient

