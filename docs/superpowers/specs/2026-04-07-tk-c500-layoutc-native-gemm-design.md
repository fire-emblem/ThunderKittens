# TK C500 LayoutC Native GEMM Design

## Goal

Add a C500-native BF16 GEMM path inside ThunderKittens that follows the effective `muxi_hgemm_layoutC` hierarchy directly:

- CTA tile `128x128x128`
- `Stage=4`
- `256` threads = `4 x wave64`
- native-layout `A`
- native-layout `B`
- continuous row-major `C`

This path is performance-first. It does not try to preserve Ampere-era fragment or pipeline semantics.

## Constraints

- Stay inside ThunderKittens sources and abstractions.
- Do not replace TK kernels with external GEMM or SDK GEMM APIs.
- Do not add substitute helper GEMM kernels or pure-CUDA fallback GEMM paths.
- Unsupported PTX/CUDA primitives must be replaced with C500-compatible builtins or MACA language/runtime features.

## Tactical Decision

Do not continue extending the transitional `layoutA` bridge path for the performance target.

Instead:

1. Add one dedicated C500 `layoutC` family under `include/arch/c500/gemm/families/`.
2. Port the kernel body structure from muxi into TK namespace and file layout.
3. Keep the public example entry in `kernels/gemm/bf16_c500/bf16_c500_gemm.cu`.
4. Precompute native `A/B` layouts once before timing.
5. Benchmark only the TK path against the existing reference path in the example.

## Data Contracts

### A native layout

Logical source: row-major `A[M, K]`

Physical layout:

- shape: `[M / 16][K / 8][16][8]`
- contiguous order after `view(m/16, 16, k/8, 8).permute(0, 2, 1, 3)`

### B native layout

Logical source: row-major `B[K, N]`

First reinterpret as logical `B_t[N, K]` where `B_t[n, k] = B[k, n]`.

Physical layout:

- shape: `[K / 32][N / 16][4][16][8]`
- contiguous order matching muxi `layoutB`

### C output

- logical and physical layout: row-major `C[M, N]`

## Implementation Scope

### New family

Add `bf16_c500_layoutc_128x128x128_stage4` with:

- C500 builtins for async global-to-shared
- direct shared-memory offsets matching muxi schedule
- native vector loads from shared
- direct `__builtin_mxc_mma_16x16x16bf16`
- direct row-major `C` export

### Example integration

Add a native-layout benchmark path in `kernels/gemm/bf16_c500/bf16_c500_gemm.cu`:

- keep existing reference GEMM generation
- build native `A/B` once on host
- allocate native device buffers
- run the new family
- compare output against existing reference

## Success Criteria

- `kernels/gemm/bf16_c500/bf16_c500_gemm.cu` compiles for `GPU=C500`
- `4096^3` BF16 path runs through the new family
- output is validated against the example reference path
- measured performance materially exceeds the current ~17 TFLOP/s TK path and becomes the new baseline for further tuning
