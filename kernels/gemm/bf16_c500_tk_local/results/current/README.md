# Current Performance Snapshot

This directory captures the current cached/smoke performance results gathered
before parameter-aligned local-tk retuning.

Files:

- `../muxi_baseline_results.csv`
  - Partial muxi baseline cache.
  - Current row count: `195`.
  - Families currently present:
    - `gemm_layoutAB_ContinuousC`: `128`
    - `gemm_layoutA`: `16`
    - `gemm_layoutABC`: `16`
    - `muxi_hgemm_layoutC`: `32`
    - `muxi_hgemm_layout`: `3`
  - The last two `muxi_hgemm_layout` cases were interrupted by an OOM in the
    older non-resumable script. The benchmark script has since been updated to
    flush incrementally and resume safely.

- `mcblas_baseline_smoke.csv`
  - Current mcBLAS smoke-only snapshot.
  - Not the final full cache.

- `muxi_local_layoutc_smoke.csv`
  - Current muxi-vs-local smoke comparison for `muxi_hgemm_layoutC` against
    local `layoutC`.

- `muxi_local_continuousc_smoke.csv`
  - Current muxi-vs-local smoke comparison for
    `gemm_layoutAB_ContinuousC` against local `continuousC`.

Status at this commit:

- `layoutC` local fp16 path is runnable and comparable against
  `muxi_hgemm_layoutC` on `N % 128 == 0` shapes.
- `continuousC` local bf16/fp16 path is runnable and correct, but still uses a
  fixed local family and is not yet parameter-aligned with muxi's dispatch
  choices.
- `muxi` and `mcBLAS` baseline generators were converted to resumable,
  incremental-cache scripts so future runs do not need to restart from zero.
