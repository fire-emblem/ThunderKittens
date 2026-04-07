# Current Performance Snapshot

This directory captures the current checked-in performance snapshots gathered
while aligning local-tk kernels against muxi and mcBLAS baselines.

Files:

- `../muxi_baseline_results.csv`
  - Full muxi baseline cache.
  - Current row count: `197`.
  - Families currently present:
    - `gemm_layoutAB_ContinuousC`: `128`
    - `gemm_layoutA`: `16`
    - `gemm_layoutABC`: `16`
    - `muxi_hgemm_layoutC`: `32`
    - `muxi_hgemm_layout`: `5`
  - The final two ultra-large `muxi_hgemm_layout` shapes are recorded with
    `status=oom` because they exceed the available device memory on this
    system.

- `../mcblas_baseline_results.csv`
  - Full mcBLAS baseline cache.
  - Current row count: `177`.
  - Most shapes are `status=ok`.
  - Two ultra-large `fp16` `muxi_hgemm_layout`-style shapes that cannot be
    completed by mcBLAS on this system are recorded as `status=failed`.

- `../muxi_local_compare_results.csv`
  - Full three-way comparison table.
  - Current row count: `197`.
  - Contains:
    - muxi runtime / reference results
    - mcBLAS cached results when available
    - local-tk runtime results when the corresponding local family is
      available
  - Local rows use muxi-style timing by default to reduce benchmark-method bias.
  - Status distribution:
    - `muxi_status`: `195 ok`, `2 oom`
    - `mcblas_status`: `195 ok`, `2 failed`
    - `local_status`: `24 ok`, `136 unsupported_shape`, `2 skipped`, `35 empty`
      The empty local status rows correspond to benchmark families that do not
      yet have a local-tk counterpart (`gemm_layoutA`, `gemm_layoutABC`,
      `muxi_hgemm_layout`).

- `mcblas_baseline_smoke.csv`
  - Early mcBLAS smoke snapshot preserved for quick inspection.

- `muxi_local_layoutc_smoke.csv`
  - Early muxi-vs-local smoke comparison for `muxi_hgemm_layoutC` against
    local `layoutC`, prior to the full comparison table.

- `muxi_local_continuousc_smoke.csv`
  - Early muxi-vs-local smoke comparison for
    `gemm_layoutAB_ContinuousC` against local `continuousC`, prior to the full
    comparison table.

Status at this commit:

- `layoutC` local fp16 path is runnable and comparable against
  `muxi_hgemm_layoutC` on `N % 128 == 0` shapes.
- `continuousC` local bf16/fp16 path now includes parameter-aligned reuseA
  families for the key `N=128/256` benchmark cases.
- The remaining giant-shape anomalies are now explicitly surfaced in cache
  status columns instead of aborting the full benchmark pass.
- `muxi` and `mcBLAS` baseline generators were converted to resumable,
  incremental-cache scripts so future runs do not need to restart from zero.
