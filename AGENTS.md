# ThunderKittens Project Rules

## Priority 1

- Do not replace ThunderKittens kernel implementations with external library or SDK GEMM APIs.
- Do not introduce separate local helper kernels or pure-CUDA fallback GEMM paths as substitutes.
- C500 adaptation must be implemented inside ThunderKittens abstractions and internals.
- When a CUDA/PTX-specific primitive is unsupported on C500, replace that primitive with a ThunderKittens-internal C500-compatible implementation using supported compiler builtins, MACA-compatible language features, or compatible runtime functionality.

## Current Scope

- The active bring-up target is `kernels/gemm/bf16_ampere/bf16_ampere_gemm.cu`.
- Prefer small, reviewable changes that progressively reduce compile and runtime gaps for this target.
