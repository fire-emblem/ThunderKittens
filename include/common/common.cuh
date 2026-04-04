/**
 * @file
 * @brief A collection of common resources on which ThunderKittens depends.
 */
 

#pragma once

#include "base_types.cuh"
#include "base_ops.cuh"
#include "util.cuh"

#if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)
#include "multimem.cuh"
#endif

#ifdef KITTENS_C500
#include "../arch/c500/fragment_layouts.cuh"
#include "../arch/c500/mma_atoms.cuh"
#include "../arch/c500/copy_atoms.cuh"
#include "../arch/c500/epilogue_atoms.cuh"
#endif
