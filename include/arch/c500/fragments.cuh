#pragma once

#include "mma_atoms.cuh"

namespace kittens::arch::c500 {

using bf16_mma_atom = mma_bf16_16x16x16_fp32;
using f16_mma_atom = mma_f16_16x16x16_fp32;

} // namespace kittens::arch::c500
