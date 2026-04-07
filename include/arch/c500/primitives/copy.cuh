#pragma once

#include "../async.cuh"

namespace kittens::arch::c500::primitives {

// Stable entry boundary for C500 GEMM copy primitives during the transition.
using kittens::arch::c500::async_copy_128b;
using kittens::arch::c500::async_token;
using kittens::arch::c500::combine;

} // namespace kittens::arch::c500::primitives
