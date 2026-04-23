#pragma once

// Unified primitive library for C500 GEMM kernels
// Organized in layers: Hardware → Atoms → Layout → Pipeline

// ============================================================================
// Layer 1: Hardware Primitives (direct mapping to MXC builtins)
// ============================================================================
#include "arch/mxc_builtins.cuh"  // Native types: float4_native, bf16_native, etc.
#include "arch/mma.cuh"           // MMA operations
#include "arch/async_copy.cuh"    // Async copy operations
#include "arch/sync.cuh"         // Barrier primitives

// ============================================================================
// Layer 2: Abstract Atoms (hardware-agnostic interface)
// ============================================================================
#include "atom/mma_atom.cuh"      // MMA atom
#include "atom/copy_atom.cuh"     // Copy atom
#include "atom/sync_atom.cuh"     // Sync atom

// ============================================================================
// Layer 3: Layout Primitives (memory layout and geometry)
// ============================================================================
#include "layout/stage_geometry.cuh"  // Stage geometry descriptor
#include "layout/stage_layout.cuh"    // Stage layout atom
#include "layout/thread_map.cuh"      // Thread mapping

namespace bf16_c500_tk_cute_local::primitives {

// Re-export atoms for convenience
using mma_atom = mma_atom_t;
using copy_atom = copy_atom_t;
using sync_atom = sync_atom_t;
using stage_layout_atom = stage_layout_atom_t;
using stage_geometry = stage_geometry_t;
using square_tile_thread_map = square_tile_thread_map_t;

} // namespace bf16_c500_tk_cute_local::primitives