#pragma once

// Unified primitive library for C500 GEMM kernels
// Organized in layers: Hardware → Operations → Pipeline → Composition

// ============================================================================
// Layer 1: Hardware Primitives (direct mapping to MXC builtins)
// ============================================================================
#include "mxc_builtins.cuh"  // Native types: float4_native, bf16_native, etc.

// ============================================================================
// Layer 2: Compute Primitives (tensor core operations)
// ============================================================================
#include "compute/mma_atom.cuh"  // MMA operations: fma_pair, accumulate_kgroup

// ============================================================================
// Layer 3: Memory Primitives (load/store operations)
// ============================================================================
#include "async_copy.cuh"             // Low-level BSM async copy
#include "pipeline/copy_atom.cuh"     // High-level BSM issue interface
#include "sync.cuh"                   // Low-level barrier primitives

// ============================================================================
// Layer 4: Pipeline Primitives (scheduling and synchronization)
// ============================================================================
#include "pipeline/sync_atom.cuh"         // Barrier wrappers
#include "pipeline/schedule_atom.cuh"      // Pipeline scheduling policies
#include "pipeline/fragment_atom.cuh"      // Shared→register fragment loads
#include "pipeline/prologue_atom.cuh"      // Prologue stage loading
#include "pipeline/reload_atom.cuh"        // Steady-state reload
#include "pipeline/tail_atom.cuh"          // Tail iteration and drain
#include "pipeline/issue_order_atom.cuh"   // Load issue ordering
#include "pipeline/mainloop_atom.cuh"      // Mainloop compute primitives

// ============================================================================
// Layer 5: Structure Primitives (data layout and geometry)
// ============================================================================
#include "structure/geometry_atom.cuh"            // Geometry providers
#include "structure/layoutc_geometry_atom.cuh"    // Layout-C specific geometry
#include "structure/stage_layout_atom.cuh"        // Stage memory layout
#include "structure/square_tt_thread_map_atom.cuh" // Thread mapping for 256x256x64

// ============================================================================
// Layer 6: Epilogue Primitives (output processing)
// ============================================================================
#include "epilogue/store_ops_atom.cuh"    // Store operations (canonical)
#include "epilogue/bias_atom.cuh"         // Bias loading
#include "epilogue/epilogue_atom.cuh"     // Unified epilogue interface

namespace bf16_c500_tk_cute_local::cute_tk::primitives {

// Re-export all atoms at the primitives namespace level for convenience
using mma_atom = ::bf16_c500_tk_cute_local::cute_tk::mma_atom;
using copy_atom = ::bf16_c500_tk_cute_local::cute_tk::copy_atom;
using sync_atom = ::bf16_c500_tk_cute_local::cute_tk::sync_atom;
using schedule_atom = ::bf16_c500_tk_cute_local::cute_tk::schedule_atom;
using fragment_atom = ::bf16_c500_tk_cute_local::cute_tk::fragment_atom;
using tail_atom = ::bf16_c500_tk_cute_local::cute_tk::tail_atom;
using geometry_atom = ::bf16_c500_tk_cute_local::cute_tk::geometry_atom;
using epilogue_atom = ::bf16_c500_tk_cute_local::cute_tk::epilogue_atom;
using mainloop_atom = ::bf16_c500_tk_cute_local::cute_tk::mainloop_atom;

} // namespace bf16_c500_tk_cute_local::cute_tk::primitives
