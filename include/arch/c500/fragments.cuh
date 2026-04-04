#pragma once

#include "mma_atoms.cuh"

namespace kittens::arch::c500 {

template<>
struct fragment_a<bf16_mma_atom> {
    uint32_t reg[bf16_mma_atom::a_registers] = {};
};

template<>
struct fragment_b<bf16_mma_atom> {
    uint32_t reg[bf16_mma_atom::b_registers] = {};
};

template<>
struct fragment_c<bf16_mma_atom> {
    float reg[bf16_mma_atom::c_registers] = {};
};

} // namespace kittens::arch::c500
