#include "testing_flags.cuh"

#ifdef TEST_C500_GEMM_BACKEND_COMPILE_PROBE

#include "kittens.cuh"
#include "arch/c500/traits.cuh"
#include "arch/c500/fragments.cuh"
#include "arch/c500/mma.cuh"

using namespace kittens;

__global__ void bf16_c500_backend_compile_probe() {
#ifdef KITTENS_C500
    static_assert(kittens::arch::c500::wave_traits::kWaveSize == 64);
    using atom = kittens::arch::c500::bf16_mma_atom;
    static_assert(atom::M == 16 && atom::N == 16 && atom::K == 16);
    kittens::arch::c500::fragment_a<atom> a{};
    kittens::arch::c500::fragment_b<atom> b{};
    kittens::arch::c500::fragment_c<atom> c{};
    auto d = kittens::arch::c500::mma(atom{}, a, b, c);
    (void)d;
#endif
}

#endif
