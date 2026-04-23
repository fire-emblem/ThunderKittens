#pragma once

#include "sync_atom.cuh"

namespace bf16_c500_tk_cute_local::cute_tk {

struct fragment_atom {
    template <typename LdsType>
    __device__ __forceinline__ static LdsType load_from_shared(
        uint8_t *wsm_lds,
        const int (&lds_offset)[4],
        int idx) {
        return *reinterpret_cast<LdsType *>(wsm_lds + lds_offset[idx]);
    }

    template <typename ALdsType, typename BLdsType,
              typename StageContract =
                  ::bf16_c500_tk_cute_local::contracts::stage_contract>
    __device__ __forceinline__ static void prime_registers(
        ALdsType (&a)[4][4],
        BLdsType (&b)[4][4],
        uint8_t *wsm_lds,
        const int (&a_lds_offset)[4],
        const int (&b_lds_offset)[4]) {
        sync_atom::wait_gmem_async<12>();
        sync_atom::barrier();

        a[0][0] = load_from_shared<ALdsType>(wsm_lds, a_lds_offset, 0);
        a[0][1] = load_from_shared<ALdsType>(wsm_lds, a_lds_offset, 1);
        a[0][2] = load_from_shared<ALdsType>(wsm_lds, a_lds_offset, 2);
        a[0][3] = load_from_shared<ALdsType>(wsm_lds, a_lds_offset, 3);

        b[0][0] = load_from_shared<BLdsType>(wsm_lds, b_lds_offset, 0);
        b[0][1] = load_from_shared<BLdsType>(wsm_lds, b_lds_offset, 1);
        b[0][2] = load_from_shared<BLdsType>(wsm_lds, b_lds_offset, 2);
        b[0][3] = load_from_shared<BLdsType>(wsm_lds, b_lds_offset, 3);

        sync_atom::wait_gmem_async<8>();
        sync_atom::barrier();

        uint8_t *stage1_lds = wsm_lds + StageContract::stage_base_offset(1);
        a[1][0] = load_from_shared<ALdsType>(stage1_lds, a_lds_offset, 0);
        a[1][1] = load_from_shared<ALdsType>(stage1_lds, a_lds_offset, 1);
        a[1][2] = load_from_shared<ALdsType>(stage1_lds, a_lds_offset, 2);
        a[1][3] = load_from_shared<ALdsType>(stage1_lds, a_lds_offset, 3);

        b[1][0] = load_from_shared<BLdsType>(stage1_lds, b_lds_offset, 0);
        b[1][1] = load_from_shared<BLdsType>(stage1_lds, b_lds_offset, 1);
        b[1][2] = load_from_shared<BLdsType>(stage1_lds, b_lds_offset, 2);
        b[1][3] = load_from_shared<BLdsType>(stage1_lds, b_lds_offset, 3);
    }

    template <typename ALdsType, typename BLdsType,
              typename StageContract =
                  ::bf16_c500_tk_cute_local::contracts::stage_contract>
    __device__ __forceinline__ static void reload_stage(
        ALdsType (&a)[4][4],
        BLdsType (&b)[4][4],
        int lds_idx,
        uint8_t *wsm_lds2,
        const int (&a_lds_offset)[4],
        const int (&b_lds_offset)[4]) {
        a[lds_idx][0] = load_from_shared<ALdsType>(wsm_lds2, a_lds_offset, 0);
        a[lds_idx][1] = load_from_shared<ALdsType>(wsm_lds2, a_lds_offset, 1);
        a[lds_idx][2] = load_from_shared<ALdsType>(wsm_lds2, a_lds_offset, 2);
        a[lds_idx][3] = load_from_shared<ALdsType>(wsm_lds2, a_lds_offset, 3);
        b[lds_idx][0] = load_from_shared<BLdsType>(wsm_lds2, b_lds_offset, 0);
        b[lds_idx][1] = load_from_shared<BLdsType>(wsm_lds2, b_lds_offset, 1);
        b[lds_idx][2] = load_from_shared<BLdsType>(wsm_lds2, b_lds_offset, 2);
        b[lds_idx][3] = load_from_shared<BLdsType>(wsm_lds2, b_lds_offset, 3);
    }
};

} // namespace bf16_c500_tk_cute_local::cute_tk
