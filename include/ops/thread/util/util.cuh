/**
 * @file
 * @brief Utilities run by a single thread.
 */

#pragma once

#include "sync.cuh"
#if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)
#include "tma.cuh"
#endif

namespace kittens {

/* ----------   To prevent generic addressing, PTX  ---------- */

template<typename T> struct move {
    __device__ static inline void lds(T& dst, uint32_t src);
    __device__ static inline void sts(uint32_t dst, const T& src);
    __device__ static inline void ldg(T& dst, T* src);
    __device__ static inline void stg(T* dst, const T& src);
};
#ifdef KITTENS_C500
template<typename T>
__device__ static inline uint16_t c500_load_shared_u16(uint32_t src) {
    uint32_t tmp;
    asm volatile("ld.shared.u16 %0, [%1];\n" : "=r"(tmp) : "r"(src));
    return static_cast<uint16_t>(tmp);
}
template<typename T>
__device__ static inline void c500_store_shared_u16(uint32_t dst, uint16_t src) {
    asm volatile("st.shared.u16 [%1], %0;\n" : : "r"(static_cast<uint32_t>(src)), "r"(dst));
}
#endif
// unpacked types
template<> struct move<bf16> {
    __device__ static inline void lds(bf16& dst, uint32_t src) {
#ifdef KITTENS_C500
        uint16_t tmp = c500_load_shared_u16<bf16>(src);
        dst = *reinterpret_cast<bf16*>(&tmp);
#else
        asm volatile("ld.shared.b16 %0, [%1];\n" : "=h"(*(uint16_t*)&dst) : "r"(src));
#endif
    }
    __device__ static inline void sts(uint32_t dst, const bf16& src) {
#ifdef KITTENS_C500
        c500_store_shared_u16<bf16>(dst, *reinterpret_cast<const uint16_t*>(&src));
#else
        asm volatile("st.shared.b16 [%1], %0;\n" : : "h"(*(uint16_t*)&src), "r"(dst));
#endif
    }
    __device__ static inline void ldg(bf16& dst, bf16* src) {
#ifdef KITTENS_C500
        dst = *src;
#else
        asm volatile("ld.global.b16 %0, [%1];\n" : "=h"(*(uint16_t*)&dst) : "l"(src));
#endif
    }
    __device__ static inline void stg(bf16* dst, const bf16& src) {
#ifdef KITTENS_C500
        *dst = src;
#else
        asm volatile("st.global.b16 [%1], %0;\n" : : "h"(*(uint16_t*)&src), "l"(dst));
#endif
    }
};
template<> struct move<half> {
    __device__ static inline void lds(half& dst, uint32_t src) {
#ifdef KITTENS_C500
        uint16_t tmp = c500_load_shared_u16<half>(src);
        dst = *reinterpret_cast<half*>(&tmp);
#else
        asm volatile("ld.shared.b16 %0, [%1];\n" : "=h"(*(uint16_t*)&dst) : "r"(src));
#endif
    }
    __device__ static inline void sts(uint32_t dst, const half& src) {
#ifdef KITTENS_C500
        c500_store_shared_u16<half>(dst, *reinterpret_cast<const uint16_t*>(&src));
#else
        asm volatile("st.shared.b16 [%1], %0;\n" : : "h"(*(uint16_t*)&src), "r"(dst));
#endif
    }
    __device__ static inline void ldg(half& dst, half* src) {
#ifdef KITTENS_C500
        dst = *src;
#else
        asm volatile("ld.global.b16 %0, [%1];\n" : "=h"(*(uint16_t*)&dst) : "l"(src));
#endif
    }
    __device__ static inline void stg(half* dst, const half& src) {
#ifdef KITTENS_C500
        *dst = src;
#else
        asm volatile("st.global.b16 [%1], %0;\n" : : "h"(*(uint16_t*)&src), "l"(dst));
#endif
    }
};
template<> struct move<float> {
    __device__ static inline void lds(float& dst, uint32_t src) {
#ifdef KITTENS_C500
        uint32_t tmp;
        asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(tmp) : "r"(src));
        dst = *reinterpret_cast<float*>(&tmp);
#else
        asm volatile("ld.shared.f32 %0, [%1];\n" : "=f"(dst) : "r"(src));
#endif
    }
    __device__ static inline void sts(uint32_t dst, const float& src) {
#ifdef KITTENS_C500
        asm volatile("st.shared.u32 [%1], %0;\n" : : "r"(*reinterpret_cast<const uint32_t*>(&src)), "r"(dst));
#else
        asm volatile("st.shared.f32 [%1], %0;\n" : : "f"(src), "r"(dst));
#endif
    }
    __device__ static inline void ldg(float& dst, float* src) {
#ifdef KITTENS_C500
        dst = *src;
#else
        asm volatile("ld.global.f32 %0, [%1];\n" : "=f"(dst) : "l"(src));
#endif
    }
    __device__ static inline void stg(float* dst, const float& src) {
#ifdef KITTENS_C500
        *dst = src;
#else
        asm volatile("st.global.f32 [%1], %0;\n" : : "f"(src), "l"(dst));
#endif
    }
};
template<> struct move<int> {
    __device__ static inline void lds(int& dst, uint32_t src) {
        asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(dst) : "r"(src));
    }
    __device__ static inline void sts(uint32_t dst, const int& src) {
        asm volatile("st.shared.u32 [%1], %0;\n" : : "r"(src), "r"(dst));
    }
    __device__ static inline void ldg(int& dst, int* src) {
#ifdef KITTENS_C500
        dst = *src;
#else
        asm volatile("ld.global.u32 %0, [%1];\n" : "=r"(dst) : "l"(src));
#endif
    }
    __device__ static inline void stg(int* dst, const int& src) {
#ifdef KITTENS_C500
        *dst = src;
#else
        asm volatile("st.global.u32 [%1], %0;\n" : : "r"(src), "l"(dst));
#endif
    }
};
// packed types
template<> struct move<bf16_2> {
    __device__ static inline void lds(bf16_2& dst, uint32_t src) {
#ifdef KITTENS_C500
        asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(*(uint32_t*)&dst) : "r"(src));
#else
        asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(*(uint32_t*)&dst) : "r"(src));
#endif
    }
    __device__ static inline void sts(uint32_t dst, const bf16_2& src) {
        asm volatile("st.shared.b32 [%1], %0;\n" : : "r"(*(uint32_t*)&src), "r"(dst));
    }
    __device__ static inline void ldg(bf16_2& dst, bf16_2* src) {
#ifdef KITTENS_C500
        dst = *src;
#else
        asm volatile("ld.global.b32 %0, [%1];\n" : "=r"(*(uint32_t*)&dst) : "l"(src));
#endif
    }
    __device__ static inline void stg(bf16_2* dst, const bf16_2& src) {
#ifdef KITTENS_C500
        *dst = src;
#else
        asm volatile("st.global.b32 [%1], %0;\n" : : "r"(*(uint32_t*)&src), "l"(dst));
#endif
    }
    __device__ static inline void ldsm4(bf16_2& dst1, bf16_2& dst2, bf16_2& dst3, bf16_2& dst4, uint32_t src) {
#ifdef KITTENS_C500
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n" :
                     "=r"(*(uint32_t*)&dst1), "=r"(*(uint32_t*)&dst2), "=r"(*(uint32_t*)&dst3), "=r"(*(uint32_t*)&dst4) : "r"(src));
#else
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0, %1, %2, %3}, [%4];\n" :
                     "=r"(*(uint32_t*)&dst1), "=r"(*(uint32_t*)&dst2), "=r"(*(uint32_t*)&dst3), "=r"(*(uint32_t*)&dst4) : "r"(src));
#endif
    }
    __device__ static inline void ldsm4t(bf16_2& dst1, bf16_2& dst2, bf16_2& dst3, bf16_2& dst4, uint32_t src) {
#ifdef KITTENS_C500
        asm volatile("ldmatrix.sync.aligned.trans.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n" :
                     "=r"(*(uint32_t*)&dst1), "=r"(*(uint32_t*)&dst2), "=r"(*(uint32_t*)&dst3), "=r"(*(uint32_t*)&dst4) : "r"(src));
#else
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared::cta.b16 {%0, %1, %2, %3}, [%4];\n" :
                     "=r"(*(uint32_t*)&dst1), "=r"(*(uint32_t*)&dst2), "=r"(*(uint32_t*)&dst3), "=r"(*(uint32_t*)&dst4) : "r"(src));
#endif
    }
    __device__ static inline void stsm4(uint32_t dst, bf16_2& src1, bf16_2& src2, bf16_2& src3, bf16_2& src4) {
#ifdef KITTENS_C500
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%4], {%0, %1, %2, %3};\n" ::
                     "r"(*(uint32_t*)&src1), "r"(*(uint32_t*)&src2), "r"(*(uint32_t*)&src3), "r"(*(uint32_t*)&src4), "r"(dst));
#else
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared::cta.b16 [%4], {%0, %1, %2, %3};\n" ::
                     "r"(*(uint32_t*)&src1), "r"(*(uint32_t*)&src2), "r"(*(uint32_t*)&src3), "r"(*(uint32_t*)&src4), "r"(dst));
#endif
    }
    __device__ static inline void stsm4t(uint32_t dst, bf16_2& src1, bf16_2& src2, bf16_2& src3, bf16_2& src4) {
#ifdef KITTENS_C500
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%4], {%0, %1, %2, %3};\n" ::
                     "r"(*(uint32_t*)&src1), "r"(*(uint32_t*)&src2), "r"(*(uint32_t*)&src3), "r"(*(uint32_t*)&src4), "r"(dst));
#else
        asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared::cta.b16 [%4], {%0, %1, %2, %3};\n" ::
                     "r"(*(uint32_t*)&src1), "r"(*(uint32_t*)&src2), "r"(*(uint32_t*)&src3), "r"(*(uint32_t*)&src4), "r"(dst));
#endif
    }
};
template<> struct move<half_2> {
    __device__ static inline void lds(half_2& dst, uint32_t src) {
        asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(*(uint32_t*)&dst) : "r"(src));
    }
    __device__ static inline void sts(uint32_t dst, const half_2& src) {
        asm volatile("st.shared.b32 [%1], %0;\n" : : "r"(*(uint32_t*)&src), "r"(dst));
    }
    __device__ static inline void ldg(half_2& dst, half_2* src) {
#ifdef KITTENS_C500
        dst = *src;
#else
        asm volatile("ld.global.b32 %0, [%1];\n" : "=r"(*(uint32_t*)&dst) : "l"(src));
#endif
    }
    __device__ static inline void stg(half_2* dst, const half_2& src) {
#ifdef KITTENS_C500
        *dst = src;
#else
        asm volatile("st.global.b32 [%1], %0;\n" : : "r"(*(uint32_t*)&src), "l"(dst));
#endif
    }
    __device__ static inline void ldsm4(half_2& dst1, half_2& dst2, half_2& dst3, half_2& dst4, uint32_t src) {
#ifdef KITTENS_C500
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n" :
                     "=r"(*(uint32_t*)&dst1), "=r"(*(uint32_t*)&dst2), "=r"(*(uint32_t*)&dst3), "=r"(*(uint32_t*)&dst4) : "r"(src));
#else
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0, %1, %2, %3}, [%4];\n" :
                     "=r"(*(uint32_t*)&dst1), "=r"(*(uint32_t*)&dst2), "=r"(*(uint32_t*)&dst3), "=r"(*(uint32_t*)&dst4) : "r"(src));
#endif
    }
    __device__ static inline void ldsm4t(half_2& dst1, half_2& dst2, half_2& dst3, half_2& dst4, uint32_t src) {
#ifdef KITTENS_C500
        asm volatile("ldmatrix.sync.aligned.trans.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n" :
                     "=r"(*(uint32_t*)&dst1), "=r"(*(uint32_t*)&dst2), "=r"(*(uint32_t*)&dst3), "=r"(*(uint32_t*)&dst4) : "r"(src));
#else
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared::cta.b16 {%0, %1, %2, %3}, [%4];\n" :
                     "=r"(*(uint32_t*)&dst1), "=r"(*(uint32_t*)&dst2), "=r"(*(uint32_t*)&dst3), "=r"(*(uint32_t*)&dst4) : "r"(src));
#endif
    }
    __device__ static inline void stsm4(uint32_t dst, half_2& src1, half_2& src2, half_2& src3, half_2& src4) {
#ifdef KITTENS_C500
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%4], {%0, %1, %2, %3};\n" ::
                     "r"(*(uint32_t*)&src1), "r"(*(uint32_t*)&src2), "r"(*(uint32_t*)&src3), "r"(*(uint32_t*)&src4), "r"(dst));
#else
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared::cta.b16 [%4], {%0, %1, %2, %3};\n" ::
                     "r"(*(uint32_t*)&src1), "r"(*(uint32_t*)&src2), "r"(*(uint32_t*)&src3), "r"(*(uint32_t*)&src4), "r"(dst));
#endif
    }
    __device__ static inline void stsm4t(uint32_t dst, half_2& src1, half_2& src2, half_2& src3, half_2& src4) {
#ifdef KITTENS_C500
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%4], {%0, %1, %2, %3};\n" ::
                     "r"(*(uint32_t*)&src1), "r"(*(uint32_t*)&src2), "r"(*(uint32_t*)&src3), "r"(*(uint32_t*)&src4), "r"(dst));
#else
        asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared::cta.b16 [%4], {%0, %1, %2, %3};\n" ::
                     "r"(*(uint32_t*)&src1), "r"(*(uint32_t*)&src2), "r"(*(uint32_t*)&src3), "r"(*(uint32_t*)&src4), "r"(dst));
#endif
    }
};
template<> struct move<float2> {
    __device__ static inline void lds(float2& dst, uint32_t src) {
#ifdef KITTENS_C500
        uint32_t x, y;
        asm volatile("ld.shared.u32 %0, [%2];\n" : "=r"(x) : "0"(x), "r"(src));
        asm volatile("ld.shared.u32 %0, [%2];\n" : "=r"(y) : "0"(y), "r"(src + 4));
        dst.x = *reinterpret_cast<float*>(&x);
        dst.y = *reinterpret_cast<float*>(&y);
#else
        asm volatile("ld.shared.v2.f32 {%0, %1}, [%2];\n" : "=f"(dst.x), "=f"(dst.y) : "r"(src));
#endif
    }
    __device__ static inline void sts(uint32_t dst, const float2& src) {
#ifdef KITTENS_C500
        asm volatile("st.shared.u32 [%1], %0;\n" : : "r"(*reinterpret_cast<const uint32_t*>(&src.x)), "r"(dst));
        asm volatile("st.shared.u32 [%1], %0;\n" : : "r"(*reinterpret_cast<const uint32_t*>(&src.y)), "r"(dst + 4));
#else
        asm volatile("st.shared.v2.f32 [%2], {%0, %1};\n" : : "f"(src.x), "f"(src.y), "r"(dst));
#endif
    }
    __device__ static inline void ldg(float2& dst, float2* src) {
#ifdef KITTENS_C500
        dst = *src;
#else
        asm volatile("ld.global.v2.f32 {%0, %1}, [%2];\n" : "=f"(dst.x), "=f"(dst.y) : "l"(src));
#endif
    }
    __device__ static inline void stg(float2* dst, const float2& src) {
#ifdef KITTENS_C500
        *dst = src;
#else
        asm volatile("st.global.v2.f32 [%2], {%0, %1};\n" : : "f"(src.x), "f"(src.y), "l"(dst));
#endif
    }
};
template<> struct move<float4> {
    __device__ static inline void lds(float4& dst, uint32_t src) {
#ifdef KITTENS_C500
        uint32_t x, y, z, w;
        asm volatile("ld.shared.u32 %0, [%2];\n" : "=r"(x) : "0"(x), "r"(src));
        asm volatile("ld.shared.u32 %0, [%2];\n" : "=r"(y) : "0"(y), "r"(src + 4));
        asm volatile("ld.shared.u32 %0, [%2];\n" : "=r"(z) : "0"(z), "r"(src + 8));
        asm volatile("ld.shared.u32 %0, [%2];\n" : "=r"(w) : "0"(w), "r"(src + 12));
        dst.x = *reinterpret_cast<float*>(&x);
        dst.y = *reinterpret_cast<float*>(&y);
        dst.z = *reinterpret_cast<float*>(&z);
        dst.w = *reinterpret_cast<float*>(&w);
#else
        asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n" : "=f"(dst.x), "=f"(dst.y), "=f"(dst.z), "=f"(dst.w) : "r"(src));
#endif
    }
    __device__ static inline void sts(uint32_t dst, const float4& src) {
#ifdef KITTENS_C500
        asm volatile("st.shared.u32 [%1], %0;\n" : : "r"(*reinterpret_cast<const uint32_t*>(&src.x)), "r"(dst));
        asm volatile("st.shared.u32 [%1], %0;\n" : : "r"(*reinterpret_cast<const uint32_t*>(&src.y)), "r"(dst + 4));
        asm volatile("st.shared.u32 [%1], %0;\n" : : "r"(*reinterpret_cast<const uint32_t*>(&src.z)), "r"(dst + 8));
        asm volatile("st.shared.u32 [%1], %0;\n" : : "r"(*reinterpret_cast<const uint32_t*>(&src.w)), "r"(dst + 12));
#else
        asm volatile("st.shared.v4.f32 [%4], {%0, %1, %2, %3};\n" : : "f"(src.x), "f"(src.y), "f"(src.z), "f"(src.w), "r"(dst));
#endif
    }
    __device__ static inline void ldg(float4& dst, float4* src) {
#ifdef KITTENS_C500
        dst = *src;
#else
        asm volatile("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];\n" : "=f"(dst.x), "=f"(dst.y), "=f"(dst.z), "=f"(dst.w) : "l"(src));
#endif
    }
    __device__ static inline void stg(float4* dst, const float4& src) {
#ifdef KITTENS_C500
        *dst = src;
#else
        asm volatile("st.global.v4.f32 [%4], {%0, %1, %2, %3};\n" : : "f"(src.x), "f"(src.y), "f"(src.z), "f"(src.w), "l"(dst));
#endif
    }
};
#if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)
template<> struct move<fp8e4m3_4> {
    __device__ static inline void ldsm4(fp8e4m3_4& dst1, fp8e4m3_4& dst2, fp8e4m3_4& dst3, fp8e4m3_4& dst4, uint32_t src) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0, %1, %2, %3}, [%4];\n" :
                     "=r"(*(uint32_t*)&dst1),  "=r"(*(uint32_t*)&dst2), "=r"(*(uint32_t*)&dst3), "=r"(*(uint32_t*)&dst4) : "r"(src));
    }
    __device__ static inline void stsm4(uint32_t dst, fp8e4m3_4& src1, fp8e4m3_4& src2, fp8e4m3_4& src3, fp8e4m3_4& src4) {
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared::cta.b16 [%4], {%0, %1, %2, %3};\n" ::
                     "r"(*(uint32_t*)&src1), "r"(*(uint32_t*)&src2), "r"(*(uint32_t*)&src3), "r"(*(uint32_t*)&src4), "r"(dst));
    }

};
template<> struct move<fp8e5m2_4> {
    __device__ static inline void ldsm4(fp8e5m2_4& dst1, fp8e5m2_4& dst2, fp8e5m2_4& dst3, fp8e5m2_4& dst4, uint32_t src) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0, %1, %2, %3}, [%4];\n" :
                     "=r"(*(uint32_t*)&dst1),  "=r"(*(uint32_t*)&dst2), "=r"(*(uint32_t*)&dst3), "=r"(*(uint32_t*)&dst4) : "r"(src));
    }
    __device__ static inline void stsm4(uint32_t dst, fp8e5m2_4& src1, fp8e5m2_4& src2, fp8e5m2_4& src3, fp8e5m2_4& src4) {
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared::cta.b16 [%4], {%0, %1, %2, %3};\n" ::
                     "r"(*(uint32_t*)&src1), "r"(*(uint32_t*)&src2), "r"(*(uint32_t*)&src3), "r"(*(uint32_t*)&src4), "r"(dst));
    }
};
#endif

/* ----------   Constants for Cache policies  ---------- */

enum cache_policy {
    NORMAL = 0,
    EVICT_FIRST = 1,
    EVICT_LAST = 2
};
template<cache_policy policy> __device__ inline uint64_t make_cache_policy() {
    uint64_t cache_policy_val;
    constexpr float fraction = 1.0f;
    static_assert(policy == cache_policy::EVICT_FIRST || policy == cache_policy::EVICT_LAST, "Unexpected cache policy");
#ifdef KITTENS_C500
    cache_policy_val = 0;
#else
    if constexpr (policy == cache_policy::EVICT_FIRST) {
        asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, %1;\n" : "=l"(cache_policy_val) : "f"(fraction));
    }
    else {
        asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, %1;\n" : "=l"(cache_policy_val) : "f"(fraction));
    }
#endif
    return cache_policy_val;
}

/* CLC scheduler operations */

#ifdef KITTENS_BLACKWELL

namespace clc {

struct handle {
    uint4 internal_value;
}; // note that this is an opaque type, so the value should not be accessed directly.

struct result {
    uint32_t success;
    uint32_t x;
    uint32_t y;
    uint32_t z;
};

/**
 * @brief Schedules a new threadblock. Must be called by a single thread in the entire CTA cluster.
 *        The caller must wait on the semaphore with tma::cluster::expect_bytes followed by tma::cluster::wait.
 *        The handle is multicasted to all CTAs in the cluster and signals the semaphore of all CTAs in the cluster.
 * @param h The CLC handle.
 * @param sem The semaphore that the caller will wait on.
 */
__device__ static inline void schedule(handle &h, semaphore &sem) {
    asm volatile("{clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [%0], [%1];}"
        :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&h.internal_value))), "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&sem)))
        : "memory"
    );
}

/**
 * @brief Queries the result of a schedule operation. Calling this again after failure is undefined behavior.
 * @param h The CLC handle.
 */
__device__ static inline result query(handle &h) {
    result r;
    asm volatile(
        "{\n"
        ".reg .pred SUCCESS;\n"
        ".reg .b128 CLC_HANDLE;\n"
        "ld.shared.b128 CLC_HANDLE, [%4];\n"
        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 SUCCESS, CLC_HANDLE;\n"
        "selp.u32 %0, 1, 0, SUCCESS;\n"
        "@!SUCCESS bra.uni DONE;\n"
        "clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%1, %2, %3, _}, CLC_HANDLE;\n"
        "fence.proxy.async.shared::cta;\n"
        "DONE:\n"
        "}"
        : "=r"(r.success), "=r"(r.x), "=r"(r.y), "=r"(r.z)
        : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&h.internal_value)))
        : "memory"
    );
    return r;
}

} // namespace clc

#endif

#if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)

/**
 * @brief Programmatic Dependent Kernel Launch (PDL) utilities. Available on Hopper and later.
 *
 * PDL allows partial overlap between two consecutive kernels in the same stream.
 *
 * @note The secondary kernel must be launched with `cudaLaunchAttributeProgrammaticStreamSerialization`
 *       attribute and `programmaticStreamSerializationAllowed` set to 1.
 */
namespace pdl {

/**
 * @brief Signals that a primary kernel has completed its dependent work, enabling a secondary kernel to launch.
 *
 * @note The secondary kernel will only launch when all threadblocks in the primary kernel have called this function.
 *       If a threadblock does not call this, the arrival is implicitly triggered at threadblock exit.
 * @note This does not guarantee memory visibility. For memory visibility, the secondary kernel must call wait().
 */
__device__ static inline void arrive() {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

/**
 * @brief Blocks until the primary kernel fully completes and flushes memory.
 */
__device__ static inline void wait() {
    asm volatile("griddepcontrol.wait;" ::: "memory");
}

}

#endif

} // namespace kittens
