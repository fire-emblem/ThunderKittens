/**
 * @file
 * @brief Functions for a warpgroup to collaboratively transfer  data directly between global memory and registers and back.
 */

/**
 * @brief Collaboratively loads data into register vectors from a source array in global memory.
 *
 * @tparam RV The register vector type.
 * @tparam U The data type of the source array.
 * @param[out] dst The destination register vector to load data into.
 * @param[in] src The source array in global memory to load data from.
 */
template<ducks::rv::all RV, ducks::gl::all GL>
__device__ inline static void load(RV &dst, const GL &src, const coord<rv<typename RV::T, GROUP_WARPS*RV::length, typename RV::layout>> &idx) {
    if constexpr (GROUP_WARPS == 1) {
        using T2 = RV::dtype;
        using U = typename GL::dtype;
        using U2 = base_types::packing<U>::packed_type;
        using T = base_types::packing<T2>::unpacked_type;

        U *src_ptr = (U*)&src[(idx.template unit_coord<-1, 3>())];
        int laneid = ::kittens::laneid();
        
#ifdef KITTENS_C500
        if constexpr (std::is_same_v<typename RV::layout, align_l>) {
            #pragma unroll
            for(int i = 0; i < RV::outer_dim; i++) {
                const int base = i*16 + (laneid >> 4);
                dst[i][0].x = base_types::convertor<T, U>::convert(src_ptr[base +  0]);
                dst[i][0].y = base_types::convertor<T, U>::convert(src_ptr[base +  4]);
                dst[i][1].x = base_types::convertor<T, U>::convert(src_ptr[base +  8]);
                dst[i][1].y = base_types::convertor<T, U>::convert(src_ptr[base + 12]);
            }
        }
        else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
            #pragma unroll
            for(int i = 0; i < RV::outer_dim; i++) {
                const T v = base_types::convertor<T, U>::convert(src_ptr[i*16 + (laneid & 0xf)]);
                dst[i][0].x = v;
                dst[i][0].y = v;
            }
        }
        else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {
            #pragma unroll
            for(int i = 0; i < RV::outer_dim; i++) {
                const int logical_idx = i*32 + (laneid & 0x1f);
                if(logical_idx < RV::length) {
                    dst[i][0] = base_types::convertor<T, U>::convert(src_ptr[logical_idx]);
                }
            }
        }
#else
        if constexpr (std::is_same_v<typename RV::layout, align_l>) {
            #pragma unroll
            for(auto w = 0; w < (RV::outer_dim+3)/4; w++) {
                int idx = w*64 + (laneid/4)*8 + 2*(laneid%4);
                int o_dim = w*4 + (laneid/4) / 2;
                int i_dim = (laneid/4) % 2;
                // this should be a maximally coalesced load.
                if(idx < RV::outer_dim*16)
                    dst[o_dim][i_dim] = base_types::convertor<T2, U2>::convert(*(U2*)&src_ptr[idx]);
            }
            // now we need to do a bunch of shuffle_sync's to make sure everyone has everything they need.
            #pragma unroll
            for(auto w = 0; w < RV::outer_dim; w++) {
                int leader = 8*(w%4) + (laneid%4); // repeats every 64 columns
                dst[w][0] = packed_shfl_sync(MASK_ALL, dst[w][0], leader);
                dst[w][1] = packed_shfl_sync(MASK_ALL, dst[w][1], leader+4);
            }
        }
        else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
            // really hoping https://stackoverflow.com/questions/15029765/is-coalescing-triggered-for-accessing-memory-in-reverse-order is still true
            // otherwise there will be some pain :/
            #pragma unroll
            for(auto w = 0; w < (RV::outer_dim+1)/2; w++) {
                int idx = w*32 + (laneid%4)*8 + (laneid/4);
                int o_dim = w*2 + (laneid%4) / 2;
                // this should be a maximally coalesced load.
                if(idx < RV::outer_dim*16) {
                    T tmp = base_types::convertor<T, U>::convert(src_ptr[idx]);
                    if(laneid%2==0) dst[o_dim][0].x = tmp;
                    else dst[o_dim][0].y = tmp;
                }
            }
            // now we need to do a bunch of shuffle_sync's to make sure everyone has everything they need.
            #pragma unroll
            for(auto w = 0; w < RV::outer_dim; w++) {
                int leader = (laneid/4)*4 + 2*(w%2); // repeats every 64 columns
                dst[w][0].x = __shfl_sync(MASK_ALL, dst[w][0].x, leader);
                dst[w][0].y = __shfl_sync(MASK_ALL, dst[w][0].y, leader+1);
            }
        }
        else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {
            #pragma unroll
            for(auto w = 0; w < RV::outer_dim; w++) {
                if(w < RV::outer_dim-1 || RV::length%32 == 0 || laneid<16) {
                    dst[w][0] = base_types::convertor<T, U>::convert(src_ptr[w*32 + laneid]);
                }
            }
        }
#endif
    }
    else {
        // Call warp level load
        ::kittens::group<1>::load(dst, src, coord<RV>(idx.b, idx.d, idx.r, idx.c*GROUP_WARPS+warpid()));
    }
}
/**
 * @brief Collaboratively stores data from register vectors to a destination array in global memory.
 *
 * @tparam RV The register vector type.
 * @tparam U The data type of the destination array.
 * @param[out] dst The destination array in global memory to store data into.
 * @param[in] src The source register vector to store data from.
 */
template<ducks::rv::all RV, ducks::gl::all GL>
__device__ inline static void store(GL &dst, const RV &src, const coord<rv<typename RV::T, GROUP_WARPS*RV::length, typename RV::layout>> &idx) {
    if constexpr (GROUP_WARPS == 1) {
        using T2 = RV::dtype;
        using U = typename GL::dtype;
        using U2 = base_types::packing<U>::packed_type;
        using T = base_types::packing<T2>::unpacked_type;
        
        U *dst_ptr = (U*)&dst[(idx.template unit_coord<-1, 3>())];
        int laneid = ::kittens::laneid();

#ifdef KITTENS_C500
        if constexpr (std::is_same_v<typename RV::layout, align_l>) {
            if((laneid & 0xf) == 0) {
                #pragma unroll
                for(int i = 0; i < RV::outer_dim; i++) {
                    const int base = i*16 + (laneid >> 4);
                    dst_ptr[base +  0] = base_types::convertor<U, T>::convert(src[i][0].x);
                    dst_ptr[base +  4] = base_types::convertor<U, T>::convert(src[i][0].y);
                    dst_ptr[base +  8] = base_types::convertor<U, T>::convert(src[i][1].x);
                    dst_ptr[base + 12] = base_types::convertor<U, T>::convert(src[i][1].y);
                }
            }
        }
        else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
            if(laneid < 16) {
                #pragma unroll
                for(int i = 0; i < RV::outer_dim; i++) {
                    dst_ptr[i*16 + laneid] = base_types::convertor<U, T>::convert(src[i][0].x);
                }
            }
        }
        else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {
            if((laneid & 0x1f) == laneid) {
                #pragma unroll
                for(int i = 0; i < RV::outer_dim; i++) {
                    const int logical_idx = i*32 + laneid;
                    if(logical_idx < RV::length) {
                        dst_ptr[logical_idx] = base_types::convertor<U, T>::convert(src[i][0]);
                    }
                }
            }
        }
#else
        if constexpr (std::is_same_v<typename RV::layout, align_l>) {
            #pragma unroll
            for(auto w = 0; w < (RV::outer_dim+3)/4; w++) {
                int idx = w*64 + (laneid/4)*8 + 2*(laneid%4);
                int o_dim = w*4 + (laneid/4) / 2;
                int i_dim = (laneid/4) % 2;
                // this should be a maximally coalesced store. I hope!
                if(idx < RV::outer_dim*16)
                    *(U2*)&dst_ptr[idx] = base_types::convertor<U2, T2>::convert(src[o_dim][i_dim]);
            }
        }
        else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
            // really hoping https://stackoverflow.com/questions/15029765/is-coalescing-triggered-for-accessing-memory-in-reverse-order is still true
            // otherwise there will be some pain :/
            #pragma unroll
            for(auto w = 0; w < (RV::outer_dim+1)/2; w++) {
                int idx = w*32 + (laneid%4)*8 + (laneid/4);
                int o_dim = w*2 + (laneid%4) / 2;
                // this should be a maximally coalesced load.
                if(idx < RV::outer_dim*16) {
                    U tmp;
                    if(laneid%2==0) tmp = base_types::convertor<U, T>::convert(src[o_dim][0].x);
                    else tmp = base_types::convertor<U, T>::convert(src[o_dim][0].y);
                    dst_ptr[idx] = tmp;
                }
            }
        }
        else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {
            #pragma unroll
            for(auto w = 0; w < RV::outer_dim; w++) {
                if(w < RV::outer_dim-1 || RV::length%32 == 0 || laneid<16) {
                    dst_ptr[w*32 + laneid] = base_types::convertor<U, T>::convert(src[w][0]);
                }
            }
        }
#endif
    }
    else {
        // Call warp level store
        ::kittens::group<1>::store(dst, src, coord<RV>(idx.b, idx.d, idx.r, idx.c*GROUP_WARPS+warpid()));
    }
}
