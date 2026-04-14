/**
 * @file
 * @brief Reduction operations mapping tiles to vectors.
 */

/**
 * @brief Perform a row-wise reduction on a matrix in row-major layout.
 *
 * This function template performs a parallel reduction across the rows of a matrix using a specified operation.
 * It leverages warp shuffle functions for efficient intra-warp communication.
 *
 * @tparam op The operation to be applied for reduction.
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type with row layout.
 * @tparam reset A boolean flag indicating whether to reset the accumulator (ignore src_accum) or not.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when reset is false.
 */
template<typename op, ducks::rv::all V, ducks::rt::row_layout T, bool reset>
__device__ static inline void row_reduce(V &row_accum, const T &src, const V &src_accum) {
    // I actually like these static asserts because they give more verbose errors when things go wrong.
    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout>::col_vec_layout>); // compatible layout
    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>); // compatible type
    static_assert(V::outer_dim == T::height); // compatible size

    using dtype = V::dtype;

#ifdef KITTENS_C500
    using scalar = typename base_types::packing<dtype>::unpacked_type;
    const int warp_laneid = ::kittens::laneid();
    const int row = warp_laneid & 0xf;
    constexpr uint64_t full_mask = 0xffffffffffffffffull;
    #pragma unroll
    for(int i = 0; i < T::height; i++) {
        scalar accum_row = op::template op<scalar>(
            op::template op<scalar>(src.tiles[i][0].data[0].x, src.tiles[i][0].data[0].y),
            op::template op<scalar>(src.tiles[i][0].data[1].x, src.tiles[i][0].data[1].y)
        );
        #pragma unroll
        for(int j = 1; j < T::width; j++) {
            accum_row = op::template op<scalar>(accum_row, src.tiles[i][j].data[0].x);
            accum_row = op::template op<scalar>(accum_row, src.tiles[i][j].data[0].y);
            accum_row = op::template op<scalar>(accum_row, src.tiles[i][j].data[1].x);
            accum_row = op::template op<scalar>(accum_row, src.tiles[i][j].data[1].y);
        }

        scalar accum_row_0 = __shfl_sync(full_mask, accum_row, row +  0, WARP_THREADS);
        scalar accum_row_1 = __shfl_sync(full_mask, accum_row, row + 16, WARP_THREADS);
        scalar accum_row_2 = __shfl_sync(full_mask, accum_row, row + 32, WARP_THREADS);
        scalar accum_row_3 = __shfl_sync(full_mask, accum_row, row + 48, WARP_THREADS);
        accum_row = op::template op<scalar>(
            op::template op<scalar>(accum_row_0, accum_row_1),
            op::template op<scalar>(accum_row_2, accum_row_3)
        );

        dtype accum_packed = base_types::packing<dtype>::pack(accum_row);
        if(reset) {
            row_accum[i][0] = accum_packed;
        }
        else {
            row_accum[i][0] = op::template op<dtype>(src_accum[i][0], accum_packed);
        }
    }
#else
    const int leader = threadIdx.x & 0x1C; // 11100 in binary
    #pragma unroll
    for(int i = 0; i < T::height; i++) {
        dtype accum_top_row    = op::template op<dtype>(src.tiles[i][0].data[0], src.tiles[i][0].data[2]);
        dtype accum_bottom_row = op::template op<dtype>(src.tiles[i][0].data[1], src.tiles[i][0].data[3]);
        #pragma unroll
        for(int j = 1; j < T::width; j++) {
            #pragma unroll
            for(int k = 0; k < T::packed_per_tile; k+=2) {
                accum_top_row    = op::template op<dtype>(accum_top_row,    src.tiles[i][j].data[k+0]);
                accum_bottom_row = op::template op<dtype>(accum_bottom_row, src.tiles[i][j].data[k+1]);
            }
        }
        dtype accum_packed;
        accum_packed.x = op::template op<typename base_types::packing<dtype>::unpacked_type>(accum_top_row.x,    accum_top_row.y);
        accum_packed.y = op::template op<typename base_types::packing<dtype>::unpacked_type>(accum_bottom_row.x, accum_bottom_row.y);

        // Now we need to do a lil shuffle to make everyone happy.

        accum_packed = op::template op<dtype>(accum_packed, packed_shfl_down_sync(MASK_ALL, accum_packed, 2));
        accum_packed = op::template op<dtype>(accum_packed, packed_shfl_down_sync(MASK_ALL, accum_packed, 1));

        accum_packed = packed_shfl_sync(MASK_ALL, accum_packed, leader);

        if(reset) {
            row_accum[i][0] = accum_packed;
        }
        else {
            row_accum[i][0] = op::template op<dtype>(src_accum[i][0], accum_packed);
        }
    }
#endif
}
/**
 * @brief Perform a row-wise reduction on a matrix in column-major layout.
 *
 * This function template performs a parallel reduction across the rows of a matrix using a specified operation.
 * It leverages warp shuffle functions for efficient intra-warp communication and is optimized for column-major matrices.
 *
 * @tparam op The operation to be applied for reduction.
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type with column layout.
 * @tparam reset A boolean flag indicating whether to reset the accumulator (ignore src_accum) or not.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when reset is false.
 */
template<typename op, ducks::rv::all V, ducks::rt::col_layout T, bool reset>
__device__ static inline void row_reduce(V &row_accum, const T &src, const V &src_accum) {
    // I actually like these static asserts because they give more verbose errors when things go wrong.
    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout>::col_vec_layout>); // compatible layout
    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>); // compatible type
    static_assert(V::outer_dim == T::height); // compatible size

    using dtype = V::dtype;

#ifdef KITTENS_C500
    using scalar = typename base_types::packing<dtype>::unpacked_type;
    constexpr uint64_t full_mask = 0xffffffffffffffffull;
    #pragma unroll
    for(int i = 0; i < T::height; i++) {
        scalar accum_r0  = src.tiles[i][0].data[0].x;
        scalar accum_r4  = src.tiles[i][0].data[0].y;
        scalar accum_r8  = src.tiles[i][0].data[1].x;
        scalar accum_r12 = src.tiles[i][0].data[1].y;
        #pragma unroll
        for(int j = 1; j < T::width; j++) {
            accum_r0  = op::template op<scalar>(accum_r0,  src.tiles[i][j].data[0].x);
            accum_r4  = op::template op<scalar>(accum_r4,  src.tiles[i][j].data[0].y);
            accum_r8  = op::template op<scalar>(accum_r8,  src.tiles[i][j].data[1].x);
            accum_r12 = op::template op<scalar>(accum_r12, src.tiles[i][j].data[1].y);
        }

        accum_r0  = op::template op<scalar>(accum_r0,  __shfl_down_sync(full_mask, accum_r0,  8, 16));
        accum_r0  = op::template op<scalar>(accum_r0,  __shfl_down_sync(full_mask, accum_r0,  4, 16));
        accum_r0  = op::template op<scalar>(accum_r0,  __shfl_down_sync(full_mask, accum_r0,  2, 16));
        accum_r0  = op::template op<scalar>(accum_r0,  __shfl_down_sync(full_mask, accum_r0,  1, 16));
        accum_r4  = op::template op<scalar>(accum_r4,  __shfl_down_sync(full_mask, accum_r4,  8, 16));
        accum_r4  = op::template op<scalar>(accum_r4,  __shfl_down_sync(full_mask, accum_r4,  4, 16));
        accum_r4  = op::template op<scalar>(accum_r4,  __shfl_down_sync(full_mask, accum_r4,  2, 16));
        accum_r4  = op::template op<scalar>(accum_r4,  __shfl_down_sync(full_mask, accum_r4,  1, 16));
        accum_r8  = op::template op<scalar>(accum_r8,  __shfl_down_sync(full_mask, accum_r8,  8, 16));
        accum_r8  = op::template op<scalar>(accum_r8,  __shfl_down_sync(full_mask, accum_r8,  4, 16));
        accum_r8  = op::template op<scalar>(accum_r8,  __shfl_down_sync(full_mask, accum_r8,  2, 16));
        accum_r8  = op::template op<scalar>(accum_r8,  __shfl_down_sync(full_mask, accum_r8,  1, 16));
        accum_r12 = op::template op<scalar>(accum_r12, __shfl_down_sync(full_mask, accum_r12, 8, 16));
        accum_r12 = op::template op<scalar>(accum_r12, __shfl_down_sync(full_mask, accum_r12, 4, 16));
        accum_r12 = op::template op<scalar>(accum_r12, __shfl_down_sync(full_mask, accum_r12, 2, 16));
        accum_r12 = op::template op<scalar>(accum_r12, __shfl_down_sync(full_mask, accum_r12, 1, 16));

        accum_r0  = __shfl_sync(full_mask, accum_r0,  0, 16);
        accum_r4  = __shfl_sync(full_mask, accum_r4,  0, 16);
        accum_r8  = __shfl_sync(full_mask, accum_r8,  0, 16);
        accum_r12 = __shfl_sync(full_mask, accum_r12, 0, 16);

        dtype accum_top_rows{accum_r0, accum_r4};
        dtype accum_bottom_rows{accum_r8, accum_r12};
        if(reset) {
            row_accum[i][0] = accum_top_rows;
            row_accum[i][1] = accum_bottom_rows;
        }
        else {
            row_accum[i][0] = op::template op<dtype>(src_accum[i][0], accum_top_rows);
            row_accum[i][1] = op::template op<dtype>(src_accum[i][1], accum_bottom_rows);
        }
    }
#else
    const int leader = threadIdx.x & 0x3; // 00011 in binary
    #pragma unroll
    for(int i = 0; i < T::height; i++) {
        dtype accum_top_rows    = op::template op<dtype>(src.tiles[i][0].data[0], src.tiles[i][0].data[1]);
        dtype accum_bottom_rows = op::template op<dtype>(src.tiles[i][0].data[2], src.tiles[i][0].data[3]);
        #pragma unroll
        for(int j = 1; j < T::width; j++) {
            #pragma unroll
            for(int k = 0; k < T::packed_per_tile/2; k++) {
                accum_top_rows    = op::template op<dtype>(accum_top_rows,    src.tiles[i][j].data[k+0]);
                accum_bottom_rows = op::template op<dtype>(accum_bottom_rows, src.tiles[i][j].data[k+2]);
            }
        }

        // Now we need to do a lil shuffle to make everyone happy.

        accum_top_rows = op::template op<dtype>(accum_top_rows, packed_shfl_down_sync(MASK_ALL, accum_top_rows, 16));
        accum_top_rows = op::template op<dtype>(accum_top_rows, packed_shfl_down_sync(MASK_ALL, accum_top_rows, 8));
        accum_top_rows = op::template op<dtype>(accum_top_rows, packed_shfl_down_sync(MASK_ALL, accum_top_rows, 4));

        accum_bottom_rows = op::template op<dtype>(accum_bottom_rows, packed_shfl_down_sync(MASK_ALL, accum_bottom_rows, 16));
        accum_bottom_rows = op::template op<dtype>(accum_bottom_rows, packed_shfl_down_sync(MASK_ALL, accum_bottom_rows, 8));
        accum_bottom_rows = op::template op<dtype>(accum_bottom_rows, packed_shfl_down_sync(MASK_ALL, accum_bottom_rows, 4));

        accum_top_rows    = packed_shfl_sync(MASK_ALL, accum_top_rows,    leader);
        accum_bottom_rows = packed_shfl_sync(MASK_ALL, accum_bottom_rows, leader);

        if(reset) {
            row_accum[i][0] = accum_top_rows;
            row_accum[i][1] = accum_bottom_rows;
        }
        else {
            row_accum[i][0] = op::template op<dtype>(src_accum[i][0], accum_top_rows);
            row_accum[i][1] = op::template op<dtype>(src_accum[i][1], accum_bottom_rows);
        }
    }
#endif
}

// Col reduction.
/**
 * @brief Perform a column-wise reduction on a matrix in row-major layout.
 *
 * This function template performs a parallel reduction across the columns of a matrix using a specified operation.
 * It leverages warp shuffle functions for efficient intra-warp communication and is optimized for row-major matrices.
 *
 * @tparam op The operation to be applied for reduction.
 * @tparam V The vector type for the column accumulator.
 * @tparam T The matrix type with row layout.
 * @tparam reset A boolean flag indicating whether to reset the accumulator (ignore src_accum) or not.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when reset is false.
 */
template<typename op, ducks::rv::all V, ducks::rt::row_layout T, bool reset>
__device__ static inline void col_reduce(V &col_accum, const T &src, const V &src_accum) {
    // I actually like these static asserts because they give more verbose errors when things go wrong.
    KITTENS_CHECK_WARP
    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout>::row_vec_layout>); // compatible layout
    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>); // compatible type
    static_assert(V::outer_dim == T::width); // compatible size

    using dtype = V::dtype;

#ifdef KITTENS_C500
    using scalar = typename base_types::packing<dtype>::unpacked_type;
    constexpr uint64_t full_mask = 0xffffffffffffffffull;
    #pragma unroll
    for(int j = 0; j < T::width; j++) {
        scalar accum_c0  = src.tiles[0][j].data[0].x;
        scalar accum_c4  = src.tiles[0][j].data[0].y;
        scalar accum_c8  = src.tiles[0][j].data[1].x;
        scalar accum_c12 = src.tiles[0][j].data[1].y;
        #pragma unroll
        for(int i = 1; i < T::height; i++) {
            accum_c0  = op::template op<scalar>(accum_c0,  src.tiles[i][j].data[0].x);
            accum_c4  = op::template op<scalar>(accum_c4,  src.tiles[i][j].data[0].y);
            accum_c8  = op::template op<scalar>(accum_c8,  src.tiles[i][j].data[1].x);
            accum_c12 = op::template op<scalar>(accum_c12, src.tiles[i][j].data[1].y);
        }

        accum_c0  = op::template op<scalar>(accum_c0,  __shfl_down_sync(full_mask, accum_c0,  8, 16));
        accum_c0  = op::template op<scalar>(accum_c0,  __shfl_down_sync(full_mask, accum_c0,  4, 16));
        accum_c0  = op::template op<scalar>(accum_c0,  __shfl_down_sync(full_mask, accum_c0,  2, 16));
        accum_c0  = op::template op<scalar>(accum_c0,  __shfl_down_sync(full_mask, accum_c0,  1, 16));
        accum_c4  = op::template op<scalar>(accum_c4,  __shfl_down_sync(full_mask, accum_c4,  8, 16));
        accum_c4  = op::template op<scalar>(accum_c4,  __shfl_down_sync(full_mask, accum_c4,  4, 16));
        accum_c4  = op::template op<scalar>(accum_c4,  __shfl_down_sync(full_mask, accum_c4,  2, 16));
        accum_c4  = op::template op<scalar>(accum_c4,  __shfl_down_sync(full_mask, accum_c4,  1, 16));
        accum_c8  = op::template op<scalar>(accum_c8,  __shfl_down_sync(full_mask, accum_c8,  8, 16));
        accum_c8  = op::template op<scalar>(accum_c8,  __shfl_down_sync(full_mask, accum_c8,  4, 16));
        accum_c8  = op::template op<scalar>(accum_c8,  __shfl_down_sync(full_mask, accum_c8,  2, 16));
        accum_c8  = op::template op<scalar>(accum_c8,  __shfl_down_sync(full_mask, accum_c8,  1, 16));
        accum_c12 = op::template op<scalar>(accum_c12, __shfl_down_sync(full_mask, accum_c12, 8, 16));
        accum_c12 = op::template op<scalar>(accum_c12, __shfl_down_sync(full_mask, accum_c12, 4, 16));
        accum_c12 = op::template op<scalar>(accum_c12, __shfl_down_sync(full_mask, accum_c12, 2, 16));
        accum_c12 = op::template op<scalar>(accum_c12, __shfl_down_sync(full_mask, accum_c12, 1, 16));

        accum_c0  = __shfl_sync(full_mask, accum_c0,  0, 16);
        accum_c4  = __shfl_sync(full_mask, accum_c4,  0, 16);
        accum_c8  = __shfl_sync(full_mask, accum_c8,  0, 16);
        accum_c12 = __shfl_sync(full_mask, accum_c12, 0, 16);

        dtype accum_left_cols{accum_c0, accum_c4};
        dtype accum_right_cols{accum_c8, accum_c12};
        if(reset) {
            col_accum[j][0] = accum_left_cols;
            col_accum[j][1] = accum_right_cols;
        }
        else {
            col_accum[j][0] = op::template op<dtype>(src_accum[j][0], accum_left_cols);
            col_accum[j][1] = op::template op<dtype>(src_accum[j][1], accum_right_cols);
        }
    }
#else
    const int leader = threadIdx.x & 0x3; // 00011 in binary
    #pragma unroll
    for(int j = 0; j < T::width; j++) {
        dtype accum_left_cols  = op::template op<dtype>(src.tiles[0][j].data[0], src.tiles[0][j].data[1]);
        dtype accum_right_cols = op::template op<dtype>(src.tiles[0][j].data[2], src.tiles[0][j].data[3]);
        #pragma unroll
        for(int i = 1; i < T::height; i++) {
            #pragma unroll
            for(int k = 0; k < T::packed_per_tile/2; k++) {
                accum_left_cols  = op::template op<dtype>(accum_left_cols,  src.tiles[i][j].data[k+0]);
                accum_right_cols = op::template op<dtype>(accum_right_cols, src.tiles[i][j].data[k+2]);
            }
        }

        // Now we need to do a lil shuffle to make everyone happy.

        accum_left_cols = op::template op<dtype>(accum_left_cols, packed_shfl_down_sync(MASK_ALL, accum_left_cols, 16));
        accum_left_cols = op::template op<dtype>(accum_left_cols, packed_shfl_down_sync(MASK_ALL, accum_left_cols, 8));
        accum_left_cols = op::template op<dtype>(accum_left_cols, packed_shfl_down_sync(MASK_ALL, accum_left_cols, 4));

        accum_right_cols = op::template op<dtype>(accum_right_cols, packed_shfl_down_sync(MASK_ALL, accum_right_cols, 16));
        accum_right_cols = op::template op<dtype>(accum_right_cols, packed_shfl_down_sync(MASK_ALL, accum_right_cols, 8));
        accum_right_cols = op::template op<dtype>(accum_right_cols, packed_shfl_down_sync(MASK_ALL, accum_right_cols, 4));

        accum_left_cols  = packed_shfl_sync(MASK_ALL, accum_left_cols,  leader);
        accum_right_cols = packed_shfl_sync(MASK_ALL, accum_right_cols, leader);

        if(reset) {
            col_accum[j][0] = accum_left_cols;
            col_accum[j][1] = accum_right_cols;
        }
        else {
            col_accum[j][0] = op::template op<dtype>(src_accum[j][0], accum_left_cols);
            col_accum[j][1] = op::template op<dtype>(src_accum[j][1], accum_right_cols);
        }
    }
#endif
}
/**
 * @brief Perform a column-wise reduction on a matrix in column-major layout.
 *
 * This function template performs a parallel reduction across the columns of a matrix using a specified operation.
 * It leverages warp shuffle functions for efficient intra-warp communication and is optimized for column-major matrices.
 *
 * @tparam op The operation to be applied for reduction.
 * @tparam V The vector type for the column accumulator.
 * @tparam T The matrix type with column layout.
 * @tparam reset A boolean flag indicating whether to reset the accumulator (ignore src_accum) or not.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when reset is false.
 */
template<typename op, ducks::rv::all V, ducks::rt::col_layout T, bool reset>
__device__ static inline void col_reduce(V &col_accum, const T &src, const V &src_accum) {
    // I actually like these static asserts because they give more verbose errors when things go wrong.
    KITTENS_CHECK_WARP
    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout>::row_vec_layout>); // compatible layout
    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>); // compatible type
    static_assert(V::outer_dim == T::width); // compatible size

    using dtype = V::dtype;
#ifdef KITTENS_C500
    using scalar = typename base_types::packing<dtype>::unpacked_type;
    const int warp_laneid = ::kittens::laneid();
    const int col = warp_laneid & 0xf;
    constexpr uint64_t full_mask = 0xffffffffffffffffull;
    #pragma unroll
    for(int j = 0; j < T::width; j++) { // note now width is the outer loop
        scalar accum_col = op::template op<scalar>(
            op::template op<scalar>(src.tiles[0][j].data[0].x, src.tiles[0][j].data[0].y),
            op::template op<scalar>(src.tiles[0][j].data[1].x, src.tiles[0][j].data[1].y)
        );
        #pragma unroll
        for(int i = 1; i < T::height; i++) { // and height is the inner loop
            accum_col = op::template op<scalar>(accum_col, src.tiles[i][j].data[0].x);
            accum_col = op::template op<scalar>(accum_col, src.tiles[i][j].data[0].y);
            accum_col = op::template op<scalar>(accum_col, src.tiles[i][j].data[1].x);
            accum_col = op::template op<scalar>(accum_col, src.tiles[i][j].data[1].y);
        }

        scalar accum_col_0 = __shfl_sync(full_mask, accum_col, col +  0, WARP_THREADS);
        scalar accum_col_1 = __shfl_sync(full_mask, accum_col, col + 16, WARP_THREADS);
        scalar accum_col_2 = __shfl_sync(full_mask, accum_col, col + 32, WARP_THREADS);
        scalar accum_col_3 = __shfl_sync(full_mask, accum_col, col + 48, WARP_THREADS);
        accum_col = op::template op<scalar>(
            op::template op<scalar>(accum_col_0, accum_col_1),
            op::template op<scalar>(accum_col_2, accum_col_3)
        );

        dtype accum_packed = base_types::packing<dtype>::pack(accum_col);
        if(reset) {
            col_accum[j][0] = accum_packed;
        }
        else {
            col_accum[j][0] = op::template op<dtype>(src_accum[j][0], accum_packed);
        }
    }
#else
    const int leader = threadIdx.x & 0x1C; // 11100 in binary
    #pragma unroll
    for(int j = 0; j < T::width; j++) { // note now width is the outer loop
        dtype accum_left_col  = op::template op<dtype>(src.tiles[0][j].data[0], src.tiles[0][j].data[2]);
        dtype accum_right_col = op::template op<dtype>(src.tiles[0][j].data[1], src.tiles[0][j].data[3]);
        #pragma unroll
        for(int i = 1; i < T::height; i++) { // and height is the inner loop
            #pragma unroll
            for(int k = 0; k < T::packed_per_tile; k+=2) {
                accum_left_col  = op::template op<dtype>(accum_left_col,  src.tiles[i][j].data[k+0]);
                accum_right_col = op::template op<dtype>(accum_right_col, src.tiles[i][j].data[k+1]);
            }
        }
        dtype accum_packed;
        accum_packed.x = op::template op<typename base_types::packing<dtype>::unpacked_type>(accum_left_col.x,  accum_left_col.y);
        accum_packed.y = op::template op<typename base_types::packing<dtype>::unpacked_type>(accum_right_col.x, accum_right_col.y);

        // Now we need to do a lil shuffle to make everyone happy.

        accum_packed = op::template op<dtype>(accum_packed, packed_shfl_down_sync(MASK_ALL, accum_packed, 2));
        accum_packed = op::template op<dtype>(accum_packed, packed_shfl_down_sync(MASK_ALL, accum_packed, 1));

        accum_packed = packed_shfl_sync(MASK_ALL, accum_packed, leader);

        if(reset) {
            col_accum[j][0] = accum_packed;
        }
        else {
            col_accum[j][0] = op::template op<dtype>(src_accum[j][0], accum_packed);
        }
    }
#endif
}


/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

// two-operand row reductions. (Accumulate and REPLACE.)
/**
 * @brief Store the maximum of each row of the src register tile in the row_accum column vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void row_max(V &row_accum, const T &src)  {
    row_reduce<base_ops::max, V, T, true>(row_accum, src, row_accum);
}
/**
 * @brief Store the minimum of each row of the src register tile in the row_accum column vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void row_min(V &row_accum, const T &src)  {
    row_reduce<base_ops::min, V, T, true>(row_accum, src, row_accum);
}
/**
 * @brief Store the sum of each row of the src register tile in the row_accum column vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void row_sum(V &row_accum, const T &src)  {
    row_reduce<base_ops::sum, V, T, true>(row_accum, src, row_accum);
}
/**
 * @brief Store the product of each row of the src register tile in the row_accum column vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void row_prod(V &row_accum, const T &src) {
    row_reduce<base_ops::mul, V, T, true>(row_accum, src, row_accum);
}
// three-operand row reductions. (Accumulate ONTO.)
/**
 * @brief Store the maximum of each row of the src register tile, as well as the src_accum column vector, in the row_accum column vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void row_max(V &row_accum, const T &src, const V &src_accum)  {
    row_reduce<base_ops::max, V, T, false>(row_accum, src, src_accum);
}
/**
 * @brief Store the minimum of each row of the src register tile, as well as the src_accum column vector, in the row_accum column vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void row_min(V &row_accum, const T &src, const V &src_accum)  {
    row_reduce<base_ops::min, V, T, false>(row_accum, src, src_accum);
}
/**
 * @brief Store the sum of each row of the src register tile, as well as the src_accum column vector, in the row_accum column vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void row_sum(V &row_accum, const T &src, const V &src_accum)  {
    row_reduce<base_ops::sum, V, T, false>(row_accum, src, src_accum);
}
/**
 * @brief Store the product of each row of the src register tile, as well as the src_accum column vector, in the row_accum column vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void row_prod(V &row_accum, const T &src, const V &src_accum) {
    row_reduce<base_ops::mul, V, T, false>(row_accum, src, src_accum);
}

// two-operand col reductions. (Accumulate and REPLACE.)

/**
 * @brief Store the maximum of each column of the src register tile in the col_accum row vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void col_max(V &col_accum, const T &src)  {
    col_reduce<base_ops::max, V, T, true>(col_accum, src, col_accum);
}
/**
 * @brief Store the minimum of each column of the src register tile in the col_accum row vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void col_min(V &col_accum, const T &src)  {
    col_reduce<base_ops::min, V, T, true>(col_accum, src, col_accum);
}
/**
 * @brief Store the sum of each column of the src register tile in the col_accum row vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void col_sum(V &col_accum, const T &src)  {
    col_reduce<base_ops::sum, V, T, true>(col_accum, src, col_accum);
}
/**
 * @brief Store the product of each column of the src register tile in the col_accum row vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void col_prod(V &col_accum, const T &src) {
    col_reduce<base_ops::mul, V, T, true>(col_accum, src, col_accum);
}
// three-operand col reductions. (Accumulate ONTO.)
/**
 * @brief Store the maximum of each column of the src register tile, as well as the src_accum row vector, in the col_accum row vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void col_max(V &col_accum, const T &src, const V &src_accum)  {
    col_reduce<base_ops::max, V, T, false>(col_accum, src, src_accum);
}
/**
 * @brief Store the minimum of each column of the src register tile, as well as the src_accum row vector, in the col_accum row vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void col_min(V &col_accum, const T &src, const V &src_accum)  {
    col_reduce<base_ops::min, V, T, false>(col_accum, src, src_accum);
}
/**
 * @brief Store the sum of each column of the src register tile, as well as the src_accum row vector, in the col_accum row vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void col_sum(V &col_accum, const T &src, const V &src_accum)  {
    col_reduce<base_ops::sum, V, T, false>(col_accum, src, src_accum);
}
/**
 * @brief Store the product of each column of the src register tile, as well as the src_accum row vector, in the col_accum row vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void col_prod(V &col_accum, const T &src, const V &src_accum) {
    col_reduce<base_ops::mul, V, T, false>(col_accum, src, src_accum);
}

// templated versions of each

template<int ax, ducks::rv::all RV, ducks::rt::all T>
__device__ static inline void max(RV &dst, const T &src, const RV &src_accum) {
    if constexpr (ax == axis::COL) row_max(dst, src, src_accum);
    else col_max(dst, src, src_accum);
}
template<int ax, ducks::rv::all RV, ducks::rt::all T>
__device__ static inline auto max(const T &src, const RV &src_accum) {
    RV dst;
    if constexpr (ax == axis::COL) row_max(dst, src, src_accum);
    else col_max(dst, src, src_accum);
    return dst;
}
template<int ax, ducks::rv::all RV, ducks::rt::all T>
__device__ static inline void max(RV &dst, const T &src) {
    if constexpr (ax == axis::COL) row_max(dst, src);
    else col_max(dst, src);
}
template<int ax, ducks::rt::all T>
__device__ static inline auto max(const T &src) {
    using RV = std::conditional_t<ax==axis::COL, typename T::col_vec, typename T::row_vec>;
    RV dst;
    if constexpr (ax == axis::COL) row_max(dst, src);
    else col_max(dst, src);
    return dst;
}

template<int ax, ducks::rv::all RV, ducks::rt::all T>
__device__ static inline void min(RV &dst, const T &src, const RV &src_accum) {
    if constexpr (ax == axis::COL) row_min(dst, src, src_accum);
    else col_min(dst, src, src_accum);
}
template<int ax, ducks::rv::all RV, ducks::rt::all T>
__device__ static inline auto min(const T &src, const RV &src_accum) {
    RV dst;
    if constexpr (ax == axis::COL) row_min(dst, src, src_accum);
    else col_min(dst, src, src_accum);
    return dst;
}
template<int ax, ducks::rv::all RV, ducks::rt::all T>
__device__ static inline void min(RV &dst, const T &src) {
    if constexpr (ax == axis::COL) row_min(dst, src);
    else col_min(dst, src);
}
template<int ax, ducks::rt::all T>
__device__ static inline auto min(const T &src) {
    using RV = std::conditional_t<ax==axis::COL, typename T::col_vec, typename T::row_vec>;
    RV dst;
    if constexpr (ax == axis::COL) row_min(dst, src);
    else col_min(dst, src);
    return dst;
}

template<int ax, ducks::rv::all RV, ducks::rt::all T>
__device__ static inline void sum(RV &dst, const T &src, const RV &src_accum) {
    if constexpr (ax == axis::COL) row_sum(dst, src, src_accum);
    else col_sum(dst, src, src_accum);
}
template<int ax, ducks::rv::all RV, ducks::rt::all T>
__device__ static inline auto sum(const T &src, const RV &src_accum) {
    RV dst;
    if constexpr (ax == axis::COL) row_sum(dst, src, src_accum);
    else col_sum(dst, src, src_accum);
    return dst;
}
template<int ax, ducks::rv::all RV, ducks::rt::all T>
__device__ static inline void sum(RV &dst, const T &src) {
    if constexpr (ax == axis::COL) row_sum(dst, src);
    else col_sum(dst, src);
}
template<int ax, ducks::rt::all T>
__device__ static inline auto sum(const T &src) {
    using RV = std::conditional_t<ax==axis::COL, typename T::col_vec, typename T::row_vec>;
    RV dst;
    if constexpr (ax == axis::COL) row_sum(dst, src);
    else col_sum(dst, src);
    return dst;
}

template<int ax, ducks::rv::all RV, ducks::rt::all T>
__device__ static inline void prod(RV &dst, const T &src, const RV &src_accum) {
    if constexpr (ax == axis::COL) row_prod(dst, src, src_accum);
    else col_prod(dst, src, src_accum);
}
template<int ax, ducks::rv::all RV, ducks::rt::all T>
__device__ static inline auto prod(const T &src, const RV &src_accum) {
    RV dst;
    if constexpr (ax == axis::COL) row_prod(dst, src, src_accum);
    else col_prod(dst, src, src_accum);
    return dst;
}
template<int ax, ducks::rv::all RV, ducks::rt::all T>
__device__ static inline void prod(RV &dst, const T &src) {
    if constexpr (ax == axis::COL) row_prod(dst, src);
    else col_prod(dst, src);
}
template<int ax, ducks::rt::all T>
__device__ static inline auto prod(const T &src) {
    using RV = std::conditional_t<ax==axis::COL, typename T::col_vec, typename T::row_vec>;
    RV dst;
    if constexpr (ax == axis::COL) row_prod(dst, src);
    else col_prod(dst, src);
    return dst;
}
