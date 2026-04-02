#include "kittens.cuh"

#if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)
#include "prototype.cuh"
#endif

#ifdef TORCH_COMPILE
#define TK_COMPILE_FUSED_ROTARY
#endif

using namespace kittens;

#if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)

using namespace kittens::prototype;
using namespace kittens::prototype::lcsf;

template<int _headdim, int _warps> struct rotary_layout {
    static constexpr int headdim = _headdim, warps = _warps;
    using seq_tile    = st_bf<16, headdim>;
    using seq_global  = gl<bf16, -1, -1, -1, headdim, seq_tile>;
    using rope_global = gl<bf16,  1,  1, -1, headdim/2>;
    struct globals {
        seq_global o, x;
        rope_global sin, cos;
        int batches;
    };
    struct input_block    { seq_tile x[warps]; };
    struct output_block   { seq_tile o[warps]; };
    struct producer_state { int active_warps;  };
    struct consumer_state { rt_fl<16, headdim/2> sin, cos; };
};

template<int _headdim> struct rotary_template {
    static constexpr int headdim=_headdim, NUM_CONSUMER_WARPS=8, NUM_BLOCKS=1, OUTPUT_PIPE_STAGES=3, INPUT_PIPE_STAGES=3;
    using layout = rotary_layout<headdim, NUM_CONSUMER_WARPS>;

    __device__ static inline void common_setup(common_setup_args<layout> args) {
        if(args.task_iter == 0) {
            args.num_iters = min(args.globals.batches, (int)(args.globals.x.batch()-blockIdx.y*args.globals.batches)) * args.globals.x.depth();
        }
        else {
            args.num_iters = -1;
        }
    }

    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::producer_registers();
            args.state.active_warps = min((int)NUM_CONSUMER_WARPS,
                                          (int)(args.globals.x.rows()/16 - blockIdx.x*NUM_CONSUMER_WARPS));
        }
        __device__ static void load(producer_load_args<layout> args) {
            if(warpgroup::warpid() == args.iter%4) {
                kittens::coord idx = { blockIdx.y*args.globals.batches+args.iter/args.globals.x.depth(),
                                       args.iter%args.globals.x.depth(),
                                       blockIdx.x*NUM_CONSUMER_WARPS,
                                       0 };
                warp::tma::expect_bytes(args.inputs_arrived, sizeof(layout::seq_tile)*args.state.active_warps);
                for(int i = 0; i < args.state.active_warps; i++) {
                    warp::tma::load_async(args.input.x[i], args.globals.x, {idx.b,idx.d,idx.r+i,idx.c}, args.inputs_arrived);
                }
                if(laneid() == 0) arrive(args.inputs_arrived, 3);
                __syncwarp();
            }
        }
        __device__ static void store(producer_store_args<layout> args) {
            if(warpgroup::warpid() == args.iter%4) {
                kittens::coord idx = { blockIdx.y*args.globals.batches+args.iter/args.globals.x.depth(),
                                       args.iter%args.globals.x.depth(),
                                       blockIdx.x*NUM_CONSUMER_WARPS,
                                       0 };
                for(int i = 0; i < args.state.active_warps; i++) {
                    warp::tma::store_async(args.globals.o, args.output.o[i], {idx.b,idx.d,idx.r+i,idx.c});
                }
                warp::tma::store_async_read_wait();
                if(laneid() == 0) arrive(args.outputs_finished, 4);
                __syncwarp();
            }
        }
    };

    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::consumer_registers<NUM_CONSUMER_WARPS/4>();
            kittens::coord idx = { blockIdx.x*NUM_CONSUMER_WARPS + warpid(), 0 };
            warp::load(args.state.sin, args.globals.sin, idx);
            warp::load(args.state.cos, args.globals.cos, idx);
        }
        __device__ static void compute(consumer_compute_args<layout> args) {
            rt_fl<16, headdim> x;
            rt_fl<16, headdim/2> x1, x2, temp1, temp2;
            warp::load(x, args.input.x[warpid()]);
            if(laneid() == 0) arrive(args.inputs_finished);
            __syncwarp();
            for(int i = 0; i < headdim/32; i++) {
                #pragma unroll
                for(int j = 0; j < 4; j++) {
                    x1.tiles[0][i].data[j] = x.tiles[0][i].data[j];
                    x2.tiles[0][i].data[j] = x.tiles[0][i+headdim/32].data[j];
                }
            }
            warp::mul(temp1, x1, args.state.cos);
            warp::mul(temp2, x2, args.state.cos);
            warp::mul(x2, x2, -1.f);
            warp::mul(x1, x1, args.state.sin);
            warp::mul(x2, x2, args.state.sin);
            warp::add(temp1, temp1, x2);
            warp::add(temp2, temp2, x1);
            for(int i = 0; i < headdim/32; i++) {
                #pragma unroll
                for(int j = 0; j < 4; j++) {
                    x.tiles[0][i].data[j]            = temp1.tiles[0][i].data[j];
                    x.tiles[0][i+headdim/32].data[j] = temp2.tiles[0][i].data[j];
                }
            }
            warp::store(args.output.o[warpid()], x);
            __syncwarp();
            if(laneid() == 0) arrive(args.outputs_arrived);
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            if(laneid() == 0) arrive(args.finish_finished);
        }
    };
};

template<int ATTN_D>
void dispatch_fused_rotary_impl(
    bf16 *d_o,
    bf16 *d_x,
    bf16 *d_sin_in,
    bf16 *d_cos_in,
    const int ATTN_B,
    const int ATTN_H,
    const int ATTN_N
) {
    using rope_t = rotary_template<ATTN_D>;
    constexpr int BATCHES_PER_BLOCK = 4;

    using seq_globals   = typename rope_t::layout::seq_global;
    using rope_globals  = typename rope_t::layout::rope_global;
    using globals = typename rope_t::layout::globals;

    seq_globals Og{d_o, ATTN_B, ATTN_H, ATTN_N, nullptr};
    seq_globals Xg{d_x, ATTN_B, ATTN_H, ATTN_N, nullptr};
    rope_globals SINg{d_sin_in, nullptr, nullptr, ATTN_N, nullptr};
    rope_globals COSg{d_cos_in, nullptr, nullptr, ATTN_N, nullptr};
    globals g{Og, Xg, SINg, COSg, BATCHES_PER_BLOCK};

    unsigned long mem_size = (MAX_SHARED_MEMORY-2048);
    constexpr int ROWS_PER_BLOCK = rope_t::NUM_CONSUMER_WARPS * rope_t::layout::seq_tile::rows;
    cudaFuncSetAttribute(prototype::lcsf::kernel<rope_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    dim3 grid((ATTN_N+ROWS_PER_BLOCK-1)/ROWS_PER_BLOCK, (ATTN_B+BATCHES_PER_BLOCK-1)/BATCHES_PER_BLOCK);
    dim3 block(kittens::prototype::detail::NUM_THREADS_v<rope_t>);
    kittens::prototype::lcsf::kernel<rope_t><<<grid, block, mem_size>>>(g);
}

#else

__global__ void rotary_fallback_kernel(
    bf16 *o,
    const bf16 *x,
    const bf16 *cos,
    const bf16 *sin,
    int total_rows,
    int seq_len,
    int head_dim,
    int rotary_dim
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elems = total_rows * head_dim;
    if (idx >= total_elems) {
        return;
    }

    const int row = idx / head_dim;
    const int dim = idx % head_dim;
    const int seq = row % seq_len;
    const int base = row * head_dim;
    const int half = rotary_dim / 2;

    if (dim < half) {
        const float x1 = __bfloat162float(x[base + dim]);
        const float x2 = __bfloat162float(x[base + dim + half]);
        const float c = __bfloat162float(cos[seq * half + dim]);
        const float s = __bfloat162float(sin[seq * half + dim]);
        o[base + dim] = __float2bfloat16(x1 * c - x2 * s);
    }
    else if (dim < rotary_dim) {
        const int pair = dim - half;
        const float x1 = __bfloat162float(x[base + pair]);
        const float x2 = __bfloat162float(x[base + dim]);
        const float c = __bfloat162float(cos[seq * half + pair]);
        const float s = __bfloat162float(sin[seq * half + pair]);
        o[base + dim] = __float2bfloat16(x2 * c + x1 * s);
    }
    else {
        o[base + dim] = x[base + dim];
    }
}

#endif

void dispatch_fused_rotary(
    bf16 *d_o,
    bf16 *d_x,
    bf16 *d_sin_in,
    bf16 *d_cos_in,
    const int ATTN_B,
    const int ATTN_H,
    const int ATTN_N,
    const int ATTN_D,
    const int ROTARY_DIM
) {
#if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)
    if (ROTARY_DIM != ATTN_D) {
        throw std::runtime_error("Hopper rotary path expects rotary_dim == head_dim.");
    }
    if (ATTN_D == 64) {
        dispatch_fused_rotary_impl<64>(d_o, d_x, d_sin_in, d_cos_in, ATTN_B, ATTN_H, ATTN_N);
    }
    else if (ATTN_D == 128) {
        dispatch_fused_rotary_impl<128>(d_o, d_x, d_sin_in, d_cos_in, ATTN_B, ATTN_H, ATTN_N);
    }
    else {
        throw std::runtime_error("Unsupported head dimension for fused_rotary.");
    }
#else
    const int total_rows = ATTN_B * ATTN_H * ATTN_N;
    const int total_elems = total_rows * ATTN_D;
    constexpr int THREADS = 256;
    const int blocks = kittens::cdiv(total_elems, THREADS);
    rotary_fallback_kernel<<<blocks, THREADS>>>(
        d_o,
        d_x,
        d_cos_in,
        d_sin_in,
        total_rows,
        ATTN_N,
        ATTN_D,
        ROTARY_DIM
    );
#endif
}

#ifdef TK_COMPILE_FUSED_ROTARY
#include <ATen/Functions.h>
#include <iostream>
#include <torch/extension.h>

#define CHECK_CUDA_TENSOR(x) TORCH_CHECK((x).device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS_TENSOR(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT_TENSOR(x) \
    CHECK_CUDA_TENSOR(x);     \
    CHECK_CONTIGUOUS_TENSOR(x)

at::Tensor fused_rotary(
    const at::Tensor x,
    const at::Tensor cos_in,
    const at::Tensor sin_in
) {
    CHECK_INPUT_TENSOR(x);
    CHECK_INPUT_TENSOR(sin_in);
    CHECK_INPUT_TENSOR(cos_in);

    TORCH_CHECK(x.dim() == 4, "x must have shape [B, H, N, D]");
    TORCH_CHECK(cos_in.dim() == 2, "cos_in must have shape [N, D/2]");
    TORCH_CHECK(sin_in.dim() == 2, "sin_in must have shape [N, D/2]");

    const int B = x.size(0);
    const int H = x.size(1);
    const int N = x.size(2);
    const int D = x.size(3);
    const int rotary_dim = cos_in.size(1) * 2;

    TORCH_CHECK(sin_in.size(0) == N, "sin_in sequence length mismatch");
    TORCH_CHECK(cos_in.size(0) == N, "cos_in sequence length mismatch");
    TORCH_CHECK(sin_in.size(1) == cos_in.size(1), "sin_in/cos_in rotary dimension mismatch");
    TORCH_CHECK(D == 64 || D == 128, "Hidden size mismatch");
    TORCH_CHECK(rotary_dim > 0 && rotary_dim % 2 == 0, "rotary dimension must be positive and even");
    TORCH_CHECK(rotary_dim <= D, "rotary dimension must not exceed hidden size");
    TORCH_CHECK(x.size(2) % 16 == 0, "Sequence length must be multiple of 16");
    TORCH_CHECK(cos_in.size(0) % 16 == 0, "Sequence length must be multiple of 16");
    TORCH_CHECK(sin_in.size(0) % 16 == 0, "Sequence length must be multiple of 16");

    at::Tensor out = at::empty({B, H, N, D}, x.options());

    c10::BFloat16 *x_bf16 = x.data_ptr<c10::BFloat16>();
    c10::BFloat16 *sin_in_bf16 = sin_in.data_ptr<c10::BFloat16>();
    c10::BFloat16 *cos_in_bf16 = cos_in.data_ptr<c10::BFloat16>();
    c10::BFloat16 *out_bf16 = out.data_ptr<c10::BFloat16>();

    bf16 *d_x = reinterpret_cast<bf16*>(x_bf16);
    bf16 *d_sin_in = reinterpret_cast<bf16*>(sin_in_bf16);
    bf16 *d_cos_in = reinterpret_cast<bf16*>(cos_in_bf16);
    bf16 *d_out = reinterpret_cast<bf16*>(out_bf16);

    dispatch_fused_rotary(
        d_out,
        d_x,
        d_sin_in,
        d_cos_in,
        B,
        H,
        N,
        D,
        rotary_dim
    );

    CHECK_CUDA_ERROR(cudaGetLastError());
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_rotary", fused_rotary, "Rotary TK. Takes tensors (x, cos_in, sin_in). All tensors are bf16. Returns (B, H, N, D) in bf16.");
}
#else
#if defined(KITTENS_AMPERE)
#include "harness_ampere.impl"
#else
#include "harness.impl"
#endif
#endif
