# ThunderKittens 在 RTX 4080 上的用户层 Feature 缺口矩阵

## 结论

对 4080 而言，当前项目的主要缺口不在底层 primitive，而在用户可见接口层：

1. 缺少统一的 4080 kernel 构建入口。
2. 缺少大部分单卡 operator 的 4080 官方模块入口。
3. 缺少 attention / GEMM 的 4080 模型级接口。
4. 缺少围绕这些接口的 benchmark / correctness / 文档工作流。

`tests/` 已经证明大量公共底座在 4080 上可运行；当前已经开始把这些能力收敛成正式面向用户的 4080 feature surface，其中 `layernorm` 和 `rotary` 已打通。

---

## A. 单卡用户可见 Operator

| Operator | 用户可见接口 | 当前官方入口 | 4080 状态 |
|---|---|---|---|
| Based | `based` | `kernels/based/Makefile` | 入口已补齐，但源码仍是 Hopper-only |
| Hedgehog | `hedgehog` | `kernels/hedgehog/Makefile` | 入口已补齐，但源码仍是 Hopper-only |
| LayerNorm | `fused_layernorm` | `kernels/layernorm/Makefile` | 已打通 4080 standalone 运行 |
| Rotary | `fused_rotary` | `kernels/rotary/Makefile` | 已打通 4080 standalone + PyTorch 模块 |
| FFTConv | `fftconv` | `kernels/fftconv/Makefile` | 入口已补齐，源码状态待继续收敛 |
| Flux Gate | `tk_flux_linear_gate` | `kernels/flux/Makefile` | 入口已补齐，源码状态待继续收敛 |
| Flux GELU | `tk_flux_linear_gelu` | `kernels/flux/Makefile` | 入口已补齐，源码状态待继续收敛 |
| Mamba2 | `mamba2` | `kernels/mamba2/Makefile` | 入口已补齐，但源码仍是 Hopper-only |

### 证据

- `based`: `m.def("based", based, ...)`，见 `kernels/based/linear_attn.cu`。
- `hedgehog`: `m.def("hedgehog", hedgehog, ...)`，见 `kernels/hedgehog/hedgehog.cu`。
- `fused_layernorm`: `m.def("fused_layernorm", fused_layernorm, ...)`，见 `kernels/layernorm/layernorm.cu`。
- `fused_rotary`: `m.def("fused_rotary", fused_rotary, ...)`，见 `kernels/rotary/rotary.cu`。
- `fftconv`: `m.def("fftconv", fftconv, ...)`，见 `kernels/fftconv/fftconv_pc.cu`。
- `tk_flux_linear_gate`: `m.def("tk_flux_linear_gate", ...)`，见 `kernels/flux/flux_gate.cu`。
- `tk_flux_linear_gelu`: `m.def("tk_flux_linear_gelu", ...)`，见 `kernels/flux/flux_gelu.cu`。
- `mamba2`: `m.def("mamba2", mamba2, ...)`，见 `kernels/mamba2/mamba2.cu`。

这些 operator API 已存在；当前 4080 侧已经完成第一批入口面补齐，并开始把其中一部分真正跑通：

- `kernels/based/Makefile`
- `kernels/hedgehog/Makefile`
- `kernels/layernorm/Makefile`
- `kernels/rotary/Makefile`
- `kernels/fftconv/Makefile`
- `kernels/flux/Makefile`
- `kernels/mamba2/Makefile`

其中最新已验证结果：

- `layernorm`: `make GPU=RTX4080 CONFIG=standalone -C kernels/layernorm run` 可通过
- `rotary`: `make GPU=RTX4080 CONFIG=standalone -C kernels/rotary run` 可通过
- `rotary`: `make GPU=RTX4080 -C kernels/rotary -j4` 可构建 `_C*.so`，并能在 Python 中调用 `fused_rotary(...)`

---

## B. Attention 的模型级接口

| 接口族 | 当前接口 | 当前官方 GPU | 4080 缺口 |
|---|---|---|---|
| 双向 MHA | `mha_forward`, `mha_backward` | H100 | 无 4080 attention 模块 |
| causal MHA | `forward`, `forward_persistent` | B300 | 无 4080 causal attention 模块 |
| non-causal MHA | `forward` | B300 | 无 4080 non-causal attention 模块 |

### 证据

- H100 MHA 暴露：
  - `m.def("mha_forward", ...)`
  - `m.def("mha_backward", ...)`
  - 文件：`kernels/attention/mha_h100/mha_h100.cu`
- B300 causal MHA 暴露：
  - `m.def("forward", ...)`
  - `m.def("forward_persistent", ...)`
  - 文件：`kernels/attention/bf16_b300_mha_causal/bf16_b300_mha_causal.cu`
- B300 non-causal MHA 暴露：
  - `m.def("forward", ...)`
  - 文件：`kernels/attention/bf16_b300_mha_noncausal/bf16_b300_mha_noncausal.cu`

配套构建入口分别固定在：

- `kernels/attention/mha_h100/Makefile`
- `kernels/attention/bf16_b300_mha_causal/Makefile`
- `kernels/attention/bf16_b300_mha_noncausal/Makefile`

### 从用户视角真正缺的东西

不是“4080 没有 WGMMA/TMA”，而是：

1. 没有 `mha_4080` / `mha_ampere` / `mha_ada` 模块。
2. 没有统一的 `attention.forward(...)` 用户接口按 GPU 自动分派。
3. 没有 4080 attention correctness / benchmark / test workflow。

---

## C. GEMM 的模型级接口

### C1. 单卡 GEMM

| 接口族 | 当前接口 | 当前官方 GPU | 4080 缺口 |
|---|---|---|---|
| BF16 GEMM | 独立 binary，无统一 pybind 接口 | H100 / B200 | 无 4080 BF16 GEMM 模块 |
| FP8 GEMM | 独立 binary，无统一 pybind 接口 | H100 / B200 | 无 4080 FP8 GEMM 模块 |
| MXFP8 GEMM | `mxfp8_gemm`, `mxfp8_quantize` | B200 | 无 4080 接口 |
| NVFP4 GEMM | `nvfp4_gemm`, `nvfp4_quantize`, conversion API | B200 | 无 4080 接口 |

### 证据

- `mxfp8_gemm`, `mxfp8_quantize` 在 `kernels/gemm/mxfp8_b200/mxfp8_b200_gemm.cu`
- `nvfp4_gemm`, `nvfp4_quantize` 等在 `kernels/gemm/nvfp4_b200/nvfp4_b200_gemm.cu`
- BF16/FP8 的单卡入口仍按 GPU 目录切分：
  - `kernels/gemm/bf16_h100/Makefile`
  - `kernels/gemm/bf16_b200/Makefile`
  - `kernels/gemm/fp8_h100/Makefile`
  - `kernels/gemm/fp8_b200/Makefile`

### C2. 多卡/模型级 GEMM 与相关接口

| 接口 | 当前官方 GPU | 4080 缺口 |
|---|---|---|
| `all_gather_matmul` | H100 / B200 | 无 4080 版本 |
| `matmul_reduce_scatter` | H100 / B200 | 无 4080 版本 |
| `matmul_all_reduce` | H100 only | 无 4080 版本 |
| `moe_dispatch_gemm` | H100 only | 无 4080 版本 |

### 证据

- `all_gather_matmul`：
  - `kernels/parallel/ag_gemm/ag_gemm_h100.cu`
  - `kernels/parallel/ag_gemm/ag_gemm_b200.cu`
- `matmul_reduce_scatter`：
  - `kernels/parallel/gemm_rs/gemm_rs_h100.cu`
  - `kernels/parallel/gemm_rs/gemm_rs_b200.cu`
- `matmul_all_reduce`：
  - `kernels/parallel/gemm_ar/gemm_ar_h100.cu`
- `moe_dispatch_gemm`：
  - `kernels/parallel/moe_dispatch_gemm/moe_dispatch_gemm_h100.cu`

配套 Makefile 和 benchmark 也都把 GPU 限死在 H100/B200：

- `kernels/parallel/ag_gemm/Makefile`
- `kernels/parallel/gemm_rs/Makefile`
- `kernels/parallel/gemm_ar/Makefile`
- `kernels/parallel/moe_dispatch_gemm/Makefile`
- `kernels/parallel/ag_gemm/benchmark.py`
- `kernels/parallel/gemm_rs/benchmark.py`

---

## D. 更高一层的统一接口缺失

对 4080 来说，最本质的用户层缺口其实是这几个：

1. 仍缺少 attention / GEMM 这种更高层模块的统一 `GPU=RTX4080` 用户入口。
2. 仍没有完整的单卡 operator 4080 模块集合。
3. 没有 attention / GEMM 的统一 GPU dispatcher。
4. 没有 README 级别把 4080 作为正式 feature target 写进去。
5. 没有与 4080 模块配套的 benchmark / correctness / demo / 文档。

---

## E. 建议的补齐顺序

### 第一优先级：先补“用户能调用”的最小表面

1. 在 `kernels/common.mk` 增加 `RTX4080`。
2. 让单卡 operator 的 Makefile 支持 `GPU=RTX4080`。
3. 明确哪些 operator 在 4080 上是“功能支持、性能非最优”。

### 第二优先级：补 attention / GEMM 的统一命名

1. 提供 `attention` 的 4080 版本模块或 dispatcher。
2. 提供 `gemm` 的 4080 版本模块或 dispatcher。
3. 把 H100/B200/B300 的 GPU-SKU 命名，提升成用户层的统一 API。

### 第三优先级：补完整工作流

1. 4080 benchmark 脚本。
2. 4080 correctness 脚本。
3. README 支持矩阵。
4. 最小 demo。
