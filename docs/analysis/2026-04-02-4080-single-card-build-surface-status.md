# RTX 4080 单卡 Operator 入口打通状态

## 本轮改动

本轮不是在做 4080 kernel 完整移植，而是在补第一优先级的用户入口：

1. `kernels/common.mk` 新增 `GPU=RTX4080`
2. `kernels/common.mk` 的 `CONFIG=pytorch` 现在会自动：
   - 在本机未安装 `pybind11` Python 模块时回退到 `torch/include/pybind11`
   - 跟随本机 `torch._C._GLIBCXX_USE_CXX11_ABI` 追加 ABI 宏
2. 第一批单卡 operator Makefile 从写死 H100 改成：
   - `GPU ?= H100`
   - `CONFIG ?= pytorch`
   - 支持 `CONFIG=standalone`

覆盖目录：

- `kernels/based/`
- `kernels/fftconv/`
- `kernels/flux/`
- `kernels/hedgehog/`
- `kernels/layernorm/`
- `kernels/mamba2/`
- `kernels/rotary/`
- `kernels/linear_attention/`

## 入口级验证结论

### 1. 已经打通到真实 4080 编译路径

这些入口现在都能进入带 `sm_89` 的真实 NVCC 编译，而不再在 Makefile/GPU 选择层直接报错：

- `kernels/linear_attention`
- `kernels/based`
- `kernels/rotary`
- `kernels/layernorm`
- `kernels/hedgehog`
- `kernels/mamba2`

验证命令示例：

```bash
make GPU=RTX4080 -C kernels/linear_attention -j4
make GPU=RTX4080 CONFIG=standalone -C kernels/rotary -j4
make GPU=RTX4080 CONFIG=standalone -C kernels/layernorm -j4
make GPU=RTX4080 CONFIG=standalone -C kernels/based -j4
make GPU=RTX4080 CONFIG=standalone -C kernels/hedgehog -j4
make GPU=RTX4080 CONFIG=standalone -C kernels/mamba2 -j4
```

### 2. 已验证可成功编译

- `layernorm`
- `rotary`

命令：

```bash
make GPU=RTX4080 CONFIG=standalone -C kernels/layernorm -j4
make GPU=RTX4080 CONFIG=standalone -C kernels/rotary run
make GPU=RTX4080 -C kernels/rotary -j4
```

结果：

- `layernorm.out` 成功生成并运行
- `rotary.out` 在 4080 上运行通过，输出 `Correct out :)`，`Max diff: 0`
- `rotary` 的 PyTorch 扩展 `_C*.so` 成功在 `sm_89` 下构建
- 直接 `import _C; fused_rotary(...)` 可调用，和 torch 参考实现满足 `atol=1/32 + 1e-6`

### 3. 已验证仍是源码级 H100-only / Hopper-only

以下入口虽然 Makefile 已支持 4080，但源码编译时仍直接依赖 Hopper 路径：

- `linear_attention`
- `based`
- `hedgehog`
- `mamba2`

共性报错模式：

- `tma_swizzle_allocator is undefined`
- `warp::tma::*` / `tma::*` 不存在
- `warpgroup::mm_AB`, `warpgroup::mm_ABt`, `mma_async_wait` 等不存在
- `increase_registers` / `decrease_registers` / `producer_registers` / `consumer_registers` 不存在

这说明这些模块的缺口已经从“用户入口缺失”收敛成“源码实现仍绑定 Hopper 风格执行模型”。

## 当前阶段判断

第一优先级已经完成一半：

- 已完成：4080 用户入口表面补齐
- 部分完成：`layernorm`、`rotary` 已经变成 4080 可编译、可运行的用户可见入口
- 未完成：把其余入口背后的源码实现真正改成 4080 可编译、可运行

## 下一步建议

下一轮应按收益排序处理源码级缺口：

1. 先挑最接近可移植的单卡 operator
   - `based`
   - `layernorm`、`rotary` 已打通，可作为 4080 可见入口样板
2. 暂时后置明显依赖 TMA / warpgroup producer-consumer 模板的模块
   - `linear_attention`
   - `hedgehog`
   - `mamba2`
