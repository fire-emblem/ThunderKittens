#!/usr/bin/env python3

import csv
import ctypes
import gc
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch


ROOT = Path("/data/muxi_native_layout_kernels")
TORCH_LIB = Path(torch.__file__).resolve().parent / "lib"


def _preload_torch_libs() -> None:
    for name in [
        "libc10.so",
        "libtorch_cpu.so",
        "libc10_cuda.so",
        "libtorch_cuda.so",
        "libtorch_python.so",
    ]:
        path = TORCH_LIB / name
        if path.exists():
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)


_preload_torch_libs()
sys.path.insert(0, str(ROOT))
import muxi_layout_kernels  # noqa: E402


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None else default


def env_list(name: str) -> set[str] | None:
    value = os.environ.get(name)
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


WARMUP = env_int("MUXI_BASELINE_WARMUP", 1)
BENCH = env_int("MUXI_BASELINE_BENCH", 3)
EXEC = env_int("MUXI_BASELINE_EXEC", 1)
SELECTED = env_list("MUXI_BASELINE_FAMILIES")
OUT_CSV = os.environ.get(
    "MUXI_BASELINE_OUT",
    str(Path("/data/ThunderKittens/kernels/gemm/bf16_c500_tk_local/muxi_baseline_results.csv")),
)


def tflops(m: int, n: int, k: int, runtime_ns: float) -> float:
    return 2.0 * m * n * k / runtime_ns / 1e3


def layout_a(a: torch.Tensor) -> torch.Tensor:
    return a.view(a.shape[0] // 16, 16, a.shape[1] // 8, 8).permute(0, 2, 1, 3).contiguous()


def layout_c_to_contiguous(c: torch.Tensor, m: int, n: int) -> torch.Tensor:
    reshaped = c.view(m // 32, n // 16, 4, 16, 8)
    transposed = reshaped.permute(1, 3, 0, 2, 4).contiguous()
    return transposed.view(n, m)


def max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.abs(a.cpu() - b.cpu()).max().item()


def time_callable(fn: Callable[[], torch.Tensor], warmup: int, bench: int, exec_times: int) -> tuple[torch.Tensor, float]:
    out = None
    for _ in range(warmup):
        for _ in range(exec_times):
            out = fn()
    torch.cuda.synchronize()

    total = 0.0
    for _ in range(bench):
        t0 = time.perf_counter()
        for _ in range(exec_times):
            out = fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        total += (t1 - t0)
    assert out is not None
    return out, total / bench * 1e9


@dataclass(frozen=True)
class Case:
    family: str
    dtype: str
    m: int
    n: int
    k: int


def run_continuous_case(case: Case) -> dict:
    dtype = torch.bfloat16 if case.dtype == "bf16" else torch.float16
    a = torch.rand(case.m, case.k, device="cuda", dtype=dtype) * 2 - 1
    b = torch.rand(case.n, case.k, device="cuda", dtype=dtype) * 2 - 1
    bias = torch.ones(1, case.m, device="cuda", dtype=dtype) * 100
    trans_a = a.t().contiguous()
    a_native = layout_a(a)
    b_native = muxi_layout_kernels.layoutB(b)

    ref_fn = lambda: muxi_layout_kernels.gemmEx(trans_a, b, 1.0, 0.0)
    if case.family == "gemm_layoutAB_ContinuousC":
        kernel_fn = lambda: muxi_layout_kernels.gemm_layoutAB_ContinuousC(a_native, b_native, 1.0, 0.0, bias)
    elif case.family == "gemm_layoutA":
        kernel_fn = lambda: muxi_layout_kernels.gemm_layoutA(a_native, b, 1.0, 0.0, bias)
    elif case.family == "gemm_layoutABC":
        kernel_fn = lambda: muxi_layout_kernels.gemm_layoutABC(a_native, b_native, 1.0, 0.0, bias)
    else:
        raise ValueError(case.family)

    ref, ref_ns = time_callable(ref_fn, WARMUP, BENCH, EXEC)
    ref = ref + bias
    out, out_ns = time_callable(kernel_fn, WARMUP, BENCH, EXEC)

    if case.family == "gemm_layoutABC":
        out = layout_c_to_contiguous(out, case.m, case.n)

    result = {
        "family": case.family,
        "dtype": case.dtype,
        "m": case.m,
        "n": case.n,
        "k": case.k,
        "kernel_ns": out_ns,
        "kernel_tflops": tflops(case.m, case.n, case.k, out_ns),
        "ref_ns": ref_ns,
        "ref_tflops": tflops(case.m, case.n, case.k, ref_ns),
        "speedup": ref_ns / out_ns,
        "max_error": max_diff(out, ref),
    }
    del a, b, bias, trans_a, a_native, b_native, ref, out
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_hgemm_layoutc_case(case: Case) -> dict:
    dtype = torch.bfloat16 if case.dtype == "bf16" else torch.float16
    a = torch.rand(case.m, case.k, device="cuda", dtype=dtype) * 2 - 1
    b = torch.rand(case.n, case.k, device="cuda", dtype=dtype) * 2 - 1
    bias = torch.rand(1, case.m, device="cuda", dtype=dtype)
    trans_a = a.t().contiguous()
    a_native = layout_a(a)
    b_native = muxi_layout_kernels.layoutB(b)

    ref_fn = lambda: muxi_layout_kernels.gemmEx(trans_a, b, 1.0, 0.0)
    kernel_fn = lambda: muxi_layout_kernels.muxi_hgemm_layoutC(a_native, b_native, 1.0, 0.0, bias)

    ref, ref_ns = time_callable(ref_fn, WARMUP, BENCH, EXEC)
    ref = ref + bias
    out, out_ns = time_callable(kernel_fn, WARMUP, BENCH, EXEC)
    out = layout_c_to_contiguous(out, case.m, case.n)

    result = {
        "family": case.family,
        "dtype": case.dtype,
        "m": case.m,
        "n": case.n,
        "k": case.k,
        "kernel_ns": out_ns,
        "kernel_tflops": tflops(case.m, case.n, case.k, out_ns),
        "ref_ns": ref_ns,
        "ref_tflops": tflops(case.m, case.n, case.k, ref_ns),
        "speedup": ref_ns / out_ns,
        "max_error": max_diff(out, ref),
    }
    del a, b, bias, trans_a, a_native, b_native, ref, out
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_hgemm_layout_case(case: Case) -> dict:
    dtype = torch.float16
    a = torch.rand(case.m, case.k, device="cuda", dtype=dtype) * 2 - 1
    b = torch.rand(case.n, case.k, device="cuda", dtype=dtype) * 2 - 1
    a_native = layout_a(a)
    b_native = muxi_layout_kernels.layoutB(b)

    ref_fn = lambda: muxi_layout_kernels.muxi_hgemm(a, b, 1.0, 0.0)
    kernel_fn = lambda: muxi_layout_kernels.muxi_hgemm_layout(a_native, b_native, 1.0, 0.0)

    ref, ref_ns = time_callable(ref_fn, WARMUP, BENCH, EXEC)
    out, out_ns = time_callable(kernel_fn, WARMUP, BENCH, EXEC)

    result = {
        "family": case.family,
        "dtype": case.dtype,
        "m": case.m,
        "n": case.n,
        "k": case.k,
        "kernel_ns": out_ns,
        "kernel_tflops": tflops(case.m, case.n, case.k, out_ns),
        "ref_ns": ref_ns,
        "ref_tflops": tflops(case.m, case.n, case.k, ref_ns),
        "speedup": ref_ns / out_ns,
        "max_error": max_diff(out, ref),
    }
    del a, b, a_native, b_native, ref, out
    gc.collect()
    torch.cuda.empty_cache()
    return result


def should_run(family: str) -> bool:
    return SELECTED is None or family in SELECTED


def build_cases() -> list[Case]:
    cases: list[Case] = []

    if should_run("gemm_layoutAB_ContinuousC"):
        for dtype in ["fp16", "bf16"]:
            for m, k in [(4608, 3584), (3584, 3584), (37888, 3584), (3584, 18944)]:
                for n in range(16, 257, 16):
                    cases.append(Case("gemm_layoutAB_ContinuousC", dtype, m, n, k))

    if should_run("gemm_layoutA"):
        for m, k in [(512, 7168), (7168, 256)]:
            for n in range(16, 129, 16):
                cases.append(Case("gemm_layoutA", "fp16", m, n, k))

    if should_run("gemm_layoutABC"):
        for m, k in [(512, 7168), (7168, 256)]:
            for n in range(16, 129, 16):
                cases.append(Case("gemm_layoutABC", "fp16", m, n, k))

    if should_run("muxi_hgemm_layoutC"):
        for dtype in ["fp16"]:
            for m, k in [(4608, 3584), (3584, 3584), (37888, 3584), (3584, 18944)]:
                for n in range(256, 512, 32):
                    cases.append(Case("muxi_hgemm_layoutC", dtype, m, n, k))

    if should_run("muxi_hgemm_layout"):
        for n in [16 * 1024, 32 * 1024, 48 * 1024, 64 * 1024, 80 * 1024]:
            cases.append(Case("muxi_hgemm_layout", "fp16", 37888, n, 3584))

    return cases


def run_case(case: Case) -> dict:
    if case.family in {"gemm_layoutAB_ContinuousC", "gemm_layoutA", "gemm_layoutABC"}:
        return run_continuous_case(case)
    if case.family == "muxi_hgemm_layoutC":
        return run_hgemm_layoutc_case(case)
    if case.family == "muxi_hgemm_layout":
        return run_hgemm_layout_case(case)
    raise ValueError(case.family)


def main() -> int:
    cases = build_cases()
    out_path = Path(OUT_CSV)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if out_path.exists():
        with out_path.open() as f:
            for row in csv.DictReader(f):
                existing[(row["family"], row["dtype"], int(row["m"]), int(row["n"]), int(row["k"]))] = row

    print(f"Running {len(cases)} muxi baseline cases with warmup={WARMUP}, bench={BENCH}, exec={EXEC}")
    write_header = not out_path.exists()
    with out_path.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "family",
                "dtype",
                "m",
                "n",
                "k",
                "kernel_ns",
                "kernel_tflops",
                "ref_ns",
                "ref_tflops",
                "speedup",
                "max_error",
            ],
        )
        if write_header:
            writer.writeheader()
        for idx, case in enumerate(cases, 1):
            key = (case.family, case.dtype, case.m, case.n, case.k)
            if key in existing:
                print(f"[{idx}/{len(cases)}] skip {case.family} {case.dtype} M={case.m} N={case.n} K={case.k}")
                continue
            result = run_case(case)
            writer.writerow(result)
            f.flush()
            print(
                f"[{idx}/{len(cases)}] {case.family} {case.dtype} "
                f"M={case.m} N={case.n} K={case.k} "
                f"kernel={result['kernel_tflops']:.3f} TFLOP/s "
                f"ref={result['ref_tflops']:.3f} TFLOP/s "
                f"speedup={result['speedup']:.3f} "
                f"err={result['max_error']:.6f}"
            )
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
