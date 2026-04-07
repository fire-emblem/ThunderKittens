#!/usr/bin/env python3

import csv
import os
import re
import subprocess
from pathlib import Path

from muxi_baseline import (
    BENCH,
    WARMUP,
    build_cases,
    run_case as run_muxi_case,
)


TK_LOCAL_DIR = Path("/data/ThunderKittens/kernels/gemm/bf16_c500_tk_local")
OUT_COMPARE = os.environ.get(
    "MUXI_LOCAL_COMPARE_OUT",
    str(TK_LOCAL_DIR / "muxi_local_compare_results.csv"),
)
MUXI_CACHE = Path(
    os.environ.get(
        "MUXI_BASELINE_CACHE",
        str(TK_LOCAL_DIR / "muxi_baseline_results.csv"),
    )
)
MCBLAS_CACHE = Path(
    os.environ.get(
        "MCBLAS_BASELINE_CACHE",
        str(TK_LOCAL_DIR / "mcblas_baseline_results.csv"),
    )
)


def make_binary(out_name: str, extra_flags: str) -> Path:
    cmd = [
        "make",
        "-C",
        str(TK_LOCAL_DIR),
        "GPU=C500",
        "-j4",
        "SRC=tk_local_runtime_gemm.cu",
        f"OUT={out_name}",
        f"CMD=./{out_name}",
        f"EXTRA_NVCCFLAGS={extra_flags}",
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return TK_LOCAL_DIR / out_name


def ensure_local_binaries() -> dict[tuple[str, str], Path]:
    return {
        ("gemm_layoutAB_ContinuousC", "bf16"): make_binary(
            "tk_local_runtime_continuousc_bf16.out", "-DTK_LOCAL_USE_CONTINUOUSC"
        ),
        ("gemm_layoutAB_ContinuousC", "fp16"): make_binary(
            "tk_local_runtime_continuousc_fp16.out",
            "-DTK_LOCAL_USE_CONTINUOUSC -DTK_LOCAL_USE_FP16",
        ),
        ("muxi_hgemm_layoutC", "fp16"): make_binary(
            "tk_local_runtime_layoutc_fp16.out", "-DTK_LOCAL_USE_FP16"
        ),
    }


PERF_RE = re.compile(r"Performance:\s+([0-9.]+)\s+TFLOP/s")
ERR_RE = re.compile(r"err max:\s+([0-9.]+)")
RUNTIME_RE = re.compile(r"Average runtime:\s+([0-9.]+)\s+ms")
FAMILY_RE = re.compile(r"Family:\s+(.+)")


def run_local_case(binary: Path, m: int, n: int, k: int) -> dict:
    env = os.environ.copy()
    env["TK_LOCAL_M"] = str(m)
    env["TK_LOCAL_N"] = str(n)
    env["TK_LOCAL_K"] = str(k)
    env["TK_LOCAL_WARMUP"] = str(WARMUP)
    env["TK_LOCAL_PROFILE"] = str(BENCH)
    proc = subprocess.run(
        [str(binary)],
        check=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    text = proc.stdout
    perf = float(PERF_RE.search(text).group(1))
    err = float(ERR_RE.search(text).group(1))
    runtime_ms = float(RUNTIME_RE.search(text).group(1))
    family = FAMILY_RE.search(text).group(1).strip()
    return {
        "local_family": family,
        "local_runtime_ns": runtime_ms * 1e6,
        "local_tflops": perf,
        "local_err_max": err,
    }


def local_shape_supported(case) -> bool:
    return (case.m % 128) == 0 and (case.n % 128) == 0 and (case.k % 128) == 0


def main() -> int:
    binaries = ensure_local_binaries()
    muxi_cache = {}
    if MUXI_CACHE.exists():
        with MUXI_CACHE.open() as f:
            for row in csv.DictReader(f):
                muxi_cache[(row["family"], row["dtype"], int(row["m"]), int(row["n"]), int(row["k"]))] = row
    mcblas_cache = {}
    if MCBLAS_CACHE.exists():
        with MCBLAS_CACHE.open() as f:
            for row in csv.DictReader(f):
                mcblas_cache[(row["dtype"], int(row["m"]), int(row["n"]), int(row["k"]))] = row

    results = []
    cases = build_cases()
    print(f"Comparing {len(cases)} muxi cases with local support where available")
    for idx, case in enumerate(cases, 1):
        muxi_key = (case.family, case.dtype, case.m, case.n, case.k)
        if muxi_key in muxi_cache:
            cached = muxi_cache[muxi_key]
            muxi = {
                "kernel_ns": float(cached["kernel_ns"]),
                "kernel_tflops": float(cached["kernel_tflops"]),
                "ref_ns": float(cached["ref_ns"]),
                "ref_tflops": float(cached["ref_tflops"]),
                "speedup": float(cached["speedup"]),
                "max_error": float(cached["max_error"]),
            }
        else:
            muxi = run_muxi_case(case)
        row = {
            "family": case.family,
            "dtype": case.dtype,
            "m": case.m,
            "n": case.n,
            "k": case.k,
            "muxi_runtime_ns": muxi["kernel_ns"],
            "muxi_tflops": muxi["kernel_tflops"],
            "muxi_ref_runtime_ns": muxi["ref_ns"],
            "muxi_ref_tflops": muxi["ref_tflops"],
            "muxi_speedup": muxi["speedup"],
            "muxi_err_max": muxi["max_error"],
            "local_family": "unavailable",
            "local_runtime_ns": "",
            "local_tflops": "",
            "local_err_max": "",
            "local_vs_muxi_ratio": "",
            "mcblas_runtime_ns": "",
            "mcblas_tflops": "",
            "mcblas_err_max": "",
            "local_vs_mcblas_ratio": "",
        }
        mcblas_key = (case.dtype, case.m, case.n, case.k)
        if mcblas_key in mcblas_cache:
            cached = mcblas_cache[mcblas_key]
            row["mcblas_runtime_ns"] = cached["mcblas_runtime_ns"]
            row["mcblas_tflops"] = cached["mcblas_tflops"]
            row["mcblas_err_max"] = cached["mcblas_err_max"]
        binary = binaries.get((case.family, case.dtype))
        if binary is not None and local_shape_supported(case):
            try:
                local = run_local_case(binary, case.m, case.n, case.k)
                row.update(local)
                row["local_vs_muxi_ratio"] = local["local_tflops"] / muxi["kernel_tflops"]
                if row["mcblas_tflops"] != "":
                    row["local_vs_mcblas_ratio"] = local["local_tflops"] / float(row["mcblas_tflops"])
            except subprocess.CalledProcessError:
                row["local_family"] = "launch_failed"
        elif binary is not None:
            row["local_family"] = "unsupported_shape"
        results.append(row)
        print(
            f"[{idx}/{len(cases)}] {case.family} {case.dtype} "
            f"M={case.m} N={case.n} K={case.k} "
            f"muxi={row['muxi_tflops']:.3f} "
            f"local={row['local_tflops'] if row['local_tflops'] != '' else 'NA'}"
        )

    out_path = Path(OUT_COMPARE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "family",
                "dtype",
                "m",
                "n",
                "k",
                "muxi_runtime_ns",
                "muxi_tflops",
                "muxi_ref_runtime_ns",
                "muxi_ref_tflops",
                "muxi_speedup",
                "muxi_err_max",
                "local_family",
                "local_runtime_ns",
                "local_tflops",
                "local_err_max",
                "local_vs_muxi_ratio",
                "mcblas_runtime_ns",
                "mcblas_tflops",
                "mcblas_err_max",
                "local_vs_mcblas_ratio",
            ],
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
