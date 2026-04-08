#!/usr/bin/env python3

import csv
import os
import re
import subprocess
from pathlib import Path

from muxi_baseline import BENCH, WARMUP, build_cases


TK_LOCAL_DIR = Path("/data/ThunderKittens/kernels/gemm/bf16_c500_tk_local")
OUT_CSV = os.environ.get(
    "MCBLAS_BASELINE_OUT",
    str(TK_LOCAL_DIR / "mcblas_baseline_results.csv"),
)


def make_binary(out_name: str, extra_flags: str) -> Path:
    cmd = [
        "make",
        "-C",
        str(TK_LOCAL_DIR),
        "GPU=C500",
        "-j4",
        "SRC=mcblas_runtime_gemm.cu",
        f"OUT={out_name}",
        f"CMD=./{out_name}",
        f"EXTRA_NVCCFLAGS={extra_flags}",
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return TK_LOCAL_DIR / out_name


PERF_RE = re.compile(r"Performance:\s+([0-9.]+)\s+TFLOP/s")
ERR_RE = re.compile(r"err max:\s+([0-9.]+)")
RUNTIME_RE = re.compile(r"Average runtime:\s+([0-9.]+)\s+ms")


def run_binary(binary: Path, dtype: str, m: int, n: int, k: int) -> dict:
    env = os.environ.copy()
    env["TK_MCBLAS_M"] = str(m)
    env["TK_MCBLAS_N"] = str(n)
    env["TK_MCBLAS_K"] = str(k)
    env["TK_MCBLAS_WARMUP"] = str(WARMUP)
    env["TK_MCBLAS_PROFILE"] = str(BENCH)
    proc = subprocess.run(
        [str(binary)],
        check=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    text = proc.stdout
    return {
        "dtype": dtype,
        "m": m,
        "n": n,
        "k": k,
        "mcblas_runtime_ns": float(RUNTIME_RE.search(text).group(1)) * 1e6,
        "mcblas_tflops": float(PERF_RE.search(text).group(1)),
        "mcblas_err_max": float(ERR_RE.search(text).group(1)),
        "status": "ok",
    }


def main() -> int:
    binaries = {
        "bf16": make_binary("tk_mcblas_runtime_bf16.out", "-lmcblas"),
        "fp16": make_binary("tk_mcblas_runtime_fp16.out", "-lmcblas -DTK_MCBLAS_USE_FP16"),
    }

    unique = sorted({(case.dtype, case.m, case.n, case.k) for case in build_cases()})
    print(f"Running {len(unique)} unique mcBLAS baseline shapes")
    out_path = Path(OUT_CSV)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if out_path.exists():
        with out_path.open() as f:
            for row in csv.DictReader(f):
                existing[(row["dtype"], int(row["m"]), int(row["n"]), int(row["k"]))] = row

    write_header = not out_path.exists()
    with out_path.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dtype",
                "m",
                "n",
                "k",
                "mcblas_runtime_ns",
                "mcblas_tflops",
                "mcblas_err_max",
                "status",
            ],
        )
        if write_header:
            writer.writeheader()
        for idx, (dtype, m, n, k) in enumerate(unique, 1):
            key = (dtype, m, n, k)
            if key in existing:
                print(f"[{idx}/{len(unique)}] skip mcblas {dtype} M={m} N={n} K={k}")
                continue
            try:
                result = run_binary(binaries[dtype], dtype, m, n, k)
            except subprocess.CalledProcessError:
                result = {
                    "dtype": dtype,
                    "m": m,
                    "n": n,
                    "k": k,
                    "mcblas_runtime_ns": "",
                    "mcblas_tflops": "",
                    "mcblas_err_max": "",
                    "status": "failed",
                }
            writer.writerow(result)
            f.flush()
            if result["status"] == "ok":
                print(
                    f"[{idx}/{len(unique)}] mcblas {dtype} "
                    f"M={m} N={n} K={k} {float(result['mcblas_tflops']):.3f} TFLOP/s"
                )
            else:
                print(
                    f"[{idx}/{len(unique)}] mcblas {dtype} "
                    f"M={m} N={n} K={k} status={result['status']}"
                )
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
