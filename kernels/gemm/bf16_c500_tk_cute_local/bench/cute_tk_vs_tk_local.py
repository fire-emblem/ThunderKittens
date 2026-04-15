#!/usr/bin/env python3

import csv
import os
import re
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path("/data/ThunderKittens/kernels/gemm/bf16_c500_tk_cute_local")
OUT_CSV = Path(
    os.environ.get(
        "CUTE_TK_VS_TK_LOCAL_OUT",
        str(ROOT / "results/current/cute_tk_vs_tk_local.csv"),
    )
)
RUNS = int(os.environ.get("CUTE_TK_VS_TK_LOCAL_RUNS", "3"))
TARGET_FILTER = tuple(filter(None, os.environ.get("CUTE_TK_TARGET_FILTER", "").split(",")))

PERF_RE = re.compile(r"Performance:\s+([0-9.]+)\s+TFLOP/s")
RUNTIME_RE = re.compile(r"Average runtime:\s+([0-9.]+)\s+ms")
FAMILY_RE = re.compile(r"Family:\s+(.+)")
PROBLEM_RE = re.compile(r"Problem size:\s+M=(\d+), N=(\d+), K=(\d+)")
ERR_RE = re.compile(r"err max:\s+([0-9.]+)")


@dataclass(frozen=True)
class Target:
    name: str
    dtype: str
    src: str
    out_name: str
    extra_flags: str
    env: dict[str, str] | None = None


def cute_runtime_target(name: str, m: int, n: int, k: int, *, env: dict[str, str] | None = None) -> Target:
    merged_env = {
        "TK_CUTE_M": str(m),
        "TK_CUTE_N": str(n),
        "TK_CUTE_K": str(k),
        "TK_CUTE_WARMUP": "1",
        "TK_CUTE_PROFILE": "3",
    }
    if env:
        merged_env.update(env)
    return Target(
        name=name,
        dtype="bf16",
        src="cute_tk_runtime_gemm.cu",
        out_name=f"{name}_cmp.out",
        extra_flags="",
        env=merged_env,
    )

def tk_local_target(name: str, m: int, n: int, k: int) -> Target:
    return Target(
        name=name,
        dtype="bf16",
        src="tk_local_runtime_gemm.cu",
        out_name=f"{name}_cmp.out",
        extra_flags="-DTK_LOCAL_USE_CONTINUOUSC",
        env={
            "TK_LOCAL_M": str(m),
            "TK_LOCAL_N": str(n),
            "TK_LOCAL_K": str(k),
            "TK_LOCAL_WARMUP": "1",
            "TK_LOCAL_PROFILE": "3",
        },
    )

def tk_local_layoutc_target(name: str, m: int, n: int, k: int) -> Target:
    return Target(
        name=name,
        dtype="bf16",
        src="bf16_c500_tk_local_gemm.cu",
        out_name=f"{name}_cmp.out",
        extra_flags=(
            f"-DBF16_C500_MUXI_NATIVE_M={m} "
            f"-DBF16_C500_MUXI_NATIVE_N={n} "
            f"-DBF16_C500_MUXI_NATIVE_K={k} "
            "-DBF16_C500_MUXI_NATIVE_WARMUP_ITERS=1 "
            "-DBF16_C500_MUXI_NATIVE_PROFILE_ITERS=3"
        ),
    )

TARGETS = [
    cute_runtime_target("cute_square_tt_256x256x64_bf16", 256, 256, 64, env={"TK_CUTE_USE_SQUARE_TT256": "1"}),

    cute_runtime_target("cute_shape_selected_best_1664x1024x16384_bf16", 1664, 1024, 16384, env={"TK_CUTE_USE_SHAPE_AWARE": "1"}),
    cute_runtime_target("cute_swizzled_tn_1664x1024x16384_bf16", 1664, 1024, 16384, env={"TK_CUTE_USE_SWIZZLED_TN": "1"}),
    cute_runtime_target("cute_layoutc_1664x1024x16384_bf16", 1664, 1024, 16384),

    cute_runtime_target("cute_shape_selected_best_2048cube_bf16", 2048, 2048, 2048, env={"TK_CUTE_USE_SHAPE_AWARE": "1"}),
    cute_runtime_target("cute_square_tt_2048x2048x2048_bf16", 2048, 2048, 2048, env={"TK_CUTE_USE_SQUARE_TT256": "1"}),
    cute_runtime_target("cute_swizzled_tn_2048cube_bf16", 2048, 2048, 2048, env={"TK_CUTE_USE_SWIZZLED_TN": "1"}),
    cute_runtime_target("cute_layoutc_2048x2048x2048_bf16", 2048, 2048, 2048),
    tk_local_layoutc_target("tk_local_layoutc_2048x2048x2048_bf16", 2048, 2048, 2048),

    cute_runtime_target("cute_shape_selected_best_4096cube_bf16", 4096, 4096, 4096, env={"TK_CUTE_USE_SHAPE_AWARE": "1"}),
    cute_runtime_target("cute_square_tt_4096x4096x4096_bf16", 4096, 4096, 4096, env={"TK_CUTE_USE_SQUARE_TT256": "1"}),
    cute_runtime_target("cute_swizzled_tn_4096cube_bf16", 4096, 4096, 4096, env={"TK_CUTE_USE_SWIZZLED_TN": "1"}),
    cute_runtime_target("cute_layoutc_4096x4096x4096_bf16", 4096, 4096, 4096),
    tk_local_layoutc_target("tk_local_layoutc_4096x4096x4096_bf16", 4096, 4096, 4096),

    cute_runtime_target("cute_shape_selected_best_8192cube_bf16", 8192, 8192, 8192, env={"TK_CUTE_USE_SHAPE_AWARE": "1"}),
    cute_runtime_target("cute_swizzled_tn_8192cube_bf16", 8192, 8192, 8192, env={"TK_CUTE_USE_SWIZZLED_TN": "1"}),
    cute_runtime_target("cute_layoutc_8192x8192x8192_bf16", 8192, 8192, 8192),

    cute_runtime_target("cute_shape_selected_best_4608x128x3584_bf16", 4608, 128, 3584, env={"TK_CUTE_USE_SHAPE_AWARE": "1"}),
    cute_runtime_target("cute_swizzled_tn_4608x128x3584_bf16", 4608, 128, 3584, env={"TK_CUTE_USE_SWIZZLED_TN": "1"}),
    cute_runtime_target("cute_continuousc_reusea_4608x128x3584_bf16", 4608, 128, 3584),
    tk_local_target("tk_local_continuousc_4608x128x3584_bf16", 4608, 128, 3584),

    cute_runtime_target("cute_shape_selected_best_4608x256x3584_bf16", 4608, 256, 3584, env={"TK_CUTE_USE_SHAPE_AWARE": "1"}),
    cute_runtime_target("cute_swizzled_tn_4608x256x3584_bf16", 4608, 256, 3584, env={"TK_CUTE_USE_SWIZZLED_TN": "1"}),
    cute_runtime_target("cute_continuousc_reusea_4608x256x3584_bf16", 4608, 256, 3584),
    tk_local_target("tk_local_continuousc_4608x256x3584_bf16", 4608, 256, 3584),

    cute_runtime_target("cute_shape_selected_best_3584x128x3584_bf16", 3584, 128, 3584, env={"TK_CUTE_USE_SHAPE_AWARE": "1"}),
    cute_runtime_target("cute_swizzled_tn_3584x128x3584_bf16", 3584, 128, 3584, env={"TK_CUTE_USE_SWIZZLED_TN": "1"}),
    cute_runtime_target("cute_continuousc_reusea_3584x128x3584", 3584, 128, 3584),
    tk_local_target("tk_local_3584x128x3584", 3584, 128, 3584),

    cute_runtime_target("cute_shape_selected_best_3584x128x18944_bf16", 3584, 128, 18944, env={"TK_CUTE_USE_SHAPE_AWARE": "1"}),
    cute_runtime_target("cute_swizzled_tn_3584x128x18944_bf16", 3584, 128, 18944, env={"TK_CUTE_USE_SWIZZLED_TN": "1"}),
    cute_runtime_target("cute_continuousc_reusea_3584x128x18944", 3584, 128, 18944),
    tk_local_target("tk_local_3584x128x18944", 3584, 128, 18944),

    cute_runtime_target("cute_shape_selected_best_37888x256x3584_bf16", 37888, 256, 3584, env={"TK_CUTE_USE_SHAPE_AWARE": "1"}),
    cute_runtime_target("cute_swizzled_tn_37888x256x3584_bf16", 37888, 256, 3584, env={"TK_CUTE_USE_SWIZZLED_TN": "1"}),
    cute_runtime_target("cute_continuousc_reusea_37888x256x3584", 37888, 256, 3584),
    tk_local_target("tk_local_37888x256x3584", 37888, 256, 3584),

    cute_runtime_target("cute_shape_selected_best_37888x128x3584_bf16", 37888, 128, 3584, env={"TK_CUTE_USE_SHAPE_AWARE": "1"}),
    cute_runtime_target("cute_swizzled_tn_37888x128x3584_bf16", 37888, 128, 3584, env={"TK_CUTE_USE_SWIZZLED_TN": "1"}),
]

def run_cmd(cmd: list[str], env: dict[str, str] | None = None) -> str:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc.stdout


def build_binary(target: Target) -> Path:
    run_cmd(
        [
            "make",
            "-B",
            "-C",
            str(ROOT),
            "GPU=C500",
            f"SRC={target.src}",
            f"OUT={target.out_name}",
            f"CMD=./{target.out_name}",
            f"EXTRA_NVCCFLAGS={target.extra_flags}",
        ]
    )
    return ROOT / target.out_name


def parse_mcblas_tflops(text: str) -> float:
    return float(PERF_RE.search(text).group(1))


MCBLAS_DIRECT_TARGETS: dict[tuple[str, int, int, int], dict[str, str]] = {}


def build_and_run_mcblas(dtype: str, m: int, n: int, k: int) -> float | None:
    config = MCBLAS_DIRECT_TARGETS.get((dtype, m, n, k))
    if config is None:
        return None
    run_cmd(
        [
            "make",
            "-B",
            "-C",
            str(ROOT),
            "GPU=C500",
            f"SRC={config['src']}",
            f"OUT={config['out_name']}",
            f"CMD=./{config['out_name']}",
            f"EXTRA_NVCCFLAGS={config['extra_flags']}",
        ]
    )
    text = run_cmd([str(ROOT / config["out_name"])])
    return parse_mcblas_tflops(text)


def parse_output(text: str) -> dict[str, str | float | int]:
    perf = float(PERF_RE.search(text).group(1))
    runtime_ms = float(RUNTIME_RE.search(text).group(1))
    family = FAMILY_RE.search(text).group(1).strip()
    m, n, k = [int(v) for v in PROBLEM_RE.search(text).groups()]
    err_max = float(ERR_RE.search(text).group(1))
    return {
        "family": family,
        "m": m,
        "n": n,
        "k": k,
        "runtime_ms": runtime_ms,
        "tflops": perf,
        "err_max": err_max,
    }


def sample_target(target: Target, binary: Path) -> tuple[dict[str, str | float | int], list[float], list[float]]:
    parsed_runs: list[dict[str, str | float | int]] = []
    env = os.environ.copy()
    if target.env is not None:
        env.update(target.env)
    for _ in range(RUNS):
        parsed_runs.append(parse_output(run_cmd([str(binary)], env=env)))
    first = parsed_runs[0]
    return (
        first,
        [float(item["runtime_ms"]) for item in parsed_runs],
        [float(item["tflops"]) for item in parsed_runs],
    )


def main() -> int:
    mcblas = {}
    with (ROOT / "mcblas_baseline_results.csv").open() as f:
        for row in csv.DictReader(f):
            key = (row["dtype"], int(row["m"]), int(row["n"]), int(row["k"]))
            mcblas[key] = row

    mcblas_direct_cache: dict[tuple[str, int, int, int], float] = {}
    rows: list[dict[str, str | float | int]] = []
    targets = [t for t in TARGETS if not TARGET_FILTER or any(f in t.name for f in TARGET_FILTER)]
    for target in targets:
        env = os.environ.copy()
        if target.env is not None:
            env.update(target.env)
        target_m = int(env.get("TK_CUTE_M", env.get("TK_LOCAL_M", "0")))
        target_n = int(env.get("TK_CUTE_N", env.get("TK_LOCAL_N", "0")))
        target_k = int(env.get("TK_CUTE_K", env.get("TK_LOCAL_K", "0")))
        try:
            binary = build_binary(target)
        except subprocess.CalledProcessError:
            rows.append(
                {
                    "target": target.name,
                    "dtype": target.dtype,
                    "status": "build_failed",
                    "family": "build_failed",
                    "m": target_m,
                    "n": target_n,
                    "k": target_k,
                    "runs": RUNS,
                    "runtime_ms_min": "",
                    "runtime_ms_avg": "",
                    "runtime_ms_max": "",
                    "tflops_min": "",
                    "tflops_avg": "",
                    "tflops_max": "",
                    "err_max": "",
                    "mcblas_tflops": "",
                    "vs_mcblas_ratio": "",
                }
            )
            continue
        try:
            first, runtimes_ms, tflops = sample_target(target, binary)
            status = "ok"
        except subprocess.CalledProcessError:
            rows.append(
                {
                    "target": target.name,
                    "dtype": target.dtype,
                    "status": "runtime_failed",
                    "family": "runtime_failed",
                    "m": target_m,
                    "n": target_n,
                    "k": target_k,
                    "runs": RUNS,
                    "runtime_ms_min": "",
                    "runtime_ms_avg": "",
                    "runtime_ms_max": "",
                    "tflops_min": "",
                    "tflops_avg": "",
                    "tflops_max": "",
                    "err_max": "",
                    "mcblas_tflops": "",
                    "vs_mcblas_ratio": "",
                }
            )
            continue
        mcblas_row = mcblas.get((target.dtype, int(first["m"]), int(first["n"]), int(first["k"])))
        mcblas_tflops = (
            float(mcblas_row["mcblas_tflops"])
            if mcblas_row and mcblas_row.get("mcblas_tflops")
            else None
        )
        if mcblas_tflops is None:
            key = (target.dtype, int(first["m"]), int(first["n"]), int(first["k"]))
            if key not in mcblas_direct_cache:
                value = build_and_run_mcblas(*key)
                if value is not None:
                    mcblas_direct_cache[key] = value
            mcblas_tflops = mcblas_direct_cache.get(key)
        rows.append(
            {
                "target": target.name,
                "dtype": target.dtype,
                "status": status,
                "family": first["family"],
                "m": first["m"],
                "n": first["n"],
                "k": first["k"],
                "runs": RUNS,
                "runtime_ms_min": min(runtimes_ms),
                "runtime_ms_avg": statistics.mean(runtimes_ms),
                "runtime_ms_max": max(runtimes_ms),
                "tflops_min": min(tflops),
                "tflops_avg": statistics.mean(tflops),
                "tflops_max": max(tflops),
                "err_max": first["err_max"],
                "mcblas_tflops": mcblas_tflops if mcblas_tflops is not None else "",
                "vs_mcblas_ratio": (
                    statistics.mean(tflops) / mcblas_tflops
                    if mcblas_tflops not in (None, 0.0)
                    else ""
                ),
            }
        )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "target",
                "dtype",
                "status",
                "family",
                "m",
                "n",
                "k",
                "runs",
                "runtime_ms_min",
                "runtime_ms_avg",
                "runtime_ms_max",
                "tflops_min",
                "tflops_avg",
                "tflops_max",
                "err_max",
                "mcblas_tflops",
                "vs_mcblas_ratio",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {OUT_CSV}")
    for row in rows:
        print(
            (
                f"{row['target']}: status={row['status']}"
                if row['status'] != 'ok' else
                f"{row['target']}: {row['tflops_min']:.3f}/{row['tflops_avg']:.3f}/{row['tflops_max']:.3f} TFLOP/s"
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
