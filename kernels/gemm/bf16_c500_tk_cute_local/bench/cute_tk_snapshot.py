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
        "CUTE_TK_SNAPSHOT_OUT",
        str(ROOT / "results/current/cute_tk_reusea_snapshot.csv"),
    )
)
RUNS = int(os.environ.get("CUTE_TK_SNAPSHOT_RUNS", "5"))

PERF_RE = re.compile(r"Performance:\s+([0-9.]+)\s+TFLOP/s")
RUNTIME_RE = re.compile(r"Average runtime:\s+([0-9.]+)\s+ms")
FAMILY_RE = re.compile(r"Family:\s+(.+)")
CASE_RE = re.compile(r"Case:\s+(.+)")
PROBLEM_RE = re.compile(r"Problem size:\s+M=(\d+), N=(\d+), K=(\d+)")
ERR_RE = re.compile(r"err max:\s+([0-9.]+)")
PERSIST_RE = re.compile(r"MetaX C500\s+\|\s+\d+\s+(\w+)\s+\|")


@dataclass(frozen=True)
class Target:
    name: str
    out_name: str
    extra_flags: str


TARGETS = [
    Target(
        name="reusea_continuousc",
        out_name="bf16_c500_tk_cute_reusea.out",
        extra_flags="-DTK_CUTE_LOCAL_USE_CONTINUOUSC -DTK_CUTE_LOCAL_USE_REUSEA",
    ),
    Target(
        name="reusea_layoutc",
        out_name="bf16_c500_tk_cute_reusea_layoutc.out",
        extra_flags="-DTK_CUTE_LOCAL_USE_CONTINUOUSC -DTK_CUTE_LOCAL_USE_REUSEA_LAYOUTC",
    ),
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
            "SRC=bf16_c500_tk_cute_local_gemm.cu",
            f"OUT={target.out_name}",
            f"CMD=./{target.out_name}",
            f"EXTRA_NVCCFLAGS={target.extra_flags}",
        ]
    )
    return ROOT / target.out_name


def parse_output(text: str) -> dict[str, str | float | int]:
    perf = float(PERF_RE.search(text).group(1))
    runtime_ms = float(RUNTIME_RE.search(text).group(1))
    family = FAMILY_RE.search(text).group(1).strip()
    case_name = CASE_RE.search(text).group(1).strip()
    m, n, k = [int(v) for v in PROBLEM_RE.search(text).groups()]
    err_max = float(ERR_RE.search(text).group(1))
    return {
        "case": case_name,
        "family": family,
        "m": m,
        "n": n,
        "k": k,
        "runtime_ms": runtime_ms,
        "tflops": perf,
        "err_max": err_max,
    }


def query_persistence_mode() -> str:
    text = run_cmd(["mx-smi"])
    match = PERSIST_RE.search(text)
    return match.group(1) if match else "unknown"


def sample_target(binary: Path) -> tuple[dict[str, str | float | int], list[float], list[float]]:
    parsed_runs: list[dict[str, str | float | int]] = []
    for _ in range(RUNS):
        parsed_runs.append(parse_output(run_cmd([str(binary)])))
    first = parsed_runs[0]
    return (
        first,
        [float(item["runtime_ms"]) for item in parsed_runs],
        [float(item["tflops"]) for item in parsed_runs],
    )


def main() -> int:
    persistence_mode = query_persistence_mode()
    rows: list[dict[str, str | float | int]] = []
    for target in TARGETS:
        binary = build_binary(target)
        first, runtimes_ms, tflops = sample_target(binary)
        rows.append(
            {
                "target": target.name,
                "case": first["case"],
                "family": first["family"],
                "m": first["m"],
                "n": first["n"],
                "k": first["k"],
                "runs": RUNS,
                "persistence_mode": persistence_mode,
                "runtime_ms_min": min(runtimes_ms),
                "runtime_ms_avg": statistics.mean(runtimes_ms),
                "runtime_ms_max": max(runtimes_ms),
                "tflops_min": min(tflops),
                "tflops_avg": statistics.mean(tflops),
                "tflops_max": max(tflops),
                "err_max": first["err_max"],
            }
        )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "target",
                "case",
                "family",
                "m",
                "n",
                "k",
                "runs",
                "persistence_mode",
                "runtime_ms_min",
                "runtime_ms_avg",
                "runtime_ms_max",
                "tflops_min",
                "tflops_avg",
                "tflops_max",
                "err_max",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {OUT_CSV}")
    for row in rows:
        print(
            f"{row['target']}: "
            f"TFLOP/s min/avg/max = {row['tflops_min']:.3f}/"
            f"{row['tflops_avg']:.3f}/{row['tflops_max']:.3f}, "
            f"runtime ms min/avg/max = {row['runtime_ms_min']:.6f}/"
            f"{row['runtime_ms_avg']:.6f}/{row['runtime_ms_max']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
