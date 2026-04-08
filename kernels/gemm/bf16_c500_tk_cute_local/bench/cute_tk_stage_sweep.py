#!/usr/bin/env python3

import csv
import os
import re
import subprocess
from pathlib import Path


ROOT = Path("/data/ThunderKittens/kernels/gemm/bf16_c500_tk_cute_local")
OUT_CSV = Path(
    os.environ.get(
        "CUTE_TK_STAGE_SWEEP_OUT",
        str(ROOT / "results/current/cute_tk_stage_sweep.csv"),
    )
)

M = int(os.environ.get("CUTE_TK_SWEEP_M", "4608"))
N = int(os.environ.get("CUTE_TK_SWEEP_N", "128"))
K = int(os.environ.get("CUTE_TK_SWEEP_K", "3584"))
NTILE = int(os.environ.get("CUTE_TK_SWEEP_NTILE", str(N)))
APERWARP = int(os.environ.get("CUTE_TK_SWEEP_APERWARP", "2"))
SPLITN = int(os.environ.get("CUTE_TK_SWEEP_SPLITN", "2"))
SPLITK = int(os.environ.get("CUTE_TK_SWEEP_SPLITK", "1"))
WARMUP = int(os.environ.get("CUTE_TK_SWEEP_WARMUP", "1"))
PROFILE = int(os.environ.get("CUTE_TK_SWEEP_PROFILE", "3"))
STAGES = [int(item) for item in os.environ.get("CUTE_TK_SWEEP_STAGES", "2,3,4").split(",")]

PERF_RE = re.compile(r"Performance:\s+([0-9.]+)\s+TFLOP/s")
RUNTIME_RE = re.compile(r"Average runtime:\s+([0-9.]+)\s+ms")
ERR_RE = re.compile(r"err max:\s+([0-9.]+)")


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def build_binary(stage_count: int) -> Path:
    out_name = f"cute_tk_stage_sweep_s{stage_count}.out"
    cmd = [
        "make",
        "-B",
        "-C",
        str(ROOT),
        "GPU=C500",
        "SRC=bf16_c500_tk_cute_local_gemm.cu",
        f"OUT={out_name}",
        f"CMD=./{out_name}",
        (
            "EXTRA_NVCCFLAGS="
            f"-DTK_CUTE_LOCAL_USE_CONTINUOUSC -DTK_CUTE_LOCAL_USE_REUSEA "
            f"-DBF16_C500_TK_CUTE_LOCAL_M={M} "
            f"-DBF16_C500_TK_CUTE_LOCAL_N={N} "
            f"-DBF16_C500_TK_CUTE_LOCAL_K={K} "
            f"-DTK_CUTE_LOCAL_NTILE={NTILE} "
            f"-DTK_CUTE_LOCAL_APERWARP={APERWARP} "
            f"-DTK_CUTE_LOCAL_SPLITN={SPLITN} "
            f"-DTK_CUTE_LOCAL_SPLITK={SPLITK} "
            f"-DTK_CUTE_LOCAL_STAGES={stage_count} "
            f"-DBF16_C500_TK_CUTE_LOCAL_WARMUP_ITERS={WARMUP} "
            f"-DBF16_C500_TK_CUTE_LOCAL_PROFILE_ITERS={PROFILE}"
        ),
    ]
    proc = run_cmd(cmd)
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout)
    return ROOT / out_name


def main() -> int:
    rows: list[dict[str, str | int | float]] = []
    for stage_count in STAGES:
        binary = build_binary(stage_count)
        proc = run_cmd([str(binary)])
        row: dict[str, str | int | float] = {
            "m": M,
            "n": N,
            "k": K,
            "n_tile": NTILE,
            "a_per_warp": APERWARP,
            "split_n": SPLITN,
            "split_k": SPLITK,
            "stages": stage_count,
        }
        if proc.returncode == 0:
            text = proc.stdout
            row["status"] = "ok"
            row["runtime_ms"] = float(RUNTIME_RE.search(text).group(1))
            row["tflops"] = float(PERF_RE.search(text).group(1))
            row["err_max"] = float(ERR_RE.search(text).group(1))
        else:
            row["status"] = "invalid_config"
            row["runtime_ms"] = ""
            row["tflops"] = ""
            row["err_max"] = ""
            row["message"] = proc.stdout.strip().splitlines()[-1]
        rows.append(row)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "m",
                "n",
                "k",
                "n_tile",
                "a_per_warp",
                "split_n",
                "split_k",
                "stages",
                "status",
                "runtime_ms",
                "tflops",
                "err_max",
                "message",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {OUT_CSV}")
    for row in rows:
        if row["status"] == "ok":
            print(
                f"stage={row['stages']}: "
                f"{row['tflops']} TFLOP/s, runtime={row['runtime_ms']} ms, "
                f"err_max={row['err_max']}"
            )
        else:
            print(f"stage={row['stages']}: {row['status']} ({row['message']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
