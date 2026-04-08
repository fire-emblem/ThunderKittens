#!/usr/bin/env python3

import csv
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path("/data/ThunderKittens/kernels/gemm/bf16_c500_tk_cute_local")
BASELINE = Path(
    os.environ.get(
        "CUTE_TK_BASELINE_CSV",
        str(ROOT / "results/current/cute_tk_reusea_snapshot.csv"),
    )
)
CURRENT = Path(
    os.environ.get(
        "CUTE_TK_CURRENT_CSV",
        str(ROOT / "results/current/cute_tk_reusea_snapshot_check.csv"),
    )
)
MIN_AVG_RATIO = float(os.environ.get("CUTE_TK_MIN_AVG_RATIO", "0.97"))


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open() as f:
        return {row["target"]: row for row in csv.DictReader(f)}


def main() -> int:
    env = os.environ.copy()
    env["CUTE_TK_SNAPSHOT_OUT"] = str(CURRENT)
    subprocess.run(
        ["python3", str(ROOT / "bench/cute_tk_snapshot.py")],
        cwd=ROOT,
        env=env,
        check=True,
    )

    baseline_rows = read_rows(BASELINE)
    current_rows = read_rows(CURRENT)

    failures: list[str] = []
    for target, baseline_row in baseline_rows.items():
        current_row = current_rows.get(target)
        if current_row is None:
            failures.append(f"{target}: missing in current snapshot")
            continue
        baseline_avg = float(baseline_row["tflops_avg"])
        current_avg = float(current_row["tflops_avg"])
        ratio = current_avg / baseline_avg if baseline_avg else 0.0
        print(
            f"{target}: baseline_avg={baseline_avg:.3f} "
            f"current_avg={current_avg:.3f} ratio={ratio:.4f}"
        )
        if ratio < MIN_AVG_RATIO:
            failures.append(
                f"{target}: avg TFLOP/s ratio {ratio:.4f} < required {MIN_AVG_RATIO:.4f}"
            )

    if failures:
        print("Performance regression detected:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print(f"Snapshot check passed against {BASELINE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
