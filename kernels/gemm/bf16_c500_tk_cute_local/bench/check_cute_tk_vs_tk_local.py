#!/usr/bin/env python3

import csv
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path("/data/ThunderKittens/kernels/gemm/bf16_c500_tk_cute_local")
CURRENT = Path(
    os.environ.get(
        "CUTE_TK_VS_TK_LOCAL_CURRENT",
        str(ROOT / "results/current/cute_tk_vs_tk_local_check.csv"),
    )
)
MIN_RATIO = float(os.environ.get("CUTE_TK_VS_TK_LOCAL_MIN_RATIO", "0.95"))

PAIRS = [
    ("cute_reusea_n128", "tk_local_n128"),
    ("cute_reusea_n256", "tk_local_n256"),
    ("cute_reusea_3584x128x3584", "tk_local_3584x128x3584"),
    ("cute_reusea_3584x128x18944", "tk_local_3584x128x18944"),
    ("cute_reusea_37888x256x3584", "tk_local_37888x256x3584"),
    ("cute_continuousc_37888x128x3584_bf16", "tk_local_37888x128x3584_bf16"),
    ("cute_continuousc_37888x128x3584_fp16", "tk_local_37888x128x3584_fp16"),
]


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open() as f:
        return {row["target"]: row for row in csv.DictReader(f)}


def main() -> int:
    env = os.environ.copy()
    env["CUTE_TK_VS_TK_LOCAL_OUT"] = str(CURRENT)
    subprocess.run(
        ["python3", str(ROOT / "bench/cute_tk_vs_tk_local.py")],
        cwd=ROOT,
        env=env,
        check=True,
    )

    rows = read_rows(CURRENT)
    failures: list[str] = []
    for cute_key, tk_key in PAIRS:
        cute_row = rows.get(cute_key)
        tk_row = rows.get(tk_key)
        if cute_row is None or tk_row is None:
            failures.append(f"missing pair rows for {cute_key}/{tk_key}")
            continue
        cute_avg = float(cute_row["tflops_avg"])
        tk_avg = float(tk_row["tflops_avg"])
        ratio = cute_avg / tk_avg if tk_avg else 0.0
        print(
            f"{cute_key} vs {tk_key}: "
            f"cute_avg={cute_avg:.3f} tk_avg={tk_avg:.3f} ratio={ratio:.4f}"
        )
        if ratio < MIN_RATIO:
            failures.append(
                f"{cute_key}: avg ratio {ratio:.4f} < required {MIN_RATIO:.4f}"
            )

    if failures:
        print("cute_tk vs tk_local regression detected:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print(f"cute_tk vs tk_local check passed against ratio >= {MIN_RATIO:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
