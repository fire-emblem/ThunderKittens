from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = REPO_ROOT / "kernels" / "gemm" / "reports"

RUNTIME_PATTERNS = (
    re.compile(r"Average kernel execution time:\s*([0-9]+(?:\.[0-9]+)?)\s*us"),
    re.compile(r"Average runtime:\s*([0-9]+(?:\.[0-9]+)?)\s*ms"),
)
TFLOPS_PATTERNS = (
    re.compile(r"Achieved performance:\s*([0-9]+(?:\.[0-9]+)?)\s*TFLOPs"),
    re.compile(r"Performance:\s*([0-9]+(?:\.[0-9]+)?)\s*TFLOP/s"),
)
PROBLEM_BLOCK_PATTERN = re.compile(
    r"Problem size:\s*M=(\d+),\s*N=(\d+),\s*K=(\d+)(.*?)(?=Problem size:\s*M=|\Z)",
    re.DOTALL,
)


@dataclass(frozen=True)
class Implementation:
    name: str
    rel_dir: str
    supported_targets: tuple[str, ...]
    multi_problem: bool = False


@dataclass(frozen=True)
class ShapeSpec:
    m: int
    n: int
    k: int

    def compact(self) -> str:
        return f"{self.m}x{self.n}x{self.k}"


IMPLEMENTATIONS = (
    Implementation("TK H100", "kernels/gemm/bf16_h100", ("H100",)),
    Implementation("TK Ampere", "kernels/gemm/bf16_ampere", ("RTX4080",)),
    Implementation(
        "cuBLAS",
        "kernels/gemm/baselines/bf16_cublas",
        ("H100", "B200", "B300", "A100", "RTX4080"),
        multi_problem=True,
    ),
    Implementation(
        "cuBLASLt",
        "kernels/gemm/baselines/bf16_cublas_lt",
        ("H100", "B200", "B300", "A100", "RTX4080"),
        multi_problem=True,
    ),
)

DEFAULT_IMPLEMENTATIONS_BY_TARGET = {
    "RTX4080": ("TK Ampere", "cuBLAS"),
    "H100": ("TK H100", "cuBLAS", "cuBLASLt"),
    "B200": ("cuBLAS", "cuBLASLt"),
    "B300": ("cuBLAS", "cuBLASLt"),
    "A100": ("cuBLAS", "cuBLASLt"),
}

DEFAULT_SHAPES_BY_TARGET = {
    "RTX4080": (
        ShapeSpec(512, 512, 512),
        ShapeSpec(1024, 1024, 1024),
        ShapeSpec(2048, 2048, 2048),
        ShapeSpec(4096, 4096, 4096),
        ShapeSpec(4096, 8192, 4096),
        ShapeSpec(8192, 4096, 4096),
    ),
    "H100": (
        ShapeSpec(4096, 4096, 4096),
        ShapeSpec(4096, 8192, 4096),
        ShapeSpec(8192, 4096, 4096),
    ),
}


def parse_benchmark_output(output: str) -> dict[str, float]:
    avg_time_us = None
    for pattern in RUNTIME_PATTERNS:
        match = pattern.search(output)
        if match is None:
            continue
        value = float(match.group(1))
        avg_time_us = value if "us" in pattern.pattern else value * 1000.0
        break

    tflops = None
    for pattern in TFLOPS_PATTERNS:
        match = pattern.search(output)
        if match is None:
            continue
        tflops = float(match.group(1))
        break

    if avg_time_us is None or tflops is None:
        raise ValueError("Could not parse runtime and TFLOP/s from benchmark output")

    return {"avg_time_us": avg_time_us, "tflops": tflops}


def parse_shape_spec(value: str) -> ShapeSpec:
    match = re.fullmatch(r"(\d+)x(\d+)x(\d+)", value.strip())
    if match is None:
        raise ValueError(f"Invalid shape spec {value!r}. Expected format MxNxK, for example 4096x4096x4096.")
    return ShapeSpec(m=int(match.group(1)), n=int(match.group(2)), k=int(match.group(3)))


def find_problem_result(output: str, m: int, n: int, k: int) -> dict[str, float]:
    for match in PROBLEM_BLOCK_PATTERN.finditer(output):
        block_m = int(match.group(1))
        block_n = int(match.group(2))
        block_k = int(match.group(3))
        if (block_m, block_n, block_k) != (m, n, k):
            continue
        return parse_benchmark_output(match.group(0))
    raise ValueError(f"Could not find benchmark block for M={m}, N={n}, K={k}")


def filter_implementations(
    implementations: tuple[Implementation, ...],
    gpu_target: str | None,
    implementation_names: tuple[str, ...] | None,
) -> list[Implementation]:
    if implementation_names:
        selected = [item for item in implementations if item.name in implementation_names]
        name_set = {item.name for item in selected}
        missing = [name for name in implementation_names if name not in name_set]
        if missing:
            raise ValueError(f"Unknown implementations requested: {', '.join(missing)}")
        return selected

    if gpu_target is None:
        return []

    default_names = DEFAULT_IMPLEMENTATIONS_BY_TARGET.get(gpu_target, ())
    return [item for item in implementations if item.name in default_names]


def rank_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_tflops = max(
        (result["tflops"] for result in results if result["status"] == "ok" and result["tflops"] is not None),
        default=None,
    )

    ranked = []
    for result in results:
        ranked_result = dict(result)
        if best_tflops and ranked_result["status"] == "ok" and ranked_result["tflops"] is not None:
            ranked_result["relative_to_best"] = ranked_result["tflops"] / best_tflops
        else:
            ranked_result["relative_to_best"] = None
        ranked.append(ranked_result)
    return ranked


def format_relative_to_best(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def summarize_shape_results(results_by_shape: dict[ShapeSpec, list[dict[str, Any]]]) -> str:
    lines = []
    for shape, results in results_by_shape.items():
        ok_results = [result for result in results if result["status"] == "ok"]
        if not ok_results:
            lines.append(f"{shape.compact()}: no successful runs")
            continue
        best = max(ok_results, key=lambda item: item["tflops"])
        lines.append(f"{shape.compact()}: {best['implementation']} at {best['tflops']:.2f} TFLOP/s")
    return "\n".join(lines)


def write_markdown_report(
    path: Path,
    gpu_name: str,
    generated_at: str,
    problem_size: tuple[int, int, int],
    results: list[dict[str, Any]],
) -> None:
    m, n, k = problem_size
    lines = [
        "# BF16 GEMM Compare Report",
        "",
        f"- Generated at: `{generated_at}`",
        f"- GPU: `{gpu_name}`",
        f"- Problem size: `M={m}, N={n}, K={k}`",
        "",
        "| Implementation | Status | Avg Time (us) | TFLOP/s | Relative to Best |",
        "| --- | --- | ---: | ---: | ---: |",
    ]

    for result in results:
        avg_time = "-" if result["avg_time_us"] is None else f"{result['avg_time_us']:.1f}"
        tflops = "-" if result["tflops"] is None else f"{result['tflops']:.2f}"
        rel = format_relative_to_best(result.get("relative_to_best"))
        lines.append(
            f"| {result['implementation']} | {result['status']} | {avg_time} | {tflops} | {rel} |"
        )

    path.write_text("\n".join(lines) + "\n")


def write_json_report(
    path: Path,
    gpu_name: str,
    generated_at: str,
    problem_size: tuple[int, int, int],
    results: list[dict[str, Any]],
) -> None:
    payload = {
        "generated_at": generated_at,
        "gpu_name": gpu_name,
        "problem_size": {"m": problem_size[0], "n": problem_size[1], "k": problem_size[2]},
        "results": results,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def detect_gpu_name() -> str:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"

    lines = completed.stdout.strip().splitlines()
    return lines[0].strip() if lines else "unknown"


def infer_gpu_target(gpu_name: str) -> str | None:
    normalized = gpu_name.lower()
    if "h100" in normalized:
        return "H100"
    if "b200" in normalized:
        return "B200"
    if "b300" in normalized:
        return "B300"
    if "a100" in normalized:
        return "A100"
    if "rtx 4080" in normalized:
        return "RTX4080"
    return None


def run_command(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def run_implementation(
    implementation: Implementation,
    gpu_target: str | None,
    problem_size: tuple[int, int, int],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "implementation": implementation.name,
        "status": "unsupported",
        "avg_time_us": None,
        "tflops": None,
    }

    if gpu_target is None or gpu_target not in implementation.supported_targets:
        return result

    workdir = REPO_ROOT / implementation.rel_dir
    build = run_command(["make", f"GPU={gpu_target}", "-j4"], workdir)
    if build.returncode != 0:
        result["status"] = "build_failed"
        result["stderr"] = build.stderr.strip() or build.stdout.strip()
        return result

    executable = next(workdir.glob("*.out"), None)
    if executable is None:
        result["status"] = "missing_binary"
        return result

    run = run_command(
        [
            str(executable),
            "--m",
            str(problem_size[0]),
            "--n",
            str(problem_size[1]),
            "--k",
            str(problem_size[2]),
            "--no-check",
        ],
        workdir,
    )
    if run.returncode != 0:
        result["status"] = "run_failed"
        result["stderr"] = run.stderr.strip() or run.stdout.strip()
        return result

    output = run.stdout + "\n" + run.stderr
    try:
        parsed = (
            find_problem_result(output, *problem_size)
            if implementation.multi_problem
            else parse_benchmark_output(output)
        )
    except ValueError:
        result["status"] = "parse_failed"
        result["stderr"] = output.strip()
        return result

    result["status"] = "ok"
    result.update(parsed)
    return result


def summarize_results(results: list[dict[str, Any]]) -> str:
    ok_results = [result for result in results if result["status"] == "ok"]
    if not ok_results:
        return "No successful BF16 GEMM runs."
    best = max(ok_results, key=lambda item: item["tflops"])
    return f"Best: {best['implementation']} at {best['tflops']:.2f} TFLOP/s"


def print_console_summary(results: list[dict[str, Any]]) -> None:
    print("Implementation       Status         Avg Time (us)   TFLOP/s   Relative")
    print("---------------------------------------------------------------------")
    for result in results:
        avg_time = "-" if result["avg_time_us"] is None else f"{result['avg_time_us']:.1f}"
        tflops = "-" if result["tflops"] is None else f"{result['tflops']:.2f}"
        rel = format_relative_to_best(result.get("relative_to_best"))
        print(
            f"{result['implementation']:<20} {result['status']:<13} "
            f"{avg_time:>13}   {tflops:>7}   {rel:>8}"
        )


def print_console_summary_for_shape(shape: ShapeSpec, results: list[dict[str, Any]]) -> None:
    print(f"Problem size: M={shape.m}, N={shape.n}, K={shape.k}")
    print_console_summary(results)


def resolve_shapes(gpu_target: str | None, shape_args: list[str] | None) -> list[ShapeSpec]:
    if shape_args:
        return [parse_shape_spec(arg) for arg in shape_args]
    return list(DEFAULT_SHAPES_BY_TARGET.get(gpu_target or "", (ShapeSpec(4096, 4096, 4096),)))


def write_multi_shape_markdown_report(
    path: Path,
    gpu_name: str,
    generated_at: str,
    results_by_shape: dict[ShapeSpec, list[dict[str, Any]]],
) -> None:
    lines = [
        "# BF16 GEMM Compare Report",
        "",
        f"- Generated at: `{generated_at}`",
        f"- GPU: `{gpu_name}`",
        "",
    ]
    for shape, results in results_by_shape.items():
        lines.extend(
            [
                f"## {shape.compact()}",
                "",
                "| Implementation | Status | Avg Time (us) | TFLOP/s | Relative to Best |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for result in results:
            avg_time = "-" if result["avg_time_us"] is None else f"{result['avg_time_us']:.1f}"
            tflops = "-" if result["tflops"] is None else f"{result['tflops']:.2f}"
            rel = format_relative_to_best(result.get("relative_to_best"))
            lines.append(
                f"| {result['implementation']} | {result['status']} | {avg_time} | {tflops} | {rel} |"
            )
        lines.append("")
    path.write_text("\n".join(lines))


def write_multi_shape_json_report(
    path: Path,
    gpu_name: str,
    generated_at: str,
    results_by_shape: dict[ShapeSpec, list[dict[str, Any]]],
) -> None:
    payload = {
        "generated_at": generated_at,
        "gpu_name": gpu_name,
        "results_by_shape": [
            {
                "shape": {"m": shape.m, "n": shape.n, "k": shape.k},
                "results": results,
            }
            for shape, results in results_by_shape.items()
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare BF16 GEMM performance across TK and BLAS baselines.")
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR, help="Directory for Markdown and JSON reports.")
    parser.add_argument("--m", type=int, default=4096, help="Problem size M.")
    parser.add_argument("--n", type=int, default=4096, help="Problem size N.")
    parser.add_argument("--k", type=int, default=4096, help="Problem size K.")
    parser.add_argument(
        "--shape",
        action="append",
        dest="shapes",
        help="Benchmark shape in MxNxK form. Repeat to run multiple shapes.",
    )
    parser.add_argument(
        "--implementations",
        nargs="+",
        help="Optional implementation names to run. Defaults depend on detected GPU.",
    )
    args = parser.parse_args()

    gpu_name = detect_gpu_name()
    gpu_target = infer_gpu_target(gpu_name)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    shapes = resolve_shapes(gpu_target, args.shapes)

    selected = filter_implementations(
        IMPLEMENTATIONS,
        gpu_target=gpu_target,
        implementation_names=tuple(args.implementations) if args.implementations else None,
    )
    if not selected:
        raise SystemExit(f"No implementations selected for GPU target {gpu_target!r}.")

    results_by_shape: dict[ShapeSpec, list[dict[str, Any]]] = {}
    for shape in shapes:
        problem_size = (shape.m, shape.n, shape.k)
        results = [run_implementation(item, gpu_target, problem_size) for item in selected]
        results_by_shape[shape] = rank_results(results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_bf16_gemm_compare")
    markdown_path = args.output_dir / f"{stem}.md"
    json_path = args.output_dir / f"{stem}.json"

    if len(shapes) == 1:
        only_shape = shapes[0]
        only_results = results_by_shape[only_shape]
        write_markdown_report(
            markdown_path,
            gpu_name=gpu_name,
            generated_at=generated_at,
            problem_size=(only_shape.m, only_shape.n, only_shape.k),
            results=only_results,
        )
        write_json_report(
            json_path,
            gpu_name=gpu_name,
            generated_at=generated_at,
            problem_size=(only_shape.m, only_shape.n, only_shape.k),
            results=only_results,
        )
    else:
        write_multi_shape_markdown_report(markdown_path, gpu_name, generated_at, results_by_shape)
        write_multi_shape_json_report(json_path, gpu_name, generated_at, results_by_shape)

    print(f"GPU: {gpu_name}")
    for shape in shapes:
        print_console_summary_for_shape(shape, results_by_shape[shape])
    print(summarize_shape_results(results_by_shape))
    print(f"Markdown report: {markdown_path}")
    print(f"JSON report: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
