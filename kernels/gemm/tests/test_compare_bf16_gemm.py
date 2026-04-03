import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kernels.gemm.compare_bf16_gemm import (
    DEFAULT_SHAPES_BY_TARGET,
    Implementation,
    ShapeSpec,
    default_implementations_for_shape,
    find_problem_result,
    format_relative_to_best,
    filter_implementations,
    parse_shape_spec,
    parse_benchmark_output,
    rank_results,
    summarize_shape_results,
    write_json_report,
    write_markdown_report,
)


def test_parse_benchmark_output_extracts_runtime_and_tflops_from_tk_output():
    output = """
    Average kernel execution time: 182.5 us
    Achieved performance: 751.25 TFLOPs
    """

    parsed = parse_benchmark_output(output)

    assert parsed["avg_time_us"] == 182.5
    assert parsed["tflops"] == 751.25


def test_parse_benchmark_output_extracts_runtime_and_tflops_from_cublas_output():
    output = """
    Average runtime: 0.84 ms
    Performance: 162.45 TFLOP/s
    """

    parsed = parse_benchmark_output(output)

    assert parsed["avg_time_us"] == 840.0
    assert parsed["tflops"] == 162.45


def test_find_problem_result_extracts_requested_problem_from_multi_problem_output():
    output = """
    ----------------------------------------
    Problem size: M=2048, N=2048, K=2048
    Average runtime: 0.21 ms
    Performance: 82.09 TFLOP/s

    ----------------------------------------
    Problem size: M=4096, N=4096, K=4096
    Average runtime: 1.34 ms
    Performance: 102.38 TFLOP/s
    """

    parsed = find_problem_result(output, m=4096, n=4096, k=4096)

    assert parsed["avg_time_us"] == 1340.0
    assert parsed["tflops"] == 102.38


def test_filter_implementations_selects_rtx4080_tk_and_cublas_pair():
    implementations = (
        Implementation("TK", "kernels/gemm/bf16_h100", ("H100",)),
        Implementation("TK Ampere", "kernels/gemm/bf16_ampere", ("RTX4080",)),
        Implementation("cuBLAS", "kernels/gemm/baselines/bf16_cublas", ("RTX4080", "H100")),
    )

    filtered = filter_implementations(
        implementations,
        gpu_target="RTX4080",
        implementation_names=("TK Ampere", "cuBLAS"),
    )

    assert [item.name for item in filtered] == ["TK Ampere", "cuBLAS"]


def test_filter_implementations_accepts_small_kernel_when_requested():
    implementations = (
        Implementation("TK Ampere", "kernels/gemm/bf16_ampere", ("RTX4080",)),
        Implementation("TK Ampere Small", "kernels/gemm/bf16_ampere_small", ("RTX4080",)),
        Implementation("cuBLAS", "kernels/gemm/baselines/bf16_cublas", ("RTX4080",)),
    )

    filtered = filter_implementations(
        implementations,
        gpu_target="RTX4080",
        implementation_names=("TK Ampere Small", "cuBLAS"),
    )

    assert [item.name for item in filtered] == ["TK Ampere Small", "cuBLAS"]


def test_parse_shape_spec_accepts_compact_mnk_triplet():
    shape = parse_shape_spec("4096x8192x4096")

    assert shape == ShapeSpec(m=4096, n=8192, k=4096)


def test_rtx4080_default_shapes_include_smaller_cases():
    shapes = DEFAULT_SHAPES_BY_TARGET["RTX4080"]

    assert ShapeSpec(512, 512, 512) in shapes
    assert ShapeSpec(1024, 1024, 1024) in shapes
    assert ShapeSpec(2048, 2048, 2048) in shapes


def test_default_implementations_for_512_include_small_kernel():
    names = default_implementations_for_shape("RTX4080", ShapeSpec(512, 512, 512))

    assert names == ("TK Ampere Small", "TK Ampere", "cuBLAS")


def test_default_implementations_for_1024_keep_main_kernel_first():
    names = default_implementations_for_shape("RTX4080", ShapeSpec(1024, 1024, 1024))

    assert names == ("TK Ampere", "TK Ampere Small", "cuBLAS")


def test_rank_results_marks_relative_to_best_for_successful_runs():
    results = [
        {"implementation": "TK", "status": "ok", "avg_time_us": 200.0, "tflops": 700.0},
        {"implementation": "cuBLAS", "status": "ok", "avg_time_us": 250.0, "tflops": 560.0},
        {"implementation": "cuBLASLt", "status": "build_failed", "avg_time_us": None, "tflops": None},
    ]

    ranked = rank_results(results)

    assert ranked[0]["relative_to_best"] == 1.0
    assert ranked[1]["relative_to_best"] == 0.8
    assert ranked[2]["relative_to_best"] is None


def test_format_relative_to_best_renders_compact_percent_string():
    assert format_relative_to_best(1.0) == "100.0%"
    assert format_relative_to_best(0.8125) == "81.2%"
    assert format_relative_to_best(None) == "-"


def test_summarize_shape_results_lists_best_per_shape():
    results_by_shape = {
        ShapeSpec(4096, 4096, 4096): rank_results(
            [
                {"implementation": "TK Ampere", "status": "ok", "avg_time_us": 1460.0, "tflops": 94.0},
                {"implementation": "cuBLAS", "status": "ok", "avg_time_us": 1340.0, "tflops": 102.0},
            ]
        ),
        ShapeSpec(4096, 8192, 4096): rank_results(
            [
                {"implementation": "TK Ampere", "status": "ok", "avg_time_us": 2920.0, "tflops": 93.0},
                {"implementation": "cuBLAS", "status": "ok", "avg_time_us": 2800.0, "tflops": 97.0},
            ]
        ),
    }

    summary = summarize_shape_results(results_by_shape)

    assert "4096x4096x4096: cuBLAS at 102.00 TFLOP/s" in summary
    assert "4096x8192x4096: cuBLAS at 97.00 TFLOP/s" in summary


def test_report_writers_emit_markdown_and_json(tmp_path: Path):
    results = rank_results(
        [
            {"implementation": "TK", "status": "ok", "avg_time_us": 200.0, "tflops": 700.0},
            {"implementation": "cuBLAS", "status": "ok", "avg_time_us": 250.0, "tflops": 560.0},
            {"implementation": "cuBLASLt", "status": "unsupported", "avg_time_us": None, "tflops": None},
        ]
    )
    markdown_path = tmp_path / "report.md"
    json_path = tmp_path / "report.json"

    write_markdown_report(
        markdown_path,
        gpu_name="NVIDIA H100",
        generated_at="2026-04-03T12:00:00Z",
        problem_size=(4096, 4096, 4096),
        results=results,
    )
    write_json_report(
        json_path,
        gpu_name="NVIDIA H100",
        generated_at="2026-04-03T12:00:00Z",
        problem_size=(4096, 4096, 4096),
        results=results,
    )

    markdown = markdown_path.read_text()
    payload = json.loads(json_path.read_text())

    assert "| TK | ok | 200.0 | 700.00 | 100.0% |" in markdown
    assert "| cuBLASLt | unsupported | - | - | - |" in markdown
    assert payload["gpu_name"] == "NVIDIA H100"
    assert payload["problem_size"] == {"m": 4096, "n": 4096, "k": 4096}
    assert payload["results"][1]["implementation"] == "cuBLAS"
