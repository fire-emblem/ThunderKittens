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


TARGETS = [
    Target(
        name="cute_shape_aware_best_1664x1024x16384_bf16",
        dtype="bf16",
        src="cute_tk_runtime_gemm.cu",
        out_name="cute_shape_aware_best_1664x1024x16384_bf16_cmp.out",
        extra_flags="",
        env={
            "TK_CUTE_USE_SHAPE_AWARE": "1",
            "TK_CUTE_M": "1664",
            "TK_CUTE_N": "1024",
            "TK_CUTE_K": "16384",
            "TK_CUTE_WARMUP": "1",
            "TK_CUTE_PROFILE": "3",
        },
    ),
    Target(
        name="cute_tn_conservative_1664x1024x16384_bf16",
        dtype="bf16",
        src="cute_tk_runtime_gemm.cu",
        out_name="cute_tn_conservative_1664x1024x16384_bf16_cmp.out",
        extra_flags="",
        env={
            "TK_CUTE_USE_TN_CONSERVATIVE": "1",
            "TK_CUTE_M": "1664",
            "TK_CUTE_N": "1024",
            "TK_CUTE_K": "16384",
            "TK_CUTE_WARMUP": "1",
            "TK_CUTE_PROFILE": "3",
        },
    ),
    Target(
        name="cute_tn_example_1664x1024x16384_bf16",
        dtype="bf16",
        src="cute_tk_runtime_gemm.cu",
        out_name="cute_tn_example_1664x1024x16384_bf16_cmp.out",
        extra_flags="",
        env={
            "TK_CUTE_USE_TN_EXAMPLE": "1",
            "TK_CUTE_M": "1664",
            "TK_CUTE_N": "1024",
            "TK_CUTE_K": "16384",
            "TK_CUTE_WARMUP": "1",
            "TK_CUTE_PROFILE": "3",
        },
    ),
    Target(
        name="cute_tn_linear_geom_1664x1024x16384_bf16",
        dtype="bf16",
        src="cute_tk_runtime_gemm.cu",
        out_name="cute_tn_linear_geom_1664x1024x16384_bf16_cmp.out",
        extra_flags="",
        env={
            "TK_CUTE_USE_TN_LINEAR_GEOMETRY": "1",
            "TK_CUTE_M": "1664",
            "TK_CUTE_N": "1024",
            "TK_CUTE_K": "16384",
            "TK_CUTE_WARMUP": "1",
            "TK_CUTE_PROFILE": "3",
        },
    ),
    Target(
        name="cute_layoutc_tuned_1664x1024x16384_bf16",
        dtype="bf16",
        src="cute_tk_runtime_gemm.cu",
        out_name="cute_layoutc_tuned_1664x1024x16384_bf16_cmp.out",
        extra_flags="",
        env={
            "TK_CUTE_USE_LAYOUTC_TN_TUNING": "1",
            "TK_CUTE_M": "1664",
            "TK_CUTE_N": "1024",
            "TK_CUTE_K": "16384",
            "TK_CUTE_WARMUP": "1",
            "TK_CUTE_PROFILE": "3",
        },
    ),
    Target(
        name="cute_layoutc_1664x1024x16384_bf16",
        dtype="bf16",
        src="cute_tk_runtime_gemm.cu",
        out_name="cute_layoutc_1664x1024x16384_bf16_cmp.out",
        extra_flags="",
        env={
            "TK_CUTE_M": "1664",
            "TK_CUTE_N": "1024",
            "TK_CUTE_K": "16384",
            "TK_CUTE_WARMUP": "1",
            "TK_CUTE_PROFILE": "3",
        },
    ),
    Target(
        name="cute_shape_aware_best_2048cube_bf16",
        dtype="bf16",
        src="cute_tk_runtime_gemm.cu",
        out_name="cute_shape_aware_best_2048cube_bf16_cmp.out",
        extra_flags="",
        env={
            "TK_CUTE_USE_SHAPE_AWARE": "1",
            "TK_CUTE_M": "2048",
            "TK_CUTE_N": "2048",
            "TK_CUTE_K": "2048",
            "TK_CUTE_WARMUP": "1",
            "TK_CUTE_PROFILE": "3",
        },
    ),
    Target(
        name="cute_tn_conservative_2048cube_bf16",
        dtype="bf16",
        src="cute_tk_runtime_gemm.cu",
        out_name="cute_tn_conservative_2048cube_bf16_cmp.out",
        extra_flags="",
        env={
            "TK_CUTE_USE_TN_CONSERVATIVE": "1",
            "TK_CUTE_M": "2048",
            "TK_CUTE_N": "2048",
            "TK_CUTE_K": "2048",
            "TK_CUTE_WARMUP": "1",
            "TK_CUTE_PROFILE": "3",
        },
    ),
    Target(
        name="cute_tn_example_2048cube_bf16",
        dtype="bf16",
        src="cute_tk_runtime_gemm.cu",
        out_name="cute_tn_example_2048cube_bf16_cmp.out",
        extra_flags="",
        env={
            "TK_CUTE_USE_TN_EXAMPLE": "1",
            "TK_CUTE_M": "2048",
            "TK_CUTE_N": "2048",
            "TK_CUTE_K": "2048",
            "TK_CUTE_WARMUP": "1",
            "TK_CUTE_PROFILE": "3",
        },
    ),
    Target(
        name="cute_tn_linear_geom_2048cube_bf16",
        dtype="bf16",
        src="cute_tk_runtime_gemm.cu",
        out_name="cute_tn_linear_geom_2048cube_bf16_cmp.out",
        extra_flags="",
        env={
            "TK_CUTE_USE_TN_LINEAR_GEOMETRY": "1",
            "TK_CUTE_M": "2048",
            "TK_CUTE_N": "2048",
            "TK_CUTE_K": "2048",
            "TK_CUTE_WARMUP": "1",
            "TK_CUTE_PROFILE": "3",
        },
    ),
    Target(
        name="cute_layoutc_tuned_2048cube_bf16",
        dtype="bf16",
        src="cute_tk_runtime_gemm.cu",
        out_name="cute_layoutc_tuned_2048cube_bf16_cmp.out",
        extra_flags="",
        env={
            "TK_CUTE_USE_LAYOUTC_TN_TUNING": "1",
            "TK_CUTE_M": "2048",
            "TK_CUTE_N": "2048",
            "TK_CUTE_K": "2048",
            "TK_CUTE_WARMUP": "1",
            "TK_CUTE_PROFILE": "3",
        },
    ),
    Target(
        name="cute_layoutc_2048cube_bf16",
        dtype="bf16",
        src="cute_tk_runtime_gemm.cu",
        out_name="cute_tk_runtime_2048cube_bf16_cmp.out",
        extra_flags="",
        env={
            "TK_CUTE_M": "2048",
            "TK_CUTE_N": "2048",
            "TK_CUTE_K": "2048",
            "TK_CUTE_WARMUP": "1",
            "TK_CUTE_PROFILE": "3",
        },
    ),
    Target(
        name="tk_local_layoutc_2048cube_bf16",
        dtype="bf16",
        src="bf16_c500_tk_local_gemm.cu",
        out_name="tk_local_layoutc_2048cube_bf16_cmp.out",
        extra_flags=(
            "-DBF16_C500_MUXI_NATIVE_M=2048 "
            "-DBF16_C500_MUXI_NATIVE_N=2048 "
            "-DBF16_C500_MUXI_NATIVE_K=2048 "
            "-DBF16_C500_MUXI_NATIVE_WARMUP_ITERS=1 "
            "-DBF16_C500_MUXI_NATIVE_PROFILE_ITERS=3"
        ),
    ),
    Target(
        name="cute_shape_aware_best_4096cube_bf16",
        dtype="bf16",
        src="cute_tk_runtime_gemm.cu",
        out_name="cute_shape_aware_best_4096cube_bf16_cmp.out",
        extra_flags="",
        env={
            "TK_CUTE_USE_SHAPE_AWARE": "1",
            "TK_CUTE_M": "4096",
            "TK_CUTE_N": "4096",
            "TK_CUTE_K": "4096",
            "TK_CUTE_WARMUP": "1",
            "TK_CUTE_PROFILE": "3",
        },
    ),
    Target(
        name="cute_tn_conservative_4096cube_bf16",
        dtype="bf16",
        src="cute_tk_runtime_gemm.cu",
        out_name="cute_tn_conservative_4096cube_bf16_cmp.out",
        extra_flags="",
        env={
            "TK_CUTE_USE_TN_CONSERVATIVE": "1",
            "TK_CUTE_M": "4096",
            "TK_CUTE_N": "4096",
            "TK_CUTE_K": "4096",
            "TK_CUTE_WARMUP": "1",
            "TK_CUTE_PROFILE": "3",
        },
    ),
    Target(
        name="cute_tn_example_4096cube_bf16",
        dtype="bf16",
        src="cute_tk_runtime_gemm.cu",
        out_name="cute_tn_example_4096cube_bf16_cmp.out",
        extra_flags="",
        env={
            "TK_CUTE_USE_TN_EXAMPLE": "1",
            "TK_CUTE_M": "4096",
            "TK_CUTE_N": "4096",
            "TK_CUTE_K": "4096",
            "TK_CUTE_WARMUP": "1",
            "TK_CUTE_PROFILE": "3",
        },
    ),
    Target(
        name="cute_tn_linear_geom_4096cube_bf16",
        dtype="bf16",
        src="cute_tk_runtime_gemm.cu",
        out_name="cute_tn_linear_geom_4096cube_bf16_cmp.out",
        extra_flags="",
        env={
            "TK_CUTE_USE_TN_LINEAR_GEOMETRY": "1",
            "TK_CUTE_M": "4096",
            "TK_CUTE_N": "4096",
            "TK_CUTE_K": "4096",
            "TK_CUTE_WARMUP": "1",
            "TK_CUTE_PROFILE": "3",
        },
    ),
    Target(
        name="cute_layoutc_tuned_4096cube_bf16",
        dtype="bf16",
        src="cute_tk_runtime_gemm.cu",
        out_name="cute_layoutc_tuned_4096cube_bf16_cmp.out",
        extra_flags="",
        env={
            "TK_CUTE_USE_LAYOUTC_TN_TUNING": "1",
            "TK_CUTE_M": "4096",
            "TK_CUTE_N": "4096",
            "TK_CUTE_K": "4096",
            "TK_CUTE_WARMUP": "1",
            "TK_CUTE_PROFILE": "3",
        },
    ),
    Target(
        name="cute_layoutc_4096cube_bf16",
        dtype="bf16",
        src="cute_tk_runtime_gemm.cu",
        out_name="cute_tk_runtime_4096cube_bf16_cmp.out",
        extra_flags="",
        env={
            "TK_CUTE_M": "4096",
            "TK_CUTE_N": "4096",
            "TK_CUTE_K": "4096",
            "TK_CUTE_WARMUP": "1",
            "TK_CUTE_PROFILE": "3",
        },
    ),
    Target(
        name="cute_shape_aware_best_8192cube_bf16",
        dtype="bf16",
        src="cute_tk_runtime_gemm.cu",
        out_name="cute_shape_aware_best_8192cube_bf16_cmp.out",
        extra_flags="",
        env={
            "TK_CUTE_USE_SHAPE_AWARE": "1",
            "TK_CUTE_M": "8192",
            "TK_CUTE_N": "8192",
            "TK_CUTE_K": "8192",
            "TK_CUTE_WARMUP": "1",
            "TK_CUTE_PROFILE": "3",
        },
    ),
    Target(
        name="cute_tn_example_8192cube_bf16",
        dtype="bf16",
        src="cute_tk_runtime_gemm.cu",
        out_name="cute_tn_example_8192cube_bf16_cmp.out",
        extra_flags="",
        env={
            "TK_CUTE_USE_TN_EXAMPLE": "1",
            "TK_CUTE_M": "8192",
            "TK_CUTE_N": "8192",
            "TK_CUTE_K": "8192",
            "TK_CUTE_WARMUP": "1",
            "TK_CUTE_PROFILE": "3",
        },
    ),
    Target(
        name="cute_layoutc_8192cube_bf16",
        dtype="bf16",
        src="cute_tk_runtime_gemm.cu",
        out_name="cute_tk_runtime_8192cube_bf16_cmp.out",
        extra_flags="",
        env={
            "TK_CUTE_M": "8192",
            "TK_CUTE_N": "8192",
            "TK_CUTE_K": "8192",
            "TK_CUTE_WARMUP": "1",
            "TK_CUTE_PROFILE": "3",
        },
    ),
    Target(
        name="tk_local_layoutc_4096cube_bf16",
        dtype="bf16",
        src="bf16_c500_tk_local_gemm.cu",
        out_name="tk_local_layoutc_4096cube_bf16_cmp.out",
        extra_flags=(
            "-DBF16_C500_MUXI_NATIVE_M=4096 "
            "-DBF16_C500_MUXI_NATIVE_N=4096 "
            "-DBF16_C500_MUXI_NATIVE_K=4096 "
            "-DBF16_C500_MUXI_NATIVE_WARMUP_ITERS=1 "
            "-DBF16_C500_MUXI_NATIVE_PROFILE_ITERS=3"
        ),
    ),
    Target(
        name="cute_reusea_n128",
        dtype="bf16",
        src="cute_tk_runtime_gemm.cu",
        out_name="cute_tk_runtime_n128_cmp.out",
        extra_flags="",
        env={
            "TK_CUTE_M": "4608",
            "TK_CUTE_N": "128",
            "TK_CUTE_K": "3584",
            "TK_CUTE_WARMUP": "1",
            "TK_CUTE_PROFILE": "3",
        },
    ),
    Target(
        name="cute_shape_aware_best_4608x128x3584_bf16",
        dtype="bf16",
        src="cute_tk_runtime_gemm.cu",
        out_name="cute_shape_aware_best_4608x128x3584_bf16_cmp.out",
        extra_flags="",
        env={
            "TK_CUTE_USE_SHAPE_AWARE": "1",
            "TK_CUTE_M": "4608",
            "TK_CUTE_N": "128",
            "TK_CUTE_K": "3584",
            "TK_CUTE_WARMUP": "1",
            "TK_CUTE_PROFILE": "3",
        },
    ),
    Target(
        name="cute_tn_example_4608x128x3584_bf16",
        dtype="bf16",
        src="cute_tk_runtime_gemm.cu",
        out_name="cute_tn_example_4608x128x3584_bf16_cmp.out",
        extra_flags="",
        env={
            "TK_CUTE_USE_TN_EXAMPLE": "1",
            "TK_CUTE_M": "4608",
            "TK_CUTE_N": "128",
            "TK_CUTE_K": "3584",
            "TK_CUTE_WARMUP": "1",
            "TK_CUTE_PROFILE": "3",
        },
    ),
    Target(
        name="tk_local_n128",
        dtype="bf16",
        src="tk_local_runtime_gemm.cu",
        out_name="tk_local_runtime_continuousc_n128_cmp.out",
        extra_flags="-DTK_LOCAL_USE_CONTINUOUSC",
        env={
            "TK_LOCAL_M": "4608",
            "TK_LOCAL_N": "128",
            "TK_LOCAL_K": "3584",
            "TK_LOCAL_WARMUP": "1",
            "TK_LOCAL_PROFILE": "3",
        },
    ),
    Target(
        name="cute_reusea_n256",
        dtype="bf16",
        src="cute_tk_runtime_gemm.cu",
        out_name="cute_tk_runtime_n256_cmp.out",
        extra_flags="",
        env={
            "TK_CUTE_M": "4608",
            "TK_CUTE_N": "256",
            "TK_CUTE_K": "3584",
            "TK_CUTE_WARMUP": "1",
            "TK_CUTE_PROFILE": "3",
        },
    ),
    Target(
        name="cute_shape_aware_best_3584x128x3584_bf16",
        dtype="bf16",
        src="cute_tk_runtime_gemm.cu",
        out_name="cute_shape_aware_best_3584x128x3584_bf16_cmp.out",
        extra_flags="",
        env={
            "TK_CUTE_USE_SHAPE_AWARE": "1",
            "TK_CUTE_M": "3584",
            "TK_CUTE_N": "128",
            "TK_CUTE_K": "3584",
            "TK_CUTE_WARMUP": "1",
            "TK_CUTE_PROFILE": "3",
        },
    ),
    Target(
        name="cute_tn_example_3584x128x3584_bf16",
        dtype="bf16",
        src="cute_tk_runtime_gemm.cu",
        out_name="cute_tn_example_3584x128x3584_bf16_cmp.out",
        extra_flags="",
        env={
            "TK_CUTE_USE_TN_EXAMPLE": "1",
            "TK_CUTE_M": "3584",
            "TK_CUTE_N": "128",
            "TK_CUTE_K": "3584",
            "TK_CUTE_WARMUP": "1",
            "TK_CUTE_PROFILE": "3",
        },
    ),
    Target(
        name="tk_local_3584x128x3584",
        dtype="bf16",
        src="tk_local_runtime_gemm.cu",
        out_name="tk_local_runtime_3584x128x3584_cmp.out",
        extra_flags="-DTK_LOCAL_USE_CONTINUOUSC",
        env={
            "TK_LOCAL_M": "3584",
            "TK_LOCAL_N": "128",
            "TK_LOCAL_K": "3584",
            "TK_LOCAL_WARMUP": "1",
            "TK_LOCAL_PROFILE": "3",
        },
    ),
    Target(
        name="cute_reusea_3584x128x3584",
        dtype="bf16",
        src="cute_tk_runtime_gemm.cu",
        out_name="cute_tk_runtime_3584x128x3584_cmp.out",
        extra_flags="",
        env={
            "TK_CUTE_M": "3584",
            "TK_CUTE_N": "128",
            "TK_CUTE_K": "3584",
            "TK_CUTE_WARMUP": "1",
            "TK_CUTE_PROFILE": "3",
        },
    ),
    Target(
        name="cute_shape_aware_best_4608x256x3584_bf16",
        dtype="bf16",
        src="cute_tk_runtime_gemm.cu",
        out_name="cute_shape_aware_best_4608x256x3584_bf16_cmp.out",
        extra_flags="",
        env={
            "TK_CUTE_USE_SHAPE_AWARE": "1",
            "TK_CUTE_M": "4608",
            "TK_CUTE_N": "256",
            "TK_CUTE_K": "3584",
            "TK_CUTE_WARMUP": "1",
            "TK_CUTE_PROFILE": "3",
        },
    ),
    Target(
        name="cute_tn_example_4608x256x3584_bf16",
        dtype="bf16",
        src="cute_tk_runtime_gemm.cu",
        out_name="cute_tn_example_4608x256x3584_bf16_cmp.out",
        extra_flags="",
        env={
            "TK_CUTE_USE_TN_EXAMPLE": "1",
            "TK_CUTE_M": "4608",
            "TK_CUTE_N": "256",
            "TK_CUTE_K": "3584",
            "TK_CUTE_WARMUP": "1",
            "TK_CUTE_PROFILE": "3",
        },
    ),
    Target(
        name="tk_local_n256",
        dtype="bf16",
        src="tk_local_runtime_gemm.cu",
        out_name="tk_local_runtime_continuousc_n256_cmp.out",
        extra_flags="-DTK_LOCAL_USE_CONTINUOUSC",
        env={
            "TK_LOCAL_M": "4608",
            "TK_LOCAL_N": "256",
            "TK_LOCAL_K": "3584",
            "TK_LOCAL_WARMUP": "1",
            "TK_LOCAL_PROFILE": "3",
        },
    ),
    Target(
        name="cute_shape_aware_best_37888x256x3584_bf16",
        dtype="bf16",
        src="cute_tk_runtime_gemm.cu",
        out_name="cute_shape_aware_best_37888x256x3584_bf16_cmp.out",
        extra_flags="",
        env={
            "TK_CUTE_USE_SHAPE_AWARE": "1",
            "TK_CUTE_M": "37888",
            "TK_CUTE_N": "256",
            "TK_CUTE_K": "3584",
            "TK_CUTE_WARMUP": "1",
            "TK_CUTE_PROFILE": "3",
        },
    ),
    Target(
        name="cute_tn_example_37888x256x3584_bf16",
        dtype="bf16",
        src="cute_tk_runtime_gemm.cu",
        out_name="cute_tn_example_37888x256x3584_bf16_cmp.out",
        extra_flags="",
        env={
            "TK_CUTE_USE_TN_EXAMPLE": "1",
            "TK_CUTE_M": "37888",
            "TK_CUTE_N": "256",
            "TK_CUTE_K": "3584",
            "TK_CUTE_WARMUP": "1",
            "TK_CUTE_PROFILE": "3",
        },
    ),
    Target(
        name="tk_local_37888x256x3584",
        dtype="bf16",
        src="tk_local_runtime_gemm.cu",
        out_name="tk_local_runtime_37888x256x3584_cmp.out",
        extra_flags="-DTK_LOCAL_USE_CONTINUOUSC",
        env={
            "TK_LOCAL_M": "37888",
            "TK_LOCAL_N": "256",
            "TK_LOCAL_K": "3584",
            "TK_LOCAL_WARMUP": "1",
            "TK_LOCAL_PROFILE": "3",
        },
    ),
    Target(
        name="cute_reusea_37888x256x3584",
        dtype="bf16",
        src="cute_tk_runtime_gemm.cu",
        out_name="cute_tk_runtime_37888x256x3584_cmp.out",
        extra_flags="",
        env={
            "TK_CUTE_M": "37888",
            "TK_CUTE_N": "256",
            "TK_CUTE_K": "3584",
            "TK_CUTE_WARMUP": "1",
            "TK_CUTE_PROFILE": "3",
        },
    ),
    Target(
        name="cute_shape_aware_best_3584x128x18944_bf16",
        dtype="bf16",
        src="cute_tk_runtime_gemm.cu",
        out_name="cute_shape_aware_best_3584x128x18944_bf16_cmp.out",
        extra_flags="",
        env={
            "TK_CUTE_USE_SHAPE_AWARE": "1",
            "TK_CUTE_M": "3584",
            "TK_CUTE_N": "128",
            "TK_CUTE_K": "18944",
            "TK_CUTE_WARMUP": "1",
            "TK_CUTE_PROFILE": "3",
        },
    ),
    Target(
        name="cute_tn_example_3584x128x18944_bf16",
        dtype="bf16",
        src="cute_tk_runtime_gemm.cu",
        out_name="cute_tn_example_3584x128x18944_bf16_cmp.out",
        extra_flags="",
        env={
            "TK_CUTE_USE_TN_EXAMPLE": "1",
            "TK_CUTE_M": "3584",
            "TK_CUTE_N": "128",
            "TK_CUTE_K": "18944",
            "TK_CUTE_WARMUP": "1",
            "TK_CUTE_PROFILE": "3",
        },
    ),
    Target(
        name="tk_local_3584x128x18944",
        dtype="bf16",
        src="tk_local_runtime_gemm.cu",
        out_name="tk_local_runtime_3584x128x18944_cmp.out",
        extra_flags="-DTK_LOCAL_USE_CONTINUOUSC",
        env={
            "TK_LOCAL_M": "3584",
            "TK_LOCAL_N": "128",
            "TK_LOCAL_K": "18944",
            "TK_LOCAL_WARMUP": "1",
            "TK_LOCAL_PROFILE": "3",
        },
    ),
    Target(
        name="cute_reusea_3584x128x18944",
        dtype="bf16",
        src="cute_tk_runtime_gemm.cu",
        out_name="cute_tk_runtime_3584x128x18944_cmp.out",
        extra_flags="",
        env={
            "TK_CUTE_M": "3584",
            "TK_CUTE_N": "128",
            "TK_CUTE_K": "18944",
            "TK_CUTE_WARMUP": "1",
            "TK_CUTE_PROFILE": "3",
        },
    ),
    Target(
        name="cute_shape_aware_best_37888x128x3584_bf16",
        dtype="bf16",
        src="cute_tk_runtime_gemm.cu",
        out_name="cute_shape_aware_best_37888x128x3584_bf16_cmp.out",
        extra_flags="",
        env={
            "TK_CUTE_USE_SHAPE_AWARE": "1",
            "TK_CUTE_M": "37888",
            "TK_CUTE_N": "128",
            "TK_CUTE_K": "3584",
            "TK_CUTE_WARMUP": "1",
            "TK_CUTE_PROFILE": "3",
        },
    ),
    Target(
        name="cute_tn_example_37888x128x3584_bf16",
        dtype="bf16",
        src="cute_tk_runtime_gemm.cu",
        out_name="cute_tn_example_37888x128x3584_bf16_cmp.out",
        extra_flags="",
        env={
            "TK_CUTE_USE_TN_EXAMPLE": "1",
            "TK_CUTE_M": "37888",
            "TK_CUTE_N": "128",
            "TK_CUTE_K": "3584",
            "TK_CUTE_WARMUP": "1",
            "TK_CUTE_PROFILE": "3",
        },
    ),
    Target(
        name="tk_local_37888x128x3584_bf16",
        dtype="bf16",
        src="tk_local_runtime_gemm.cu",
        out_name="tk_local_runtime_37888x128x3584_bf16_cmp.out",
        extra_flags="-DTK_LOCAL_USE_CONTINUOUSC",
        env={
            "TK_LOCAL_M": "37888",
            "TK_LOCAL_N": "128",
            "TK_LOCAL_K": "3584",
            "TK_LOCAL_WARMUP": "1",
            "TK_LOCAL_PROFILE": "3",
        },
    ),
    Target(
        name="cute_continuousc_37888x128x3584_bf16",
        dtype="bf16",
        src="cute_tk_runtime_gemm.cu",
        out_name="cute_tk_runtime_37888x128x3584_bf16_cmp.out",
        extra_flags="",
        env={
            "TK_CUTE_M": "37888",
            "TK_CUTE_N": "128",
            "TK_CUTE_K": "3584",
            "TK_CUTE_WARMUP": "1",
            "TK_CUTE_PROFILE": "3",
        },
    ),
    Target(
        name="tk_local_37888x128x3584_fp16",
        dtype="fp16",
        src="tk_local_runtime_gemm.cu",
        out_name="tk_local_runtime_37888x128x3584_fp16_cmp.out",
        extra_flags="-DTK_LOCAL_USE_CONTINUOUSC -DTK_LOCAL_USE_FP16",
        env={
            "TK_LOCAL_M": "37888",
            "TK_LOCAL_N": "128",
            "TK_LOCAL_K": "3584",
            "TK_LOCAL_WARMUP": "1",
            "TK_LOCAL_PROFILE": "3",
        },
    ),
    Target(
        name="cute_continuousc_37888x128x3584_fp16",
        dtype="fp16",
        src="cute_tk_runtime_gemm.cu",
        out_name="cute_tk_runtime_37888x128x3584_fp16_cmp.out",
        extra_flags="-DTK_CUTE_USE_FP16",
        env={
            "TK_CUTE_M": "37888",
            "TK_CUTE_N": "128",
            "TK_CUTE_K": "3584",
            "TK_CUTE_WARMUP": "1",
            "TK_CUTE_PROFILE": "3",
        },
    ),
]

MCBLAS_DIRECT_TARGETS = {
    ("bf16", 1664, 1024, 16384): {
        "src": "../baselines/bf16_mcblas/bf16_mcblas_gemm.cu",
        "out_name": "mcblas_1664x1024x16384_bf16_cmp.out",
        "extra_flags": (
            "-lmcblas "
            "-DBF16_MCBLAS_PROBLEM_M=1664 "
            "-DBF16_MCBLAS_PROBLEM_N=1024 "
            "-DBF16_MCBLAS_PROBLEM_K=16384 "
            "-DBF16_MCBLAS_WARMUP_ITERS=5 "
            "-DBF16_MCBLAS_PROFILE_ITERS=20"
        ),
    },
    ("bf16", 2048, 2048, 2048): {
        "src": "../baselines/bf16_mcblas/bf16_mcblas_gemm.cu",
        "out_name": "mcblas_2048cube_bf16_cmp.out",
        "extra_flags": (
            "-lmcblas "
            "-DBF16_MCBLAS_PROBLEM_M=2048 "
            "-DBF16_MCBLAS_PROBLEM_N=2048 "
            "-DBF16_MCBLAS_PROBLEM_K=2048 "
            "-DBF16_MCBLAS_WARMUP_ITERS=5 "
            "-DBF16_MCBLAS_PROFILE_ITERS=20"
        ),
    },
    ("bf16", 8192, 8192, 8192): {
        "src": "../baselines/bf16_mcblas/bf16_mcblas_gemm.cu",
        "out_name": "mcblas_8192cube_bf16_cmp.out",
        "extra_flags": (
            "-lmcblas "
            "-DBF16_MCBLAS_PROBLEM_M=8192 "
            "-DBF16_MCBLAS_PROBLEM_N=8192 "
            "-DBF16_MCBLAS_PROBLEM_K=8192 "
            "-DBF16_MCBLAS_WARMUP_ITERS=5 "
            "-DBF16_MCBLAS_PROFILE_ITERS=20"
        ),
    },
    ("bf16", 4096, 4096, 4096): {
        "src": "../baselines/bf16_mcblas/bf16_mcblas_gemm.cu",
        "out_name": "mcblas_4096cube_bf16_cmp.out",
        "extra_flags": (
            "-lmcblas "
            "-DBF16_MCBLAS_PROBLEM_M=4096 "
            "-DBF16_MCBLAS_PROBLEM_N=4096 "
            "-DBF16_MCBLAS_PROBLEM_K=4096 "
            "-DBF16_MCBLAS_WARMUP_ITERS=5 "
            "-DBF16_MCBLAS_PROFILE_ITERS=20"
        ),
    },
}


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
