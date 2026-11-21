# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

r"""Microbenchmark the forward of the public triangle_attention op.

Each case is first run once: the whole output must be finite, and a few full
rows of it must match the fp64 reference to the bounds the tests use
(``TRIANGLE_ATTENTION_TOL``). A case that fails is recorded as FAILED and not
timed. Full correctness lives in ``tests/test_triangle_attention.py``.

Timing is the ``do_bench`` median under ``inference_mode``, after a warmup and
with the inputs built once outside the timed call, so compilation is excluded.
By default it times the public call as a user makes it, host overhead
included: use that for what a caller gets. ``--graph`` instead replays a
captured CUDA/HIP graph of the call, which leaves only the GPU time: use it to
compare kernels on shapes small enough to be host-bound (``rowwise``).

Presets (B=1, random bias, ~5% of keys masked, fixed seeds):
    of3-pair      H=4 D=32, N in 256..2048, N_seq=N (every row of the pair)
    of3-template  H=4 D=16, N in 256..2048, N_seq=N
    rowwise       H=4 D=32, N in 32..4096, N_seq=1

TFLOP/s counts QK^T and PV only: 4 * B * N_seq * N^2 * H * D.

To compare two checkouts, run this script once per checkout with that checkout
first on PYTHONPATH and diff the CSVs (see ``benchmarks/README.md``).

Usage:
    python benchmarks/bench_triattn_micro.py --preset of3-pair \
        --dtypes bf16 fp32 --out results/triattn_micro.csv
"""

import argparse
from pathlib import Path

import torch
import triton
from _bench_common import (
    add_timing_arguments,
    describe_case,
    finish,
    fmt,
    git_commit,
    print_provenance,
    release_blas_workspaces,
    require_gpu,
)
from triton.testing import do_bench, do_bench_cudagraph

from geometric_nn_ops_torch.triangle import triangle_attention
from geometric_nn_ops_torch.triangle._reference.triangle_attention import (
    triangle_attention_torch,
)
from testutil import cast_tree
from testutil.metrics import TRIANGLE_ATTENTION_TOL, assert_matches_oracle, rel_fro
from testutil.precision import torch_matmul_precision
from testutil.triangle_attention import make_inputs, make_key_mask

CASE_KEYS = ("dtype", "heads", "dim", "n_seq", "size")

OF3_SIZES = [256, 512, 768, 1024, 1536, 2048]

# name -> (heads, head dims, sizes, N_seq). N_seq None means N_seq = N.
PRESETS = {
    "of3-pair": (4, [32], OF3_SIZES, None),
    "of3-template": (4, [16], OF3_SIZES, None),
    "rowwise": (4, [32], [32, 64, 128, 256, 512, 1024, 2048, 4096], 1),
}

# name -> (input dtype, torch fp32 matmul precision)
DTYPE_MODES = {
    "bf16": (torch.bfloat16, "ieee"),
    "fp16": (torch.float16, "ieee"),
    "fp32": (torch.float32, "ieee"),
    "tf32": (torch.float32, "tf32"),
}

# Rows of N_seq checked against the reference, evenly spaced, first and last
# included. Each row is a full attention over N keys, and the fp64 logits of
# the slice are rows * H * N^2 * 8 bytes: 1 GiB at N=2048.
CHECK_ROWS = 8

# Where the inputs live. A module constant so the CPU test can redirect it.
DEVICE = "cuda"


def check_rows(n_seq, device):
    """Indices of the rows of N_seq compared with the reference."""
    rows = torch.linspace(0, n_seq - 1, min(n_seq, CHECK_ROWS), device=device)
    return rows.round().long().unique()


def check_against_reference(out, tensors, mask, tol, label):
    """Check all of ``out`` is finite, then compare a few full rows of it with
    the fp64 reference.

    The finiteness check covers every row, since the comparison does not. The
    rows of N_seq are independent attentions that share the bias, so the
    reference on a slice of them is exact for those rows.
    """
    q = tensors["q"]
    assert out.shape == q.shape and out.dtype == q.dtype, (
        f"[{label}] output is {tuple(out.shape)} {out.dtype}, "
        f"expected {tuple(q.shape)} {q.dtype}"
    )
    assert torch.isfinite(out).all(), f"[{label}] output has non-finite values"
    rows = check_rows(out.shape[1], out.device)
    sliced = {
        key: val if key == "bias" else val.index_select(1, rows)
        for key, val in tensors.items()
    }
    oracle = triangle_attention_torch(
        **cast_tree(sliced, torch.float64, requires_grad=False),
        mask=mask.index_select(1, rows),
    )
    candidate = out.index_select(1, rows)
    assert_matches_oracle([("out", candidate, oracle)], tol, label=label)
    return rel_fro(candidate, oracle)


def bench_case(dtype_name, heads, dim, n_seq, n, warmup, rep, graph):
    dtype, precision = DTYPE_MODES[dtype_name]
    tensors = make_inputs(1, n_seq, n, heads, dim, DEVICE, dtype, seed=0)
    mask = make_key_mask(1, n_seq, n, DEVICE, seed=1, frac_keep=0.95)

    def call():
        return triangle_attention(**tensors, mask=mask)

    with torch_matmul_precision(precision), torch.inference_mode():
        error = check_against_reference(
            call(),
            tensors,
            mask,
            TRIANGLE_ATTENTION_TOL[dtype, precision],
            label=f"{dtype_name} H={heads} D={dim} N_seq={n_seq} N={n}",
        )
        if graph:
            ms = do_bench_cudagraph(call, rep=rep, return_mode="median")
        else:
            ms = do_bench(call, warmup=warmup, rep=rep, return_mode="median")
    flops = 4 * n_seq * n * n * heads * dim
    return {"ms": ms, "tflops": flops / (ms * 1e-3) / 1e12, "rel_fro": error}


def run_case(dtype, heads, dim, n_seq, size, *, warmup, rep, graph):
    """One row of results. A case that fails its check or raises is recorded
    with the reason in ``error`` and no measurements, rather than raising."""
    case = {"dtype": dtype, "heads": heads, "dim": dim, "n_seq": n_seq, "size": size}
    row = case | {"ms": None, "tflops": None, "rel_fro": None, "error": None}
    try:
        row |= bench_case(dtype, heads, dim, n_seq, size, warmup, rep, graph)
    except Exception as error:  # noqa: BLE001
        row["error"] = f"{type(error).__name__}: {error}".splitlines()[0]
    return row


def report(rows, graph):
    header = (
        f"{'dtype':<5} {'H':>3} {'D':>4} {'N_seq':>6} {'N':>5} "
        f"{'ms':>10} {'TFLOP/s':>10} {'rel_fro':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        if row["error"]:
            print(f"{describe_case(row, CASE_KEYS)}  FAILED {row['error']}")
            continue
        print(
            f"{row['dtype']:<5} {row['heads']:>3} {row['dim']:>4} "
            f"{row['n_seq']:>6} {row['size']:>5} {fmt(row['ms'])} "
            f"{fmt(row['tflops'], prec=2)} {row['rel_fro']:>10.2e}"
        )
    if not graph and any(row["n_seq"] == 1 for row in rows):
        print(
            "\n# N_seq=1: the small cases are host-bound (Python and launch "
            "overhead, a flat floor across N); rerun with --graph for GPU time."
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--preset", default="of3-pair", choices=list(PRESETS))
    parser.add_argument(
        "--sizes", type=int, nargs="+", help="N_res; default: the preset's"
    )
    parser.add_argument(
        "--dims", type=int, nargs="+", help="head dim D; default: the preset's"
    )
    parser.add_argument("--heads", type=int, help="H; default: the preset's")
    parser.add_argument(
        "--n-seq",
        type=int,
        help="rows N_seq; default: the preset's (N for of3-*, 1 for rowwise)",
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        default=["bf16", "fp32"],
        choices=list(DTYPE_MODES),
        help="fp32 is IEEE; tf32 is fp32 inputs with "
        "torch.backends.cuda.matmul.fp32_precision='tf32'",
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help="time a captured CUDA/HIP graph replay (GPU time only, no host "
        "overhead; --warmup is unused) instead of the eager call",
    )
    add_timing_arguments(parser)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    heads, dims, sizes, n_seq = PRESETS[args.preset]
    heads = args.heads or heads
    dims = args.dims or dims
    sizes = args.sizes or sizes
    n_seq = args.n_seq if args.n_seq is not None else n_seq

    # How the numbers were made, in the header and in every row: the compiler
    # and timing method change the result as much as the kernel does.
    runtime = torch.version.hip and f"hip-{torch.version.hip}"
    context = {
        "timing": "do_bench_cudagraph" if args.graph else "do_bench",
        "warmup_ms": None if args.graph else args.warmup,
        "rep_ms": args.rep,
        "gpu_runtime": runtime or f"cuda-{torch.version.cuda}",
        "triton": triton.__version__,
    }

    require_gpu()
    print_provenance(
        ",".join(args.dtypes),
        preset=args.preset,
        heads=heads,
        dims=",".join(map(str, dims)),
        fp32_precision=",".join(f"{d}:{DTYPE_MODES[d][1]}" for d in args.dtypes),
        **context,
    )

    rows = []
    for dtype_name in args.dtypes:
        for dim in dims:
            for n in sizes:
                row = run_case(
                    dtype_name,
                    heads,
                    dim,
                    n_seq or n,
                    n,
                    warmup=args.warmup,
                    rep=args.rep,
                    graph=args.graph,
                )
                rows.append(
                    {"preset": args.preset}
                    | row
                    | {"fp32_precision": DTYPE_MODES[dtype_name][1]}
                    | context
                    | {
                        "torch": torch.__version__,
                        "device": torch.cuda.get_device_name(0),
                        "commit": git_commit(),
                    }
                )
                release_blas_workspaces()

    report(rows, args.graph)
    finish(rows, args.out, case_keys=CASE_KEYS)


if __name__ == "__main__":
    main()
