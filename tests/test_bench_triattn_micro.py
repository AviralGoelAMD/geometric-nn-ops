# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The triangle attention benchmark only times a backend that passed its check.

Runs on CPU: the inputs, the fake backends and the fp64 reference all live
there and the timers are replaced, so no GPU is needed. Triton must still be
importable, because the benchmark imports ``triton.testing``.
"""

import importlib
import pathlib

import pytest

pytest.importorskip("triton")

from geometric_nn_ops_torch.triangle._reference.triangle_attention import (  # noqa: E402
    triangle_attention_torch,
)

BENCHMARKS = pathlib.Path(__file__).resolve().parents[1] / "benchmarks"

N_SEQ, N, H, D = 32, 16, 2, 8


@pytest.fixture
def bench(monkeypatch):
    monkeypatch.syspath_prepend(str(BENCHMARKS))
    module = importlib.import_module("bench_triattn_micro")
    monkeypatch.setattr(module, "DEVICE", "cpu")
    return module


@pytest.fixture
def timer_calls(bench, monkeypatch):
    """Replace both timers with one that records its calls."""
    calls = []

    def timer(fn, **kwargs):
        calls.append(fn)
        return 1.0

    monkeypatch.setattr(bench, "do_bench", timer)
    monkeypatch.setattr(bench, "do_bench_cudagraph", timer)
    return calls


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
def test_correct_backend_is_timed(bench, timer_calls, monkeypatch, graph):
    monkeypatch.setattr(bench, "triangle_attention", triangle_attention_torch)

    row = bench.run_case("fp32", H, D, N_SEQ, N, warmup=1, rep=1, graph=graph)

    assert row["error"] is None
    assert row["ms"] == 1.0
    assert len(timer_calls) == 1


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_non_finite_unchecked_row_fails_untimed(
    bench, timer_calls, monkeypatch, graph, bad
):
    """A non-finite value in a row the reference comparison skips still fails
    the case, and the case is not timed."""
    checked = set(bench.check_rows(N_SEQ, "cpu").tolist())
    unchecked = next(row for row in range(N_SEQ) if row not in checked)

    def broken(**kwargs):
        out = triangle_attention_torch(**kwargs)
        out[:, unchecked, 0, 0, 0] = bad
        return out

    monkeypatch.setattr(bench, "triangle_attention", broken)

    row = bench.run_case("fp32", H, D, N_SEQ, N, warmup=1, rep=1, graph=graph)

    assert "non-finite" in row["error"]
    assert row["ms"] is None
    assert timer_calls == []


def test_wrong_output_fails_untimed(bench, timer_calls, monkeypatch):
    """An output outside the oracle tolerance fails the case untimed."""

    def scaled(**kwargs):
        return 1.2 * triangle_attention_torch(**kwargs)

    monkeypatch.setattr(bench, "triangle_attention", scaled)

    row = bench.run_case("fp32", H, D, N_SEQ, N, warmup=1, rep=1, graph=False)

    assert row["error"].startswith("AssertionError")
    assert row["ms"] is None
    assert timer_calls == []
