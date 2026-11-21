# Benchmarks

Run from the repository root with a GPU and the checkout on `PYTHONPATH` (the
scripts import `testutil`). Every script prints a provenance header (device,
torch, commit and path of the imported package), takes `--warmup`/`--rep` in
milliseconds, writes `--out` as CSV, records failed cases and exits non-zero
if any case failed.

## Triangle attention: `bench_triattn_micro.py`

Times the forward of the public `triangle_attention` (B=1, random bias, ~5% of
keys masked, fixed seeds). Before timing, each case runs once; the whole output
must have the input's shape and dtype and be finite, and up to 8 full rows of it
must match the fp64 reference to the bounds the tests use
(`TRIANGLE_ATTENTION_TOL`). A case that fails either is reported FAILED and not
timed. Output columns are the median ms, TFLOP/s (`4 * N_seq * N^2 * H * D`,
QK^T and PV only) and the relative Frobenius error of the checked rows. The
header and every CSV row also record how the number was made: timer
(`do_bench` or `do_bench_cudagraph`), `warmup_ms`/`rep_ms`, fp32 matmul
precision, torch, HIP/CUDA runtime and Triton versions, device and commit.

| preset | H | D | N | N_seq |
|---|---|---|---|---|
| `of3-pair` (default) | 4 | 32 | 256, 512, 768, 1024, 1536, 2048 | N |
| `of3-template` | 4 | 16 | 256, 512, 768, 1024, 1536, 2048 | N |
| `rowwise` | 4 | 32 | 32, 64, ..., 4096 | 1 |

`of3-*` run every row of the pair, which is the model's workload. `rowwise` is
one row, so its small sizes measure the host, not the kernel. `--sizes`,
`--dims`, `--heads` and `--n-seq` override the preset. `--dtypes` takes `bf16`,
`fp16`, `fp32` (IEEE) and `tf32` (fp32 inputs with
`torch.backends.cuda.matmul.fp32_precision = "tf32"`); the default is
`bf16 fp32`.

```bash
python benchmarks/bench_triattn_micro.py --out results/triattn_pair.csv
python benchmarks/bench_triattn_micro.py --preset rowwise --dtypes bf16 fp16 fp32 tf32
python benchmarks/bench_triattn_micro.py --preset rowwise --graph
```

Without `--graph` the script times the eager call, Python and launch overhead
included: what a caller gets. With `--graph` it replays a captured CUDA/HIP
graph of the same call, which leaves only GPU time. Use `--graph` when the
eager times sit on a flat floor across sizes (the host cost, which depends on
the machine and the Python stack), since a kernel change can hide under it.
`do_bench` flushes the L2 cache between calls and graph replay does not, so
the two modes differ slightly even where the host cost is negligible.

### Comparing two checkouts

The script reports on whichever `geometric_nn_ops_torch` is imported, so run it
once per checkout with that checkout first on `PYTHONPATH`, on the same GPU,
and diff the CSVs:

```bash
PYTHONPATH=/path/to/base python benchmarks/bench_triattn_micro.py --out base.csv
PYTHONPATH=/path/to/head python benchmarks/bench_triattn_micro.py --out head.csv
```

Alternate the runs (base, head, base, head) and compare medians across them: a
single pair can differ by a few percent from clock and thermal drift alone.
