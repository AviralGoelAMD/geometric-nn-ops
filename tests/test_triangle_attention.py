# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for triangle_attention."""

# The kernel imports run after `importorskip` so the module skips cleanly when
# Triton is missing; E402 (import not at top of file) is disabled accordingly.
# ruff: noqa: E402

import functools
import math

import pytest
import torch

triton = pytest.importorskip("triton")

from geometric_nn_ops_torch.triangle import triangle_attention
from geometric_nn_ops_torch.triangle._kernels.triangle_attention import (
    triangle_attention_triton,
)
from geometric_nn_ops_torch.triangle._masking import masked_key_bias
from geometric_nn_ops_torch.triangle._reference.triangle_attention import (
    triangle_attention_torch,
)
from testutil import cast_tree, generator, masked_noise, seeded_randn
from testutil.metrics import (
    Tolerance,
    assert_matches_oracle,
)
from testutil.precision import torch_matmul_precision
from testutil.triangle_attention import (
    assert_agrees_with_the_oracle,
    comparisons,
    fwd_bwd,
    make_inputs,
    make_key_mask,
)
from utils import (
    DTYPE_PRECISION_PARAM,
    DTYPES_PARAM,
    gpu_arch,
    requires_gpu,
    skip_without_reduced_precision_dot,
)

SHAPES = [
    pytest.param((1, 1, 32, 4, 32), id="pairformer_32"),
    pytest.param((1, 4, 32, 8, 8), id="msa_d8"),
    pytest.param((1, 1, 64, 16, 48), id="atom_d48"),
    pytest.param((2, 2, 48, 4, 32), id="batched"),
]

# There is no torch fallback: the public entrypoint raises when Triton is
# unavailable, so it is the Triton path. The reference takes the canonical 5D
# layout only, and is the oracle rather than something the op can run.
IMPLEMENTATIONS = {
    "reference": triangle_attention_torch,
    "triton": triangle_attention_triton,
}
IMPLEMENTATION_PARAM = pytest.mark.parametrize(
    "impl", list(IMPLEMENTATIONS.values()), ids=list(IMPLEMENTATIONS.keys())
)


# Structural tests: These use contrived scenarios where we know the answer
# analytically. We test both the Triton and Torch reference implementation,
# except where the case is the adapter in the public entrypoint, which the
# reference does not have.


def uniform_attention_inputs(b, n_seq, n_res, h, d, dtype, *, seed=0):
    """Zero q, so every logit is equal and softmax is exactly uniform.

    ``n_res`` is a power of two, making each probability exactly ``1/n_res``,
    and the values are small integers so the sums stay inside a bf16 mantissa.
    """
    g = generator("cuda", seed)
    return {
        "q": torch.zeros(b, n_seq, n_res, h, d, device="cuda", dtype=dtype),
        "k": torch.zeros(b, n_seq, n_res, h, d, device="cuda", dtype=dtype),
        "v": torch.randint(
            -4, 5, (b, n_seq, n_res, h, d), device="cuda", dtype=dtype, generator=g
        ),
        "bias": torch.zeros(b, 1, h, n_res, n_res, device="cuda", dtype=dtype),
    }


@requires_gpu
@IMPLEMENTATION_PARAM
@DTYPES_PARAM
@pytest.mark.parametrize("kept_key", [0, 7, 31], ids=["first", "interior", "last"])
def test_one_unmasked_key_returns_that_key(impl, dtype, kept_key):
    """One unmasked key saturates the softmax: its probability is exactly 1 and
    every other exactly 0, so the output is exactly ``v`` at that key."""
    B, N_seq, N_res, H, D = 1, 2, 32, 4, 32
    tensors = make_inputs(B, N_seq, N_res, H, D, "cuda", dtype, seed=3)
    mask = torch.zeros(B, N_seq, 1, 1, N_res, device="cuda", dtype=torch.bool)
    mask[..., kept_key] = True

    with torch.no_grad():
        out = impl(**tensors, mask=mask, scale=D**-0.5)

    expected = tensors["v"][:, :, kept_key : kept_key + 1, :, :].expand_as(out)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@requires_gpu
@IMPLEMENTATION_PARAM
@DTYPES_PARAM
def test_uniform_logits(impl, dtype):
    """Uniform probabilities over an exactly representable mean.

    The reference is exact in every dtype. The kernel is exact in fp32 and lands
    within one fp32 epsilon of the mean in bf16 and fp16.
    """
    # TODO(gmnngeoffrey): figure out why there's a residual error in bf16/fp16
    # even with exactly representable (small int) inputs.
    B, N_seq, N_res, H, D = 1, 1, 32, 4, 16
    tensors = uniform_attention_inputs(B, N_seq, N_res, H, D, dtype, seed=5)

    with torch.no_grad():
        out = impl(**tensors, mask=None, scale=D**-0.5)

    expected = tensors["v"].double().mean(dim=2, keepdim=True).expand_as(out)
    accumulator_ulp = torch.finfo(torch.float32).eps * expected.abs().max().item()
    torch.testing.assert_close(out.double(), expected, rtol=0, atol=accumulator_ulp)


@requires_gpu
@IMPLEMENTATION_PARAM
def test_mask_none(impl):
    """``mask=None`` matches an all-keep mask."""
    B, N_seq, N_res, H, D = 1, 2, 32, 4, 32
    tensors = make_inputs(B, N_seq, N_res, H, D, "cuda", torch.bfloat16, seed=13)
    all_keep = torch.ones(B, N_seq, 1, 1, N_res, device="cuda", dtype=torch.bool)

    with torch.no_grad():
        out = impl(**tensors, mask=None, scale=D**-0.5)
        expected = impl(**tensors, mask=all_keep, scale=D**-0.5)

    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@requires_gpu
@IMPLEMENTATION_PARAM
@DTYPES_PARAM
def test_masked_keys_are_ignored(request, impl, dtype):
    """A masked key's ``v`` reaches the output only through a zero softmax
    weight, so it should leave the output bit-identical however large it is."""

    B, N_seq, N_res, H, D = 1, 1, 32, 4, 32
    tensors = make_inputs(B, N_seq, N_res, H, D, "cuda", dtype, seed=3)
    mask = make_key_mask(B, N_seq, N_res, "cuda", seed=5)
    perturbed = {
        **tensors,
        "v": masked_noise(tensors["v"], ~mask.reshape(B, N_seq, N_res, 1, 1), 9),
    }

    with torch.no_grad():
        out = impl(**tensors, mask=mask, scale=D**-0.5)
        out_perturbed = impl(**perturbed, mask=mask, scale=D**-0.5)
        unmasked = impl(**tensors, mask=None, scale=D**-0.5)
        unmasked_perturbed = impl(**perturbed, mask=None, scale=D**-0.5)

    # fp16 Triton on gfx1151 is off by one ULP. Do a close test first and then
    # xfail that combination before doing the exact test.
    # TODO(gmngeoffrey): root cause this issue
    torch.testing.assert_close(out, out_perturbed, rtol=0, atol=1e-6)
    assert not torch.allclose(unmasked, unmasked_perturbed), (
        "the perturbation does not change the result even unmasked"
    )

    if (
        impl is triangle_attention_triton
        and dtype is torch.float16
        and gpu_arch() == "gfx1151"
    ):
        request.node.add_marker(
            pytest.mark.xfail(
                strict=True,
                reason="gfx1151 fp16: masked v perturbs the output by one ulp",
            )
        )

    torch.testing.assert_close(out, out_perturbed, rtol=0, atol=0)


@requires_gpu
@IMPLEMENTATION_PARAM
@DTYPES_PARAM
def test_masked_logits_are_suppressed(impl, dtype):
    """``k`` and the bias at masked keys move those logits, but masked logits
    are replaced with a constant, so whatever they were cannot reach the
    output.
    """
    B, N_seq, N_res, H, D = 1, 1, 32, 4, 32
    tensors = make_inputs(B, N_seq, N_res, H, D, "cuda", dtype, seed=3)
    mask = make_key_mask(B, N_seq, N_res, "cuda", seed=5)
    perturbed = {
        **tensors,
        "k": masked_noise(tensors["k"], ~mask.reshape(B, N_seq, N_res, 1, 1), 7),
        "bias": masked_noise(tensors["bias"], ~mask, 11),
    }

    with torch.no_grad():
        out = impl(**tensors, mask=mask, scale=D**-0.5)
        out_perturbed = impl(**perturbed, mask=mask, scale=D**-0.5)
        unmasked = impl(**tensors, mask=None, scale=D**-0.5)
        unmasked_perturbed = impl(**perturbed, mask=None, scale=D**-0.5)

    torch.testing.assert_close(out, out_perturbed, rtol=0, atol=0)
    assert not torch.allclose(unmasked, unmasked_perturbed), (
        "the perturbation does not change the result even unmasked"
    )


@requires_gpu
@IMPLEMENTATION_PARAM
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("kept_logit", [-2e30, -1e38, -3e38], ids=str)
@pytest.mark.parametrize("use_exp2", [False, True], ids=["exp", "exp2"])
def test_masked_keys_lose_to_a_very_negative_kept_logit(
    request, impl, dtype, kept_logit, use_exp2
):
    """Masked keys get zero weight however low the one kept logit is and however
    high the masked ones are (``+2e9`` and ``+1e30`` via the bias): the output
    is exactly the kept key's ``v``, and masked keys get no ``dk``, ``dv`` or
    ``dbias``.

    The bound is the kept logit staying above ``finfo(float32).min``, the value
    masked logits are replaced with. With ``use_exp2`` the kernel scales logits
    by ``log2(e)`` first, so the bound is ``finfo(float32).min / log2(e)``
    (about ``-2.36e38``): below it the kept logit overflows to ``-inf`` and the
    masked keys win.
    """
    if use_exp2:
        if impl is triangle_attention_torch:
            pytest.skip("the reference has no base-2 path")
        impl = functools.partial(impl, use_exp2=True)
        if kept_logit < torch.finfo(torch.float32).min / math.log2(math.e):
            request.node.add_marker(
                pytest.mark.xfail(
                    strict=True,
                    reason="kept logit * log2(e) overflows fp32 to -inf",
                )
            )
    B, N_seq, N_res, H, D = 1, 1, 32, 4, 32
    kept = 7
    tensors = make_inputs(B, N_seq, N_res, H, D, "cuda", dtype, seed=3)
    mask = torch.zeros(B, N_seq, 1, 1, N_res, device="cuda", dtype=torch.bool)
    mask[..., kept] = True
    bias = torch.empty(B, 1, H, N_res, N_res, device="cuda", dtype=torch.float32)
    bias[..., 0::2] = 2e9
    bias[..., 1::2] = 1e30
    bias[..., kept] = kept_logit
    tensors["bias"] = bias.to(dtype)
    leaves = {k: t.detach().clone().requires_grad_(True) for k, t in tensors.items()}

    out = impl(**leaves, mask=mask, scale=D**-0.5)
    out.backward(torch.ones_like(out))

    expected = tensors["v"][:, :, kept : kept + 1].expand_as(out)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    for name, leaf in leaves.items():
        assert torch.isfinite(leaf.grad).all(), f"non-finite grad_{name}"
    masked = ~mask.reshape(N_res)
    assert (leaves["k"].grad[:, :, masked] == 0).all(), "masked key got dk"
    assert (leaves["v"].grad[:, :, masked] == 0).all(), "masked key got dv"
    assert (leaves["bias"].grad[..., masked] == 0).all(), "masked key got dbias"


@requires_gpu
@IMPLEMENTATION_PARAM
def test_no_bias(impl):
    """Omitting the bias matches an explicit zero bias."""
    B, N_seq, N_res, H, D = 1, 2, 32, 4, 32
    tensors = make_inputs(B, N_seq, N_res, H, D, "cuda", torch.bfloat16, seed=15)
    without = {k: v for k, v in tensors.items() if k != "bias"}
    zeroed = {**tensors, "bias": torch.zeros_like(tensors["bias"])}

    with torch.no_grad():
        out = impl(**without, mask=None, scale=D**-0.5)
        expected = impl(**zeroed, mask=None, scale=D**-0.5)

    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@requires_gpu
@IMPLEMENTATION_PARAM
@DTYPES_PARAM
@pytest.mark.parametrize("logits", ["small", "large"])
@pytest.mark.parametrize("use_exp2", [False, True], ids=["exp", "exp2"])
def test_fully_masked_row_is_uniform(impl, dtype, logits, use_exp2):
    """With every key masked the softmax is uniform, as in AlphaFold3: the
    output is the mean of ``v``, every key gets ``dv = sum_i dO[i] / N_res``,
    and ``q``, ``k`` and the bias get no gradient.

    Masked logits are replaced, not offset, so every logit in the row is the
    same constant however large ``q . k`` and the bias were. ``"large"`` puts
    them around ``1e4``, which an additive ``-1e9`` (fp32 ulp 64) would have
    left distinct. ``v`` and ``dO`` hold small integers over a power-of-two
    ``N_res`` so the means are exactly representable.

    Softmax over no keys is undefined; uniform is the convention, not a
    mathematical truth.
    """
    if use_exp2:
        if impl is triangle_attention_torch:
            pytest.skip("the reference has no base-2 path")
        impl = functools.partial(impl, use_exp2=True)
    B, N_seq, N_res, H, D = 1, 1, 32, 4, 16
    shape = (B, N_seq, N_res, H, D)
    values = torch.randint(-4, 5, shape, device="cuda", generator=generator("cuda", 5))
    grad_out = torch.randint(
        -4, 5, shape, device="cuda", generator=generator("cuda", 6)
    )
    q = torch.randn(shape, device="cuda", generator=generator("cuda", 7))
    bias = torch.zeros(B, 1, H, N_res, N_res, device="cuda")
    if logits == "large":
        q = q * 100
        bias = 1e4 * torch.randn(
            bias.shape, device="cuda", generator=generator("cuda", 9)
        )
    tensors = {
        "q": q.to(dtype),
        "k": torch.randn(shape, device="cuda", generator=generator("cuda", 8)).to(
            dtype
        ),
        "v": values.to(dtype),
        "bias": bias.to(dtype),
    }
    leaves = {k: t.detach().clone().requires_grad_(True) for k, t in tensors.items()}
    mask = torch.zeros(B, N_seq, 1, 1, N_res, device="cuda", dtype=torch.bool)

    out = impl(**leaves, mask=mask, scale=D**-0.5)
    out.backward(grad_out.to(dtype))

    expected = values.double().mean(dim=2, keepdim=True).expand_as(out)
    accumulator_ulp = torch.finfo(torch.float32).eps * expected.abs().max().item()
    torch.testing.assert_close(out.double(), expected, rtol=0, atol=accumulator_ulp)
    expected_dv = grad_out.double().mean(dim=2, keepdim=True).expand_as(out)
    accumulator_ulp = torch.finfo(torch.float32).eps * expected_dv.abs().max().item()
    torch.testing.assert_close(
        leaves["v"].grad.double(), expected_dv, rtol=0, atol=accumulator_ulp
    )
    for name in ("q", "k", "bias"):
        assert (leaves[name].grad == 0).all(), f"fully masked row gave d{name}"


@requires_gpu
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.uint8], ids=["fp32", "bf16", "uint8"]
)
def test_non_boolean_mask_is_rejected(dtype):
    B, N_seq, N_res, H, D = 1, 1, 32, 4, 32
    tensors = make_inputs(B, N_seq, N_res, H, D, "cuda", torch.bfloat16, seed=11)
    boolean = make_key_mask(B, N_seq, N_res, "cuda", seed=4)

    with pytest.raises(ValueError, match="mask must be bool"):
        triangle_attention(**tensors, mask=boolean.to(dtype), scale=D**-0.5)


@DTYPES_PARAM
def test_mask_marker_is_finite(dtype):
    """The kernels tell masked keys (finite negative ``masked_key_bias``) from
    padding keys past ``N_res`` (``-inf``), so the marker must not overflow."""
    marker = masked_key_bias(torch.tensor([True, False]), dtype).to(dtype)
    assert marker[0] == 0
    assert marker[1].isfinite() and marker[1] < 0


@requires_gpu
@IMPLEMENTATION_PARAM
def test_default_scale(impl):
    """``scale=None`` matches an explicit ``D ** -0.5``, and the argument has an
    effect at all."""
    B, N_seq, N_res, H, D = 1, 1, 32, 4, 32
    tensors = make_inputs(B, N_seq, N_res, H, D, "cuda", torch.float32, seed=13)

    with torch.no_grad():
        default = impl(**tensors, mask=None, scale=None)
        explicit = impl(**tensors, mask=None, scale=D**-0.5)
        other = impl(**tensors, mask=None, scale=D**-0.5 * 1.75)

    torch.testing.assert_close(default, explicit, rtol=0, atol=0)
    assert not torch.equal(default, other), "the scale argument had no effect"


@requires_gpu
def test_4d_input():
    """Ranks below 5 are padded onto the canonical layout and come back in the
    input shape. The reference takes 5D only, so this is Triton only."""
    N_seq, N_res, H, D = 2, 32, 4, 32
    unbatched = make_inputs((), N_seq, N_res, H, D, "cuda", torch.bfloat16, seed=7)
    batched = make_inputs(1, N_seq, N_res, H, D, "cuda", torch.bfloat16, seed=7)
    mask = make_key_mask(1, N_seq, N_res, "cuda", seed=9)

    with torch.no_grad():
        out_4d = triangle_attention(**unbatched, mask=mask, scale=D**-0.5)
        out_5d = triangle_attention(**batched, mask=mask, scale=D**-0.5)

    assert out_4d.shape == unbatched["q"].shape
    torch.testing.assert_close(out_4d, out_5d[0], rtol=0, atol=0)


@requires_gpu
def test_four_dimensional_bias():
    """A ``[B, H, N, N]`` bias gains the N_seq axis. Rank normalization, so it
    is the public op's job -- the functional entry takes shapes as given."""
    B, N_seq, N_res, H, D = 1, 2, 32, 4, 32
    tensors = make_inputs(B, N_seq, N_res, H, D, "cuda", torch.bfloat16, seed=5)
    without_seq = {**tensors, "bias": tensors["bias"][:, 0]}  # [B, H, N, N]

    with torch.no_grad():
        out_4d = triangle_attention(**without_seq, mask=None, scale=D**-0.5)
        out_5d = triangle_attention(**tensors, mask=None, scale=D**-0.5)

    torch.testing.assert_close(out_4d, out_5d, rtol=0, atol=0)


@requires_gpu
@pytest.mark.parametrize("rank", [5, 4, 3], ids=["canonical", "no_batch", "no_seq"])
def test_lower_rank_mask(rank):
    """A mask below the canonical rank broadcasts onto it.

    ``[B, N_seq, 1, 1, N_res]`` is the canonical layout, but the documented
    contract is any shape broadcastable to it. Notably, cuequivariance allows
    omitting the batch axis when ``B`` is 1.
    """
    B, N_seq, N_res, H, D = 1, 2, 32, 4, 32
    tensors = make_inputs(B, N_seq, N_res, H, D, "cuda", torch.bfloat16, seed=5)

    canonical = make_key_mask(B, N_seq, N_res, "cuda", seed=9)
    lower = canonical[(0,) * (5 - rank)]
    explicit = lower.expand_as(canonical).contiguous()
    assert lower.dim() == rank

    with torch.no_grad():
        out_lower = triangle_attention(**tensors, mask=lower, scale=D**-0.5)
        out_explicit = triangle_attention(**tensors, mask=explicit, scale=D**-0.5)

    torch.testing.assert_close(out_lower, out_explicit, rtol=0, atol=0)


@requires_gpu
def test_6d_input():
    """Ranks above 5 fold their leading dims into the batch axis."""
    B0, B1, N_seq, N_res, H, D = 2, 3, 1, 32, 4, 32
    nested = make_inputs((B0, B1), N_seq, N_res, H, D, "cuda", torch.bfloat16, seed=11)
    flat = make_inputs(B0 * B1, N_seq, N_res, H, D, "cuda", torch.bfloat16, seed=11)

    with torch.no_grad():
        out_6d = triangle_attention(**nested, mask=None, scale=D**-0.5)
        out_5d = triangle_attention(**flat, mask=None, scale=D**-0.5)

    assert out_6d.shape == nested["q"].shape
    torch.testing.assert_close(out_6d.reshape_as(out_5d), out_5d, rtol=0, atol=0)


@requires_gpu
def test_dynamic_entry_point():
    """The variable-shape entry trades a per-call cost for no per-size
    recompile. Its output is identical, not merely close."""
    B, N_seq, N_res, H, D = 1, 1, 32, 4, 32
    tensors = make_inputs(B, N_seq, N_res, H, D, "cuda", torch.bfloat16, seed=19)
    mask = make_key_mask(B, N_seq, N_res, "cuda", seed=21)

    with torch.no_grad():
        specialized = triangle_attention(
            **tensors, mask=mask, scale=D**-0.5, dynamic=False
        )
        dynamic = triangle_attention(**tensors, mask=mask, scale=D**-0.5, dynamic=True)

    torch.testing.assert_close(specialized, dynamic, rtol=0, atol=0)


# We don't test the reference here because it's torch's own behavior, and
# whether it honors the setting is inconsistent and not really our busines to
# test.


@requires_gpu
@pytest.mark.parametrize(
    "impl",
    [
        triangle_attention_triton,
        functools.partial(triangle_attention_triton, dynamic=True),
    ],
    ids=["triton", "triton_dynamic"],
)
def test_tf32_changes_the_result(impl):
    """Make sure the reduced precision actually affects the output."""
    skip_without_reduced_precision_dot()
    B, N_seq, N_res, H, D = 1, 1, 32, 4, 32
    mask = make_key_mask(B, N_seq, N_res, "cuda", seed=5)

    results = {}
    for precision in ("ieee", "tf32"):
        tensors = make_inputs(
            B, N_seq, N_res, H, D, "cuda", torch.float32, seed=3, requires_grad=True
        )
        grad_out = seeded_randn(
            *tensors["q"].shape, device="cuda", dtype=torch.float32, seed=31
        )
        with torch_matmul_precision(precision):
            results[precision] = fwd_bwd(impl, tensors, mask, D**-0.5, grad_out)

    out_ieee, keys, grads_ieee = results["ieee"]
    out_tf32, _, grads_tf32 = results["tf32"]
    for name, full, reduced in comparisons(
        out_ieee, keys, grads_ieee, out_tf32, grads_tf32
    ):
        assert not torch.allclose(full, reduced, rtol=0, atol=0), (
            f"{name} is bit-identical for ieee and tf32"
        )


# Reference tests: these compare the Triton kernel to the pure Torch reference
# in fp64.


@requires_gpu
@pytest.mark.parametrize("shape", SHAPES)
@DTYPE_PRECISION_PARAM
def test_triton_accuracy(shape, dtype, precision):
    """The kernel at ``dtype`` and ``precision`` against the reference in fp64."""
    B, N_seq, N_res, H, D = shape
    tensors = make_inputs(
        B, N_seq, N_res, H, D, "cuda", dtype, seed=3, requires_grad=True
    )
    mask = make_key_mask(B, N_seq, N_res, "cuda", seed=1)
    with torch_matmul_precision(precision):
        assert_agrees_with_the_oracle(
            tensors,
            mask,
            D**-0.5,
            impl=triangle_attention_triton,
            precision=precision,
            label=f"{shape}/{dtype}/{precision}",
        )


@requires_gpu
@pytest.mark.parametrize("shape", SHAPES)
@DTYPES_PARAM
def test_exp2_softmax(shape, dtype):
    """The base-2 softmax is an alternate evaluation of the same function, so it
    holds to the same tolerance as the default base-e path. The backward has to
    pick up the base the forward used, hence the gradient comparison."""
    B, N_seq, N_res, H, D = shape
    tensors = make_inputs(
        B, N_seq, N_res, H, D, "cuda", dtype, seed=3, requires_grad=True
    )
    mask = make_key_mask(B, N_seq, N_res, "cuda", seed=1)
    assert_agrees_with_the_oracle(
        tensors,
        mask,
        D**-0.5,
        impl=functools.partial(triangle_attention, use_exp2=True),
        label=f"exp2/{shape}/{dtype}",
    )


@requires_gpu
@DTYPES_PARAM
def test_fully_masked_row(dtype):
    """Fully masked row, gradients included, against the reference in fp64.

    Both replace masked logits with ``finfo.min`` of their logit dtype, so the
    row is uniform in every dtype (see ``test_fully_masked_row_is_uniform``).
    """
    B, N_seq, N_res, H, D = 1, 1, 64, 4, 32
    tensors = make_inputs(
        B, N_seq, N_res, H, D, "cuda", dtype, seed=3, requires_grad=True
    )
    mask = torch.zeros(B, N_seq, 1, 1, N_res, device="cuda", dtype=torch.bool)

    assert_agrees_with_the_oracle(
        tensors,
        mask,
        D**-0.5,
        impl=triangle_attention_triton,
        label=f"fully_masked/{dtype}",
    )


@requires_gpu
@pytest.mark.parametrize("case", ["no_bias", "no_mask", "custom_scale"])
def test_optional_arguments(case):
    """One case per branch through the adapter's optional arguments."""
    B, N_seq, N_res, H, D = 1, 2, 32, 4, 32
    dtype = torch.bfloat16
    tensors = make_inputs(
        B,
        N_seq,
        N_res,
        H,
        D,
        "cuda",
        dtype,
        seed=5,
        with_bias=case != "no_bias",
        requires_grad=True,
    )
    mask = None if case == "no_mask" else make_key_mask(B, N_seq, N_res, "cuda")
    scale = D**-0.5 * (1.75 if case == "custom_scale" else 1.0)

    assert_agrees_with_the_oracle(
        tensors, mask, scale, impl=triangle_attention_triton, label=case
    )


@requires_gpu
def test_pairformer_shape():
    """N=384, H=16, D=48 -- the pairformer operating point."""
    B, N_seq, N_res, H, D = 1, 1, 384, 16, 48
    dtype = torch.bfloat16
    tensors = make_inputs(B, N_seq, N_res, H, D, "cuda", dtype, seed=21)

    with torch.no_grad():
        out = triangle_attention(**tensors, scale=None)
        out_oracle = triangle_attention_torch(
            **cast_tree(tensors, torch.float64, requires_grad=False), scale=None
        )

    assert_matches_oracle(
        [("out", out, out_oracle)],
        Tolerance(1.0 - 1e-4, 5e-2),
        label="pairformer_384",
    )
