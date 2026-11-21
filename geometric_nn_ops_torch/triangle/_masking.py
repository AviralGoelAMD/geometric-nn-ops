# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Masking keys out of a softmax.

Two schemes live here:

- **Additive** (``mask_val``, ``masked_key_bias``): push a masked key's logit
  down by a large dtype-dependent amount. Used by attention with pair bias.
- **Replacement** (``MASKED_LOGIT``, ``replace_masked_logits``): set a masked
  key's final logit to ``finfo.min``, as AlphaFold3/tokamax do. Used by
  triangle attention.
"""

import torch

# Triangle attention replaces a masked key's logit with this constant after
# scale, pair bias and any base-2 ``* log2e`` scaling, instead of adding
# ``-mask_val``. Every masked logit in a row is then the same value whatever
# ``q``, ``k`` or the bias were, so a fully masked row is exactly uniform (output
# ``mean(v)``) in every dtype and masked entries carry no gradient. It is
# ``finfo.min`` of the logit dtype, like tokamax: the kernels compute logits in
# fp32 and replace the final tile, so the constant itself is never scaled.
# Masked keys lose to any kept logit above ``finfo(fp32).min``; on the base-2
# path the kept logit must stay above ``finfo(fp32).min / log2(e)`` (~-2.36e38),
# below which ``* log2e`` overflows it to ``-inf``.
MASKED_LOGIT = torch.finfo(torch.float32).min


def replace_masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked logits (``mask == False``) -> ``finfo(logits.dtype).min``.

    The torch counterpart of the kernels' ``MASKED_LOGIT`` replacement, in the
    logits' own dtype (``finfo(float64).min`` for the fp64 reference).
    """
    return torch.where(mask, logits, torch.finfo(logits.dtype).min)


def mask_val(dtype: torch.dtype) -> float:
    """How far a masked key's logit is pushed below zero, for ``dtype``.

    ``1e9`` for fp32 and bf16. fp16 cannot represent it and would get ``inf`` --
    a fully masked key *tile* then takes ``-inf`` as its running maximum and the
    online softmax evaluates ``-inf - -inf``, giving NaNs even for rows that are
    not fully masked -- so it takes half the dtype maximum instead, which still
    squashes any reasonable logit. fp64 needs a larger value than fp32 to behave
    the same way: at ``1e9`` its ulp is far below ``log(SEQ_LEN)``, so a fully
    masked row would keep its logsumexp term where fp32 loses it, and the two
    would disagree.
    """
    if dtype == torch.float64:
        return 1e18
    elif dtype == torch.float32 or dtype == torch.bfloat16:
        return 1e9
    elif dtype == torch.float16:
        return torch.finfo(dtype).max / 2
    else:
        raise ValueError(f"Unsupported dtype {dtype}")


def masked_key_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Boolean mask -> additive bias, ``0`` where kept and ``-mask_val`` where not.

    The triangle attention kernels read it only as a marker (``0`` kept, finite
    negative masked) and replace masked logits with ``MASKED_LOGIT``.
    """
    # Note that we deliberately spell this the same way openfold-3 has
    # historically, so that for now we can get bitwise identical results. A
    # masked fill or `(~mask) * -mask_val` would be more natural, but would yield
    # -0.0 on kept entries rather than +0.0, which is enough to shift bf16 and
    # fp16 results.
    return mask_val(dtype) * (mask.to(torch.float32) - 1)
