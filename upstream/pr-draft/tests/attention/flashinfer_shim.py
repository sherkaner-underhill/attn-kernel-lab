# SPDX-License-Identifier: Apache-2.0
"""Adapter: FlashInfer call shapes and test helpers onto ``attn_kernel_lab.ops``.

**This file does not go upstream.** It exists so the test module beside it can be
written in FlashInfer's house style today, against this repository's operator,
and become an in-tree test by rewriting its imports. Everything the shim exports
has an in-tree counterpart:

===========================  ==================================================
this shim                    FlashInfer in-tree replacement
===========================  ==================================================
``get_compute_capability``   ``flashinfer.utils.get_compute_capability``
``is_sm120a_supported``      ``flashinfer.utils.is_sm120a_supported``
``is_sm121a_supported``      ``flashinfer.utils.is_sm121a_supported``
``ref_single_prefill``       ``tests.test_helpers.test_helpers.ref_single_prefill``
``flip_coin``                ``tests.test_helpers.test_helpers.flip_coin``
``fp8_paged_prefill_sm120``  ``flashinfer.<op module>.fp8_paged_prefill_sm120``,
                             decorated ``@supported_compute_capability([120, 121])``
``GENERALIZATION_SURFACE``   nothing -- upstream would simply accept any
                             divisible GQA ratio, so the sweep needs no opt-in
===========================  ==================================================

Three differences are deliberate and are the ones to look at when porting:

1. **Pool layout.** FlashInfer's paged NHD cache is
   ``[num_pages, page_size, num_kv_heads, head_dim]``; ``ops.prefill_extend``
   takes ``[pool, num_kv_heads, head_dim]`` because ``page_size`` is 1 by
   construction. The shim squeezes the page axis *after* rejecting
   ``page_size != 1``, so the rejection is a ``ValueError`` and never a
   mis-shaped read.
2. **Error type.** Upstream raises bare ``ValueError``. This project raises
   ``CapabilityError``, which *is* a ``ValueError``, so every
   ``pytest.raises(ValueError, match=...)`` in the test module ports unchanged.
3. **Bottom-right causal is derived, not passed.** ``prefix = kv_len - qo_len``,
   matching the contract's statement that the causal diagonal sits at offset
   ``K - Q``. Upstream's paged wrappers spell the same thing ``causal=True``
   over an indptr pair.
"""
from __future__ import annotations

import hashlib
import math
import pathlib
import sys

import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from attn_kernel_lab import ops  # noqa: E402
from attn_kernel_lab.capability import (  # noqa: E402
    GENERALIZATION_CAPABILITY,
    V1_CAPABILITY,
    check_supported,
)

#: The surface the generalization matrix (gap G11) runs against. Upstream has no
#: analogue because upstream would accept any divisible ratio outright.
GENERALIZATION_SURFACE = GENERALIZATION_CAPABILITY
DECLARED_SURFACE = V1_CAPABILITY

__all__ = [
    "DECLARED_SURFACE",
    "GENERALIZATION_SURFACE",
    "flip_coin",
    "make_workspace",
    "fp8_paged_prefill_sm120",
    "get_compute_capability",
    "is_sm120a_supported",
    "is_sm121a_supported",
    "ref_single_prefill",
]


def make_workspace(device: torch.device):
    """The reusable operator workspace.

    Upstream's equivalent is a caller-owned flat ``float_workspace_buffer`` sized
    from a ``plan()``-returned byte requirement; this project still uses a set of
    monotonically grown named buffers (gap G10, and the reason the CUDA-graph
    declaration is ``eager_only``). The factory exists so the test module never
    reaches into package internals -- when the flat buffer lands, only this
    function changes.
    """
    from attn_kernel_lab.quant import FP8PrefillWorkspace

    return FP8PrefillWorkspace(device)


# ----------------------------------------------------------- capability gating


def get_compute_capability(device: torch.device) -> tuple[int, int]:
    """``flashinfer.utils.get_compute_capability``."""
    if device.type != "cuda":
        raise ValueError("device must be a CUDA device")
    return torch.cuda.get_device_capability(device)


def is_sm120a_supported(device: torch.device) -> bool:
    """SM120a: ``major == 12 and minor == 0`` with CUDA >= 12.8."""
    major, minor = get_compute_capability(device)
    return major == 12 and minor == 0 and _cuda_at_least(12, 8)


def is_sm121a_supported(device: torch.device) -> bool:
    """SM121a: ``major == 12 and minor == 1`` with CUDA >= 12.9."""
    major, minor = get_compute_capability(device)
    return major == 12 and minor == 1 and _cuda_at_least(12, 9)


def _cuda_at_least(major: int, minor: int) -> bool:
    version = torch.version.cuda
    if not version:
        return False
    parts = version.split(".")
    return (int(parts[0]), int(parts[1] if len(parts) > 1 else 0)) >= (major, minor)


# ------------------------------------------------------------- test utilities


def flip_coin(*args) -> bool:
    """Deterministic alternation over a parametrization, upstream's trick for
    covering "caller supplies ``out=``" vs "kernel allocates" without doubling
    the matrix.

    Upstream uses the builtin ``hash``, which is salted per process for strings;
    a digest is used here so a given ``pytest.param`` id always lands on the same
    side of the coin regardless of ``PYTHONHASHSEED``. Keep this version when
    porting -- it is strictly the more reproducible of the two. Which digest is
    arbitrary; blake2b splits the length matrix in this module 3/4 rather than
    2/5, which is the only reason it is this one.
    """
    digest = hashlib.blake2b(repr(args).encode(), digest_size=8).digest()
    return digest[0] % 2 == 0


def ref_single_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP64 single-request reference, mask and return shape as in-tree.

    ``q`` is ``[qo_len, num_heads, head_dim]`` and ``k``/``v`` are
    ``[kv_len, num_heads, head_dim]`` -- GQA is expressed by the caller passing
    ``repeat_interleave``d K/V, exactly as upstream does it.

    The mask is upstream's, verbatim in effect:
    ``k_pos[None, :] - (kv_len - qo_len) > q_pos[:, None]`` -- i.e. **bottom
    right** causal for ``qo_len < kv_len``, which is the mask this operator
    implements and which ``F.scaled_dot_product_attention(is_causal=True)`` does
    *not* implement for ``Q < K``.

    Returns ``(out, lse)`` with **base-2** LSE, ``log2(sum(exp(scores)))``
    computed stably as ``row_max * log2(e) + log2(sum_exp)``.

    Base is worth checking on port. The gap analysis records the in-tree helper
    as returning ``row_max + log2(sum_exp)``, which is the same expression only
    if ``row_max`` is already in base-2 units; FA2/CUTLASS wrappers are base-2
    and the SM120 standalone op is natural-log, and upstream issue #4485 is open
    precisely because the convention is not uniform. The operator under test
    returns no LSE through its public API at v1 (gap G1), so nothing here depends
    on it -- the second element exists so swapping in the in-tree helper is an
    import change and nothing else.
    """
    qo_len, num_heads, head_dim = q.shape
    kv_len = k.shape[0]
    scale = (1.0 / math.sqrt(head_dim)) if sm_scale is None else sm_scale

    q64 = q.to(torch.float64).transpose(0, 1)
    k64 = k.to(torch.float64).transpose(0, 1)
    v64 = v.to(torch.float64).transpose(0, 1)

    scores = torch.einsum("hqd,hkd->hqk", q64, k64) * scale
    if causal:
        q_pos = torch.arange(qo_len, device=q.device)[:, None]
        k_pos = torch.arange(kv_len, device=q.device)[None, :]
        scores.masked_fill_(k_pos - (kv_len - qo_len) > q_pos, float("-inf"))

    row_max = scores.amax(dim=-1, keepdim=True)
    row_max = torch.where(torch.isinf(row_max), torch.zeros_like(row_max), row_max)
    exp_scores = torch.exp(scores - row_max)
    sum_exp = exp_scores.sum(dim=-1, keepdim=True)
    out = torch.einsum("hqk,hkd->hqd", exp_scores / sum_exp, v64)
    lse = (row_max * math.log2(math.e) + torch.log2(sum_exp)).squeeze(-1)
    return out.transpose(0, 1), lse.transpose(0, 1)


# ------------------------------------------------------------------ the operator


def fp8_paged_prefill_sm120(
    q: torch.Tensor,
    paged_k_cache: torch.Tensor,
    paged_v_cache: torch.Tensor,
    kv_indices: torch.Tensor,
    *,
    page_size: int = 1,
    causal: bool = True,
    sm_scale: float | None = None,
    out: torch.Tensor | None = None,
    return_lse: bool = False,
    workspace_buffer=None,
    capability=DECLARED_SURFACE,
):
    """FlashInfer-shaped entry: one request over a paged BF16 KV cache.

    Args:
        q: ``[qo_len, num_qo_heads, head_dim]`` BF16, post-RoPE.
        paged_k_cache, paged_v_cache: ``[num_pages, page_size, num_kv_heads,
            head_dim]`` BF16, NHD layout.
        kv_indices: ``[kv_len]`` int64 page ids in position order, prefix first.
        page_size: 1 is the declared surface; anything else is rejected here,
            before the cache is reshaped.
        causal: ``True`` selects the bottom-right causal mask, the only mask this
            operator implements; ``False`` is rejected.
        out: optional ``[qo_len, num_qo_heads, head_dim]`` BF16 destination.
        return_lse: refused at v1 (gap G1).
        workspace_buffer: an ``FP8PrefillWorkspace``. Named after the upstream
            argument; it is not yet a caller-owned flat byte buffer (gap G10).

    Raises:
        ValueError: the request is outside the declared surface.
    """
    if not isinstance(q, torch.Tensor) or q.dim() != 3:
        raise ValueError("q must be a 3-D [qo_len, num_qo_heads, head_dim] tensor")
    if not isinstance(paged_k_cache, torch.Tensor) or paged_k_cache.dim() != 4:
        raise ValueError(
            "paged_k_cache must be a 4-D [num_pages, page_size, num_kv_heads, "
            f"head_dim] NHD tensor, got {paged_k_cache.dim()}-D"
        )

    # The declared-surface axes the pool reshape below would otherwise silently
    # reinterpret: page_size and the mask mode. Checked first, and with no
    # device work, so both are ValueErrors rather than a mis-shaped read.
    check_supported(
        head_dim=int(q.shape[2]),
        q_heads=int(q.shape[1]),
        kv_heads=int(paged_k_cache.shape[2]),
        page_size=int(page_size),
        mode="extend",
        mask="bottom_right_causal" if causal else "noncausal",
        kv_dtype=str(paged_k_cache.dtype).rsplit(".", 1)[-1],
        capability=capability,
    )

    k_pool = paged_k_cache.squeeze(1)
    v_pool = paged_v_cache.squeeze(1)
    prefix = int(kv_indices.numel()) - int(q.shape[0])

    return ops.prefill_extend(
        q,
        k_pool,
        v_pool,
        kv_indices,
        prefix,
        workspace=workspace_buffer,
        out=out,
        sm_scale=sm_scale,
        return_lse=return_lse,
        capability=capability,
    )
