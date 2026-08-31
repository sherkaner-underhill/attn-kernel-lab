# SPDX-License-Identifier: Apache-2.0
"""GQA ratio generalization beyond the declared 24:4 (gap G11).

WHAT IS AND IS NOT BEING CLAIMED
================================
Contract §1 says GQA maps query head ``h`` to KV head ``h // (Hq / Hkv)`` and
requires only that ``Hq`` be divisible by ``Hkv``. The kernel agrees: its sole
head constraint is ``TORCH_CHECK(H % KVH == 0)``. Contract §2 nevertheless
DECLARES 24:4 and nothing else, because a declaration is a promise about the
pinned production target and nobody had measured any other ratio.

This file measures them. It does not promote anything. ``V1_CAPABILITY`` still
declares 24:4, ``tests/test_ops_rejection.py`` still pins every non-24:4 request
as a ``CapabilityError`` against the default surface, and
``test_declared_surface_stays_strict`` below asserts that from this side too, so
a ratio cannot be promoted by accident.

The tests drive the PUBLIC ``ops.prefill_extend`` with
``capability=GENERALIZATION_CAPABILITY``. That is what the second capability is
for: the ratios run positively through the whole validated path -- ops gate,
quantization, kernel -- rather than underneath the gate, and nothing about the
declared surface is weakened to allow it.

PROMOTING A RATIO (the qualification session's job)
=========================================
The one-line flip is ``capability.DECLARED_GQA_RATIOS = TESTED_GQA_RATIOS``.
It is gated on SM120 goldens for at least one non-24:4 ratio. Generate them on
the pinned target with:

    CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 -m pytest -q \
        tests/kernel/test_golden_bitexact.py --write-golden

after adding the chosen ratio's shape to that file's matrix -- goldens are
regenerated in their own commit and nothing else, per that file's regeneration
policy. Then flip the constant and record the promotion.
"""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch

HEAD_DIM = 256
PKG = os.path.join(os.path.dirname(__file__), "..", "..", "src", "attn_kernel_lab")
sys.path.insert(0, os.path.abspath(PKG))
sys.path.insert(0, os.path.abspath(os.path.join(PKG, "..")))

from attn_kernel_lab import ops  # noqa: E402
from attn_kernel_lab.capability import (  # noqa: E402
    DECLARED_GQA_RATIOS,
    GENERALIZATION_CAPABILITY,
    TESTED_GQA_RATIOS,
    V1_CAPABILITY,
    CapabilityError,
    check_supported,
)

#: The ratios under test, ids spelled ``Hq:Hkv=grp``. 24:4 rides along as the
#: control: if the declared shape drifts from the generalization ones, the
#: comparison in every other row is worthless.
RATIO_IDS = [f"{q}:{kv}=grp{q // kv}" for q, kv in TESTED_GQA_RATIOS]


def _qkv(q_len, kv_len, q_heads, kv_heads, seed, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)

    def mk(*shape):
        return (
            torch.empty(*shape, dtype=torch.float32)
            .uniform_(-1.0, 1.0, generator=g)
            .to(torch.bfloat16)
            .to(device)
        )

    return (
        mk(q_len, q_heads, HEAD_DIM),
        mk(kv_len, kv_heads, HEAD_DIM),
        mk(kv_len, kv_heads, HEAD_DIM),
    )


def _ref_attention(q, k, v, prefix):
    """FP32 bottom-right-causal attention, contract §1's mask built explicitly.

    ``F.scaled_dot_product_attention(is_causal=True)`` does NOT implement this
    for ``Q < K`` -- contract §1 says so in as many words -- so the mask is
    constructed here rather than delegated.
    """
    q_len, q_heads, head_dim = q.shape
    kv_len, kv_heads, _ = k.shape
    grp = q_heads // kv_heads
    out = torch.empty_like(q, dtype=torch.float32)
    cols = torch.arange(kv_len, device=q.device)
    lim = prefix + torch.arange(q_len, device=q.device)
    mask = cols[None, :] > lim[:, None]
    for h in range(q_heads):
        s = (q[:, h].float() @ k[:, h // grp].float().T) / math.sqrt(head_dim)
        s.masked_fill_(mask, float("-inf"))
        out[:, h] = torch.softmax(s, dim=1) @ v[:, h // grp].float()
    return out


def _row_rel(got, want):
    return (got - want).norm(dim=-1) / want.norm(dim=-1).clamp_min(1e-6)


# =========================================================================
# The declared surface is NOT widened by any of this
# =========================================================================


def test_declared_surface_admits_exactly_the_tested_ratios():
    """Post-promotion (2026-08-31, SM120 goldens for 32:4 landed): the DEFAULT
    surface admits every tested ratio and NOTHING else. The original guard's
    principle survives inverted: measuring a ratio is still never the thing
    that declares it -- the declaration moved because the goldens landed, and
    an untested divisible ratio must still be refused.
    """
    from attn_kernel_lab.capability import DECLARED_GQA_RATIOS

    for q_heads, kv_heads in DECLARED_GQA_RATIOS:
        check_supported(
            head_dim=256,
            q_heads=q_heads,
            kv_heads=kv_heads,
            page_size=1,
            mode="extend",
            mask="bottom_right_causal",
            kv_dtype="bfloat16",
        )
    # the per-axis schema must not over-admit: untested cross-pairs and 8:2
    # (tested but generalization-tier until pairs are expressible) all refuse
    for q_heads, kv_heads in ((8, 2), (8, 4), (16, 2), (48, 4), (24, 8)):
        assert (q_heads, kv_heads) not in DECLARED_GQA_RATIOS
        with pytest.raises(CapabilityError):
            check_supported(
                head_dim=256,
                q_heads=q_heads,
                kv_heads=kv_heads,
                page_size=1,
                mode="extend",
                mask="bottom_right_causal",
                kv_dtype="bfloat16",
            )


def test_every_declared_ratio_is_also_a_tested_one():
    """The invariant that survives promotion: a declaration must be backed by a
    numerical test. Flipping ``DECLARED_GQA_RATIOS`` to ``TESTED_GQA_RATIOS``
    keeps this true; declaring anything else does not."""
    assert set(DECLARED_GQA_RATIOS) <= set(TESTED_GQA_RATIOS)


def test_generalization_capability_admits_every_tested_ratio():
    """Otherwise the parameterized rows below could not run through the public
    path at all, and would have to reach under the ops gate."""
    for q_heads, kv_heads in TESTED_GQA_RATIOS:
        check_supported(
            head_dim=256,
            q_heads=q_heads,
            kv_heads=kv_heads,
            page_size=1,
            mode="extend",
            mask="bottom_right_causal",
            kv_dtype="bfloat16",
            capability=GENERALIZATION_CAPABILITY,
        )


def test_indivisible_ratio_is_refused_even_by_the_generalization_surface():
    """Divisibility is contract §1, not a declaration detail.

    Two refusals, because they are two different guarantees:

    * an out-of-tuple head count is refused by the *tuple*, and
    * an indivisible pair is refused by the *divisibility check*, which is the
      one that must survive any future widening of the tuples. Reaching it needs
      a capability whose tuples admit the pair -- with today's tuples every
      admitted ``q_heads`` is a multiple of every admitted ``kv_heads``, which is
      itself worth pinning: a widening that breaks it would silently make the
      second refusal unreachable rather than untrue.
    """
    import dataclasses

    with pytest.raises(CapabilityError) as excinfo:
        check_supported(
            head_dim=256,
            q_heads=24,
            kv_heads=16,
            page_size=1,
            mode="extend",
            mask="bottom_right_causal",
            kv_dtype="bfloat16",
            capability=GENERALIZATION_CAPABILITY,
        )
    assert excinfo.value.field == "kv_heads"

    assert all(
        q % kv == 0
        for q in GENERALIZATION_CAPABILITY.q_heads
        for kv in GENERALIZATION_CAPABILITY.kv_heads
    ), (
        "the generalization surface now admits an indivisible pair through its "
        "tuples; check_supported's divisibility guard is the only thing "
        "stopping it and it must gain a test row of its own"
    )

    indivisible = dataclasses.replace(GENERALIZATION_CAPABILITY, q_heads=(24,), kv_heads=(16,))
    with pytest.raises(CapabilityError) as excinfo:
        check_supported(
            head_dim=256,
            q_heads=24,
            kv_heads=16,
            page_size=1,
            mode="extend",
            mask="bottom_right_causal",
            kv_dtype="bfloat16",
            capability=indivisible,
        )
    assert excinfo.value.field == "q_heads"
    assert "divisible" in str(excinfo.value)


# =========================================================================
# Numerics at each ratio
# =========================================================================


@pytest.mark.parametrize("q_heads,kv_heads", TESTED_GQA_RATIOS, ids=RATIO_IDS)
@pytest.mark.parametrize(
    "q_len,kv_len,prefix",
    [
        (128, 1024, 896),  # the anchor shape, multi-tile prefix
        (100, 999, 899),  # ragged everything
    ],
)
def test_ratio_matches_fp32_reference(q_heads, kv_heads, q_len, kv_len, prefix):
    """The gross-breakage envelope of ``test_kernel_vs_sdpa`` at each ratio.

    Bounds are deliberately the SAME as the declared shape's rather than
    per-ratio-tuned: the claim is that the head mapping generalizes, so any
    ratio drifting outside the envelope the declared one sits in is the finding,
    not a threshold to widen.
    """
    q, k, v = _qkv(q_len, kv_len, q_heads, kv_heads, seed=q_heads * 101 + kv_len)
    idx = torch.arange(kv_len, device="cuda")
    got = ops.prefill_extend(q, k, v, idx, prefix, capability=GENERALIZATION_CAPABILITY).float()
    want = _ref_attention(q, k, v, prefix)
    rel = _row_rel(got, want)
    assert rel.mean().item() < 0.06, rel.mean().item()
    assert rel.max().item() < 0.15, rel.max().item()
    assert torch.isfinite(got).all()


@pytest.mark.parametrize("q_heads,kv_heads", TESTED_GQA_RATIOS, ids=RATIO_IDS)
def test_head_mapping_is_h_over_grp(q_heads, kv_heads):
    """Contract §1's mapping, asserted structurally rather than by tolerance.

    Perturbing KV head ``j`` must move exactly the query heads
    ``[j*grp, (j+1)*grp)`` and no others. A ``qh % KVH`` mapping, a
    ``qh & (KVH-1)`` mask, or an off-by-one in ``grp`` all fail here, and none of
    them would necessarily fail an error-envelope check.
    """
    q_len, kv_len, prefix = 64, 512, 448
    grp = q_heads // kv_heads
    q, k, v = _qkv(q_len, kv_len, q_heads, kv_heads, seed=q_heads * 7 + 3)
    idx = torch.arange(kv_len, device="cuda")

    base = ops.prefill_extend(q, k, v, idx, prefix, capability=GENERALIZATION_CAPABILITY).clone()
    for j in range(kv_heads):
        v_j = v.clone()
        v_j[:, j] *= -1.0
        moved = ops.prefill_extend(q, k, v_j, idx, prefix, capability=GENERALIZATION_CAPABILITY)
        touched = {h for h in range(q_heads) if not torch.equal(base[:, h], moved[:, h])}
        assert touched == set(range(j * grp, (j + 1) * grp)), (
            f"kv head {j} moved q heads {sorted(touched)}, expected "
            f"{list(range(j * grp, (j + 1) * grp))}"
        )


@pytest.mark.parametrize("q_heads,kv_heads", TESTED_GQA_RATIOS, ids=RATIO_IDS)
def test_ratio_survives_workspace_reuse_shrinking_n(q_heads, kv_heads):
    """Long-after-short on ONE workspace at each ratio: the stale-tail hazard of
    contract §5 is per-KV-head bookkeeping, so it deserves a row per ratio."""
    from attn_kernel_lab.quant import FP8PrefillWorkspace

    ws = FP8PrefillWorkspace(torch.device("cuda"))
    q_s, k_s, v_s = _qkv(32, 128, q_heads, kv_heads, seed=q_heads * 13)
    idx_s = torch.arange(128, device="cuda")
    first = ops.prefill_extend(
        q_s, k_s, v_s, idx_s, 96, workspace=ws, capability=GENERALIZATION_CAPABILITY
    ).clone()

    q_l, k_l, v_l = _qkv(128, 2048, q_heads, kv_heads, seed=q_heads * 13 + 1)
    ops.prefill_extend(
        q_l,
        k_l,
        v_l,
        torch.arange(2048, device="cuda"),
        1920,
        workspace=ws,
        capability=GENERALIZATION_CAPABILITY,
    )

    again = ops.prefill_extend(
        q_s, k_s, v_s, idx_s, 96, workspace=ws, capability=GENERALIZATION_CAPABILITY
    )
    assert torch.equal(first.view(torch.uint16), again.view(torch.uint16))


@pytest.mark.parametrize("q_heads,kv_heads", TESTED_GQA_RATIOS, ids=RATIO_IDS)
def test_ratio_runs_under_a_capacity_reserved_workspace(q_heads, kv_heads):
    """The two roadmap items meet here: a reserved workspace must be
    bit-identical to a grown one at every tested ratio, not only at 24:4."""
    q_len, kv_len, prefix = 96, 700, 604
    q, k, v = _qkv(q_len, kv_len, q_heads, kv_heads, seed=q_heads * 17 + 5)
    idx = torch.arange(kv_len, device="cuda")

    reserved = ops.plan_workspace(
        torch.device("cuda"),
        max_kv_len=1024,
        max_q_len=128,
        q_heads=q_heads,
        kv_heads=kv_heads,
        capability=GENERALIZATION_CAPABILITY,
    )
    assert reserved.capacity.q_heads == q_heads
    got = ops.prefill_extend(
        q, k, v, idx, prefix, workspace=reserved, capability=GENERALIZATION_CAPABILITY
    )
    want = ops.prefill_extend(q, k, v, idx, prefix, capability=GENERALIZATION_CAPABILITY)
    assert torch.equal(got.view(torch.uint16), want.view(torch.uint16))
