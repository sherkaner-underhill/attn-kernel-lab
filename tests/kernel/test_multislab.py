# SPDX-License-Identifier: Apache-2.0
"""The multi-slab path — the coverage gap the fruit report named Q0.

Production runs N = 446,335 = 14 slabs of 32,768, but every byte-exact
assertion in the repository ran at exactly one slab (goldens max N=1024,
oracle boundary tests max N=448). The channel-mean accumulation is
slab-ORDER-dependent fp32 arithmetic, so quant.py's bytes provably depend on
SLAB. These tests exercise 4–16 slabs by shrinking quant.SLAB, pinning:

  1. quant.py == oracle(slab=quant.SLAB), byte-exact, above one slab —
     including ragged tails that cross slab boundaries.
  2. The one-shot mathematical mean and the slab-ordered mean genuinely
     differ at the byte level (the divergence is REAL, documented, and the
     slab order is the normative one per contract §3.3).
  3. Slab size itself changes bytes — so SLAB is numerics-visible and its
     value is part of the operator's identity, not a tuning knob.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

HEAD_DIM = 256
PKG = os.path.join(os.path.dirname(__file__), "..", "..", "src", "attn_kernel_lab")
sys.path.insert(0, os.path.abspath(PKG))

import oracle_a  # noqa: E402
import quant as q_mod  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _kv(N, KVH=2, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    mk = lambda: (torch.empty(N, KVH, HEAD_DIM, dtype=torch.float32)
                  .uniform_(-1, 1, generator=g).to(torch.bfloat16).to(DEVICE))
    return mk(), mk()


def _run_quant(k, v):
    ws = q_mod.FP8PrefillWorkspace(torch.device(DEVICE))
    idx = torch.arange(k.shape[0], device=DEVICE)
    return q_mod.gather_quantize_kv(ws, k, v, idx, need_vt8=True, need_vb16=False,
                                    center_k=True, qk_i8=True, rotate=True)


@pytest.mark.parametrize("slab,N", [
    (64, 256),     # 4 exact slabs, slab == BLK
    (64, 1000),    # 16 slabs, ragged tail (1000 % 64 = 40) crossing both a
                   # slab boundary and a tile boundary
    (128, 1024),   # 8 slabs of 2 tiles each
    (128, 500),    # ragged: last slab is a partial tile
])
def test_quant_matches_oracle_above_one_slab(monkeypatch, slab, N):
    monkeypatch.setattr(q_mod, "SLAB", slab)
    k, v = _kv(N, seed=slab * 7 + N)
    theirs = _run_quant(k, v)
    rot = oracle_a.hadamard(HEAD_DIM, device=DEVICE)
    ours = oracle_a.preprocess_kv(k, v, rot, slab=slab)

    KVH = k.shape[1]
    nt = ours.k8.shape[1] // 64
    assert torch.equal(ours.k8.view(torch.uint8), theirs["k8"].view(KVH, -1, HEAD_DIM))
    assert torch.equal(ours.kscale, theirs["kscale"][:, :nt])
    assert torch.equal(ours.vt8.reshape(KVH, nt, HEAD_DIM, 64).view(torch.uint8),
                       theirs["vt8"].view(KVH, -1, HEAD_DIM, 64)[:, :nt])
    assert torch.equal(ours.vscale, theirs["vscale"])
    assert torch.equal(ours.vlog2r, theirs["vlog2r"][:, :nt])
    assert torch.equal(ours.vinvr, theirs["vinvr"][:, :nt])
    assert torch.equal(ours.vmean, theirs["vmean"])


def test_slab_order_diverges_from_one_shot_mean():
    """The finding itself: the slab-accumulated mean and the mathematical
    one-shot mean differ in fp32 once N spans multiple slabs. Not a bug — a
    reduction-order fact the contract now pins. If this test ever fails
    (i.e. they became identical), the pin can be revisited."""
    N, slab = 4096, 64
    k, v = _kv(N, seed=99)
    slabbed = oracle_a._channel_mean(v, N, slab)
    one_shot = oracle_a._channel_mean(v, N, None)
    assert torch.allclose(slabbed, one_shot, atol=1e-5)   # mathematically same
    assert not torch.equal(slabbed, one_shot), (
        "slab-ordered and one-shot means are bitwise identical at N=64*64; "
        "the documented divergence premise no longer holds — re-examine "
        "contract §3.3's slab pin")


def test_slab_size_is_numerics_visible(monkeypatch):
    """Different SLAB, different bytes: SLAB is part of the operator identity."""
    N = 1024
    k, v = _kv(N, seed=5)
    monkeypatch.setattr(q_mod, "SLAB", 64)
    a = _run_quant(k, v)
    a_mean = a["vmean"].clone()
    monkeypatch.setattr(q_mod, "SLAB", 1024)
    b = _run_quant(k, v)
    assert not torch.equal(a_mean, b["vmean"]), (
        "vmean identical across SLAB=64 and SLAB=1024 at N=1024 — if fp32 "
        "accumulation order stopped mattering, contract §3.3 should be relaxed")
