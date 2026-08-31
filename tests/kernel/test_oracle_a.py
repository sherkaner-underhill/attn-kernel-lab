# SPDX-License-Identifier: Apache-2.0
"""Oracle A vs the production implementation.

Three-way agreement: docs/OPERATOR_CONTRACT.md is the text, quant.py + the CUDA
kernel are the production implementation, oracle_a.py is an independent
reimplementation from the text. These tests are the comparison.

Intermediate boundaries must agree BYTE-EXACTLY: both sides perform the same
normatively specified fp32 tensor operations, so any byte difference is a
divergence between code and contract, not noise.

The final output comparison runs under a small CONTRACT TOLERANCE that absorbs
only what contract section 4 declares non-normative: ex2.approx vs exact exp,
fp32 accumulation order, and the bf16 store. It is far tighter than the
kernel-vs-BF16-attention envelopes in test_kernel_vs_sdpa.py, because the
intended quantization error is on BOTH sides here and cancels.
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

import oracle_a  # noqa: E402
import quant as q_mod  # noqa: E402
from attn_kernel_lab import kernel as kernel_mod  # noqa: E402


def _inputs(T, N, H=8, KVH=2, seed=0, device="cpu"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.empty(T, H, HEAD_DIM, dtype=torch.float32).uniform_(-1, 1, generator=g)
    k = torch.empty(N, KVH, HEAD_DIM, dtype=torch.float32).uniform_(-1, 1, generator=g)
    v = torch.empty(N, KVH, HEAD_DIM, dtype=torch.float32).uniform_(-1, 1, generator=g)
    return (q.to(torch.bfloat16).to(device),
            k.to(torch.bfloat16).to(device),
            v.to(torch.bfloat16).to(device))


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- constants

def test_hadamard_matches_production():
    ours = oracle_a.hadamard(HEAD_DIM, device=DEVICE)
    theirs = q_mod._hadamard(HEAD_DIM, DEVICE)
    assert torch.equal(ours, theirs)


def test_hadamard_is_orthonormal():
    h = oracle_a.hadamard(HEAD_DIM, dtype=torch.float64)
    eye = h @ h.T
    assert torch.allclose(eye, torch.eye(HEAD_DIM, dtype=torch.float64), atol=1e-12)


def test_sigma_permutation_matches_and_is_a_bijection():
    assert oracle_a.SIGMA64 == q_mod.SIGMA64
    assert sorted(oracle_a.SIGMA64) == list(range(64))


# ------------------------------------------------- intermediate boundaries

@pytest.mark.parametrize("T,N", [(64, 128), (100, 200), (128, 448), (37, 65)])
def test_q_boundary_bytes_exact(T, N):
    q, _, _ = _inputs(T, N, device=DEVICE)
    sm = 1.0 / math.sqrt(HEAD_DIM)
    rot = oracle_a.hadamard(HEAD_DIM, device=DEVICE)
    ours = oracle_a.preprocess_q(q, sm, rot)

    ws = q_mod.FP8PrefillWorkspace(torch.device(DEVICE))
    q8, qscale, mpad = q_mod.quantize_q(ws, q, sm, qk_i8=True, rotate=True)

    assert ours.mpad == mpad
    assert torch.equal(ours.q8.view(torch.uint8), q8)
    assert torch.equal(ours.qscale, qscale)


@pytest.mark.parametrize("T,N", [(64, 128), (64, 200), (64, 448), (16, 65)])
def test_kv_boundary_bytes_exact(T, N):
    """K bytes/scales/mean, V bytes in the SIGMA layout, every fold array."""
    _, k, v = _inputs(T, N, device=DEVICE)
    rot = oracle_a.hadamard(HEAD_DIM, device=DEVICE)
    ours = oracle_a.preprocess_kv(k, v, rot)

    ws = q_mod.FP8PrefillWorkspace(torch.device(DEVICE))
    kp = k.view(N, -1, HEAD_DIM)
    idx = torch.arange(N, device=DEVICE)
    theirs = q_mod.gather_quantize_kv(ws, kp, v.view(N, -1, HEAD_DIM), idx,
                                      need_vt8=True, need_vb16=False,
                                      center_k=True, qk_i8=True, rotate=True)
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


def test_boundary_tile_hygiene_zeroes_permuted_tail():
    """Ragged N: every vt8 column whose SOURCE position is padding must be
    zero — the stale-V 0*NaN hazard is contract-level (section 5)."""
    _, k, v = _inputs(16, 65, device=DEVICE)
    rot = oracle_a.hadamard(HEAD_DIM, device=DEVICE)
    ours = oracle_a.preprocess_kv(k, v, rot)
    sigma = torch.tensor(oracle_a.SIGMA64)
    tail = (sigma >= 1).nonzero(as_tuple=True)[0]  # N%64 == 1
    assert (ours.vt8[:, 1][:, :, tail].view(torch.uint8) == 0).all()
    kept = (sigma < 1).nonzero(as_tuple=True)[0]
    assert kept.numel() == 1  # exactly one real column survives


def test_v_underflow_floor_engages():
    """A quiet tile must be floored at vs_max/16 (P-underflow guard).

    Construction note: "quiet" means quiet AFTER centering. Setting tile 1 to
    the mean of tile 0 makes the global channel mean equal that constant, so
    tile 1 centers to exactly zero while tile 0 stays loud."""
    _, k, v = _inputs(16, 128, device=DEVICE)
    v = v.clone()
    v[64:] = v[:64].to(torch.float32).mean(dim=0).to(v.dtype)
    ours = oracle_a.preprocess_kv(k, v, oracle_a.hadamard(HEAD_DIM, device=DEVICE))
    ratio = 1.0 / ours.vinvr
    assert torch.isclose(ratio[:, 1], torch.tensor(1.0 / 16.0, device=ratio.device)).all()
    assert (ratio >= 1.0 / 16.0 - 1e-7).all() and (ratio <= 1.0 + 1e-7).all()


# ----------------------------------------------------- final output (GPU)

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="kernel comparison needs a GPU")


@pytest.fixture(scope="module")
def ext():
    """The one extension the whole repo builds: ``attn_kernel_lab_fp8_prefill``.

    This carried a THIRD name (``attn_kernel_lab_oracle_test``) for a build with
    flags identical to the other two -- a third redundant nvcc compile per cold
    run, because ``cpp_extension`` keys its build directory on the name alone.
    See the same fixture in ``test_kernel_vs_sdpa.py`` for the full note,
    including the rule that any future test needing DIFFERENT ``-D`` flags must
    take its own name.
    """
    return kernel_mod.load()


def _run_kernel(ext, q, k, v, prefix):
    T, H, _ = q.shape
    ws = q_mod.FP8PrefillWorkspace(q.device)
    idx = torch.arange(k.shape[0], device=q.device)
    kv = q_mod.gather_quantize_kv(ws, k, v, idx, need_vt8=True, need_vb16=False,
                                  center_k=True, qk_i8=True, rotate=True)
    q8, qscale, mpad = q_mod.quantize_q(ws, q, 1.0 / math.sqrt(HEAD_DIM),
                                        qk_i8=True, rotate=True)
    o = ws.get("o", (H, mpad, HEAD_DIM), torch.bfloat16)
    mask = torch.ones(H, dtype=torch.uint8, device=q.device)
    ext.fp8_prefill_attn(q8, kv["k8"], kv["vt8"], kv["vb16"], o, qscale,
                         kv["kscale"], kv["vscale"], kv["vlog2r"], kv["vinvr"],
                         kv["vmean"], mask, kv["n"], prefix, True, True, True)
    return o.view(H, mpad, HEAD_DIM)[:, :T].permute(1, 0, 2).contiguous()


# Contract tolerance: only ex2.approx-vs-exp, fp32 accumulation order, and the
# bf16 store separate the two sides. Calibrated on SM89 2026-08-30: worst
# row-relative L2 observed 4.60e-3 (at the deepest shape, T=128/N=448; the
# shallow shapes sit at ~1e-3); asserted at ~2x the observed worst. A breach
# means an implementation divergence from the contract, NOT quantization
# error — do not widen without a finding. Re-calibrate on SM120 in B1.
CONTRACT_ROW_REL = 9e-3


def _row_rel(a, b):
    diff = (a.float() - b.float()).norm(dim=-1)
    return (diff / b.float().norm(dim=-1).clamp_min(1e-6)).max().item()


@needs_gpu
@pytest.mark.parametrize("T,N,prefix", [
    (64, 64, 0), (64, 128, 64), (128, 448, 320), (100, 199, 99), (37, 65, 28),
])
def test_kernel_matches_oracle_within_contract_tolerance(ext, T, N, prefix):
    q, k, v = _inputs(T, N, device="cuda", seed=T + N)
    got = _run_kernel(ext, q, k.view(N, -1, HEAD_DIM), v.view(N, -1, HEAD_DIM), prefix)
    want = oracle_a.reference(q.cpu(), k.cpu(), v.cpu(), prefix).to("cuda")
    assert _row_rel(got, want) < CONTRACT_ROW_REL


@needs_gpu
def test_oracle_catches_a_wrong_scale(ext):
    """Sensitivity check: the oracle must FAIL against a deliberately perturbed
    operator, or agreement means nothing. Perturb one K tile scale by 3%."""
    T, N, prefix = 64, 128, 64
    q, k, v = _inputs(T, N, device="cuda", seed=7)
    ws = q_mod.FP8PrefillWorkspace(q.device)
    idx = torch.arange(N, device="cuda")
    kv = q_mod.gather_quantize_kv(ws, k.view(N, -1, HEAD_DIM), v.view(N, -1, HEAD_DIM),
                                  idx, need_vt8=True, need_vb16=False)
    q8, qscale, mpad = q_mod.quantize_q(ws, q, 1.0 / math.sqrt(HEAD_DIM))
    kv["kscale"][:, 0] *= 1.03
    o = ws.get("o", (q.shape[1], mpad, HEAD_DIM), torch.bfloat16)
    mask = torch.ones(q.shape[1], dtype=torch.uint8, device="cuda")
    ext.fp8_prefill_attn(q8, kv["k8"], kv["vt8"], kv["vb16"], o, qscale,
                         kv["kscale"], kv["vscale"], kv["vlog2r"], kv["vinvr"],
                         kv["vmean"], mask, kv["n"], prefix, True, True, True)
    got = o.view(-1, mpad, HEAD_DIM)[:, :T].permute(1, 0, 2)
    want = oracle_a.reference(q.cpu(), k.cpu(), v.cpu(), prefix).to("cuda")
    assert _row_rel(got, want) > CONTRACT_ROW_REL


# ------------------------------------------------------------------- LSE

# Calibrated on SM89 2026-08-30 alongside CONTRACT_ROW_REL: worst |lse| abs
# deviation observed 1.6e-3 across the matrix (ex2.approx enters both m-updates
# and the l accumulation); asserted at ~3x. Same rule: a breach is a contract
# divergence, never quantization error.
LSE_ABS_TOL = 5e-3


def _run_kernel_lse(ext, q, k, v, prefix):
    T, H, _ = q.shape
    ws = q_mod.FP8PrefillWorkspace(q.device)
    idx = torch.arange(k.shape[0], device=q.device)
    kv = q_mod.gather_quantize_kv(ws, k, v, idx, need_vt8=True, need_vb16=False,
                                  center_k=True, qk_i8=True, rotate=True)
    q8, qscale, mpad = q_mod.quantize_q(ws, q, 1.0 / math.sqrt(HEAD_DIM),
                                        qk_i8=True, rotate=True)
    o = ws.get("o", (H, mpad, HEAD_DIM), torch.bfloat16)
    lse = torch.full((H, mpad), float("nan"), dtype=torch.float32, device=q.device)
    mask = torch.ones(H, dtype=torch.uint8, device=q.device)
    ext.fp8_prefill_attn(q8, kv["k8"], kv["vt8"], kv["vb16"], o, qscale,
                         kv["kscale"], kv["vscale"], kv["vlog2r"], kv["vinvr"],
                         kv["vmean"], mask, kv["n"], prefix, True, True, True,
                         lse)
    out = o.view(H, mpad, HEAD_DIM)[:, :T].permute(1, 0, 2).contiguous()
    return out, lse[:, :T].permute(1, 0).contiguous()


@needs_gpu
@pytest.mark.parametrize("T,N,prefix", [
    (64, 64, 0), (64, 128, 64), (128, 448, 320), (100, 199, 99), (37, 65, 28),
])
def test_lse_matches_oracle(ext, T, N, prefix):
    """The kernel's base-2 LSE against the contract oracle's."""
    q, k, v = _inputs(T, N, device="cuda", seed=1000 + T + N)
    _, got_lse = _run_kernel_lse(ext, q, k.view(N, -1, HEAD_DIM),
                                 v.view(N, -1, HEAD_DIM), prefix)
    _, want_lse = oracle_a.reference(q.cpu(), k.cpu(), v.cpu(), prefix,
                                     return_lse=True)
    assert torch.isfinite(got_lse).all()
    assert (got_lse - want_lse.to("cuda")).abs().max().item() < LSE_ABS_TOL


@needs_gpu
def test_lse_consistent_with_logsumexp_of_dequantized_scores(ext):
    """Independent route: base-2 logsumexp of the masked, dequantized-score
    matrix must agree with the kernel LSE. Different code path end to end."""
    T, N, prefix = 64, 128, 64
    q, k, v = _inputs(T, N, device="cuda", seed=42)
    _, got_lse = _run_kernel_lse(ext, q, k.view(N, -1, HEAD_DIM),
                                 v.view(N, -1, HEAD_DIM), prefix)

    rot = oracle_a.hadamard(HEAD_DIM)
    qp = oracle_a.preprocess_q(q.cpu(), 1.0 / math.sqrt(HEAD_DIM), rot)
    kv = oracle_a.preprocess_kv(k.cpu(), v.cpu(), rot)
    H = q.shape[1]
    group = H // k.shape[1]
    want = torch.empty(T, H)
    cols = torch.arange(N)
    for h in range(H):
        kvh = h // group
        k8f = kv.k8.view(kv.k8.shape[0], -1, 64, HEAD_DIM).to(torch.float64)
        scores = torch.empty(T, N, dtype=torch.float64)
        for t in range((N + 63) // 64):
            lo, hi = t * 64, min((t + 1) * 64, N)
            scores[:, lo:hi] = (
                qp.q8[h, :T].to(torch.float64) @ k8f[kvh, t, : hi - lo].T
            ) * qp.qscale[h, :T, None].to(torch.float64) * float(kv.kscale[kvh, t])
        mask = cols[None, :] > (prefix + torch.arange(T))[:, None]
        scores.masked_fill_(mask, -math.inf)
        want[:, h] = (torch.logsumexp(scores, dim=1) / math.log(2.0)).float()
    assert (got_lse.cpu() - want).abs().max().item() < LSE_ABS_TOL


@needs_gpu
def test_output_bytes_unchanged_by_lse_request(ext):
    """Requesting LSE must not perturb the value path: O byte-identical with
    and without the writeback on the same build and inputs."""
    T, N, prefix = 64, 128, 64
    q, k, v = _inputs(T, N, device="cuda", seed=9)
    out_plain = _run_kernel(ext, q, k.view(N, -1, HEAD_DIM),
                            v.view(N, -1, HEAD_DIM), prefix).clone()
    out_lse, _ = _run_kernel_lse(ext, q, k.view(N, -1, HEAD_DIM),
                                 v.view(N, -1, HEAD_DIM), prefix)
    assert torch.equal(out_plain.view(torch.uint16), out_lse.view(torch.uint16))
