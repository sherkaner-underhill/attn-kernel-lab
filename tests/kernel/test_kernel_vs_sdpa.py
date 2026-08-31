# SPDX-License-Identifier: Apache-2.0
"""Kernel-level tests: fused fp8 prefill attention vs a torch fp32 reference.

Runs on an SM120-class GPU with a CUDA toolchain for the JIT:

    python3 -m pytest test_kernel_vs_sdpa.py -x -q

Bounds are calibrated from the measured numerics program (synthetic
uniform data, see package README): fp8-PV row-relative error shrinks with
depth (7.5e-3 @128 ... 1.0e-4 abs @446k); bf16-PV stays ~10x tighter.
The tests assert conservative envelopes, plus structural properties
(ragged tails, GQA mapping, per-head mask routing, causal boundary).
"""

import math
import os
import sys

import pytest
import torch

HEAD_DIM = 256
PKG = os.path.join(os.path.dirname(__file__), "..", "..", "src", "attn_kernel_lab")
sys.path.insert(0, os.path.abspath(PKG))
sys.path.insert(0, os.path.abspath(os.path.join(PKG, "..")))

import quant as q_mod  # noqa: E402  (the package's quant.py, imported standalone)
from attn_kernel_lab import kernel as kernel_mod  # noqa: E402


@pytest.fixture(scope="module")
def ext():
    """The one extension the whole repo builds: ``attn_kernel_lab_fp8_prefill``.

    This used to JIT the same ``.cu`` under a second name with byte-identical
    flags. ``cpp_extension._get_build_directory`` keys the build directory on
    the extension NAME alone, and the source/flag hash cache is an in-process
    dict reset by every new interpreter -- so a second name meant a second real
    nvcc compile of identical source, ~56 s on a cold cache, every cold run.
    Delegating to the package loader collapses that to one build directory
    shared with the public path, and leaves ``kernel.gencode_for`` as the single
    place the sm_120a gencode trap is handled.

    Forward-looking rule: a test that ever needs DIFFERENT ``-D`` flags must
    take its own distinct extension name. Sharing a name across differing flags
    would silently serve the wrong binary.
    """
    return kernel_mod.load()


def ref_attention(q, k, v, prefix):
    """fp32 bottom-right-causal attention. q [T,H,D], k/v [N,KVH,D]."""
    T, H, D = q.shape
    N, KVH, _ = k.shape
    grp = H // KVH
    out = torch.empty_like(q, dtype=torch.float32)
    cols = torch.arange(N, device=q.device)
    lim = prefix + torch.arange(T, device=q.device)
    mask = cols[None, :] > lim[:, None]
    for h in range(H):
        s = (q[:, h].float() @ k[:, h // grp].float().T) / math.sqrt(D)
        s.masked_fill_(mask, float("-inf"))
        out[:, h] = torch.softmax(s, dim=1) @ v[:, h // grp].float()
    return out


def run_kernel(
    ext, q, k, v, prefix, bf16_heads=(), ws=None, center_k=True, qk_i8=True, rotate=True
):
    """Quantize with the package pipeline and run the fused kernel."""
    T, H, D = q.shape
    N, KVH, _ = k.shape
    if ws is None:
        ws = q_mod.FP8PrefillWorkspace(q.device)
    mask = torch.ones(H, dtype=torch.uint8, device=q.device)
    for h in bf16_heads:
        mask[h] = 0
    any_pv8 = bool(mask.max().item())
    all_pv8 = bool(mask.min().item())

    # feed the gather path an identity index (pool == contiguous K/V here)
    idx = torch.arange(N, device=q.device)
    kv = q_mod.gather_quantize_kv(
        ws,
        k,
        v,
        idx,
        need_vt8=any_pv8,
        need_vb16=not all_pv8,
        center_k=center_k,
        qk_i8=qk_i8,
        rotate=rotate,
    )
    q8, qscale, mpad = q_mod.quantize_q(ws, q, 1.0 / math.sqrt(D), qk_i8=qk_i8, rotate=rotate)
    o = ws.get("o", (H, mpad, HEAD_DIM), torch.bfloat16)
    ext.fp8_prefill_attn(
        q8,
        kv["k8"],
        kv["vt8"],
        kv["vb16"],
        o,
        qscale,
        kv["kscale"],
        kv["vscale"],
        kv["vlog2r"],
        kv["vinvr"],
        kv["vmean"],
        mask,
        kv["n"],
        prefix,
        any_pv8,
        all_pv8,
        qk_i8,
    )
    return o[:, :T].permute(1, 0, 2).float()


def rand_qkv(T, N, H=8, KVH=2, seed=0, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = (torch.rand((T, H, HEAD_DIM), generator=g) * 2 - 1).to(device, torch.bfloat16)
    k = (torch.rand((N, KVH, HEAD_DIM), generator=g) * 2 - 1).to(device, torch.bfloat16)
    v = (torch.rand((N, KVH, HEAD_DIM), generator=g) * 2 - 1).to(device, torch.bfloat16)
    return q, k, v


def row_rel(a, b):
    return (a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-6)


@pytest.mark.parametrize(
    "T,N,prefix",
    [
        (64, 128, 64),  # 2 tiles, causal boundary
        (128, 1024, 896),  # multi-tile prefix
        (192, 4096, 3904),  # deeper
        (100, 999, 899),  # ragged everything
        (64, 63, 0),  # N smaller than one tile (self-attn only, ragged)
    ],
)
def test_fp8_pv_vs_reference(ext, T, N, prefix):
    q, k, v = rand_qkv(T, N, seed=T + N)
    o = run_kernel(ext, q, k, v, prefix)
    ref = ref_attention(q, k, v, prefix)
    rr = row_rel(o, ref)
    # Against a TRUE fp32 reference the fp8-QK logit noise dominates on
    # uniform synthetic data (~3.5-4% row-rel mean at every depth; measured
    # real-workload tensors are ~2%, see README). These bounds are the gross-
    # breakage envelope; precision-grade checks are the exact-equality
    # tests below plus the dequantized-reference verify in the research
    # kernel (1.0e-4 at 446k).
    assert rr.mean().item() < 0.06, rr.mean().item()
    assert rr.max().item() < 0.15, rr.max().item()


def test_bf16_pv_mode_is_tight(ext):
    T, N, prefix = 128, 2048, 1920
    q, k, v = rand_qkv(T, N, seed=7)
    o = run_kernel(ext, q, k, v, prefix, bf16_heads=range(8))
    ref = ref_attention(q, k, v, prefix)
    rr = row_rel(o, ref)
    # bf16 PV removes the P/V quantization term; the residual is fp8-QK
    # logit noise, so the bound is ~3x tighter than the fp8-PV envelope
    assert rr.mean().item() < 0.02, rr.mean().item()
    assert rr.max().item() < 0.08, rr.max().item()


def test_mixed_head_mask_routes_correctly(ext):
    """bf16-PV heads must match the all-bf16 run bit-exactly; fp8 heads the
    all-fp8 run: proves the mask selects the intended path per head."""
    T, N, prefix = 128, 1024, 896
    q, k, v = rand_qkv(T, N, seed=11)
    o_all8 = run_kernel(ext, q, k, v, prefix)
    o_all16 = run_kernel(ext, q, k, v, prefix, bf16_heads=range(8))
    o_mixed = run_kernel(ext, q, k, v, prefix, bf16_heads=(2, 5))
    assert torch.equal(o_mixed[:, 2], o_all16[:, 2])
    assert torch.equal(o_mixed[:, 5], o_all16[:, 5])
    assert torch.equal(o_mixed[:, 0], o_all8[:, 0])
    assert torch.equal(o_mixed[:, 7], o_all8[:, 7])


def test_causal_boundary_exact(ext):
    """A row must not see future kv positions.  Perturbation must land in a
    kv TILE that is wholly invisible to the protected rows (per-64-tile K
    scales couple columns within a tile), and only K may be perturbed (the
    global V scale couples every position).

    prefix=224, T=64: rows [0,32) see cols <= 255 = tiles 0..3 exactly;
    tile 4 (cols 256..319) is invisible to them and visible to rows >= 32.
    """
    T, N, prefix = 64, 320, 224
    q, k, v = rand_qkv(T, N, seed=3)
    # center_k=False: centering couples ALL rows through the channel means,
    # so a single-row perturbation would (legitimately) move every row's
    # quantization; this test isolates the kernel's causal masking.
    o1 = run_kernel(ext, q, k, v, prefix, center_k=False)
    k2 = k.clone()
    k2[280] += 3.0  # inside tile 4
    o2 = run_kernel(ext, q, k2, v, prefix, center_k=False)
    assert torch.equal(o1[:32], o2[:32])  # rows that cannot see tile 4
    assert not torch.equal(o1[32:], o2[32:])  # rows that can


def test_gqa_mapping(ext):
    """Making kv-head 1 radically different must affect exactly q heads
    grp..2*grp-1."""
    T, N, prefix = 64, 512, 448
    q, k, v = rand_qkv(T, N, H=8, KVH=2, seed=5)
    o1 = run_kernel(ext, q, k, v, prefix)
    v2 = v.clone()
    v2[:, 1] *= -1.0
    o2 = run_kernel(ext, q, k, v2, prefix)
    grp = 8 // 2
    assert torch.equal(o1[:, :grp], o2[:, :grp])
    assert not torch.equal(o1[:, grp:], o2[:, grp:])


def test_gather_path_matches_contiguous(ext):
    """A shuffled pool with the inverse index must reproduce the contiguous
    result exactly (validates the gather/index plumbing)."""
    T, N, prefix = 64, 512, 448
    q, k, v = rand_qkv(T, N, seed=9)
    perm = torch.randperm(N, device=q.device)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(N, device=q.device)
    ws = q_mod.FP8PrefillWorkspace(q.device)
    mask = torch.ones(8, dtype=torch.uint8, device=q.device)
    kv = q_mod.gather_quantize_kv(ws, k[perm], v[perm], inv, need_vt8=True, need_vb16=False)
    q8, qscale, mpad = q_mod.quantize_q(ws, q, 1.0 / math.sqrt(HEAD_DIM))
    o = ws.get("o", (8, mpad, HEAD_DIM), torch.bfloat16)
    ext.fp8_prefill_attn(
        q8,
        kv["k8"],
        kv["vt8"],
        kv["vb16"],
        o,
        qscale,
        kv["kscale"],
        kv["vscale"],
        kv["vlog2r"],
        kv["vinvr"],
        kv["vmean"],
        mask,
        kv["n"],
        prefix,
        True,
        True,
        True,
    )
    o_gather = o[:, :T].permute(1, 0, 2).float()
    o_direct = run_kernel(ext, q, k, v, prefix)
    assert torch.equal(o_gather, o_direct)


def test_workspace_reuse_shrinking_n(ext):
    """A short request after a long one must not read the long request's
    stale workspace tails (the NaN-hygiene + masking contract). Uses ONE
    workspace across both calls, as the backend does."""
    ws = q_mod.FP8PrefillWorkspace(torch.device("cuda"))
    q1, k1, v1 = rand_qkv(64, 4096, seed=13)
    run_kernel(ext, q1, k1, v1, 4032, ws=ws)  # long first (populates)
    q2, k2, v2 = rand_qkv(64, 200, seed=14)  # short + ragged after
    o = run_kernel(ext, q2, k2, v2, 136, ws=ws)
    ref = ref_attention(q2, k2, v2, 136)
    rr = row_rel(o, ref)
    assert torch.isfinite(o).all()
    assert rr.max().item() < 0.12, rr.max().item()


def test_k_centering_reduces_offset_error(ext):
    """K mean-centering must (a) be softmax-shift-exact on offset-free data
    (equal within quantization noise) and (b) substantially reduce error
    when K carries large per-channel offsets (the RoPE'd-key regime)."""
    T, N, prefix = 128, 2048, 1920
    q, k, v = rand_qkv(T, N, seed=21)
    # add strong per-channel offsets to K (same for every kv position)
    offs = torch.rand(1, k.shape[1], HEAD_DIM, device=k.device) * 8 - 4
    k_off = (k.float() + offs).to(torch.bfloat16)
    ref = ref_attention(q, k_off, v, prefix)
    # fp8 QK: the mode where K-offset damage exists (int8 QK has the range
    # to absorb offsets; measured: with int8 QK both variants sit at the
    # PV-fp8 noise floor and centering has nothing left to fix)
    rr_plain = row_rel(
        run_kernel(ext, q, k_off, v, prefix, center_k=False, qk_i8=False, rotate=False), ref
    )
    rr_cent = row_rel(
        run_kernel(ext, q, k_off, v, prefix, center_k=True, qk_i8=False, rotate=False), ref
    )
    # offset-free floor: the same shape on plain data with centering on
    q2, k2, v2 = rand_qkv(T, N, seed=22)
    ref2 = ref_attention(q2, k2, v2, prefix)
    rr_floor = row_rel(
        run_kernel(ext, q2, k2, v2, prefix, center_k=True, qk_i8=False, rotate=False), ref2
    )
    # centering must (a) strictly beat the uncentered run on offset-heavy K,
    # and (b) remove the offset excess almost entirely: the centered error
    # lands near the offset-free fp8 noise floor (P/V-fp8 + Q-quant), which
    # centering cannot touch. Measured here: 0.054 -> 0.0395 with floor
    # 0.037 -- the K-offset excess is gone.
    assert rr_cent.mean().item() < 0.85 * rr_plain.mean().item(), (
        rr_cent.mean().item(),
        rr_plain.mean().item(),
    )
    assert rr_cent.mean().item() < 1.3 * rr_floor.mean().item(), (
        rr_cent.mean().item(),
        rr_floor.mean().item(),
    )
    assert rr_floor.mean().item() < 0.06, rr_floor.mean().item()


def test_v_outlier_channels_handled(ext):
    """Massive-activation V channels (the real-tensor regime measured in
    the numerics program) must NOT degrade the rest of the output: with V
    mean-centering + per-tile V scales, error on outlier-V data stays near
    the plain-data floor.  Also: a huge V in one kv TILE must not degrade
    contributions from other tiles (per-tile scales)."""
    T, N, prefix = 128, 2048, 1920
    q, k, v = rand_qkv(T, N, seed=31)
    # (a) massive channel OFFSETS (mean-centering's job)
    offs = torch.zeros(1, v.shape[1], HEAD_DIM, device=v.device)
    offs[..., :8] = 40.0  # 8 huge-mean channels
    v_off = (v.float() + offs).to(torch.bfloat16)
    ref = ref_attention(q, k, v_off, prefix)
    rr_off = row_rel(run_kernel(ext, q, k, v_off, prefix), ref)
    # (b) one huge-MAGNITUDE kv tile (per-tile scales' job)
    v_tile = v.clone()
    v_tile[:64] *= 50.0  # first kv tile blows up
    ref_t = ref_attention(q, k, v_tile, prefix)
    rr_tile = row_rel(run_kernel(ext, q, k, v_tile, prefix), ref_t)
    # plain-data floor on the same shape
    ref_p = ref_attention(q, k, v, prefix)
    rr_plain = row_rel(run_kernel(ext, q, k, v, prefix), ref_p)
    assert rr_off.mean().item() < 2.0 * rr_plain.mean().item(), (
        rr_off.mean().item(),
        rr_plain.mean().item(),
    )
    assert rr_tile.mean().item() < 2.0 * rr_plain.mean().item(), (
        rr_tile.mean().item(),
        rr_plain.mean().item(),
    )


def test_int8_qk_beats_fp8_qk(ext):
    """INT8 QK (exact int32 dots, ~0.4% input rounding) must substantially
    beat e4m3 QK (~3% per element).  Isolated in bf16-PV mode: with fp8 PV
    active, PV noise (~3.5%) dominates the full-kernel row error and masks
    the QK difference (measured 0.0421 vs 0.0439) — itself a load-bearing
    finding recorded in the README."""
    T, N, prefix = 128, 4096, 3968
    q, k, v = rand_qkv(T, N, seed=41)
    ref = ref_attention(q, k, v, prefix)
    rr_i8 = row_rel(run_kernel(ext, q, k, v, prefix, bf16_heads=range(8), qk_i8=True), ref)
    rr_f8 = row_rel(run_kernel(ext, q, k, v, prefix, bf16_heads=range(8), qk_i8=False), ref)
    assert rr_i8.mean().item() < 0.5 * rr_f8.mean().item(), (
        rr_i8.mean().item(),
        rr_f8.mean().item(),
    )


def test_hadamard_rotation_exact_and_helpful(ext):
    """Rotation must (a) leave outputs within quantization noise on plain
    data (scores are exactly invariant pre-quantization) and (b) fix
    within-row outlier structure (RoPE'd-key regime) that linear int8
    cannot otherwise handle."""
    T, N, prefix = 128, 2048, 1920
    q, k, v = rand_qkv(T, N, seed=51)
    # within-row outliers: a few dims carry 30x the energy, same dims every row
    boost = torch.ones(1, 1, HEAD_DIM, device=q.device)
    boost[..., ::37] = 30.0
    q_o = (q.float() * boost).to(torch.bfloat16)
    k_o = (k.float() * boost).to(torch.bfloat16)
    ref = ref_attention(q_o, k_o, v, prefix)
    # isolate under bf16 PV (fp8-PV noise otherwise floors the comparison,
    # same masking as in the int8-vs-fp8 QK test)
    rr_rot = row_rel(run_kernel(ext, q_o, k_o, v, prefix, bf16_heads=range(8), rotate=True), ref)
    rr_no = row_rel(run_kernel(ext, q_o, k_o, v, prefix, bf16_heads=range(8), rotate=False), ref)
    assert rr_rot.mean().item() < 0.6 * rr_no.mean().item(), (
        rr_rot.mean().item(),
        rr_no.mean().item(),
    )
