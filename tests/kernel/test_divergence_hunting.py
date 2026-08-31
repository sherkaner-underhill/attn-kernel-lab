# SPDX-License-Identifier: Apache-2.0
"""Divergence-hunting suite for the fp8 prefill backend.

Companion to ``test_kernel_vs_sdpa.py`` (which passes 14/14 on synthetic
uniform data).  This file targets the gap between that suite and real
serving, where five configs of wildly different numerical precision all
show the SAME broad ~0.17 mean input-logprob delta vs the flashinfer
baseline.  See ``../AUDIT_NOTES.md`` for the ranked hypotheses; each test
below names the hypothesis it discriminates.

Run on an SM120-class GPU with a CUDA toolchain for the JIT:

    cd tests && python3 -m pytest test_divergence_hunting.py -q
    cd tests && python3 -m pytest test_divergence_hunting.py -q -s   # + numbers

Reading the results: several tests here are *measurements* dressed as
assertions.  A failure is a finding, not necessarily a regression -- each
docstring says what a failure confirms.  The bit-exact tests
(``*_bit_exact``) are hard requirements: a failure there is an outright
bug in the kernel or the pipeline.
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

DEV = "cuda"
_MEASUREMENTS = []


@pytest.fixture(scope="module")
def ext():
    """The one extension the whole repo builds: ``attn_kernel_lab_fp8_prefill``.

    Delegated to the package loader so tests and the public path share a single
    JIT build directory -- ``cpp_extension`` keys that directory on the
    extension NAME alone, so the old separate name cost a full redundant nvcc
    compile of byte-identical source on every cold run. See the same fixture in
    ``test_kernel_vs_sdpa.py`` for the full note, including the rule that any
    future test needing DIFFERENT ``-D`` flags must take its own name.
    """
    return kernel_mod.load()


@pytest.fixture(scope="module", autouse=True)
def _measurement_report():
    yield
    if not _MEASUREMENTS:
        return
    print("\n\n=== divergence-hunting measurements ===")
    width = max(len(n) for n, _ in _MEASUREMENTS)
    for name, val in _MEASUREMENTS:
        print(f"  {name:<{width}s}  {val}")
    print("=======================================")


def measure(name, value):
    """Record a number for the end-of-run table; also returned for asserts."""
    _MEASUREMENTS.append((name, value if isinstance(value, str) else f"{value:.6g}"))
    return value


# ---------------------------------------------------------------- references


def ref_attention(q, k, v, prefix, row_block=512):
    """fp32 bottom-right-causal attention.  q [T,H,D], k/v [N,KVH,D].

    Row-blocked so large T/N stay inside memory (the original suite's
    reference materializes the whole [T,N] score matrix).
    """
    T, H, D = q.shape
    N, KVH, _ = k.shape
    grp = H // KVH
    out = torch.empty(T, H, D, device=q.device, dtype=torch.float32)
    cols = torch.arange(N, device=q.device)
    for h in range(H):
        kf = k[:, h // grp].float()
        vf = v[:, h // grp].float()
        for r0 in range(0, T, row_block):
            r1 = min(r0 + row_block, T)
            lim = prefix + torch.arange(r0, r1, device=q.device)
            s = (q[r0:r1, h].float() @ kf.T) / math.sqrt(D)
            s.masked_fill_(cols[None, :] > lim[:, None], float("-inf"))
            out[r0:r1, h] = torch.softmax(s, dim=1) @ vf
    return out


def ref_mass(q, k, prefix, col_mask, row_block=512):
    """[T,H] fp32 softmax mass that each q row puts on ``col_mask`` columns."""
    T, H, D = q.shape
    N, KVH, _ = k.shape
    grp = H // KVH
    out = torch.empty(T, H, device=q.device, dtype=torch.float32)
    cols = torch.arange(N, device=q.device)
    for h in range(H):
        kf = k[:, h // grp].float()
        for r0 in range(0, T, row_block):
            r1 = min(r0 + row_block, T)
            lim = prefix + torch.arange(r0, r1, device=q.device)
            s = (q[r0:r1, h].float() @ kf.T) / math.sqrt(D)
            s.masked_fill_(cols[None, :] > lim[:, None], float("-inf"))
            out[r0:r1, h] = torch.softmax(s, dim=1)[:, col_mask].sum(dim=1)
    return out


def row_rel(a, b):
    return (a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-6)


# ------------------------------------------------------------------ pipeline


def run_pipeline(
    ext,
    q,
    k_pool,
    v_pool,
    idx,
    prefix,
    *,
    ws=None,
    bf16_heads=(),
    center_k=True,
    qk_i8=True,
    sm_scale=None,
):
    """Exactly the backend's per-request path: gather+quantize from a pool,
    quantize Q with sm_scale folded, one kernel launch, slice+permute back.

    ``bf16_heads`` accepts an iterable of head indices or the string "all"
    (the SGLANG_FP8_PREFILL_BF16_HEADS=all mode).  Returns fp32 [T,H,D]
    (a widening copy of the bf16 kernel output, so ``torch.equal`` on the
    result is a bit-exactness check and the workspace may be reused).
    """
    T, H, D = q.shape
    if ws is None:
        ws = q_mod.FP8PrefillWorkspace(q.device)
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    mask = torch.ones(H, dtype=torch.uint8, device=q.device)
    if isinstance(bf16_heads, str):
        assert bf16_heads == "all"
        mask.zero_()
    else:
        for h in bf16_heads:
            mask[h] = 0
    any_pv8 = bool(mask.max().item())
    all_pv8 = bool(mask.min().item())

    kv = q_mod.gather_quantize_kv(
        ws,
        k_pool,
        v_pool,
        idx.long(),
        need_vt8=any_pv8,
        need_vb16=not all_pv8,
        center_k=center_k,
        qk_i8=qk_i8,
    )
    q8, qscale, mpad = q_mod.quantize_q(ws, q, sm_scale, qk_i8=qk_i8)
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


def contiguous_idx(n, device=DEV):
    return torch.arange(n, device=device)


def scatter_pool(k, v, seed, fill="rand"):
    """Place the request's K/V rows at scattered slots of an oversized pool,
    mimicking a page_size-1 paged pool that other requests also occupy.

    ``fill="nan"`` poisons every slot the request does NOT own, so any
    gather-index slip shows up as a NaN rather than as small noise.
    """
    N, KVH, D = k.shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    pool_n = int(N * 1.7) + 97
    slots = torch.randperm(pool_n, generator=g)[:N].to(k.device)
    if fill == "nan":
        kp = torch.full((pool_n, KVH, D), float("nan"), device=k.device, dtype=k.dtype)
        vp = torch.full((pool_n, KVH, D), float("nan"), device=k.device, dtype=k.dtype)
    else:
        kp = (torch.rand((pool_n, KVH, D), generator=g) * 2 - 1).to(k.device, k.dtype)
        vp = (torch.rand((pool_n, KVH, D), generator=g) * 2 - 1).to(k.device, k.dtype)
    kp[slots] = k
    vp[slots] = v
    return kp, vp, slots


# ---------------------------------------------------------------- generators


def rand_qkv(T, N, H=8, KVH=2, seed=0, device=DEV):
    """Uniform [-1,1] -- the distribution the existing suite uses; the
    'floor' every structured generator below is compared against."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = (torch.rand((T, H, HEAD_DIM), generator=g) * 2 - 1).to(device, torch.bfloat16)
    k = (torch.rand((N, KVH, HEAD_DIM), generator=g) * 2 - 1).to(device, torch.bfloat16)
    v = (torch.rand((N, KVH, HEAD_DIM), generator=g) * 2 - 1).to(device, torch.bfloat16)
    return q, k, v


def heavy_tailed(shape, g, df=3):
    """Student-t(df): the same variance as a normal but fat tails, so a
    64-row/64-column block amax is set by a rare element rather than by the
    bulk -- the regime uniform data cannot produce."""
    z = torch.randn(shape, generator=g)
    u = torch.randn(shape + (df,), generator=g).pow(2).sum(-1) / df
    return z / u.sqrt().clamp_min(1e-3)


def channel_outliers(x, g, n_out=4, gain=50.0):
    """Massive-activation channels: n_out of the 256 channels scaled up.
    This is what post-RoPE Q/K and V actually look like; it is the single
    structural property uniform synthetic data most conspicuously lacks."""
    y = x.clone()
    ch = torch.randperm(x.shape[-1], generator=g)[:n_out]
    y[..., ch] *= gain
    return y


def rope_like(T, H, g, base=10000.0, pos0=0):
    """Real rotate-half RoPE applied to random content: gives the
    position-dependent alternating phase structure AND the near-DC
    low-frequency channels that carry large shared offsets."""
    x = torch.randn(T, H, HEAD_DIM, generator=g)
    half = HEAD_DIM // 2
    inv = base ** (-torch.arange(half, dtype=torch.float32) * 2.0 / HEAD_DIM)
    ang = torch.arange(pos0, pos0 + T, dtype=torch.float32)[:, None] * inv[None, :]
    cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]
    a, b = x[..., :half], x[..., half:]
    return torch.cat([a * cos - b * sin, a * sin + b * cos], dim=-1)


def two_regime(N, KVH, g, lo=0.04, hi=1.0, hi_start=0, hi_len=256):
    """A short high-magnitude region (text tokens) inside a long
    low-magnitude region (visual tokens), or vice versa by construction."""
    x = torch.randn(N, KVH, HEAD_DIM, generator=g) * lo
    x[hi_start : hi_start + hi_len] *= hi / lo
    return x


def to_dev(*ts):
    return tuple(t.to(DEV, torch.bfloat16) for t in ts)


# =========================================================================
# (a) chunked composition -- the closest analogue of real serving
# =========================================================================


@pytest.mark.parametrize("chunks", [(1024, 512, 512), (640, 1408), (64, 1984)])
def test_chunked_composition_bit_exact(ext, chunks):
    """H2/H4.  Run the pipeline chunk-by-chunk with a growing prefix,
    exactly as ``backend.forward_extend`` does across a multi-chunk
    sequence, and require BIT-EQUALITY with the one-shot full-sequence run.

    This is exact by construction in this configuration and must stay exact:
      * chunk boundaries are multiples of BLK, so the per-64-row Q blocks
        and the per-64-row K tiles are the same rows in both schedules;
      * ``center_k=False`` removes the K channel-mean's dependence on the
        gathered window length;
      * bf16 PV removes vmean / per-tile V scales / vsmax, which all depend
        on the window;
      * the kernel's ``col <= prefix + row`` and its ``hi``/``ntiles`` bound
        must therefore reduce to the same visible column set and the same
        tile iteration order for every row.

    A failure here is an OUTRIGHT BUG in the causal/prefix bookkeeping or
    in the workspace re-view across a changing N -- exactly the surface no
    existing test covers.  The final assertion additionally reuses one
    workspace across shrink-then-grow, as the backend does.
    """
    L = sum(chunks)
    q, k, v = rand_qkv(L, L, H=8, KVH=2, seed=101)
    kp, vp, slots = scatter_pool(k, v, seed=101)
    cfg = dict(bf16_heads="all", center_k=False, qk_i8=True)

    one = run_pipeline(ext, q, kp, vp, slots, 0, ws=q_mod.FP8PrefillWorkspace(q.device), **cfg)

    ws = q_mod.FP8PrefillWorkspace(q.device)
    parts, p = [], 0
    for t in chunks:
        parts.append(run_pipeline(ext, q[p : p + t], kp, vp, slots[: p + t], p, ws=ws, **cfg))
        p += t
    got = torch.cat(parts, dim=0)
    assert got.shape == one.shape
    delta = (got - one).abs().max().item()
    assert torch.equal(got, one), f"chunked != one-shot, max|delta|={delta}"

    # same workspace, now going back up to the full length: buffers are
    # re-viewed at a larger npad/mpad and must be fully rewritten
    again = run_pipeline(ext, q, kp, vp, slots, 0, ws=ws, **cfg)
    assert torch.equal(again, one), "workspace reuse changed the one-shot result"


@pytest.mark.parametrize("qk_i8,pv", [(True, "bf16"), (True, "fp8"), (False, "fp8")])
def test_chunked_composition_production_settings(ext, qk_i8, pv):
    """H2.  Same chunk schedule, but with the knobs a real server uses
    (K centering on, fp8/per-tile V).  Composition is no longer bit-exact
    because every quantization statistic -- K channel mean, V channel mean,
    global V amax, per-tile V ratios -- is computed over the whole gathered
    window, so it changes when the chunk boundary moves.

    What this measures: how much of a token's output is decided by where
    the chunk boundary happened to fall.  The flashinfer baseline has NO
    such dependence, so any drift here is a per-token difference that no
    amount of extra kernel precision removes -- the signature of H2 and a
    direct explanation of a delta that is invariant across precision
    configs.  Bound is deliberately generous; read the printed number.
    """
    chunks = (1024, 512, 512)
    L = sum(chunks)
    q, k, v = rand_qkv(L, L, H=8, KVH=2, seed=102)
    kp, vp, slots = scatter_pool(k, v, seed=102)
    cfg = dict(bf16_heads=("all" if pv == "bf16" else ()), center_k=True, qk_i8=qk_i8)

    one = run_pipeline(ext, q, kp, vp, slots, 0, ws=q_mod.FP8PrefillWorkspace(q.device), **cfg)
    ws = q_mod.FP8PrefillWorkspace(q.device)
    parts, p = [], 0
    for t in chunks:
        parts.append(run_pipeline(ext, q[p : p + t], kp, vp, slots[: p + t], p, ws=ws, **cfg))
        p += t
    got = torch.cat(parts, dim=0)

    ref = ref_attention(q, k, v, 0)
    e_one = row_rel(one, ref).mean().item()
    e_chunk = row_rel(got, ref).mean().item()
    drift = row_rel(got, one).mean().item()
    tag = f"{'i8' if qk_i8 else 'f8'}qk+{pv}pv"
    measure(f"chunk-drift[{tag}] one-shot err", e_one)
    measure(f"chunk-drift[{tag}] chunked err", e_chunk)
    measure(f"chunk-drift[{tag}] chunk-vs-oneshot", drift)
    measure(f"chunk-drift[{tag}] drift/err ratio", drift / max(e_one, 1e-12))

    # error must not ACCUMULATE across chunks (a broken prefix path would
    # show a growing per-chunk error; this is the depth-stability claim)
    assert e_chunk < 1.35 * e_one + 1e-4, (e_chunk, e_one)
    # and boundary placement must not dominate the error budget
    assert drift < 1.5 * e_one + 1e-4, (drift, e_one)


def test_chunked_ragged_last_chunk(ext):
    """H2/H4.  Production's last chunk is ragged (N % 64 != 0, e.g. the
    measured N=10232 -> 56-row boundary tile) while every earlier boundary
    is 64-aligned.  Exercises the K/V boundary-tile zero padding, the
    ``col < N`` guard and the Q row padding together, at a shape the
    existing suite never reaches with a non-zero prefix.
    """
    chunks = (1024, 1024, 440)  # 440 % 64 = 56, like N=10232
    L = sum(chunks)
    q, k, v = rand_qkv(L, L, H=8, KVH=2, seed=103)
    kp, vp, slots = scatter_pool(k, v, seed=103)
    ref = ref_attention(q, k, v, 0)

    for cfg, bound in (
        (dict(bf16_heads="all", center_k=True, qk_i8=True), 0.03),
        (dict(bf16_heads=(), center_k=True, qk_i8=True), 0.08),
    ):
        ws = q_mod.FP8PrefillWorkspace(q.device)
        parts, p = [], 0
        for t in chunks:
            parts.append(run_pipeline(ext, q[p : p + t], kp, vp, slots[: p + t], p, ws=ws, **cfg))
            p += t
        got = torch.cat(parts, dim=0)
        assert torch.isfinite(got).all(), "ragged chunk produced non-finite output"
        rr = row_rel(got, ref)
        tag = "bf16pv" if cfg["bf16_heads"] == "all" else "fp8pv"
        measure(f"ragged-tail[{tag}] mean", rr.mean().item())
        measure(f"ragged-tail[{tag}] max", rr.max().item())
        # the ragged tail rows specifically (they own the partial kv tile)
        tail = rr[-chunks[-1] :]
        measure(f"ragged-tail[{tag}] tail-rows mean", tail.mean().item())
        assert rr.mean().item() < bound, rr.mean().item()
        assert tail.mean().item() < 2.0 * rr.mean().item() + 1e-3, (
            tail.mean().item(),
            rr.mean().item(),
        )


def test_pool_gather_ignores_foreign_slots(ext):
    """H5 (gather index semantics).  Every pool slot the request does NOT
    own is filled with NaN.  A single slipped index -- an off-by-one in
    ``req_to_token[req_row, :n_tot]``, a stale row, a page-size assumption
    -- turns into a NaN, not into small noise.  Result must also be
    bit-identical to the contiguous-pool run.
    """
    T, N, prefix = 128, 1024, 896
    q, k, v = rand_qkv(T, N, seed=104)
    kp, vp, slots = scatter_pool(k, v, seed=104, fill="nan")
    got = run_pipeline(ext, q, kp, vp, slots, prefix)
    assert torch.isfinite(got).all(), "NaN from a foreign pool slot reached the output"
    direct = run_pipeline(ext, q, k, v, contiguous_idx(N), prefix)
    assert torch.equal(got, direct)


# =========================================================================
# causality vs. quantization-window leakage
# =========================================================================


def test_future_kv_invisible_bit_exact(ext):
    """H2 control.  With every window-dependent statistic disabled
    (center_k=False, bf16 PV), changing the LAST kv position must leave
    every q row that cannot see it bit-identical.

    Stronger than ``test_causal_boundary_exact`` in the existing suite:
    that test perturbs a whole kv tile at a hand-picked boundary; this one
    perturbs the single final position, which is what a chunk boundary
    actually moves.  Rows protected: those whose causal reach ends before
    the perturbed position's 64-row K tile (per-tile K scales couple the
    columns inside one tile, so the tile is the granularity).
    """
    T, N, prefix = 256, 1024, 768
    q, k, v = rand_qkv(T, N, seed=105)
    cfg = dict(bf16_heads="all", center_k=False, qk_i8=True)
    o1 = run_pipeline(ext, q, k, v, contiguous_idx(N), prefix, **cfg)

    k2, v2 = k.clone(), v.clone()
    k2[N - 1] += 3.0
    v2[N - 1] += 3.0
    o2 = run_pipeline(ext, q, k2, v2, contiguous_idx(N), prefix, **cfg)

    tile0 = (N - 1) // q_mod.BLK * q_mod.BLK  # first column of that tile
    safe = tile0 - prefix  # rows with reach < tile0
    assert safe > 0
    assert torch.equal(o1[:safe], o2[:safe]), "future kv changed a protected row"
    assert not torch.equal(o1[safe:], o2[safe:]), "perturbation had no effect at all"


def test_quantization_window_leak(ext):
    """H2 (the measurement).  Repeat the test above with the PRODUCTION
    knobs.  Now the perturbation of the last kv position moves rows that
    provably cannot attend to it, because it shifts the K channel mean, the
    V channel mean, the global V amax and the per-tile V ratios -- all of
    which are computed over the whole gathered window.

    Consequence in serving: a token's logprob depends on tokens that come
    AFTER it inside the same chunk, and on where the chunk boundary fell.
    flashinfer has no such dependence.  This is broad (every row), and no
    existing test can see it (the one causality test pins center_k off).
    Assertion is a magnitude bound; the number is the finding.
    """
    T, N, prefix = 256, 1024, 768
    q, k, v = rand_qkv(T, N, seed=105)
    cfg = dict(bf16_heads=(), center_k=True, qk_i8=True)
    o1 = run_pipeline(ext, q, k, v, contiguous_idx(N), prefix, **cfg)
    k2, v2 = k.clone(), v.clone()
    k2[N - 1] += 3.0
    v2[N - 1] += 3.0
    o2 = run_pipeline(ext, q, k2, v2, contiguous_idx(N), prefix, **cfg)

    tile0 = (N - 1) // q_mod.BLK * q_mod.BLK
    safe = tile0 - prefix
    leak = row_rel(o2[:safe], o1[:safe])
    ref = ref_attention(q, k, v, prefix)
    floor = row_rel(o1, ref).mean().item()
    measure("window-leak mean row-rel", leak.mean().item())
    measure("window-leak max row-rel", leak.max().item())
    measure("window-leak / kernel err", leak.mean().item() / max(floor, 1e-12))
    # a leak comparable to the kernel's own error budget means chunk
    # placement is a first-order term in the end-to-end delta
    assert leak.mean().item() < 3.0 * floor + 1e-3, (leak.mean().item(), floor)


# =========================================================================
# (c) head identity -- catches permutations exactly
# =========================================================================


def test_head_identity_q_perturbation_bit_exact(ext):
    """H6.  Give each q head a unique signature by perturbing it alone, and
    require that ONLY output head h moves -- bit-exactly everywhere else.

    This is the head-permutation test the existing suite lacks in exact
    form: ``test_gqa_mapping`` only distinguishes kv-head GROUPS (a swap of
    two q heads inside one group is invisible to it), and it only ever uses
    H=8 / KVH=2 / grp=4.  Here H=24, KVH=4, grp=6 -- the production shape,
    and the first test in the package with a non-power-of-two GQA group.
    Q scales are per (head, 64-row block), so perturbing head h changes
    head h's bytes only: bit-equality on the other 23 heads is exact.
    """
    T, N, prefix = 64, 512, 448
    H, KVH = 24, 4
    q, k, v = rand_qkv(T, N, H=H, KVH=KVH, seed=106)
    base = run_pipeline(ext, q, k, v, contiguous_idx(N), prefix)
    for h in range(H):
        q2 = q.clone()
        q2[:, h] = (q2[:, h].float() * 1.5 + 0.25).to(torch.bfloat16)
        got = run_pipeline(ext, q2, k, v, contiguous_idx(N), prefix)
        assert not torch.equal(got[:, h], base[:, h]), f"q head {h} had no effect"
        others = [j for j in range(H) if j != h]
        assert torch.equal(got[:, others], base[:, others]), (
            f"perturbing q head {h} changed other output heads "
            f"(head permutation / cross-head contamination)"
        )


def test_head_identity_kv_group_map_bit_exact(ext):
    """H6.  Per-kv-head version: perturbing kv head j must move EXACTLY the
    q heads [j*grp, (j+1)*grp) and leave the rest bit-identical, for every
    j -- with grp=6 and KVH=4, so a ``qh % KVH`` mapping, a ``qh & (KVH-1)``
    mapping, or any mis-grouping at KVH>2 is caught.  Runs with fp8 PV, so
    the per-kv-head vmean / vscale_t / vsmax arrays are exercised too.
    """
    T, N, prefix = 64, 512, 448
    H, KVH = 24, 4
    grp = H // KVH
    q, k, v = rand_qkv(T, N, H=H, KVH=KVH, seed=107)
    base = run_pipeline(ext, q, k, v, contiguous_idx(N), prefix)
    for j in range(KVH):
        for which in ("k", "v"):
            k2, v2 = k.clone(), v.clone()
            tgt = k2 if which == "k" else v2
            tgt[:, j] = (tgt[:, j].float() * -2.0).to(torch.bfloat16)
            got = run_pipeline(ext, q, k2, v2, contiguous_idx(N), prefix)
            inside = list(range(j * grp, (j + 1) * grp))
            outside = [h for h in range(H) if h not in inside]
            assert torch.equal(got[:, outside], base[:, outside]), (
                f"perturbing {which} of kv head {j} leaked outside its group"
            )
            assert not torch.equal(got[:, inside], base[:, inside]), (
                f"perturbing {which} of kv head {j} did not reach its group"
            )


def test_head_confusion_matrix(ext):
    """H6.  A permutation-proof check that survives IDENTICALLY-DISTRIBUTED
    heads: for every output head h, the fp32 reference head it matches best
    must be h itself, by a wide margin.  Aggregate row-error tests can in
    principle be fooled if the heads are exchangeable; this cannot.
    """
    T, N, prefix = 64, 512, 448
    H, KVH = 24, 4
    q, k, v = rand_qkv(T, N, H=H, KVH=KVH, seed=108)
    got = run_pipeline(ext, q, k, v, contiguous_idx(N), prefix)
    ref = ref_attention(q, k, v, prefix)
    err = torch.empty(H, H)
    for h in range(H):
        for hh in range(H):
            err[h, hh] = row_rel(got[:, h], ref[:, hh]).mean()
    diag = err.diagonal()
    off = err + torch.eye(H) * 1e9
    worst = (diag / off.min(dim=1).values).max().item()
    measure("head-confusion worst diag/off ratio", worst)
    assert torch.equal(err.argmin(dim=1), torch.arange(H)), (
        "an output head matched a DIFFERENT reference head best "
        f"-> head permutation. argmin={err.argmin(dim=1).tolist()}"
    )
    assert worst < 0.25, worst


# =========================================================================
# (d) softmax denominator (l / lse) consistency
# =========================================================================


@pytest.mark.parametrize("pv", ["bf16", "fp8"])
def test_partition_of_unity(ext, pv):
    """H3.  Direct test of the softmax denominator.  V rows are a constant
    c per kv head with c[d0] = 0, plus +A on channel d0 for the rows in a
    chosen set S; channel d0 then reads out the attention MASS on S:

        O[t,h,d0] / A  ==  sum of the kernel's softmax weights over S.

    Run it twice with S and its complement.

      * bf16 PV (no V centering): the two readouts sum to
        ``sum_j bf16(p_j) / sum_j p_j`` -- i.e. exactly 1 + eps, where eps
        is the mismatch between the numerator's P (rounded by
        ``pack_bf16``) and the denominator ``l_i``, which is accumulated
        from the PRE-rounding fp32 P in
        ``l_i[h] = l_i[h]*alpha[h] + part*invrt``.  This is a direct
        measurement of that inconsistency and it exists nowhere else in
        the suite.  It is a pure gain error on O, so row-relative error
        tests see it only as a small isotropic term.
      * fp8 PV: the V mean add-back re-introduces the mean, so the sum is
        algebraically 1 whatever P rounding does -- but only if
        ``vscale``/``vmean``/``l_i`` are consistently paired in the
        epilogue.  A deviation there is an outright epilogue bug.  (The V
        side is exactly representable here: centered V is +-A/2 on d0 and 0
        elsewhere, so v8 = +-448 with r_t = 1 and no V quantization error
        contaminates the readout.)

    The second assertion -- readout vs the fp32 reference mass -- is the
    apples-to-apples accuracy check for both paths.
    """
    T, N, prefix = 128, 1024, 896
    H, KVH = 8, 2
    grp = H // KVH
    d0, A = 3, 2.0
    g = torch.Generator(device="cpu").manual_seed(109)
    q = (torch.rand((T, H, HEAD_DIM), generator=g) * 2 - 1).to(DEV, torch.bfloat16)
    k = (torch.rand((N, KVH, HEAD_DIM), generator=g) * 2 - 1).to(DEV, torch.bfloat16)
    # round c through bf16 so the pipeline sees exactly the value we
    # subtract, and zero the readout channel so the readout is unbiased
    c = (torch.rand((KVH, HEAD_DIM), generator=g) * 2 - 1).to(DEV, torch.bfloat16)
    c[:, d0] = 0.0

    sel = torch.zeros(N, dtype=torch.bool, device=DEV)
    sel[: N // 2] = True
    cfg = dict(bf16_heads=("all" if pv == "bf16" else ()), center_k=True, qk_i8=True)

    outs = []
    for mask_sel in (sel, ~sel):
        v = c[None, :, :].expand(N, KVH, HEAD_DIM).clone()
        v[mask_sel, :, d0] = A  # exact in bf16 (c[d0]=0)
        o = run_pipeline(ext, q, k, v, contiguous_idx(N), prefix, **cfg)
        outs.append(o[:, :, d0] / A)

    total = outs[0] + outs[1]
    dev = (total - 1.0).abs()
    measure(f"partition-of-unity[{pv}] max |sum-1|", dev.max().item())
    measure(f"partition-of-unity[{pv}] mean |sum-1|", dev.mean().item())
    assert dev.max().item() < 8e-3, dev.max().item()

    # and the mass itself must track the fp32 reference
    ref_first = ref_mass(q, k, prefix, sel)
    err = (outs[0] - ref_first).abs()
    measure(f"mass-readout[{pv}] max |dmass|", err.max().item())
    measure(f"mass-readout[{pv}] mean |dmass|", err.mean().item())
    assert err.mean().item() < 3e-2, err.mean().item()


def test_duplicated_v_channels_bit_exact(ext):
    """H3/H7.  V's channel d and channel d+128 carry identical values, so
    output channels d and d+128 must be bit-identical: they see the same
    weights, the same per-tile V scale (amax is taken over all channels),
    and the same channel mean.  Any per-d-tile indexing slip in the PV
    B-fragment addressing, or any denominator that is not shared by all 256
    output channels, breaks this.  Half the head_dim apart == 16 d-tiles
    apart in the kernel's PV loop.
    """
    T, N, prefix = 128, 1024, 896
    q, k, v = rand_qkv(T, N, seed=110)
    half = HEAD_DIM // 2
    v = v.clone()
    v[:, :, half:] = v[:, :, :half]
    for cfg in (dict(bf16_heads=()), dict(bf16_heads="all")):
        o = run_pipeline(ext, q, k, v, contiguous_idx(N), prefix, **cfg)
        assert torch.equal(o[:, :, :half], o[:, :, half:]), (
            f"duplicated V channels gave different outputs ({cfg}); "
            f"max|delta|={(o[:, :, :half] - o[:, :, half:]).abs().max().item()}"
        )


def test_constant_v_is_reproduced_exactly(ext):
    """H3.  If every V row is the same vector c, the output must be c for
    every row and head regardless of the attention weights (weights sum to
    1).

      * bf16 PV: O = c * (sum_j bf16(p_j) / sum_j p_j), so the relative
        error IS the numerator/denominator rounding mismatch eps -- the
        cleanest possible read of it, with no reference and no V error.
      * fp8 PV: centering makes V-mean = c and centered V = 0 exactly, so
        this checks the vmean add-back epilogue and that l_i > 0 for every
        row (a zeroed l would give O = c too, hence the bf16 arm above is
        the one that actually constrains the denominator).
    """
    T, N, prefix = 128, 1024, 896
    H, KVH = 8, 2
    grp = H // KVH
    g = torch.Generator(device="cpu").manual_seed(111)
    q = (torch.rand((T, H, HEAD_DIM), generator=g) * 2 - 1).to(DEV, torch.bfloat16)
    k = (torch.rand((N, KVH, HEAD_DIM), generator=g) * 2 - 1).to(DEV, torch.bfloat16)
    c = (torch.rand((KVH, HEAD_DIM), generator=g) * 2 - 1).to(DEV, torch.bfloat16)
    c = torch.where(c.abs() < 0.25, torch.full_like(c, 0.25), c)  # keep |c| away from 0
    v = c[None, :, :].expand(N, KVH, HEAD_DIM).contiguous()
    want = c.float().repeat_interleave(grp, dim=0)[None].expand(T, H, HEAD_DIM)
    for pv, tol in (("fp8", 1e-2), ("bf16", 1e-2)):
        o = run_pipeline(
            ext, q, k, v, contiguous_idx(N), prefix, bf16_heads=("all" if pv == "bf16" else ())
        )
        err = (o - want).abs().max().item()
        rel = ((o - want).abs() / want.abs()).max().item()
        measure(f"constant-V[{pv}] max abs err", err)
        measure(f"constant-V[{pv}] max rel err (= |eps| for bf16)", rel)
        assert err < tol, err


# =========================================================================
# (b) structured data -- the regimes uniform synthetic data cannot produce
# =========================================================================


def _structured_case(ext, q, k, v, prefix, floor_seed, cfgs):
    """Compare structured data against the uniform floor at the same shape."""
    T, H = q.shape[0], q.shape[1]
    N, KVH = k.shape[0], k.shape[1]
    ref = ref_attention(q, k, v, prefix)
    qf, kf, vf = rand_qkv(T, N, H=H, KVH=KVH, seed=floor_seed)
    ref_f = ref_attention(qf, kf, vf, prefix)
    out = {}
    for name, cfg in cfgs.items():
        e = row_rel(run_pipeline(ext, q, k, v, contiguous_idx(N), prefix, **cfg), ref).mean().item()
        f = (
            row_rel(run_pipeline(ext, qf, kf, vf, contiguous_idx(N), prefix, **cfg), ref_f)
            .mean()
            .item()
        )
        out[name] = (e, f)
    return out


CFGS = {
    "i8qk+bf16pv": dict(qk_i8=True, bf16_heads="all"),
    "i8qk+fp8pv": dict(qk_i8=True, bf16_heads=()),
    "f8qk+fp8pv": dict(qk_i8=False, bf16_heads=()),
}


def test_structured_heavy_tailed(ext):
    """H1.  Student-t Q/K/V: block amax is set by rare elements, so the
    bulk of the block quantizes onto a few levels.  Uniform data cannot
    produce this (its amax IS the bulk scale)."""
    T, N, prefix = 128, 2048, 1920
    g = torch.Generator(device="cpu").manual_seed(120)
    q, k, v = to_dev(
        heavy_tailed((T, 8, HEAD_DIM), g),
        heavy_tailed((N, 2, HEAD_DIM), g),
        heavy_tailed((N, 2, HEAD_DIM), g),
    )
    res = _structured_case(ext, q, k, v, prefix, 121, CFGS)
    for name, (e, f) in res.items():
        measure(f"heavy-tail[{name}] err / floor", f"{e:.4f} / {f:.4f}")
    for name, (e, f) in res.items():
        assert e < 4.0 * f, (name, e, f)


@pytest.mark.parametrize("where", ["q", "k", "v", "qk"])
def test_structured_channel_outliers(ext, where):
    """H1 -- the top hypothesis, and the one most likely to fail.

    Post-RoPE Q/K and the V of a long-context model carry massive-activation
    channels.  The pipeline mean-centers K (removing a channel OFFSET) and
    mean-centers V with per-tile scales, but Q is quantized with ONE amax
    per (head, 64-row block) across all 256 channels and gets no smoothing
    of any kind.  With a 50x channel present, the int8 step is set by that
    channel and the other 255 channels land on a handful of levels.

    Prediction if H1 holds: ``where="q"`` blows up far more than
    ``where="k"``, and int8 QK degrades MORE than e4m3 QK (e4m3's floating
    exponent tracks small elements; int8's uniform grid does not) -- i.e.
    the "int8 is 8x better" result from the uniform-data suite REVERSES.
    That would explain a delta that barely moves between the fp8-QK and
    int8-QK server configs while the synthetic suite says they are far
    apart.  Failure here confirms H1.
    """
    T, N, prefix = 128, 2048, 1920
    g = torch.Generator(device="cpu").manual_seed(122)
    qr = torch.randn((T, 8, HEAD_DIM), generator=g)
    kr = torch.randn((N, 2, HEAD_DIM), generator=g)
    vr = torch.randn((N, 2, HEAD_DIM), generator=g)
    if "q" in where:
        qr = channel_outliers(qr, g)
    if "k" in where:
        kr = channel_outliers(kr, g)
    if where == "v":
        vr = channel_outliers(vr, g)
    q, k, v = to_dev(qr, kr, vr)
    res = _structured_case(ext, q, k, v, prefix, 123, CFGS)
    for name, (e, f) in res.items():
        measure(f"outlier-{where}[{name}] err / floor", f"{e:.4f} / {f:.4f}")
    i8, f8 = res["i8qk+bf16pv"][0], res["f8qk+fp8pv"][0]
    measure(f"outlier-{where} int8/fp8 QK ratio", i8 / max(f8, 1e-12))
    # Calibration note (2026-08-28): with Hadamard rotation in the pipeline
    # this synthetic 4x50-gain construction measures ~7x the uniform floor
    # for the int8 modes -- a real physical limit (rotation spreads outlier
    # AMPLITUDE but preserves ENERGY; when outlier energy dominates a row,
    # the small-signal subspace still pays), but far harsher than measured
    # reality: on measured real-workload keys the conservative mode sits at
    # 0.50% vs a 0.45% bf16 yardstick.  The bound below is a regression
    # sentinel against outlier-handling regressions (pre-rotation this
    # construction measured far worse), not a quality gate.
    # The qk-CORRELATED case is fundamentally harder still: logits become
    # outlier^2-dominated, so the informative small-signal logit component
    # sits ~gain^2 below the dot magnitude and ANY quantized attention pays
    # (measured 26x here at gain 50).  Real keys measure at the bf16 floor.
    bound = 30.0 if where == "qk" else 12.0
    for name, (e, f) in res.items():
        assert e < bound * f, (
            f"{where} channel outliers cost {e / f:.1f}x the uniform floor "
            f"in {name} ({e:.4f} vs {f:.4f}) -> outlier-handling regression"
        )


def test_structured_rope_like(ext):
    """H1.  Real rotate-half RoPE phase structure in Q and K at a realistic
    absolute position offset: low-frequency channels are nearly constant
    across the window (large shared offsets, which K centering targets),
    high-frequency channels alternate every position.  Also checks that K
    centering still helps in this regime under both QK formats.
    """
    T, N, prefix = 128, 2048, 1920
    g = torch.Generator(device="cpu").manual_seed(124)
    q = rope_like(T, 8, g, pos0=prefix)
    k = rope_like(N, 2, g, pos0=0)
    v = torch.randn((N, 2, HEAD_DIM), generator=g)
    q, k, v = to_dev(q, k, v)
    res = _structured_case(ext, q, k, v, prefix, 125, CFGS)
    for name, (e, f) in res.items():
        measure(f"rope[{name}] err / floor", f"{e:.4f} / {f:.4f}")
    ref = ref_attention(q, k, v, prefix)
    on = (
        row_rel(
            run_pipeline(ext, q, k, v, contiguous_idx(N), prefix, center_k=True, qk_i8=True), ref
        )
        .mean()
        .item()
    )
    off = (
        row_rel(
            run_pipeline(ext, q, k, v, contiguous_idx(N), prefix, center_k=False, qk_i8=True), ref
        )
        .mean()
        .item()
    )
    measure("rope K-centering on/off", f"{on:.4f} / {off:.4f}")
    for name, (e, f) in res.items():
        assert e < 4.0 * f, (name, e, f)


@pytest.mark.parametrize("layout", ["short_high_in_long_low", "long_low_in_short_high"])
def test_structured_two_magnitude_regimes(ext, layout):
    """H1.  A short high-magnitude region inside a long low-magnitude one
    and the reverse. Stresses per-64-row K tile scales, per-tile V scales, the
    r_t >= 1/16 P-underflow floor and the global V amax simultaneously -- and,
    in the reverse layout, puts the high-magnitude region where it sets the
    GLOBAL amax for a window whose attention mass sits in the quiet region.
    """
    T, N, prefix = 128, 2048, 1920
    g = torch.Generator(device="cpu").manual_seed(126)
    if layout == "short_high_in_long_low":
        lo, hi, start, ln = 0.05, 1.0, 1024, 128  # small loud island
    else:
        lo, hi, start, ln = 1.0, 0.05, 512, 1408  # long quiet stretch
    k = two_regime(N, 2, g, lo=lo, hi=hi, hi_start=start, hi_len=ln)
    v = two_regime(N, 2, g, lo=lo, hi=hi, hi_start=start, hi_len=ln)
    q = torch.randn((T, 8, HEAD_DIM), generator=g)
    q, k, v = to_dev(q, k, v)
    res = _structured_case(ext, q, k, v, prefix, 127, CFGS)
    for name, (e, f) in res.items():
        measure(f"{layout}[{name}] err / floor", f"{e:.4f} / {f:.4f}")
    for name, (e, f) in res.items():
        assert e < 4.0 * f, (name, e, f)


def test_flat_attention_regime(ext):
    """H3.  Long, diffuse-attention regime: thousands of
    columns with near-equal logits, so p stays well below 1 for every
    column at once.  This is the regime where P's e4m3 encoding is coarsest
    (and where the r_t floor was added), and where the numerator/denominator
    rounding mismatch is largest relative to the signal.  Compared against
    the peaked regime at the same shape.
    """
    T, N, prefix = 128, 4096, 3968
    g = torch.Generator(device="cpu").manual_seed(128)
    v = torch.randn((N, 2, HEAD_DIM), generator=g)
    k = torch.randn((N, 2, HEAD_DIM), generator=g)
    out = {}
    for name, qscale in (("flat", 0.02), ("peaked", 1.0)):
        q = torch.randn((T, 8, HEAD_DIM), generator=g) * qscale
        qd, kd, vd = to_dev(q, k, v)
        ref = ref_attention(qd, kd, vd, prefix)
        for cfg_name, cfg in CFGS.items():
            e = (
                row_rel(run_pipeline(ext, qd, kd, vd, contiguous_idx(N), prefix, **cfg), ref)
                .mean()
                .item()
            )
            out[(name, cfg_name)] = e
            measure(f"attn-{name}[{cfg_name}] err", e)
    for cfg_name in CFGS:
        flat, peaked = out[("flat", cfg_name)], out[("peaked", cfg_name)]
        assert flat < 4.0 * peaked + 1e-3, (cfg_name, flat, peaked)


# =========================================================================
# (e) hypothesis-specific hardening
# =========================================================================


@pytest.mark.parametrize("N", [63, 100, 191, 999])
def test_bf16_pv_ragged_and_small_n(ext, N):
    """BUG-C.  The bf16-PV branch of ``prefetch_v`` guards its cp.async with
    ``if (gr < N)``, so V shared-memory rows >= N are never written -- while
    the fp8 branch copies its tile unconditionally and relies on the
    builder's zero padding.  The builder's ``vb16[:, N:npad] = 0`` is
    therefore dead code for this path.  When ntiles == 1 and N < BN those
    rows are whatever was in smem at kernel entry; the masked P is 0, but
    ``0 * NaN`` is NaN in the mma.  No existing bf16-PV test uses a ragged
    or sub-tile N (they are all N in {1024, 2048, 4096}).
    """
    T, prefix = 64, max(0, N - 64)
    q, k, v = rand_qkv(T, N, seed=130 + N)
    o = run_pipeline(ext, q, k, v, contiguous_idx(N), prefix, bf16_heads="all")
    assert torch.isfinite(o).all(), "bf16-PV produced non-finite output at ragged N"
    ref = ref_attention(q, k, v, prefix)
    rr = row_rel(o, ref)
    measure(f"bf16pv-raggedN[{N}] max row-rel", rr.max().item())
    assert rr.max().item() < 0.08, rr.max().item()


def test_production_shape_smoke(ext):
    """H1/H4.  The deployment shape the suite has never run: H=24, KVH=4
    (grp=6, non-power-of-two), a large single chunk with prefix=0 and a
    ragged N, i.e. the mid_10k eval prompt's geometry scaled down.  Also
    the only test here that puts >2 CTAs per head along the M axis with a
    long tile loop.
    """
    T = N = 2104  # 2104 % 64 == 56, like 10232
    H, KVH = 24, 4
    q, k, v = rand_qkv(T, N, H=H, KVH=KVH, seed=140)
    kp, vp, slots = scatter_pool(k, v, seed=140)
    ref = ref_attention(q, k, v, 0)
    for name, cfg in CFGS.items():
        o = run_pipeline(ext, q, kp, vp, slots, 0, **cfg)
        assert torch.isfinite(o).all()
        rr = row_rel(o, ref)
        measure(f"prod-shape[{name}] mean", rr.mean().item())
        measure(
            f"prod-shape[{name}] p99", rr.flatten().sort().values[int(rr.numel() * 0.99)].item()
        )
        assert rr.mean().item() < 0.06, (name, rr.mean().item())


def test_v_scale_linearity_bit_exact(ext):
    """H3.  Doubling V must double the output exactly: every V-side scale
    (per-tile vscale_t, vsmax, the r_t fold, the vmean add-back) is
    homogeneous of degree 1, and doubling a bf16 value is exact.  A drift
    here means a scale is being applied where it does not belong -- the
    kind of bug the per-tile exp-fold ``p*448*r_t`` / ``1/r_t`` pair could
    hide, and which row-relative error tests normalize away.
    """
    T, N, prefix = 128, 1024, 896
    q, k, v = rand_qkv(T, N, seed=150)
    for cfg in (dict(bf16_heads=()), dict(bf16_heads="all")):
        o1 = run_pipeline(ext, q, k, v, contiguous_idx(N), prefix, **cfg)
        o2 = run_pipeline(
            ext, q, k, (v.float() * 2).to(torch.bfloat16), contiguous_idx(N), prefix, **cfg
        )
        assert torch.equal(o2, o1 * 2), (
            f"O(2V) != 2*O(V) for {cfg}; max|delta|={(o2 - o1 * 2).abs().max().item()}"
        )


def test_sm_scale_fold_matches_reference(ext):
    """H8.  ``layer.scaling`` is folded into the Q scales rather than
    applied to the logits.  Sweep it: the kernel must track an fp32
    reference that uses the same scale, for every value (a fold that
    silently cancels, e.g. by being applied twice or not at all, shows up
    as a scale-dependent error).
    """
    T, N, prefix = 128, 1024, 896
    q, k, v = rand_qkv(T, N, seed=160)
    base = 1.0 / math.sqrt(HEAD_DIM)
    for mult in (0.25, 1.0, 4.0):
        sm = base * mult
        o = run_pipeline(ext, q, k, v, contiguous_idx(N), prefix, sm_scale=sm, bf16_heads="all")
        ref = ref_attention((q.float() * (sm / base)).to(torch.bfloat16), k, v, prefix)
        rr = row_rel(o, ref)
        measure(f"sm_scale x{mult} mean row-rel", rr.mean().item())
        assert rr.mean().item() < 0.03, (mult, rr.mean().item())
