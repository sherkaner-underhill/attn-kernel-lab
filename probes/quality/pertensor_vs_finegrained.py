#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Per-tensor FP8 (the cuDNN model) vs the lab's fine-grained scales, at depth.

WHAT THIS IS
------------
A *quantization-scheme simulator* at the attention-output boundary.  It answers
one narrow question:

    Holding the operator, the geometry and the data fixed, how much output
    error does each SCALE SCHEME contribute, and how does that error move as
    the context deepens toward the production 446,335 tokens?

It is a **prediction instrument** for the concurrently-ported per-tensor FP8
D256 kernel: it says what that port's quality should look like if it is
implemented correctly, and what it should NOT look like.

WHAT THIS IS NOT (read ``upstream/CLAIMS.md`` "Explicitly NOT claimed")
----------------------------------------------------------------------
* **Not** a model-quality statement. Attention-output error is only the
  engine-free numerical boundary; downstream task outputs, logprobs and task
  accuracy are separate boundaries and are not simulated here.
  The logprob-saturation lesson applies in reverse too: a metric that moves
  here has not thereby been shown to move anything downstream.
* **Not** a measurement of either kernel.  Nothing here executes
  ``csrc/fp8_prefill_attn.cu`` or cuDNN.  The lab scheme is evaluated through
  ``quant.py`` -- the normative preprocessing implementation -- so the
  *quantization* is the real production code; the *attention* is plain fp32
  PyTorch for every scheme, which is what makes the schemes comparable.
* **Not** real activations.  Synthetic distributions, chosen to isolate the
  failure modes the lab's transforms exist for.  Real-capture evidence is the
  private lane of the quality-gate plan and is not reproduced here.  Absolute
  levels here are NOT transferable to real data -- see the ``v_massive`` row,
  which exists to show exactly how much the denominator moves them.
* **Not** cuDNN.  Schemes 3/4 model the per-tensor FP8 descale contract (one
  scalar per operand), not cuDNN itself -- which per claim K11 does not accept
  D256 at all.

METHOD
------
Geometry mirrors production: T=128 query rows are the *tail* of an N-token
context (prefix = N - T, bottom-right causal), Hq=24, Hkv=4 (group 6), D=256,
sm_scale = 1/sqrt(256).  Attention rows are independent, so quality-at-depth-N
needs only a few query rows against a full N-deep K/V; at T=128 the whole score
rectangle is a few hundred MB and nothing needs streaming (the chunked pattern
in ``bench/candidate_bench.py::_streaming_masked_attention`` is unnecessary
here, and materializing keeps every scheme on one code path).

Data is generated in **fp32** (the master), then rounded to **bf16** -- the
dtype the operator actually consumes.  So:

    1   ref            fp32 attention on the fp32 master           (0 by def.)
    2   bf16           fp32 attention on the bf16-rounded input    <- yardstick
    2b  bf16_arith     + bf16 P and bf16 output (fuller impl-swap floor)
    3   pertensor      per-TENSOR E4M3 Q/K/V, no centering, no rotation
    4   pertensor_p    3 + E4M3 P at the online-softmax scale 448
    4b  pertensor_pT   3 + E4M3 P at ONE per-tensor amax on the normalized P
    3b  tile_nocenter  per-row-Q / per-64-tile-K/V E4M3, NO centering, NO
                       rotation -- isolates SCALE GRANULARITY from the transforms
    5   lab            quant.py's actual q8/k8/vt8, dequantized (quantization
                       error in isolation, matching how 3 and 3b are evaluated)
    5b  lab_p          5 + the kernel's normative E4M3 P rounding at 448*r_t

Pairing matters when reading the table: **3 vs 5** compares operand
quantization alone; **4 vs 5b** compares the two complete FP8 attention
pipelines.  Mixing them across the P boundary flatters whichever side is
missing its P rounding.

P rounding is modelled the way an online-softmax kernel actually does it: tile
t's P is rounded against the RUNNING row max after tile t, then rescaled by the
alpha product, not against the final row max.  That is not a detail -- rounding
against the final max instead disagrees with ``oracle_a.attention`` by ~2e-2
row-rel, while the running-max form agrees to ~2e-3 (the bf16 output-cast floor
of oracle_a's own normative final cast).  ``--check-oracle`` asserts this.
The vectorized form uses ``cummax`` over per-tile maxima, because oracle_a's
per-row Python loop is O(H*T*ntiles) = 2e7 iterations at N=446k.

Metrics follow the repo's conventions (quality-gate plan section 3.2): row-
relative L2 (mean / p99 / max), cosine similarity (mean / worst row), relative
L1, RMSE, output-norm ratio, NaN/Inf counts, and worst-head aggregation.  Each
scheme is reported alongside the bf16 yardstick for the SAME cell and the ratio
to it, because the metric core is a ratio, not an absolute.

Usage
-----
    python3 probes/quality/pertensor_vs_finegrained.py                  # full sweep
    python3 probes/quality/pertensor_vs_finegrained.py --quick          # smoke
    python3 probes/quality/pertensor_vs_finegrained.py --check-oracle   # 5b vs oracle_a
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

# The rotation GEMMs inside quant.py are reproducible only under a pinned cuBLAS
# workspace, and the variable must be set before torch creates its first cuBLAS
# handle (contract 3.1; same reasoning as bench/candidate_bench.py).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "attn_kernel_lab"))

import quant as labquant  # noqa: E402  -- standalone, the way tests/kernel imports it

HERE = Path(__file__).resolve().parent

T_ROWS = 128
Q_HEADS = 24
KV_HEADS = 4
GROUP = Q_HEADS // KV_HEADS
HEAD_DIM = 256
BLK = labquant.BLK
FP8_MAX = 448.0
SM_SCALE = 1.0 / math.sqrt(HEAD_DIM)

GEN_CHUNK = 32768  # K/V generation block; makes a shallow cell a bit-prefix of a deep one
DEPTHS = [4096, 32768, 131072, 262144, 446335]

SCHEMES = [
    "bf16",
    "bf16_arith",
    "pertensor",
    "pertensor_p",
    "pertensor_pT",
    "tile_nocenter",
    "lab",
    "lab_p",
]

# --------------------------------------------------------------------------
# distributions (deterministic; every seed is recorded in the JSON)
# --------------------------------------------------------------------------
#
# One base seed per distribution.  Q is drawn once per distribution and does not
# depend on N, so the depth axis varies only the context.  K/V are drawn in
# fixed GEN_CHUNK-row blocks with a per-block seed, so the N=4096 context is a
# bit-exact prefix of the N=446335 context: depth is then the only thing that
# changes along a row of the table.
#
# Unit per-element variance everywhere it is not the point of the case, so the
# score scale (and hence softmax sharpness) is comparable across distributions:
# q.k over D=256 has std 16, times sm_scale 1/16 gives scores ~ N(0,1).

DISTRIBUTIONS = {
    "gaussian": dict(
        seed=20260830,
        doc="Q,K,V ~ N(0,1) iid. Baseline: no structure for any transform to exploit.",
    ),
    "heavy_t3": dict(
        seed=20260831,
        doc="Q,K,V ~ Student-t(df=3) scaled to unit variance, built as "
        "z0*sqrt(1/(z1^2+z2^2+z3^2)) so it is deterministic under a seeded "
        "generator. Kurtosis is infinite; per-element outliers are the case a "
        "fine-grained (per-row / per-64-tile) amax is supposed to contain.",
    ),
    "rope_like": dict(
        seed=20260832,
        doc="K = randn + 5*channel_bias, channel_bias a fixed randn[256] shared by "
        "every position and every KV head; Q,V ~ N(0,1). A large shared "
        "per-channel offset is what RoPE'd keys carry, and is the case K "
        "mean-centering exists for (contract 3.3). The offset is constant "
        "across KV columns, so softmax shift-invariance makes centering exact.",
    ),
    "v_outlier": dict(
        seed=20260833,
        outlier_channels=(11, 67, 150, 233),
        doc="V has 4 fixed channels multiplied by 50 (zero-mean, huge variance); "
        "Q,K ~ N(0,1). The 'a few channels burn the FP8 range' case that "
        "per-tile V scales address. HONESTY NOTE: V mean-centering targets "
        "large channel MEANS, not large channel VARIANCE -- see v_massive.",
    ),
    "v_massive": dict(
        seed=20260834,
        outlier_channels=(11, 67, 150, 233),
        doc="EXTRA (beyond the four requested). V has 4 fixed channels offset by "
        "+50: a massive-activation channel, large magnitude and consistent "
        "sign; Q,K ~ N(0,1). This is the case V mean-centering actually exists "
        "for (contract 3.4), separated from v_outlier on purpose so the "
        "centering claim is tested against the right distribution. It is also "
        "the only cell whose output norm is dominated by a channel mean, which "
        "is the regime real activations live in -- so it calibrates how far the "
        "zero-mean cells' absolute levels are from a real-capture number.",
    ),
}


def _gen(shape, seed: int, device) -> torch.Tensor:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return torch.randn(shape, generator=g, device=device, dtype=torch.float32)


def _gen_t3(shape, seed: int, device) -> torch.Tensor:
    """Student-t(df=3) normalized to unit variance, from four seeded normals."""
    z0 = _gen(shape, seed, device)
    chi2 = torch.zeros(shape, device=device, dtype=torch.float32)
    for i in range(3):
        z = _gen(shape, seed + 101 * (i + 1), device)
        chi2.addcmul_(z, z)
        del z
    # t3 = z0*sqrt(3/chi2) has Var 3; z0*sqrt(1/chi2) is the unit-variance form.
    return z0.mul_(torch.rsqrt(chi2))


def _fill_kv(dist: str, n: int, device, cfg: dict):
    """K/V as [N, KV_HEADS, HEAD_DIM] fp32, generated in fixed GEN_CHUNK blocks."""
    base = cfg["seed"]
    k = torch.empty(n, KV_HEADS, HEAD_DIM, device=device, dtype=torch.float32)
    v = torch.empty(n, KV_HEADS, HEAD_DIM, device=device, dtype=torch.float32)
    shape = (GEN_CHUNK, KV_HEADS, HEAD_DIM)
    draw = _gen_t3 if dist == "heavy_t3" else _gen
    for c0 in range(0, n, GEN_CHUNK):
        c1 = min(c0 + GEN_CHUNK, n)
        ci = c0 // GEN_CHUNK
        kc = draw(shape, base + 10_000 + 7 * ci, device)
        vc = draw(shape, base + 20_000 + 7 * ci, device)
        if dist == "rope_like":
            kc.add_(_gen((HEAD_DIM,), base + 777, device) * 5.0)
        if dist in ("v_outlier", "v_massive"):
            ch = torch.tensor(cfg["outlier_channels"], device=device, dtype=torch.long)
            if dist == "v_outlier":
                vc[:, :, ch] *= 50.0
            else:
                vc[:, :, ch] += 50.0
        k[c0:c1] = kc[: c1 - c0]
        v[c0:c1] = vc[: c1 - c0]
        del kc, vc
    return k, v


def _gen_q(dist: str, device, cfg: dict) -> torch.Tensor:
    shape = (T_ROWS, Q_HEADS, HEAD_DIM)
    draw = _gen_t3 if dist == "heavy_t3" else _gen
    return draw(shape, cfg["seed"] + 900_000, device)


# --------------------------------------------------------------------------
# attention machinery (materialized; T=128 keeps the score rectangle small)
# --------------------------------------------------------------------------


def _e4m3_rt_(x: torch.Tensor, scale) -> torch.Tensor:
    """Round-trip ``x`` through E4M3 at ``scale``, IN PLACE (``x`` must be owned).

    torch's float8_e4m3fn cast saturates at +-448 on this build (verified), so an
    amax-derived scale can never produce a NaN encoding; no clamp is needed.
    """
    x.div_(scale)
    x.copy_(x.to(torch.float8_e4m3fn))
    x.mul_(scale)
    return x


def _tail_mask(device) -> torch.Tensor:
    """[T, T] bool: True where the bottom-right causal mask forbids attention.

    Row r of the extend sees the P prefix positions plus positions 0..r of the
    current chunk, so over the LAST T columns, column j is masked iff j > r.
    """
    ar = torch.arange(T_ROWS, device=device)
    return ar[None, :] > ar[:, None]


def _scores(
    qf: torch.Tensor, kf: torch.Tensor, g: int, prefix: int, n: int, ncols: int, tail: torch.Tensor
) -> torch.Tensor:
    """qf [g*T, D] -> bottom-right-masked scores [g, T, ncols] (ncols >= n)."""
    s = (qf @ kf.t()).view(g, T_ROWS, ncols)
    if ncols > n:
        s[:, :, n:] = float("-inf")
    s[:, :, prefix:n].masked_fill_(tail, float("-inf"))
    return s


def _pv(
    s: torch.Tensor,
    vf: torch.Tensor,
    mode: str,
    *,
    fold=None,
    amax_p: float | None = None,
    vscale: float | None = None,
    vmean: torch.Tensor | None = None,
) -> torch.Tensor:
    """Softmax + PV.  ``s`` is consumed in place.  Returns [g*T, D] fp32.

    modes
      plain     : fp32 softmax, fp32 PV.  The evaluation used for schemes 1, 2,
                  3, 3b and 5 -- error is entirely from the dequantized operands.
      p_bf16    : P cast to bf16 and PV in bf16 (the bf16 implementation swap).
      p_tensor  : P NORMALIZED, then E4M3-rounded at one per-tensor scale
                  amax_p/448.  The literal per-tensor-P reading; a flash-style
                  kernel does not do this, which is the point of comparing it.
      p_online  : the online-softmax form.  Per 64-column tile, P is rounded to
                  E4M3 at ``fold`` (448 for scheme 4, 448*r_t for scheme 5b --
                  contract 3.6 / claim K3) against the RUNNING row max, then
                  carried forward by the alpha product exp(m_t - m_final).
                  ``l`` accumulates the PRE-rounding sum, as the kernel does.
                  With ``vscale``/``vmean``, ``vf`` holds the RAW stored fp8 V
                  bytes and the lab epilogue (vs_max scale, exact V-mean
                  add-back) is applied; without them ``vf`` is dequantized V.
    """
    g, t, c = s.shape
    b = g * t

    if mode == "p_online":
        nt = c // BLK
        sv = s.view(b, nt, BLK)
        m_run = sv.amax(dim=-1).cummax(dim=1).values  # [b, nt]
        neg = torch.isneginf(m_run)  # only if tile 0 is
        m_safe = m_run.masked_fill(neg, 0.0)  # fully masked (never here)
        w = (m_safe - m_safe[:, -1:]).exp().masked_fill_(neg, 0.0)
        sv.sub_(m_safe[:, :, None]).exp_()  # p_t = exp(s - m_t)
        lsum = (sv.sum(dim=-1) * w).sum(dim=-1, keepdim=True)  # == sum exp(s - m_final)
        sv.mul_(fold)
        sv.copy_(sv.to(torch.float8_e4m3fn))  # the normative rounding
        sv.mul_(w[:, :, None])  # alpha product
        acc = s.view(b, c) @ vf
        if vscale is None:
            acc.div_(FP8_MAX * lsum)
            return acc
        acc.mul_(vscale / (FP8_MAX * lsum))
        acc.add_(vmean)
        return acc

    m = s.amax(dim=-1, keepdim=True)
    s.sub_(m).exp_()
    ell = s.sum(dim=-1, keepdim=True)
    if mode == "plain":
        s.div_(ell)
        return s.view(b, c) @ vf
    if mode == "p_bf16":
        s.div_(ell)
        p = s.to(torch.bfloat16)
        return (p.view(b, c) @ vf.to(torch.bfloat16)).float()
    if mode == "p_tensor":
        s.div_(ell)
        _e4m3_rt_(s, amax_p / FP8_MAX)
        return s.view(b, c) @ vf
    raise ValueError(mode)


def _run_scheme(
    q_head_fn, kv_fn, n: int, ncols: int, prefix: int, tail, mode: str, kw_fn, head_batch: int
) -> torch.Tensor:
    """Drive one scheme over all 24 query heads; returns [T, Hq, D] fp32.

    ``kv_fn(kvh) -> (Kf, Vf, extra)`` builds the dequantized [ncols, D] K/V for
    one KV head, once, reused by its 6 query heads.  ``q_head_fn(h) -> [T, D]``
    builds one query head's dequantized rows, with sm_scale already applied
    where the scheme does not fold it into the Q scales.
    """
    out = torch.empty(T_ROWS, Q_HEADS, HEAD_DIM, device="cuda", dtype=torch.float32)
    for kvh in range(KV_HEADS):
        kf, vf, extra = kv_fn(kvh)
        heads = list(range(kvh * GROUP, (kvh + 1) * GROUP))
        for b0 in range(0, GROUP, head_batch):
            hs = heads[b0 : b0 + head_batch]
            qf = torch.cat([q_head_fn(h) for h in hs], dim=0)
            s = _scores(qf, kf, len(hs), prefix, n, ncols, tail)
            o = _pv(s, vf, mode, **kw_fn(kvh, extra))
            out[:, hs, :] = o.view(len(hs), T_ROWS, HEAD_DIM).permute(1, 0, 2)
            del qf, s, o
        del kf, vf, extra
    return out


def _head_batch_for(ncols: int, budget_bytes: float = 384e6) -> int:
    return max(1, min(GROUP, int(budget_bytes // (T_ROWS * ncols * 4))))


def _pad_head(src: torch.Tensor, kvh: int, npad: int, n: int) -> torch.Tensor:
    """[N, KVH, D] (any dtype) -> owned [npad, D] fp32 with a zeroed tile tail."""
    out = torch.zeros(npad, HEAD_DIM, device=src.device, dtype=torch.float32)
    out[:n] = src[:, kvh]
    return out


def _fp32_reference(q_f32, k_f32, v_f32, n, npad, prefix, tail, hb) -> torch.Tensor:
    """Scheme 1.  A free function so the caller can ``del`` the 3.7 GiB of fp32
    K/V the moment the reference exists, without those names being captured by a
    closure that outlives them."""
    return _run_scheme(
        lambda h: q_f32[:, h] * SM_SCALE,
        lambda kvh: (_pad_head(k_f32, kvh, npad, n), _pad_head(v_f32, kvh, npad, n), None),
        n,
        npad,
        prefix,
        tail,
        "plain",
        lambda *_: {},
        hb,
    )


# --------------------------------------------------------------------------
# metrics (quality-gate plan section 3.2)
# --------------------------------------------------------------------------


def _metrics(out: torch.Tensor, ref: torch.Tensor) -> dict:
    o = out.reshape(-1, HEAD_DIM).double()
    r = ref.reshape(-1, HEAD_DIM).double()
    d = o - r
    rn = r.norm(dim=1)
    tiny = torch.finfo(torch.float64).tiny
    rel = d.norm(dim=1) / rn.clamp_min(tiny)  # per (row, head)
    cos = (o * r).sum(1) / (o.norm(dim=1) * rn).clamp_min(tiny)
    per_head = rel.view(T_ROWS, Q_HEADS).mean(dim=0)
    wh = int(per_head.argmax())
    return {
        "row_rel_l2_mean": float(rel.mean()),
        "row_rel_l2_p99": float(rel.quantile(0.99)),
        "row_rel_l2_max": float(rel.max()),
        "cos_sim_mean": float(cos.mean()),
        "cos_sim_worst_row": float(cos.min()),
        "rel_l1": float(d.abs().sum() / r.abs().sum()),
        "rmse": float((d * d).mean().sqrt()),
        "nrmse": float((d * d).mean().sqrt() / (r * r).mean().sqrt()),
        "norm_ratio": float(o.norm() / r.norm()),
        "worst_head": {"head": wh, "row_rel_l2_mean": float(per_head[wh])},
        "nan_count": int(torch.isnan(out).sum()),
        "inf_count": int(torch.isinf(out).sum()),
    }


# --------------------------------------------------------------------------
# one cell
# --------------------------------------------------------------------------


def run_cell(dist: str, n: int, cfg: dict, verbose: bool = True) -> dict:
    dev = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()
    prefix = n - T_ROWS
    npad = (n + BLK - 1) // BLK * BLK
    ntiles = npad // BLK
    tail = _tail_mask(dev)
    hb = _head_batch_for(npad)
    none_kw = lambda *_: {}  # noqa: E731 -- schemes with no extra PV state
    res: dict = {"schemes": {}, "npad": npad, "head_batch": hb}

    # ---- generation -------------------------------------------------------
    q_f32 = _gen_q(dist, dev, cfg)
    k_f32, v_f32 = _fill_kv(dist, n, dev, cfg)
    q_bf, k_bf, v_bf = (x.to(torch.bfloat16) for x in (q_f32, k_f32, v_f32))

    def _amax(x):
        mn, mx = torch.aminmax(x)
        return max(abs(float(mn)), abs(float(mx)))

    amax_q, amax_k, amax_v = _amax(q_bf), _amax(k_bf), _amax(v_bf)
    res["per_tensor_amax"] = {"q": amax_q, "k": amax_k, "v": amax_v}

    # ---- scheme 1: the fp32 reference -------------------------------------
    ref = _fp32_reference(q_f32, k_f32, v_f32, n, npad, prefix, tail, hb)
    del k_f32, v_f32
    torch.cuda.empty_cache()

    # ---- schemes 2 / 2b: the bf16 yardstick -------------------------------
    q_bff = q_bf.float()

    def kv_bf(kvh):
        return _pad_head(k_bf, kvh, npad, n), _pad_head(v_bf, kvh, npad, n), None

    res["schemes"]["bf16"] = _metrics(
        _run_scheme(
            lambda h: q_bff[:, h] * SM_SCALE, kv_bf, n, npad, prefix, tail, "plain", none_kw, hb
        ),
        ref,
    )
    res["schemes"]["bf16_arith"] = _metrics(
        _run_scheme(
            lambda h: q_bff[:, h] * SM_SCALE, kv_bf, n, npad, prefix, tail, "p_bf16", none_kw, hb
        )
        .to(torch.bfloat16)
        .float(),
        ref,
    )

    # ---- schemes 3 / 4 / 4b: per-tensor E4M3 ------------------------------
    # One amax/448 descale per operand over the WHOLE tensor (all heads), which
    # is the cuDNN FP8 SDPA contract.  Using this tensor's own amax rather than a
    # calibrated constant is the BEST case for per-tensor and is deliberate.
    sq, sk, sv = amax_q / FP8_MAX, amax_k / FP8_MAX, amax_v / FP8_MAX
    q_pt = _e4m3_rt_(q_bf.float(), sq)

    def kv_pt(kvh):
        return (
            _e4m3_rt_(_pad_head(k_bf, kvh, npad, n), sk),
            _e4m3_rt_(_pad_head(v_bf, kvh, npad, n), sv),
            None,
        )

    res["schemes"]["pertensor"] = _metrics(
        _run_scheme(
            lambda h: q_pt[:, h] * SM_SCALE, kv_pt, n, npad, prefix, tail, "plain", none_kw, hb
        ),
        ref,
    )
    res["schemes"]["pertensor_p"] = _metrics(
        _run_scheme(
            lambda h: q_pt[:, h] * SM_SCALE,
            kv_pt,
            n,
            npad,
            prefix,
            tail,
            "p_online",
            lambda *_: {"fold": FP8_MAX},
            hb,
        ),
        ref,
    )

    # A per-TENSOR P scale needs the global amax of the NORMALIZED P first.  The
    # row max exponentiates to exactly 1, so max_j P[r,j] = 1/l_r and the extra
    # pass only has to find min_r l_r -- but it IS an extra pass, which is part
    # of what a non-row-local scale costs.
    min_l = math.inf
    for kvh in range(KV_HEADS):
        kf = _e4m3_rt_(_pad_head(k_bf, kvh, npad, n), sk)
        for b0 in range(0, GROUP, hb):
            hs = list(range(kvh * GROUP + b0, min(kvh * GROUP + b0 + hb, (kvh + 1) * GROUP)))
            qf = torch.cat([q_pt[:, h] * SM_SCALE for h in hs], dim=0)
            s = _scores(qf, kf, len(hs), prefix, n, npad, tail)
            s.sub_(s.amax(dim=-1, keepdim=True)).exp_()
            min_l = min(min_l, float(s.sum(dim=-1).min()))
            del s, qf
        del kf
    amax_p = 1.0 / min_l
    res["amax_p_normalized"] = amax_p
    res["schemes"]["pertensor_pT"] = _metrics(
        _run_scheme(
            lambda h: q_pt[:, h] * SM_SCALE,
            kv_pt,
            n,
            npad,
            prefix,
            tail,
            "p_tensor",
            lambda *_: {"amax_p": amax_p},
            hb,
        ),
        ref,
    )
    q_pt = None  # free 3 MiB + let the padded K/V go
    torch.cuda.empty_cache()

    # ---- diagnostic 3b: fine granularity, NO centering, NO rotation -------
    # Per-ROW Q amax and per-64-tile K/V amax over (rows x channels) jointly,
    # exactly as quant.py takes them -- but E4M3 and no transforms.  Scheme 3 ->
    # 3b is granularity alone; 3b -> 5 is centering + rotation + INT8 QK + the
    # vs_max/16 floor.  Without this column the headline number cannot be
    # attributed, and "per-tensor is worse" would be an unearned conclusion.
    qt = _e4m3_rt_(
        q_bf.float(), q_bf.float().abs().amax(dim=2, keepdim=True).clamp_min(1e-8) / FP8_MAX
    )

    def kv_tile(kvh):
        outs = []
        for src in (k_bf, v_bf):
            x = _pad_head(src, kvh, npad, n)
            xb = x.view(ntiles, BLK, HEAD_DIM)
            _e4m3_rt_(xb, xb.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-8) / FP8_MAX)
            outs.append(x)
        return outs[0], outs[1], None

    res["schemes"]["tile_nocenter"] = _metrics(
        _run_scheme(
            lambda h: qt[:, h] * SM_SCALE, kv_tile, n, npad, prefix, tail, "plain", none_kw, hb
        ),
        ref,
    )
    qt = None
    torch.cuda.empty_cache()

    # ---- schemes 5 / 5b: the lab scheme, via quant.py itself --------------
    ws = labquant.FP8PrefillWorkspace(dev)
    idx = torch.arange(n, device=dev, dtype=torch.long)
    q8, qs, mpad = labquant.quantize_q(ws, q_bf, SM_SCALE, qk_i8=True, rotate=True)
    packs = labquant.gather_quantize_kv(
        ws, k_bf, v_bf, idx, need_vt8=True, need_vb16=False, center_k=True, qk_i8=True, rotate=True
    )
    assert mpad == T_ROWS and packs["ntmax"] == ntiles, (mpad, packs["ntmax"], ntiles)
    # vscale_t is the exact per-tile V scale max(amax, vs_max/16)/448.  quant.py
    # keeps it in the workspace instead of returning it; ws.get hands back the
    # very same buffer, which is exact where vscale/vinvr would round twice.
    vst = ws.get("vscale_t", (KV_HEADS, ntiles), torch.float32)
    k8 = packs["k8"].view(torch.int8)
    ks, vsmax, vinvr, vmean = (packs[x] for x in ("kscale", "vscale", "vinvr", "vmean"))
    sigma = torch.tensor(labquant.SIGMA64, device=dev, dtype=torch.long)
    inv = torch.empty_like(sigma)
    inv[sigma] = torch.arange(BLK, device=dev)  # inverse SIGMA64 (contract 3.5)
    # Q bytes carry the rotation; the per-row scales carry sm_scale already.
    q_lab = q8.view(torch.int8).float() * qs[:, :, None]

    def _k_lab(kvh):
        kf = k8[kvh].view(ntiles, BLK, HEAD_DIM).float()
        kf.mul_(ks[kvh][:, None, None])  # rotated, mean-centered K
        return kf.reshape(npad, HEAD_DIM)

    def _v8_raw(kvh):
        """vt8 [KVH, nt, D, BLK-permuted] -> raw fp8 byte VALUES [npad, D]."""
        vt = packs["vt8"].view(KV_HEADS, ntiles, HEAD_DIM, BLK)[kvh]
        return (
            vt.view(torch.float8_e4m3fn)
            .permute(0, 2, 1)
            .index_select(1, inv)
            .float()
            .reshape(npad, HEAD_DIM)
        )

    def kv_lab(kvh):
        vf = _v8_raw(kvh).view(ntiles, BLK, HEAD_DIM)
        vf.mul_(vst[kvh][:, None, None])
        vf = vf.reshape(npad, HEAD_DIM)
        vf.add_(vmean[kvh])  # exact epilogue add-back
        return _k_lab(kvh), vf, None

    res["schemes"]["lab"] = _metrics(
        _run_scheme(lambda h: q_lab[h], kv_lab, n, npad, prefix, tail, "plain", none_kw, hb), ref
    )

    def kw_lab_p(kvh, _extra):
        r_t = (1.0 / vinvr[kvh, :ntiles]).clamp(1.0 / 16.0, 1.0)
        return {
            "fold": (FP8_MAX * r_t).view(1, ntiles, 1),
            "vscale": float(vsmax[kvh]),
            "vmean": vmean[kvh],
        }

    res["schemes"]["lab_p"] = _metrics(
        _run_scheme(
            lambda h: q_lab[h],
            lambda kvh: (_k_lab(kvh), _v8_raw(kvh), None),
            n,
            npad,
            prefix,
            tail,
            "p_online",
            kw_lab_p,
            hb,
        ),
        ref,
    )

    # ---- ratios to the bf16 yardstick -- the metric core is a ratio -------
    yard = res["schemes"]["bf16"]
    for m in res["schemes"].values():
        m["ratio_to_bf16"] = {
            k: (m[k] / yard[k] if yard[k] else float("inf"))
            for k in ("row_rel_l2_mean", "row_rel_l2_max", "rel_l1", "rmse")
        }

    res["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30
    res["seconds"] = time.perf_counter() - t_start
    res["ref_row_norm_mean"] = float(ref.reshape(-1, HEAD_DIM).norm(dim=1).mean())

    if verbose:
        s = res["schemes"]
        print(
            f"    bf16 {yard['row_rel_l2_mean']:.3%} | lab {s['lab']['row_rel_l2_mean']:.3%}"
            f" lab_p {s['lab_p']['row_rel_l2_mean']:.3%} | pertensor "
            f"{s['pertensor']['row_rel_l2_mean']:.3%} +P "
            f"{s['pertensor_p']['row_rel_l2_mean']:.3%}"
            f"  ({res['seconds']:.1f}s, {res['peak_mem_gib']:.2f} GiB)",
            flush=True,
        )
    return res


# --------------------------------------------------------------------------
# per-operand ablation of the lab scheme (the "why isn't it ~1%?" investigation)
# --------------------------------------------------------------------------


def ablate_cell(dist: str, n: int, cfg: dict) -> dict:
    """Decompose the lab scheme's error into its Q / K / V / P terms.

    Each row quantizes ONE operand through quant.py and leaves the others at
    bf16-exact.  The Hadamard is orthonormal, so rotating the un-quantized
    partner in fp32 is exact and keeps both sides of the QK dot in the same
    space; K centering is a per-row score shift and softmax removes it, so the
    un-centered exact K is an equally valid partner.

    This exists because the gaussian cells land near 3% row-rel where the
    repo's real-capture history records ~1%, and a number that far from the
    recorded class must be explained before it is reported.
    """
    dev = torch.device("cuda")
    prefix = n - T_ROWS
    npad = (n + BLK - 1) // BLK * BLK
    ntiles = npad // BLK
    tail = _tail_mask(dev)
    hb = _head_batch_for(npad)
    none_kw = lambda *_: {}  # noqa: E731

    q_f32, k_f32, v_f32 = _gen_q(dist, dev, cfg), *_fill_kv(dist, n, dev, cfg)
    q_bf, k_bf, v_bf = (x.to(torch.bfloat16) for x in (q_f32, k_f32, v_f32))
    ref = _fp32_reference(q_f32, k_f32, v_f32, n, npad, prefix, tail, hb)
    del k_f32, v_f32
    torch.cuda.empty_cache()

    ws = labquant.FP8PrefillWorkspace(dev)
    q8, qs, _ = labquant.quantize_q(ws, q_bf, SM_SCALE, qk_i8=True, rotate=True)
    packs = labquant.gather_quantize_kv(
        ws, k_bf, v_bf, torch.arange(n, device=dev, dtype=torch.long), True, False
    )
    vst = ws.get("vscale_t", (KV_HEADS, ntiles), torch.float32)
    rot = ws.hadamard
    k8 = packs["k8"].view(torch.int8)
    inv = torch.empty(BLK, device=dev, dtype=torch.long)
    inv[torch.tensor(labquant.SIGMA64, device=dev)] = torch.arange(BLK, device=dev)

    q_lab = q8.view(torch.int8).float() * qs[:, :, None]
    q_rot = (q_bf.float().permute(1, 0, 2) @ rot) * SM_SCALE  # [H, T, D] exact
    q_plain = q_bf.float().permute(1, 0, 2) * SM_SCALE

    def k_lab(kvh):
        kf = k8[kvh].view(ntiles, BLK, HEAD_DIM).float()
        kf.mul_(packs["kscale"][kvh][:, None, None])
        return kf.reshape(npad, HEAD_DIM)

    def k_rot(kvh):
        return _pad_head(k_bf, kvh, npad, n) @ rot

    def v_lab(kvh):
        vt = packs["vt8"].view(KV_HEADS, ntiles, HEAD_DIM, BLK)[kvh]
        vf = vt.view(torch.float8_e4m3fn).permute(0, 2, 1).index_select(1, inv).float()
        vf.mul_(vst[kvh][:, None, None])
        vf = vf.reshape(npad, HEAD_DIM)
        vf.add_(packs["vmean"][kvh])
        return vf

    def v_exact(kvh):
        return _pad_head(v_bf, kvh, npad, n)

    def k_exact(kvh):
        return _pad_head(k_bf, kvh, npad, n)

    combos = {
        "q_only": (lambda h: q_lab[h], k_rot, v_exact, "plain", none_kw),
        "k_only": (lambda h: q_rot[h], k_lab, v_exact, "plain", none_kw),
        "qk_only": (lambda h: q_lab[h], k_lab, v_exact, "plain", none_kw),
        "v_only": (lambda h: q_plain[h], k_exact, v_lab, "plain", none_kw),
        "p_only": (
            lambda h: q_plain[h],
            k_exact,
            v_exact,
            "p_online",
            lambda *_: {"fold": FP8_MAX},
        ),
    }
    out = {}
    for name, (qfn, kfn, vfn, mode, kw) in combos.items():
        o = _run_scheme(
            qfn, lambda kvh: (kfn(kvh), vfn(kvh), None), n, npad, prefix, tail, mode, kw, hb
        )
        out[name] = _metrics(o, ref)["row_rel_l2_mean"]
        del o
        torch.cuda.empty_cache()

    # Structural facts that explain the level, not just the split.
    r_t = (1.0 / packs["vinvr"][:, :ntiles]).clamp(1.0 / 16.0, 1.0)
    rows = ref.reshape(-1, HEAD_DIM)
    vm = packs["vmean"]
    out["v_floor_fraction"] = float((r_t <= 1.0 / 16.0 + 1e-6).float().mean())
    out["r_t_min"] = float(r_t.min())
    out["r_t_mean"] = float(r_t.mean())
    # ||O|| against ||V channel mean||: with zero-mean V the output is an average
    # of ~N_eff random vectors and is therefore SMALL, which inflates every
    # relative metric. Real activations carry large channel means.
    out["ref_row_norm_mean"] = float(rows.norm(dim=1).mean())
    out["v_channel_mean_norm"] = float(vm.norm(dim=1).mean())
    out["v_rms"] = float(v_bf.float().pow(2).mean().sqrt())
    return out


# --------------------------------------------------------------------------
# oracle_a cross-check for scheme 5b
# --------------------------------------------------------------------------


def check_oracle() -> dict:
    """Assert the vectorized 5b equals ``oracle_a.attention`` at a small shape.

    oracle_a is the contract's independent implementation of the intended
    quantized operator; 5b is a vectorized restatement of the same arithmetic,
    and agreement is the licence to use 5b at N=446k where oracle_a's per-row
    Python loop cannot go.  Both sides are fed the SAME packs from quant.py, so
    this compares the attention arithmetic only -- preprocessing agreement
    between quant.py and oracle_a is tests/kernel/test_oracle_a.py's job.
    """
    import oracle_a

    dev = torch.device("cuda")
    n, t, hq, kvh = 512, 128, 6, 1
    g = torch.Generator(device=dev)
    g.manual_seed(4242)
    q = torch.randn(t, hq, HEAD_DIM, generator=g, device=dev, dtype=torch.bfloat16)
    k = torch.randn(n, kvh, HEAD_DIM, generator=g, device=dev, dtype=torch.bfloat16)
    v = torch.randn(n, kvh, HEAD_DIM, generator=g, device=dev, dtype=torch.bfloat16)
    prefix = n - t

    ws = labquant.FP8PrefillWorkspace(dev)
    q8, qs, mpad = labquant.quantize_q(ws, q, SM_SCALE, qk_i8=True, rotate=True)
    packs = labquant.gather_quantize_kv(
        ws, k, v, torch.arange(n, device=dev, dtype=torch.long), True, False
    )
    ntiles = packs["ntmax"]
    npad = ntiles * BLK
    vst = ws.get("vscale_t", (kvh, ntiles), torch.float32)

    qp = oracle_a.QPack(q8=q8.view(torch.int8).cpu(), qscale=qs.cpu(), mpad=mpad)
    kv = oracle_a.KVPack(
        k8=packs["k8"].view(torch.int8).cpu(),
        kscale=packs["kscale"].cpu(),
        kmean=torch.zeros(kvh, HEAD_DIM),
        vt8=packs["vt8"].view(kvh, ntiles, HEAD_DIM, BLK).view(torch.float8_e4m3fn).cpu(),
        vst_scaled=vst.cpu(),
        vscale=packs["vscale"].cpu(),
        vlog2r=packs["vlog2r"].cpu(),
        vinvr=packs["vinvr"].cpu(),
        vmean=packs["vmean"].cpu(),
        n=n,
    )
    ora = oracle_a.attention(qp, kv, prefix, t, group=hq // kvh).float().to(dev)

    sigma = torch.tensor(labquant.SIGMA64, device=dev, dtype=torch.long)
    inv = torch.empty_like(sigma)
    inv[sigma] = torch.arange(BLK, device=dev)
    q_lab = q8.view(torch.int8).float() * qs[:, :, None]
    kf = packs["k8"].view(torch.int8)[0].view(ntiles, BLK, HEAD_DIM).float()
    kf.mul_(packs["kscale"][0][:, None, None])
    vraw = (
        packs["vt8"]
        .view(kvh, ntiles, HEAD_DIM, BLK)[0]
        .view(torch.float8_e4m3fn)
        .permute(0, 2, 1)
        .index_select(1, inv)
        .float()
        .reshape(npad, HEAD_DIM)
    )
    r_t = (1.0 / packs["vinvr"][0, :ntiles]).clamp(1.0 / 16.0, 1.0)
    s = _scores(
        q_lab.reshape(hq * t, HEAD_DIM),
        kf.reshape(npad, HEAD_DIM),
        hq,
        prefix,
        n,
        npad,
        _tail_mask(dev),
    )
    mine = (
        _pv(
            s,
            vraw,
            "p_online",
            fold=(FP8_MAX * r_t).view(1, ntiles, 1),
            vscale=float(packs["vscale"][0]),
            vmean=packs["vmean"][0],
        )
        .view(hq, t, HEAD_DIM)
        .permute(1, 0, 2)
    )

    d = mine.double() - ora.double()
    rel = float((d.norm(dim=-1) / ora.double().norm(dim=-1).clamp_min(1e-30)).max())
    print(f"  [check-oracle] vectorized 5b vs oracle_a.attention: max row-rel L2 = {rel:.3e}")
    # oracle_a returns bf16 (the normative final cast) while 5b stays fp32, so
    # the floor here is bf16 output rounding (~2e-3), not zero.
    assert rel < 5e-3, f"5b disagrees with oracle_a beyond the bf16 cast floor: {rel}"
    return {
        "shape": {"T": t, "q_heads": hq, "kv_heads": kvh, "N": n},
        "max_row_rel_l2_vs_oracle_a": rel,
        "floor": "oracle_a returns bf16 (normative final cast); 5b stays fp32",
    }


# --------------------------------------------------------------------------
# environment record
# --------------------------------------------------------------------------


def _nvidia_smi() -> dict:
    query = (
        "name,driver_version,uuid,power.limit,power.draw,clocks.sm,"
        "clocks.max.sm,temperature.gpu,persistence_mode"
    )
    try:
        out = (
            subprocess.run(
                ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            .stdout.strip()
            .splitlines()[0]
        )
        return dict(zip(query.split(","), [p.strip() for p in out.split(",")]))
    except Exception as exc:  # noqa: BLE001 -- record the absence, never fail the run
        return {"error": f"nvidia-smi unavailable: {exc}"}


def _env() -> dict:
    props = torch.cuda.get_device_properties(0)
    src = (ROOT / "src" / "attn_kernel_lab" / "quant.py").read_bytes()
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": props.name,
        "capability": f"sm_{props.major}{props.minor}",
        "total_mem_gib": props.total_memory / 2**30,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "quant_py_sha256": hashlib.sha256(src).hexdigest(),
        "nvidia_smi": _nvidia_smi(),
    }


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--depths", type=int, nargs="+", default=DEPTHS)
    ap.add_argument("--dists", nargs="+", default=list(DISTRIBUTIONS))
    ap.add_argument(
        "--quick", action="store_true", help="smoke: gaussian + rope_like at 4096 and 32768"
    )
    ap.add_argument("--check-oracle", action="store_true")
    ap.add_argument(
        "--ablate",
        nargs="*",
        metavar="DIST:N",
        help="per-operand ablation of the lab scheme; default "
        "gaussian and v_massive at the shallowest and deepest N",
    )
    ap.add_argument("--out", default=str(HERE / "pertensor_vs_finegrained.json"))
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device", file=sys.stderr)
        return 2
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if args.quick:
        args.depths = [4096, 32768]
        args.dists = ["gaussian", "rope_like"]

    record = {
        "probe": "pertensor_vs_finegrained",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "question": "Per-tensor FP8 (one descale per operand, the cuDNN model) vs the "
        "lab's per-row-Q / per-tile-K / per-tile-V scheme, simulated at the "
        "attention-output level as context depth grows to 446,335 tokens.",
        "not_claimed": [
            "Not a model-quality statement: boundary (a) of the quality-gate plan only. "
            "See the logprob-saturation rule in upstream/CLAIMS.md.",
            "Not a measurement of either kernel: no CUDA kernel and no cuDNN engine "
            "executes here. Only quant.py's quantization is production code.",
            "Not real activations: synthetic distributions chosen to isolate the failure "
            "modes the lab's transforms exist for. Absolute levels do not transfer.",
            "Not cuDNN: schemes 3/4 model the per-tensor FP8 descale contract. Per claim "
            "K11 cuDNN's FP8 SDPA does not accept D256 at all.",
        ],
        "geometry": {
            "T": T_ROWS,
            "q_heads": Q_HEADS,
            "kv_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "sm_scale": SM_SCALE,
            "blk": BLK,
            "mask": "bottom-right causal, prefix = N - T",
            "eval": "dequantize-then-fp32-attention for every scheme (materialized "
            "score rectangle; T=128 keeps it small). P rounding, where "
            "modelled, uses the online running-tile-max form and is checked "
            "against oracle_a.attention by --check-oracle.",
        },
        "schemes": {
            "bf16": "fp32 attention on bf16-rounded inputs (the yardstick)",
            "bf16_arith": "bf16 inputs + bf16 P + bf16 output (fuller impl-swap floor)",
            "pertensor": "one amax/448 E4M3 descale per operand, no centering/rotation",
            "pertensor_p": "pertensor + E4M3 P at the online-softmax scale 448",
            "pertensor_pT": "pertensor + E4M3 P at one per-tensor amax on normalized P",
            "tile_nocenter": "per-row-Q / per-64-tile-K/V E4M3, no centering/rotation "
            "(granularity isolated from the transforms)",
            "lab": "quant.py quantize_q + gather_quantize_kv, dequantized "
            "(quantization error in isolation)",
            "lab_p": "lab + the kernel's normative E4M3 P rounding at 448*r_t",
        },
        "distributions": {
            k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()}
            for k, v in DISTRIBUTIONS.items()
        },
        "gen_chunk_rows": GEN_CHUNK,
        "env": _env(),
        "cells": [],
    }

    if args.check_oracle:
        record["oracle_a_crosscheck"] = check_oracle()

    if args.ablate is not None:
        targets = args.ablate or [
            f"{d}:{n}" for d in ("gaussian", "v_massive") for n in (args.depths[0], args.depths[-1])
        ]
        record["lab_ablation"] = {}
        record["lab_ablation_note"] = (
            "row_rel_l2_mean vs the fp32 reference with exactly ONE operand "
            "quantized through quant.py; p_only is E4M3 P at 448 with exact "
            "operands. Explains the level of the lab column."
        )
        for spec in targets:
            d, _, nn = spec.partition(":")
            print(f"[ablate {d}:{nn}]", flush=True)
            abl = ablate_cell(d, int(nn), DISTRIBUTIONS[d])
            record["lab_ablation"][spec] = abl
            print("    " + "  ".join(f"{k}={v:.4g}" for k, v in abl.items()), flush=True)
            torch.cuda.empty_cache()

    t0 = time.perf_counter()
    for dist in args.dists:
        cfg = DISTRIBUTIONS[dist]
        for n in args.depths:
            print(f"[{dist}] N={n}", flush=True)
            cell = {"dist": dist, "N": n, "seed_base": cfg["seed"]}
            try:
                cell.update(run_cell(dist, n, cfg))
                cell["status"] = "ok"
            except torch.cuda.OutOfMemoryError as exc:
                cell["status"] = "oom"
                cell["error"] = str(exc)[:400]
                print(f"    OOM: {str(exc)[:120]}", flush=True)
            except RuntimeError as exc:  # noqa: BLE001 -- one bad cell must not kill the sweep
                cell["status"] = "oom" if "out of memory" in str(exc).lower() else "error"
                cell["error"] = str(exc)[:400]
                print(f"    {cell['status']}: {str(exc)[:200]}", flush=True)
            torch.cuda.empty_cache()
            record["cells"].append(cell)
    record["total_seconds"] = time.perf_counter() - t0
    record["env_after"] = {"nvidia_smi": _nvidia_smi()}

    Path(args.out).write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}  ({record['total_seconds']:.1f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
