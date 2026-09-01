#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Adaptive EXACT distributional transforms inside the lab's attention operator.

The attention score is bilinear, so the space of transforms that leave the
operator's output mathematically unchanged is affine-linear and larger than the
one the shipped scheme uses:

  * **QK.**  For any invertible ``A`` and any per-channel offset ``mu``,

        q . k  ==  (A^-T q) . (A (k - mu))  +  (q . mu)

    and the residual ``q . mu`` is constant across every kv column of a query
    row, so row-wise softmax removes it exactly.  Both the transform ``A`` and
    the offset ``mu`` may be chosen from the data.
  * **PV.**  For any invertible ``M``, ``out = M^-1 (sum_j p_j M v_j)`` because
    ``sum_j p_j == 1``; the same identity absorbs an additive V offset, which is
    why the shipped epilogue can add ``vmean`` back exactly.
  * **Key order.**  Attention sums over keys.  Every prefix key is visible to
    every query row of an extend chunk, so any permutation of the PREFIX columns
    (applied to K and V rows together) leaves the output unchanged.  The current
    chunk's tail keys carry the causal mask and must keep their order.

The shipped scheme (``src/attn_kernel_lab/quant.py``, contract section 3) uses
the FIRST moment adaptively -- K mean-centering, V mean-centering with an exact
epilogue add-back -- and touches the second moment only through a FIXED,
data-independent orthonormal Hadamard plus per-64-tile local scales.  Nothing in
the operator is adapted to the second moment of the actual keys.

This probe measures what closing that gap is worth, on the donor probe's grid:

  ``lab_diag``      channel equilibration.  After centering, divide each K
                    channel by its per-channel spread (per KV head, over the
                    whole sequence) and multiply the matching Q channels by the
                    same diagonal.  The diagonals cancel in the dot product.
                    This is the SmoothQuant / Outlier-Suppression+ migration
                    move applied inside attention rather than to a linear layer.
  ``lab_whiten``    adaptive rotation.  ``A = Sigma^-1/2`` of the centered K
                    (per KV head, fp32, whole sequence); K gets ``A``, Q gets
                    ``A^-1 = Sigma^+1/2``.
  ``lab_whiten_bal``the balanced split ``Sigma^-1/4`` on K against
                    ``Sigma^+1/4`` on Q.  Quantization error is CONSERVED across
                    the dot product -- whatever the K side stops paying, the Q
                    side starts paying -- so the split exponent is the real
                    question and full whitening is only one point on it.
  ``lab_perm``      prefix-permutation tiling.  Reorder the prefix keys so each
                    64-token quantization tile groups tokens of similar
                    magnitude, and apply the identical permutation to V rows.
  ``lab_veq``       V-channel equilibration with the inverse folded into the
                    (fp32) epilogue that already adds ``vmean`` back.

Each is also measured composed with the existing Hadamard (``*_had``), because
the shipped baseline HAS that rotation and a variant that drops it is not
comparable to it.

Validity controls, both mandatory and both reported in the record:

  1. **Identity control.**  With the transform set to the plain Hadamard, no
     permutation and no V gain, this file's runner is the donor's ``lab`` /
     ``lab_p`` code path exactly.  Every metric must agree to 0.0.
  2. **Exactness control.**  Each transform plus its compensation is run at FULL
     precision -- no quantization anywhere -- against the transform-free fp32
     reference.  The invariance algebra has to hold before quantization is
     allowed to enter, and the residual is fp32 rounding only (~1e-6).

Every fixture, reference, metric and scheme convention is imported unchanged
from ``pertensor_vs_finegrained.py`` (the donor probe), whose bytes are sealed
by the published scorecard's ``donor_sha256``: this probe adds schemes in a
separate file and a separate JSON record and modifies nothing under ``src/``.
The transforms are injected through the workspace's own ``hadamard`` attribute,
which ``quant.py`` reads at call time and which broadcasts over a per-head batch
of matrices -- so the REAL quantizer runs, unmodified, on transformed operands.

Usage:
    python3 probes/quality/adaptive_transforms.py            # full grid
    python3 probes/quality/adaptive_transforms.py --quick    # 2x2 smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Same reason as the donor: the rotation GEMMs inside quant.py are reproducible
# only under a pinned cuBLAS workspace, and the variable must be set before
# torch creates its first cuBLAS handle (contract 3.1).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import importlib.util  # noqa: E402

import torch  # noqa: E402

HERE = Path(__file__).resolve().parent

_spec = importlib.util.spec_from_file_location(
    "pertensor_vs_finegrained", HERE / "pertensor_vs_finegrained.py"
)
donor = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = donor
_spec.loader.exec_module(donor)

labquant = donor.labquant
D = donor.HEAD_DIM
KVH = donor.KV_HEADS
QH = donor.Q_HEADS
GROUP = donor.GROUP
BLK = donor.BLK
T_ROWS = donor.T_ROWS
STAT_CHUNK = 32768  # statistics pass block; bounds transient fp32 memory

#: The committed preprocessing budget is a whole-schedule figure (K10: 6.19 s
#: for the 14-chunk x 16-layer schedule of ``workloads/generated``).  Statistics
#: measured here are for ONE terminal-depth call over 4 KV heads; the schedule
#: sums 3,428,223 kv tokens per layer across its 14 chunks, so one terminal call
#: is 446,335 of those and the schedule is 16 * 3428223 / 446335 = 122.9
#: terminal-call equivalents of gather traffic.
SCHEDULE_EQUIV_TERMINAL_CALLS = 16 * 3428223 / 446335
PREPROCESSING_BUDGET_S = 6.19

#: One EXTRA fixture beyond the donor's five, in the spirit of its own
#: ``v_massive`` ("EXTRA beyond the four requested").  The donor's five all have
#: essentially ISOTROPIC keys once centered -- gaussian, t3 and the RoPE-like
#: offset case are iid per channel, so their key covariance is a multiple of the
#: identity and there is no second-moment structure for an adaptive second
#: moment to exploit.  Measuring adaptive whitening only there would report a
#: null result with no way to tell "the transform does not work" from "the
#: fixture has nothing to transform".  This is the positive control.
EXTRA_DISTRIBUTIONS = {
    "k_aniso": dict(
        seed=20260901,
        kappa=1.0e4,
        doc="EXTRA (beyond the donor's five). K = z @ L with z ~ N(0,1) and L "
        "the symmetric square root of a per-KV-head covariance whose "
        "eigenvalues are a geometric spectrum of condition number 1e4 in a "
        "seeded random eigenbasis, renormalized to unit MEAN per-element "
        "variance so the score scale matches the donor's fixtures. Q,V ~ "
        "N(0,1) iid. The key second moment is anisotropic and NOT aligned to "
        "the channel axes, which is what an adaptive rotation exists for and "
        "what none of the donor's five fixtures contain. Q is left isotropic "
        "on purpose: the compensation A^-1 then genuinely costs the Q side "
        "something, which is the conservation question lab_whiten_bal asks.",
        mode="aniso",
    ),
    "k_chan": dict(
        seed=20260902,
        kappa=1.0e4,
        doc="EXTRA (beyond the donor's five). K channels are scaled by a fixed "
        "per-channel std whose spectrum is geometric with ratio 1e4 in a seeded "
        "channel order, renormalized to unit MEAN per-element variance; Q,V ~ "
        "N(0,1) iid. This is the AXIS-ALIGNED anisotropy an outlier-channel "
        "argument assumes and the case a diagonal equilibration can actually "
        "see -- k_aniso's covariance is anisotropic in a RANDOM basis, where "
        "the per-channel variances are all near their mean by concentration "
        "and a diagonal has nothing to work with. Separating the two is the "
        "only way to tell channel equilibration and adaptive rotation apart.",
        mode="chan",
    ),
}

#: Transform names, resolved to (bq, bk, idx, vgain) by ``_build_transform``.
#: Kept as an ordered list so the record's scheme order is stable and the
#: analysis tables can be generated from it.
TRANSFORMS = [
    "lab_diag",
    "lab_diag_had",
    "lab_whiten",
    "lab_whiten_had",
    "lab_whiten_bal",
    "lab_whiten_bal_had",
    "lab_perm",
    "lab_perm_v",
    "lab_veq",
    "lab_diag_veq_had",
]


def _scrubbed_env() -> dict:
    """The donor's env fingerprint with the raw GPU UUID replaced by the
    reviewed hardware label -- the donor file is sealed by ``donor_sha256`` and
    cannot take the fix itself; records in this repository never carry a
    physical-card serial (see docs/hardware-labels.md)."""
    env = donor._env()
    smi = env.get("nvidia_smi")
    if isinstance(smi, dict) and "uuid" in smi:
        smi["uuid"] = os.environ.get("ATTN_KERNEL_LAB_GPU_LABEL", "local-dev-gpu")
    return env


# --------------------------------------------------------------------------
# fixtures: the donor's five, plus the anisotropic positive control
# --------------------------------------------------------------------------


def _spectrum(cfg: dict, device) -> torch.Tensor:
    """[D] geometric eigenvalue spectrum, unit mean (so per-element variance is 1)."""
    i = torch.arange(D, device=device, dtype=torch.float32)
    lam = cfg["kappa"] ** (-i / (D - 1))
    return lam / lam.mean()


def _aniso_mixer(cfg: dict, device) -> torch.Tensor:
    """[KVH, D, D] symmetric psd mixing matrix with a fixed condition number.

    ``mode="aniso"``: the spectrum sits in a seeded RANDOM orthonormal basis, so
    the anisotropy is invisible to any per-channel statistic.
    ``mode="chan"``: the spectrum sits on the channel axes in a seeded order, so
    it is exactly a per-channel variance spread.
    """
    g = torch.Generator(device=device)
    g.manual_seed(cfg["seed"] + 555)
    lam = _spectrum(cfg, device)
    if cfg["mode"] == "chan":
        s = torch.stack([lam[torch.randperm(D, generator=g, device=device)] for _ in range(KVH)])
        return torch.diag_embed(s.sqrt())
    a = torch.randn(KVH, D, D, generator=g, device=device, dtype=torch.float32)
    u, _ = torch.linalg.qr(a)
    return u @ torch.diag_embed(lam.sqrt()) @ u.transpose(-1, -2)


def _fill_kv(dist: str, n: int, device, cfg: dict):
    """The donor's K/V fixtures, extended with the anisotropic control.

    The GEN_CHUNK block structure is the donor's, so a shallow cell stays a
    bit-exact prefix of a deep one on the extra fixture too.
    """
    if dist not in EXTRA_DISTRIBUTIONS:
        return donor._fill_kv(dist, n, device, cfg)
    base = cfg["seed"]
    gc = donor.GEN_CHUNK
    mix = _aniso_mixer(cfg, device)
    k = torch.empty(n, KVH, D, device=device, dtype=torch.float32)
    v = torch.empty(n, KVH, D, device=device, dtype=torch.float32)
    for c0 in range(0, n, gc):
        c1 = min(c0 + gc, n)
        ci = c0 // gc
        kc = donor._gen((gc, KVH, D), base + 10_000 + 7 * ci, device)
        vc = donor._gen((gc, KVH, D), base + 20_000 + 7 * ci, device)
        kc = torch.bmm(kc.permute(1, 0, 2), mix).permute(1, 0, 2)
        k[c0:c1] = kc[: c1 - c0]
        v[c0:c1] = vc[: c1 - c0]
        del kc, vc
    return k, v


def _gen_q(dist: str, device, cfg: dict) -> torch.Tensor:
    if dist not in EXTRA_DISTRIBUTIONS:
        return donor._gen_q(dist, device, cfg)
    return donor._gen((T_ROWS, QH, D), cfg["seed"] + 900_000, device)


def _all_distributions() -> dict:
    return {**donor.DISTRIBUTIONS, **EXTRA_DISTRIBUTIONS}


# --------------------------------------------------------------------------
# statistics: at most ONE extra pass over K/V beyond the shipped pipeline's two
# --------------------------------------------------------------------------


def _stats(k_bf: torch.Tensor, v_bf: torch.Tensor, n: int, h0: torch.Tensor) -> dict:
    """Every statistic any transform below needs, in two blocked passes.

    Pass A: channel sums for the K and V means (the shipped pipeline's own
    statistics gather already computes exactly these, so pass A is free).
    Pass B: the centered K covariance, per-token Linf of the centered-rotated K,
    per-token Linf of the centered V, and the per-channel V second moment.  This
    is the ONE extra full pass an adaptive transform costs; it is assumed to have
    the same access pattern as the existing centering pass, as directed.
    """
    dev = k_bf.device
    ksum = torch.zeros(KVH, D, device=dev, dtype=torch.float32)
    vsum = torch.zeros(KVH, D, device=dev, dtype=torch.float32)
    for c0 in range(0, n, STAT_CHUNK):
        c1 = min(c0 + STAT_CHUNK, n)
        ksum += k_bf[c0:c1].float().sum(dim=0)
        vsum += v_bf[c0:c1].float().sum(dim=0)
    kmean = ksum / n
    vmean = vsum / n

    sigma = torch.zeros(KVH, D, D, device=dev, dtype=torch.float32)
    vsq = torch.zeros(KVH, D, device=dev, dtype=torch.float32)
    kstat = torch.empty(KVH, n, device=dev, dtype=torch.float32)
    vstat = torch.empty(KVH, n, device=dev, dtype=torch.float32)
    for c0 in range(0, n, STAT_CHUNK):
        c1 = min(c0 + STAT_CHUNK, n)
        kc = (k_bf[c0:c1].float() - kmean).permute(1, 0, 2).contiguous()
        sigma += kc.transpose(1, 2) @ kc
        kstat[:, c0:c1] = (kc @ h0).abs().amax(dim=2)
        del kc
        vc = (v_bf[c0:c1].float() - vmean).permute(1, 0, 2).contiguous()
        vsq += (vc * vc).sum(dim=1)
        vstat[:, c0:c1] = vc.abs().amax(dim=2)
        del vc
    sigma /= n
    return {
        "kmean": kmean,
        "vmean": vmean,
        "sigma": sigma,
        "vrms": (vsq / n).sqrt(),
        "kstat": kstat,
        "vstat": vstat,
    }


def _sym_power(sigma: torch.Tensor, p: float, rel_floor: float = 1e-6):
    """([KVH,D,D] symmetric psd) ** p, with the spectrum floored relatively.

    Returns (matrix, condition_number_after_flooring).  The floor is what makes
    an inverse power well posed on a rank-deficient or near-deficient second
    moment; a real deployment would floor for the same reason and the record
    carries the resulting condition number so the reader can see how hard the
    floor had to work.
    """
    lam, u = torch.linalg.eigh(sigma.double())
    lmax = lam.amax(dim=1, keepdim=True)
    lam = lam.clamp_min(rel_floor * lmax)
    cond = float((lam.amax(dim=1) / lam.amin(dim=1)).max())
    out = (u * lam.pow(p)[:, None, :]) @ u.transpose(1, 2)
    return out.float(), cond


def _unit_geomean(x: torch.Tensor) -> torch.Tensor:
    """Rescale a positive per-channel vector to geometric mean 1, per head.

    The overall scale of a diagonal is irrelevant to exactness (it cancels) and
    irrelevant to quality (the tile scales are amax-derived), but pinning it
    keeps the transformed operands in the same numeric decade as the untransformed
    ones, which keeps the recorded per-tile scales comparable across schemes.
    """
    return x / x.log().mean(dim=1, keepdim=True).exp()


def _permutation(stat: torch.Tensor, n: int, prefix: int) -> torch.Tensor:
    """Sort the PREFIX positions by a per-token magnitude statistic.

    ``stat`` is [KVH, n]: one number per token per KV head.  A single index
    vector drives the gather for all four KV heads, so the per-head statistics
    must be reduced to one ordering; each head is normalized by its own mean
    first (heads differ in scale, and an unnormalized mean would let the loudest
    head dictate the order alone) and the heads are then averaged.

    Only ``[0, prefix)`` moves.  The current chunk's tail keys carry the causal
    mask and their order is load-bearing.
    """
    key = (stat[:, :prefix] / stat[:, :prefix].mean(dim=1, keepdim=True)).mean(dim=0)
    order = torch.argsort(key)
    idx = torch.arange(n, device=stat.device, dtype=torch.long)
    idx[:prefix] = order
    return idx


def _pow2(x: torch.Tensor) -> torch.Tensor:
    """Round a positive tensor to the nearest power of two.

    V equilibration is applied here by pre-scaling the bf16 V the quantizer
    reads.  A power-of-two gain makes that pre-scale EXACT in bf16 (it is an
    exponent adjust; the mantissa is untouched), so the measurement is of the
    equilibration and not of a bf16 rounding artefact introduced by the harness.
    In a kernel the gain folds into the fp32 conversion the V quantization pass
    already performs and the inverse folds into the epilogue that already adds
    ``vmean`` back, so neither the restriction nor the pass is needed there.
    """
    return torch.pow(2.0, torch.round(torch.log2(x)))


# --------------------------------------------------------------------------
# transforms
# --------------------------------------------------------------------------


def _build_transform(name: str, st: dict, h0: torch.Tensor, n: int, prefix: int) -> dict:
    """-> {bq [QH,D,D] or [D,D], bk [KVH,D,D] or [D,D], idx [n], vgain [KVH,D]|None}.

    Row-vector convention throughout, matching ``quant.py``: Q rows are right-
    multiplied by ``bq`` and centered K rows by ``bk``, so the score is
    ``q bq bk^T k^T`` and exactness is exactly the condition ``bq bk^T == I``.
    ``vgain`` is the diagonal ``M^-1`` the epilogue re-applies.
    """
    dev = h0.device
    idx = torch.arange(n, device=dev, dtype=torch.long)
    vgain = None
    info: dict = {}

    def _per_q(m: torch.Tensor) -> torch.Tensor:
        return m.repeat_interleave(GROUP, dim=0)

    if name == "identity":
        return {"bq": h0, "bk": h0, "idx": idx, "vgain": None, "info": {}}

    if name.startswith("lab_diag"):
        d = _unit_geomean(torch.diagonal(st["sigma"], dim1=1, dim2=2).clamp_min(1e-12).sqrt())
        info["diag_spread_ratio"] = float((d.amax(dim=1) / d.amin(dim=1)).max())
        bk = torch.diag_embed(1.0 / d)
        bq = torch.diag_embed(d)
    elif name.startswith("lab_whiten"):
        p = 0.25 if "bal" in name else 0.5
        bk, cond = _sym_power(st["sigma"], -p)
        bq, _ = _sym_power(st["sigma"], p)
        info["sigma_cond"] = cond
        info["split_exponent"] = p
    elif name.startswith("lab_perm") or name.startswith("lab_veq"):
        bk = bq = None
    else:
        raise ValueError(name)

    if name.startswith("lab_perm"):
        stat = st["vstat"] if name.endswith("_v") else st["kstat"]
        idx = _permutation(stat, n, prefix)
        bk, bq = h0, h0
    elif "veq" in name:
        w = _pow2(_unit_geomean(st["vrms"].clamp_min(1e-12)))
        vgain = w
        info["v_gain_ratio"] = float((w.amax(dim=1) / w.amin(dim=1)).max())
        if bk is None:  # plain lab_veq: the shipped Hadamard on QK, V equalized
            bk, bq = h0, h0

    if name.endswith("_had") or name == "lab_diag_veq_had":
        bk, bq = bk @ h0, bq @ h0
    if bk.dim() == 3:
        bq = _per_q(bq)
    return {"bq": bq, "bk": bk, "idx": idx, "vgain": vgain, "info": info}


# --------------------------------------------------------------------------
# the lab scheme, run through quant.py, under a transform
# --------------------------------------------------------------------------


def _lab_run(ws, q_bf, k_bf, v_bf, tf: dict, geo: dict, ref: torch.Tensor,
             floors: tuple | None = None) -> dict:
    """Run schemes 5 and 5b (``lab`` / ``lab_p``) under one transform.

    Line for line the donor's ``run_cell`` lab section, with three injections and
    nothing else: ``ws.hadamard`` carries ``bq`` for the Q quantization and
    ``bk`` for the K quantization (``quant.py`` reads the attribute at call time
    and its ``x @ ws.hadamard`` broadcasts over a leading per-head batch), ``idx``
    carries the key permutation, and ``vgain`` is re-applied to the fp32 output,
    which is where the shipped epilogue's ``vmean`` add-back already lives.
    """
    n, npad, ntiles, prefix, tail, hb = (
        geo[x] for x in ("n", "npad", "ntiles", "prefix", "tail", "hb")
    )
    dev = q_bf.device

    v_use = v_bf
    if tf["vgain"] is not None:
        v_use = v_bf * (1.0 / tf["vgain"]).to(torch.bfloat16)  # exact: power-of-two gain

    ws.hadamard = tf["bq"]
    q8, qs, mpad = labquant.quantize_q(ws, q_bf, donor.SM_SCALE, qk_i8=True, rotate=True)
    q_lab = q8.view(torch.int8).float() * qs[:, :, None]
    q_step = float((qs[:, :T_ROWS] / donor.SM_SCALE).mean())
    # q_lab carries sm_scale (quantize_q folds it into the returned scales);
    # divide it back out so q_step / q_rms is a pure relative-step number
    # comparable with the K side, which carries no sm_scale.
    q_rms = float(q_lab[:, :T_ROWS].pow(2).mean().sqrt()) / donor.SM_SCALE

    ws.hadamard = tf["bk"]
    packs = labquant.gather_quantize_kv(
        ws, k_bf, v_use, tf["idx"], need_vt8=True, need_vb16=False,
        center_k=True, qk_i8=True, rotate=True,
    )
    assert mpad == T_ROWS and packs["ntmax"] == ntiles, (mpad, packs["ntmax"], ntiles)
    if v_use is not v_bf:
        del v_use
    vst = ws.get("vscale_t", (KVH, ntiles), torch.float32)
    k8 = packs["k8"].view(torch.int8)
    ks, vsmax, vinvr, vmean = (packs[x] for x in ("kscale", "vscale", "vinvr", "vmean"))
    sigma64 = torch.tensor(labquant.SIGMA64, device=dev, dtype=torch.long)
    inv = torch.empty_like(sigma64)
    inv[sigma64] = torch.arange(BLK, device=dev)

    def _k_lab(kvh):
        kf = k8[kvh].view(ntiles, BLK, D).float()
        kf.mul_(ks[kvh][:, None, None])
        return kf.reshape(npad, D)

    def _v8_raw(kvh):
        vt = packs["vt8"].view(KVH, ntiles, D, BLK)[kvh]
        return (
            vt.view(torch.float8_e4m3fn)
            .permute(0, 2, 1)
            .index_select(1, inv)
            .float()
            .reshape(npad, D)
        )

    def kv_lab(kvh):
        vf = _v8_raw(kvh).view(ntiles, BLK, D)
        vf.mul_(vst[kvh][:, None, None])
        vf = vf.reshape(npad, D)
        vf.add_(vmean[kvh])
        return _k_lab(kvh), vf, None

    def kw_lab_p(kvh, _extra):
        r_t = (1.0 / vinvr[kvh, :ntiles]).clamp(1.0 / 16.0, 1.0)
        return {
            "fold": (donor.FP8_MAX * r_t).view(1, ntiles, 1),
            "vscale": float(vsmax[kvh]),
            "vmean": vmean[kvh],
        }

    def _epilogue(out: torch.Tensor) -> torch.Tensor:
        if tf["vgain"] is not None:
            out.mul_(tf["vgain"].repeat_interleave(GROUP, dim=0)[None, :, :])
        return out

    none_kw = lambda *_: {}  # noqa: E731 -- schemes with no extra PV state
    out_lab = _epilogue(
        donor._run_scheme(
            lambda h: q_lab[h], kv_lab, n, npad, prefix, tail, "plain", none_kw, hb
        )
    )
    res = {"lab": donor._metrics(out_lab, ref)}
    del out_lab
    out_p = _epilogue(
        donor._run_scheme(
            lambda h: q_lab[h],
            lambda kvh: (_k_lab(kvh), _v8_raw(kvh), None),
            n, npad, prefix, tail, "p_online", kw_lab_p, hb,
        )
    )
    res["lab_p"] = donor._metrics(out_p, ref)
    k_rms = float(torch.stack([_k_lab(kvh).pow(2).mean() for kvh in range(KVH)]).mean().sqrt())
    # r_t = vs_t / vs_max is the per-tile V dequant ratio quant.py folds into the
    # packed P constant (448 * r_t), floored at 1/16 to keep packed P out of E4M3
    # subnormals.  Its DISPERSION is therefore a P-quality statistic as much as a
    # V one: tiles whose V is quiet relative to the loudest tile get a coarse P
    # grid, and every tile pinned at the floor has given up on finer V scaling.
    # A transform that homogenizes per-tile V magnitude moves r_t toward 1 and
    # improves the P path without touching P's own arithmetic.
    r_t = 1.0 / vinvr[:, :ntiles]
    res["operands"] = {
        # Relative quantization step per operand: the step size an int8/fp8 grid
        # imposes, divided by the operand's own RMS after the transform.  A
        # transform that helps K by hurting Q shows up here as one number falling
        # while the other rises -- the conservation the balanced variant asks about.
        "q_rel_step": q_step / q_rms,
        "k_rel_step": float(ks[:, :ntiles].mean()) / k_rms,
        "v_scale_mean": float(vst[:, :ntiles].mean()),
        "q_rms": q_rms,
        "k_rms": k_rms,
        "v_ratio_mean": float(r_t.mean()),
        "v_ratio_min": float(r_t.min()),
        "v_floor_fraction": float((r_t <= 1.0 / 16.0 * (1.0 + 1e-6)).float().mean()),
    }
    if floors is not None:
        # What ANY transform confined to one operand's grid can possibly reach.
        # qk_exact_floor: Q and K exact (the Hadamard is orthonormal, so rotating
        # an unquantized operand is exact), V and P through the lab path -- the
        # bound on lab_diag / lab_whiten* / lab_perm, which touch the QK grid
        # only. v_exact_floor: Q and K through the lab path, V exact, P at the
        # plain 448 fold -- the bound on lab_veq, and slightly optimistic
        # because an exact V also removes the per-tile r_t coupling from P.
        h0, kmean = floors
        qe = (q_bf.float().permute(1, 0, 2) @ h0).permute(1, 0, 2) * donor.SM_SCALE

        def _k_exact(kvh):
            kk = torch.zeros(npad, D, device=dev, dtype=torch.float32)
            kk[:n] = (k_bf[:, kvh].float() - kmean[kvh]) @ h0
            return kk

        res["qk_exact_floor"] = donor._metrics(
            _epilogue(
                donor._run_scheme(
                    lambda h: qe[:, h],
                    lambda kvh: (_k_exact(kvh), _v8_raw(kvh), None),
                    n, npad, prefix, tail, "p_online", kw_lab_p, hb,
                )
            ),
            ref,
        )
        del qe
        res["v_exact_floor"] = donor._metrics(
            donor._run_scheme(
                lambda h: q_lab[h],
                lambda kvh: (_k_lab(kvh), donor._pad_head(v_bf, kvh, npad, n), None),
                n, npad, prefix, tail, "p_online",
                lambda *_: {"fold": donor.FP8_MAX}, hb,
            ),
            ref,
        )
        torch.cuda.empty_cache()
    del out_p
    return res


# --------------------------------------------------------------------------
# exactness control: the algebra, at full precision, before quantization
# --------------------------------------------------------------------------


def _exactness(dist: str, n: int, cfg: dict, names: list[str]) -> dict:
    """Run transform + compensation with NO quantization against the fp32 ref.

    If a transform is exact, this is the identity map composed with fp32
    rounding, and the row-relative L2 must land at the fp32 noise level.  If it
    is not exact, no amount of quantization measurement downstream means
    anything.
    """
    dev = torch.device("cuda")
    prefix = n - T_ROWS
    npad = (n + BLK - 1) // BLK * BLK
    tail = donor._tail_mask(dev)
    hb = donor._head_batch_for(npad)
    q_f32 = _gen_q(dist, dev, cfg)
    k_f32, v_f32 = _fill_kv(dist, n, dev, cfg)
    ref = donor._fp32_reference(q_f32, k_f32, v_f32, n, npad, prefix, tail, hb)
    k_bf, v_bf = k_f32.to(torch.bfloat16), v_f32.to(torch.bfloat16)
    h0 = labquant._hadamard(D, dev)
    st = _stats(k_bf, v_bf, n, h0)
    none_kw = lambda *_: {}  # noqa: E731
    out: dict = {}
    for name in names:
        tf = _build_transform(name, st, h0, n, prefix)
        bq = tf["bq"] if tf["bq"].dim() == 3 else tf["bq"].expand(QH, D, D)
        bk = tf["bk"] if tf["bk"].dim() == 3 else tf["bk"].expand(KVH, D, D)
        qt = torch.bmm(q_f32.permute(1, 0, 2), bq).permute(1, 0, 2).contiguous()
        gain = tf["vgain"]

        def kv_fn(kvh, _bk=bk, _gain=gain, _tf=tf):
            kk = torch.zeros(npad, D, device=dev, dtype=torch.float32)
            vv = torch.zeros(npad, D, device=dev, dtype=torch.float32)
            kk[:n] = (k_f32[:, kvh].index_select(0, _tf["idx"]) - st["kmean"][kvh]) @ _bk[kvh]
            vv[:n] = v_f32[:, kvh].index_select(0, _tf["idx"])
            if _gain is not None:
                vv[:n] /= _gain[kvh]
            return kk, vv, None

        got = donor._run_scheme(
            lambda h: qt[:, h] * donor.SM_SCALE, kv_fn, n, npad, prefix, tail,
            "plain", none_kw, hb,
        )
        if gain is not None:
            got.mul_(gain.repeat_interleave(GROUP, dim=0)[None, :, :])
        m = donor._metrics(got, ref)
        out[name] = {
            "row_rel_l2_mean": m["row_rel_l2_mean"],
            "row_rel_l2_max": m["row_rel_l2_max"],
            "cos_sim_worst_row": m["cos_sim_worst_row"],
            **tf["info"],
        }
        got = qt = None
        torch.cuda.empty_cache()
    q_f32 = k_f32 = v_f32 = k_bf = v_bf = ref = st = None
    torch.cuda.empty_cache()
    return out


# --------------------------------------------------------------------------
# permutation ablation: WHERE a permutation's win comes from
# --------------------------------------------------------------------------


def perm_ablation(dist: str, n: int, cfg: dict, names: list[str]) -> dict:
    """Decompose a key-permutation cell into its V-reconstruction and V-path parts.

    The headline ``lab_perm_v`` result needed a mechanism and the first guess --
    "sorting makes the per-tile V ratios r_t friendlier, so the packed P grid
    gets finer" -- is contradicted by the r_t statistics the cells record (r_t
    falls and the 1/16 floor binds MORE under the permutation).  So this
    measures, rather than asserts:

      ``v_recon_rel_fro``  ||dequant(V) - V||_F / ||V||_F over the gathered
                           rows.  Whether the permutation made the stored V
                           more accurate AS A TENSOR.
      ``v_only``           attention-output error with Q, K exact and fp32 P,
                           so the ONLY quantized operand is V.  Whether the
                           permutation made V's contribution TO THE OUTPUT
                           smaller, which is a different question -- the output
                           is a p-weighted sum, so per-tile error placement
                           matters even when the Frobenius norm does not.
      ``qk_only``          the mirror: Q and K through the lab path, V exact,
                           fp32 P.
    """
    dev = torch.device("cuda")
    prefix = n - T_ROWS
    npad = (n + BLK - 1) // BLK * BLK
    ntiles = npad // BLK
    tail = donor._tail_mask(dev)
    hb = donor._head_batch_for(npad)
    q_f32 = _gen_q(dist, dev, cfg)
    k_f32, v_f32 = _fill_kv(dist, n, dev, cfg)
    ref = donor._fp32_reference(q_f32, k_f32, v_f32, n, npad, prefix, tail, hb)
    k_bf, v_bf = k_f32.to(torch.bfloat16), v_f32.to(torch.bfloat16)
    ws = labquant.FP8PrefillWorkspace(dev)
    h0 = ws.hadamard
    st = _stats(k_bf, v_bf, n, h0)
    sigma64 = torch.tensor(labquant.SIGMA64, device=dev, dtype=torch.long)
    inv = torch.empty_like(sigma64)
    inv[sigma64] = torch.arange(BLK, device=dev)
    none_kw = lambda *_: {}  # noqa: E731
    out: dict = {}
    for name in names:
        tf = _build_transform(name, st, h0, n, prefix)
        ws.hadamard = h0
        packs = labquant.gather_quantize_kv(
            ws, k_bf, v_bf, tf["idx"], need_vt8=True, need_vb16=False,
            center_k=True, qk_i8=True, rotate=True,
        )
        vst = ws.get("vscale_t", (KVH, ntiles), torch.float32)
        q8, qs, _ = labquant.quantize_q(
            ws, q_f32.to(torch.bfloat16), donor.SM_SCALE, qk_i8=True, rotate=True
        )
        q_lab = q8.view(torch.int8).float() * qs[:, :, None]
        k8 = packs["k8"].view(torch.int8)
        qe = (q_f32.permute(1, 0, 2) @ h0).permute(1, 0, 2) * donor.SM_SCALE
        num = den = 0.0

        def _vdeq(kvh, _p=packs, _vst=vst):
            vt = _p["vt8"].view(KVH, ntiles, D, BLK)[kvh].view(torch.float8_e4m3fn)
            raw = vt.permute(0, 2, 1).index_select(1, inv).float().reshape(ntiles, BLK, D)
            return (raw * _vst[kvh][:, None, None]).reshape(npad, D) + _p["vmean"][kvh]

        for kvh in range(KVH):
            deq = _vdeq(kvh)
            tru = v_f32[:, kvh].index_select(0, tf["idx"])
            num += float((deq[:n] - tru).pow(2).sum())
            den += float(tru.pow(2).sum())
            del deq, tru

        def _kv_vquant(kvh, _tf=tf):
            kk = torch.zeros(npad, D, device=dev, dtype=torch.float32)
            kk[:n] = (k_f32[:, kvh].index_select(0, _tf["idx"]) - st["kmean"][kvh]) @ h0
            return kk, _vdeq(kvh), None

        def _kv_kquant(kvh, _tf=tf, _ks=packs["kscale"]):
            kf = k8[kvh].view(ntiles, BLK, D).float()
            kf.mul_(_ks[kvh][:, None, None])
            vv = torch.zeros(npad, D, device=dev, dtype=torch.float32)
            vv[:n] = v_f32[:, kvh].index_select(0, _tf["idx"])
            return kf.reshape(npad, D), vv, None

        out[name] = {
            "v_recon_rel_fro": (num / den) ** 0.5,
            "v_only": donor._metrics(
                donor._run_scheme(
                    lambda h: qe[:, h], _kv_vquant, n, npad, prefix, tail,
                    "plain", none_kw, hb,
                ),
                ref,
            )["row_rel_l2_mean"],
            "qk_only": donor._metrics(
                donor._run_scheme(
                    lambda h: q_lab[h], _kv_kquant, n, npad, prefix, tail,
                    "plain", none_kw, hb,
                ),
                ref,
            )["row_rel_l2_mean"],
        }
        qe = q_lab = None
        torch.cuda.empty_cache()
    q_f32 = k_f32 = v_f32 = k_bf = v_bf = ref = st = ws = None
    torch.cuda.empty_cache()
    return out


# --------------------------------------------------------------------------
# one cell
# --------------------------------------------------------------------------


def run_cell(dist: str, n: int, cfg: dict, names: list[str]) -> dict:
    dev = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    prefix = n - T_ROWS
    npad = (n + BLK - 1) // BLK * BLK
    geo = {
        "n": n, "npad": npad, "ntiles": npad // BLK, "prefix": prefix,
        "tail": donor._tail_mask(dev), "hb": donor._head_batch_for(npad),
    }

    q_f32 = _gen_q(dist, dev, cfg)
    k_f32, v_f32 = _fill_kv(dist, n, dev, cfg)
    q_bf, k_bf, v_bf = (x.to(torch.bfloat16) for x in (q_f32, k_f32, v_f32))
    ref = donor._fp32_reference(q_f32, k_f32, v_f32, n, npad, prefix, geo["tail"], geo["hb"])
    del q_f32, k_f32, v_f32
    torch.cuda.empty_cache()

    none_kw = lambda *_: {}  # noqa: E731 -- schemes with no extra PV state
    q_bff = q_bf.float()
    cell_bf16 = donor._metrics(
        donor._run_scheme(
            lambda h: q_bff[:, h] * donor.SM_SCALE,
            lambda kvh: (
                donor._pad_head(k_bf, kvh, npad, n),
                donor._pad_head(v_bf, kvh, npad, n),
                None,
            ),
            n, npad, prefix, geo["tail"], "plain", none_kw, geo["hb"],
        ),
        ref,
    )
    q_bff = None
    torch.cuda.empty_cache()

    ws = labquant.FP8PrefillWorkspace(dev)
    h0 = ws.hadamard

    torch.cuda.synchronize()
    t_stat = time.perf_counter()
    st = _stats(k_bf, v_bf, n, h0)
    torch.cuda.synchronize()
    stat_seconds = time.perf_counter() - t_stat

    cell: dict = {
        "dist": dist, "N": n, "seed_base": cfg["seed"], "npad": npad,
        "schemes": {}, "transform_info": {}, "bf16_anchor": cell_bf16,
        "stats_pass_seconds": stat_seconds,
        "stats_pass_schedule_equiv_seconds": stat_seconds * SCHEDULE_EQUIV_TERMINAL_CALLS,
    }

    for name in ["identity", *names]:
        torch.cuda.synchronize()
        t_b = time.perf_counter()
        tf = _build_transform(name, st, h0, n, prefix)
        torch.cuda.synchronize()
        build_s = time.perf_counter() - t_b
        res = _lab_run(
            ws, q_bf, k_bf, v_bf, tf, geo, ref,
            floors=(h0, st["kmean"]) if name == "identity" else None,
        )
        if name == "identity":
            cell["identity_control"] = {"lab": res["lab"], "lab_p": res["lab_p"]}
            cell["floors"] = {
                "qk_exact_floor": res["qk_exact_floor"],
                "v_exact_floor": res["v_exact_floor"],
            }
        else:
            cell["schemes"][name] = {"lab": res["lab"], "lab_p": res["lab_p"]}
        cell["transform_info"][name] = {
            **tf["info"], "build_seconds": build_s, "operands": res["operands"],
        }
        del tf, res
        torch.cuda.empty_cache()

    ws.hadamard = h0
    del st, ws
    torch.cuda.empty_cache()

    cell["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30
    cell["seconds"] = time.perf_counter() - t0
    q_bf = k_bf = v_bf = ref = None
    torch.cuda.empty_cache()
    return cell


def _max_metric_delta(a: dict, b: dict) -> float:
    """Largest absolute difference over the shared scalar metric fields."""
    worst = 0.0
    for k, va in a.items():
        vb = b.get(k)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            worst = max(worst, abs(float(va) - float(vb)))
    return worst


# --------------------------------------------------------------------------


def main() -> int:
    dists_all = _all_distributions()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--depths", type=int, nargs="+", default=donor.DEPTHS)
    ap.add_argument("--dists", nargs="+", default=list(dists_all))
    ap.add_argument("--transforms", nargs="+", default=TRANSFORMS)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--exact-n", type=int, default=4096)
    ap.add_argument("--out", default=str(HERE / "adaptive_transforms.json"))
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device", file=sys.stderr)
        return 2
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.quick:
        args.depths = [4096, 32768]
        args.dists = ["gaussian", "k_aniso"]

    record = {
        "probe": "adaptive_transforms",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "question": "What do EXACT adaptive second-moment transforms -- channel "
        "equilibration, adaptive whitening at a split exponent, and prefix-key "
        "permutation tiling -- buy the lab's quantization scheme, whose current "
        "adaptivity stops at the first moment (mean-centering) plus a FIXED "
        "Hadamard rotation?",
        "frame": "The score is bilinear, so q.k == (A^-T q).(A(k-mu)) up to a "
        "per-row constant softmax removes, and out == M^-1 (sum p_j M v_j) since "
        "sum p == 1. Attention also sums over keys, so any permutation of the "
        "PREFIX columns applied to K and V rows together is exact. Every scheme "
        "here is an exact rewrite of the same operator; only the quantization "
        "grid it presents to quant.py changes.",
        "not_claimed": [
            "Not a kernel measurement: quant.py's REAL quantizer runs, but the "
            "attention itself is the donor probe's materialized evaluation form, "
            "not the fused kernel. Tiling and accumulation order differ, as they "
            "do for every scheme on this grid.",
            "Not real activations: seeded synthetic fixtures, boundary (a) only, "
            "exactly as the donor probe declares. The anisotropic k_aniso "
            "fixture is a constructed positive control, not a captured "
            "covariance; its condition number was chosen, not measured.",
            "Not a novelty claim. Channel equilibration is the SmoothQuant / "
            "Outlier-Suppression+ move, adaptive whitening is the adaptive "
            "cousin of QuaRot/SpinQuant/FlashAttention-3 incoherent processing, "
            "and V-side smoothing is in the SageAttention2 family. Prefix-key "
            "permutation tiling for paged EXTEND is not something the author has "
            "found published, but absence of a citation is not evidence of "
            "absence and no priority is claimed.",
            "Preprocessing costs are measured as torch-level statistics passes "
            "on this evaluation harness, not as a fused kernel's preprocessing. "
            "They bound what a transform would cost, they do not predict it.",
        ],
        "controls": {
            "identity": "with the transform set to the plain Hadamard, no "
            "permutation and no V gain, this file's runner IS the donor's lab / "
            "lab_p code path; every metric must agree to 0.0.",
            "exactness": "each transform + compensation run at FULL precision "
            "against the transform-free fp32 reference; residual must be fp32 "
            "rounding only.",
        },
        "budget": {
            "committed_preprocessing_seconds_per_schedule": PREPROCESSING_BUDGET_S,
            "schedule_equiv_terminal_calls": SCHEDULE_EQUIV_TERMINAL_CALLS,
            "note": "16 layers x 3,428,223 kv tokens summed over the 14-chunk "
            "schedule = 122.9 terminal-depth-call equivalents of gather traffic.",
        },
        "distributions": {k: v["doc"] for k, v in dists_all.items()},
        "transforms": args.transforms,
        "env": _scrubbed_env(),
        "cells": [],
    }

    t0 = time.perf_counter()
    print(f"exactness control at N={args.exact_n} ...", flush=True)
    record["exactness_control"] = {
        d: _exactness(d, args.exact_n, dists_all[d], args.transforms) for d in args.dists
    }
    worst_exact = max(
        v["row_rel_l2_max"]
        for per in record["exactness_control"].values()
        for v in per.values()
    )
    record["exactness_control_worst_row_rel_l2"] = worst_exact
    print(f"  worst exactness residual (row_rel_l2_max): {worst_exact:.3e}", flush=True)

    max_ident = 0.0
    for dist in args.dists:
        cfg = dists_all[dist]
        for n in args.depths:
            print(f"[{dist}] N={n}", flush=True)
            try:
                base = donor.run_cell(dist, n, cfg, verbose=False) \
                    if dist not in EXTRA_DISTRIBUTIONS else None
                torch.cuda.empty_cache()
                cell = run_cell(dist, n, cfg, args.transforms)
                if base is not None:
                    cell["donor_schemes"] = base["schemes"]
                    dl = _max_metric_delta(cell["identity_control"]["lab"], base["schemes"]["lab"])
                    dp = _max_metric_delta(
                        cell["identity_control"]["lab_p"], base["schemes"]["lab_p"]
                    )
                    cell["identity_delta"] = {"lab": dl, "lab_p": dp}
                    max_ident = max(max_ident, dl, dp)
                    ref_p = base["schemes"]["lab_p"]["row_rel_l2_mean"]
                    anchor = base["schemes"]["bf16"]["row_rel_l2_mean"]
                else:
                    ref_p = cell["identity_control"]["lab_p"]["row_rel_l2_mean"]
                    anchor = cell["bf16_anchor"]["row_rel_l2_mean"]
                cell["status"] = "ok"
                print(
                    f"    lab_p baseline {ref_p * 100:8.4f}%"
                    f"   (bf16 anchor {anchor * 100:.4f}%)"
                    f"   stats pass {cell['stats_pass_seconds'] * 1e3:.1f} ms",
                    flush=True,
                )
                for name in args.transforms:
                    m = cell["schemes"][name]["lab_p"]["row_rel_l2_mean"]
                    o = cell["transform_info"][name]["operands"]
                    print(
                        f"    {name:20s} {m * 100:8.4f}%  ({m / ref_p:6.3f}x lab_p)"
                        f"  q_step {o['q_rel_step']:.4f} k_step {o['k_rel_step']:.4f}",
                        flush=True,
                    )
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                cell = {"dist": dist, "N": n, "status": "error", "error": str(exc)[:400]}
                print(f"    error: {str(exc)[:200]}", flush=True)
            torch.cuda.empty_cache()
            record["cells"].append(cell)

    record["identity_control_max_delta"] = max_ident

    ab_dists = [d for d in ("gaussian", "heavy_t3", "v_outlier") if d in args.dists]
    ab_n = max(args.depths)
    print(f"permutation ablation at N={ab_n} ...", flush=True)
    record["perm_ablation_note"] = (
        "row_rel_l2_mean with exactly one operand group quantized, plus the V "
        "tensor's own reconstruction error, at the deepest depth. Isolates "
        "whether a key permutation's win is V-reconstruction, V-placement, or "
        "neither."
    )
    record["perm_ablation"] = {
        d: perm_ablation(d, ab_n, dists_all[d], ["identity", "lab_perm", "lab_perm_v"])
        for d in ab_dists
    }
    for d, per in record["perm_ablation"].items():
        for nm, v in per.items():
            print(
                f"    {d:10s} {nm:12s} v_recon {v['v_recon_rel_fro']:.6f}"
                f"  v_only {v['v_only'] * 100:.4f}%  qk_only {v['qk_only'] * 100:.4f}%",
                flush=True,
            )

    record["total_seconds"] = time.perf_counter() - t0
    record["env_after"] = _scrubbed_env()
    Path(args.out).write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\nidentity-control max metric delta: {max_ident:.3e}")
    print(f"exactness-control worst residual:  {worst_exact:.3e}")
    print(f"wrote {args.out}  ({record['total_seconds']:.1f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
