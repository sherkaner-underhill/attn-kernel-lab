#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Emulate FlashInfer PR #4714's FP8 quantization scheme on the 25-cell grid.

The neighbouring proposal (flashinfer-ai/flashinfer#4714, "cute-dsl-prims
backend for SM120 FP8 GQA prefill", pinned at its head commit ``004d1aea``)
quantizes differently from this lab's operator, and the difference is the
question this probe answers. Its scheme, read from that PR's own source:

  * Q, K and V are E4M3 tensors with ONE optional scalar (per-tensor) scale
    each — ``q_scale``/``k_scale``/``v_scale`` are ``Optional[float]`` in
    ``flashinfer/attention/cute_dsl/sm120_fmha.py``; ``q_scale*k_scale`` folds
    into the softmax scale and ``v_scale`` into the output scale. The KV pool
    itself stores E4M3 (storage quantization).
  * QK is FP8xFP8 MMA with fp32 accumulation; the online softmax runs in fp32
    (base-2, running tile max).
  * P is converted to E4M3 **directly, with no scale**
    (``fmha_prefill_fp8_tma.py``: ``p = exp2(...)`` then
    ``cvt_f32x2_to_f8x2(p1, p0, self.in_dtype)``), while the row-sum
    denominator accumulates the PRE-conversion fp32 values. E4M3's smallest
    denormal is 2^-9, so probabilities below ~2e-3 quantize coarsely and below
    ~2^-10 flush to zero.
  * No mean-centering and no rotation anywhere.

Faithfulness rules, applied deliberately:

  * The scheme is granted BEST-CASE per-tensor calibration: each operand's
    scale is its own amax/448 (the same charity the donor probe extends to the
    cuDNN per-tensor cells). The PR's *default* is scale=1.0, which would saturate
    catastrophically on the heavy-tailed fixtures; using it would be a straw
    man, so it is not used.
  * The unscaled-P conversion is reproduced exactly as their kernel does it.
    It is NOT "fixed": the 448-fold diagnostic cell below exists precisely to
    isolate what that choice costs, and doubles as a cross-check — with
    fold=448 this file's PV reimplementation must agree with the donor
    probe's ``pertensor_p`` scheme computed in the same cell.

Every fixture, reference, metric and scheme convention is imported unchanged
from ``pertensor_vs_finegrained.py`` (the donor probe), whose bytes are sealed
by the published scorecard's ``donor_sha256`` and therefore must not change:
this probe adds cells in a separate file and a separate JSON record.

Usage:
    python3 probes/quality/fp8_pool_emulation.py            # full 5x5 grid
    python3 probes/quality/fp8_pool_emulation.py --quick    # 2x2 smoke
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent

_spec = importlib.util.spec_from_file_location(
    "pertensor_vs_finegrained", HERE / "pertensor_vs_finegrained.py"
)
donor = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = donor
_spec.loader.exec_module(donor)

PR_PIN = {
    "pr": "flashinfer-ai/flashinfer#4714",
    "head_commit": "004d1aea",
    "scale_contract": "per-tensor scalar q_scale/k_scale/v_scale "
    "(flashinfer/attention/cute_dsl/sm120_fmha.py, folded into sm/output scale)",
    "p_contract": "E4M3 conversion of exp2 probabilities with NO scale; fp32 "
    "pre-conversion row sums (fmha_prefill_fp8_tma.py online_softmax)",
}


def _pv_pr4714(s: torch.Tensor, vf: torch.Tensor, fold: float) -> torch.Tensor:
    """The donor's ``p_online`` PV with the denominator generalized to ``fold``.

    The donor hardcodes ``acc /= 448 * lsum`` because its only fold is 448.
    #4714's fold is 1.0 (no P scale), so the denominator must be
    ``fold * lsum``; at fold=448 this function reproduces the donor's
    ``p_online`` arithmetic operation-for-operation, which the runner asserts.
    ``s`` is consumed in place; returns [g*T, D] fp32.
    """
    g, t, c = s.shape
    b = g * t
    nt = c // donor.BLK
    sv = s.view(b, nt, donor.BLK)
    m_run = sv.amax(dim=-1).cummax(dim=1).values
    neg = torch.isneginf(m_run)
    m_safe = m_run.masked_fill(neg, 0.0)
    w = (m_safe - m_safe[:, -1:]).exp().masked_fill_(neg, 0.0)
    sv.sub_(m_safe[:, :, None]).exp_()
    lsum = (sv.sum(dim=-1) * w).sum(dim=-1, keepdim=True)
    sv.mul_(fold)
    sv.copy_(sv.to(torch.float8_e4m3fn))
    sv.mul_(w[:, :, None])
    acc = s.view(b, c) @ vf
    acc.div_(fold * lsum)
    return acc


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


def emulation_schemes(dist: str, n: int, cfg: dict) -> dict:
    """The two #4714 cells for one (dist, N), on the donor's seeded fixtures."""
    dev = torch.device("cuda")
    prefix = n - donor.T_ROWS
    npad = (n + donor.BLK - 1) // donor.BLK * donor.BLK
    tail = donor._tail_mask(dev)
    hb = donor._head_batch_for(npad)

    q_f32 = donor._gen_q(dist, dev, cfg)
    k_f32, v_f32 = donor._fill_kv(dist, n, dev, cfg)
    q_bf, k_bf, v_bf = (x.to(torch.bfloat16) for x in (q_f32, k_f32, v_f32))

    ref = donor._fp32_reference(q_f32, k_f32, v_f32, n, npad, prefix, tail, hb)
    del k_f32, v_f32
    torch.cuda.empty_cache()

    def _amax(x):
        mn, mx = torch.aminmax(x)
        return max(abs(float(mn)), abs(float(mx)))

    sq = _amax(q_bf) / donor.FP8_MAX
    sk = _amax(k_bf) / donor.FP8_MAX
    sv_ = _amax(v_bf) / donor.FP8_MAX
    q_pt = donor._e4m3_rt_(q_bf.float(), sq)

    def kv_pt(kvh):
        return (
            donor._e4m3_rt_(donor._pad_head(k_bf, kvh, npad, n), sk),
            donor._e4m3_rt_(donor._pad_head(v_bf, kvh, npad, n), sv_),
            None,
        )

    def run(fold: float) -> torch.Tensor:
        out = torch.empty(
            donor.T_ROWS, donor.Q_HEADS, donor.HEAD_DIM, device=dev, dtype=torch.float32
        )
        for kvh in range(donor.KV_HEADS):
            kf, vf, _ = kv_pt(kvh)
            heads = list(range(kvh * donor.GROUP, (kvh + 1) * donor.GROUP))
            for b0 in range(0, donor.GROUP, hb):
                hs = heads[b0 : b0 + hb]
                qf = torch.cat([q_pt[:, h] * donor.SM_SCALE for h in hs], dim=0)
                s = donor._scores(qf, kf, len(hs), prefix, n, npad, tail)
                o = _pv_pr4714(s, vf, fold)
                out[:, hs, :] = (
                    o.view(len(hs), donor.T_ROWS, donor.HEAD_DIM).permute(1, 0, 2)
                )
                del qf, s, o
            del kf, vf
        return out

    result = {
        "per_tensor_amax": {"q": sq * donor.FP8_MAX, "k": sk * donor.FP8_MAX, "v": sv_ * donor.FP8_MAX},
        "schemes": {
            "pr4714": donor._metrics(run(1.0), ref),
            "pr4714_p448diag": donor._metrics(run(donor.FP8_MAX), ref),
        },
    }
    del q_f32, q_bf, ref
    torch.cuda.empty_cache()
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--depths", type=int, nargs="+", default=donor.DEPTHS)
    ap.add_argument("--dists", nargs="+", default=list(donor.DISTRIBUTIONS))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(HERE / "fp8_pool_emulation.json"))
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
        "probe": "fp8_pool_emulation",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "question": "FlashInfer #4714's FP8 scheme (per-tensor scales, E4M3 KV "
        "pool, unscaled E4M3 P) vs this lab's transform pipeline, emulated at "
        "the attention-output level on the donor probe's 25-cell grid.",
        "pr_pin": PR_PIN,
        "not_claimed": [
            "Not a measurement of #4714's kernel: none of its code executes "
            "here. This reproduces its quantization ARITHMETIC (operand "
            "granularity, storage dtype, P handling) in the donor probe's "
            "oracle-checked evaluation form; tiling and accumulation order "
            "differ, as they do for every scheme on this grid.",
            "Calibration charity: the emulation grants per-tensor amax/448 "
            "scales. The PR's default scales are 1.0; results here are its "
            "BEST case, not its default behaviour.",
            "Not real activations, not a model-quality statement: seeded "
            "synthetic distributions, boundary (a) only, exactly as the donor "
            "probe declares.",
            "The pr4714_p448diag scheme is NOT #4714: it is this lab's P "
            "handling grafted onto their operands, included to attribute the "
            "cost of the unscaled-P choice and to cross-check this file's PV "
            "reimplementation against the donor's pertensor_p cell.",
        ],
        "cross_check": "in every cell, pr4714_p448diag must equal the donor "
        "run_cell's pertensor_p (same arithmetic, independent code path); "
        "max |row_rel_l2_mean delta| reported below.",
        "env": _scrubbed_env(),
        "cells": [],
    }

    t0 = time.perf_counter()
    max_xcheck = 0.0
    for dist in args.dists:
        cfg = donor.DISTRIBUTIONS[dist]
        for n in args.depths:
            print(f"[{dist}] N={n}", flush=True)
            cell = {"dist": dist, "N": n, "seed_base": cfg["seed"]}
            try:
                base = donor.run_cell(dist, n, cfg, verbose=False)
                emu = emulation_schemes(dist, n, cfg)
                cell["npad"] = base["npad"]
                cell["schemes"] = {**base["schemes"], **emu["schemes"]}
                cell["per_tensor_amax"] = emu["per_tensor_amax"]
                a = cell["schemes"]["pr4714_p448diag"]["row_rel_l2_mean"]
                b = cell["schemes"]["pertensor_p"]["row_rel_l2_mean"]
                cell["xcheck_delta"] = abs(a - b)
                max_xcheck = max(max_xcheck, cell["xcheck_delta"])
                cell["status"] = "ok"
                anchor = cell["schemes"]["bf16"]["row_rel_l2_mean"]
                for name in ("pr4714", "pr4714_p448diag", "pertensor", "lab_p"):
                    m = cell["schemes"][name]["row_rel_l2_mean"]
                    print(
                        f"    {name:16s} row_rel_l2 {m * 100:8.4f}%"
                        f"  ({m / anchor:7.2f}x bf16)",
                        flush=True,
                    )
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                cell["status"] = "error"
                cell["error"] = str(exc)[:400]
                print(f"    error: {str(exc)[:200]}", flush=True)
            torch.cuda.empty_cache()
            record["cells"].append(cell)
    record["total_seconds"] = time.perf_counter() - t0
    record["max_xcheck_delta"] = max_xcheck

    Path(args.out).write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\nmax cross-check delta: {max_xcheck:.3e}")
    print(f"wrote {args.out}  ({record['total_seconds']:.1f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
