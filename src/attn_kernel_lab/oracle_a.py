# SPDX-License-Identifier: Apache-2.0
"""Oracle A: an independent implementation of the exact quantized-operator contract.

``docs/OPERATOR_CONTRACT.md`` is the normative text; ``quant.py`` + the CUDA
kernel are the production implementation. This module is the *third* leg: a
deliberately simple, high-precision reimplementation of the same contract,
written from the document rather than from the code, so that agreement between
the three is evidence and disagreement is a defect with an address.

It answers: **did the implementation execute the intended quantized operator?**
A failure against this oracle can never be waived as "expected quantization
error" -- the intended quantization IS this oracle. Fidelity of the intended
operator to BF16 attention is Oracle B's question, not this module's.

Two comparison surfaces:

- **Intermediate boundaries** (``preprocess_q`` / ``preprocess_kv``): quantized
  bytes, scales, means, fold arrays, packed layouts. Compared byte-exactly
  against ``quant.py`` at small shapes -- both sides use the same normatively
  specified fp32 tensor operations, so bytes must agree.
- **Final output** (``attention``): the contract's own arithmetic -- INT32-exact
  QK dots, per-tile softmax with the 448*r_t P fold, E4M3 rounding of P (the
  rounding IS normative), fp32 PV accumulation, the vs_max epilogue and exact
  V-mean add-back -- evaluated in a straightforward per-row loop with float64
  softmax state. Compared against the kernel under a small *contract tolerance*
  that absorbs only what section 4 declares non-normative: ``ex2.approx``
  versus exact exp, fp32 accumulation order, and the bf16 output cast.

Everything here is plain PyTorch and runs on CPU; only the comparison against
the real kernel needs a GPU.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

HEAD_DIM = 256
BLK = 64
MPAD = 128
INT8_MAX = 127.0
FP8_MAX = 448.0

# Fixed KV-position permutation inside each 32-row half of a 64-row V tile
# (contract section 3.5). Restated independently from the document, not
# imported from quant.py -- agreement is one of the things under test.
SIGMA32 = [
    0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15,
    16, 17, 24, 25, 18, 19, 26, 27, 20, 21, 28, 29, 22, 23, 30, 31,
]
SIGMA64 = [(j // 32) * 32 + SIGMA32[j % 32] for j in range(64)]


def _ceil_to(x: int, m: int) -> int:
    return (x + m - 1) // m * m


def hadamard(n: int, dtype=torch.float32, device="cpu") -> torch.Tensor:
    """Orthonormal Sylvester Hadamard, H_{2n} = [[H, H], [H, -H]] / sqrt(n)."""
    if n & (n - 1):
        raise ValueError(f"Hadamard order must be a power of two, got {n}")
    h = torch.ones(1, 1, dtype=dtype, device=device)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)


def _quant_int8(values: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Contract rounding for the INT8 path: divide, round-nearest, saturate."""
    return (values / scale).round_().clamp_(-INT8_MAX, INT8_MAX).to(torch.int8)


def _e4m3(values: torch.Tensor) -> torch.Tensor:
    """Contract rounding for the FP8 path: IEEE round-nearest-even, saturating
    to +-448 (torch's float8_e4m3fn cast semantics are the normative rounding)."""
    return values.to(torch.float8_e4m3fn)


# --------------------------------------------------------------------------
# Preprocessing (contract section 3), independently implemented
# --------------------------------------------------------------------------

@dataclass
class QPack:
    q8: torch.Tensor        # [H, Mpad, D] int8 (viewed uint8 by the kernel)
    qscale: torch.Tensor    # [H, Mpad] f32, sm_scale folded in
    mpad: int


@dataclass
class KVPack:
    k8: torch.Tensor        # [KVH, Npad, D] int8
    kscale: torch.Tensor    # [KVH, ntiles] f32
    kmean: torch.Tensor     # [KVH, D] f32
    vt8: torch.Tensor       # [KVH, ntiles, D, BLK] float8_e4m3fn, SIGMA-permuted, d-major
    vst_scaled: torch.Tensor  # [KVH, ntiles] f32: max(amax, vsmax/16)/448
    vscale: torch.Tensor    # [KVH] f32: vsmax_raw / 448 (the epilogue scale)
    vlog2r: torch.Tensor    # [KVH, ntiles] f32
    vinvr: torch.Tensor     # [KVH, ntiles] f32
    vmean: torch.Tensor     # [KVH, D] f32
    n: int


def preprocess_q(q: torch.Tensor, sm_scale: float, rotation: torch.Tensor) -> QPack:
    """Contract section 3.2: rotate; per-ROW amax clamped at 1e-8; INT8; the
    softmax scale folds into the exported per-row scales; MPAD zero padding."""
    T, H, D = q.shape
    mpad = _ceil_to(T, MPAD)
    qf = q.permute(1, 0, 2).to(torch.float32) @ rotation           # [H, T, D]
    if mpad != T:
        qf = torch.cat([qf, qf.new_zeros(H, mpad - T, D)], dim=1)
    amax = qf.abs().amax(dim=2).clamp_min(1e-8)                    # per row
    scale = amax / INT8_MAX
    q8 = _quant_int8(qf, scale[:, :, None])
    return QPack(q8=q8, qscale=scale * sm_scale, mpad=mpad)


def _channel_mean(x: torch.Tensor, n: int, slab: int | None) -> torch.Tensor:
    """Channel mean over the first ``n`` rows of ``[N, KVH, D]``.

    ``slab=None`` is the mathematical definition (one reduction). ``slab=k``
    replicates the PRODUCTION reduction order: fp32 partial sums accumulated
    slab by slab, exactly as ``quant.py`` iterates -- which is the order the
    72 goldens embody. The two differ in final-ulp fp32 rounding once ``n``
    spans more than one slab, so byte-level comparisons above one slab MUST
    pass the production ``slab``. Contract section 3.3 pins the slab order as
    normative (SLAB = 32768 is numerics-visible).
    """
    xf = x.to(torch.float32)
    if slab is None:
        return xf.sum(dim=0) / max(n, 1)
    total = torch.zeros(x.shape[1], x.shape[2], dtype=torch.float32, device=x.device)
    for s0 in range(0, n, slab):
        total += xf[s0:s0 + slab].sum(dim=0)
    return total / max(n, 1)


def preprocess_kv(
    k: torch.Tensor,   # [N, KVH, D] bf16 (already gathered, position order)
    v: torch.Tensor,   # [N, KVH, D] bf16
    rotation: torch.Tensor,
    slab: int | None = None,
) -> KVPack:
    """Contract sections 3.3-3.5: K channel-mean centering over all N (exact,
    softmax shift-invariance), rotation, per-tile INT8 K; V channel-mean
    centering, per-tile amax with the vs_max/16 P-underflow floor, E4M3, the
    SIGMA64-permuted transposed tile-major layout, and boundary-tile hygiene."""
    N, KVH, D = k.shape
    npad = _ceil_to(N, BLK)
    ntiles = npad // BLK

    # --- K ---
    kmean = _channel_mean(k, N, slab)                              # [KVH, D]
    # The rotation GEMM's slab structure is ALSO numerics-visible: quant.py
    # rotates per BLK-aligned slab, and cuBLAS picks its reduction by GEMM
    # shape, so a [KVH, 64, D] @ [D, D] per slab and one [KVH, N, D] @ [D, D]
    # differ in final-ulp fp32. slab=None keeps the one-GEMM mathematical
    # form; slab=k replicates production (contract section 3.3).
    if slab is None:
        kf = (k.to(torch.float32).permute(1, 0, 2) - kmean[:, None, :]) @ rotation
    else:
        slab_aligned = _ceil_to(slab, BLK)
        pieces = []
        for s0 in range(0, N, slab_aligned):
            piece = (k[s0:s0 + slab_aligned].to(torch.float32).permute(1, 0, 2)
                     - kmean[:, None, :]) @ rotation
            pieces.append(piece)
        kf = torch.cat(pieces, dim=1)
    if npad != N:
        kf = torch.cat([kf, kf.new_zeros(KVH, npad - N, D)], dim=1)
    kb = kf.view(KVH, ntiles, BLK, D)
    kamax = kb.abs().amax(dim=(2, 3)).clamp_min(1e-8)
    kscale = kamax / INT8_MAX
    k8 = _quant_int8(kb, kscale[:, :, None, None]).view(KVH, npad, D)

    # --- V ---
    vmean = _channel_mean(v, N, slab)                              # [KVH, D]
    vf = v.to(torch.float32).permute(1, 0, 2) - vmean[:, None, :]
    if npad != N:
        vf = torch.cat([vf, vf.new_zeros(KVH, npad - N, D)], dim=1)
    vb = vf.view(KVH, ntiles, BLK, D)
    vamax = vb.abs().amax(dim=(2, 3))                              # raw per-tile amax
    vsmax_raw = vamax.amax(dim=1).clamp_min(1e-8)                  # [KVH]
    # P-underflow guard: floor vs_t at vs_max/16 BEFORE the /448 (normative).
    vst_scaled = torch.maximum(vamax, vsmax_raw[:, None] / 16.0) / FP8_MAX
    v8 = _e4m3(vb / vst_scaled[:, :, None, None])                  # [KVH, nt, BLK, D]
    # SIGMA permutation over kv rows inside each tile, then d-major transpose.
    sigma = torch.tensor(SIGMA64, device=v.device, dtype=torch.long)
    vt8 = v8.index_select(2, sigma).permute(0, 1, 3, 2).contiguous()  # [KVH, nt, D, BLK]

    # Boundary-tile hygiene (contract section 5): permuted columns whose SOURCE
    # position is padding are zeroed so stale-decoding-as-NaN cannot poison PV.
    if N % BLK:
        boundary = N // BLK
        tail_cols = (sigma >= (N % BLK)).nonzero(as_tuple=True)[0]
        vt8[:, boundary][:, :, tail_cols] = 0

    ratio = (vst_scaled * FP8_MAX / vsmax_raw[:, None]).clamp(1.0 / 16.0, 1.0)
    return KVPack(
        k8=k8, kscale=kscale, kmean=kmean,
        vt8=vt8, vst_scaled=vst_scaled, vscale=vsmax_raw / FP8_MAX,
        vlog2r=torch.log2(ratio), vinvr=1.0 / ratio, vmean=vmean, n=N,
    )


# --------------------------------------------------------------------------
# The operator (contract sections 1 and 3.6), evaluated plainly
# --------------------------------------------------------------------------

def attention(
    qp: QPack,
    kv: KVPack,
    prefix: int,
    T: int,
    group: int,
    return_lse: bool = False,
):
    """The intended quantized operator, per query row, float64 softmax state.

    Bottom-right causal: row ``r`` attends to columns ``0 .. prefix + r``.
    Per KV tile ``t``: INT32-exact dequantized scores; P is E4M3-rounded at
    scale ``448 * r_t`` (via the fold, here applied explicitly); PV accumulates
    the ROUNDED P against the stored ``vt8`` bytes in float64; ``l`` accumulates
    the UNROUNDED ``p * 448`` (the kernel sums pre-rounding fp32 and undoes
    ``r_t``); the epilogue multiplies ``vscale / l`` and adds the V mean back
    exactly. Returns BF16 ``[T, H, D]`` -- the final cast is normative.
    """
    H = qp.q8.shape[0]
    KVH, npad, D = kv.k8.shape
    ntiles = npad // BLK
    n = kv.n
    out = torch.empty(T, H, D, dtype=torch.bfloat16)
    lse = torch.empty(T, H, dtype=torch.float32) if return_lse else None

    # Depermute vt8 back to position order once: PV math is over logical rows;
    # the permutation is a layout fact both sides agree on (any bijection is
    # exact), and the *stored bytes* are what carry the rounding.
    sigma = torch.tensor(SIGMA64, device=kv.vt8.device, dtype=torch.long)
    inv = torch.empty_like(sigma)
    inv[sigma] = torch.arange(BLK, device=sigma.device)
    # vt8 is [KVH, nt, D, BLK-permuted]; bring back to [KVH, nt, BLK, D] rows.
    v8_logical = kv.vt8.permute(0, 1, 3, 2).index_select(2, inv)
    v8f = v8_logical.to(torch.float64)

    k8f = kv.k8.view(KVH, ntiles, BLK, D).to(torch.float64)
    kscale = kv.kscale.to(torch.float64)
    q8f = qp.q8.to(torch.float64)
    qscale = qp.qscale.to(torch.float64)

    for h in range(H):
        kvh = h // group
        for r in range(T):
            visible = min(prefix + r + 1, n)
            last_tile = (visible - 1) // BLK
            m = -math.inf
            l = 0.0
            acc = torch.zeros(D, dtype=torch.float64)
            for t in range(last_tile + 1):
                lo, hi = t * BLK, min((t + 1) * BLK, visible)
                # INT8 dots are exact in INT32; dequantize with per-row Q scale
                # (sm_scale folded) and per-tile K scale.
                s = (q8f[h, r] @ k8f[kvh, t, : hi - lo].T) * qscale[h, r] * kscale[kvh, t]
                tile_max = float(s.max())
                m_new = max(m, tile_max)
                alpha = 0.0 if m == -math.inf else math.exp(m - m_new)
                p = torch.exp(s - m_new)                       # float64, unrounded
                r_t = 1.0 / float(kv.vinvr[kvh, t])
                # P rounding at scale 448*r_t is normative; PV uses the ROUNDED
                # bytes, l uses the UNROUNDED sum (matching the kernel exactly).
                p8 = _e4m3((p * (FP8_MAX * r_t)).to(torch.float32)).to(torch.float64)
                acc = acc * alpha + p8 @ v8f[kvh, t, : hi - lo]
                l = l * alpha + float(p.sum()) * FP8_MAX
                m = m_new
            inv_l = float(kv.vscale[kvh]) / l if l > 0 else 0.0
            out[r, h] = (acc * inv_l + kv.vmean[kvh].to(torch.float64)).to(torch.bfloat16)
            if return_lse:
                # Base-2 (FA2/CUTLASS wrapper convention). ``l`` carries the
                # kernel's fold -- 448 * sum(exp(s - m)), accumulated
                # PRE-rounding -- so subtract log2(448) to report the true
                # log-sum-exp of the masked scores.
                lse[r, h] = (
                    (m / math.log(2.0)) + math.log2(l) - math.log2(FP8_MAX)
                    if l > 0 else -math.inf
                )
    return (out, lse) if return_lse else out


def reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, prefix: int,
    sm_scale: float | None = None, return_lse: bool = False,
):
    """Convenience: full contract pipeline from BF16 inputs to BF16 output
    (optionally with the base-2 LSE)."""
    T, H, D = q.shape
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    rot = hadamard(D, device=q.device)
    qp = preprocess_q(q, sm_scale, rot)
    kv = preprocess_kv(k, v, rot)
    return attention(qp, kv, prefix, T, group=H // k.shape[1], return_lse=return_lse)
