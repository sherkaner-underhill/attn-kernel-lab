"""Gather + quantization pipeline for the low-precision prefill attention operator.

SPDX-License-Identifier: Apache-2.0
Adapted from the ``origin-private`` implementation defined in
THIRD_PARTY_NOTICES.md. This module is the normative implementation of the
preprocessing half of docs/OPERATOR_CONTRACT.md (section 3).

Per qualifying extend forward, per layer:

  1. gather the request's K/V rows (prefix + current chunk) from the paged
     pool into planar per-kv-head fp8 workspaces, quantizing slab-wise so
     no full-context bf16 copy is ever materialized (one statistics gather
     carrying means and per-tile min/max, one quantization gather);
  2. build the SIGMA-permuted, transposed, tile-major fp8 ``V^T`` the
     kernel's zero-shuffle PV path consumes (and, only when some heads use
     the bf16-PV fallback, a planar bf16 V);
  3. quantize the chunk's Q per 64-row block with the layer's softmax scale
     folded into the Q scales.

Scale scheme: Q per-64-row block x sm_scale, K per-64-row tile (mean-
centered), V mean-centered with PER-64-ROW-TILE scales, P the constant 448
(folded into the kernel's exp).  The per-tile V dequant ratio r_t =
vs_t/vs_max is folded into the same per-tile exp constant the kernel
already computes (P is packed as p*448*r_t), so per-tile V costs zero
inner-loop instructions; the l-sum gets a one-FMUL-per-tile correction.
V's channel mean is added back exactly in the kernel epilogue (softmax
weights sum to 1), which neutralizes massive-activation V channels that
would otherwise burn the fp8 range — the two production-grade V tricks
(SageAttention2 family) this package previously lacked.

All buffers live in one persistent workspace that grows monotonically and is
reused across layers and forwards, so steady-state cost is gather+quant
traffic only.  (The absolute total across a 446k-token, 14-chunk,
16-full-attn-layer prefill is DISPUTED -- ~0.2-0.3 s was claimed here
historically, while the checked per-call baseline JSON implies ~7-8 s; the
direct measurement is a Phase 3 blocker.  Do not cite either figure.)

The workspace has two modes, and they differ ONLY in where memory comes
from:

  * **grow-on-demand** (the default, unchanged): ``get()`` reallocates a
    buffer whenever a request exceeds what it holds.
  * **capacity** (opt-in, ``ws.reserve(max_kv_len, max_q_len, ...)`` or the
    constructor's ``max_kv_len=``/``max_q_len=``): every buffer the pipeline
    and the kernel launch can ask for is allocated up front at the declared
    maxima, and any later request that WOULD have reallocated raises
    :class:`WorkspaceCapacityError` instead of silently growing.  That is
    what makes the buffer addresses stable across calls, which is what CUDA
    graph capture requires (contract section 5: "the caller owns workspace
    and destination output; their addresses and capacity must remain stable
    across graph replay").

Capacity mode is NOT a numerics variant.  Both modes hand every consumer
``buf[:numel].view(shape)`` -- offset 0, standard contiguous strides, the
exact per-call shape -- so the only thing that differs between them is the
device address of the parent allocation.  Per-call shapes, the slab loop
bounds, the rotation GEMM batching and every accumulation order are
functions of N, T, ``BLK`` and ``SLAB`` alone, none of which capacity mode
touches.  Bit-identity between the two modes is therefore structural, not a
tolerance, and ``tests/kernel/test_cuda_graph_capacity.py`` asserts it.

Correctness notes:

  * Workspace rows beyond the current N can hold stale data from a longer
    earlier request.  Stale K/Q is harmless (masked before exp / sliced
    output).  Stale V could contain fp8 NaN encodings which would poison
    ``0 * NaN`` in the PV matmul, so the kv-boundary tile's tail columns
    are explicitly zeroed each build.
  * Q rows in [T, Mpad) are compute padding; they are zeroed to keep NaNs
    out of (discarded) padded-row accumulators.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

HEAD_DIM = 256
BLK = 64  # kernel tile size in both the q-row and kv-row dimensions
# Q rows are padded to whole CTA tiles, and the kernel's widest CTA covers
# BM=128 Q rows (csrc/fp8_prefill_attn.cu, BM_D).  This is a padding
# granularity ONLY -- BLK stays the quantization/scale tile everywhere
# (K tiles, kv-row blocks), and Q scales are per row, so the extra padded
# rows are just zeros with their own scales and never touch a real row.
MPAD = 128
SLAB = 32768  # gather/quant slab (tokens); bounds transient fp32 memory
# SLAB is numerics-visible (contract section 3: mean accumulation order and
# per-slab rotation GEMM shape) and the merged statistics pass below requires
# BLK alignment so mean partial sums and per-tile min/max share one loop.
assert SLAB % BLK == 0, "SLAB must be BLK-aligned (numerics-visible, see contract)"
FP8_MAX = 448.0

# kv-position permutation inside each 32-block of V^T (see kernel header)
_SIGMA32 = [
    0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15,
    16, 17, 24, 25, 18, 19, 26, 27, 20, 21, 28, 29, 22, 23, 30, 31,
]
# 64-wide version: applies SIGMA32 inside each 32-half of a 64-row kv tile
SIGMA64 = [(jk // 32) * 32 + _SIGMA32[jk % 32] for jk in range(64)]


def _ceil_to(x: int, m: int) -> int:
    return (x + m - 1) // m * m


def _hadamard(n: int, device) -> torch.Tensor:
    """Orthonormal Hadamard matrix (n a power of two)."""
    h = torch.ones(1, 1, device=device)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / (n ** 0.5)


class _true_fp32_matmul:
    """Pin the rotation GEMMs to true-fp32 math regardless of ambient
    TF32 state.  The serving stack may enable TF32 globally; a 10-bit-
    mantissa rotation would silently shift every downstream quantization
    (and break golden bit-exactness) based on global flags rather than
    code.  Save/flip/restore keeps the pin local to the two matmuls and
    never perturbs the host process's own math settings."""

    #: Normative cuBLAS workspace pin (contract section 3.1): the rotation is
    #: reproducible only under a fixed cuBLAS workspace configuration, and the
    #: variable must be set BEFORE the first cuBLAS handle exists -- far too
    #: early for this library to set it itself. So it checks and warns
    #: (fruit report Q6); tests and the bench set it at import time.
    CUBLAS_PIN = ":4096:8"

    def __enter__(self):
        import os as _os
        import warnings as _warnings

        configured = _os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if configured != self.CUBLAS_PIN and torch.cuda.is_available():
            _warnings.warn(
                f"CUBLAS_WORKSPACE_CONFIG={configured!r} != {self.CUBLAS_PIN!r}: "
                "the fp32 rotation may not be reproducible against the golden "
                "records (operator contract section 3.1). Set it before torch "
                "creates its first cuBLAS handle.",
                RuntimeWarning,
                stacklevel=3,
            )
        self._tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        return self

    def __exit__(self, *exc):
        torch.backends.cuda.matmul.allow_tf32 = self._tf32
        return False


#: Head counts the capacity planner assumes when a caller does not name them.
#: These MUST agree with ``capability.V1_CAPABILITY`` (24:4, contract section 2).
#: They are spelled out here rather than imported because ``quant.py`` is also
#: imported standalone as a top-level ``quant`` module by the kernel test lane,
#: where a package-relative import would not resolve; ``ops.plan_workspace``
#: always passes the capability's own values explicitly, so the declared surface
#: still has exactly one source of truth on the public path.
DECLARED_Q_HEADS = 24
DECLARED_KV_HEADS = 4


class WorkspaceCapacityError(RuntimeError):
    """A capacity-reserved workspace was asked for more than it planned for.

    Deliberately **not** a ``CapabilityError``: that class is the one a
    consuming framework may catch to route around an unsupported request, and
    catching this one the same way would turn "the graph I captured is now
    replaying against a reallocated pointer" into a quiet fallback. A plan miss
    is a defect in the caller's plan, and it must fail the request.
    """


@dataclass(frozen=True)
class CapacityPlan:
    """The maxima a reserved workspace was sized for, and the padded geometry.

    Returned by :attr:`FP8PrefillWorkspace.capacity` so a caller -- a bench
    lane, a test, a serving planner -- can check a request against the plan
    without re-deriving the padding rules.
    """

    max_kv_len: int
    max_q_len: int
    q_heads: int
    kv_heads: int
    need_vb16: bool
    need_lse: bool
    npad: int
    ntmax: int
    mpad: int

    def admits(self, kv_len: int, q_len: int) -> bool:
        """Whether a request of this geometry fits without reallocation."""
        return 1 <= kv_len <= self.max_kv_len and 1 <= q_len <= self.max_q_len


class FP8PrefillWorkspace:
    """Persistent device buffers for one backend: grown on demand, or reserved.

    ``FP8PrefillWorkspace(device)`` is the historical grow-on-demand workspace
    and is unchanged. Passing ``max_kv_len``/``max_q_len`` (or calling
    :meth:`reserve` later) switches it to capacity mode -- see the module
    docstring, and :meth:`reserve` for what that buys and what it forbids.
    """

    def __init__(
        self,
        device: torch.device,
        *,
        max_kv_len: int | None = None,
        max_q_len: int | None = None,
        q_heads: int = DECLARED_Q_HEADS,
        kv_heads: int = DECLARED_KV_HEADS,
        need_vb16: bool = False,
        need_lse: bool = True,
    ):
        self.device = device
        self.sigma64 = torch.tensor(SIGMA64, device=device, dtype=torch.long)
        # incoherent processing (FA3-style): rotate Q and K by the same
        # orthonormal Hadamard before quantization. Scores are exactly
        # invariant; within-row outliers (RoPE'd keys) spread across dims,
        # which is what linear int8 needs. Measured on real keys: worst-head
        # per-layer error 2.2% -> 0.55% (at the bf16-input noise floor).
        self.hadamard = _hadamard(HEAD_DIM, device)
        self._bufs: dict[str, torch.Tensor] = {}
        self._tail_cols: dict[int, torch.Tensor] = {}
        self._plan: CapacityPlan | None = None
        # The "this output is not requested" sentinels gather_quantize_kv
        # returns for vt8/vb16. Allocated once (they carry no storage bytes) so
        # that even the zero-size allocation is off the per-call path.
        self.empty_u8 = torch.empty(0, dtype=torch.uint8, device=device)
        self.empty_bf16 = torch.empty(0, dtype=torch.bfloat16, device=device)
        if max_kv_len is not None or max_q_len is not None:
            if max_kv_len is None or max_q_len is None:
                raise ValueError(
                    "capacity mode needs BOTH max_kv_len and max_q_len: a plan "
                    "that pins one dimension and grows the other still "
                    "reallocates, which is the thing capacity mode exists to "
                    "prevent"
                )
            self.reserve(
                max_kv_len, max_q_len, q_heads=q_heads, kv_heads=kv_heads,
                need_vb16=need_vb16, need_lse=need_lse,
            )

    @property
    def capacity(self) -> CapacityPlan | None:
        """The active :class:`CapacityPlan`, or ``None`` in grow-on-demand mode."""
        return self._plan

    def reserve(
        self,
        max_kv_len: int,
        max_q_len: int,
        *,
        q_heads: int = DECLARED_Q_HEADS,
        kv_heads: int = DECLARED_KV_HEADS,
        need_vb16: bool = False,
        need_lse: bool = True,
    ) -> CapacityPlan:
        """Preallocate every buffer at ``max_kv_len``/``max_q_len`` and lock.

        After this returns, :meth:`get` raises :class:`WorkspaceCapacityError`
        rather than allocating, so a request that outgrows the plan is a loud
        failure instead of a silently moved pointer. That is the property CUDA
        graph capture needs: a captured launch bakes in the device addresses it
        saw, and a workspace that reallocates between capture and replay makes
        the replay read freed memory.

        ``need_vb16`` reserves the planar BF16 V the per-head bf16-PV numerics
        lane consumes; the declared surface is all-fp8-PV, so it defaults off.
        ``need_lse`` reserves the optional base-2 LSE buffer (contract section 2).

        Reserving is idempotent-ish rather than incremental: it re-plans from
        scratch, and because :meth:`get` only ever grows, calling it a second
        time with larger maxima enlarges the buffers as expected.

        Returns the :class:`CapacityPlan` it installed.
        """
        for name, value in (("max_kv_len", max_kv_len), ("max_q_len", max_q_len),
                            ("q_heads", q_heads), ("kv_heads", kv_heads)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an int >= 1, got {value!r}")
        if q_heads % kv_heads:
            raise ValueError(
                f"q_heads={q_heads} must be divisible by kv_heads={kv_heads} "
                "(contract section 1: GQA maps query head h to kv head "
                "h // (Hq / Hkv))"
            )

        # Planning itself allocates, so drop the lock for the duration and take
        # it again at the end. Every buffer goes through get() -- the same call
        # the per-call path uses -- so a reserved buffer and a grown one are the
        # same object built the same way, differing only in size.
        self._plan = None
        npad = _ceil_to(max_kv_len, BLK)
        ntmax = npad // BLK
        mpad = _ceil_to(max_q_len, MPAD)
        h, kvh = q_heads, kv_heads

        self.get("q8", (h, mpad, HEAD_DIM), torch.uint8)
        self.get("qscale", (h, mpad), torch.float32)
        self.get("o", (h, mpad, HEAD_DIM), torch.bfloat16)
        self.get("pv8_mask", (h,), torch.uint8)
        if need_lse:
            self.get("lse", (h, mpad), torch.float32)
        self.get("k8", (kvh, npad, HEAD_DIM), torch.uint8)
        self.get("kscale", (kvh, ntmax), torch.float32)
        self.get("vt8", (kvh, ntmax, HEAD_DIM, BLK), torch.uint8)
        if need_vb16:
            self.get("vb16", (kvh, npad, HEAD_DIM), torch.bfloat16)
        for name in ("vscale_t", "vlog2r", "vinvr"):
            self.get(name, (kvh, ntmax), torch.float32)
        self.get("vscale", (kvh,), torch.float32)
        self.get("vmean", (kvh, HEAD_DIM), torch.float32)
        self.get("vtile_max", (kvh, ntmax, HEAD_DIM), torch.float32)
        self.get("vtile_min", (kvh, ntmax, HEAD_DIM), torch.float32)
        self.get("vsum", (kvh, HEAD_DIM), torch.float32)
        self.get("ksum", (kvh, HEAD_DIM), torch.float32)

        # The boundary-tile tail-column index sets. Only tails in [1, BLK) are
        # ever asked for (N % BLK == 0 skips the hygiene branch entirely), and
        # each is 63 int64s at most, so planning all of them costs nothing and
        # removes the last host->device copy from the per-call path.
        for tail in range(1, BLK):
            self.tail_cols(tail)

        self._plan = CapacityPlan(
            max_kv_len=max_kv_len, max_q_len=max_q_len,
            q_heads=q_heads, kv_heads=kv_heads,
            need_vb16=need_vb16, need_lse=need_lse,
            npad=npad, ntmax=ntmax, mpad=mpad,
        )
        return self._plan

    def release_capacity(self) -> None:
        """Return to grow-on-demand. The buffers already allocated are kept.

        For a caller that reserved for a graph, finished with it, and now wants
        the workspace back for arbitrary geometry.
        """
        self._plan = None

    def buffer_pointers(self) -> dict[str, int]:
        """``{name: data_ptr}`` for every allocated buffer.

        The evidence a graph-safety test actually needs: capacity mode's claim
        is that these do not move across calls, and this is how a test says so
        without reaching into ``_bufs``.
        """
        return {name: buf.data_ptr() for name, buf in self._bufs.items()}

    # Debug discriminator (SGLANG_FP8_PREFILL_ZERO_WS=1): zero-fill every
    # workspace allocation instead of torch.empty.  If a nondeterministic
    # quality failure vanishes under this knob, some path reads
    # uninitialized workspace whose content depends on allocator history.
    import os as _os
    ZERO_WS = _os.environ.get("SGLANG_FP8_PREFILL_ZERO_WS") == "1"

    def tail_cols(self, tail: int) -> torch.Tensor:
        """Permuted V columns whose SOURCE position is >= ``tail`` (the padding
        columns of a ragged boundary tile). Pure function of ``tail`` and the
        compile-time SIGMA64; cached as small device tensors so the hygiene
        zeroing never synchronises (see gather_quantize_kv)."""
        cached = self._tail_cols.get(tail)
        if cached is None:
            if self._plan is not None:
                # Unreachable via reserve(), which pre-populates every tail in
                # [1, BLK). Kept because building one here is a host->device
                # copy, which is illegal during graph capture and would be a
                # sync on the hot path -- exactly what capacity mode promises
                # is gone.
                raise WorkspaceCapacityError(
                    f"tail_cols({tail}) is not in the reserved cache; building "
                    "it now would copy from the host on the per-call path"
                )
            cols = [j for j in range(BLK) if SIGMA64[j] >= tail]
            cached = torch.tensor(cols, device=self.device, dtype=torch.long)
            self._tail_cols[tail] = cached
        return cached

    def get(self, name: str, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        buf = self._bufs.get(name)
        numel = 1
        for s in shape:
            numel *= s
        if buf is None or buf.numel() < numel or buf.dtype != dtype:
            if self._plan is not None:
                held = "unallocated" if buf is None else f"{buf.numel()} x {buf.dtype}"
                raise WorkspaceCapacityError(
                    f"buffer {name!r} needs {numel} x {dtype} but the capacity "
                    f"plan holds {held}. The plan is {self._plan}; a request "
                    "outside it would reallocate, which invalidates the device "
                    "addresses any captured CUDA graph baked in. Re-reserve at "
                    "larger maxima, or call release_capacity() to go back to "
                    "growing on demand."
                )
            if self.ZERO_WS:
                buf = torch.zeros(numel, dtype=dtype, device=self.device)
            else:
                buf = torch.empty(numel, dtype=dtype, device=self.device)
            self._bufs[name] = buf
        # Both modes return the SAME view: element 0 of the parent, the exact
        # requested shape, standard contiguous strides. Only the parent's size
        # and address differ between a reserved buffer and a grown one, and
        # neither is visible to any operation downstream -- which is the whole
        # of the bit-identity argument for capacity mode.
        return buf[:numel].view(shape)


INT8_MAX = 127.0


def _block_quant(blocks: torch.Tensor, amax: torch.Tensor, qk_i8: bool):
    """blocks [..., BLK, HD] / per-block amax -> (bytes, scale)."""
    if qk_i8:
        scale = amax / INT8_MAX
        # No clamp: scale = amax/127 and |x| <= amax bound the quotient at
        # 127*(1+2eps); the first representable value the round could clamp
        # is 127.5, needing ~0.4% relative error where fp32 has ~1e-7. The
        # quotient is <= 127.0 exactly across eight decades of amax (fruit
        # report Q3); the clamp was a dead elementwise pass over all of K.
        q = ((blocks / scale[..., None, None]).round_()
             .to(torch.int8).view(torch.uint8))
    else:
        scale = amax / FP8_MAX
        q = (blocks / scale[..., None, None]).to(torch.float8_e4m3fn).view(torch.uint8)
    return q, scale


def quantize_q(
    ws: FP8PrefillWorkspace,
    q: torch.Tensor,  # [T, H, HEAD_DIM] bf16 (post-RoPE)
    sm_scale: float,
    qk_i8: bool = True,
    rotate: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """-> (q8 [H, Mpad, HD] uint8, qscale [H, Mpad] f32 PER ROW, Mpad).

    Per-ROW Q scales (not per-64-block): block-granular scales let a single
    outlier row coarsen 63 neighbours — measured as the dominant term of
    the conservative mode's per-layer excess vs the bf16 yardstick.  The
    kernel reads two scales per thread (its two fixed rows) instead of one
    per CTA.

    Rows in [T, MPAD-aligned) are compute padding: their Q bytes are zero
    and their amax clamps to 1e-8, so their scale is a tiny positive
    number (never 0/NaN), their scores are all-zero, and their outputs are
    sliced off by the caller.  Because the scales are per ROW, widening
    the padding from BLK to MPAD adds zero rows only — it cannot coarsen
    any real row's scale.
    """
    T, H, _ = q.shape
    mpad = _ceil_to(T, MPAD)
    q8 = ws.get("q8", (H, mpad, HEAD_DIM), torch.uint8)
    qs = ws.get("qscale", (H, mpad), torch.float32)
    qf = q.permute(1, 0, 2).to(torch.float32)  # [H, T, HD]
    if rotate:
        with _true_fp32_matmul():
            qf = qf @ ws.hadamard
    if mpad != T:
        qf = torch.cat([qf, qf.new_zeros(H, mpad - T, HEAD_DIM)], dim=1)
    # aminmax + max(mx, -mn) == abs().amax() bitwise (max/min/negation are
    # exact), without materialising the full |qf| copy (fruit report Q2).
    qmn, qmx = torch.aminmax(qf, dim=2)
    amax = torch.maximum(qmx, -qmn).clamp_min_(1e-8)  # [H, mpad] per row
    lim = INT8_MAX if qk_i8 else FP8_MAX
    scale = amax / lim
    if qk_i8:
        # No clamp -- provably dead, see _block_quant.
        q8.view(H, mpad, HEAD_DIM)[:] = (
            (qf / scale[:, :, None]).round_()
            .to(torch.int8).view(torch.uint8))
    else:
        q8.view(H, mpad, HEAD_DIM)[:] = (
            (qf / scale[:, :, None]).to(torch.float8_e4m3fn).view(torch.uint8))
    qs[:] = scale * sm_scale
    return q8, qs, mpad


def gather_quantize_kv(
    ws: FP8PrefillWorkspace,
    k_buffer: torch.Tensor,  # [pool, KVH, HEAD_DIM] bf16
    v_buffer: torch.Tensor,  # [pool, KVH, HEAD_DIM] bf16
    idx: torch.Tensor,  # [N] long: pool rows, position order
    need_vt8: bool,
    need_vb16: bool,
    center_k: bool = True,
    qk_i8: bool = True,
    rotate: bool = True,
) -> dict:
    """Gather + quantize the request's K/V.  Returns kernel-ready tensors.

    ``center_k`` enables K mean-centering (the SageAttention smoothing
    trick): quantize ``K - mean_channel(K)`` instead of K.  The dropped
    ``q . mean`` term is constant across every kv column of a q row, and
    softmax is shift-invariant per row, so the result is mathematically
    identical while the centered K has a much smaller amax — finer fp8
    steps exactly where RoPE'd keys carry large shared channel offsets.
    No kernel change and no correction term are needed.
    """
    N = idx.shape[0]
    KVH = k_buffer.shape[1]
    npad = _ceil_to(N, BLK)
    ntmax = npad // BLK

    k8 = ws.get("k8", (KVH, npad, HEAD_DIM), torch.uint8)
    ks = ws.get("kscale", (KVH, ntmax), torch.float32)
    vt8 = ws.get("vt8", (KVH, ntmax, HEAD_DIM, BLK), torch.uint8) if need_vt8 \
        else ws.empty_u8
    vb16 = ws.get("vb16", (KVH, npad, HEAD_DIM), torch.bfloat16) if need_vb16 \
        else ws.empty_bf16
    # per-tile V scales + fold arrays + channel mean (see module docstring)
    vst = ws.get("vscale_t", (KVH, ntmax), torch.float32)
    vlog2r = ws.get("vlog2r", (KVH, ntmax), torch.float32)
    vinvr = ws.get("vinvr", (KVH, ntmax), torch.float32)
    vsmax = ws.get("vscale", (KVH,), torch.float32)
    vmean = ws.get("vmean", (KVH, HEAD_DIM), torch.float32)

    # pass 1 (slab-wise, SLAB is BLK-aligned by the module assert): channel
    # sums for the means AND per-tile per-channel V min/max in the SAME
    # gather. The former pass 1.5 (a second full gather + a full centred-
    # |V| materialisation) is replaced by the exact identity
    #     max_r |V[r,c] - m[c]| = max(fl(mx[c] - m[c]), fl(m[c] - mn[c]))
    # -- fp32 subtraction with a fixed right operand is monotone, negation
    # is exact, and min/max are exact and order-independent, so this is
    # bit-exact unconditionally (fruit report Q1). Padding rows contributed
    # exactly 0 to the old centred amax and the result is always >= 0, so
    # excluding them here changes nothing.
    kmean = None
    if need_vt8 or center_k:
        # Workspace-owned rather than freshly allocated: a per-call
        # ``torch.zeros`` is an allocation on the hot path, and capacity mode's
        # claim is that there are none. Bit-identical -- an fp32 zero is an fp32
        # zero, and the ``+=`` accumulation order below is untouched.
        vsum = ws.get("vsum", (KVH, HEAD_DIM), torch.float32).zero_()
        ksum = ws.get("ksum", (KVH, HEAD_DIM), torch.float32).zero_()
        if need_vt8:
            vtmx = ws.get("vtile_max", (KVH, ntmax, HEAD_DIM), torch.float32)
            vtmn = ws.get("vtile_min", (KVH, ntmax, HEAD_DIM), torch.float32)
        for s0 in range(0, N, SLAB):
            s1 = min(s0 + SLAB, N)
            sub = idx[s0:s1]
            if need_vt8:
                vsl = v_buffer.index_select(0, sub).to(torch.float32)
                vsum += vsl.sum(dim=0)
                vslh = vsl.permute(1, 0, 2)                  # [KVH, cur, HD]
                cur = s1 - s0
                rag = cur % BLK
                t0 = s0 // BLK
                full = cur - rag
                if full:
                    fb = vslh[:, :full].reshape(KVH, full // BLK, BLK, HEAD_DIM)
                    mn, mx = torch.aminmax(fb, dim=2)        # [KVH, nt_full, HD]
                    vtmn[:, t0:t0 + full // BLK] = mn
                    vtmx[:, t0:t0 + full // BLK] = mx
                if rag:
                    mn, mx = torch.aminmax(vslh[:, full:], dim=1)
                    vtmn[:, t0 + full // BLK] = mn
                    vtmx[:, t0 + full // BLK] = mx
            if center_k:
                ksum += k_buffer.index_select(0, sub).to(torch.float32).sum(dim=0)
        if need_vt8:
            vmean[:] = vsum / max(N, 1)
        if center_k:
            kmean = ksum / max(N, 1)  # [KVH, HEAD_DIM]

    # P-underflow guard unchanged: the kernel packs P as p*448*(vs_t/vs_max);
    # ratios below 1/16 would push packed P into e4m3 subnormals exactly when
    # a quiet-V tile carries high attention mass (measured output damage on
    # real workloads).
    # Flooring vs_t at vs_max/16 keeps packed P in the normal range while
    # quiet tiles still get 16x finer scales than a global-scale scheme.
    if need_vt8:
        nt_used = _ceil_to(N, BLK) // BLK
        vam = torch.maximum(vtmx[:, :nt_used] - vmean[:, None, :],
                            vmean[:, None, :] - vtmn[:, :nt_used]).amax(dim=2)
        vst[:, :nt_used] = vam
        vsmax[:] = vst[:, :ntmax].amax(dim=1).clamp_min_(1e-8)
        vst[:, :ntmax] = (
            torch.maximum(vst[:, :ntmax], vsmax[:, None] / 16.0) / FP8_MAX
        )

    # K (+ V pass 2) slab-wise, aligned to BLK so tile scales are exact.
    # The true-fp32 pin is entered ONCE around the loop (it used to flip the
    # process-global TF32 flag per slab -- 28 times at 446k; fruit report Q7).
    slab = _ceil_to(SLAB, BLK)
    _fp32_pin = _true_fp32_matmul()
    _fp32_pin.__enter__()
    for s0 in range(0, N, slab):
        s1 = min(s0 + slab, N)
        cur = s1 - s0
        curp = _ceil_to(cur, BLK)
        ksl = k_buffer.index_select(0, idx[s0:s1]).to(torch.float32)  # [cur,KVH,HD]
        ksl = ksl.permute(1, 0, 2)  # [KVH, cur, HD]
        if kmean is not None:
            ksl = ksl - kmean[:, None, :]
        if rotate:
            ksl = ksl @ ws.hadamard
        if curp != cur:
            ksl = torch.cat([ksl, ksl.new_zeros(KVH, curp - cur, HEAD_DIM)], 1)
        kb = ksl.view(KVH, curp // BLK, BLK, HEAD_DIM)
        # aminmax over merged (row, channel) dims == abs().amax bitwise, one
        # pass instead of a full |kb| materialisation (fruit report Q2).
        kmn, kmx = torch.aminmax(kb.reshape(KVH, curp // BLK, BLK * HEAD_DIM), dim=2)
        kam = torch.maximum(kmx, -kmn).clamp_min_(1e-8)
        kbytes, kscale_sl = _block_quant(kb, kam, qk_i8)
        ks[:, s0 // BLK:s0 // BLK + curp // BLK] = kscale_sl
        k8.view(KVH, ntmax, BLK, HEAD_DIM)[:, s0 // BLK:s0 // BLK + curp // BLK] = kbytes

        if need_vt8 or need_vb16:
            vslf = v_buffer.index_select(0, idx[s0:s1])  # [cur, KVH, HD] bf16
            if need_vb16:
                vb = vslf.permute(1, 0, 2)  # [KVH, cur, HD]
                vb16[:, s0:s1] = vb
            if need_vt8:
                vf = vslf.permute(1, 0, 2).to(torch.float32) - vmean[:, None, :]
                if curp != cur:
                    vf = torch.cat([vf, vf.new_zeros(KVH, curp - cur, HEAD_DIM)], 1)
                vfb = vf.view(KVH, curp // BLK, BLK, HEAD_DIM)
                tsl = slice(s0 // BLK, s0 // BLK + curp // BLK)
                v8 = (vfb / vst[:, tsl, None, None]).to(torch.float8_e4m3fn)
                # SIGMA-permute kv positions inside each tile, then transpose
                # to d-major [KVH, nblk, HD, BLK] (the kernel's B layout).
                # index_select along the last axis of the TRANSPOSED view
                # materialises the permuted-transposed tensor in one copy;
                # the strided slice-assign is the second. (Was three full
                # copies of the fp8 V; fruit report Q4a. Same bytes:
                # out[..., d, j] = v8[..., SIGMA64[j], d] either way.)
                vt8[:, tsl] = (
                    v8.permute(0, 1, 3, 2).index_select(3, ws.sigma64)
                    .view(torch.uint8)
                )

    _fp32_pin.__exit__(None, None, None)

    # NaN hygiene: zero the kv-boundary tile's tail (stale fp8 bytes there
    # could decode as NaN and poison 0*NaN in the PV matmul).
    if N % BLK != 0:
        t = N // BLK
        if need_vt8:
            # tail columns of the boundary tile map to permuted positions;
            # zero the whole tile's columns >= N%BLK in PERMUTED coords by
            # zeroing every column whose source position >= N%BLK.
            # permuted column j holds source SIGMA64[j]; the column set
            # depends only on N % BLK and the compile-time SIGMA64, so it is
            # computed host-side and cached -- torch.nonzero on a CUDA tensor
            # synchronises the stream, and this was the ONLY sync in an
            # otherwise sync-free path (fruit report Q5).
            #
            # ``index_fill_`` rather than ``[..., tail_cols] = 0``. They write
            # the SAME bytes -- the same column set of the same tile set to the
            # same zero -- but the advanced-indexing assignment lowers to
            # ``index_put_``, which synchronises the stream on CUDA (measured
            # under ``set_sync_debug_mode("warn")``: one sync per ragged call).
            # Caching ``tail_cols`` removed the ``nonzero()`` sync (fruit report
            # Q5) and left this one behind, hidden because the audit was looking
            # for the index computation rather than the store. It matters twice
            # over: it is a host stall on every ragged request, and a
            # synchronising op is ILLEGAL inside a CUDA graph capture, so it
            # made exactly the ragged shapes uncapturable while BLK-aligned ones
            # captured fine.
            tail_cols = ws.tail_cols(N % BLK)
            vt8.view(KVH, ntmax, HEAD_DIM, BLK)[:, t].index_fill_(2, tail_cols, 0)
        if need_vb16:
            vb16[:, N:npad] = 0

    # per-tile fold arrays: r_t = vs_t/vs_max (already floored at 1/16),
    # packed P carries 448*r_t via the exp constant; epilogue multiplies
    # by vs_max (see kernel)
    if need_vt8:
        r = (vst[:, :ntmax] * FP8_MAX / vsmax[:, None]).clamp(1.0 / 16.0, 1.0)
        vlog2r[:, :ntmax] = torch.log2(r)
        vinvr[:, :ntmax] = 1.0 / r
        # kernel epilogue expects vs_max as a SCALE (amax/448), while the
        # ratio math above needed the raw amax
        vsmax /= FP8_MAX
    else:
        # Defined values, not stale workspace: these are returned (and passed
        # to the kernel by generic callers) unconditionally, and contract
        # section 5 makes exactly this stale-reuse class a contract hazard
        # (fruit report Q8 / gap G4b). Identity fold, zero scale, zero mean.
        vst[:, :ntmax].zero_()
        vlog2r[:, :ntmax].zero_()
        vinvr[:, :ntmax].fill_(1.0)
        vsmax.zero_()
        vmean.zero_()

    return {
        "k8": k8, "kscale": ks, "vt8": vt8, "vb16": vb16, "vscale": vsmax,
        "vlog2r": vlog2r, "vinvr": vinvr, "vmean": vmean,
        "n": N, "ntmax": ntmax,
    }
