# SPDX-License-Identifier: Apache-2.0
"""The public operator entry point: validate the declared surface, then run.

``docs/OPERATOR_CONTRACT.md`` is the normative statement of what this operator
*is*; ``capability.py`` is the executable form of its §2 support table. This
module is the only thing that puts the two on the call path, which is the whole
point of it: a request outside the declared surface must fail as a
``CapabilityError`` -- a ``ValueError``, the one class a consuming framework may
catch -- and not as a ``RuntimeError`` raised by a ``TORCH_CHECK`` deep inside
the extension, still less as a silent misread. That distinction is gap G4 of
``upstream/EVIDENCE_GAP_ANALYSIS_2026-08-29.md`` and it is the shape upstream
asks for: FlashInfer #4272 asserts five distinct ``ValueError`` messages, and
#3518 gained a rejection test purely because a reviewer asked for one.

Ordering is deliberate, and it is the property the rejection tests pin:

1. the few structural facts needed to *derive* the geometry (ranks, shapes);
2. the declared surface, via :func:`capability.check_supported`;
3. the structural checks the contract implies but the capability table does not
   express -- dtypes, contiguity, device agreement, index and prefix ranges, a
   caller-supplied ``out``;
4. only then the lazy extension load, the preprocessing pipeline, the launch.

Nothing before step 4 imports torch at module scope, initialises CUDA, or builds
the JIT extension. That is what lets the entire rejection matrix run on a
CPU-only machine with neither a GPU nor a compiler, which is where it is cheap
enough to run on every commit.

Two checks are deliberately *absent*, and their absence is contract-level rather
than an oversight:

- **KV index bounds.** Validating ``idx`` values against the pool size means
  reading device memory, i.e. a synchronisation; contract §5 says ``run``
  enqueues on the caller's current stream and does not implicitly synchronise.
  Out-of-range indices are the caller's invariant.
- **CUDA residency.** Only device *agreement* is checked here. Requiring
  ``is_cuda`` would make the validation surface untestable off a GPU for no
  gain: the extension's own ``TORCH_CHECK`` rejects host tensors at launch.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .capability import V1_CAPABILITY, CapabilityError, OperatorCapability, check_supported

#: **Seam for the LSE writeback (gap G1).** The kernel epilogue gained an
#: optional base-2 LSE output on 2026-08-30 (contract §2); *exposing* it through
#: this API is the ``package_api_version`` bump, and until that lands v1 is
#: output-only and ``return_lse=True`` is refused. Everything below is already
#: written for both cases -- the launch passes an ``lse`` buffer and
#: :func:`prefill_extend` returns ``(out, lse)`` -- so the exposure is this flag
#: plus a ``returns_lse=True`` capability declaration. Flipping the flag alone
#: leaves ``check_supported`` refusing the request one line later, which is the
#: intended backstop against a half-declared surface.
# Flipped 2026-08-30 after the kernel-side base-2 LSE writeback landed
# (e0c33bc) with the 72 goldens surviving bit-exact and the output path proven
# byte-identical with and without the request. Exposure through this public
# API is the package_api_version bump the contract names; the next release
# record carries package_api_version 2.
_LSE_AVAILABLE = True

#: The pool is one KV position per row, so ``page_size`` is 1 by construction at
#: this entry point rather than by argument. Larger pages are a different gather
#: (contract §6.3); the rejection lives in ``check_supported`` for callers that
#: pass a page size through an adapter.
_PAGE_SIZE = 1

_MODE = "extend"
_MASK = "bottom_right_causal"

__all__ = ["PrefillRequest", "check_request", "plan_workspace", "prefill_extend"]


@dataclass(frozen=True)
class PrefillRequest:
    """The geometry a validated request implies, derived once and reused.

    Returned by :func:`check_request` so a caller that wants the validation
    without the launch -- a planner, a bench harness, a test -- gets the derived
    numbers rather than re-deriving them from shapes it has already checked.
    """

    q_len: int
    q_heads: int
    kv_heads: int
    head_dim: int
    kv_len: int
    prefix: int
    sm_scale: float
    kv_dtype: str
    return_lse: bool


def _dtype_name(dtype) -> str:
    """``torch.bfloat16`` -> ``"bfloat16"``: the contract's dtype spelling."""
    return str(dtype).rsplit(".", 1)[-1]


def _reject(message: str, *, field: str, value: object, supported: object):
    raise CapabilityError(message, field=field, value=value, supported=supported)


def _same_device(a, b) -> bool:
    """Device agreement, tolerating an unindexed ``cuda`` against ``cuda:0``.

    A workspace built as ``FP8PrefillWorkspace(torch.device("cuda"))`` and a
    tensor on ``cuda:0`` are the same device; ``==`` says otherwise.
    """
    if a.type != b.type:
        return False
    if a.index is None or b.index is None:
        return True
    return a.index == b.index


def _shares_storage(a, b) -> bool:
    """Whether two tensors are backed by the same allocation.

    Meta tensors all report a null data pointer, so they are excluded rather
    than reported as universally aliased.
    """
    if a.device.type == "meta" or b.device.type == "meta":
        return False
    try:
        ptr_a = a.untyped_storage().data_ptr()
        ptr_b = b.untyped_storage().data_ptr()
    except Exception:  # noqa: BLE001 -- a storage-less tensor cannot alias
        return False
    return bool(ptr_a) and ptr_a == ptr_b


def check_request(
    q,
    k_pool,
    v_pool,
    idx,
    prefix,
    *,
    workspace=None,
    out=None,
    lse_out=None,
    sm_scale=None,
    return_lse: bool = False,
    capability: OperatorCapability = V1_CAPABILITY,
) -> PrefillRequest:
    """Validate one request against the declared surface; return its geometry.

    Raises :class:`~attn_kernel_lab.capability.CapabilityError` -- a
    ``ValueError`` -- naming the offending field. Touches no device memory,
    builds no extension, and does not synchronise, so it is safe to call from a
    dispatcher deciding whether to route elsewhere.

    ``capability`` exists for the generalization matrix (gap G11), which
    exercises GQA ratios v1 does not *declare*. Widening it in production would
    make the support surface a runtime argument, which is exactly the thing the
    contract exists to prevent.
    """
    import torch

    # -- 1. derivation prerequisites ------------------------------------------
    for name, tensor in (("q", q), ("k_pool", k_pool), ("v_pool", v_pool), ("idx", idx)):
        if not isinstance(tensor, torch.Tensor):
            _reject(
                f"{name} must be a torch.Tensor, got {type(tensor).__name__}",
                field=name, value=type(tensor).__name__, supported="torch.Tensor",
            )
    if q.dim() != 3:
        _reject(
            f"q must be 3-D [q_len, q_heads, head_dim], got {q.dim()}-D "
            f"with shape {tuple(q.shape)}",
            field="q", value=tuple(q.shape), supported="[q_len, q_heads, head_dim]",
        )
    if k_pool.dim() != 3:
        _reject(
            f"k_pool must be 3-D [pool, kv_heads, head_dim], got {k_pool.dim()}-D "
            f"with shape {tuple(k_pool.shape)}",
            field="k_pool", value=tuple(k_pool.shape), supported="[pool, kv_heads, head_dim]",
        )
    if tuple(v_pool.shape) != tuple(k_pool.shape):
        _reject(
            f"v_pool shape {tuple(v_pool.shape)} must equal k_pool shape "
            f"{tuple(k_pool.shape)}: one pool geometry, two planes",
            field="v_pool", value=tuple(v_pool.shape), supported=tuple(k_pool.shape),
        )

    q_len, q_heads, head_dim = (int(size) for size in q.shape)
    kv_heads = int(k_pool.shape[1])
    if int(k_pool.shape[2]) != head_dim:
        _reject(
            f"k_pool head_dim {int(k_pool.shape[2])} must equal q head_dim {head_dim}",
            field="k_pool", value=int(k_pool.shape[2]), supported=head_dim,
        )

    # -- 2. the declared surface ----------------------------------------------
    if return_lse and not _LSE_AVAILABLE:
        _reject(
            "return_lse=True is unavailable: operator contract v1 writes no LSE "
            "back (gap G1). The flag is present at the boundary so the kernel-side "
            "writeback lands without an API change; see _LSE_AVAILABLE in "
            "attn_kernel_lab.ops",
            field="return_lse", value=True, supported=(False,),
        )
    check_supported(
        head_dim=head_dim,
        q_heads=q_heads,
        kv_heads=kv_heads,
        page_size=_PAGE_SIZE,
        mode=_MODE,
        mask=_MASK,
        kv_dtype=_dtype_name(k_pool.dtype),
        return_lse=return_lse,
        capability=capability,
    )

    # -- 3. structural checks the contract implies ----------------------------
    if q.dtype != torch.bfloat16:
        _reject(
            f"q dtype must be bfloat16 (contract §2), got {_dtype_name(q.dtype)}",
            field="q", value=_dtype_name(q.dtype), supported=("bfloat16",),
        )
    if not q.is_contiguous():
        _reject(
            "q must be contiguous: the preprocessing pass permutes it to "
            "[q_heads, q_len, head_dim] and a strided view would be re-laid out "
            "silently on every call",
            field="q", value=tuple(q.stride()), supported="contiguous",
        )
    if q_len < 1:
        _reject(
            f"q_len must be >= 1, got {q_len}: an empty extend has no operator "
            "meaning and must be skipped by the caller, not padded here",
            field="q_len", value=q_len, supported=">= 1",
        )
    if v_pool.dtype != k_pool.dtype:
        _reject(
            f"v_pool dtype {_dtype_name(v_pool.dtype)} must equal k_pool dtype "
            f"{_dtype_name(k_pool.dtype)}",
            field="v_pool", value=_dtype_name(v_pool.dtype),
            supported=_dtype_name(k_pool.dtype),
        )
    if idx.dim() != 1:
        _reject(
            f"idx must be 1-D [kv_len] in position order, got {idx.dim()}-D "
            f"with shape {tuple(idx.shape)}",
            field="idx", value=tuple(idx.shape), supported="[kv_len]",
        )
    if idx.dtype != torch.int64:
        _reject(
            f"idx dtype must be int64, got {_dtype_name(idx.dtype)}: the gather "
            "indexes a pool that exceeds int32 rows at production depth",
            field="idx", value=_dtype_name(idx.dtype), supported=("int64",),
        )
    kv_len = int(idx.numel())
    if kv_len < 1:
        _reject(
            "idx must name at least one kv position, got an empty index",
            field="idx", value=kv_len, supported=">= 1",
        )
    for name, tensor in (("k_pool", k_pool), ("v_pool", v_pool), ("idx", idx)):
        if not _same_device(tensor.device, q.device):
            _reject(
                f"{name} is on {tensor.device} but q is on {q.device}: every "
                "operand of one call must live on one device",
                field=name, value=str(tensor.device), supported=str(q.device),
            )
    if isinstance(prefix, bool) or not isinstance(prefix, int):
        _reject(
            f"prefix must be a Python int, got {type(prefix).__name__}: a tensor "
            "prefix would have to be read from the device to be checked",
            field="prefix", value=type(prefix).__name__, supported="int",
        )
    if not 0 <= prefix <= kv_len - q_len:
        _reject(
            f"prefix={prefix} is outside [0, kv_len - q_len] = "
            f"[0, {kv_len - q_len}] for kv_len={kv_len}, q_len={q_len}: the "
            "bottom-right causal diagonal sits at offset kv_len - q_len, so the "
            "T x (prefix + T) score rectangle must fit inside the gathered kv",
            field="prefix", value=prefix, supported=(0, kv_len - q_len),
        )
    if sm_scale is not None:
        if isinstance(sm_scale, bool) or not isinstance(sm_scale, (int, float)):
            _reject(
                f"sm_scale must be a real number or None, got "
                f"{type(sm_scale).__name__}",
                field="sm_scale", value=type(sm_scale).__name__, supported="float | None",
            )
        if not math.isfinite(sm_scale) or sm_scale <= 0.0:
            _reject(
                f"sm_scale must be positive and finite, got {sm_scale!r}: it is "
                "folded into the Q scales, so a non-positive value silently "
                "inverts or zeroes every logit",
                field="sm_scale", value=sm_scale, supported="> 0, finite",
            )
    if out is not None:
        if not isinstance(out, torch.Tensor):
            _reject(
                f"out must be a torch.Tensor or None, got {type(out).__name__}",
                field="out", value=type(out).__name__, supported="torch.Tensor | None",
            )
        if tuple(out.shape) != (q_len, q_heads, head_dim):
            _reject(
                f"out shape {tuple(out.shape)} must equal q shape "
                f"{(q_len, q_heads, head_dim)}",
                field="out", value=tuple(out.shape),
                supported=(q_len, q_heads, head_dim),
            )
        if out.dtype != torch.bfloat16:
            _reject(
                f"out dtype must be bfloat16, got {_dtype_name(out.dtype)}",
                field="out", value=_dtype_name(out.dtype), supported=("bfloat16",),
            )
        if not out.is_contiguous():
            _reject(
                "out must be contiguous", field="out",
                value=tuple(out.stride()), supported="contiguous",
            )
        if not _same_device(out.device, q.device):
            _reject(
                f"out is on {out.device} but q is on {q.device}",
                field="out", value=str(out.device), supported=str(q.device),
            )
        if _shares_storage(out, q):
            _reject(
                "out must not overlap q storage: the epilogue writes out while "
                "the padded q rows are still live",
                field="out", value="aliases q", supported="a distinct allocation",
            )
    if lse_out is not None:
        # The LSE twin of ``out``, and it exists for the same reason contract §5
        # gives for ``out``: a caller capturing a CUDA graph needs the
        # destination address to be its own and to stay put across replay. An
        # internally allocated result tensor would be a fresh address per eager
        # call and a graph-pool address the caller never sees under capture.
        if not return_lse:
            _reject(
                "lse_out was supplied but return_lse=False: a destination for "
                "an output that will not be produced is a caller mistake, not a "
                "buffer to ignore",
                field="lse_out", value="supplied", supported="return_lse=True",
            )
        if not isinstance(lse_out, torch.Tensor):
            _reject(
                f"lse_out must be a torch.Tensor or None, got {type(lse_out).__name__}",
                field="lse_out", value=type(lse_out).__name__,
                supported="torch.Tensor | None",
            )
        if tuple(lse_out.shape) != (q_len, q_heads):
            _reject(
                f"lse_out shape {tuple(lse_out.shape)} must be "
                f"{(q_len, q_heads)} -- one base-2 log-sum-exp per (row, head)",
                field="lse_out", value=tuple(lse_out.shape),
                supported=(q_len, q_heads),
            )
        if lse_out.dtype != torch.float32:
            _reject(
                f"lse_out dtype must be float32, got {_dtype_name(lse_out.dtype)}",
                field="lse_out", value=_dtype_name(lse_out.dtype),
                supported=("float32",),
            )
        if not lse_out.is_contiguous():
            _reject(
                "lse_out must be contiguous", field="lse_out",
                value=tuple(lse_out.stride()), supported="contiguous",
            )
        if not _same_device(lse_out.device, q.device):
            _reject(
                f"lse_out is on {lse_out.device} but q is on {q.device}",
                field="lse_out", value=str(lse_out.device), supported=str(q.device),
            )
    if workspace is not None:
        from .quant import FP8PrefillWorkspace

        if not isinstance(workspace, FP8PrefillWorkspace):
            # Name the MODULE too: quant.py is importable both as top-level
            # `quant` (the kernel tests' legacy style) and as
            # `attn_kernel_lab.quant`; those are distinct module objects with
            # distinct classes, and "got FP8PrefillWorkspace, expected
            # FP8PrefillWorkspace" is undebuggable without the qualifier.
            got = f"{type(workspace).__module__}.{type(workspace).__qualname__}"
            _reject(
                f"workspace must be {FP8PrefillWorkspace.__module__}."
                f"FP8PrefillWorkspace or None, got {got}"
                + (" (same class name from a different module: the package was "
                   "imported under two names)" if type(workspace).__name__ ==
                   "FP8PrefillWorkspace" else ""),
                field="workspace", value=got,
                supported=f"{FP8PrefillWorkspace.__module__}.FP8PrefillWorkspace | None",
            )
        ws_device = torch.device(workspace.device)
        if not _same_device(ws_device, q.device):
            _reject(
                f"workspace is on {ws_device} but q is on {q.device}",
                field="workspace", value=str(ws_device), supported=str(q.device),
            )

    return PrefillRequest(
        q_len=q_len,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        kv_len=kv_len,
        prefix=prefix,
        sm_scale=(1.0 / math.sqrt(head_dim)) if sm_scale is None else float(sm_scale),
        kv_dtype=_dtype_name(k_pool.dtype),
        return_lse=return_lse,
    )


def plan_workspace(
    device,
    *,
    max_kv_len: int,
    max_q_len: int,
    q_heads: int | None = None,
    kv_heads: int | None = None,
    need_vb16: bool = False,
    return_lse: bool | None = None,
    capability: OperatorCapability = V1_CAPABILITY,
):
    """A capacity-reserved workspace, sized from the declared surface.

    The capacity-mode entry point on the public path. ``quant.reserve`` will
    happily size a workspace for any divisible head pair; this wrapper takes the
    head counts from ``capability`` when the caller does not name them, and
    validates the pair through :func:`~attn_kernel_lab.capability.check_supported`
    either way, so a plan cannot quietly describe a geometry the operator
    refuses to run.

    Why this exists at all: contract §5 requires the workspace's addresses and
    capacity to stay stable across graph replay, and the default grow-on-demand
    workspace cannot promise that. A caller that intends to capture must reserve
    for the deepest request it will ever replay -- the workspace then raises
    ``WorkspaceCapacityError`` rather than reallocating under a live graph.

    Args:
        device: the device the buffers live on.
        max_kv_len: the largest ``idx.numel()`` any replayed request will carry.
        max_q_len: the largest ``q_len`` (extend chunk length).
        q_heads, kv_heads: default to the largest values ``capability`` declares.
        need_vb16: also reserve the planar BF16 V of the per-head bf16-PV
            numerics lane. The declared surface is all-fp8-PV, so it is off.
        return_lse: reserve the optional base-2 LSE buffer (contract §2).
            ``None`` (the default) follows ``capability.returns_lse``, so a
            capability that does not declare LSE plans no buffer for it rather
            than being refused for asking.
        capability: the surface to validate the head pair against.

    Returns:
        An ``FP8PrefillWorkspace`` in capacity mode. Its ``.capacity`` is the
        installed ``CapacityPlan``.

    Raises:
        CapabilityError: the head pair is outside ``capability``.
        ValueError: the maxima are not positive ints, or the pair is not
            divisible.
    """
    from .quant import FP8PrefillWorkspace

    q_heads = max(capability.q_heads) if q_heads is None else int(q_heads)
    kv_heads = max(capability.kv_heads) if kv_heads is None else int(kv_heads)
    return_lse = capability.returns_lse if return_lse is None else bool(return_lse)
    check_supported(
        head_dim=max(capability.head_dim),
        q_heads=q_heads,
        kv_heads=kv_heads,
        page_size=_PAGE_SIZE,
        mode=_MODE,
        mask=_MASK,
        kv_dtype=capability.kv_dtypes[0],
        return_lse=return_lse,
        capability=capability,
    )
    return FP8PrefillWorkspace(
        device,
        max_kv_len=max_kv_len,
        max_q_len=max_q_len,
        q_heads=q_heads,
        kv_heads=kv_heads,
        need_vb16=need_vb16,
        need_lse=return_lse,
    )


def prefill_extend(
    q,
    k_pool,
    v_pool,
    idx,
    prefix,
    *,
    workspace=None,
    out=None,
    lse_out=None,
    sm_scale=None,
    qk_i8: bool = True,
    rotate: bool = True,
    center_k: bool = True,
    return_lse: bool = False,
    capability: OperatorCapability = V1_CAPABILITY,
):
    """Bottom-right-causal paged EXTEND attention over a page_size-1 BF16 pool.

    Args:
        q: ``[q_len, q_heads, 256]`` BF16, contiguous, post-RoPE.
        k_pool, v_pool: ``[pool, kv_heads, 256]`` BF16 pools, one KV position per
            row (``page_size`` 1). Unquantized: an FP8/FP4 pool would
            double-quantize and is refused.
        idx: ``[kv_len]`` int64 pool rows in position order -- the request's
            prefix followed by the current chunk. Values are the caller's
            invariant; checking them would mean a device read.
        prefix: number of already-cached positions, in ``[0, kv_len - q_len]``.
            Query row ``r`` attends to all ``prefix`` prefix positions plus
            positions ``0..r`` of the current chunk.
        workspace: an ``FP8PrefillWorkspace`` to reuse across calls. One is
            allocated per call if omitted, which is correct but throws away the
            steady state the design exists for. A workspace from
            :func:`plan_workspace` is capacity-reserved: it never reallocates,
            and a request outside its plan raises ``WorkspaceCapacityError``
            rather than moving the addresses a captured graph is holding.
        out: ``[q_len, q_heads, 256]`` BF16 destination. Allocated if omitted.
            Must not alias ``q``.
        lse_out: ``[q_len, q_heads]`` FP32 destination for the base-2 LSE, the
            twin of ``out`` and refused unless ``return_lse=True``. Allocated if
            omitted, which is correct but leaves a per-call allocation the
            caller cannot address -- pass one when capturing a CUDA graph, where
            contract §5 requires the destination to be caller-owned and stable.
        sm_scale: softmax scale, default ``1/sqrt(head_dim)``. Folded into the Q
            scales during preprocessing; the kernel never applies it separately.
        qk_i8, rotate, center_k: preprocessing variants (contract §3). Flipping
            any of them changes the numerics of a *declared* configuration and is
            a contract-variant question, still open in contract §7 -- they are
            exposed for the numerics lanes, not as deployment knobs.
        return_lse: when True, also return the BASE-2 log-sum-exp
            (``[q_len, q_heads]`` fp32; the FA2/CUTLASS wrapper convention).
        capability: the surface to validate against. Widen only for the
            generalization matrix (gap G11).

    Returns:
        ``out`` -- ``[q_len, q_heads, 256]`` BF16; ``(out, lse)`` with ``lse``
        ``[q_len, q_heads]`` FP32 **base-2** once the LSE seam opens.

    Raises:
        CapabilityError: a ``ValueError``; the request is outside the declared
            surface. Raised before the extension is loaded and before any device
            work, so catching it costs a caller nothing.
    """
    request = check_request(
        q, k_pool, v_pool, idx, prefix,
        workspace=workspace, out=out, lse_out=lse_out, sm_scale=sm_scale,
        return_lse=return_lse, capability=capability,
    )

    # -- 4. everything below this line touches the device ---------------------
    import torch

    from . import kernel as kernel_mod
    from . import quant as quant_mod

    ext = kernel_mod.load()
    ws = quant_mod.FP8PrefillWorkspace(q.device) if workspace is None else workspace

    kv = quant_mod.gather_quantize_kv(
        ws, k_pool, v_pool, idx,
        need_vt8=True, need_vb16=False,
        center_k=center_k, qk_i8=qk_i8, rotate=rotate,
    )
    q8, qscale, mpad = quant_mod.quantize_q(
        ws, q, request.sm_scale, qk_i8=qk_i8, rotate=rotate
    )
    scratch = ws.get("o", (request.q_heads, mpad, request.head_dim), torch.bfloat16)
    # All heads take the fp8-PV path: the per-head bf16-PV fallback is a numerics
    # lane (contract §7's open env-switch item), not part of the declared surface.
    # Workspace-owned so that a capacity-reserved workspace has no per-call
    # allocation left here either; the bytes are all-ones in both modes.
    pv8_mask = ws.get("pv8_mask", (request.q_heads,), torch.uint8).fill_(1)
    # ``None`` reaches the binding's optional `lse` as nullopt, which is the
    # no-buffer path the goldens are pinned against; a buffer is only ever
    # allocated when the seam above is open.
    lse_scratch = (
        ws.get("lse", (request.q_heads, mpad), torch.float32)
        if request.return_lse
        else None
    )

    ext.fp8_prefill_attn(
        q8, kv["k8"], kv["vt8"], kv["vb16"], scratch,
        qscale, kv["kscale"], kv["vscale"],
        kv["vlog2r"], kv["vinvr"], kv["vmean"], pv8_mask,
        kv["n"], request.prefix, True, True, qk_i8, lse_scratch,
    )

    # ``scratch`` is workspace-owned and the next call overwrites it, so the
    # padded [q_heads, mpad, D] result is copied out rather than returned as a
    # view -- a returned view would alias across calls, which is the exact class
    # of bug the workspace-reuse tests exist to catch.
    if out is None:
        out = torch.empty_like(q)
    out.copy_(scratch[:, : request.q_len].permute(1, 0, 2))
    if not request.return_lse:
        return out
    lse = (
        torch.empty((request.q_len, request.q_heads), dtype=torch.float32,
                    device=q.device)
        if lse_out is None
        else lse_out
    )
    lse.copy_(lse_scratch[:, : request.q_len].permute(1, 0))
    return out, lse
