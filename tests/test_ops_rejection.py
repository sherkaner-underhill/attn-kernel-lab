# SPDX-License-Identifier: Apache-2.0
"""Every declared-surface rejection, asserted on CPU with no kernel launch.

This is gap G4's other half: ``capability.check_supported`` has existed since
Phase 1 but was off the call path, so an out-of-surface request reached the
extension and came back as a ``RuntimeError`` from a ``TORCH_CHECK`` -- an
exception a consuming framework cannot distinguish from a defect.
``attn_kernel_lab.ops`` puts the check on the path; this file pins the result.

Upstream form (``upstream/EVIDENCE_GAP_ANALYSIS_2026-08-29.md`` §2.2): FlashInfer
#4272 asserts five *distinct* ``ValueError`` messages and #3518 gained a
rejection test because a reviewer asked "nor is it covered in the tests?". So
each case here asserts the message text as well as the typed ``field``, because
a rejection that says the wrong thing is a rejection a caller cannot act on.

Two properties are load-bearing beyond the individual cases:

- **No device, no compiler.** Everything runs on small CPU tensors of the
  production *geometry* (24:4 GQA, head_dim 256) at toy lengths. The module
  skips cleanly where torch is absent, so the CPU CI lane collects it and the
  no-torch lane does not error.
- **Validation never loads the extension.** An autouse fixture replaces
  ``kernel.load`` with a tripwire. A rejection that reached it would raise
  ``_KernelLoadAttempted`` instead of ``CapabilityError`` and fail the case,
  which is what makes "before any CUDA work" a tested claim rather than a
  comment.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

torch = pytest.importorskip(
    "torch", reason="the operator surface needs torch tensors; CPU-only is enough"
)

from attn_kernel_lab import (  # noqa: E402
    CapabilityError,
    GENERALIZATION_CAPABILITY,
    kernel as kernel_mod,
    ops,
)

Q_LEN, Q_HEADS, KV_HEADS, HEAD_DIM, KV_LEN = 8, 24, 4, 256, 32
PREFIX = KV_LEN - Q_LEN


class _KernelLoadAttempted(RuntimeError):
    """Tripwire: raised if a code path under test tries to build the extension."""


@pytest.fixture(autouse=True)
def no_kernel_load(monkeypatch):
    """Make any launch attempt loud, and distinct from a capability rejection."""

    def _trip(*args, **kwargs):
        raise _KernelLoadAttempted("kernel.load() must not be reached by validation")

    monkeypatch.setattr(kernel_mod, "load", _trip)


def _call(**overrides) -> dict:
    """One valid production-geometry request, as keyword arguments.

    Toy lengths, real geometry: the declared surface is about head_dim, head
    counts, page size, mode, mask and dtypes, none of which depend on length.
    """
    call = dict(
        q=torch.zeros((Q_LEN, Q_HEADS, HEAD_DIM), dtype=torch.bfloat16),
        k_pool=torch.zeros((KV_LEN, KV_HEADS, HEAD_DIM), dtype=torch.bfloat16),
        v_pool=torch.zeros((KV_LEN, KV_HEADS, HEAD_DIM), dtype=torch.bfloat16),
        idx=torch.arange(KV_LEN, dtype=torch.int64),
        prefix=PREFIX,
    )
    call.update(overrides)
    return call


def _geometry(*, q_heads=Q_HEADS, kv_heads=KV_HEADS, head_dim=HEAD_DIM) -> dict:
    """A request at a different head geometry, everything else unchanged."""
    return _call(
        q=torch.zeros((Q_LEN, q_heads, head_dim), dtype=torch.bfloat16),
        k_pool=torch.zeros((KV_LEN, kv_heads, head_dim), dtype=torch.bfloat16),
        v_pool=torch.zeros((KV_LEN, kv_heads, head_dim), dtype=torch.bfloat16),
    )


# --------------------------------------------------------- the declared surface

SURFACE_CASES = [
    pytest.param(
        lambda: _geometry(head_dim=128),
        {},
        "head_dim",
        r"head_dim=128 is outside the declared support surface",
        id="head_dim-128",
    ),
    pytest.param(
        lambda: _geometry(head_dim=64),
        {},
        "head_dim",
        r"head_dim=64 is outside the declared support surface",
        id="head_dim-64",
    ),
    pytest.param(
        lambda: _geometry(q_heads=8, kv_heads=4),
        {},
        "q_heads",
        r"q_heads=8 is outside the declared support surface",
        id="q_heads-8",
    ),
    pytest.param(
        lambda: _geometry(q_heads=24, kv_heads=2),
        {},
        "kv_heads",
        r"kv_heads=2 is outside the declared support surface",
        id="kv_heads-2",
    ),
    pytest.param(
        lambda: _call(
            k_pool=torch.zeros((KV_LEN, KV_HEADS, HEAD_DIM), dtype=torch.float8_e4m3fn),
            v_pool=torch.zeros((KV_LEN, KV_HEADS, HEAD_DIM), dtype=torch.float8_e4m3fn),
        ),
        {},
        "kv_dtype",
        r"kv_dtype='float8_e4m3fn' is outside the declared support surface",
        id="kv_pool-fp8-would-double-quantize",
    ),
]

# --------------------------------- structure the contract implies but §2 omits

STRUCTURE_CASES = [
    pytest.param(
        lambda: _call(q=torch.zeros((Q_LEN, HEAD_DIM), dtype=torch.bfloat16)),
        {}, "q", r"q must be 3-D \[q_len, q_heads, head_dim\], got 2-D",
        id="q-rank-2",
    ),
    pytest.param(
        lambda: _call(q="not a tensor"),
        {}, "q", r"q must be a torch\.Tensor, got str",
        id="q-not-a-tensor",
    ),
    pytest.param(
        lambda: _call(
            q=torch.zeros((Q_LEN, Q_HEADS, HEAD_DIM * 2), dtype=torch.bfloat16)[..., ::2]
        ),
        {}, "q", r"q must be contiguous",
        id="q-strided",
    ),
    pytest.param(
        lambda: _call(q=torch.zeros((Q_LEN, Q_HEADS, HEAD_DIM), dtype=torch.float16)),
        {}, "q", r"q dtype must be bfloat16",
        id="q-dtype-fp16",
    ),
    pytest.param(
        lambda: _call(q=torch.zeros((0, Q_HEADS, HEAD_DIM), dtype=torch.bfloat16)),
        {}, "q_len", r"q_len must be >= 1",
        id="q_len-zero",
    ),
    pytest.param(
        lambda: _call(k_pool=torch.zeros((KV_LEN, HEAD_DIM), dtype=torch.bfloat16)),
        {}, "k_pool", r"k_pool must be 3-D \[pool, kv_heads, head_dim\], got 2-D",
        id="k_pool-rank-2",
    ),
    pytest.param(
        lambda: _call(
            k_pool=torch.zeros((KV_LEN, KV_HEADS, 128), dtype=torch.bfloat16),
            v_pool=torch.zeros((KV_LEN, KV_HEADS, 128), dtype=torch.bfloat16),
        ),
        {}, "k_pool", r"k_pool head_dim 128 must equal q head_dim 256",
        id="k_pool-head_dim-disagrees-with-q",
    ),
    pytest.param(
        lambda: _call(v_pool=torch.zeros((KV_LEN, KV_HEADS + 1, HEAD_DIM), dtype=torch.bfloat16)),
        {}, "v_pool", r"v_pool shape \(32, 5, 256\) must equal k_pool shape",
        id="v_pool-shape-disagrees",
    ),
    pytest.param(
        lambda: _call(v_pool=torch.zeros((KV_LEN, KV_HEADS, HEAD_DIM), dtype=torch.float16)),
        {}, "v_pool", r"v_pool dtype float16 must equal k_pool dtype bfloat16",
        id="v_pool-dtype-disagrees",
    ),
    pytest.param(
        lambda: _call(idx=torch.zeros((2, KV_LEN), dtype=torch.int64)),
        {}, "idx", r"idx must be 1-D \[kv_len\] in position order, got 2-D",
        id="idx-rank-2",
    ),
    pytest.param(
        lambda: _call(idx=torch.arange(KV_LEN, dtype=torch.int32)),
        {}, "idx", r"idx dtype must be int64",
        id="idx-int32",
    ),
    pytest.param(
        lambda: _call(idx=torch.zeros((0,), dtype=torch.int64)),
        {}, "idx", r"idx must name at least one kv position",
        id="idx-empty",
    ),
    pytest.param(
        lambda: _call(idx=torch.zeros((KV_LEN,), dtype=torch.int64, device="meta")),
        {}, "idx", r"idx is on meta but q is on cpu",
        id="idx-wrong-device",
    ),
    pytest.param(
        lambda: _call(prefix=-1),
        {}, "prefix", r"prefix=-1 is outside \[0, kv_len - q_len\] = \[0, 24\]",
        id="prefix-negative",
    ),
    pytest.param(
        lambda: _call(prefix=KV_LEN),
        {}, "prefix", r"prefix=32 is outside \[0, kv_len - q_len\] = \[0, 24\]",
        id="prefix-past-the-gathered-kv",
    ),
    pytest.param(
        lambda: _call(prefix=8.0),
        {}, "prefix", r"prefix must be a Python int, got float",
        id="prefix-float",
    ),
    pytest.param(
        lambda: _call(), {"sm_scale": 0.0}, "sm_scale",
        r"sm_scale must be positive and finite",
        id="sm_scale-zero",
    ),
    pytest.param(
        lambda: _call(), {"sm_scale": float("inf")}, "sm_scale",
        r"sm_scale must be positive and finite",
        id="sm_scale-inf",
    ),
    pytest.param(
        lambda: _call(), {"sm_scale": "0.0625"}, "sm_scale",
        r"sm_scale must be a real number or None, got str",
        id="sm_scale-str",
    ),
    pytest.param(
        lambda: _call(),
        {"out": torch.zeros((Q_LEN, Q_HEADS, HEAD_DIM // 2), dtype=torch.bfloat16)},
        "out", r"out shape \(8, 24, 128\) must equal q shape \(8, 24, 256\)",
        id="out-wrong-shape",
    ),
    pytest.param(
        lambda: _call(),
        {"out": torch.zeros((Q_LEN, Q_HEADS, HEAD_DIM), dtype=torch.float32)},
        "out", r"out dtype must be bfloat16",
        id="out-wrong-dtype",
    ),
    pytest.param(
        lambda: _call(),
        {"out": torch.zeros((Q_LEN, Q_HEADS, HEAD_DIM * 2), dtype=torch.bfloat16)[..., ::2]},
        "out", r"out must be contiguous",
        id="out-strided",
    ),
    pytest.param(
        lambda: _call(),
        {"out": torch.zeros((Q_LEN, Q_HEADS, HEAD_DIM), dtype=torch.bfloat16, device="meta")},
        "out", r"out is on meta but q is on cpu",
        id="out-wrong-device",
    ),
    pytest.param(
        lambda: _call(), {"workspace": object()}, "workspace",
        r"workspace must be attn_kernel_lab\.quant\.FP8PrefillWorkspace or None, got builtins\.object",
        id="workspace-wrong-type",
    ),
]


@pytest.mark.parametrize("build, kwargs, field, match", SURFACE_CASES + STRUCTURE_CASES)
def test_rejected_before_any_device_work(build, kwargs, field, match):
    """Outside the surface -> CapabilityError naming the field, no launch."""
    with pytest.raises(CapabilityError, match=match) as excinfo:
        ops.prefill_extend(**build(), **kwargs)
    assert excinfo.value.field == field


def test_out_must_not_alias_q():
    """Upstream's own wording (`out must not overlap q storage`): the epilogue
    writes ``out`` while the padded Q rows are still live, so an aliased buffer
    corrupts the tail of its own input."""
    call = _call()
    with pytest.raises(CapabilityError, match=r"out must not overlap q storage") as excinfo:
        ops.prefill_extend(**call, out=call["q"])
    assert excinfo.value.field == "out"


def test_capability_error_is_a_valueerror():
    """The upstream convention, and the one class a consuming framework catches.

    FlashInfer rejection tests are written ``pytest.raises(ValueError, ...)``;
    ours must satisfy that spelling without the caller importing our types.
    """
    with pytest.raises(ValueError):
        ops.prefill_extend(**_geometry(head_dim=128))


def test_every_rejection_is_distinguishable_from_a_defect():
    """A ``RuntimeError`` from the extension means a bug; a ``CapabilityError``
    means "route elsewhere". Collapsing them is how a silent fallback becomes a
    silent performance regression, so the tripwire must never fire."""
    with pytest.raises(CapabilityError):
        ops.prefill_extend(**_call(prefix=-1))
    with pytest.raises(CapabilityError):
        ops.prefill_extend(**_geometry(q_heads=8, kv_heads=4))


# ------------------------------------------------------------- the happy path


def test_production_shape_validates_without_touching_the_kernel():
    """The declared shape must pass validation and then -- and only then -- try
    to load the extension. The tripwire standing in for ``kernel.load`` is what
    proves the ordering: reaching it means every check passed, and reaching it
    *first* would mean validation had a device dependency."""
    with pytest.raises(_KernelLoadAttempted):
        ops.prefill_extend(**_call())


def test_check_request_returns_the_derived_geometry():
    """A caller that wants the validation without the launch gets the numbers."""
    request = ops.check_request(**_call())
    assert (request.q_len, request.q_heads, request.kv_heads) == (Q_LEN, Q_HEADS, KV_HEADS)
    assert (request.head_dim, request.kv_len, request.prefix) == (HEAD_DIM, KV_LEN, PREFIX)
    assert request.kv_dtype == "bfloat16"
    assert request.return_lse is False
    assert request.sm_scale == pytest.approx(1.0 / HEAD_DIM**0.5)


def test_supplied_sm_scale_survives_validation():
    assert ops.check_request(**_call(), sm_scale=0.125).sm_scale == pytest.approx(0.125)


def test_supplied_out_and_workspace_are_accepted():
    out = torch.zeros((Q_LEN, Q_HEADS, HEAD_DIM), dtype=torch.bfloat16)
    assert ops.check_request(**_call(), out=out).q_len == Q_LEN


def test_lse_seam_is_open_and_validation_accepts_it():
    """The kernel-side writeback landed (gap G1 closed); the seam is open and a
    return_lse request validates cleanly."""
    assert ops._LSE_AVAILABLE is True
    assert ops.check_request(**_call(), return_lse=True).return_lse is True


def test_lse_refusal_branch_survives_for_a_closed_seam(monkeypatch):
    """The refusal path must keep working for any future capability that does
    not write LSE (e.g. a wgmma implementation before its own writeback)."""
    monkeypatch.setattr(ops, "_LSE_AVAILABLE", False)
    with pytest.raises(CapabilityError, match=r"return_lse"):
        ops.check_request(**_call(), return_lse=True)


def test_generalization_capability_widens_only_what_g11_sweeps():
    """The GQA sweep the upstream-shape matrix runs (gap G11) is not a support
    declaration: 8:2 validates against the generalization surface and is refused
    by the declared one, and head_dim stays pinned in both."""
    ops.check_request(**_geometry(q_heads=8, kv_heads=2), capability=GENERALIZATION_CAPABILITY)
    with pytest.raises(CapabilityError):
        ops.check_request(**_geometry(q_heads=8, kv_heads=2))
    with pytest.raises(CapabilityError, match=r"head_dim"):
        ops.check_request(**_geometry(head_dim=128), capability=GENERALIZATION_CAPABILITY)
