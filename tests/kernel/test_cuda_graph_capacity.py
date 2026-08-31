# SPDX-License-Identifier: Apache-2.0
"""Capacity-stable workspace + CUDA-graph capture/replay (gap G5).

WHAT THIS FILE IS EVIDENCE FOR, AND WHAT IT IS NOT
==================================================
``capability.V1_CAPABILITY.cuda_graph`` says ``eager_only``. The reason it said
so was structural: ``FP8PrefillWorkspace.get()`` reallocated on growth, a
captured graph bakes in the device addresses it saw, and a reallocation between
capture and replay makes the replay read freed memory. ``ws.reserve()`` removes
that reason -- it preallocates at declared maxima and raises
``WorkspaceCapacityError`` instead of growing -- and this file is the test that
says so out loud.

It does **not** promote the declaration. This lane runs on the local RTX 4090
(SM89), which the repository treats as a *development* target: it proves
correctness and never performance or support. The flip to ``"supported"`` is one
constant (``capability.CUDA_GRAPH_DECLARED``) and it waits on this file passing
on the pinned SM120 target. See that constant's comment for the full checklist.

THE PATTERN
===========
Borrowed wholesale from FlashInfer PR #4272's ``test_fmha_v2_prefill.py``, as
catalogued in ``upstream/EVIDENCE_GAP_ANALYSIS_2026-08-29.md``: warm up on a
side ``torch.cuda.Stream()``, capture, **fill the output with NaN**, replay,
assert ``rtol=0, atol=0`` against the eager result -- then do it again with
*changed values* copied into the same static tensors, which is the half that
catches a graph that captured data instead of pointers.

Two properties beyond #4272's, both of them contract-level here:

* **Workspace poisoning across replay** (contract section 5). The reserved
  ``vt8`` is filled with ``0x7F`` -- an E4M3 NaN encoding -- before a replay. The
  boundary tile's tail zeroing must still keep ``0 * NaN`` out of the PV matmul.
* **The sync tripwire stays silent.** A capacity-mode call must not synchronise
  the host against the device; ``torch.cuda.set_sync_debug_mode("warn")`` is
  armed around a block of steady-state calls and any warning is a failure. This
  is the same tripwire ``ATTN_KERNEL_LAB_SYNC_DEBUG=1`` arms globally in
  ``conftest.py``; here it is armed locally so the assertion holds on every run
  rather than only on an opted-in one.

RUNNING IT
==========
    python3 -m pytest -q tests/kernel/test_cuda_graph_capacity.py
"""

from __future__ import annotations

import os
import sys
import warnings

import pytest
import torch

HEAD_DIM = 256
PKG = os.path.join(os.path.dirname(__file__), "..", "..", "src", "attn_kernel_lab")
sys.path.insert(0, os.path.abspath(PKG))
sys.path.insert(0, os.path.abspath(os.path.join(PKG, "..")))

from attn_kernel_lab import ops  # noqa: E402
from attn_kernel_lab.quant import (  # noqa: E402
    BLK,
    FP8PrefillWorkspace,
    WorkspaceCapacityError,
)

# The DECLARED surface: capacity mode is a memory-provenance change, so it is
# exercised at the geometry the contract actually declares (24:4, contract §2).
Q_HEADS, KV_HEADS = 24, 4
MAX_Q, MAX_KV = 128, 1024


def _data(q_len: int, kv_len: int, seed: int, device="cuda"):
    """Post-RoPE-shaped BF16 inputs; CPU generator so the bytes are fixed."""
    g = torch.Generator(device="cpu").manual_seed(seed)

    def mk(*shape):
        return (
            torch.empty(*shape, dtype=torch.float32)
            .uniform_(-1.0, 1.0, generator=g)
            .to(torch.bfloat16)
            .to(device)
        )

    return (
        mk(q_len, Q_HEADS, HEAD_DIM),
        mk(kv_len, KV_HEADS, HEAD_DIM),
        mk(kv_len, KV_HEADS, HEAD_DIM),
    )


def _reserved() -> FP8PrefillWorkspace:
    return ops.plan_workspace(
        torch.device("cuda"),
        max_kv_len=MAX_KV,
        max_q_len=MAX_Q,
        return_lse=True,
    )


def _bits(t: torch.Tensor) -> torch.Tensor:
    """A view whose equality is BYTE equality (NaN != NaN would hide a poison)."""
    return t.view(
        torch.uint8 if t.element_size() == 1 else {2: torch.int16, 4: torch.int32}[t.element_size()]
    )


# =========================================================================
# 1. Capacity mode is a memory decision, not a numerics one
# =========================================================================


def test_capacity_output_is_bit_identical_to_grow_on_demand():
    """THE bit-identity claim, asserted at the byte level on both outputs.

    Both modes hand every consumer ``buf[:numel].view(shape)`` -- offset 0, the
    exact per-call shape, standard contiguous strides -- so the only difference
    is the parent allocation's address and size. Per-call shapes, the slab loop
    bounds, the rotation GEMM batching and every accumulation order are
    functions of N, T, BLK and SLAB alone. If this ever fails, capacity mode has
    become a contract variant and must be declared as one.
    """
    q, k, v = _data(96, 700, seed=1)
    idx = torch.arange(700, device="cuda")

    grown = FP8PrefillWorkspace(torch.device("cuda"))
    out_g, lse_g = ops.prefill_extend(q, k, v, idx, 604, workspace=grown, return_lse=True)

    reserved = _reserved()
    out_c, lse_c = ops.prefill_extend(q, k, v, idx, 604, workspace=reserved, return_lse=True)

    assert reserved.capacity is not None and grown.capacity is None
    assert torch.equal(_bits(out_g), _bits(out_c)), "capacity mode moved the output bytes"
    assert torch.equal(_bits(lse_g), _bits(lse_c)), "capacity mode moved the LSE bytes"


@pytest.mark.parametrize(
    "q_len,kv_len,prefix",
    [
        (128, 1024, 896),  # the plan's maxima exactly
        (63, 63, 0),  # sub-tile, ragged, self-attention only
        (100, 999, 899),  # ragged everything
        (1, 1, 0),  # degenerate
    ],
)
def test_capacity_matches_grow_on_demand_across_geometries(q_len, kv_len, prefix):
    """One plan, many shapes: every geometry the plan admits must be identical
    to what the grown workspace produces for it."""
    q, k, v = _data(q_len, kv_len, seed=q_len * 31 + kv_len)
    idx = torch.arange(kv_len, device="cuda")

    grown = FP8PrefillWorkspace(torch.device("cuda"))
    reserved = _reserved()
    assert reserved.capacity.admits(kv_len, q_len)

    out_g = ops.prefill_extend(q, k, v, idx, prefix, workspace=grown)
    out_c = ops.prefill_extend(q, k, v, idx, prefix, workspace=reserved)
    assert torch.equal(_bits(out_g), _bits(out_c))


# =========================================================================
# 2. The lock: a plan miss is loud, and pointers do not move inside the plan
# =========================================================================


def test_outgrowing_the_plan_raises_instead_of_reallocating():
    """The whole point of the lock. Silently growing here is what made the
    declaration ``eager_only`` in the first place."""
    ws = ops.plan_workspace(torch.device("cuda"), max_kv_len=256, max_q_len=64)
    q, k, v = _data(128, 512, seed=2)  # both maxima exceeded
    idx = torch.arange(512, device="cuda")
    with pytest.raises(WorkspaceCapacityError, match="capacity plan"):
        ops.prefill_extend(q, k, v, idx, 384, workspace=ws)


def test_capacity_error_is_not_a_capability_error():
    """A consuming framework catches ``CapabilityError`` to route to a stock
    backend. A plan miss must NOT be catchable that way: it is a defect in the
    caller's plan, and routing around it would hide a live graph replaying
    against a reallocated pointer."""
    from attn_kernel_lab.capability import CapabilityError

    ws = ops.plan_workspace(torch.device("cuda"), max_kv_len=128, max_q_len=64)
    q, k, v = _data(64, 512, seed=3)
    idx = torch.arange(512, device="cuda")
    with pytest.raises(WorkspaceCapacityError) as excinfo:
        ops.prefill_extend(q, k, v, idx, 448, workspace=ws)
    assert not isinstance(excinfo.value, CapabilityError)
    assert not isinstance(excinfo.value, ValueError)


def test_release_capacity_restores_growth():
    ws = ops.plan_workspace(torch.device("cuda"), max_kv_len=128, max_q_len=64)
    ws.release_capacity()
    assert ws.capacity is None
    q, k, v = _data(64, 512, seed=4)
    ops.prefill_extend(q, k, v, torch.arange(512, device="cuda"), 448, workspace=ws)


def test_workspace_pointers_are_stable_across_calls():
    """Contract §5: "their addresses and capacity must remain stable across
    graph replay". Asserted directly, across a shrinking-then-growing sequence
    that would have reallocated under the default workspace."""
    ws = _reserved()
    before = ws.buffer_pointers()
    assert before, "reserve() allocated nothing"
    for q_len, kv_len, prefix in ((32, 64, 32), (128, 1024, 896), (16, 200, 184)):
        q, k, v = _data(q_len, kv_len, seed=q_len + kv_len)
        ops.prefill_extend(
            q, k, v, torch.arange(kv_len, device="cuda"), prefix, workspace=ws, return_lse=True
        )
    after = ws.buffer_pointers()
    assert after == before, {
        name: (before.get(name), after.get(name))
        for name in set(before) | set(after)
        if before.get(name) != after.get(name)
    }


def test_reserve_rejects_an_indivisible_head_pair():
    with pytest.raises(ValueError, match="divisible"):
        FP8PrefillWorkspace(torch.device("cuda")).reserve(128, 64, q_heads=25, kv_heads=4)


# =========================================================================
# 3. The #4272 pattern: capture, NaN-poison, replay, change values, replay
# =========================================================================


def _capture(ws, q, k, v, idx, prefix, out, lse_out):
    """Warm on a side stream, then capture exactly one iteration.

    The side-stream warm-up is required by the CUDA graph API (it forces the
    lazy one-time allocations -- cuBLAS workspaces, the JIT'd module's own
    first-call state -- to happen before capture) and it is what #4272's test
    does too.
    """

    def step():
        ops.prefill_extend(
            q, k, v, idx, prefix, workspace=ws, out=out, return_lse=True, lse_out=lse_out
        )

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    return graph, step


@pytest.mark.parametrize(
    "q_len,kv_len,prefix",
    [
        (128, 1024, 896),  # BLK-aligned N: no boundary tile, no tail zeroing
        (100, 999, 899),  # RAGGED N: exercises the boundary-tile tail zeroing,
        # which used to lower to a synchronising index_put_ and
        # therefore made exactly these shapes uncapturable while
        # the aligned ones captured fine
    ],
    ids=["aligned", "ragged"],
)
def test_graph_capture_replay_matches_eager_and_survives_changed_values(q_len, kv_len, prefix):
    """The full quantize+kernel call, captured and replayed.

    Three replays, each answering a different question:

      1. after a NaN poison of the output buffer -- does the replay actually
         write, or was the eager warm-up's result still sitting there?
      2. with different data copied into the same static tensors -- did the
         graph capture pointers (correct) or values (a graph that silently
         returns a stale answer forever)?
      3. after a NaN poison of the reserved fp8 ``V^T`` workspace -- contract §5's
         stale-V hazard, which the boundary-tile tail zeroing must neutralise,
         under replay rather than under an eager call.
    """
    ws = _reserved()
    plan = ws.capacity

    q_s, k_s, v_s = _data(q_len, kv_len, seed=101)
    idx_s = torch.arange(kv_len, device="cuda")
    out_s = torch.empty_like(q_s)
    lse_s = torch.empty((q_len, Q_HEADS), dtype=torch.float32, device="cuda")

    graph, _ = _capture(ws, q_s, k_s, v_s, idx_s, prefix, out_s, lse_s)

    def eager_reference():
        """The same inputs through the same workspace, eagerly."""
        out_e = torch.empty_like(q_s)
        lse_e = torch.empty_like(lse_s)
        ops.prefill_extend(
            q_s, k_s, v_s, idx_s, prefix, workspace=ws, out=out_e, return_lse=True, lse_out=lse_e
        )
        return out_e, lse_e

    # -- replay 1: NaN-poisoned output ------------------------------------
    out_s.fill_(float("nan"))
    lse_s.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    got_out, got_lse = out_s.clone(), lse_s.clone()
    assert torch.isfinite(got_out).all(), "replay left NaN in the output"
    assert torch.isfinite(got_lse).all(), "replay left NaN in the LSE"

    want_out, want_lse = eager_reference()
    torch.testing.assert_close(got_out, want_out, rtol=0, atol=0)
    torch.testing.assert_close(got_lse, want_lse, rtol=0, atol=0)
    assert torch.equal(_bits(got_out), _bits(want_out))
    assert torch.equal(_bits(got_lse), _bits(want_lse))

    # -- replay 2: changed values, same addresses -------------------------
    q_b, k_b, v_b = _data(q_len, kv_len, seed=202)
    q_s.copy_(q_b)
    k_s.copy_(k_b)
    v_s.copy_(v_b)
    out_s.fill_(float("nan"))
    lse_s.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    got_out_b, got_lse_b = out_s.clone(), lse_s.clone()

    want_out_b, want_lse_b = eager_reference()
    torch.testing.assert_close(got_out_b, want_out_b, rtol=0, atol=0)
    torch.testing.assert_close(got_lse_b, want_lse_b, rtol=0, atol=0)
    assert not torch.equal(_bits(got_out_b), _bits(got_out)), (
        "changed inputs produced identical output bytes: the graph is replaying "
        "a captured VALUE, not re-reading the static tensors"
    )

    # -- replay 3: NaN-poisoned fp8 V^T workspace (contract §5) ------------
    ws.get("vt8", (KV_HEADS, plan.ntmax, HEAD_DIM, BLK), torch.uint8).fill_(0x7F)
    out_s.fill_(float("nan"))
    lse_s.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(out_s).all(), (
        "an E4M3 NaN left in the reserved V^T poisoned 0*NaN in the PV matmul "
        "across replay -- the boundary-tile tail zeroing is not covering it"
    )
    assert torch.equal(_bits(out_s), _bits(got_out_b)), (
        "a poisoned workspace changed the answer: the pipeline is reading rows "
        "it did not write this call"
    )


def test_graph_replay_is_repeatable():
    """N replays of one capture must all agree, byte for byte."""
    q_len, kv_len, prefix = 64, 512, 448
    ws = _reserved()
    q_s, k_s, v_s = _data(q_len, kv_len, seed=303)
    idx_s = torch.arange(kv_len, device="cuda")
    out_s = torch.empty_like(q_s)
    lse_s = torch.empty((q_len, Q_HEADS), dtype=torch.float32, device="cuda")

    graph, _ = _capture(ws, q_s, k_s, v_s, idx_s, prefix, out_s, lse_s)
    graph.replay()
    torch.cuda.synchronize()
    first = out_s.clone()
    for _ in range(4):
        out_s.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(_bits(out_s), _bits(first))


def test_capture_does_not_move_the_workspace():
    """Capture and replay must not have reallocated anything: if they did, the
    graph is holding addresses nobody owns any more."""
    ws = _reserved()
    q_s, k_s, v_s = _data(64, 512, seed=404)
    idx_s = torch.arange(512, device="cuda")
    out_s = torch.empty_like(q_s)
    lse_s = torch.empty((64, Q_HEADS), dtype=torch.float32, device="cuda")
    graph, _ = _capture(ws, q_s, k_s, v_s, idx_s, 448, out_s, lse_s)
    before = ws.buffer_pointers()
    graph.replay()
    torch.cuda.synchronize()
    assert ws.buffer_pointers() == before


# =========================================================================
# 4. The promotion is genuinely one line
# =========================================================================


def test_the_cuda_graph_declaration_is_a_single_constant():
    """Both capabilities read ``CUDA_GRAPH_DECLARED``, so promoting the claim is
    editing one constant rather than hunting for string literals.

    Deliberately asserts the WIRING and not the value: this test must keep
    passing after the qualification session flips the constant, or it becomes a thing to
    delete during a promotion, which is when nobody wants to be reading tests.
    """
    from attn_kernel_lab.capability import (
        CUDA_GRAPH_DECLARED,
        GENERALIZATION_CAPABILITY,
        V1_CAPABILITY,
    )

    assert V1_CAPABILITY.cuda_graph == CUDA_GRAPH_DECLARED
    assert GENERALIZATION_CAPABILITY.cuda_graph == CUDA_GRAPH_DECLARED
    assert CUDA_GRAPH_DECLARED in {"eager_only", "supported"}


# =========================================================================
# 5. The sync tripwire
# =========================================================================


@pytest.mark.parametrize("kv_len", [1024, 700], ids=["aligned", "ragged"])
def test_capacity_mode_call_does_not_synchronise(kv_len):
    """``set_sync_debug_mode("warn")`` armed around steady-state calls.

    The ``ragged`` row is the one that earns its keep: it is the only shape that
    reaches the boundary-tile tail zeroing, and that store used to synchronise.

    An implicit device->host synchronisation is not merely slow here: it is
    ILLEGAL during graph capture, so this assertion is the cheap standing guard
    that keeps the capture above capturable. The mode is process-global and
    fires on legitimate syncs in test bodies too, which is why the armed window
    holds nothing but the calls themselves -- no ``.item()``, no ``.clone()``,
    no ``synchronize()``.

    ``ATTN_KERNEL_LAB_SYNC_DEBUG=1`` arms the same tripwire globally for a whole
    run (see conftest.py); this test does not depend on it.
    """
    ws = _reserved()
    q, k, v = _data(96, kv_len, seed=505)
    idx = torch.arange(kv_len, device="cuda")
    out = torch.empty_like(q)
    lse = torch.empty((96, Q_HEADS), dtype=torch.float32, device="cuda")
    prefix = kv_len - 96

    # Warm-up OUTSIDE the armed window: first-call laziness (extension load,
    # cuBLAS handle and workspace, allocator segments) legitimately synchronises.
    for _ in range(3):
        ops.prefill_extend(
            q, k, v, idx, prefix, workspace=ws, out=out, return_lse=True, lse_out=lse
        )
    torch.cuda.synchronize()

    previous = torch.cuda.get_sync_debug_mode()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            for _ in range(3):
                ops.prefill_extend(
                    q, k, v, idx, 604, workspace=ws, out=out, return_lse=True, lse_out=lse
                )
        finally:
            torch.cuda.set_sync_debug_mode(previous)

    # ``set_sync_debug_mode`` announces itself ("prototype feature ...") the
    # moment it is armed; that one is the tripwire reporting for duty, not a
    # finding. Only the operation-level message counts.
    syncs = [str(w.message) for w in caught if "called a synchronizing" in str(w.message)]
    assert not syncs, (
        "the capacity-mode call path synchronised the host against the device, "
        f"which is illegal during graph capture: {syncs}"
    )
