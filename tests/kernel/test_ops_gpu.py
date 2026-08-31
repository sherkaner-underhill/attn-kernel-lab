# SPDX-License-Identifier: Apache-2.0
"""The public ops surface on a real GPU: the wrapper must add nothing and lose
nothing relative to the raw pipeline it wraps."""
from __future__ import annotations

import math
import os
import sys

import pytest
import torch

HEAD_DIM = 256
PKG = os.path.join(os.path.dirname(__file__), "..", "..", "src", "attn_kernel_lab")
sys.path.insert(0, os.path.abspath(PKG))
sys.path.insert(0, os.path.abspath(os.path.join(PKG, "..")))

import oracle_a  # noqa: E402
from attn_kernel_lab import ops  # noqa: E402


def _inputs(T, N, H=24, KVH=4, seed=0):  # the DECLARED surface: 24:4 only
    g = torch.Generator(device="cpu").manual_seed(seed)
    mk = lambda *shape: (torch.empty(*shape, dtype=torch.float32)
                         .uniform_(-1, 1, generator=g).to(torch.bfloat16).cuda())
    return mk(T, H, HEAD_DIM), mk(N, KVH, HEAD_DIM), mk(N, KVH, HEAD_DIM)


def test_prefill_extend_matches_oracle_with_lse():
    """Oracle attention over the kernel's OWN packed bytes.

    Comparing against a CPU-side oracle preprocessing would smuggle in
    cross-device cuBLAS rotation differences: the two devices pick different
    fp32 GEMM algorithms, occasional values land on quantization boundaries,
    and a flipped byte is a DIFFERENT quantized operator, not a kernel error
    (observed: same test 5e-3 on SM89, 1.5e-2 on SM120, purely from packing).
    Fixing the bytes makes the tolerance mean what it claims: mainloop vs
    contract math only."""
    import math as _math

    import quant as q_mod  # the kernel-side (GPU) packing

    T, N, prefix = 64, 128, 64
    q, k, v = _inputs(T, N, seed=3)
    idx = torch.arange(N, device="cuda")
    out, lse = ops.prefill_extend(q, k, v, idx, prefix, return_lse=True)

    ws = q_mod.FP8PrefillWorkspace(q.device)
    kv = q_mod.gather_quantize_kv(ws, k, v, idx, need_vt8=True, need_vb16=False)
    q8, qscale, mpad = q_mod.quantize_q(ws, q, 1.0 / _math.sqrt(HEAD_DIM))
    KVH = k.shape[1]
    nt = (N + 63) // 64
    qp = oracle_a.QPack(q8=q8.cpu().view(torch.int8), qscale=qscale.cpu(), mpad=mpad)
    kvp = oracle_a.KVPack(
        k8=kv["k8"].cpu().view(torch.int8).view(KVH, -1, HEAD_DIM)[:, : nt * 64],
        kscale=kv["kscale"].cpu()[:, :nt],
        kmean=torch.zeros(KVH, HEAD_DIM),  # unused by attention()
        vt8=kv["vt8"].cpu().view(KVH, -1, HEAD_DIM, 64)[:, :nt].view(torch.float8_e4m3fn),
        vst_scaled=torch.zeros(KVH, nt),   # unused by attention()
        vscale=kv["vscale"].cpu(),
        vlog2r=kv["vlog2r"].cpu()[:, :nt],
        vinvr=kv["vinvr"].cpu()[:, :nt],
        vmean=kv["vmean"].cpu(),
        n=N,
    )
    want_out, want_lse = oracle_a.attention(qp, kvp, prefix, T,
                                            group=q.shape[1] // KVH,
                                            return_lse=True)
    diff = (out.float() - want_out.cuda().float()).norm(dim=-1)
    rel = (diff / want_out.cuda().float().norm(dim=-1).clamp_min(1e-6)).max()
    assert rel.item() < 9e-3
    assert (lse - want_lse.cuda()).abs().max().item() < 5e-3
    assert out.shape == (T, q.shape[1], HEAD_DIM) and lse.shape == (T, q.shape[1])


def test_supplied_out_buffer_is_filled_and_returned():
    T, N, prefix = 37, 65, 28
    q, k, v = _inputs(T, N, seed=5)
    idx = torch.arange(N, device="cuda")
    buffer = torch.zeros_like(q)
    returned = ops.prefill_extend(q, k, v, idx, prefix, out=buffer)
    assert returned.data_ptr() == buffer.data_ptr()
    fresh = ops.prefill_extend(q, k, v, idx, prefix)
    assert torch.equal(returned, fresh)


def test_supplied_lse_buffer_is_filled_and_returned():
    """``lse_out`` is the LSE twin of ``out`` and behaves like it.

    It exists because contract §5 requires the DESTINATION to be caller-owned
    and address-stable across graph replay, and an internally allocated result
    tensor is neither. Same identity property as ``out``: the buffer passed in
    is the object returned, and it holds what a fresh call would have produced.
    """
    T, N, prefix = 37, 65, 28
    q, k, v = _inputs(T, N, seed=7)
    idx = torch.arange(N, device="cuda")
    buffer = torch.zeros((T, q.shape[1]), dtype=torch.float32, device="cuda")
    out, lse = ops.prefill_extend(q, k, v, idx, prefix, return_lse=True,
                                  lse_out=buffer)
    assert lse.data_ptr() == buffer.data_ptr()
    _, fresh = ops.prefill_extend(q, k, v, idx, prefix, return_lse=True)
    assert torch.equal(lse, fresh)
    assert out.shape == (T, q.shape[1], HEAD_DIM)


@pytest.mark.parametrize("make,field", [
    (lambda T, H: torch.zeros((T, H), dtype=torch.float16, device="cuda"), "lse_out"),
    (lambda T, H: torch.zeros((T + 1, H), dtype=torch.float32, device="cuda"), "lse_out"),
    (lambda T, H: torch.zeros((H, T), dtype=torch.float32, device="cuda").T, "lse_out"),
    (lambda T, H: torch.zeros((T, H), dtype=torch.float32, device="cpu"), "lse_out"),
    (lambda T, H: "not a tensor", "lse_out"),
])
def test_bad_lse_out_is_a_typed_capability_error(make, field):
    """One refusal per way of getting it wrong, as the rest of the rejection
    matrix does: a mis-shaped destination must be a ``CapabilityError`` raised
    before any device work, never a ``RuntimeError`` from inside the copy."""
    from attn_kernel_lab.capability import CapabilityError

    T, N, prefix = 32, 64, 32
    q, k, v = _inputs(T, N, seed=8)
    idx = torch.arange(N, device="cuda")
    with pytest.raises(CapabilityError) as excinfo:
        ops.prefill_extend(q, k, v, idx, prefix, return_lse=True,
                           lse_out=make(T, q.shape[1]))
    assert excinfo.value.field == field


def test_lse_out_without_return_lse_is_refused():
    """A destination for an output that will not be produced is a caller
    mistake, and silently ignoring it would leave stale values in a buffer the
    caller is about to read."""
    from attn_kernel_lab.capability import CapabilityError

    T, N, prefix = 32, 64, 32
    q, k, v = _inputs(T, N, seed=9)
    idx = torch.arange(N, device="cuda")
    buffer = torch.zeros((T, q.shape[1]), dtype=torch.float32, device="cuda")
    with pytest.raises(CapabilityError, match="return_lse=False"):
        ops.prefill_extend(q, k, v, idx, prefix, lse_out=buffer)


def test_output_only_call_is_bitwise_stable_across_workspace_reuse():
    """Long-after-short through the PUBLIC surface: the reused workspace must
    not leak the longer request into the shorter one."""
    # The CANONICAL class identity: ops validates isinstance against
    # attn_kernel_lab.quant, and top-level `quant` (the legacy sys.path import
    # style) is a DIFFERENT module object with a different class. Public-surface
    # callers must use the package import.
    from attn_kernel_lab.quant import FP8PrefillWorkspace

    ws = FP8PrefillWorkspace(torch.device("cuda"))
    q1, k1, v1 = _inputs(32, 64, seed=11)
    idx1 = torch.arange(64, device="cuda")
    first = ops.prefill_extend(q1, k1, v1, idx1, 32, workspace=ws).clone()
    qL, kL, vL = _inputs(128, 448, seed=12)
    ops.prefill_extend(qL, kL, vL, torch.arange(448, device="cuda"), 320, workspace=ws)
    again = ops.prefill_extend(q1, k1, v1, idx1, 32, workspace=ws)
    assert torch.equal(first.view(torch.uint16), again.view(torch.uint16))
