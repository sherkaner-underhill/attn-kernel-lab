# SPDX-License-Identifier: Apache-2.0
"""FROST DSL SM120 SDPA at D256/24:4/bottom-right — probe v2 (the real surface).

The pygraph layer silently ignored diagonal alignment (probe v1's 2580% error
and impossible flat timing caught it). This drives cudnn.sdpa.fwd.SdpaFwdDslSm120
directly: is_causal + causal_bottom_right, BSHD bf16, numerics gate BEFORE any
timing is trusted.
"""

import math, torch
from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm120

DEV = torch.device("cuda")


def run(SQ, SK, time_iters=10, check=False, lse=False):
    torch.manual_seed(SQ + SK)
    B, Hq, Hkv, D = 1, 24, 4, 256
    q = (torch.randn(B, Hq, SQ, D, device=DEV) * 0.5).to(torch.bfloat16)  # BHSD
    k = (torch.randn(B, Hkv, SK, D, device=DEV) * 0.5).to(torch.bfloat16)
    v = (torch.randn(B, Hkv, SK, D, device=DEV) * 0.5).to(torch.bfloat16)
    o = torch.empty_like(q)
    lse_t = torch.empty(B, Hq, SQ, device=DEV, dtype=torch.float32) if lse else None

    api = SdpaFwdDslSm120(
        sample_q=q,
        sample_k=k,
        sample_v=v,
        sample_o=o,
        sample_lse=lse_t,
        is_causal=True,
        causal_bottom_right=True,
        scale_softmax=1.0 / math.sqrt(D),
    )
    api.check_support()
    api.compile()
    ws_bytes = api.scratch_workspace_bytes()
    ws = torch.empty(ws_bytes, device=DEV, dtype=torch.uint8) if ws_bytes else None
    kw = dict(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o)
    if lse_t is not None:
        kw["lse_tensor"] = lse_t
    if ws is not None:
        kw["workspace"] = ws
    api.execute(**kw)
    torch.cuda.synchronize()

    if check:
        grp = Hq // Hkv
        qh = q.float()
        kh = k.repeat_interleave(grp, 1).float()
        vh = v.repeat_interleave(grp, 1).float()
        s = (qh @ kh.transpose(-1, -2)) / math.sqrt(D)
        cols = torch.arange(SK, device=DEV)
        rows = torch.arange(SQ, device=DEV)
        s.masked_fill_(cols[None, :] > rows[:, None] + (SK - SQ), float("-inf"))
        ref = torch.softmax(s, -1) @ vh
        got = o.float()
        rel = ((got - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-6)).max().item()
        print(
            f"  numerics ({SQ}x{SK}): max row-rel {rel:.3e}  "
            + ("OK" if rel < 5e-2 else "*** WRONG MASK/VALUES ***")
        )
    for _ in range(3):
        api.execute(**kw)
    torch.cuda.synchronize()
    times = []
    for _ in range(time_iters):
        t0 = torch.cuda.Event(True)
        t1 = torch.cuda.Event(True)
        t0.record()
        api.execute(**kw)
        t1.record()
        torch.cuda.synchronize()
        times.append(t0.elapsed_time(t1))
    times.sort()
    ms = times[len(times) // 2]
    flops = 4 * Hq * D * SQ * ((SK - SQ) + (SQ + 1) / 2)
    print(f"  {SQ:>6} x {SK:>6}: {ms:>9.2f} ms   {flops / ms / 1e9:>6.1f} TF/s")
    return ms


run(512, 1024, time_iters=5, check=True)
run(1024, 4096, time_iters=5, check=True)
print("real chunk shapes (BR causal, 24:4, D256, bf16, SM120 FROST DSL):")
tot = 0.0
for SQ, SK in [(32768, 32768), (32768, 98304), (32768, 229376), (32768, 425984), (20351, 446335)]:
    tot += run(SQ, SK)
print(f"(5-shape subtotal {tot:.1f} ms)")
print("CUDNN_DSL_PROBE_COMPLETE")
