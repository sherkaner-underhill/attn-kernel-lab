# SPDX-License-Identifier: Apache-2.0
"""cuDNN FROST FP8 SDPA probe: D256 + 24:4 GQA + bottom-right causal at Q<K.

Answers the three pre-registered gotchas before any benchmark number is
trusted: (1) does the FP8 engine SELECT at our shape, (2) is the causal mask
bottom-right aligned for rectangular Q<K, (3) do the numerics agree with an
fp32 reference within fp8-plausible error.
"""

import math, sys, traceback
import torch, cudnn

DEV = torch.device("cuda")
E4M3 = torch.float8_e4m3fn


def build_and_run(B, Hq, Hkv, SQ, SK, D, causal, bottom_right, time_iters=0):
    torch.manual_seed(1234 + SQ + SK)
    q32 = torch.randn(B, Hq, SQ, D, device=DEV) * 0.5
    k32 = torch.randn(B, Hkv, SK, D, device=DEV) * 0.5
    v32 = torch.randn(B, Hkv, SK, D, device=DEV) * 0.5
    # per-tensor quantization (cuDNN's model): scale = amax/448
    sq_, sk_, sv_ = (t.abs().max() / 448.0 for t in (q32, k32, v32))
    q8 = (q32 / sq_).to(E4M3)
    k8 = (k32 / sk_).to(E4M3)
    v8 = (v32 / sv_).to(E4M3)
    one = lambda v: torch.full((1, 1, 1, 1), float(v), device=DEV, dtype=torch.float32)
    descale_q, descale_k, descale_v = one(sq_), one(sk_), one(sv_)
    descale_s, scale_s, scale_o = one(1.0), one(1.0), one(1.0)

    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.FP8_E4M3,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    tq = g.tensor_like(q8)
    tk = g.tensor_like(k8)
    tv = g.tensor_like(v8)
    tdq = g.tensor_like(descale_q)
    tdk = g.tensor_like(descale_k)
    tdv = g.tensor_like(descale_v)
    tds = g.tensor_like(descale_s)
    tss = g.tensor_like(scale_s)
    tso = g.tensor_like(scale_o)
    kwargs = dict(attn_scale=1.0 / math.sqrt(D), use_causal_mask=causal, is_inference=True)
    if causal and bottom_right:
        kwargs["diagonal_alignment"] = cudnn.diagonal_alignment.BOTTOM_RIGHT
    O, Stats, AmaxS, AmaxO = g.sdpa_fp8(
        tq, tk, tv, tdq, tdk, tdv, tds, tss, tso, name="probe", **kwargs
    )
    O.set_output(True).set_data_type(cudnn.data_type.FP8_E4M3)
    AmaxS.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    AmaxO.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    g.check_support()
    g.build_plans()

    o8 = torch.empty(B, Hq, SQ, D, device=DEV, dtype=E4M3)
    amax_s = torch.empty(1, 1, 1, 1, device=DEV)
    amax_o = torch.empty_like(amax_s)
    ws = torch.empty(g.get_workspace_size(), device=DEV, dtype=torch.uint8)
    pack = {
        tq: q8,
        tk: k8,
        tv: v8,
        tdq: descale_q,
        tdk: descale_k,
        tdv: descale_v,
        tds: descale_s,
        tss: scale_s,
        tso: scale_o,
        O: o8,
        AmaxS: amax_s,
        AmaxO: amax_o,
    }
    g.execute(pack, ws)
    torch.cuda.synchronize()

    # fp32 reference with EXPLICIT bottom-right mask
    grp = Hq // Hkv
    kr = k32.repeat_interleave(grp, dim=1)
    vr = v32.repeat_interleave(grp, dim=1)
    s = (q32.float() @ kr.transpose(-1, -2).float()) / math.sqrt(D)
    if causal:
        cols = torch.arange(SK, device=DEV)
        rows = torch.arange(SQ, device=DEV)
        off = (SK - SQ) if bottom_right else 0
        mask = cols[None, :] > (rows[:, None] + off)
        s.masked_fill_(mask, float("-inf"))
    ref = torch.softmax(s, dim=-1) @ vr.float()
    got = o8.float()  # scale_o = 1
    rel = ((got - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-6)).max().item()

    ms = None
    if time_iters:
        for _ in range(3):
            g.execute(pack, ws)
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(True)
        t1 = torch.cuda.Event(True)
        times = []
        for _ in range(time_iters):
            t0.record()
            g.execute(pack, ws)
            t1.record()
            torch.cuda.synchronize()
            times.append(t0.elapsed_time(t1))
        times.sort()
        ms = times[len(times) // 2]
    return rel, ms


CASES = [
    (
        "tiny sanity, TL causal, MHA",
        dict(B=1, Hq=4, Hkv=4, SQ=128, SK=128, D=256, causal=True, bottom_right=False),
    ),
    (
        "D256 + 24:4 GQA, no mask",
        dict(B=1, Hq=24, Hkv=4, SQ=256, SK=256, D=256, causal=False, bottom_right=False),
    ),
    (
        "D256 + 24:4 + BOTTOM_RIGHT, Q<K rect",
        dict(B=1, Hq=24, Hkv=4, SQ=128, SK=448, D=256, causal=True, bottom_right=True),
    ),
    (
        "chunk0 shape 32k x 32k",
        dict(
            B=1,
            Hq=24,
            Hkv=4,
            SQ=32768,
            SK=32768,
            D=256,
            causal=True,
            bottom_right=True,
            time_iters=10,
        ),
    ),
    (
        "chunk6 shape 32k x 229k",
        dict(
            B=1,
            Hq=24,
            Hkv=4,
            SQ=32768,
            SK=229376,
            D=256,
            causal=True,
            bottom_right=True,
            time_iters=10,
        ),
    ),
    (
        "chunk13 tail 20351 x 446335",
        dict(
            B=1,
            Hq=24,
            Hkv=4,
            SQ=20351,
            SK=446335,
            D=256,
            causal=True,
            bottom_right=True,
            time_iters=10,
        ),
    ),
]
for name, kw in CASES:
    try:
        rel, ms = build_and_run(**kw)
        flops = (
            4 * kw["Hq"] * kw["D"] * kw["SQ"] * ((kw["SK"] - kw["SQ"]) + (kw["SQ"] + 1) / 2)
            if kw["causal"]
            else 4 * kw["Hq"] * kw["D"] * kw["SQ"] * kw["SK"]
        )
        tf = f"  {flops / ms / 1e9:.1f} TF/s" if ms else ""
        print(f"PASS  {name}: max row-rel {rel:.3e}" + (f"  {ms:.2f} ms{tf}" if ms else ""))
    except Exception as e:
        print(f"FAIL  {name}: {type(e).__name__}: {str(e)[:300]}")
print("PROBE_COMPLETE")
