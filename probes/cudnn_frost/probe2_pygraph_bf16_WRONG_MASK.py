# SPDX-License-Identifier: Apache-2.0
"""cuDNN BF16 dense SDPA at D256/24:4/bottom-right — their strongest engine
that can enter the ring at our geometry. Pre-gathered inputs (their best case)."""

import math, torch, cudnn

DEV = torch.device("cuda")


def run(SQ, SK, time_iters=10, check=False):
    torch.manual_seed(SQ + SK)
    B, Hq, Hkv, D = 1, 24, 4, 256
    q = (torch.randn(B, Hq, SQ, D, device=DEV) * 0.5).to(torch.bfloat16)
    k = (torch.randn(B, Hkv, SK, D, device=DEV) * 0.5).to(torch.bfloat16)
    v = (torch.randn(B, Hkv, SK, D, device=DEV) * 0.5).to(torch.bfloat16)
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    tq, tk, tv = g.tensor_like(q), g.tensor_like(k), g.tensor_like(v)
    O, Stats = g.sdpa(
        tq,
        tk,
        tv,
        name="ctl",
        attn_scale=1.0 / math.sqrt(D),
        use_causal_mask=True,
        diagonal_alignment=cudnn.diagonal_alignment.BOTTOM_RIGHT,
        is_inference=True,
    )
    O.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    g.check_support()
    g.build_plans()
    o = torch.empty(B, Hq, SQ, D, device=DEV, dtype=torch.bfloat16)
    ws = torch.empty(g.get_workspace_size(), device=DEV, dtype=torch.uint8)
    pack = {tq: q, tk: k, tv: v, O: o}
    g.execute(pack, ws)
    torch.cuda.synchronize()
    if check:
        grp = Hq // Hkv
        s = (q.float() @ k.repeat_interleave(grp, 1).transpose(-1, -2).float()) / math.sqrt(D)
        cols = torch.arange(SK, device=DEV)
        rows = torch.arange(SQ, device=DEV)
        s.masked_fill_(cols[None, :] > rows[:, None] + (SK - SQ), float("-inf"))
        ref = torch.softmax(s, -1) @ v.repeat_interleave(grp, 1).float()
        rel = ((o.float() - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-6)).max().item()
        print(f"  numerics sanity ({SQ}x{SK}): max row-rel {rel:.3e}")
    for _ in range(3):
        g.execute(pack, ws)
    torch.cuda.synchronize()
    times = []
    for _ in range(time_iters):
        t0 = torch.cuda.Event(True)
        t1 = torch.cuda.Event(True)
        t0.record()
        g.execute(pack, ws)
        t1.record()
        torch.cuda.synchronize()
        times.append(t0.elapsed_time(t1))
    times.sort()
    ms = times[len(times) // 2]
    flops = 4 * Hq * D * SQ * ((SK - SQ) + (SQ + 1) / 2)
    print(f"  {SQ:>6} x {SK:>6}: {ms:>9.2f} ms   {flops / ms / 1e9:>6.1f} TF/s")
    return ms


run(512, 1024, time_iters=5, check=True)
print("real chunk shapes (bottom-right causal, 24:4, D256, bf16 dense):")
for SQ, SK in [(32768, 32768), (32768, 229376), (32768, 425984), (20351, 446335)]:
    run(SQ, SK)
print("CUDNN_BF16_PROBE_COMPLETE")
