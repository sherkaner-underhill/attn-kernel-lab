# SPDX-License-Identifier: Apache-2.0
"""Drive the FP8-QK port of the FROST SM120 template: numerics gate, then timing.

Per-tensor model: q8 = e4m3(q / sq), k8 = e4m3(k / sk) with per-TENSOR scalar
scales; the descales fold into the softmax scale EXACTLY:
    softmax_scale' = sm_scale * sq * sk
V is fp8 per-tensor too (dequantized to bf16 in-kernel); its descale folds into
a single post-kernel elementwise multiply of O (included in timing: it is part
of the operator).
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cudnn.frost.template_loader import load_template
from cudnn.sdpa.fwd.config_sm120 import TemplateParams

import cutlass

TPL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prefill_fp8qk_sm120.py")
DEV = torch.device("cuda")
E4M3 = torch.float8_e4m3fn


def build(SQ, SK, q_tile=128, kv_tile=128):
    params = TemplateParams(
        dtype_qkv=0,  # E4M3 -> FP8_QK path in the ported template
        is_causal=True,
        causal_bottom_right=True,
        q_tile=q_tile,
        kv_tile=kv_tile,
    )
    mod = load_template(TPL, params, tag="fp8qk_sm120_port")
    cc = torch.cuda.get_device_capability()
    return mod.compile(cc, b=1, qh=24, kh=4, sq=SQ, skv=SK, d=256)


def run(kernel, SQ, SK, seed, time_iters=0):
    torch.manual_seed(seed)
    B, Hq, Hkv, D = 1, 24, 4, 256
    q32 = torch.randn(B, SQ, Hq, D, device=DEV) * 0.5
    k32 = torch.randn(B, SK, Hkv, D, device=DEV) * 0.5
    v32 = torch.randn(B, SK, Hkv, D, device=DEV) * 0.5
    sq_ = (q32.abs().max() / 448.0).item()
    sk_ = (k32.abs().max() / 448.0).item()
    sv_ = (v32.abs().max() / 448.0).item()
    q8 = (q32 / sq_).to(E4M3)
    k8 = (k32 / sk_).to(E4M3)
    v8 = (v32 / sv_).to(E4M3)
    o = torch.empty(B, SQ, Hq, D, device=DEV, dtype=torch.bfloat16)
    lse = torch.empty(B, Hq, SQ, device=DEV, dtype=torch.float32)
    sinks = torch.zeros(Hq, device=DEV, dtype=torch.float32)
    sl_q = torch.zeros(1, device=DEV, dtype=torch.int32)
    sl_k = torch.zeros(1, device=DEV, dtype=torch.int32)
    scale_log2 = (1.0 / math.sqrt(D)) * sq_ * sk_ * math.log2(math.e)
    stream = torch.cuda.current_stream().cuda_stream

    def step():
        kernel(q8, k8, v8, o, lse, sinks, sl_q, sl_k,
               cutlass.Float32(scale_log2), stream)
        o.mul_(sv_)  # per-tensor V descale: part of the operator, timed

    step()
    torch.cuda.synchronize()

    # numerics vs an fp32 reference OF THE PER-TENSOR MODEL'S INPUTS is checked
    # by the caller; here return raw pieces.
    result = {"o": o.clone(), "q32": q32, "k32": k32, "v32": v32,
              "q8": q8, "k8": k8, "v8": v8, "scales": (sq_, sk_, sv_)}
    if time_iters:
        for _ in range(3):
            step()
        torch.cuda.synchronize()
        times = []
        for _ in range(time_iters):
            t0 = torch.cuda.Event(True); t1 = torch.cuda.Event(True)
            t0.record(); step(); t1.record(); torch.cuda.synchronize()
            times.append(t0.elapsed_time(t1))
        times.sort()
        result["ms"] = times[len(times) // 2]
    return result


def reference_fp32(q32, k32, v32, SQ, SK):
    grp = 24 // 4
    qh = q32.permute(0, 2, 1, 3).float()
    kh = k32.permute(0, 2, 1, 3).repeat_interleave(grp, 1).float()
    vh = v32.permute(0, 2, 1, 3).repeat_interleave(grp, 1).float()
    s = (qh @ kh.transpose(-1, -2)) / math.sqrt(256)
    cols = torch.arange(SK, device=DEV)
    rows = torch.arange(SQ, device=DEV)
    s.masked_fill_(cols[None, :] > rows[:, None] + (SK - SQ), float("-inf"))
    return (torch.softmax(s, -1) @ vh).permute(0, 2, 1, 3)


def main():
    print("== build 128x448 (small gate) ==")
    kern = build(128, 448)
    r = run(kern, 128, 448, seed=7)
    ref = reference_fp32(r["q32"], r["k32"], r["v32"], 128, 448)
    got = r["o"].float()
    rel = ((got - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-6)).max().item()
    print(f"  vs fp32 reference: max row-rel {rel:.3e} "
          f"({'OK for per-tensor fp8-QK' if rel < 8e-2 else '*** INVESTIGATE ***'})")

    print("== real chunk shapes ==")
    for SQ, SK in [(32768, 32768), (32768, 229376), (32768, 425984), (20351, 446335)]:
        kern = build(SQ, SK)
        r = run(kern, SQ, SK, seed=SQ + SK, time_iters=10)
        flops = 4 * 24 * 256 * SQ * ((SK - SQ) + (SQ + 1) / 2)
        print(f"  {SQ:>6} x {SK:>6}: {r['ms']:>9.2f} ms   {flops / r['ms'] / 1e9:>6.1f} TF/s")
    print("FP8QK_PORT_PROBE_COMPLETE")


if __name__ == "__main__":
    main()
