#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""TF/s regression bench for the fp8 prefill attention kernel.

Standalone (NOT pytest). Runs on a rented GPU, next to a live serving process.

    python3 kernel_bench.py run --out before.json --label baseline
    # ... make the scheduling change, rebuild ...
    python3 kernel_bench.py run --out after.json  --label swizzle
    python3 kernel_bench.py compare before.json after.json     # exit 1 on regression

`compare` is the gate: any case whose kernel-side median time got worse by
more than --threshold (default 2%) fails the run.  Quant-side timings are
reported and gated separately -- the perf phase targets the kernel, but a
gather+quantize regression is its own bug and must not hide inside a kernel
win.

===========================================================================
WHAT IT MEASURES, AND HOW
===========================================================================
Two independent timings per case:

  kernel : CUDA events around the ``ext.fp8_prefill_attn`` launch alone,
           3 warmup + 10 timed iterations.  This is the number the
           performance phase is optimizing.  Reported as median ms and
           TF/s; min is reported too (the least noisy estimator when the
           GPU is shared).
  quant  : wall clock, synchronized, around gather_quantize_kv +
           quantize_q -- the whole Python-side pipeline for the case.
           1 warmup + 3 timed iterations (it is far more expensive per
           iteration and far less noisy).  A regression here does not
           touch TF/s but does touch end-to-end prefill time.

FLOP accounting (the repo convention, bottom-right causal, QK + PV):

    flops = 4 * T * (prefix + (T + 1) / 2) * HD * H

i.e. 2 flops per multiply-add, twice (QK and PV), over HD channels, for
each of the sum_{r<T} (prefix + r + 1) = T*(prefix + (T+1)/2) visible
(query, key) pairs, for each of H query heads.

Calibration (real rented-GPU measurements, for sanity-bounding only -- these are
NOT assertions and they must never be used to massage a number):
  * fp8-PV deployment shape   ~446 TF/s average over the chunk schedule
  * bf16-PV class             ~260-300 TF/s
  * single head, 32k x 446k   ~390-400 TF/s
  * known cliffs: 1 CTA/SM at 72.7 KB smem; +16B padded rows (the swizzle
    target)

THE PV AXIS IS ALSO THE CTA TILE-SHAPE AXIS.  The launcher takes the wide
tile (BM=128) only when every head is fp8-PV -- the bf16-PV V staging buffer
does not fit beside a 128-row Q tile in SM120's ~99 KB opt-in -- so
deploy_fp8 measures the wide path while deploy_bf16 and deploy_mixed4
measure the narrow one.  That is the whole point of keeping the mixed case:
the per-head PV dial's cost is a NARROW-tile cost, and a wide-tile win does
not transfer to it.

Runtime: ~576 TFLOP per timed iteration for one PV mode's deployment
schedule, x3 PV modes, x13 iterations (3 warmup + 10 timed) -> ~60-90 s of
kernel time at the calibrated rates, plus JIT (cached after the first
build), pool setup and ~4 quant iterations per case.  Budget ~3 minutes.

===========================================================================
MEMORY -- THIS RUNS BESIDE A SERVING PROCESS
===========================================================================
Budget assumption: ~3-4 GiB free on the GPU.  Every case carries an
estimated peak allocation and is SKIPPED (loudly, and recorded as skipped
in the json) if it exceeds --budget-gib (default 3.5).  Components, at the
deepest deployment case (N=446464, KVH=4, HD=256, H=24, T=20480/32768):

    K/V pools (bf16)   2 * POOL_ROWS * KVH * 256 * 2      512 MiB @ 131072
    q       (bf16)     T * H * 256 * 2                    252 MiB
    q8      (uint8)    H * Mpad * 256                     120 MiB
    o       (bf16)     H * Mpad * 256 * 2                 240 MiB
    k8      (uint8)    KVH * Npad * 256                   436 MiB
    vt8     (uint8)    KVH * Npad * 256   (fp8-PV heads)  436 MiB
    vb16    (bf16)     KVH * Npad * 256 * 2 (bf16-PV)     872 MiB
    transient fp32     SLAB(32768) * KVH * 256 * 4 * ~3   384 MiB

Estimator output at the defaults (pool_rows=131072), worst case per mode:

    deploy_fp8_p262144    2.61 GiB      deploy_bf16_p425984   3.01 GiB
    deploy_fp8_p425984    2.54 GiB      deploy_mixed4_p425984 3.48 GiB

so everything fits a 3.5 GiB budget, with the mixed deepest case at the
edge -- it is the first to skip on a busy GPU.  ``--budget-gib 6`` runs
everything on an idle one; a workspace is allocated and freed per case, so
the peak is per case, not cumulative.

POOL CAVEAT: the K/V pool is a fixed POOL_ROWS-row physical buffer (default
131072) indexed through a fixed random permutation, wrapping when N >
POOL_ROWS.  A full N-row pool at N=446464 would cost 1.75 GiB on its own
and blow the budget.  Consequence: at the deepest prefixes the gather
re-reads pool rows, so the QUANT-side timing there is mildly optimistic.
The KERNEL-side timing -- the number this bench exists for -- is unaffected:
the kernel reads only the full-size quantized workspace.  Use --pool-rows
to change it; a compare across two runs is only valid at the same value
(compare warns if they differ).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import platform
import statistics
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

HEAD_DIM = 256
BLK = 64  # kv quantization / K-tile granularity (quant.py BLK)
MPAD_GRAN = 128  # Q row padding granularity (quant.py MPAD == kernel BM)
_HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(_HERE, "..", "sglang_files/python/sglang/srt/layers/attention/fp8_prefill")
sys.path.insert(0, os.path.abspath(PKG))

BENCH_SCHEMA = 1
_OOM = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)


# =========================================================================
# extension + env
# =========================================================================


def load_ext(verbose=False):
    from torch.utils.cpp_extension import load

    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}a" if (major, minor) >= (9, 0) else f"{major}{minor}"
    # A/B support: FP8PA_EXTRA_CUDA_FLAGS adds -D switches (e.g. the kernel's
    # FP8PA_DISABLE_* toggles); FP8PA_BUILD_TAG suffixes the JIT cache name so
    # two flag sets never share a build directory.
    extra = os.environ.get("FP8PA_EXTRA_CUDA_FLAGS", "").split()
    tag = os.environ.get("FP8PA_BUILD_TAG", "")
    return load(
        name="sgl_fp8_prefill_attn_bench" + tag,
        sources=[os.path.join(PKG, "csrc", "fp8_prefill_attn.cu")],
        extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}"] + extra,
        verbose=verbose,
    )


def pin_numerics():
    """Same pins as the golden harness: keep the Hadamard GEMM out of TF32 so
    quant-side timings are comparable run to run (and so running this bench
    never leaves the process in a different precision mode than the tests)."""
    for obj, attr in (
        (torch.backends.cuda.matmul, "allow_tf32"),
        (torch.backends.cudnn, "allow_tf32"),
    ):
        if hasattr(obj, attr):
            setattr(obj, attr, False)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")


def env_fingerprint():
    p = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device_name": p.name,
        "capability": f"{p.major}.{p.minor}",
        "sm_count": p.multi_processor_count,
        "total_mem_gib": round(total / 2**30, 2),
        "free_mem_gib_at_start": round(free / 2**30, 2),
        "python": platform.python_version(),
        # Deliberately a constant, not platform.node(). The hostname of a rented
        # container or a workstation is an identity/correlation fingerprint and
        # carries no evidential value; the device fields above are the fingerprint
        # that matters.
        "host": "redacted",
    }


# =========================================================================
# cases
# =========================================================================


def ceil_to(x, m):
    return (x + m - 1) // m * m


def flops(T, prefix, H, hd=HEAD_DIM):
    """4 * T * (prefix + (T+1)/2) * HD * H -- the repo's convention."""
    return 4.0 * T * (prefix + (T + 1) / 2.0) * hd * H


# The deployment chunked-prefill schedule: 14 chunks of 32768 over a 446464
# token context; the 4 sampled points span the whole arithmetic-intensity
# range (the last one is the real ragged tail chunk, T=20480).
DEPLOY_SCHEDULE = (
    # (prefix,  T)
    (0, 32768),
    (131072, 32768),
    (262144, 32768),
    (425984, 20480),  # 13*32768 prefix + 20480 = 446464 = full context
)

# 4 protected heads, one per kv group at H=24/KVH=4/grp=6: the shape of the
# per-head PV dial the numerics program landed on, and the mask that forces
# BOTH PV code paths through every kv head's data.
MIXED_BF16_HEADS = (0, 6, 12, 18)

# Soft calibration bands from real rented-GPU measurements. Printed as a note when a
# measurement falls outside; NEVER a gate, NEVER a target to tune toward.
BAND_DEPLOY_FP8 = (350.0, 720.0)
BAND_DEPLOY_BF16 = (170.0, 420.0)
BAND_SINGLE_HEAD = (300.0, 620.0)


def build_cases():
    cases = []
    for pv, tag in (("fp8", "fp8"), ("bf16", "bf16"), ("mixed", "mixed4")):
        for prefix, T in DEPLOY_SCHEDULE:
            band = None
            if prefix >= 262144:
                band = (
                    BAND_DEPLOY_FP8 if pv == "fp8" else (BAND_DEPLOY_BF16 if pv == "bf16" else None)
                )
            cases.append(
                dict(
                    name=f"deploy_{tag}_p{prefix}",
                    group=f"deploy_{tag}",
                    T=T,
                    prefix=prefix,
                    N=prefix + T,
                    H=24,
                    KVH=4,
                    pv=pv,
                    band=band,
                    seed=9000 + prefix // 1024,
                )
            )
    # single head at the deployment depth: the 388-401 TF/s calibration point
    cases.append(
        dict(
            name="single_head_fp8_32kx446k",
            group="single_head",
            T=32768,
            prefix=413696,
            N=446464,
            H=1,
            KVH=1,
            pv="fp8",
            band=BAND_SINGLE_HEAD,
            seed=9500,
        )
    )
    # small-shape latency: launch overhead + a short tile loop dominate, so
    # this is where an occupancy / smem-size change shows up as a latency
    # regression even when the big shapes look fine.  Timed with inner=32
    # launches per event pair -- one launch is shorter than the CPU launch
    # path and the events would otherwise time GPU idle.
    for pv in ("fp8", "bf16"):
        cases.append(
            dict(
                name=f"small_latency_{pv}",
                group="small_latency",
                T=512,
                prefix=0,
                N=512,
                H=24,
                KVH=4,
                pv=pv,
                band=None,
                seed=9600,
                inner=32,
            )
        )
    for c in cases:
        c["flops"] = flops(c["T"], c["prefix"], c["H"])
        c.setdefault("inner", 1)
    return cases


def estimate_bytes(case, pool_rows, slab=32768):
    """Rough peak device allocation for a case, in bytes.  Deliberately
    generous: better a skipped case than an OOM next to a serving process."""
    T, N, H, KVH = case["T"], case["N"], case["H"], case["KVH"]
    npad, mpad = ceil_to(N, BLK), ceil_to(T, MPAD_GRAN)
    pv = case["pv"]
    need_vt8 = pv in ("fp8", "mixed")
    need_vb16 = pv in ("bf16", "mixed")
    b = 0
    b += 2 * min(pool_rows, ceil_to(N, BLK)) * KVH * HEAD_DIM * 2  # K/V pools
    b += T * H * HEAD_DIM * 2  # q bf16
    b += H * mpad * HEAD_DIM  # q8
    b += H * mpad * HEAD_DIM * 2  # o
    b += KVH * npad * HEAD_DIM  # k8
    if need_vt8:
        b += KVH * npad * HEAD_DIM  # vt8
    if need_vb16:
        b += KVH * npad * HEAD_DIM * 2  # vb16
    # transient fp32 slabs inside gather_quantize_kv (index_select -> f32 ->
    # centered -> hadamard product): ~3 live copies of one slab
    b += 3 * min(slab, npad) * KVH * HEAD_DIM * 4
    return int(b * 1.10)  # 10% slack


# =========================================================================
# data
# =========================================================================


def _rand_bf16(shape, g, device):
    """Uniform [-1,1] bf16 on `device`, generated on the CPU.

    Seeded CPU generation means two runs of this script bench IDENTICAL
    bytes -- quantization work (and therefore the quant-side timing) is
    data-dependent, so this is what makes A-vs-B meaningful.  The cast to
    bf16 happens on the CPU on purpose: `.to(device, bfloat16)` in one step
    can stage an fp32 copy on the GPU, and at these sizes that fp32 copy is
    half a gigabyte the memory estimate does not account for.
    """
    t = torch.empty(shape, dtype=torch.float32)
    t.uniform_(-1.0, 1.0, generator=g)
    return t.to(torch.bfloat16).to(device)


def make_pool(pool_rows, KVH, seed, device="cuda"):
    """A page_size-1 style pool plus a fixed scattering permutation."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    kp = _rand_bf16((pool_rows, KVH, HEAD_DIM), g, device)
    vp = _rand_bf16((pool_rows, KVH, HEAD_DIM), g, device)
    perm = torch.randperm(pool_rows, generator=g)
    return kp, vp, perm.to(device)


def make_idx(perm, N):
    """N pool rows in position order.  Wraps when N > pool_rows (see the POOL
    CAVEAT in the module docstring)."""
    P = perm.numel()
    if N <= P:
        return perm[:N].contiguous()
    return perm[torch.arange(N, device=perm.device) % P].contiguous()


def make_q(T, H, seed, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    return _rand_bf16((T, H, HEAD_DIM), g, device)


def pv_mask(H, pv, device="cuda"):
    m = torch.ones(H, dtype=torch.uint8, device=device)
    if pv == "bf16":
        m.zero_()
    elif pv == "mixed":
        for h in MIXED_BF16_HEADS:
            if h < H:
                m[h] = 0
    return m


# =========================================================================
# timing
# =========================================================================


def _stats(ms):
    ms = sorted(ms)
    return {
        "median_ms": statistics.median(ms),
        "mean_ms": statistics.fmean(ms),
        "min_ms": ms[0],
        "max_ms": ms[-1],
        "std_ms": statistics.pstdev(ms) if len(ms) > 1 else 0.0,
        "iters": len(ms),
    }


def time_events(fn, warmup=3, iters=10, inner=1):
    """CUDA-event timing, ``inner`` launches per event pair.

    ``inner > 1`` is for shapes whose kernel is shorter than the CPU launch
    path: with one launch per pair the events would time GPU idle waiting on
    the CPU rather than the kernel.  The reported ms is per launch.
    """
    for _ in range(warmup):
        for _ in range(inner):
            fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        for _ in range(inner):
            fn()
        ends[i].record()
    torch.cuda.synchronize()
    return _stats([s.elapsed_time(e) / inner for s, e in zip(starts, ends)])


def time_wall(fn, warmup=1, iters=3):
    """Synchronized wall clock -- for the Python-side quant pipeline, whose
    cost is CPU launch overhead + many kernels, not one launch."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t0) * 1e3)
    return _stats(out)


# =========================================================================
# run
# =========================================================================


def run_case(ext, q_mod, case, pool_rows, args):
    T, N, H, KVH, pv = case["T"], case["N"], case["H"], case["KVH"], case["pv"]
    dev = torch.device("cuda")
    ws = q_mod.FP8PrefillWorkspace(dev)
    kp, vp, perm = make_pool(min(pool_rows, ceil_to(N, BLK)), KVH, case["seed"])
    idx = make_idx(perm, N)
    q = make_q(T, H, case["seed"])
    mask = pv_mask(H, pv)
    any_pv8 = bool(mask.max().item())
    all_pv8 = bool(mask.min().item())
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    def quant_step():
        kv = q_mod.gather_quantize_kv(
            ws,
            kp,
            vp,
            idx,
            need_vt8=any_pv8,
            need_vb16=not all_pv8,
            center_k=True,
            qk_i8=True,
            rotate=True,
        )
        q8, qscale, mpad = q_mod.quantize_q(ws, q, sm_scale, qk_i8=True, rotate=True)
        return kv, q8, qscale, mpad

    quant = time_wall(quant_step, warmup=1, iters=args.quant_iters)
    kv, q8, qscale, mpad = quant_step()
    o = ws.get("o", (H, mpad, HEAD_DIM), torch.bfloat16)

    def kernel_step():
        ext.fp8_prefill_attn(
            q8,
            kv["k8"],
            kv["vt8"],
            kv["vb16"],
            o,
            qscale,
            kv["kscale"],
            kv["vscale"],
            kv["vlog2r"],
            kv["vinvr"],
            kv["vmean"],
            mask,
            kv["n"],
            case["prefix"],
            any_pv8,
            all_pv8,
            True,
        )

    kernel = time_events(
        kernel_step, warmup=args.warmup, iters=args.iters, inner=case.get("inner", 1)
    )
    kernel["tflops_median"] = case["flops"] / (kernel["median_ms"] * 1e-3) / 1e12
    kernel["tflops_min_time"] = case["flops"] / (kernel["min_ms"] * 1e-3) / 1e12
    peak = torch.cuda.max_memory_allocated()

    del ws, kp, vp, perm, idx, q, kv, q8, qscale, o, mask
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return {"kernel": kernel, "quant": quant, "peak_alloc_gib": peak / 2**30}


def cmd_run(args):
    pin_numerics()
    if not torch.cuda.is_available():
        print("ERROR: no CUDA device. This bench must run on the rented target.")
        return 2
    import quant as q_mod  # noqa: E402  (the package's quant.py, standalone)

    env = env_fingerprint()
    print(f"device : {env['device_name']}  sm{env['capability']}  {env['sm_count']} SMs")
    print(f"torch  : {env['torch']} / CUDA {env['cuda']}")
    print(
        f"memory : {env['free_mem_gib_at_start']:.2f} GiB free of "
        f"{env['total_mem_gib']:.2f} GiB   budget={args.budget_gib:.2f} GiB"
    )
    print(f"pool   : {args.pool_rows} rows (see POOL CAVEAT in the header)")
    print("compiling kernel (JIT, ~1 min cold) ...", flush=True)
    ext = load_ext(verbose=args.verbose)
    torch.cuda.reset_peak_memory_stats()

    cases = build_cases()
    if args.cases:
        cases = [c for c in cases if args.cases in c["name"]]
        if not cases:
            print(f"ERROR: --cases {args.cases!r} matched nothing.")
            return 2

    results = []
    print()
    hdr = (
        f"{'case':<28s} {'T':>6s} {'prefix':>7s} {'TFLOP':>7s} "
        f"{'kern ms':>9s} {'TF/s':>7s} {'quant ms':>9s} {'GiB':>5s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for c in cases:
        est = estimate_bytes(c, args.pool_rows)
        if est > args.budget_gib * 2**30:
            print(
                f"{c['name']:<28s} SKIPPED: estimated peak "
                f"{est / 2**30:.2f} GiB > budget {args.budget_gib:.2f} GiB "
                f"(raise --budget-gib on an idle GPU)"
            )
            results.append(
                {
                    **{
                        k: c[k]
                        for k in ("name", "group", "T", "N", "prefix", "H", "KVH", "pv", "flops")
                    },
                    "skipped": f"over budget ({est / 2**30:.2f} GiB)",
                }
            )
            continue
        try:
            r = run_case(ext, q_mod, c, args.pool_rows, args)
        except _OOM as e:  # pragma: no cover (rented target only)
            torch.cuda.empty_cache()
            print(f"{c['name']:<28s} SKIPPED: OOM ({e.__class__.__name__})")
            results.append(
                {
                    **{
                        k: c[k]
                        for k in ("name", "group", "T", "N", "prefix", "H", "KVH", "pv", "flops")
                    },
                    "skipped": "OOM",
                }
            )
            continue
        row = {
            **{k: c[k] for k in ("name", "group", "T", "N", "prefix", "H", "KVH", "pv", "flops")},
            "skipped": None,
            **r,
        }
        results.append(row)
        note = ""
        if c["band"] is not None:
            lo, hi = c["band"]
            tf = r["kernel"]["tflops_median"]
            if not (lo <= tf <= hi):
                note = f"   <-- calibration note: outside {lo:.0f}-{hi:.0f} TF/s"
        print(
            f"{c['name']:<28s} {c['T']:>6d} {c['prefix']:>7d} "
            f"{c['flops'] / 1e12:>7.1f} "
            f"{r['kernel']['median_ms']:>9.3f} "
            f"{r['kernel']['tflops_median']:>7.1f} "
            f"{r['quant']['median_ms']:>9.3f} "
            f"{r['peak_alloc_gib']:>5.2f}{note}",
            flush=True,
        )

    doc = {
        "schema": BENCH_SCHEMA,
        "label": args.label,
        "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "argv": sys.argv[1:],
        "config": {
            "pool_rows": args.pool_rows,
            "warmup": args.warmup,
            "iters": args.iters,
            "quant_iters": args.quant_iters,
            "budget_gib": args.budget_gib,
        },
        "env": env,
        "cases": results,
    }
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    ran = sum(1 for r in results if not r["skipped"])
    print(f"\nwrote {args.out}  ({ran}/{len(results)} cases measured)")
    if ran < len(results):
        print(
            "NOTE: skipped cases are recorded as skipped; `compare` will "
            "report them as MISSING rather than as a regression."
        )
    return 0


# =========================================================================
# compare
# =========================================================================


def _by_name(doc):
    return {c["name"]: c for c in doc["cases"]}


def _pct(a, b):
    """percent change from a to b (positive = b is larger)."""
    if a == 0:
        return float("nan")
    return (b - a) / a * 100.0


def cmd_compare(args):
    with open(args.a) as f:
        A = json.load(f)
    with open(args.b) as f:
        B = json.load(f)
    return report_compare(
        A, B, args.threshold, args.kernel_only, names=(args.a, args.b), out=sys.stdout
    )


def report_compare(A, B, threshold=2.0, kernel_only=False, names=("A", "B"), out=sys.stdout):
    """Pure-python comparison + verdict.  Returns the process exit code
    (0 pass, 1 regression). No torch, no device -- unit-testable off-target."""
    a_cases, b_cases = _by_name(A), _by_name(B)
    la = A.get("label") or os.path.basename(names[0])
    lb = B.get("label") or os.path.basename(names[1])

    def w(s=""):
        print(s, file=out)

    w(f"A = {la:<24s} {A.get('utc', '?')}   {A.get('env', {}).get('device_name', '?')}")
    w(f"B = {lb:<24s} {B.get('utc', '?')}   {B.get('env', {}).get('device_name', '?')}")
    for k in ("device_name", "torch", "cuda"):
        va, vb = A.get("env", {}).get(k), B.get("env", {}).get(k)
        if va != vb:
            w(
                f"WARNING: env.{k} differs ({va!r} vs {vb!r}) -- the comparison "
                f"is not apples to apples."
            )
    ca, cb = A.get("config", {}), B.get("config", {})
    for k in ("pool_rows", "iters", "warmup"):
        if ca.get(k) != cb.get(k):
            w(
                f"WARNING: config.{k} differs ({ca.get(k)} vs {cb.get(k)}) -- "
                f"timings are not comparable."
            )
    w()

    hdr = (
        f"{'case':<28s} "
        f"{'kern ms A':>10s} {'kern ms B':>10s} {'d%':>7s}  "
        f"{'TF/s A':>7s} {'TF/s B':>7s}  "
        f"{'quant A':>8s} {'quant B':>8s} {'d%':>7s}  {'':<4s}"
    )
    w(hdr)
    w("-" * len(hdr))

    kernel_regressions, quant_regressions, missing, improvements = [], [], [], []
    for name in sorted(set(a_cases) | set(b_cases)):
        ra, rb = a_cases.get(name), b_cases.get(name)
        if ra is None or rb is None or ra.get("skipped") or rb.get("skipped"):
            why = "absent" if ra is None or rb is None else "skipped"
            missing.append((name, why))
            w(
                f"{name:<28s} {'--':>10s} {'--':>10s} {'--':>7s}  "
                f"{'--':>7s} {'--':>7s}  {'--':>8s} {'--':>8s} {'--':>7s}  "
                f"MISSING({why})"
            )
            continue
        ka, kb = ra["kernel"], rb["kernel"]
        qa, qb = ra["quant"], rb["quant"]
        dk = _pct(ka["median_ms"], kb["median_ms"])
        dq = _pct(qa["median_ms"], qb["median_ms"])
        flag = "ok"
        if dk > threshold:
            kernel_regressions.append((name, dk))
            flag = "SLOW"
        elif dk < -threshold:
            improvements.append((name, dk))
            flag = "fast"
        if dq > threshold:
            quant_regressions.append((name, dq))
            flag = "SLOW-Q" if flag == "ok" else flag + "+Q"
        w(
            f"{name:<28s} "
            f"{ka['median_ms']:>10.3f} {kb['median_ms']:>10.3f} {dk:>+7.2f}  "
            f"{ka['tflops_median']:>7.1f} {kb['tflops_median']:>7.1f}  "
            f"{qa['median_ms']:>8.3f} {qb['median_ms']:>8.3f} {dq:>+7.2f}  "
            f"{flag:<4s}"
        )

    w()
    for name, group in (("kernel", kernel_regressions), ("quant", quant_regressions)):
        if group:
            w(f"{name} regressions (> {threshold:.1f}% slower):")
            for n, d in sorted(group, key=lambda x: -x[1]):
                w(f"    {n:<28s} {d:+.2f}%")
    if improvements:
        w(f"kernel improvements (> {threshold:.1f}% faster):")
        for n, d in sorted(improvements, key=lambda x: x[1]):
            w(f"    {n:<28s} {d:+.2f}%")
    if missing:
        w(f"not compared ({len(missing)}): " + ", ".join(f"{n}[{why}]" for n, why in missing))

    fail = bool(kernel_regressions) or (bool(quant_regressions) and not kernel_only)
    w()
    if kernel_regressions and quant_regressions:
        w(
            f"VERDICT: REGRESSION -- {len(kernel_regressions)} kernel case(s) and "
            f"{len(quant_regressions)} quant case(s) slower by > {threshold:.1f}%."
        )
    elif kernel_regressions:
        w(
            f"VERDICT: REGRESSION -- {len(kernel_regressions)} kernel case(s) "
            f"slower by > {threshold:.1f}%."
        )
    elif quant_regressions:
        w(
            f"VERDICT: {'REGRESSION' if not kernel_only else 'QUANT REGRESSION (not gated)'}"
            f" -- {len(quant_regressions)} quant case(s) slower by > {threshold:.1f}%; "
            f"kernel is clean."
        )
    else:
        w(f"VERDICT: PASS -- no case slower by > {threshold:.1f}% ({len(improvements)} improved).")
    if missing and fail is False:
        w(f"         (note: {len(missing)} case(s) were not compared.)")
    w(
        "REMINDER: a speed result is only valid if test_golden_bitexact.py is "
        "green on the same build."
    )
    return 1 if fail else 0


# =========================================================================
# cli
# =========================================================================


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="measure and write a results json")
    r.add_argument("--out", required=True)
    r.add_argument("--label", default=None, help="name for this build/config, shown by compare")
    r.add_argument("--cases", default=None, help="substring filter on case names")
    r.add_argument("--warmup", type=int, default=3)
    r.add_argument("--iters", type=int, default=10)
    r.add_argument("--quant-iters", type=int, default=3)
    r.add_argument(
        "--pool-rows",
        type=int,
        default=131072,
        help="physical K/V pool rows (memory vs gather realism; see POOL CAVEAT)",
    )
    r.add_argument(
        "--budget-gib", type=float, default=3.5, help="skip cases whose estimated peak exceeds this"
    )
    r.add_argument("--verbose", action="store_true")
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("compare", help="A vs B, exit 1 on regression")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument(
        "--threshold", type=float, default=2.0, help="percent slower that counts as a regression"
    )
    c.add_argument(
        "--kernel-only",
        action="store_true",
        help="gate on kernel timings only (still reports quant)",
    )
    c.set_defaults(func=cmd_compare)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
