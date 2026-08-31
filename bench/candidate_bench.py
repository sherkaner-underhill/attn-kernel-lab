#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Candidate-zero qualification bench: the protected schedule, measured honestly.

Successor to the historical ``legacy/kernel_bench.py`` bootstrap, fixing its
recorded debts for a DEDICATED qualification GPU (it assumed ~3.5 GiB beside a
serving process; a qualification target is idle):

  - **All 14 protected cases**, loaded from the hashed workload data
    (``workloads/generated/<profile>.cases.json``) -- never loop arithmetic.
    The cases file hash is recorded in the result.
  - **No pool wrap.** The physical K/V pool holds all N rows (1.7 GiB at 446k),
    scattered by a fixed permutation, so deep-prefix gather timing is real.
  - **Per-(chunk, layer) seeds.** ``--layers L`` replays each geometry L times
    with distinct data, so nothing exploits repeated identical buffers.

    CHANGED 2026-08-30 (report item B1): pools are generated **on the device**
    from a CUDA generator, straight into bf16. The old host path burned ~35 s
    per block on single-threaded fp32 RNG that was thrown away in the bf16 cast
    -- entirely outside every timed span, so no recorded measurement moves.
    Same-seed-same-bytes still holds *within* a process, which is the property
    an A/B needs. It no longer holds across the CPU/CUDA backends, so a given
    ``--seed`` does not reproduce a pre-2026-08-30 run's exact pool bytes;
    ``--cpu-pools`` restores the old path for that. No output field was removed
    or renamed, and ``config.cpu_pools`` appears only when the flag is passed.
  - **Three lanes, reported separately** (a lane is a claim type, not a flag):
      preprocessing  gather+quantize only        -> resolves the disputed cost
      core           kernel only, prequantized   -> diagnosis
      inclusive      quant + kernel, one span    -> the promotion-relevant lane
  - **Raw samples retained** in the JSON, alongside median/p5/p95.
  - **Environment recorded**, including clocks/power via nvidia-smi, and the
    timing backend named explicitly (cuda_events; a CUPTI lane is future work).

The schedule-replay aggregate is ``--layers 16``: 224 operator calls with
distinct data, workspace reused across all of them exactly as the serving
integration reuses it.

v1 measurement notes (recorded in the output):
  - Warm-L2, eager. Cold-L2 and CUDA-graph lanes are not yet implemented.
  - Single-process. Interleaved multi-process A/B blocks are Phase 3 work; for
    candidate zero this bench IS the baseline being established, so the paired
    statistics apply from the first comparison onward, not to this run.

Usage (rented target, from the repo root):
    python bench/candidate_bench.py --profile d256-24x4-446k \
        --label candidate-zero --layers 16
    python bench/candidate_bench.py --profile d256-24x4-446k \
        --smoke   # 2 shallow cases, 1 layer: harness self-check, any GPU

Three additions close gaps G6/G7/G13 of the evidence gap analysis. All are
opt-in: with no new flag the run, the printed table and the JSON are exactly
what they were.

  --control {sdpa-bf16,flashinfer-bf16,cudnn-frost-bf16}
      A denominator. The same hashed cases through a stock control **in the
      same process**, interleaved A-B-B-A per case, with a paired bootstrap 95%
      CI over cases. A speedup with no denominator measured beside it is not
      evidence, which is why this is a lane and not a footnote. Expect to run it
      with --layers 1 or a --chunks subset: the control is far slower than the
      candidate and pays the full schedule cost.

  --lane upstream-comparability
      FlashInfer #4502's protocol instead of the three-lane run, so a number of
      ours can sit beside a number of theirs: prequantized inputs (quantization
      excluded), preallocated out, CUDA-graph capture with one iteration
      in-graph, 10 warm-up replays, 100 timed replays, median, clocks requested
      and observed. Capture can legitimately fail here -- the workspace grows on
      demand, which is illegal mid-graph, and that is the whole of why the
      capability declares eager_only -- so a failure is recorded with its
      exception text and the run falls back to eager under the loud lane name
      ``upstream_comparability_FALLBACK_eager``. It is never relabelled silently.

  --lane candidate-graph
      The same capture/replay protocol as above, but around the operator this
      package actually ships: gather + quantize + kernel, all inside one graph,
      on a **capacity-reserved** workspace (``ws.reserve()``, added with the
      capture/replay test in ``tests/kernel/test_cuda_graph_capacity.py``). The
      comparability lane answers "how does our kernel compare with theirs under
      their protocol"; this one answers "what does the whole operator cost when
      one graph launch replaces ~14 per call". Its TF/s divides ATTENTION flops
      by a span containing the preprocessing, so it is a lower bound and shares
      no column with the comparability lane -- the JSON records that as
      ``flops_basis``. Capture failure here is a FINDING, not the expected
      outcome, because the reservation removes the reason capture used to fail.
      No number from this lane is promotable off a development target.

  --emit-repro
      One paste-able command line with every count spelled out, printed and
      embedded in the JSON. Precedent says the currency a reviewer accepts is a
      pasted command plus a named GPU plus warmup/iteration counts, not a
      manifest.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import pathlib
import statistics
import subprocess
import sys

# cuBLAS determinism, pinned at IMPORT scope on purpose (mirrors
# ``tests/kernel/conftest.py``).  ``quant.py`` runs an fp32 GEMM (the Hadamard
# rotation of Q and K); cuBLAS picks its algorithm partly from the size of the
# workspace it is handed, and split-K variants reduce with atomics, which is
# run-to-run nondeterministic.  The variable must be set BEFORE the first cuBLAS
# handle is created, so it cannot live inside ``pin_numerics()`` -- that is
# called after ``import torch``, which is benign only for as long as nothing
# above it touches cuBLAS.  The flag half of the pin stays in the function,
# where it needs a live ``torch``.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "attn_kernel_lab"))

HEAD_DIM = 256
BLK = 64

#: The lanes. A lane is a claim type, not a flag -- see the module docstring.
LANE_CHOICES = ("default", "upstream-comparability", "candidate-graph")

#: CLI spelling -> the key the lane's stats appear under in the JSON record.
LANE_RECORD_KEY = {
    "upstream-comparability": "upstream_comparability",
    "candidate-graph": "candidate_graph",
}


# ---------------------------------------------------------------- environment


def _nvidia_smi() -> dict:
    query = (
        "name,driver_version,uuid,power.limit,power.draw,clocks.sm,"
        "clocks.max.sm,temperature.gpu,persistence_mode"
    )
    try:
        out = (
            subprocess.run(
                ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            .stdout.strip()
            .splitlines()[0]
        )
        keys = query.split(",")
        return dict(zip(keys, [part.strip() for part in out.split(",")]))
    except Exception as exc:  # noqa: BLE001 -- record the absence, never fail the run
        return {"error": f"nvidia-smi unavailable: {exc}"}


def environment() -> dict:
    import torch

    props = torch.cuda.get_device_properties(0)
    from attn_kernel_lab.kernel import source_build_id

    return {
        "device_name": props.name,
        "capability": f"{props.major}.{props.minor}",
        "sm_count": props.multi_processor_count,
        "total_mem_gib": round(props.total_memory / 2**30, 2),
        "free_mem_gib_at_start": round(torch.cuda.mem_get_info()[0] / 2**30, 2),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": sys.version.split()[0],
        "source_build_id": source_build_id(),
        "nvidia_smi": _nvidia_smi(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def pin_numerics() -> None:
    """The half of the numerics pin that needs a live ``torch``.

    ``CUBLAS_WORKSPACE_CONFIG`` is NOT set here -- it is set at module import
    scope above, before anything can create a cuBLAS handle.
    """
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


# ------------------------------------------------------------------ workload


def load_workload(profile: str) -> dict:
    path = ROOT / "workloads" / "generated" / f"{profile}.cases.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {"payload": payload, "cases_sha256": digest, "path": str(path)}


def case_flops(q_len: int, prefix: int, q_heads: int) -> float:
    """Bottom-right-causal QK+PV over logically attended pairs."""
    return 4.0 * q_heads * HEAD_DIM * q_len * (prefix + (q_len + 1) / 2.0)


# --------------------------------------------------------------------- data


def _pool_generator(seed: int, cpu_pools: bool):
    """The generator every buffer of one case is drawn from.

    CUDA by default; CPU only under ``--cpu-pools`` (see ``_rand_bf16``).
    """
    import torch

    return torch.Generator(device=("cpu" if cpu_pools else "cuda")).manual_seed(seed)


def _rand_bf16(shape, generator, device):
    """Uniform bf16 noise drawn from ``generator``.

    The invariant an A/B needs is **same seed -> same bytes within a process**:
    quant work is data-dependent, so both arms of a comparison must see
    identical buffers.  A CUDA generator gives that just as the old CPU
    generator did -- ``manual_seed`` fixes the Philox stream, and these calls
    are issued in a fixed order on one stream.  (The older docstring claimed
    CPU *seeding* was the thing that made an A/B meaningful; it never was.
    Determinism was, and it is preserved.)

    What a device generator does NOT preserve is the byte sequence ACROSS
    backends: device Philox is a different stream from the host RNG, so
    ``--seed S`` today does not reproduce the exact bytes ``--seed S`` produced
    before 2026-08-30.  No recorded number depends on those bytes -- nothing
    downstream reads or hashes the generated data, only ``workload_cases_sha256``
    (a hash of the schedule *spec*) and timing/shape fields -- and pools are
    built strictly OUTSIDE every ``time_events`` / ``time_cuda_graph`` span.
    ``--cpu-pools`` restores the old host path for exact-byte reproduction of a
    historical run.

    Why it moved: the host path generated ~36 GiB of single-threaded fp32 per
    block and then threw half of it away in the bf16 cast -- ~35 s per block,
    ~560 s on a ``--layers 16`` run, all of it dead time beside a 66.7 s core
    lane.  bf16 is allocated directly here, so no transient fp32 buffer lands on
    the GPU and ``peak_alloc_gib`` keeps meaning what it meant.
    """
    import torch

    if generator.device.type == "cpu":
        t = torch.empty(shape, dtype=torch.float32)
        t.uniform_(-1.0, 1.0, generator=generator)
        return t.to(torch.bfloat16).to(device)
    t = torch.empty(shape, dtype=torch.bfloat16, device=device)
    t.uniform_(-1.0, 1.0, generator=generator)
    return t


def make_pool(n_rows: int, kv_heads: int, seed: int, device, cpu_pools: bool = False):
    """Full-size page_size-1 pool: every position is a distinct physical row,
    scattered by a fixed permutation (non-monotonic gather, like a radix pool
    after real traffic). No wrap, by design.

    The permutation is drawn from the same generator as the data, so it follows
    it onto the device; on the CUDA path it is built on-device and never
    crosses the bus."""
    import torch

    g = _pool_generator(seed, cpu_pools)
    rows = _ceil_to(n_rows, BLK)
    k_pool = _rand_bf16((rows, kv_heads, HEAD_DIM), g, device)
    v_pool = _rand_bf16((rows, kv_heads, HEAD_DIM), g, device)
    if cpu_pools:
        perm = torch.randperm(rows, generator=g)[:n_rows].contiguous().to(device)
    else:
        perm = torch.randperm(rows, generator=g, device=device)[:n_rows].contiguous()
    return k_pool, v_pool, perm


def _ceil_to(x: int, m: int) -> int:
    return (x + m - 1) // m * m


# ------------------------------------------------------------------- timing


def time_events(fn, warmup: int, iters: int) -> dict:
    """CUDA-event timing, warm L2, eager. Raw samples retained."""
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    return _stats(_sample_events(fn, iters))


def _sample_events(fn, iters: int) -> list[float]:
    """``iters`` CUDA-event-timed calls with no warm-up: one block of an A-B-B-A
    schedule. Split out of ``time_events`` so the interleaved lane and the
    standalone lanes time by exactly the same code rather than by two copies of
    it that drift."""
    import torch

    samples = []
    for _ in range(iters):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def time_cuda_graph(
    fn, warmup_replays: int, timed_replays: int, lane: str = "upstream_comparability"
) -> tuple[dict, str, str]:
    """#4502's timing protocol: capture one iteration, replay and time.

    Returns ``(stats, lane_name, capture_error)``. Capture is warmed on a side
    stream first, which the CUDA graph API requires and which #4272's test
    pattern also does.

    On capture failure the caller gets eager timing at the same replay counts
    under the name ``<lane>_FALLBACK_eager`` and the exception text. This is the
    honest outcome, not a defect to be smoothed over: under the DEFAULT
    workspace the buffers reallocate on growth, which is illegal mid-graph, and
    it is the reason ``V1_CAPABILITY.cuda_graph`` says ``eager_only``. A run
    that quietly reported eager numbers under a graph lane name would be worse
    than no run. (``--lane candidate-graph`` reserves a capacity-stable
    workspace first, so a capture failure there is a finding, not the expected
    outcome.)
    """
    import torch

    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()  # exactly one iteration in-graph, as #4502 does
        for _ in range(warmup_replays):
            graph.replay()
        torch.cuda.synchronize()

        samples = []
        for _ in range(timed_replays):
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            graph.replay()
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end))
        return _stats(samples), lane, None
    except Exception as exc:  # noqa: BLE001 -- recorded, never swallowed
        stats = time_events(fn, warmup=warmup_replays, iters=timed_replays)
        return stats, f"{lane}_FALLBACK_eager", f"{type(exc).__name__}: {exc}"


def _sm_clock_mhz():
    """Observed SM clock, or None. Requesting a clock is the operator's job
    (``nvidia-smi -lgc <mhz>``, which needs privileges); this records what the
    device was actually running at, so requested and observed can disagree in
    the record instead of silently in the numbers."""
    try:
        out = (
            subprocess.run(
                ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            .stdout.strip()
            .splitlines()[0]
        )
        return int(out.strip())
    except Exception:  # noqa: BLE001
        return None


def _stats(samples: list[float]) -> dict:
    ordered = sorted(samples)
    return {
        "samples_ms": [round(sample, 4) for sample in samples],
        "median_ms": round(statistics.median(samples), 4),
        "min_ms": round(ordered[0], 4),
        "p5_ms": round(ordered[max(0, int(0.05 * (len(ordered) - 1)))], 4),
        "p95_ms": round(ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))], 4),
        "iters": len(samples),
    }


# ------------------------------------------------------------------ controls
#
# A speedup is a ratio, and a ratio needs a denominator measured in the same
# process, on the same data, in the same thermal state. Gap G6. Both controls
# consume the identical hashed cases and the identical gathered K/V, and both
# are timed against a candidate block in the same A-B-B-A sandwich.

CONTROL_KINDS = ("none", "sdpa-bf16", "flashinfer-bf16", "cudnn-frost-bf16")


def _streaming_masked_attention(q, k, v, prefix, sm_scale, out, q_chunk, kv_chunk):
    """Chunked, streaming, bottom-right-causal attention at BF16.

    A **functional** control, not a peak-performance one, and the JSON says so.
    It is a stock-PyTorch composition -- batched matmul plus an FP32 online
    softmax, no fusion, no tiling for the memory hierarchy -- so it establishes
    that the candidate computes the same operator at a comparable cost class and
    nothing whatever about how a tuned BF16 kernel would perform. For a
    peak-performance denominator use ``--control flashinfer-bf16``.

    Nothing quadratic is materialised beyond one ``[H, q_chunk, kv_chunk]`` score
    block, so peak transient memory is set by the two chunk knobs and not by the
    context depth -- at 446 k positions a single materialised score matrix would
    be several hundred GiB.

    ``F.scaled_dot_product_attention`` is deliberately not used. It returns no
    LSE, so the fully-visible prefix and the diagonal band cannot be merged; the
    alternatives are one call per q-chunk over a mask as wide as the whole
    context, or this. GQA is handled by reshaping the query heads into the KV
    batch dimension rather than by ``repeat_interleave``, so K and V are read
    once per chunk instead of once per group.
    """
    import torch

    q_len, q_heads, head_dim = q.shape
    kv_heads = k.shape[1]
    group = q_heads // kv_heads
    device = q.device

    for r0 in range(0, q_len, q_chunk):
        r1 = min(r0 + q_chunk, q_len)
        rows = r1 - r0
        # [H, rows, D] -> [KVH, group*rows, D]: head h belongs to kv head
        # h // group, so the head axis folds into the batch axis directly.
        qc = q[r0:r1].permute(1, 0, 2).contiguous().view(kv_heads, group * rows, head_dim)
        limit = prefix + r1  # no row of this chunk can see past here
        row_pos = torch.arange(r0, r1, device=device)[:, None] + prefix

        run_max = torch.full(
            (kv_heads, group * rows, 1), float("-inf"), dtype=torch.float32, device=device
        )
        run_sum = torch.zeros_like(run_max)
        acc = torch.zeros((kv_heads, group * rows, head_dim), dtype=torch.float32, device=device)

        for c0 in range(0, limit, kv_chunk):
            c1 = min(c0 + kv_chunk, limit)
            kc = k[c0:c1].permute(1, 2, 0).contiguous()  # [KVH, D, cols]
            scores = torch.bmm(qc, kc).float() * sm_scale  # [KVH, group*rows, cols]
            if c1 > prefix + r0:  # the diagonal band; earlier chunks are wholly visible
                cols = torch.arange(c0, c1, device=device)[None, :]
                masked = (cols > row_pos).view(1, rows, c1 - c0)
                masked = masked.expand(group, rows, c1 - c0).reshape(1, group * rows, c1 - c0)
                scores.masked_fill_(masked, float("-inf"))

            chunk_max = scores.amax(dim=-1, keepdim=True)
            new_max = torch.maximum(run_max, chunk_max)
            # A chunk wholly masked for some row leaves that row at -inf; the
            # substitution keeps (-inf) - (-inf) out of the rescale.
            safe_max = torch.where(torch.isneginf(new_max), torch.zeros_like(new_max), new_max)
            alpha = torch.exp(run_max - safe_max)
            probs = torch.exp(scores - safe_max)
            run_sum = run_sum * alpha + probs.sum(dim=-1, keepdim=True)
            vc = v[c0:c1].permute(1, 0, 2).contiguous()  # [KVH, cols, D]
            acc = acc * alpha + torch.bmm(probs.to(torch.bfloat16), vc).float()
            run_max = new_max

        result = (acc / run_sum.clamp_min(1e-30)).to(torch.bfloat16)
        out[r0:r1] = (
            result.view(kv_heads, group, rows, head_dim)
            .permute(2, 0, 1, 3)
            .reshape(rows, q_heads, head_dim)
        )


def _make_sdpa_control(k_pool, v_pool, idx, q, prefix, args):
    """Gathered-BF16 streaming control. The gather is excluded from the timed
    step, which is the fair pairing: the candidate's core lane is measured on
    already-quantized inputs, so the control is measured on already-gathered
    ones."""
    import torch

    k = k_pool.index_select(0, idx).contiguous()
    v = v_pool.index_select(0, idx).contiguous()
    out = torch.empty_like(q)
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    def step():
        _streaming_masked_attention(
            q, k, v, prefix, sm_scale, out, args.control_q_chunk, args.control_kv_chunk
        )

    step()  # fail here, before timing, if it is going to fail at all
    torch.cuda.synchronize()
    return step, {
        "available": True,
        "impl": "chunked streaming bmm + fp32 online softmax, bf16 operands",
        "peak_performance_control": False,
        "interpretation": "functional denominator only: a stock-PyTorch "
        "composition with no fusion. It bounds what the "
        "candidate is computing, not what a tuned BF16 kernel "
        "would cost.",
        "excluded_from_timing": "kv gather (pool -> contiguous position order)",
        "q_chunk": args.control_q_chunk,
        "kv_chunk": args.control_kv_chunk,
    }


def _make_flashinfer_control(k_pool, v_pool, idx, q, prefix, geometry, args):
    """Stock FlashInfer BF16 paged prefill at D256 / page_size 1 / 24:4.

    Never fatal. An absent or incompatible FlashInfer is a fact about the
    environment and belongs in the record; a bench that dies because the
    denominator is missing loses the numerator too.
    """
    try:
        from flashinfer import BatchPrefillWithPagedKVCacheWrapper
    except Exception as exc:  # noqa: BLE001
        return None, {
            "available": False,
            "reason": f"control unavailable: flashinfer not installed ({exc})",
        }

    import torch

    try:
        q_len, q_heads, _ = q.shape
        kv_len = int(idx.numel())
        workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=q.device)
        wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")
        wrapper.plan(
            torch.tensor([0, q_len], dtype=torch.int32, device=q.device),
            torch.tensor([0, kv_len], dtype=torch.int32, device=q.device),
            idx.to(torch.int32),
            torch.tensor([1], dtype=torch.int32, device=q.device),
            q_heads,
            geometry["kv_heads"],
            HEAD_DIM,
            1,  # page_size
            causal=True,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )
        # [pool, KVH, D] -> the wrapper's NHD [num_pages, page_size, KVH, D]
        k_cache = k_pool.unsqueeze(1)
        v_cache = v_pool.unsqueeze(1)

        def step():
            wrapper.run(q, (k_cache, v_cache))

        step()
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001
        return None, {
            "available": False,
            "reason": f"control unavailable: {type(exc).__name__}: {exc}",
        }

    return step, {
        "available": True,
        "impl": "flashinfer.BatchPrefillWithPagedKVCacheWrapper, bf16, NHD, "
        "page_size 1, causal (bottom-right)",
        "peak_performance_control": True,
        "excluded_from_timing": "plan() (done once per case, as a serving stack would)",
    }


def _make_cudnn_frost_control(k_pool, v_pool, idx, q, prefix, geometry, args):
    """NVIDIA FROST DSL SM120 BF16 dense SDPA (cudnn-frontend >= 1.27),
    bottom-right causal, GQA, at the case geometry.

    Why BF16: cuDNN's FP8 and MXFP8 engines are registered at d_qk = 128 ONLY
    (frozenset({128}) in cudnn/sdpa/fwd/engines.py; probed 2026-08-30), so this
    is the strongest cuDNN engine that can enter the ring at D256 at all.

    DENSE control: consumes pre-gathered contiguous BHSD K/V built OUTSIDE the
    timed span, so it skips the paged gather the candidate's inclusive lane
    pays -- the comparison is conservative against the candidate. The mask was
    verified against an explicit bottom-right fp32 reference before any timing
    was trusted (the pygraph-layer path silently ignored diagonal alignment
    and produced top-left results; probes/cudnn_frost/ records both probes).
    Template compilation is per-geometry and cached by the frontend; it happens
    at build time here, outside the timed span, like flashinfer's plan()."""
    import math as _math

    import torch

    info = {
        "impl": "cudnn.sdpa.fwd.SdpaFwdDslSm120 (FROST DSL), bf16 BHSD dense, "
        "causal_bottom_right, default tiles/scheduler",
        "peak_performance_control": True,
        "excluded_from_timing": "gather to dense BHSD + per-shape template compile",
        "note": "dense: no paged-KV support in this engine; pre-gathered inputs",
    }
    try:
        from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm120
    except Exception as exc:  # noqa: BLE001 -- record, never crash the bench
        info.update(available=False, unavailable_reason=f"cudnn frontend unavailable: {exc}")
        return None, info

    head_dim = q.shape[2]
    dense_k = k_pool.index_select(0, idx).permute(1, 0, 2).unsqueeze(0).contiguous()
    dense_v = v_pool.index_select(0, idx).permute(1, 0, 2).unsqueeze(0).contiguous()
    dense_q = q.permute(1, 0, 2).unsqueeze(0).contiguous()
    dense_o = torch.empty_like(dense_q)
    try:
        api = SdpaFwdDslSm120(
            sample_q=dense_q,
            sample_k=dense_k,
            sample_v=dense_v,
            sample_o=dense_o,
            is_causal=True,
            causal_bottom_right=True,
            scale_softmax=1.0 / _math.sqrt(head_dim),
        )
        api.check_support()
        api.compile()
    except Exception as exc:  # noqa: BLE001
        info.update(available=False, unavailable_reason=f"{type(exc).__name__}: {str(exc)[:300]}")
        return None, info

    ws_bytes = api.scratch_workspace_bytes()
    workspace = torch.empty(ws_bytes, device=q.device, dtype=torch.uint8) if ws_bytes else None

    def step():
        kwargs = dict(q_tensor=dense_q, k_tensor=dense_k, v_tensor=dense_v, o_tensor=dense_o)
        if workspace is not None:
            kwargs["workspace"] = workspace
        api.execute(**kwargs)

    info["available"] = True
    return step, info


def run_control_pairing(candidate_step, k_pool, v_pool, idx, q, prefix, geometry, args):
    """One case's A-B-B-A sandwich. Returns the record, never raises.

    A-B-B-A rather than A-B: any monotone drift over the block -- clocks
    settling, a neighbour waking up -- biases A-B by half the drift and cancels
    to first order in A-B-B-A. The candidate side is deliberately the *eager*
    kernel step in both lanes, so that the ratio compares like with like even
    when the run's headline number comes from a graph.
    """
    if args.control == "sdpa-bf16":
        step, info = _make_sdpa_control(k_pool, v_pool, idx, q, prefix, args)
    elif args.control == "cudnn-frost-bf16":
        step, info = _make_cudnn_frost_control(k_pool, v_pool, idx, q, prefix, geometry, args)
    else:
        step, info = _make_flashinfer_control(k_pool, v_pool, idx, q, prefix, geometry, args)
    record = {
        "kind": args.control,
        "interleave": "ABBA",
        "paired_against": "core (eager, cuda_events)",
        **info,
    }
    if step is None:
        return record

    import torch

    for _ in range(args.warmup):
        candidate_step()
    for _ in range(args.control_warmup):
        step()
    torch.cuda.synchronize()

    candidate_samples = _sample_events(candidate_step, args.iters)  # A
    control_samples = _sample_events(step, args.control_iters)  # B
    control_samples += _sample_events(step, args.control_iters)  # B
    candidate_samples += _sample_events(candidate_step, args.iters)  # A

    record["candidate"] = _stats(candidate_samples)
    record["control"] = _stats(control_samples)
    record["speedup_median"] = round(
        record["control"]["median_ms"] / record["candidate"]["median_ms"], 4
    )
    return record


def paired_bootstrap(speedups: list[float], resamples: int, seed: int) -> dict:
    """Percentile bootstrap 95% CI on the geometric mean of per-case speedups.

    The resampling unit is the **case**, not the iteration: iterations inside one
    case are not independent evidence about the schedule, and treating them as
    such is how a bench manufactures a tight interval around a number it has one
    observation of. Geometric mean because the statistic is a ratio.
    """
    import random

    if not speedups:
        return {"cases": 0}
    rng = random.Random(seed)
    count = len(speedups)
    logs = [math.log(value) for value in speedups]
    draws = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(count):
            total += logs[rng.randrange(count)]
        draws.append(math.exp(total / count))
    draws.sort()
    return {
        "cases": count,
        "geomean": round(math.exp(sum(logs) / count), 4),
        "ci95_low": round(draws[int(0.025 * (resamples - 1))], 4),
        "ci95_high": round(draws[int(0.975 * (resamples - 1))], 4),
        "min": round(min(speedups), 4),
        "median": round(statistics.median(speedups), 4),
        "max": round(max(speedups), 4),
        "resamples": resamples,
        "resample_unit": "case",
        "seed": seed,
    }


# ---------------------------------------------------------------- repro line


def repro_command(args) -> str:
    """The whole run as one paste-able line, every count explicit (gap G13).

    ``--smoke`` is expanded into the counts it implies rather than passed along,
    so the line reproduces the run even if the meaning of ``--smoke`` changes.
    """
    import shlex

    parts = [
        "python",
        "bench/candidate_bench.py",
        "--profile",
        args.profile,
        "--label",
        args.label,
        "--layers",
        str(args.layers),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--quant-iters",
        str(args.quant_iters),
        "--seed",
        str(args.seed),
        "--lane",
        args.lane,
        "--control",
        args.control,
    ]
    if args.chunks:
        parts += ["--chunks", args.chunks]
    if args.control != "none":
        parts += [
            "--control-warmup",
            str(args.control_warmup),
            "--control-iters",
            str(args.control_iters),
            "--control-q-chunk",
            str(args.control_q_chunk),
            "--control-kv-chunk",
            str(args.control_kv_chunk),
            "--bootstrap",
            str(args.bootstrap),
        ]
    if args.lane != "default":
        parts += [
            "--graph-warmup-replays",
            str(args.graph_warmup_replays),
            "--graph-timed-replays",
            str(args.graph_timed_replays),
        ]
        if args.request_clock_mhz:
            parts += ["--request-clock-mhz", str(args.request_clock_mhz)]
    if args.cpu_pools:
        parts.append("--cpu-pools")
    parts.append("--emit-repro")
    return shlex.join(parts)


def _across_cases(values: list[float]) -> dict:
    """median / p5 / p95 **across cases**, one median per case.

    The schedule is the population and a case is the observation; pooling raw
    iterations here would report an interval about a number the run has one
    independent measurement of.
    """
    ordered = sorted(values)
    return {
        "median_ms": round(statistics.median(values), 4),
        "p5_ms": round(ordered[max(0, int(0.05 * (len(ordered) - 1)))], 4),
        "p95_ms": round(ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))], 4),
        "cases": len(values),
    }


# ----------------------------------------------------------------- one case


def run_case(ext, q_mod, ws, geometry: dict, case: dict, layer_seed: int, args) -> dict:
    """One (chunk geometry, layer seed): fresh data, reused workspace."""
    import torch

    device = torch.device("cuda")
    q_heads, kv_heads = geometry["q_heads"], geometry["kv_heads"]
    q_len, prefix, n = case["q_len"], case["prefix_len"], case["k_len"]
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    seed = (case["chunk"] * 1009 + layer_seed) * 2 + args.seed

    k_pool, v_pool, idx = make_pool(n, kv_heads, seed, device, args.cpu_pools)
    g = _pool_generator(seed + 1, args.cpu_pools)
    q = _rand_bf16((q_len, q_heads, HEAD_DIM), g, device)
    mask = torch.ones(q_heads, dtype=torch.uint8, device=device)  # all-fp8-PV (production)

    def quant_step():
        kv = q_mod.gather_quantize_kv(
            ws,
            k_pool,
            v_pool,
            idx,
            need_vt8=True,
            need_vb16=False,
            center_k=True,
            qk_i8=True,
            rotate=True,
        )
        q8, qscale, mpad = q_mod.quantize_q(ws, q, sm_scale, qk_i8=True, rotate=True)
        return kv, q8, qscale, mpad

    default_lane = args.lane == "default"
    preprocessing = (
        time_events(quant_step, warmup=1, iters=args.quant_iters) if default_lane else None
    )
    kv, q8, qscale, mpad = quant_step()
    out = ws.get("o", (q_heads, mpad, HEAD_DIM), torch.bfloat16)

    def kernel_step():
        ext.fp8_prefill_attn(
            q8,
            kv["k8"],
            kv["vt8"],
            kv["vb16"],
            out,
            qscale,
            kv["kscale"],
            kv["vscale"],
            kv["vlog2r"],
            kv["vinvr"],
            kv["vmean"],
            mask,
            kv["n"],
            prefix,
            True,
            True,
            True,
        )

    flops = case_flops(q_len, prefix, q_heads)
    lane_name, capture_error, clocks = "default", None, None

    if default_lane:
        core = time_events(kernel_step, warmup=args.warmup, iters=args.iters)

        def inclusive_step():
            kv2, q8_2, qscale_2, _ = quant_step()
            ext.fp8_prefill_attn(
                q8_2,
                kv2["k8"],
                kv2["vt8"],
                kv2["vb16"],
                out,
                qscale_2,
                kv2["kscale"],
                kv2["vscale"],
                kv2["vlog2r"],
                kv2["vinvr"],
                kv2["vmean"],
                mask,
                kv2["n"],
                prefix,
                True,
                True,
                True,
            )

        inclusive = time_events(inclusive_step, warmup=1, iters=args.quant_iters)
        core["tflops_median"] = round(flops / (core["median_ms"] * 1e-3) / 1e12, 1)
        core["tflops_min_time"] = round(flops / (core["min_ms"] * 1e-3) / 1e12, 1)
        lanes = {"preprocessing": preprocessing, "core": core, "inclusive": inclusive}
    elif args.lane == "candidate-graph":
        # The CANDIDATE graph lane. Same replay protocol as the comparability
        # lane above -- capture one iteration, N warm-up replays, N timed
        # replays, median -- but the captured span is the operator the package
        # actually ships: gather + quantize + kernel, the `inclusive` lane's
        # work, in one graph. That is only capturable because ``ws`` was
        # capacity-reserved by main() before the first case ran; under the
        # grow-on-demand workspace the first differently-shaped request would
        # reallocate the buffers whose addresses the graph is holding.
        #
        # It exists to answer a question the comparability lane cannot: what
        # does the whole operator cost when the launch overhead of ~14 kernels
        # per call is replaced by one graph launch? No number from it is
        # promotable off a development target.
        # ``_out=out`` binds the destination at definition time rather than
        # through a closure cell: the cleanup ``del`` at the end of this function
        # empties those cells, and a graph-lane step that outlives it must not
        # depend on one.
        def inclusive_graph_step(_out=out):
            kv2, q8_2, qscale_2, _ = quant_step()
            ext.fp8_prefill_attn(
                q8_2,
                kv2["k8"],
                kv2["vt8"],
                kv2["vb16"],
                _out,
                qscale_2,
                kv2["kscale"],
                kv2["vscale"],
                kv2["vlog2r"],
                kv2["vinvr"],
                kv2["vmean"],
                mask,
                kv2["n"],
                prefix,
                True,
                True,
                True,
            )

        clock_before = _sm_clock_mhz()
        stats, lane_name, capture_error = time_cuda_graph(
            inclusive_graph_step,
            args.graph_warmup_replays,
            args.graph_timed_replays,
            lane="candidate_graph",
        )
        stats["tflops_median"] = round(flops / (stats["median_ms"] * 1e-3) / 1e12, 1)
        stats["tflops_min_time"] = round(flops / (stats["min_ms"] * 1e-3) / 1e12, 1)
        # Named explicitly because the denominator is NOT the comparability
        # lane's: attention FLOPs over a span that also contains the gather and
        # the quantization is a lower bound on achieved TF/s, not a core number,
        # and the two must never be read off the same column.
        stats["flops_basis"] = "attention_flops_over_inclusive_span"
        stats["captured_span"] = "gather+quantize+kernel"
        stats["workspace_mode"] = "capacity_reserved"
        clocks = {
            "requested_mhz": args.request_clock_mhz,
            "observed_before_mhz": clock_before,
            "observed_after_mhz": _sm_clock_mhz(),
        }
        lanes = {"candidate_graph": stats}
    else:
        # #4502's lane: inputs are already quantized and `out` already exists, so
        # the timed span is the kernel and nothing else -- the same exclusion
        # their number carries.
        clock_before = _sm_clock_mhz()
        stats, lane_name, capture_error = time_cuda_graph(
            kernel_step, args.graph_warmup_replays, args.graph_timed_replays
        )
        stats["tflops_median"] = round(flops / (stats["median_ms"] * 1e-3) / 1e12, 1)
        stats["tflops_min_time"] = round(flops / (stats["min_ms"] * 1e-3) / 1e12, 1)
        clocks = {
            "requested_mhz": args.request_clock_mhz,
            "observed_before_mhz": clock_before,
            "observed_after_mhz": _sm_clock_mhz(),
        }
        lanes = {"upstream_comparability": stats}

    control = (
        run_control_pairing(kernel_step, k_pool, v_pool, idx, q, prefix, geometry, args)
        if args.control != "none"
        else None
    )

    peak = torch.cuda.max_memory_allocated() / 2**30

    # No ``empty_cache()`` here, deliberately. ``peak_alloc_gib`` above derives
    # from ``max_memory_allocated``, which is governed by the reset below and
    # not by whether the caching allocator handed free blocks back to the
    # driver; ``memory_reserved`` -- the only thing ``empty_cache`` moves -- is
    # never read in this file. It bought nothing and cost a device-wide sync
    # plus driver-level frees on every one of the 224 cases.
    del k_pool, v_pool, idx, q, kv, q8, qscale, out
    torch.cuda.reset_peak_memory_stats()
    record = {
        "chunk": case["chunk"],
        "layer_seed": layer_seed,
        "q_len": q_len,
        "prefix_len": prefix,
        "k_len": n,
        "flops": flops,
        **lanes,
        "peak_alloc_gib": round(peak, 3),
    }
    if not default_lane:
        record["lane"] = lane_name
        record["clocks_mhz"] = clocks
        if capture_error is not None:
            record["graph_capture_error"] = capture_error
    if control is not None:
        record["control"] = control
    return record


# --------------------------------------------------------------------- main


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--profile", default="d256-24x4-446k")
    parser.add_argument("--label", default="unnamed")
    parser.add_argument(
        "--layers",
        type=int,
        default=1,
        help="distinct layer seeds per chunk (16 = full schedule replay)",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--quant-iters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument(
        "--chunks", default=None, help="comma-separated chunk ordinals to run (default: all)"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="chunks 0,1 only, 1 layer, minimal iters: harness self-check",
    )
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--lane",
        choices=LANE_CHOICES,
        default="default",
        help="default = the three-lane run; upstream-comparability "
        "= FlashInfer #4502's protocol instead (CUDA graph, "
        "prequantized, 10/100 replays); candidate-graph = the "
        "same replay protocol around the FULL operator "
        "(gather+quantize+kernel) on a capacity-reserved "
        "workspace",
    )
    parser.add_argument(
        "--control",
        choices=CONTROL_KINDS,
        default="none",
        help="stock control run on the same cases in the same "
        "process, interleaved A-B-B-A (gap G6)",
    )
    parser.add_argument("--control-warmup", type=int, default=1)
    parser.add_argument(
        "--control-iters",
        type=int,
        default=3,
        help="timed control calls per A-B-B-A block (2 blocks per "
        "case); the control is much slower than the candidate",
    )
    parser.add_argument(
        "--control-q-chunk",
        type=int,
        default=256,
        help="sdpa-bf16 only: query rows per streaming chunk",
    )
    parser.add_argument(
        "--control-kv-chunk",
        type=int,
        default=4096,
        help="sdpa-bf16 only: kv columns per streaming chunk; with "
        "--control-q-chunk this caps transient score memory",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=10000,
        help="paired bootstrap resamples over cases for the speedup 95%% CI",
    )
    parser.add_argument(
        "--graph-warmup-replays",
        type=int,
        default=10,
        help="upstream-comparability lane: #4502's 10 warm-ups",
    )
    parser.add_argument(
        "--graph-timed-replays",
        type=int,
        default=100,
        help="upstream-comparability lane: #4502's 100 repeats",
    )
    parser.add_argument(
        "--request-clock-mhz",
        type=int,
        default=None,
        help="record the SM clock this run asked for (#4502 "
        "requested 2430). Locking it is the operator's job "
        "(nvidia-smi -lgc); observed clocks are sampled "
        "around every timed block either way",
    )
    parser.add_argument(
        "--emit-repro",
        action="store_true",
        help="print the full paste-able repro command and embed "
        "it in the JSON as repro_command (gap G13)",
    )
    parser.add_argument(
        "--cpu-pools",
        action="store_true",
        help="generate the K/V pools with a CPU generator, the "
        "pre-2026-08-30 path. ~35 s/block of untimed host "
        "RNG; the ONLY reason to use it is reproducing a "
        "historical run's exact pool bytes",
    )
    args = parser.parse_args(argv)

    import torch

    if not torch.cuda.is_available():
        print("ERROR: no CUDA device.", file=sys.stderr)
        return 2

    pin_numerics()
    workload = load_workload(args.profile)
    geometry = workload["payload"]["geometry"]
    cases = workload["payload"]["cases"]
    if args.smoke:
        args.layers, args.iters, args.quant_iters, args.warmup = 1, 3, 2, 1
        args.control_warmup, args.control_iters = 1, 2
        args.graph_warmup_replays, args.graph_timed_replays = 2, 5
        cases = cases[:2]
        # so the repro line reproduces this run by explicit counts rather than by
        # whatever --smoke happens to mean at the time it is pasted
        args.chunks = ",".join(str(case["chunk"]) for case in cases)
    elif args.chunks:
        wanted = {int(part) for part in args.chunks.split(",")}
        cases = [case for case in cases if case["chunk"] in wanted]

    env = environment()
    print(
        f"device : {env['device_name']} sm{env['capability']} {env['sm_count']} SMs, "
        f"{env['free_mem_gib_at_start']:.1f} GiB free"
    )
    print(f"profile: {args.profile}  cases_sha256={workload['cases_sha256'][:16]}…")
    lane_key = LANE_RECORD_KEY.get(args.lane)
    if args.lane == "default":
        print("lanes  : preprocessing / core / inclusive  (cuda_events, warm-L2, eager)")
    elif args.lane == "candidate-graph":
        print(
            f"lanes  : candidate-graph  (cuda graph over gather+quantize+kernel, "
            f"capacity-reserved workspace, {args.graph_warmup_replays} warm-up / "
            f"{args.graph_timed_replays} timed replays, warm-L2)"
        )
    else:
        print(
            f"lanes  : upstream-comparability  (#4502 protocol: cuda graph, "
            f"prequantized, {args.graph_warmup_replays} warm-up / "
            f"{args.graph_timed_replays} timed replays, warm-L2)"
        )
    if args.control != "none":
        print(
            f"control: {args.control}  interleaved A-B-B-A, "
            f"{args.control_iters} iters x 2 blocks per case, "
            f"{args.bootstrap} bootstrap resamples over cases"
        )
        if args.control == "sdpa-bf16":
            print(
                "         (functional control: stock-PyTorch streaming "
                "attention, NOT a peak-performance baseline)"
            )
    print(f"plan   : {len(cases)} chunk(s) x {args.layers} layer seed(s)")
    print("compiling kernel (JIT, ~1 min cold) ...", flush=True)
    from attn_kernel_lab import kernel as kernel_mod
    import quant as q_mod

    ext = kernel_mod.load()
    ws = q_mod.FP8PrefillWorkspace(torch.device("cuda"))
    if args.lane == "candidate-graph":
        # The lane's precondition, not a nicety: a captured graph holds the
        # device addresses it saw at capture, so the workspace must be sized for
        # the DEEPEST case in the run before the first capture and must never
        # reallocate afterwards. Reserving here (rather than per case) is also
        # what makes the per-case captures comparable -- every case replays
        # against the same allocation.
        max_kv = max(case["k_len"] for case in cases)
        max_q = max(case["q_len"] for case in cases)
        plan = ws.reserve(max_kv, max_q, q_heads=geometry["q_heads"], kv_heads=geometry["kv_heads"])
        print(
            f"workspc: capacity-reserved for kv<={plan.max_kv_len} "
            f"q<={plan.max_q_len} ({plan.q_heads}:{plan.kv_heads}); "
            "any request outside the plan raises instead of reallocating"
        )
    torch.cuda.reset_peak_memory_stats()

    results = []
    if args.lane == "default":
        header = (
            f"{'chunk':>5} {'layer':>5} {'q_len':>7} {'prefix':>7} "
            f"{'core ms':>9} {'TF/s':>7} {'quant ms':>9} {'incl ms':>9} {'GiB':>6}"
        )
    else:
        header = (
            f"{'chunk':>5} {'layer':>5} {'q_len':>7} {'prefix':>7} "
            f"{'graph ms':>9} {'TF/s':>7} {'clk MHz':>9} {'GiB':>6}"
        )
    print("\n" + header)
    print("-" * len(header))
    for case in cases:
        for layer_seed in range(args.layers):
            record = run_case(ext, q_mod, ws, geometry, case, layer_seed, args)
            results.append(record)
            if args.lane == "default":
                print(
                    f"{record['chunk']:>5} {layer_seed:>5} {record['q_len']:>7} "
                    f"{record['prefix_len']:>7} {record['core']['median_ms']:>9.2f} "
                    f"{record['core']['tflops_median']:>7.1f} "
                    f"{record['preprocessing']['median_ms']:>9.2f} "
                    f"{record['inclusive']['median_ms']:>9.2f} "
                    f"{record['peak_alloc_gib']:>6.2f}"
                )
            else:
                lane_stats = record[lane_key]
                observed = record["clocks_mhz"]["observed_after_mhz"]
                print(
                    f"{record['chunk']:>5} {layer_seed:>5} {record['q_len']:>7} "
                    f"{record['prefix_len']:>7} {lane_stats['median_ms']:>9.2f} "
                    f"{lane_stats['tflops_median']:>7.1f} "
                    f"{(observed if observed is not None else -1):>9} "
                    f"{record['peak_alloc_gib']:>6.2f}"
                )
                if record.get("graph_capture_error"):
                    print(
                        f"      ! graph capture failed, EAGER FALLBACK: "
                        f"{record['graph_capture_error']}"
                    )
            control = record.get("control")
            if control and control.get("available"):
                print(
                    f"      control {control['kind']}: "
                    f"{control['control']['median_ms']:.2f} ms vs candidate "
                    f"{control['candidate']['median_ms']:.2f} ms  "
                    f"-> {control['speedup_median']:.2f}x"
                )
            elif control:
                print(f"      {control['reason']}")

    layers_full = geometry["layers"]
    agg = {}
    lane_keys = ("preprocessing", "core", "inclusive") if args.lane == "default" else (lane_key,)
    for lane in lane_keys:
        per_chunk = {}
        for record in results:
            per_chunk.setdefault(record["chunk"], []).append(record[lane]["median_ms"])
        chunk_medians = {chunk: statistics.median(vals) for chunk, vals in per_chunk.items()}
        # One full schedule = every chunk once per layer. With L measured layer
        # seeds we extrapolate the remaining (layers_full - L) layers at the
        # per-chunk median; at --layers 16 nothing is extrapolated.
        measured = sum(sum(vals) for vals in per_chunk.values())
        extrapolated = sum(chunk_medians.values()) * max(0, layers_full - args.layers)
        agg[lane] = {
            "schedule_ms": round(measured + extrapolated, 1),
            "measured_calls": len(results),
            "extrapolated_calls": max(0, layers_full - args.layers) * len(per_chunk),
        }
    total_flops = (
        sum(case_flops(c["q_len"], c["prefix_len"], geometry["q_heads"]) for c in cases)
        * layers_full
    )
    if args.lane == "default":
        agg["core"]["schedule_tflops"] = round(
            total_flops / (agg["core"]["schedule_ms"] * 1e-3) / 1e12, 1
        )
        agg["inclusive"]["schedule_tflops_honest"] = round(
            total_flops / (agg["inclusive"]["schedule_ms"] * 1e-3) / 1e12, 1
        )
    else:
        agg[lane_key]["schedule_tflops"] = round(
            total_flops / (agg[lane_key]["schedule_ms"] * 1e-3) / 1e12, 1
        )

    # One honest lane name for the whole run: a single failed capture demotes it,
    # because a mixed run reported as a graph run is a mislabelled run.
    per_case_lanes = {record.get("lane", "default") for record in results}
    run_lane = (
        "default"
        if args.lane == "default"
        else (lane_key if per_case_lanes == {lane_key} else f"{lane_key}_FALLBACK_eager")
    )

    print(
        f"\nschedule aggregate over {len(cases)} chunk(s) x {layers_full} layers "
        f"({'extrapolated from ' + str(args.layers) + ' seed(s)' if args.layers < layers_full else 'fully measured'}):"
    )
    if args.lane == "default":
        print(
            f"  preprocessing : {agg['preprocessing']['schedule_ms'] / 1000:.2f} s   <-- the disputed number"
        )
        print(
            f"  core          : {agg['core']['schedule_ms'] / 1000:.2f} s   "
            f"({agg['core']['schedule_tflops']} TF/s)"
        )
        print(
            f"  inclusive     : {agg['inclusive']['schedule_ms'] / 1000:.2f} s   "
            f"({agg['inclusive']['schedule_tflops_honest']} TF/s honest)"
        )
    else:
        lane_agg = agg[lane_key]
        print(
            f"  {run_lane} : {lane_agg['schedule_ms'] / 1000:.2f} s   "
            f"({lane_agg['schedule_tflops']} TF/s)"
        )
        if args.lane == "candidate-graph":
            print(
                "  NOTE: the TF/s above divides ATTENTION flops by a span that "
                "also contains the gather and the quantization. It is a lower "
                "bound, not a core number, and does not compare with the "
                "upstream-comparability lane."
            )
        if run_lane.endswith("FALLBACK_eager"):
            print(
                f"  NOTE: at least one case could not be captured; this is an "
                f"EAGER number under the {lane_key} lane name, not a graph one."
            )

    control_agg = None
    if args.control != "none":
        controls = [record["control"] for record in results if "control" in record]
        speedups = [c["speedup_median"] for c in controls if "speedup_median" in c]
        control_agg = {
            "kind": args.control,
            "interleave": "ABBA",
            "available_cases": sum(1 for c in controls if c.get("available")),
            "total_cases": len(controls),
            "unavailable_reason": next(
                (c["reason"] for c in controls if not c.get("available")), None
            ),
            "peak_performance_control": next(
                (c.get("peak_performance_control") for c in controls if c.get("available")), None
            ),
            "paired_speedup": paired_bootstrap(speedups, args.bootstrap, args.seed),
        }
        for side in ("candidate", "control"):
            medians = [c[side]["median_ms"] for c in controls if side in c]
            if medians:
                control_agg[f"{side}_across_cases"] = _across_cases(medians)
        if speedups:
            boot = control_agg["paired_speedup"]
            print(f"\ncontrol ({args.control}, A-B-B-A, {boot['cases']} paired case(s)):")
            print(
                f"  control median across cases : "
                f"{control_agg['control_across_cases']['median_ms']:.2f} ms "
                f"(p5 {control_agg['control_across_cases']['p5_ms']:.2f}, "
                f"p95 {control_agg['control_across_cases']['p95_ms']:.2f})"
            )
            print(
                f"  paired speedup (geomean)    : {boot['geomean']:.2f}x  "
                f"95% CI [{boot['ci95_low']:.2f}, {boot['ci95_high']:.2f}]  "
                f"min {boot['min']:.2f}x / max {boot['max']:.2f}x"
            )
            if control_agg["peak_performance_control"] is False:
                print(
                    "  NOTE: functional control. This ratio is not a "
                    "peak-performance speedup and must not be quoted as one."
                )
        else:
            print(f"\ncontrol ({args.control}): {control_agg['unavailable_reason']}")

    measurement = {
        "timing_backend": "cuda_events",
        "l2_policy": "warm",
        "graph_mode": "eager",
        "process_model": "single",
    }
    if args.lane != "default":
        measurement["lane"] = run_lane
        measurement["graph_mode"] = "cuda_graph" if run_lane == lane_key else "eager"
        measurement["graph_iters_within_graph"] = 1
        measurement["graph_warmup_replays"] = args.graph_warmup_replays
        measurement["graph_timed_replays"] = args.graph_timed_replays
        measurement["quantization"] = (
            "included (gather+quantize+kernel inside the graph)"
            if args.lane == "candidate-graph"
            else "excluded (prequantized inputs, preallocated out)"
        )
        if args.lane == "candidate-graph":
            measurement["workspace_mode"] = "capacity_reserved"
        measurement["clock_requested_mhz"] = args.request_clock_mhz
    if args.control != "none":
        measurement["control"] = args.control

    config_keys = ["layers", "warmup", "iters", "quant_iters", "seed", "smoke"]
    if args.lane != "default":
        config_keys += ["lane", "graph_warmup_replays", "graph_timed_replays", "request_clock_mhz"]
    if args.control != "none":
        config_keys += [
            "control",
            "control_warmup",
            "control_iters",
            "control_q_chunk",
            "control_kv_chunk",
            "bootstrap",
        ]
    # Recorded only when set, like every other non-default switch above, so a
    # default run's `config` block stays byte-for-byte what it has always been.
    # (The default pool RNG moved to the device on 2026-08-30; `argv` and
    # `utc` already distinguish a `--cpu-pools` run from a default one.)
    if args.cpu_pools:
        config_keys += ["cpu_pools"]

    doc = {
        "schema": "candidate-bench/1",
        "label": args.label,
        "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "argv": sys.argv[1:],
        "workload_profile": args.profile,
        "workload_cases_sha256": workload["cases_sha256"],
        "measurement": measurement,
        "config": {key: getattr(args, key) for key in config_keys},
        "env": env,
        "geometry": geometry,
        "cases": results,
        "schedule_aggregate": agg,
    }
    if control_agg is not None:
        doc["control_aggregate"] = control_agg
    if args.emit_repro:
        doc["repro_command"] = repro_command(args)
    out = (
        pathlib.Path(args.out)
        if args.out
        else (
            ROOT
            / "bench"
            / "results"
            / f"{env['capability'].replace('.', '')}-{_dt.datetime.now(_dt.timezone.utc):%Y%m%dT%H%M%SZ}-{args.label}.json"
        )
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    print(f"sha256 {hashlib.sha256(out.read_bytes()).hexdigest()}")
    if args.emit_repro:
        print("\nreproduce this run:")
        print(f"  {doc['repro_command']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
