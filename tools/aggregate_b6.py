#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Aggregate the B6 campaign: independent bench blocks -> the PR's perf table.

The statistics rule (architecture doc, "Same-machine comparisons"): bootstrap
over INDEPENDENT blocks/processes, never over inner CUDA timing iterations.
Each input JSON is one independently launched process with its own CUDA
context and its own deterministic data seed; within a block, candidate and
control saw identical bytes in ABBA order. This script resamples at block
level -- the stronger unit than the bench's own per-case resampling.

    aggregate_b6.py --control 'bench/results/*b6-ctrl-b*.json' \
                    --compat  'bench/results/*b6-compat-b*.json' \
                    --out bench/results/B6-SUMMARY-<date>

Fails loudly on: mixed workload hashes, missing chunks, an unavailable control
in any block, or fewer than 2 control blocks (no CI without independence).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import pathlib
import random
import statistics
import sys

LAYERS = 16


def _load(pattern: str) -> list[dict]:
    paths = sorted(glob.glob(pattern))
    return [(p, json.loads(pathlib.Path(p).read_text())) for p in paths]


def _boot_ci(values: list[float], resamples: int = 10000, seed: int = 0):
    rng = random.Random(seed)
    n = len(values)
    stats = sorted(
        statistics.median([values[rng.randrange(n)] for _ in range(n)])
        for _ in range(resamples)
    )
    return stats[int(0.025 * resamples)], stats[int(0.975 * resamples)]


def aggregate_control(blocks: list[tuple[str, dict]]) -> dict:
    if len(blocks) < 2:
        sys.exit(f"need >=2 independent control blocks for a CI, got {len(blocks)}")
    hashes = {doc["workload_cases_sha256"] for _, doc in blocks}
    if len(hashes) != 1:
        sys.exit(f"mixed workload hashes across blocks: {hashes}")

    chunk_ids = sorted({case["chunk"] for _, doc in blocks for case in doc["cases"]})
    per_chunk: dict[int, dict] = {c: {"cand": [], "ctrl": [], "geom": None} for c in chunk_ids}
    sched_ratio, sched_core_s, sched_ctrl_s, sched_incl_s, sched_prep_s = [], [], [], [], []

    for path, doc in blocks:
        agg = doc.get("control_aggregate") or {}
        if agg.get("unavailable_reason"):
            sys.exit(f"{path}: control unavailable: {agg['unavailable_reason']}")
        cases = {case["chunk"]: case for case in doc["cases"]}
        if sorted(cases) != chunk_ids:
            sys.exit(f"{path}: chunk set {sorted(cases)} != {chunk_ids}")
        core = ctrl = incl = prep = 0.0
        for c in chunk_ids:
            case = cases[c]
            block = case["control"]
            if not block.get("available", False):
                sys.exit(f"{path}: chunk {c}: control unavailable")
            # The ABBA pairing's own candidate re-measurement is the honest
            # numerator (same visits, same interleave); ``core`` is the
            # standalone lane and stays reported separately.
            cand_ms = block["candidate"]["median_ms"]
            ctrl_ms = block["control"]["median_ms"]
            per_chunk[c]["cand"].append(cand_ms)
            per_chunk[c]["ctrl"].append(ctrl_ms)
            per_chunk[c]["geom"] = (case["q_len"], case["prefix_len"], case["flops"])
            core += cand_ms
            ctrl += ctrl_ms
            incl += case["inclusive"]["median_ms"]
            prep += case["preprocessing"]["median_ms"]
        sched_core_s.append(core * LAYERS / 1000)
        sched_ctrl_s.append(ctrl * LAYERS / 1000)
        sched_incl_s.append(incl * LAYERS / 1000)
        sched_prep_s.append(prep * LAYERS / 1000)
        sched_ratio.append(ctrl / core)

    total_flops = sum(per_chunk[c]["geom"][2] for c in chunk_ids) * LAYERS
    rows = []
    for c in chunk_ids:
        cand = per_chunk[c]["cand"]
        ctrlv = per_chunk[c]["ctrl"]
        ratios = [b / a for a, b in zip(cand, ctrlv)]
        lo, hi = _boot_ci(ratios, seed=100 + c)
        q, prefix, flops = per_chunk[c]["geom"]
        rows.append({
            "chunk": c, "q_len": q, "prefix_len": prefix,
            "candidate_ms_median": round(statistics.median(cand), 2),
            "candidate_ms_spread": [round(min(cand), 2), round(max(cand), 2)],
            "candidate_tflops": round(flops / statistics.median(cand) / 1e9, 1),
            "control_ms_median": round(statistics.median(ctrlv), 2),
            "control_tflops": round(flops / statistics.median(ctrlv) / 1e9, 1),
            "speedup_median": round(statistics.median(ratios), 3),
            "speedup_ci95_blocks": [round(lo, 3), round(hi, 3)],
        })

    lo, hi = _boot_ci(sched_ratio, seed=7)
    return {
        "blocks": len(blocks),
        "block_files": [pathlib.Path(p).name for p, _ in blocks],
        "resample_unit": "independent process block",
        "per_chunk": rows,
        "schedule_weighted": {
            "candidate_core_s_median": round(statistics.median(sched_core_s), 2),
            "candidate_core_s_spread": [round(min(sched_core_s), 2), round(max(sched_core_s), 2)],
            "candidate_core_tflops_median": round(
                total_flops / statistics.median(sched_core_s) / 1e12, 1),
            "candidate_inclusive_s_median": round(statistics.median(sched_incl_s), 2),
            "candidate_preprocessing_s_median": round(statistics.median(sched_prep_s), 2),
            "control_core_s_median": round(statistics.median(sched_ctrl_s), 2),
            "control_core_tflops_median": round(
                total_flops / statistics.median(sched_ctrl_s) / 1e12, 1),
            "speedup_median": round(statistics.median(sched_ratio), 3),
            "speedup_ci95_blocks": [round(lo, 3), round(hi, 3)],
        },
    }


def aggregate_compat(blocks: list[tuple[str, dict]]) -> dict:
    if not blocks:
        return {"blocks": 0}
    chunk_ids = sorted({case["chunk"] for _, doc in blocks for case in doc["cases"]})
    rows = []
    for c in chunk_ids:
        ms, clocks = [], []
        geom = None
        for _, doc in blocks:
            for case in doc["cases"]:
                if case["chunk"] == c:
                    ms.append(case["upstream_comparability"]["median_ms"])
                    clk = case.get("clocks_mhz") or {}
                    for key in ("observed_before_mhz", "observed_after_mhz"):
                        if clk.get(key):
                            clocks.append(clk[key])
                    geom = (case["q_len"], case["prefix_len"], case["flops"])
        rows.append({
            "chunk": c, "q_len": geom[0], "prefix_len": geom[1],
            "graph_ms_median": round(statistics.median(ms), 2),
            "graph_ms_spread": [round(min(ms), 2), round(max(ms), 2)],
            "tflops_median": round(geom[2] / statistics.median(ms) / 1e9, 1),
            "observed_clocks_mhz": sorted({int(x) for x in clocks if x}),
        })
    meas = blocks[0][1]["measurement"]
    return {
        "blocks": len(blocks),
        "protocol": {k: meas.get(k) for k in
                     ("graph_mode", "graph_warmup_replays", "graph_timed_replays",
                      "quantization", "clock_requested_mhz", "l2_policy")},
        "per_chunk": rows,
    }


def to_markdown(summary: dict) -> str:
    ctrl = summary["control"]
    sw = ctrl["schedule_weighted"]
    lines = [
        "# B6 campaign summary",
        "",
        f"- Control: FlashInfer BF16 paged prefill, same process, ABBA per case,"
        f" **{ctrl['blocks']} independent process blocks** (bootstrap unit).",
        f"- Schedule-weighted (14 chunks x 16 layers): candidate core "
        f"**{sw['candidate_core_s_median']} s ({sw['candidate_core_tflops_median']} TF/s)** "
        f"vs control **{sw['control_core_s_median']} s ({sw['control_core_tflops_median']} TF/s)** "
        f"-> speedup **{sw['speedup_median']}x**, 95% CI over blocks "
        f"[{sw['speedup_ci95_blocks'][0]}, {sw['speedup_ci95_blocks'][1]}].",
        "",
        "| chunk | q_len | prefix | cand ms | cand TF/s | ctrl ms | ctrl TF/s | speedup | CI95 (blocks) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in ctrl["per_chunk"]:
        lines.append(
            f"| {row['chunk']} | {row['q_len']} | {row['prefix_len']} | "
            f"{row['candidate_ms_median']} | {row['candidate_tflops']} | "
            f"{row['control_ms_median']} | {row['control_tflops']} | "
            f"{row['speedup_median']}x | [{row['speedup_ci95_blocks'][0]}, "
            f"{row['speedup_ci95_blocks'][1]}] |")
    compat = summary.get("compat") or {}
    if compat.get("blocks"):
        proto = compat["protocol"]
        lines += ["", f"## Upstream-comparability lane ({compat['blocks']} blocks, "
                  f"{proto['graph_warmup_replays']}/{proto['graph_timed_replays']} "
                  f"replays, {proto['graph_mode']}, quantization {proto['quantization']})", "",
                  "| chunk | q_len | prefix | graph ms | TF/s | observed clocks MHz |",
                  "|---|---|---|---|---|---|"]
        for row in compat["per_chunk"]:
            lines.append(f"| {row['chunk']} | {row['q_len']} | {row['prefix_len']} | "
                         f"{row['graph_ms_median']} | {row['tflops_median']} | "
                         f"{row['observed_clocks_mhz']} |")
    lines += ["", "Caveats: single allocation (second-allocation confirmation pending); "
              "warm-L2 eager for the control lane; the comparability lane follows "
              "FlashInfer PR #4502's protocol and excludes quantization."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--control", required=True, help="glob for control-block JSONs")
    parser.add_argument("--compat", default=None, help="glob for comparability-block JSONs")
    parser.add_argument("--out", required=True, help="output path stem (.json/.md appended)")
    args = parser.parse_args(argv)

    summary = {"control": aggregate_control(_load(args.control))}
    if args.compat:
        summary["compat"] = aggregate_compat(_load(args.compat))

    out = pathlib.Path(args.out)
    out.with_suffix(".json").write_text(json.dumps(summary, indent=1) + "\n")
    out.with_suffix(".md").write_text(to_markdown(summary))
    print(f"wrote {out.with_suffix('.json')} and .md")
    sw = summary["control"]["schedule_weighted"]
    print(f"schedule-weighted speedup {sw['speedup_median']}x "
          f"CI {sw['speedup_ci95_blocks']} over {summary['control']['blocks']} blocks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
