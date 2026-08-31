#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Q1 on the PUBLIC lane: attention-output fidelity of the real kernel at depth.

This is the engine-free, offline numerical boundary: it never loads a model,
so it fits on a 24 GiB card at the full production depth of 446,335 tokens even
though the target profile excludes the 446k *workload* -- that exclusion is about
sixteen layers of paged K/V resident at once (~27.2 GiB), and this processes one
case at a time (K+V for one case at 446k is 1.83 GiB).

For each fixture case it runs, in one process on the same tensors:

    ref   fp32 attention on the fp32 master                     (the reference)
    ctl   bf16 inputs, bf16 P, bf16 output                       (the ANCHOR)
    ctl2  fp32 attention on bf16-rounded inputs                  (input rounding)
    cand  attn_kernel_lab.ops.prefill_extend                     (the candidate)

and reports every metric four ways -- mean, worst layer, worst head, worst ROW --
each against the anchor evaluated at that same slice.  The anchor is the
sensitivity floor: FlashAttention's own suite asserts against exactly this
quantity (the same reference in the target dtype with reordered ops), and the
repository's recorded 0.31-0.45% implementation-swap band is what it reproduces.
Deltas are read as ratios to it, never as absolutes.

Every metric comes from ``probes/quality/pertensor_vs_finegrained.py`` through
``quality/metrics.py``.  Nothing here defines a metric.

WHAT THIS LANE IS FOR, AND WHAT IT IS NOT
-----------------------------------------
The fixtures are **public and redistributable by construction**: seeded synthetic
distributions with no restricted inputs, real activations, or model weights,
reproducible from the recorded seeds alone. They are the five distributions the
instrument built
to isolate the failure modes the operator's transforms exist for -- iid Gaussian,
Student-t(3) heavy tails, a RoPE-like shared channel offset, a high-variance V
channel, and a massive-activation V channel -- which is a considerably sharper
test than the ``torch.randn`` the upstream bar is set at, and considerably weaker
evidence than a real capture.

So: the *method* travels from this lane and the *production numbers* do not.
Absolute levels here are not transferable to real activations (the instrument's
own ``v_massive`` row exists to show how far the denominator moves them).  The
private lane on the production target owns the production numbers, and the
envelope this run seeds stays a DRAFT until a production-tier run confirms it.

Usage
-----
    python3 quality/q1_public.py                       # full 446,335-token depth
    python3 quality/q1_public.py --quick               # smoke at 4,096
    python3 quality/q1_public.py --no-write            # measure, write nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import platform
import subprocess
import sys
import time

# Before torch, for the reason in quality/metrics.py and contract §3.1.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import fidelity_records as rec  # noqa: E402
import metrics as qmetrics  # noqa: E402

FULL_DEPTH = 446_335
DEFAULT_DISTS = ["gaussian", "heavy_t3", "rope_like", "v_outlier", "v_massive"]

#: How far the real kernel may sit from the contract's vectorised restatement
#: before the run is attributed to a different operator rather than to
#: quantization loss -- measured in IMPLEMENTATION-SWAP UNITS, the unit the whole
#: record is anchored in.  2 is FlashAttention's own forward-output multiplier.
#: Measured: 0.01 units at 4,096 tokens, 0.70-0.78 at 446,335.
FORM_AGREEMENT_MAX = 2.0

FIXTURE_SET_ID = "pub-synth-d256-446k-v1"
SCORECARD_ID = "candidate-zero-pub-synth-446k-attention-output"
ENVELOPE_ID = "candidate-zero-v1"

#: The published release whose sources this run is measuring, IF the loaded
#: kernel's source build id matches it.  Checked, never assumed: the JIT path
#: builds whatever is in the working tree.
PUBLISHED_RELEASE = {
    "release_id": "d256-int8-fp8-v0.3.0",
    "build_id": "9df2efa339d01f4279b858778626ecad2a7ff94e99cd96820da13035068e116c",
}

SLICE_SEMANTICS = {
    "layer": "fixture_case",
    "row": "query_row",
    "note": (
        "This lane has no model, so its layer axis is the fixture case, not a "
        "model layer. The worst-layer mandate is preserved -- the worst case in "
        "the population is reported and never averaged away -- but a worst-layer "
        "number here is a statement about a distribution, not about layer 31 of "
        "anything. The private lane sets layer=model_layer over all sixteen "
        "full-attention layers."
    ),
}

NOT_CLAIMED = [
    "Not a model-quality statement. This is the attention-output boundary on "
    "captured or seeded activations. Downstream task outputs, logprobs and task "
    "accuracy are separate boundaries and are not measured here; a metric that "
    "moves here has not thereby been shown to move anything downstream.",
    "Not real activations. Seeded synthetic distributions chosen to isolate the "
    "failure modes the operator's transforms exist for. Absolute levels do not "
    "transfer to a real capture; the private lane owns the production numbers.",
    "Not a performance result. No timing appears in this record, and the RTX "
    "4090 is a development-authority target that cannot author one.",
    "Not 'quality-neutral' and not 'lossless', in any form. upstream/CLAIMS.md "
    "forbids that wording and this lane does not establish it.",
    "Not a general-model claim. One contract version, one geometry, one "
    "synthetic input distribution, one candidate.",
    "Not a run of the d256-24x4-446k workload. This fixture set borrows "
    "that workload's DEPTH so the depth axis is comparable; it does not replay "
    "its data, and the 4090 target excludes that workload for a memory reason "
    "this lane does not encounter.",
]

HARD_FAILS = [
    "nan_count > 0 or inf_count > 0 (the stale-V 0 x NaN poisoning class, an "
    "actual historical bug).",
    "cos_sim worst-head < 0.99. A floor, not a target: SageAttention's "
    "catastrophic INT8-PV case is 56.40% worst-layer cosine and their accepted "
    "configurations sit above 99%.",
    "An Oracle A contract failure, which is an implementation defect and is "
    "never waivable as quantization error.",
    "Any boot-gate or dispatch-counter deviation, for the engine-bound rungs.",
]


# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------


def _nvidia_smi() -> dict:
    query = "name,driver_version,uuid,temperature.gpu"
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
        return dict(zip(query.split(","), [p.strip() for p in out.split(",")]))
    except Exception as exc:  # noqa: BLE001 -- record the absence, never fail the run
        return {"error": f"nvidia-smi unavailable: {exc}"}


def _environment() -> dict:
    import torch

    props = torch.cuda.get_device_properties(0)
    smi = _nvidia_smi()
    quant_src = (ROOT / "src" / "attn_kernel_lab" / "quant.py").read_bytes()
    temp = smi.get("temperature.gpu")
    return {
        "device": props.name,
        "capability": f"sm_{props.major}{props.minor}",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "driver": smi.get("driver_version"),
        # Never the raw GPU UUID: a physical-card serial is a machine
        # fingerprint, and this record publishes. docs/hardware-labels.md
        # names the reviewed aliases; default to the local dev card's.
        "gpu_uuid_or_run_id": os.environ.get(
            "ATTN_KERNEL_LAB_GPU_LABEL", "local-dev-gpu"
        ),
        "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "observed_temperature_c": float(temp) if temp and temp.isdigit() else None,
        "quant_py_sha256": hashlib.sha256(quant_src).hexdigest(),
        "donor_sha256": qmetrics.donor_sha256(),
        "harness_sha256": hashlib.sha256(
            pathlib.Path(__file__).read_bytes() + (HERE / "metrics.py").read_bytes()
        ).hexdigest(),
        "python": platform.python_version(),
    }


def _subject() -> dict:
    from attn_kernel_lab import kernel as kernel_mod

    build_id = kernel_mod.source_build_id()
    matches = build_id == PUBLISHED_RELEASE["build_id"]
    # The loader has a precedence order (an installed wheel over a JIT build), so
    # which side of it won is a fact about the measured binary, not an inference
    # from the source digest. Read defensively: the loader is A2's ground and this
    # record must not fail because its shape moved.
    reporter = getattr(kernel_mod, "loaded_from", None)
    try:
        loaded_from = reporter() if callable(reporter) else None
    except Exception:  # noqa: BLE001 -- absence of the fact, never a failed run
        loaded_from = None
    return {
        "impl": "attn_kernel_lab.ops.prefill_extend",
        "loaded_from": loaded_from,
        # Named only when the loaded sources hash to the published release's
        # build id. The JIT path builds the working tree, and a scorecard that
        # asserted a release_id it had not checked would be evidence about a
        # different artifact than the one it names.
        "release_id": PUBLISHED_RELEASE["release_id"] if matches else None,
        "artifact_manifest_sha256": None,
        "source_build_id": build_id,
        "mode": "declared_default_v1",
        "env_switches": {
            "qk_i8": True,
            "rotate": True,
            "center_k": True,
            "return_lse": False,
            # Contract §7 leaves the status of these open; they change numerics
            # and survive into production, so their absence is recorded too.
            "SGLANG_FP8_PREFILL_QK": os.environ.get("SGLANG_FP8_PREFILL_QK"),
            "SGLANG_FP8_PREFILL_K_CENTER": os.environ.get("SGLANG_FP8_PREFILL_K_CENTER"),
            "SGLANG_FP8_PREFILL_BF16_HEADS": os.environ.get("SGLANG_FP8_PREFILL_BF16_HEADS"),
            "SGLANG_FP8_PREFILL_ZERO_WS": os.environ.get("SGLANG_FP8_PREFILL_ZERO_WS"),
            "SGLANG_FP8_PREFILL_MIN_TOKENS": os.environ.get("SGLANG_FP8_PREFILL_MIN_TOKENS"),
        },
    }


# --------------------------------------------------------------------------
# one case
# --------------------------------------------------------------------------


def _worst(values: list, direction: str, key):
    """Index of the worst entry; `worst` means largest error or smallest similarity."""
    pick = min if direction == "higher_is_better" else max
    return pick(range(len(values)), key=lambda i: key(values[i]))


def run_case(
    layer: int,
    dist: str,
    depth: int,
    *,
    hash_fixtures: bool,
    repeat_check: bool,
    oracle_form: bool,
    verbose: bool = True,
) -> dict:
    """Measure one fixture case and return its per-slice raw numbers."""
    import torch

    started = time.perf_counter()
    case = qmetrics.Case(dist, depth)
    ref = case.ref

    cand = case.candidate()
    bitexact = None
    if repeat_check:
        # The plan's harness floor: a same-config repeat must be bit-identical.
        # The offline path is deterministic by construction (it is the premise of
        # cmp_live.py's BITEXACT assertion), so any movement here is a harness
        # defect and the run is void, not noisy.
        again = case.candidate()
        bitexact = bool(torch.equal(cand, again))
        del again

    control = case.control_implementation_swap()
    secondary = case.control_input_rounding()

    cand_all = qmetrics.metrics_all(cand, ref)
    ctl_all = qmetrics.metrics_all(control, ref)
    sec_all = qmetrics.metrics_all(secondary, ref)
    cand_heads = qmetrics.metrics_per_head(cand, ref)
    cand_rows = qmetrics.metrics_per_row(cand, ref)

    slices: dict[str, dict] = {}
    for name in rec.GATE_METRICS:
        direction = rec.METRIC_FORM[name][0]
        key = rec.DONOR_KEY[name]
        head = _worst(cand_heads, direction, lambda m: m[key])
        flat = [(r, h) for r in range(len(cand_rows)) for h in range(len(cand_rows[0]))]
        worst_flat = _worst(flat, direction, lambda rh: cand_rows[rh[0]][rh[1]][key])
        row, row_head = flat[worst_flat]
        slices[name] = {
            "layer_value": cand_all[key],
            "layer_control": ctl_all[key],
            "head": head,
            "head_value": cand_heads[head][key],
            "head_control": qmetrics.metrics_at_head(control, ref, head)[key],
            "row": row,
            "row_head": row_head,
            "row_value": cand_rows[row][row_head][key],
            "row_control": qmetrics.metrics_at_row(control, ref, row, row_head)[key],
        }

    fixture_hashes = {}
    if hash_fixtures:
        fixture_hashes = {
            "q_sha256": qmetrics.sha256_tensor(case.q_bf),
            "k_sha256": qmetrics.sha256_tensor(case.k_bf),
            "v_sha256": qmetrics.sha256_tensor(case.v_bf),
        }

    result = {
        "layer": layer,
        "name": dist,
        "doc": case.doc,
        "seed": case.seed,
        "depth": depth,
        "prefix": case.prefix,
        "q_len": case.t_rows,
        "slices": slices,
        "norm_ratio": cand_all["norm_ratio"],
        "norm_ratio_control": ctl_all["norm_ratio"],
        "nan_count": cand_all["nan_count"],
        "inf_count": cand_all["inf_count"],
        "control_input_rounding": {k: sec_all[rec.DONOR_KEY[k]] for k in rec.GATE_METRICS},
        "bitexact_repeat": bitexact,
        "fixture_hashes": fixture_hashes,
        "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
        "seconds": time.perf_counter() - started,
    }

    if verbose:
        row_rel = slices["row_rel_l2"]
        print(
            f"    row_rel_l2 mean {row_rel['layer_value']:.3%} "
            f"(anchor {row_rel['layer_control']:.3%}, "
            f"{row_rel['layer_value'] / row_rel['layer_control']:.2f}x) | "
            f"worst row {row_rel['row_value']:.3%} @ h{row_rel['row_head']} r{row_rel['row']} | "
            f"cos worst row {slices['cos_sim']['row_value']:.6f} | "
            f"norm_ratio {cand_all['norm_ratio']:.5f} "
            f"({result['seconds']:.1f}s, {result['peak_mem_gib']:.2f} GiB)",
            flush=True,
        )

    case.free()
    del cand, control, secondary, ref
    torch.cuda.empty_cache()

    if oracle_form:
        # The offline analogue of cmp_live.py's BITEXACT assertion, and the reason
        # this lane can attribute a number to the kernel at all: re-run the SAME
        # case through the donor's simulated pipeline -- quant.py's own operands
        # plus the online running-tile-max E4M3 P rounding at 448*r_t, the form
        # --check-oracle pins against oracle_a.attention -- and compare. The
        # candidate and the contract's vectorised restatement have to agree, or
        # the kernel is not executing the operator the record describes.
        module = qmetrics.donor()
        cell = module.run_cell(dist, depth, module.DISTRIBUTIONS[dist], verbose=False)
        sim = cell["schemes"]["lab_p"]
        anchor = cell["schemes"]["bf16_arith"]
        result["oracle_form"] = {
            "lab_p": {k: sim[rec.DONOR_KEY[k]] for k in rec.GATE_METRICS},
            "lab_p_row_rel_l2_max": sim["row_rel_l2_max"],
            "candidate_vs_lab_p_rel": abs(
                slices["row_rel_l2"]["layer_value"] - sim["row_rel_l2_mean"]
            )
            / sim["row_rel_l2_mean"],
            # The number that means something. A bare relative disagreement has no
            # scale: 180% of v_massive's 5e-5 and 7% of gaussian's 4.6e-2 are the
            # same absolute quantity. Expressed in implementation-swap units -- the
            # unit the whole record is anchored in -- both read as ~0.7.
            "candidate_vs_lab_p_in_anchor_units": abs(
                slices["row_rel_l2"]["layer_value"] - sim["row_rel_l2_mean"]
            )
            / slices["row_rel_l2"]["layer_control"],
            "anchor_vs_donor_rel": abs(
                slices["row_rel_l2"]["layer_control"] - anchor["row_rel_l2_mean"]
            )
            / anchor["row_rel_l2_mean"],
        }
        if verbose:
            of = result["oracle_form"]
            print(
                f"    contract form: candidate {slices['row_rel_l2']['layer_value']:.4%} vs "
                f"simulated lab_p {sim['row_rel_l2_mean']:.4%} "
                f"({of['candidate_vs_lab_p_in_anchor_units']:.2f} anchor units, "
                f"{of['candidate_vs_lab_p_rel']:.2%} relative); anchor reproduces the "
                f"instrument's yardstick to {of['anchor_vs_donor_rel']:.2e}",
                flush=True,
            )
        torch.cuda.empty_cache()
    return result


# --------------------------------------------------------------------------
# record assembly
# --------------------------------------------------------------------------


def build_fixture_set(cases: list[dict], depth: int, hash_fixtures: bool) -> dict:
    module = qmetrics.donor()
    record = {
        "schema_version": 1,
        "fixture_set_id": FIXTURE_SET_ID,
        "lane": "public_synth",
        "redistributable": True,
        "fixture_set_sha256": "",
        "provenance": {
            "generator": "quality/q1_public.py + quality/metrics.py",
            "generator_sha256": hashlib.sha256(
                pathlib.Path(__file__).read_bytes() + (HERE / "metrics.py").read_bytes()
            ).hexdigest(),
            "donor_instrument": "probes/quality/pertensor_vs_finegrained.py",
            "donor_sha256": qmetrics.donor_sha256(),
            "media": (
                "NONE: seeded synthetic tensors with no restricted inputs, real "
                "activations, or model weights. Reproducible from the seeds "
                "recorded per case."
            ),
            "model": None,
            "engine": None,
            "workload_profile": "d256-24x4-446k",
            "workload_relationship": (
                "GEOMETRY ONLY. The depth 446,335 and the 24:4 / D256 / page-size-1 "
                "shape are taken from that workload profile so the depth axis is "
                "comparable with the production numbers. No input example or "
                "schedule from that workload is used, and this is not a run of it."
            ),
            "distributions": {
                name: {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}
                for name, cfg in module.DISTRIBUTIONS.items()
            },
            "notes": [
                "Synthetic fixtures are weaker evidence than a real capture and the "
                "plan says so; they are chosen here because the public lane must be "
                "redistributable and must contain nothing derived from restricted "
                "inputs. Each "
                "distribution isolates a failure mode one of the operator's "
                "transforms exists for, which is a sharper test than the torch.randn "
                "the upstream bar is set at, and not a substitute for Lane P.",
                "K and V are drawn in fixed 32,768-row blocks with per-block seeds, "
                "so a shallow case's context is a bit-exact prefix of a deep one's "
                "and depth is the only thing that varies along the depth axis.",
            ],
        },
        "geometry": {
            "head_dim": module.HEAD_DIM,
            "q_heads": module.Q_HEADS,
            "kv_heads": module.KV_HEADS,
            "page_size": 1,
            "mask": "bottom_right_causal",
            "mode": "extend",
            "q_len": module.T_ROWS,
            "sm_scale": module.SM_SCALE,
            "kv_dtype": "bfloat16",
            "gen_chunk_rows": module.GEN_CHUNK,
        },
        "slice_semantics": SLICE_SEMANTICS,
        "cases": [
            {
                "layer": case["layer"],
                "name": case["name"],
                "distribution": case["name"],
                "depth": case["depth"],
                "prefix": case["prefix"],
                "seed": case["seed"],
                "doc": case["doc"],
                **(case["fixture_hashes"] if hash_fixtures else {}),
            }
            for case in cases
        ],
        "notes": [
            f"Depth {depth:,} tokens per case; q_len {module.T_ROWS} query rows as the "
            "tail of the context, which is the extend geometry the operator declares.",
        ],
    }
    return rec.seal(record, "fixture_set_sha256")


def build_scorecard(
    cases: list[dict],
    fixture_set: dict,
    controls: list[dict],
    environment: dict,
    subject: dict,
    status: str,
) -> dict:
    metrics: dict = {}
    for name in rec.GATE_METRICS:
        direction = rec.METRIC_FORM[name][0]
        basis = rec.METRIC_FORM[name][1]
        per_layer = [
            rec.slice_entry(
                basis,
                c["slices"][name]["layer_value"],
                c["slices"][name]["layer_control"],
                layer=c["layer"],
                layer_name=c["name"],
            )
            for c in cases
        ]
        heads = [
            rec.slice_entry(
                basis,
                c["slices"][name]["head_value"],
                c["slices"][name]["head_control"],
                layer=c["layer"],
                layer_name=c["name"],
                head=c["slices"][name]["head"],
            )
            for c in cases
        ]
        rows = [
            rec.slice_entry(
                basis,
                c["slices"][name]["row_value"],
                c["slices"][name]["row_control"],
                layer=c["layer"],
                layer_name=c["name"],
                head=c["slices"][name]["row_head"],
                row=c["slices"][name]["row"],
            )
            for c in cases
        ]
        worst_layer = per_layer[_worst(per_layer, direction, lambda s: s["value"])]
        worst_head = heads[_worst(heads, direction, lambda s: s["value"])]
        worst_row = rows[_worst(rows, direction, lambda s: s["value"])]
        # The aggregate is the unweighted mean over the case population, which is
        # the population the worst-layer mandate is defined over.
        mean = sum(s["value"] for s in per_layer) / len(per_layer)
        mean_control = sum(s["control"] for s in per_layer) / len(per_layer)
        metrics[name] = rec.metric_block(
            name,
            mean=mean,
            mean_control=mean_control,
            per_layer=per_layer,
            worst_layer=worst_layer,
            worst_head=worst_head,
            worst_row=worst_row,
        )

    metrics["norm_ratio"] = rec.reported_block(
        [
            {
                "layer": c["layer"],
                "layer_name": c["name"],
                "value": c["norm_ratio"],
                "control": c["norm_ratio_control"],
            }
            for c in cases
        ],
        note=(
            "Reported, never gated (addendum 2026-08-30 item 2). The plan's §6.3 "
            "hard gate of [0.98, 1.02] is withdrawn by that addendum: the "
            "instrument measured a systematic E4M3 output-norm drift with depth "
            "that never leaves the band, so the gate would never fire while the "
            "trend line is informative. Read it as the cheap tell for absent "
            "K/V centering, not as a threshold."
        ),
    )
    metrics["nan_count"] = sum(c["nan_count"] for c in cases)
    metrics["inf_count"] = sum(c["inf_count"] for c in cases)

    return {
        "schema_version": 1,
        "kind": "fidelity_scorecard",
        "scorecard_id": SCORECARD_ID,
        "boundary": "attention_output",
        "status": status,
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "subject": subject,
        "reference": {
            "role": "fp32_reference",
            "impl": "fp32 attention on the fp32 master, explicit bottom-right causal mask",
            "note": (
                "TF32 is off and the cuBLAS workspace is pinned to the contract §3.1 "
                "constant; a TF32-contaminated fp32 reference is not one, and the "
                "rotation is not reproducible without the pin."
            ),
        },
        "controls": controls,
        "fixture_set": {
            "id": fixture_set["fixture_set_id"],
            "fixture_set_sha256": fixture_set["fixture_set_sha256"],
            "lane": fixture_set["lane"],
            "layers": len(cases),
            "q_heads": fixture_set["geometry"]["q_heads"],
            "kv_heads": fixture_set["geometry"]["kv_heads"],
            "head_dim": fixture_set["geometry"]["head_dim"],
            "q_len": fixture_set["geometry"]["q_len"],
            "depths": sorted({c["depth"] for c in cases}),
            "slice_semantics": SLICE_SEMANTICS,
        },
        "environment": environment,
        "metrics": metrics,
        "notes": [
            "Every ratio is against the bf16 implementation swap measured on the "
            "same tensors in the same process, evaluated at the SAME slice as the "
            "value it anchors -- not against the control's own worst slice.",
            "Aggregation is over the fixture-case population; see "
            "fixture_set.slice_semantics for what a layer is in this lane.",
            "Wall-clock and memory figures are absent by design. No number in this "
            "record is a performance claim.",
            "The candidate's contract conformance is Oracle A's lane "
            "(tests/kernel/test_oracle_a.py) and a failure there is an "
            "implementation defect, never a quality result.",
        ],
        "not_claimed": NOT_CLAIMED,
    }


def build_envelope(scorecard: dict, scorecard_digest: dict, fixture_set: dict) -> dict:
    metrics = {}
    for name in rec.GATE_METRICS:
        block = scorecard["metrics"][name]
        metrics[name] = {
            "direction": block["direction"],
            "ratio_basis": block["ratio_basis"],
            "mean_value": block["mean"],
            "mean_control": block["control"],
            "mean_ratio": block["ratio"],
            "worst_layer_ratio": block["worst_layer"]["ratio"],
            "worst_head_ratio": block["worst_head"]["ratio"],
            "worst_row_ratio": block["worst_row"]["ratio"],
            "worst_layer_value": block["worst_layer"]["value"],
            "worst_head_value": block["worst_head"]["value"],
            "worst_row_value": block["worst_row"]["value"],
            "ratio_band": None,
            "saturated": block["saturated"],
        }
    return {
        "schema_version": 1,
        "kind": "fidelity_envelope",
        "envelope_id": ENVELOPE_ID,
        "status": "draft",
        "calibrated": False,
        "boundary": "attention_output",
        "recorded_utc": scorecard["recorded_utc"],
        "supersedes": None,
        "supersede_reason": None,
        "subject": {
            "impl": scorecard["subject"]["impl"],
            "release_id": scorecard["subject"]["release_id"],
            "artifact_manifest_sha256": scorecard["subject"]["artifact_manifest_sha256"],
            "source_build_id": scorecard["subject"]["source_build_id"],
            "loaded_from": scorecard["subject"]["loaded_from"],
            "mode": scorecard["subject"]["mode"],
            "env_switches": scorecard["subject"]["env_switches"],
        },
        "fixture_sets": [
            {
                "id": fixture_set["fixture_set_id"],
                "fixture_set_sha256": fixture_set["fixture_set_sha256"],
                "lane": fixture_set["lane"],
            }
        ],
        "source_scorecards": [
            {
                "scorecard_id": scorecard["scorecard_id"],
                "scorecard_sha256": scorecard_digest,
                "path": f"quality/scorecards/{scorecard['scorecard_id']}.json",
            }
        ],
        "decision_rule": {
            "aggregate_tolerance_pct": 10.0,
            "worst_slice_tolerance_pct": 0.0,
            "worst_slice_enforced": False,
            "hard_fails": HARD_FAILS,
        },
        "metrics": metrics,
        "notes": [
            "DRAFT. This is candidate zero measured on the PUBLIC synthetic lane on "
            "a development-authority 4090. It is a place for the numbers to live and "
            "a proof that the harness runs at full depth; it is not the frozen "
            "non-inferiority band. Freezing needs a production-tier confirmation on the "
            "private real-activation lane and an explicit reviewed decision, exactly "
            "like publishing a release.",
            "calibrated=false, so the worst-slice thresholds are reported and NOT "
            "enforced: the reproducibility band needs three independent fixture sets "
            "and only one exists. Same posture as promotion_thresholds.calibrated in "
            "the target profiles.",
            "Sampling error is not the limiting uncertainty here -- the aggregate is "
            "a reduction over ~4x10^5 row observations per case. Content "
            "generalisation is: one synthetic fixture family, and its absolute "
            "levels do not transfer to real activations.",
            "norm_ratio is deliberately absent from hard_fails. The plan's §6.3 "
            "[0.98, 1.02] gate is withdrawn by the 2026-08-30 addendum, which found "
            "a systematic drift that never leaves the band; it is tracked as a "
            "reported trend line in the scorecard instead.",
            "The 0.99 cosine floor stays defined on the worst HEAD, where the plan "
            "put it. The addendum made worst-ROW a mandatory reported slice; it did "
            "not move the floor there, and moving it would be wrong here: candidate "
            "zero's worst row is cosine 0.954 on synthetic Student-t(3) keys at "
            "446k, which is the tail this fixture family was built to provoke and "
            "not a statement about real activations. Whether a real-capture worst "
            "row behaves like that is a Lane P question and is the single most "
            "important thing the production tier will answer.",
        ],
        "not_claimed": NOT_CLAIMED,
    }


# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--depth", type=int, default=FULL_DEPTH)
    parser.add_argument("--dists", nargs="+", default=DEFAULT_DISTS)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="smoke: gaussian + rope_like at 4,096 tokens, no hashes",
    )
    parser.add_argument(
        "--no-hash",
        action="store_true",
        help="skip the fixture tensor digests (they cost ~3 s per case)",
    )
    parser.add_argument(
        "--no-oracle",
        action="store_true",
        help="skip the donor's P-rounding cross-check against oracle_a",
    )
    parser.add_argument(
        "--no-oracle-form",
        action="store_true",
        help="skip re-running each case through the donor's simulated "
        "pipeline (the candidate-vs-contract-form control)",
    )
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--status", default="draft", choices=["draft", "final"])
    parser.add_argument("--out-dir", default=str(HERE))
    parser.add_argument("--envelope-dir", default=str(ROOT / "promotion" / "envelopes"))
    args = parser.parse_args(argv)

    import torch

    if not torch.cuda.is_available():
        print("no CUDA device: Q1 needs one", file=sys.stderr)
        return 2
    # A TF32-contaminated fp32 reference is not one (plan §4.2).
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if args.quick:
        args.depth = 4096
        args.dists = ["gaussian", "rope_like"]
        args.no_hash = True

    started = time.perf_counter()
    print(f"[Q1-public] {len(args.dists)} case(s) at depth {args.depth:,}", flush=True)

    oracle = None
    if not args.no_oracle:
        # The donor's own cross-check that its P-rounding form -- the online
        # running-tile-max rounding, not the final-max form, which disagrees with
        # the contract by an order of magnitude -- reproduces oracle_a.attention.
        # Run here so the licence for that form is recorded in THIS environment.
        oracle = qmetrics.donor().check_oracle()

    cases = []
    for layer, dist in enumerate(args.dists):
        print(f"  [{dist}] layer {layer}", flush=True)
        cases.append(
            run_case(
                layer,
                dist,
                args.depth,
                hash_fixtures=not args.no_hash,
                repeat_check=(layer == 0),
                oracle_form=not args.no_oracle_form,
            )
        )

    bitexact = [c["bitexact_repeat"] for c in cases if c["bitexact_repeat"] is not None]
    controls = [
        {
            "role": "implementation_swap",
            "impl": "bf16 inputs, bf16 P, bf16 output (the donor's bf16_arith scheme)",
            "primary": True,
            "note": (
                "The same reference computed in the target dtype with reordered ops "
                "-- FlashAttention's own out_pt. Every metric block's `control` is "
                "this quantity at that block's slice."
            ),
        },
        {
            "role": "input_rounding",
            "impl": "fp32 attention on bf16-rounded inputs (the donor's bf16 scheme)",
            "primary": False,
            "note": (
                "Input rounding alone, kept because the repository's recorded "
                "0.31-0.45% attention-output control band is in this unit. Per-case "
                "values are in the run log, not in a metric block, so there is one "
                "and only one anchor for the ratios."
            ),
        },
        {
            "role": "harness_floor",
            "assertion": "candidate repeat is bit-identical (same config, same process)",
            "result": (
                "pass" if bitexact and all(bitexact) else ("not_run" if not bitexact else "fail")
            ),
            "note": (
                "The offline path is deterministic by construction, so any movement "
                "here is a harness defect and voids the run rather than widening it."
            ),
        },
    ]
    if oracle is not None:
        controls.append(
            {
                "role": "oracle",
                "assertion": "donor P-rounding form (online running-tile-max) vs oracle_a.attention",
                "result": "pass",
                "value": oracle["max_row_rel_l2_vs_oracle_a"],
                "note": (
                    "Floor is oracle_a's own bf16 output cast, not zero. This licenses "
                    f"the P-rounding form the instrument uses; shape {oracle['shape']}."
                ),
            }
        )

    forms = [c["oracle_form"] for c in cases if "oracle_form" in c]
    if forms:
        worst_form = max(f["candidate_vs_lab_p_in_anchor_units"] for f in forms)
        worst_rel = max(f["candidate_vs_lab_p_rel"] for f in forms)
        worst_anchor = max(f["anchor_vs_donor_rel"] for f in forms)
        controls.append(
            {
                "role": "oracle",
                "assertion": (
                    "candidate vs the donor's simulated pipeline (lab_p: quant.py operands "
                    "plus the oracle-checked online E4M3 P rounding at 448*r_t), same "
                    "seeds, same case; max |candidate - simulation| on row_rel_l2 mean, in "
                    "implementation-swap units"
                ),
                "result": "pass" if worst_form <= FORM_AGREEMENT_MAX else "fail",
                "value": worst_form,
                "note": (
                    "The offline analogue of the live-vs-offline bit-exactness assertion: "
                    "it separates 'the kernel is lossy' from 'the kernel is computing "
                    "something else'. Chain: candidate ~ lab_p ~ oracle_a.attention. Not "
                    "bit-exact by construction -- the kernel accumulates in the tensor "
                    "cores, casts its output to bf16 and runs 6,974 online-softmax tiles "
                    "at this depth, while the simulation is fp32 throughout -- so the bar "
                    f"is {FORM_AGREEMENT_MAX:.0f} implementation-swap units, which is "
                    "FlashAttention's own forward-output multiplier. Raw relative "
                    f"disagreement at its worst is {worst_rel:.2%}; that number is scale-free "
                    "and misleading where the base is ~5e-5 (v_massive), which is why the "
                    "anchor unit is what is gated. OBSERVATION, not a defect: this gap is "
                    "depth-dependent -- 0.01 anchor units at 4,096 tokens, 0.7-0.8 at "
                    "446,335 -- i.e. the kernel's implementation adds error the fp32 "
                    "restatement of its own contract does not predict, and the amount grows "
                    "with context. Attributing it (output cast vs online rescale over ~7k "
                    "tiles) is the session owner's call and needs the private lane to say "
                    "whether it matters."
                ),
            }
        )
        controls.append(
            {
                "role": "harness_floor",
                "assertion": (
                    "the implementation-swap anchor reproduces the instrument's own bf16 "
                    "yardstick on the same case; max relative difference"
                ),
                "result": "pass" if worst_anchor <= 1e-6 else "fail",
                "value": worst_anchor,
                "note": (
                    "Same code, same seeds, so this must be zero to floating point. It is "
                    "here because a silently mis-wired anchor would rescale every ratio in "
                    "the record and nothing else would notice."
                ),
            }
        )

    fixture_set = build_fixture_set(cases, args.depth, hash_fixtures=not args.no_hash)
    scorecard = build_scorecard(
        cases, fixture_set, controls, _environment(), _subject(), args.status
    )
    if forms:
        scorecard["notes"].append(
            "OBSERVATION for the reviewer, recorded rather than resolved: the "
            "candidate sits "
            + ", ".join(
                f"{c['name']} {c['oracle_form']['candidate_vs_lab_p_in_anchor_units']:.2f}"
                for c in cases
                if "oracle_form" in c
            )
            + " implementation-swap units above the fp32 simulation of its own "
            "contract (quality/runs/*.json has the raw pairs). The same comparison "
            "at 4,096 tokens is ~0.01 units, so this is a depth-dependent property "
            "of the implementation -- output cast, or accumulation across ~7k "
            "online-softmax tiles -- and not of the contract. It does not move any "
            "gate metric here, because every metric is measured against the kernel, "
            "not against the simulation; it is recorded because a later candidate "
            "that closed this gap would look better for a reason unrelated to "
            "quantization."
        )
    digest = rec.canonical_digest(scorecard)
    envelope = build_envelope(scorecard, digest, fixture_set)

    elapsed = time.perf_counter() - started
    row = scorecard["metrics"]["row_rel_l2"]
    cos = scorecard["metrics"]["cos_sim"]
    print(
        f"\n[Q1-public] {len(cases)} case(s) in {elapsed:.1f}s\n"
        f"  row_rel_l2   mean {row['mean']:.4%}  anchor {row['control']:.4%}  "
        f"ratio {row['ratio']:.2f}x  saturated={row['saturated']}\n"
        f"  row_rel_l2   worst-layer {row['worst_layer']['value']:.4%} "
        f"({row['worst_layer']['layer_name']}, {row['worst_layer']['ratio']:.2f}x)  "
        f"worst-head {row['worst_head']['value']:.4%} "
        f"({row['worst_head']['layer_name']} h{row['worst_head']['head']}, "
        f"{row['worst_head']['ratio']:.2f}x)  "
        f"worst-ROW {row['worst_row']['value']:.4%} "
        f"({row['worst_row']['layer_name']} h{row['worst_row']['head']} "
        f"r{row['worst_row']['row']}, {row['worst_row']['ratio']:.2f}x)\n"
        f"  cos_sim      mean {cos['mean']:.6f}  worst-head {cos['worst_head']['value']:.6f}  "
        f"worst-ROW {cos['worst_row']['value']:.6f}\n"
        f"  norm_ratio   [{scorecard['metrics']['norm_ratio']['min']:.5f}, "
        f"{scorecard['metrics']['norm_ratio']['max']:.5f}] (reported, not gated)\n"
        f"  nan/inf      {scorecard['metrics']['nan_count']}/{scorecard['metrics']['inf_count']}",
        flush=True,
    )

    if args.no_write:
        print("\n--no-write: nothing written")
        return 0

    out_dir = pathlib.Path(args.out_dir)
    fixture_path = out_dir / "fixtures" / f"{FIXTURE_SET_ID}.json"
    scorecard_path = out_dir / "scorecards" / f"{SCORECARD_ID}.json"
    envelope_path = pathlib.Path(args.envelope_dir) / f"{ENVELOPE_ID}.draft.json"
    rec.write_record(fixture_path, fixture_set)
    rec.write_record(scorecard_path, scorecard)
    rec.write_record(envelope_path, envelope)
    log_path = out_dir / "runs" / f"q1-public-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        json.dumps({"cases": cases, "oracle": oracle, "elapsed_seconds": elapsed}, indent=2),
        encoding="utf-8",
    )

    for path in (fixture_path, scorecard_path, envelope_path, log_path):
        print(f"  wrote {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    print(f"  scorecard canonical sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
