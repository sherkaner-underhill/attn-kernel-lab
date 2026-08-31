#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Record assembly for the Q lane: scorecards, envelopes, fixture-set manifests.

Deliberately free of torch and of the metric library, so the shape of a record
can be tested on a CPU-only machine in the plain environment -- the same
property that lets the whole rejection matrix in ``tests/test_ops_rejection.py``
run per commit.

Three things live here rather than in the runner:

* **The ratio basis.** Every metric is reported against the same metric computed
  for a BF16 implementation swap on the same tensors in the same process.  For
  an error metric the ratio is ``value / control``.  For a *similarity* it is
  taken over the deficit, ``(1 - value) / (1 - control)``: cosine 0.9985 against
  a control of 0.99997 is a fifty-fold loss of agreement, and the naive quotient
  0.9985 reports it as none.
* **The saturation rule.**  ``control >= 0.5 * value`` on that basis.  A metric
  whose stock-vs-stock control moves as far as the candidate does cannot rank
  backends, and the rule exists so that fact lands in the record automatically
  instead of being remembered.
* **The four mandatory aggregations.**  ``mean``, ``worst_layer``,
  ``worst_head`` and ``worst_row``.  The first three are the plan's; the fourth
  is the 2026-08-30 addendum's, and it is the one that matters most at depth --
  the per-tensor collapse the instrument found (cosine 0.894 -> 0.049 at 446k)
  is non-monotone in the mean AND in the worst-head aggregate, and only shows
  as a worst ROW.

``tools/validate_registry.py`` re-derives all three independently.  That
duplication is intended: a validator that imports the producer's own opinion of
the rule cannot catch the producer getting it wrong.
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from validate_registry import canonical_digest  # noqa: E402  -- the house digest convention

__all__ = [
    "METRIC_FORM",
    "GATE_METRICS",
    "canonical_digest",
    "deficit",
    "ratio",
    "saturated",
    "slice_entry",
    "metric_block",
    "reported_block",
    "seal",
    "write_record",
]

#: (direction, ratio_basis) per gate metric.  Mirrors the same table in the
#: validator; disagreement between them is itself a caught error.
METRIC_FORM = {
    "row_rel_l2": ("lower_is_better", "direct"),
    "cos_sim": ("higher_is_better", "one_minus"),
    "rel_l1": ("lower_is_better", "direct"),
    "rmse": ("lower_is_better", "direct"),
}

GATE_METRICS = tuple(METRIC_FORM)

#: The donor's ``_metrics`` key for each gate metric's scalar value.
DONOR_KEY = {
    "row_rel_l2": "row_rel_l2_mean",
    "cos_sim": "cos_sim_mean",
    "rel_l1": "rel_l1",
    "rmse": "rmse",
}


def deficit(basis: str, value: float) -> float:
    """The quantity the ratio is taken over."""
    return (1.0 - value) if basis == "one_minus" else float(value)


def ratio(basis: str, value: float, control: float) -> float | None:
    """value/control on the ratio basis; ``None`` when the control is exactly 0."""
    denominator = deficit(basis, control)
    if denominator == 0.0:
        return None
    return deficit(basis, value) / denominator


def saturated(basis: str, value: float, control: float) -> bool:
    """The rule: a control at least half the candidate's movement is a null detector."""
    return deficit(basis, control) >= 0.5 * deficit(basis, value)


def slice_entry(basis: str, value: float, control: float, **locator) -> dict:
    """One slice of one metric: the value, its anchor, the ratio, the flag."""
    entry = {k: v for k, v in locator.items() if v is not None}
    entry.update(
        {
            "value": float(value),
            "control": float(control),
            "ratio": ratio(basis, value, control),
            "saturated": saturated(basis, value, control),
        }
    )
    return entry


def metric_block(
    name: str,
    *,
    mean: float,
    mean_control: float,
    per_layer: list[dict],
    worst_layer: dict,
    worst_head: dict,
    worst_row: dict,
    note: str | None = None,
) -> dict:
    """Assemble one gate metric's block.  Every slice argument is a `slice_entry`."""
    direction, basis = METRIC_FORM[name]
    block = {
        "direction": direction,
        "ratio_basis": basis,
        "mean": float(mean),
        "control": float(mean_control),
        "ratio": ratio(basis, mean, mean_control),
        "saturated": saturated(basis, mean, mean_control),
        "worst_layer": worst_layer,
        "worst_head": worst_head,
        "worst_row": worst_row,
        "per_layer": per_layer,
    }
    if note:
        block["note"] = note
    return block


def reported_block(per_layer: list[dict], note: str) -> dict:
    """`norm_ratio`: reported, never gated (addendum item 2).

    The instrument found a systematic E4M3 output-norm inflation that drifts with
    depth and never leaves [0.98, 1.02].  A gate on it would never fire; the
    trend line is the informative artefact, so the schema makes `gated` a
    constant false rather than a field an author can flip.
    """
    values = [entry["value"] for entry in per_layer]
    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "gated": False,
        "per_layer": per_layer,
        "note": note,
    }


def seal(record: dict, field: str) -> dict:
    """Fill ``field`` with the record's own canonical digest, minus that field."""
    record[field] = canonical_digest(record, exclude=field)
    return record


def write_record(path: pathlib.Path, record: dict) -> str:
    """Write a record and return its canonical digest (how another record cites it)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return canonical_digest(record)
