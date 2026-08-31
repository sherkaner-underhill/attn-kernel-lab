#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render N fidelity scorecards as one table: one row per metric slice.

The deliverable of the comparability half of the quality gate (plan §6.2).
"Understand and compare how different kernels' quantization loss compares" is a
rendering problem once the scorecard is uniform -- and it is deliberately NOT a
single scalar. A composite score would recreate exactly the averaging blindness
the worst-layer and worst-row mandates exist to prevent: the run that seeded this
lane has a 3.5% mean and a 29.9% worst row, and any weighting that turns those
into one number has thrown away the half that matters.

The control column is always present, because a metric without its
implementation-swap anchor is an absolute, and absolutes drift with the fixture,
the driver and the model revision while ratios survive.

    python3 quality/fidelity_diff.py quality/scorecards/*.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fidelity_records as rec  # noqa: E402

SLICES = ("mean", "worst_layer", "worst_head", "worst_row")

#: How each metric reads. row_rel_l2 and rel_l1 are dimensionless ratios of
#: magnitudes and belong in percent; rmse is an ABSOLUTE error in the output's
#: own units and does not, which is exactly the scale-blindness cosine has and
#: rmse is kept to catch.
FORMAT = {"row_rel_l2": "pct", "rel_l1": "pct", "rmse": "abs", "cos_sim": "sim"}


def _fmt(block: dict, slice_name: str, name: str) -> str:
    if slice_name == "mean":
        value, ratio, sat = block["mean"], block["ratio"], block["saturated"]
        where = ""
    else:
        entry = block[slice_name]
        value, ratio, sat = entry["value"], entry["ratio"], entry["saturated"]
        where = " " + "/".join(
            str(entry[k]) for k in ("layer_name", "head", "row") if k in entry
        )
    if sat:
        return "<saturated>"
    style = FORMAT.get(name, "pct")
    shown = {"sim": f"{value:.6f}", "abs": f"{value:.4g}"}.get(style, f"{value * 100:.3f}%")
    tail = f" ({ratio:.2f}x)" if ratio is not None else " (control 0)"
    return shown + tail + where


def render(cards: list[tuple[str, dict]]) -> str:
    width = max(len(label) for label, _ in cards) + 2
    header = "metric / slice".ljust(26) + "".join(label.ljust(width) for label, _ in cards)
    lines = [header, "-" * len(header)]
    for name in rec.GATE_METRICS:
        for slice_name in SLICES:
            row = f"{name} {slice_name}".ljust(26)
            for _, card in cards:
                block = card["metrics"].get(name)
                row += (_fmt(block, slice_name, name) if block else "-").ljust(width)
            lines.append(row)
        lines.append("")
    row = "norm_ratio [min, max]".ljust(26)
    for _, card in cards:
        norm = card["metrics"]["norm_ratio"]
        row += f"[{norm['min']:.5f}, {norm['max']:.5f}]".ljust(width)
    lines.append(row + "   reported, never gated")
    row = "nan / inf".ljust(26)
    for _, card in cards:
        row += f"{card['metrics']['nan_count']} / {card['metrics']['inf_count']}".ljust(width)
    lines.append(row)
    lines.append("")
    for label, card in cards:
        controls = ", ".join(
            f"{c['role']}={c.get('result') or 'anchor'}"
            + (f" ({c['value']:.3g})" if c.get("value") is not None else "")
            for c in card["controls"]
        )
        lines.append(
            f"{label}: {card['status']} | {card['fixture_set']['lane']} | "
            f"{card['fixture_set']['id']} | layers={card['fixture_set']['layers']} "
            f"(slice_semantics.layer={card['fixture_set']['slice_semantics']['layer']})"
        )
        lines.append(f"    controls: {controls}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("scorecards", nargs="+", type=pathlib.Path)
    args = parser.parse_args(argv)

    cards = []
    for path in args.scorecards:
        card = json.loads(path.read_text(encoding="utf-8"))
        cards.append((card.get("scorecard_id") or path.stem, card))
    print(render(cards))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
