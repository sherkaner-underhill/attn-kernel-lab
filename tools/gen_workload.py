#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Expand a workload profile into an explicit, hashed case list.

The architecture requires the request schedule to be data rather than loop
arithmetic hidden inside benchmark code, and requires the workload hash to
appear in every result and manifest.  This is the only sanctioned expander.

    gen_workload.py workloads/profiles/d256-24x4-446k.yaml            # print
    gen_workload.py <profile> --write     # write cases file + record hash in profile
    gen_workload.py <profile> --check     # fail if profile's recorded hash is stale

The canonical serialisation is ``json.dumps(payload, sort_keys=True,
separators=(",", ":"))`` encoded UTF-8; ``cases_sha256`` is the SHA-256 of those
bytes.  Anything that changes the geometry, the schedule parameters, or the
derived cases changes the hash, which is the point.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys

import yaml


def expand_chunked_prefill(total_tokens: int, chunk_size: int) -> list[dict]:
    """Bottom-right-causal chunked prefill of one request.

    Case ``i`` extends by ``q`` new query rows against ``prefix`` already-cached
    keys, so it attends over ``k = prefix + q`` keys with the causal frontier at
    the bottom right: query row ``r`` sees the prefix plus keys through
    ``prefix + r``.
    """
    if chunk_size <= 0 or total_tokens <= 0:
        raise ValueError("total_tokens and chunk_size must be positive")

    cases, prefix, ordinal = [], 0, 0
    while prefix < total_tokens:
        q = min(chunk_size, total_tokens - prefix)
        cases.append(
            {
                "chunk": ordinal,
                "q_len": q,
                "prefix_len": prefix,
                "k_len": prefix + q,
                "attended_pairs": q * prefix + q * (q + 1) // 2,
            }
        )
        prefix += q
        ordinal += 1
    return cases


GENERATORS = {"chunked_prefill": expand_chunked_prefill}


def build_payload(profile: dict) -> dict:
    geometry = profile["geometry"]
    schedule = profile["schedule"]

    generator = GENERATORS.get(schedule["generator"])
    if generator is None:
        raise ValueError(f"unknown generator: {schedule['generator']}")
    cases = generator(**schedule["params"])

    # Bottom-right-causal QK+PV over logically attended pairs. Never a
    # square-causal formula: these rectangles are not squares.
    per_layer_flops = sum(
        4 * geometry["q_heads"] * geometry["head_dim"] * case["attended_pairs"] for case in cases
    )

    return {
        "workload_profile": profile["id"],
        "schema_version": profile["schema_version"],
        "geometry": geometry,
        "generator": schedule["generator"],
        "params": schedule["params"],
        "cases": cases,
        "totals": {
            "case_count": len(cases),
            "operator_calls": len(cases) * geometry["layers"],
            "attended_pairs": sum(case["attended_pairs"] for case in cases),
            "flops_per_layer": per_layer_flops,
            "flops_total": per_layer_flops * geometry["layers"],
            "flops_basis": "logically_attended_pairs",
        },
    }


def canonical_bytes(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(payload: dict) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def _replace_scalar(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"^(\s*){key}:.*$", re.MULTILINE)
    if not pattern.search(text):
        raise KeyError(f"{key} not found in profile")
    return pattern.sub(lambda m: f"{m.group(1)}{key}: {value}", text, count=1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("profile", type=pathlib.Path)
    parser.add_argument("--write", action="store_true", help="write cases file and record hash")
    parser.add_argument("--check", action="store_true", help="fail if recorded hash is stale")
    args = parser.parse_args(argv)

    root = pathlib.Path(__file__).resolve().parents[1]
    text = args.profile.read_text(encoding="utf-8")
    profile = yaml.safe_load(text)

    payload = build_payload(profile)
    computed = digest(payload)
    recorded = profile["schedule"].get("cases_sha256")
    totals = payload["totals"]

    if args.check:
        if recorded != computed:
            print(
                f"STALE {profile['id']}: recorded={recorded} computed={computed}\n"
                f"  regenerate with: tools/gen_workload.py {args.profile} --write",
                file=sys.stderr,
            )
            return 1
        print(f"OK {profile['id']}: cases_sha256={computed}")
        return 0

    if args.write:
        cases_path = root / profile["schedule"]["cases_path"]
        cases_path.parent.mkdir(parents=True, exist_ok=True)
        cases_path.write_bytes(canonical_bytes(payload) + b"\n")
        text = _replace_scalar(text, "cases_sha256", computed)
        if totals["flops_total"]:
            text = _replace_scalar(
                text, "total_pflop", f"{totals['flops_total'] / 1e15:.2f}"
            )
        args.profile.write_text(text, encoding="utf-8")
        print(f"wrote {cases_path.relative_to(root)}")

    print(f"profile        {profile['id']}")
    print(f"cases          {totals['case_count']}")
    print(f"operator calls {totals['operator_calls']}")
    print(f"attended pairs {totals['attended_pairs']:,}")
    print(f"total FLOPs    {totals['flops_total'] / 1e15:.2f} PFLOP ({totals['flops_basis']})")
    print(f"cases_sha256   {computed}")
    print()
    print(f"{'chunk':>5} {'q_len':>8} {'prefix_len':>11} {'k_len':>8}")
    for case in payload["cases"]:
        print(
            f"{case['chunk']:>5} {case['q_len']:>8} {case['prefix_len']:>11} {case['k_len']:>8}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
