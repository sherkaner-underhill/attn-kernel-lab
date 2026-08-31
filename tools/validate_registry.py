#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate every registry record, and the invariants that hold *between* them.

Schema validation alone is not enough here.  The design's load-bearing rules are
cross-record -- a development-tier GPU must not be able to author a performance
qualification, a release must not claim qualification on a target it does not
support -- and those are exactly the rules that erode silently.  They are
checked mechanically so that they cannot.

    validate_registry.py            # validate everything, exit non-zero on error
    validate_registry.py --verbose  # list every record checked
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys

import jsonschema
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "tools"))
from gen_workload import build_payload, digest  # noqa: E402

PERF_KINDS = {"kernel_performance"}
ENGINE_KINDS = {"integration_smoke", "application", "production"}
FIDELITY_KINDS = {"fidelity"}
PROMOTION_LANES = {"inclusive", "schedule_replay"}
REQUIRED_THRESHOLDS = {
    "min_inclusive_improvement_pct",
    "min_paired_ci95_lower_pct",
    "max_single_workload_regression_pct",
    "min_independent_blocks",
}

# --- the Q lane -------------------------------------------------------------
#
# The rules below are deliberately mechanical restatements of the Q-lane's
# load-bearing decisions, and they are re-derived here rather than
# imported from the harness that produces the records: a validator that asks the
# harness whether the harness got it right checks nothing.

#: Metrics that must carry the full four-way aggregation to be gate-admissible.
GATE_METRICS = ("row_rel_l2", "cos_sim", "rel_l1", "rmse")

#: A similarity has to be anchored through its deficit; see `_deficit`.
METRIC_FORM = {
    "row_rel_l2": ("lower_is_better", "direct"),
    "cos_sim": ("higher_is_better", "one_minus"),
    "rel_l1": ("lower_is_better", "direct"),
    "rmse": ("lower_is_better", "direct"),
}

#: Healthy accept_len band 3.20 / 4.36 / 4.80 on the 900 / 10k / 40k prompts;
#: a collapsed spec graph reads 2.40-2.82. A scorecard measured through a
#: collapsed boot is self-invalidating, so the floor is a validator rule rather
#: than a caveat (plan §7.3, §9.1.5).
ACCEPT_LEN_FLOOR = 3.0

ENVELOPE_DIR = ("promotion", "envelopes")
SCORECARD_DIR = ("quality", "scorecards")
FIXTURE_DIR = ("quality", "fixtures")


def canonical_digest(record: dict, *, exclude: str | None = None) -> str:
    """SHA-256 over a record serialised canonically, optionally minus one field.

    The house convention (``tools/make_candidate_records.py``): sorted keys,
    compact separators, UTF-8. ``exclude`` lets a record carry a digest OF
    ITSELF -- the fixture-set manifest does, so that a fixture set is
    self-verifying and independent of where the file sits, which a path- and
    mode-sensitive tree digest cannot be.
    """
    body = {k: v for k, v in record.items() if k != exclude} if exclude else record
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _deficit(basis: str, value: float) -> float:
    """The quantity a ratio is taken over.

    For an error metric that is the value itself. For a similarity it is
    ``1 - value``: cosine 0.9985 against a control of 0.99997 is a 50x loss of
    agreement, and ``0.9985 / 0.99997 = 0.9985`` reports it as none.
    """
    return (1.0 - value) if basis == "one_minus" else value


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-12)


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.targets: dict[str, dict] = {}
        self.workloads: dict[str, dict] = {}
        self.engines: dict[str, dict] = {}
        self.fixture_sets: dict[str, dict] = {}
        self.envelopes: dict[str, dict] = {}
        self.scorecards: dict[str, dict] = {}
        self.blocked: list[str] = []
        self.checked: list[str] = []

    def error(self, where: str, message: str) -> None:
        self.errors.append(f"{where}: {message}")

    def block(self, where: str, message: str) -> None:
        self.blocked.append(f"{where}: {message}")


def _load_schema(root: pathlib.Path, name: str, subdir: str) -> dict:
    return json.loads((root / subdir / "schema" / name).read_text(encoding="utf-8"))


def _load_yaml_dir(path: pathlib.Path) -> list[tuple[pathlib.Path, dict]]:
    return [(p, yaml.safe_load(p.read_text(encoding="utf-8"))) for p in sorted(path.glob("*.yaml"))]


def _load_json_tree(path: pathlib.Path, pattern: str) -> list[tuple[pathlib.Path, dict]]:
    if not path.exists():
        return []
    return [(p, json.loads(p.read_text(encoding="utf-8"))) for p in sorted(path.rglob(pattern))]


def _validate(report: Report, schema: dict, record: dict, where: str) -> bool:
    try:
        jsonschema.validate(record, schema)
    except jsonschema.ValidationError as exc:
        location = "/".join(str(part) for part in exc.absolute_path) or "<root>"
        report.error(where, f"schema: {location}: {exc.message}")
        return False
    report.checked.append(where)
    return True


def check_targets(report: Report, root: pathlib.Path) -> dict[str, dict]:
    schema = _load_schema(root, "target-profile.schema.json", "targets")
    targets: dict[str, dict] = {}

    for path, record in _load_yaml_dir(root / "targets"):
        where = str(path.relative_to(root))
        if not _validate(report, schema, record, where):
            continue

        if record["id"] != path.stem:
            report.error(where, f"id {record['id']!r} does not match filename {path.stem!r}")
        if record["id"] in targets:
            report.error(where, f"duplicate target id {record['id']!r}")
        targets[record["id"]] = record

        thresholds = record.get("promotion_thresholds") or {}
        if record["authority"] == "production":
            missing = REQUIRED_THRESHOLDS - set(thresholds)
            if missing:
                report.error(where, f"production target missing thresholds: {sorted(missing)}")
        elif thresholds:
            # The rule that makes the development tier safe: a target that cannot
            # qualify a release must not carry the numbers used to qualify one.
            report.error(
                where,
                f"authority={record['authority']!r} must not define promotion_thresholds",
            )

        if record["authority"] == "production" and record["verification"]["state"] == "unverified":
            report.error(where, "a production-authority target cannot be `unverified`")

    return targets


def check_workloads(
    report: Report, root: pathlib.Path, targets: dict[str, dict]
) -> dict[str, dict]:
    schema = _load_schema(root, "workload-profile.schema.json", "workloads")
    workloads: dict[str, dict] = {}

    for path, record in _load_yaml_dir(root / "workloads" / "profiles"):
        where = str(path.relative_to(root))
        if not _validate(report, schema, record, where):
            continue

        if record["id"] != path.stem:
            report.error(where, f"id {record['id']!r} does not match filename {path.stem!r}")
        workloads[record["id"]] = record

        computed = digest(build_payload(record))
        recorded = record["schedule"].get("cases_sha256")
        if recorded is None:
            report.error(where, "cases_sha256 unset; run tools/gen_workload.py --write")
        elif recorded != computed:
            report.error(
                where, f"stale cases_sha256 (recorded {recorded[:12]}, computed {computed[:12]})"
            )

        cases_path = record["schedule"].get("cases_path")
        if cases_path and not (root / cases_path).exists():
            report.error(where, f"cases_path missing on disk: {cases_path}")

        unresolved = record["origin"].get("unresolved") or []
        if record.get("protected") and unresolved:
            report.block(
                where,
                f"protected profile has {len(unresolved)} unresolved discrepancy(ies); "
                "must not back a promotion until settled",
            )

    for target_id, target in targets.items():
        limits = target.get("workload_limits") or {}
        for key in ("max_workload_profiles", "excluded_workload_profiles"):
            for workload_id in limits.get(key) or []:
                if workload_id not in workloads:
                    report.error(
                        f"targets/{target_id}.yaml",
                        f"{key} references unknown workload {workload_id!r}",
                    )

    return workloads


def check_engines(report: Report, root: pathlib.Path) -> dict[str, dict]:
    schema = _load_schema(root, "engine-profile.schema.json", "engines")
    engines: dict[str, dict] = {}

    for path, record in _load_yaml_dir(root / "engines" / "profiles"):
        where = str(path.relative_to(root))
        if not _validate(report, schema, record, where):
            continue

        if record["id"] != path.stem:
            report.error(where, f"id {record['id']!r} does not match filename {path.stem!r}")
        engines[record["id"]] = record

        gate_ids = [gate["id"] for gate in record["gates"]]
        if len(gate_ids) != len(set(gate_ids)):
            report.error(where, "duplicate gate ids")

        for issue in record.get("known_issues") or []:
            unknown = set(issue.get("affects_gates") or []) - set(gate_ids)
            if unknown:
                report.error(
                    where, f"known issue {issue['id']!r} affects unknown gates {sorted(unknown)}"
                )
            if issue["status"] in {"open", "workaround-in-production"} and not issue.get(
                "affects_gates"
            ):
                # An engine defect that changes what a gate result MEANS must say
                # which gates it touches, or the caveat is lost at read time.
                report.error(
                    where,
                    f"known issue {issue['id']!r} is {issue['status']} but names no affected gates",
                )

        if record["status"] == "active" and not gate_ids:
            report.error(where, "an active engine must declare its gate ladder")

    return engines


def check_manifests(
    report: Report,
    root: pathlib.Path,
    targets: dict[str, dict],
    workloads: dict[str, dict],
    engines: dict[str, dict],
) -> None:
    schema = _load_schema(root, "artifact-manifest.schema.json", "promotion")

    for path, record in _load_json_tree(root / "promotion" / "releases", "artifact-manifest.json"):
        where = str(path.relative_to(root))
        if not _validate(report, schema, record, where):
            continue

        supported = set(record["targets"]["supported"])
        for target_id in supported:
            if target_id not in targets:
                report.error(where, f"supported target {target_id!r} is not in the registry")

        for entry in record["targets"]["qualified"]:
            if entry["target"] not in supported:
                report.error(where, f"qualified on {entry['target']!r} which is not in `supported`")
            target = targets.get(entry["target"])
            if target and target["authority"] != "production":
                report.error(
                    where,
                    f"qualified on {entry['target']!r} whose authority is "
                    f"{target['authority']!r}; only production-authority targets may qualify a release",
                )
            if entry["workload_profile"] not in workloads:
                report.error(
                    where, f"qualified against unknown workload {entry['workload_profile']!r}"
                )

        for entry in record["targets"]["excluded"]:
            if entry["target"] in supported:
                report.error(where, f"target {entry['target']!r} is both supported and excluded")

        for entry in (record.get("engines") or {}).get("integrated") or []:
            if entry["engine"] not in engines:
                report.error(where, f"integrated engine {entry['engine']!r} is not in the registry")
            elif engines[entry["engine"]]["status"] != "active":
                report.error(
                    where,
                    f"integrated engine {entry['engine']!r} has status "
                    f"{engines[entry['engine']]['status']!r}; only an active engine can hold an integration",
                )


def check_attestations(
    report: Report,
    root: pathlib.Path,
    targets: dict[str, dict],
    workloads: dict[str, dict],
    engines: dict[str, dict],
) -> list[tuple[pathlib.Path, dict]]:
    """Returns the attestations that passed schema validation, for the second pass."""
    schema = _load_schema(root, "qualification-attestation.schema.json", "promotion")
    well_formed: list[tuple[pathlib.Path, dict]] = []

    for path, record in _load_json_tree(root / "promotion" / "attestations", "*.json"):
        where = str(path.relative_to(root))
        if not _validate(report, schema, record, where):
            continue
        well_formed.append((path, record))

        if record["kind"] in ENGINE_KINDS:
            engine_id = record.get("engine")
            if not engine_id:
                report.error(where, f"a {record['kind']} attestation must name an engine")
            elif engine_id not in engines:
                report.error(where, f"unknown engine {engine_id!r}")
            else:
                engine = engines[engine_id]
                if not record.get("engine_revision"):
                    report.error(where, "engine_revision must be recorded")
                known = {gate["id"] for gate in engine["gates"]}
                by_id = {gate["id"]: gate for gate in engine["gates"]}
                for gate in record.get("gates") or []:
                    gate_id = gate.get("engine_gate_id")
                    if gate_id is None:
                        continue
                    if gate_id not in known:
                        report.error(
                            where, f"gate {gate_id!r} is not in the {engine_id!r} gate ladder"
                        )
                        continue
                    required = by_id[gate_id].get("requires_target_authority", "any")
                    target = targets.get(record.get("target") or "")
                    if (
                        gate["result"] == "pass"
                        and required != "any"
                        and target is not None
                        and required == "production"
                        and target["authority"] != "production"
                    ):
                        report.error(
                            where,
                            f"gate {gate_id!r} requires production hardware but ran on "
                            f"{record['target']!r} (authority {target['authority']!r})",
                        )

        if record["kind"] not in PERF_KINDS:
            continue

        target_id = record.get("target")
        if not target_id:
            report.error(where, "kernel_performance attestation must name a target")
            continue
        target = targets.get(target_id)
        if target is None:
            report.error(where, f"unknown target {target_id!r}")
        elif target["authority"] != "production":
            report.error(
                where,
                f"performance attestation on {target_id!r} (authority "
                f"{target['authority']!r}); development-tier hardware can prove a change "
                "is correct, never that it is fast",
            )

        workload_id = record.get("workload_profile")
        if workload_id not in workloads:
            report.error(
                where,
                f"kernel_performance attestation needs a known workload_profile, got {workload_id!r}",
            )
        else:
            recorded = record.get("workload_cases_sha256")
            expected = workloads[workload_id]["schedule"].get("cases_sha256")
            if recorded and expected and recorded != expected:
                report.error(where, "workload_cases_sha256 does not match the current profile")

        measurement = record.get("measurement") or {}
        lane = measurement.get("lane")
        if lane not in PROMOTION_LANES:
            report.error(
                where,
                f"lane {lane!r} has no promotion authority; use one of {sorted(PROMOTION_LANES)}",
            )
        if measurement.get("timing_backend") is None:
            report.error(where, "timing_backend must be recorded, never assumed")

        for gate in record.get("gates") or []:
            if gate["result"] == "skipped" and not gate.get("skip_reason"):
                report.error(where, f"gate {gate['name']!r} skipped without a reason")

    return well_formed


def check_fixture_sets(report: Report, root: pathlib.Path) -> dict[str, dict]:
    """Fidelity fixture-set manifests: self-verifying, and lane-safe.

    The privacy guard is the point. `upstream/CLAIMS.md` and `AGENTS.md` both
    say private captures never enter this repository; a manifest is exactly the
    artifact that makes a private result auditable WITHOUT the tensors, so it is
    also exactly the artifact that leaks if it is allowed to name where they
    live.
    """
    schema = _load_schema(root, "fidelity-fixture-set.schema.json", "promotion")
    sets: dict[str, dict] = {}

    for path, record in _load_json_tree(root.joinpath(*FIXTURE_DIR), "*.json"):
        where = str(path.relative_to(root))
        if not _validate(report, schema, record, where):
            continue

        set_id = record["fixture_set_id"]
        if set_id != path.stem:
            report.error(where, f"fixture_set_id {set_id!r} does not match filename {path.stem!r}")
        if set_id in sets:
            report.error(where, f"duplicate fixture_set_id {set_id!r}")

        recomputed = canonical_digest(record, exclude="fixture_set_sha256")
        if recomputed != record["fixture_set_sha256"]:
            report.error(
                where,
                f"stale fixture_set_sha256 (recorded {record['fixture_set_sha256'][:12]}, "
                f"computed {recomputed[:12]}): the manifest was edited after it was sealed",
            )

        lane = record["lane"]
        if lane == "private_real":
            if record["redistributable"]:
                report.error(where, "a private_real fixture set may not be marked redistributable")
            if not str(record["provenance"].get("media", "")).startswith("PRIVATE:"):
                report.error(
                    where,
                    "a private_real fixture set must record its media as PRIVATE:<sha256>, "
                    "never a title or a path",
                )
            named = [case["name"] for case in record["cases"] if "path" in case]
            if named:
                report.error(
                    where,
                    f"private_real cases name tensor paths ({sorted(named)[:3]}): the manifest "
                    "carries hashes and geometry only, so that committing it discloses nothing",
                )
        elif not record["redistributable"]:
            report.error(where, f"lane {lane!r} is a public lane and must be redistributable")

        layers = [case["layer"] for case in record["cases"]]
        if sorted(layers) != list(range(len(layers))):
            report.error(
                where,
                f"case layer ids must be 0..{len(layers) - 1} exactly once, got {sorted(layers)}: "
                "the scorecard's per_layer array is positional",
            )
        for case in record["cases"]:
            if case["prefix"] + record["geometry"]["q_len"] != case["depth"]:
                report.error(
                    where,
                    f"case {case['name']!r}: prefix {case['prefix']} + q_len "
                    f"{record['geometry']['q_len']} != depth {case['depth']}; the bottom-right "
                    "causal diagonal sits at kv_len - q_len",
                )
        sets[set_id] = record

    report.fixture_sets = sets
    return sets


def check_envelopes(
    report: Report, root: pathlib.Path, fixture_sets: dict[str, dict]
) -> dict[str, dict]:
    """Fidelity envelopes: a draft may not enforce, and only review may freeze."""
    schema = _load_schema(root, "fidelity-envelope.schema.json", "promotion")
    envelopes: dict[str, dict] = {}

    for path, record in _load_json_tree(root.joinpath(*ENVELOPE_DIR), "*.json"):
        where = str(path.relative_to(root))
        if not _validate(report, schema, record, where):
            continue

        envelope_id = record["envelope_id"]
        stem = path.stem
        is_draft_name = stem.endswith(".draft")
        expected = envelope_id + (".draft" if record["status"] == "draft" else "")
        if stem != expected:
            report.error(
                where,
                f"filename {stem!r} must be {expected!r} for envelope_id {envelope_id!r} with "
                f"status {record['status']!r}: a draft is named as one on disk",
            )
        if is_draft_name and record["status"] != "draft":
            report.error(where, "a .draft filename must carry status 'draft'")
        if envelope_id in envelopes:
            report.error(where, f"duplicate envelope_id {envelope_id!r}")

        rule = record["decision_rule"]
        if record["status"] == "draft":
            if record["calibrated"]:
                report.error(where, "a draft envelope cannot be calibrated")
            if record.get("approver"):
                report.error(
                    where,
                    "a draft envelope carries no approver: an approver is what freezing IS",
                )
            if rule["worst_slice_enforced"]:
                report.error(where, "a draft envelope may not enforce its worst-slice thresholds")
        else:
            if not record.get("approver"):
                report.error(
                    where,
                    "a frozen envelope needs an approver and a review reference; freezing is an "
                    "explicit decision, exactly like publishing a release",
                )
        if rule["worst_slice_enforced"] and not record["calibrated"]:
            report.error(
                where,
                "worst_slice_enforced requires calibrated=true: the worst-layer figure is compared "
                "against a reproducibility band from three independent fixture sets, and until "
                "those exist it is reported for a human, not enforced",
            )
        if record.get("supersedes") and not record.get("supersede_reason"):
            report.error(where, "superseding an envelope must name the reason")

        for entry in record["fixture_sets"]:
            known = fixture_sets.get(entry["id"])
            if known is None:
                continue  # a private-lane set legitimately has no in-tree manifest yet
            if known["fixture_set_sha256"] != entry["fixture_set_sha256"]:
                report.error(
                    where,
                    f"fixture set {entry['id']!r} digest does not match the in-tree manifest",
                )
            if known["lane"] != entry["lane"]:
                report.error(where, f"fixture set {entry['id']!r} lane disagrees with its manifest")

        envelopes[envelope_id] = record

    report.envelopes = envelopes
    return envelopes


def _check_metric_block(report: Report, where: str, name: str, block: dict, layers: int) -> None:
    """The four-way aggregation, the ratio, and the saturation flag, re-derived.

    Every rule here exists because the corresponding mistake is one a scorecard
    can make while looking perfectly well-formed: a worst_layer that is not the
    worst layer, a ratio that does not divide, a `saturated: false` on a metric
    whose control moved as far as the candidate did.
    """
    direction, basis = METRIC_FORM.get(name, (block["direction"], block["ratio_basis"]))
    if (block["direction"], block["ratio_basis"]) != (direction, basis):
        report.error(
            where,
            f"metric {name!r} declares {block['direction']}/{block['ratio_basis']}, but this "
            f"metric is {direction}/{basis}",
        )
        return

    per_layer = block["per_layer"]
    if len(per_layer) != layers:
        report.error(
            where,
            f"metric {name!r}: per_layer has {len(per_layer)} entries for a fixture set of "
            f"{layers} layer(s)",
        )
        return
    ids = [entry["layer"] for entry in per_layer]
    if sorted(ids) != list(range(len(ids))):
        report.error(where, f"metric {name!r}: per_layer layer ids must be 0..{len(ids) - 1}")
        return

    pick = min if direction == "higher_is_better" else max
    worst = pick(per_layer, key=lambda entry: entry["value"])
    if block["worst_layer"]["layer"] != worst["layer"] or not _close(
        block["worst_layer"]["value"], worst["value"]
    ):
        report.error(
            where,
            f"metric {name!r}: worst_layer names layer {block['worst_layer']['layer']} at "
            f"{block['worst_layer']['value']:.6g}, but per_layer's worst is layer "
            f"{worst['layer']} at {worst['value']:.6g}",
        )

    for slice_name in ("worst_layer", "worst_head", "worst_row"):
        entry = block[slice_name]
        if not 0 <= entry["layer"] < layers:
            report.error(where, f"metric {name!r}: {slice_name} names layer {entry['layer']}")

    # The worst head is at least as bad as the worst layer, and the worst row at
    # least as bad as the worst head: they are nested reductions of one tensor.
    order = [
        block["worst_layer"]["value"],
        block["worst_head"]["value"],
        block["worst_row"]["value"],
    ]
    monotone = (
        all(a >= b for a, b in zip(order, order[1:]))
        if direction == "higher_is_better"
        else all(a <= b for a, b in zip(order, order[1:]))
    )
    if not monotone:
        report.error(
            where,
            f"metric {name!r}: worst_layer/worst_head/worst_row are {order}, which is not a "
            "nested reduction of the same tensor -- a coarser slice cannot be worse than a finer one",
        )

    for label, entry in (
        [
            (
                "aggregate",
                {
                    "value": block["mean"],
                    "control": block["control"],
                    "ratio": block["ratio"],
                    "saturated": block["saturated"],
                },
            )
        ]
        + [(s, block[s]) for s in ("worst_layer", "worst_head", "worst_row")]
        + [(f"per_layer[{e['layer']}]", e) for e in per_layer]
    ):
        control = _deficit(basis, entry["control"])
        value = _deficit(basis, entry["value"])
        expected = None if control == 0.0 else value / control
        if entry["ratio"] is None:
            if expected is not None:
                report.error(
                    where, f"metric {name!r} {label}: ratio is null but the control is not 0"
                )
        elif expected is None or not _close(entry["ratio"], expected):
            report.error(
                where,
                f"metric {name!r} {label}: ratio {entry['ratio']!r} is not value/control "
                f"({expected!r}) on the {basis} basis",
            )
        if "saturated" in entry:
            expect_sat = control >= 0.5 * value
            if entry["saturated"] != expect_sat:
                report.error(
                    where,
                    f"metric {name!r} {label}: saturated={entry['saturated']} but control "
                    f"{control:.6g} {'>=' if expect_sat else '<'} 0.5 * value {value:.6g}",
                )


def check_scorecards(
    report: Report, root: pathlib.Path, fixture_sets: dict[str, dict], envelopes: dict[str, dict]
) -> dict[str, dict]:
    """Fidelity scorecards, indexed by canonical digest (how an attestation cites one)."""
    schema = _load_schema(root, "fidelity-scorecard.schema.json", "promotion")
    scorecards: dict[str, dict] = {}

    for path, record in _load_json_tree(root.joinpath(*SCORECARD_DIR), "*.json"):
        where = str(path.relative_to(root))
        if not _validate(report, schema, record, where):
            continue

        fixture = record["fixture_set"]
        known = fixture_sets.get(fixture["id"])
        if known is not None:
            if known["fixture_set_sha256"] != fixture["fixture_set_sha256"]:
                report.error(
                    where, f"fixture set {fixture['id']!r} digest does not match its manifest"
                )
            if known["lane"] != fixture["lane"]:
                report.error(
                    where, f"fixture set {fixture['id']!r} lane disagrees with its manifest"
                )
            if len(known["cases"]) != fixture["layers"]:
                report.error(
                    where,
                    f"fixture set {fixture['id']!r} has {len(known['cases'])} case(s) but the "
                    f"scorecard reports {fixture['layers']} layer(s)",
                )
            if known["slice_semantics"] != fixture["slice_semantics"]:
                report.error(where, f"fixture set {fixture['id']!r} slice_semantics disagree")

        primary = [c for c in record["controls"] if c.get("primary")]
        if len(primary) != 1:
            report.error(
                where,
                f"exactly one control must be marked primary (found {len(primary)}): every "
                "metric block's `control` field has to come from a named anchor",
            )

        missing = [m for m in GATE_METRICS if m not in record["metrics"]]
        if missing:
            report.error(where, f"missing gate metric(s) {missing}")
        # Every gate-metric block, including one a later rung adds through the
        # schema's open `additionalProperties` -- top1_flip_rate, say. A metric
        # that arrives without the four aggregations or with an unchecked ratio
        # would be exactly the loophole this validator exists to close.
        for name, block in record["metrics"].items():
            if name in ("norm_ratio", "nan_count", "inf_count") or not isinstance(block, dict):
                continue
            _check_metric_block(report, where, name, block, fixture["layers"])

        norm = record["metrics"]["norm_ratio"]
        if len(norm["per_layer"]) != fixture["layers"]:
            report.error(where, "norm_ratio.per_layer must cover every layer")

        envelope_ref = record.get("envelope")
        if envelope_ref:
            envelope = envelopes.get(envelope_ref["envelope_id"])
            if envelope is None:
                report.error(
                    where,
                    f"cites envelope {envelope_ref['envelope_id']!r}, which is not in "
                    f"{'/'.join(ENVELOPE_DIR)}",
                )
            else:
                if envelope_ref.get("envelope_status") not in (None, envelope["status"]):
                    report.error(where, "envelope_status disagrees with the envelope record")
                if envelope["status"] == "draft" and envelope_ref["comparison"] == "pass":
                    report.error(
                        where,
                        "a draft envelope cannot return a 'pass' comparison: it is a measurement, "
                        "not yet a commitment",
                    )

        scorecards[canonical_digest(record)] = {"path": where, "record": record}

    report.scorecards = scorecards
    return scorecards


def check_fidelity_attestations(
    report: Report,
    root: pathlib.Path,
    targets: dict[str, dict],
    engines: dict[str, dict],
    fixture_sets: dict[str, dict],
    envelopes: dict[str, dict],
    scorecards: dict[str, dict],
    attestations: list[tuple[pathlib.Path, dict]],
) -> None:
    """The cross-record rules that make a fidelity attestation mean something."""
    for path, record in attestations:
        if record.get("kind") not in FIDELITY_KINDS:
            continue
        where = str(path.relative_to(root))
        block = record["fidelity"]

        cited = []
        for sha in block["scorecard_sha256"]:
            entry = scorecards.get(sha)
            if entry is None:
                report.error(
                    where,
                    f"cites scorecard {sha[:12]} which is not in {'/'.join(SCORECARD_DIR)}; a "
                    "fidelity verdict whose evidence is not in the tree is unreviewable",
                )
                continue
            cited.append(entry["record"])
            subject_release = entry["record"]["subject"].get("release_id")
            if subject_release and subject_release != record["release_id"]:
                report.error(
                    where,
                    f"scorecard {sha[:12]} measures release {subject_release!r}, not "
                    f"{record['release_id']!r}",
                )
        if cited:
            boundaries = sorted({card["boundary"] for card in cited})
            if boundaries != sorted(block["boundaries"]):
                report.error(
                    where,
                    f"boundaries {sorted(block['boundaries'])} do not match the cited scorecards' "
                    f"{boundaries}",
                )

        envelope = envelopes.get(block["envelope_id"])
        if envelope is None:
            report.error(
                where,
                f"envelope {block['envelope_id']!r} is not in {'/'.join(ENVELOPE_DIR)}",
            )
        else:
            recorded = canonical_digest(envelope)
            if block.get("envelope_sha256") not in (None, recorded):
                report.error(where, "envelope_sha256 does not match the envelope record")
            if block.get("envelope_calibrated") not in (None, envelope["calibrated"]):
                report.error(
                    where,
                    f"envelope_calibrated={block['envelope_calibrated']} but envelope "
                    f"{envelope['envelope_id']!r} records calibrated={envelope['calibrated']}",
                )
            if block.get("envelope_status") not in (None, envelope["status"]):
                report.error(where, "envelope_status does not match the envelope record")
            if envelope["status"] == "draft" and not (record.get("notes") or []):
                report.error(
                    where,
                    "an attestation resting on a DRAFT envelope must say so in `notes`: the band "
                    "it is measured against has not been frozen by review",
                )

        lanes = set()
        for entry in block["fixture_sets"]:
            lanes.add(entry["lane"])
            known = fixture_sets.get(entry["id"])
            if known is None:
                continue
            if known["fixture_set_sha256"] != entry["fixture_set_sha256"]:
                report.error(
                    where, f"fixture set {entry['id']!r} digest does not match its manifest"
                )
            if known["lane"] != entry["lane"]:
                report.error(where, f"fixture set {entry['id']!r} lane disagrees with its manifest")

        target = targets.get(record.get("target") or "")
        if "private_real" in lanes and (target is None or target["authority"] != "production"):
            report.error(
                where,
                "a private_real fixture set may only back an attestation on a production-authority "
                "target: the tensors may not leave the production host, so a development-tier record could not "
                "have been produced from them",
            )

        for slice_name in ("worst_layer", "worst_head", "worst_row"):
            if not block["worst"].get(slice_name):
                report.error(where, f"fidelity.worst is missing {slice_name}")

        boot = block["boot_gate"]
        engine_id = record.get("engine")
        if engine_id:
            if not boot["applicable"]:
                report.error(
                    where,
                    "an engine-bound fidelity result must record its boot gate; a scorecard from a "
                    "collapsed boot is self-invalidating",
                )
            elif "accept_len" not in boot:
                report.error(
                    where, "boot_gate.accept_len must be recorded for an engine-bound result"
                )
            elif boot["accept_len"] < ACCEPT_LEN_FLOOR:
                report.error(
                    where,
                    f"boot_gate.accept_len {boot['accept_len']} is below the healthy band floor "
                    f"{ACCEPT_LEN_FLOOR} (collapsed reads 2.40-2.82); the run is void regardless of "
                    "verdict",
                )
            if boot.get("fallback_calls"):
                report.error(
                    where, "boot_gate records fallback calls: the measurement is not of this kernel"
                )
            engine = engines.get(engine_id)
            if engine is None:
                report.error(where, f"unknown engine {engine_id!r}")
            else:
                by_id = {gate["id"]: gate for gate in engine["gates"]}
                for gate in record.get("gates") or []:
                    gate_id = gate.get("engine_gate_id")
                    if gate_id is None:
                        continue
                    if gate_id not in by_id:
                        report.error(
                            where, f"gate {gate_id!r} is not in the {engine_id!r} gate ladder"
                        )
                    elif by_id[gate_id]["authority"] != "fidelity":
                        report.error(
                            where,
                            f"gate {gate_id!r} has authority {by_id[gate_id]['authority']!r}; a "
                            "fidelity attestation may only cite a fidelity gate",
                        )
        else:
            if boot["applicable"]:
                report.error(
                    where,
                    "boot_gate.applicable is true but the attestation names no engine: an offline "
                    "Q1 result has no boot to gate and must say so",
                )
            elif not boot.get("reason"):
                report.error(where, "boot_gate.applicable=false must carry a reason")

        workload_id = record.get("workload_profile")
        if workload_id and target is not None:
            excluded = (target.get("workload_limits") or {}).get("excluded_workload_profiles") or []
            if workload_id in excluded:
                report.error(
                    where,
                    f"target {record['target']!r} excludes workload {workload_id!r}; a fidelity "
                    "run that borrows a workload's GEOMETRY is not a run of that workload and must "
                    "not name it as one",
                )


def validate(root: pathlib.Path = ROOT) -> Report:
    """Run every schema check and cross-record invariant against ``root``."""
    report = Report()
    report.targets = check_targets(report, root)
    report.workloads = check_workloads(report, root, report.targets)
    report.engines = check_engines(report, root)
    check_manifests(report, root, report.targets, report.workloads, report.engines)
    attestations = check_attestations(
        report, root, report.targets, report.workloads, report.engines
    )
    fixture_sets = check_fixture_sets(report, root)
    envelopes = check_envelopes(report, root, fixture_sets)
    scorecards = check_scorecards(report, root, fixture_sets, envelopes)
    check_fidelity_attestations(
        report,
        root,
        report.targets,
        report.engines,
        fixture_sets,
        envelopes,
        scorecards,
        attestations,
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--root", default=None, help="registry root (default: this repository)")
    args = parser.parse_args(argv)

    report = validate(pathlib.Path(args.root).resolve() if args.root else ROOT)
    targets = report.targets
    workloads = report.workloads

    if args.verbose:
        for name in report.checked:
            print(f"  ok  {name}")

    print(
        f"{len(report.checked)} record(s) validated: "
        f"{len(targets)} target(s), {len(workloads)} workload(s), "
        f"{len(report.engines)} engine(s), "
        f"{len(report.fixture_sets)} fixture set(s), {len(report.envelopes)} envelope(s), "
        f"{len(report.scorecards)} scorecard(s)"
    )
    for line in report.blocked:
        print(f"BLOCKED  {line}")
    for line in report.errors:
        print(f"ERROR    {line}", file=sys.stderr)

    if report.errors:
        print(f"\n{len(report.errors)} error(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
