# SPDX-License-Identifier: Apache-2.0
"""The cross-record rules the design depends on, exercised from both sides.

A schema check proves a record is well-formed. These prove the rules that hold
*between* records -- above all that a development-tier GPU can never author a
performance qualification, which is the invariant that makes the local 4090 tier
safe to rely on.
"""

from __future__ import annotations

import json
import pathlib

import yaml
from validate_registry import validate

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _errors(report) -> str:
    return "\n".join(report.errors)


def _write_target(registry, target_id, **overrides):
    record = yaml.safe_load((registry / "targets" / "sm89-rtx4090-local.yaml").read_text())
    record["id"] = target_id
    record.update(overrides)
    (registry / "targets" / f"{target_id}.yaml").write_text(yaml.safe_dump(record, sort_keys=False))


def _attestation(**overrides):
    record = {
        "schema_version": 1,
        "kind": "kernel_performance",
        "artifact_manifest_sha256": "a" * 64,
        "release_id": "d256-int8-fp8-v0.3.0",
        "verdict": "pass",
        "recorded_utc": "2026-08-29T00:00:00Z",
        "approver": {"identity": "reviewer", "review": "PR #1"},
        "target": "sm120-rtxpro6000-server",
        "workload_profile": "d256-24x4-446k",
        "measurement": {"lane": "inclusive", "timing_backend": "cuda_events"},
    }
    record.update(overrides)
    return record


# --- the fixture registry itself is clean -----------------------------------


def test_fixture_registry_is_valid(registry):
    report = validate(registry)
    assert not report.errors, _errors(report)
    assert set(report.targets) == {"sm120-rtxpro6000-server", "sm89-rtx4090-local"}


def test_live_registry_is_valid():
    report = validate(ROOT)
    assert not report.errors, _errors(report)


def test_live_protected_workload_is_unblocked_with_public_geometry_on_record():
    """The public profile retains the canonical geometry without private provenance."""
    report = validate(ROOT)
    assert not report.blocked, report.blocked
    profile = report.workloads["d256-24x4-446k"]
    assert profile["origin"]["unresolved"] == []
    assert profile["origin"]["resolved"] == []
    assert "446,335" in profile["origin"]["description"]
    assert profile["schedule"]["params"]["total_tokens"] == 446335


def test_a_new_unresolved_discrepancy_still_blocks(registry):
    """Resolving one discrepancy must not have weakened the guard for the next."""
    import yaml as _yaml

    path = registry / "workloads" / "profiles" / "d256-24x4-446k.yaml"
    profile = _yaml.safe_load(path.read_text())
    profile["origin"]["unresolved"] = ["hypothetical new mismatch"]
    path.write_text(_yaml.safe_dump(profile, sort_keys=False))
    report = validate(registry)
    assert any("unresolved" in line for line in report.blocked), report.blocked


# --- development hardware cannot qualify a release --------------------------


def test_performance_attestation_on_development_target_is_rejected(registry):
    (registry / "promotion" / "attestations" / "perf.json").write_text(
        json.dumps(_attestation(target="sm89-rtx4090-local"))
    )
    report = validate(registry)
    assert any("development" in line and "sm89-rtx4090-local" in line for line in report.errors), (
        _errors(report)
    )


def test_performance_attestation_on_production_target_is_accepted(registry):
    (registry / "promotion" / "attestations" / "perf.json").write_text(json.dumps(_attestation()))
    report = validate(registry)
    assert not report.errors, _errors(report)


def test_development_target_may_not_carry_promotion_thresholds(registry):
    _write_target(
        registry,
        "sm89-clone-local",
        promotion_thresholds={"min_inclusive_improvement_pct": 3.0},
    )
    report = validate(registry)
    assert any("must not define promotion_thresholds" in line for line in report.errors), _errors(
        report
    )


def test_production_target_must_declare_thresholds(registry):
    _write_target(registry, "sm120-thresholdless", authority="production", promotion_thresholds={})
    report = validate(registry)
    assert any("missing thresholds" in line for line in report.errors), _errors(report)


def test_production_target_may_not_be_unverified(registry):
    record = yaml.safe_load((registry / "targets" / "sm120-rtxpro6000-server.yaml").read_text())
    record["verification"]["state"] = "unverified"
    (registry / "targets" / "sm120-rtxpro6000-server.yaml").write_text(
        yaml.safe_dump(record, sort_keys=False)
    )
    report = validate(registry)
    assert any("unverified" in line for line in report.errors), _errors(report)


# --- measurement provenance --------------------------------------------------


def test_diagnostic_lane_has_no_promotion_authority(registry):
    (registry / "promotion" / "attestations" / "perf.json").write_text(
        json.dumps(_attestation(measurement={"lane": "core", "timing_backend": "cupti"}))
    )
    report = validate(registry)
    assert any("no promotion authority" in line for line in report.errors), _errors(report)


def test_timing_backend_must_be_recorded(registry):
    (registry / "promotion" / "attestations" / "perf.json").write_text(
        json.dumps(_attestation(measurement={"lane": "inclusive"}))
    )
    report = validate(registry)
    assert any("timing_backend" in line for line in report.errors), _errors(report)


def test_stale_workload_hash_in_attestation_is_rejected(registry):
    (registry / "promotion" / "attestations" / "perf.json").write_text(
        json.dumps(_attestation(workload_cases_sha256="9" * 64))
    )
    report = validate(registry)
    assert any("workload_cases_sha256" in line for line in report.errors), _errors(report)


def test_skipped_gate_needs_a_reason(registry):
    """A warning must never turn an incomplete qualification into a pass."""
    (registry / "promotion" / "attestations" / "perf.json").write_text(
        json.dumps(_attestation(gates=[{"name": "sanitizer", "result": "skipped"}]))
    )
    report = validate(registry)
    assert any("without a reason" in line for line in report.errors), _errors(report)


# --- manifests ---------------------------------------------------------------


def test_manifest_may_not_qualify_on_an_unsupported_target(registry, manifest):
    release = registry / "promotion" / "releases" / "d256-int8-fp8-v0.3.0"
    release.mkdir()
    (release / "artifact-manifest.json").write_text(
        json.dumps(
            manifest(
                targets={
                    "supported": ["sm120-rtxpro6000-server"],
                    "qualified": [
                        {
                            "target": "sm89-rtx4090-local",
                            "workload_profile": "d256-24x4-446k",
                            "attestation_sha256": "1" * 64,
                        }
                    ],
                    "excluded": [],
                }
            )
        )
    )
    report = validate(registry)
    assert any("not in `supported`" in line for line in report.errors), _errors(report)


def test_manifest_may_not_qualify_on_development_hardware(registry, manifest):
    release = registry / "promotion" / "releases" / "d256-int8-fp8-v0.3.0"
    release.mkdir()
    (release / "artifact-manifest.json").write_text(
        json.dumps(
            manifest(
                targets={
                    "supported": ["sm120-rtxpro6000-server", "sm89-rtx4090-local"],
                    "qualified": [
                        {
                            "target": "sm89-rtx4090-local",
                            "workload_profile": "d256-24x4-446k",
                            "attestation_sha256": "1" * 64,
                        }
                    ],
                    "excluded": [],
                }
            )
        )
    )
    report = validate(registry)
    assert any("only production-authority targets may qualify" in line for line in report.errors), (
        _errors(report)
    )


def test_manifest_may_not_both_support_and_exclude_a_target(registry, manifest):
    release = registry / "promotion" / "releases" / "d256-int8-fp8-v0.3.0"
    release.mkdir()
    (release / "artifact-manifest.json").write_text(
        json.dumps(
            manifest(
                targets={
                    "supported": ["sm120-rtxpro6000-server"],
                    "qualified": [],
                    "excluded": [{"target": "sm120-rtxpro6000-server", "reason": "no"}],
                }
            )
        )
    )
    report = validate(registry)
    assert any("both supported and excluded" in line for line in report.errors), _errors(report)


def test_release_id_must_not_encode_an_architecture(registry, manifest):
    """One release may support several targets; baking sm120 into the permanent
    identifier would make that a lie the moment a second target is qualified."""
    release = registry / "promotion" / "releases" / "bad"
    release.mkdir()
    (release / "artifact-manifest.json").write_text(
        json.dumps(manifest(release_id="sm120-d256-int8-fp8-v0.3.0"))
    )
    report = validate(registry)
    assert any("release_id" in line for line in report.errors), _errors(report)


def test_dirty_source_tree_may_not_be_released(registry, manifest):
    release = registry / "promotion" / "releases" / "d256-int8-fp8-v0.3.0"
    release.mkdir()
    manifest = manifest()
    manifest["source"]["dirty"] = True
    (release / "artifact-manifest.json").write_text(json.dumps(manifest))
    report = validate(registry)
    assert any("dirty" in line for line in report.errors), _errors(report)


# --- workload profiles -------------------------------------------------------


def test_stale_workload_hash_in_profile_is_rejected(registry):
    path = registry / "workloads" / "profiles" / "d256-24x4-446k.yaml"
    profile = yaml.safe_load(path.read_text())
    profile["schedule"]["params"]["chunk_size"] = 16384
    path.write_text(yaml.safe_dump(profile, sort_keys=False))
    report = validate(registry)
    assert any("stale cases_sha256" in line for line in report.errors), _errors(report)


def test_target_may_not_reference_an_unknown_workload(registry):
    _write_target(
        registry,
        "sm89-other-local",
        workload_limits={"excluded_workload_profiles": ["llama-nonexistent-8k"]},
    )
    report = validate(registry)
    assert any("unknown workload" in line for line in report.errors), _errors(report)
