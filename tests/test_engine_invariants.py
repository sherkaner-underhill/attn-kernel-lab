# SPDX-License-Identifier: Apache-2.0
"""The engine axis.

An integration or application qualification is a fact about
``(artifact, engine, target, workload)``. These tests hold the engine coordinate
in place -- above all that a gate needing production hardware cannot be
satisfied on the development tier, which is the seam where a second engine would
otherwise quietly inherit the first one's evidence.
"""
from __future__ import annotations

import json
import pathlib

import pytest
import yaml
from validate_registry import validate

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _errors(report) -> str:
    return "\n".join(report.errors)


def _attestation(**overrides):
    record = {
        "schema_version": 1,
        "kind": "application",
        "artifact_manifest_sha256": "a" * 64,
        "release_id": "d256-int8-fp8-v0.3.0",
        "verdict": "pass",
        "recorded_utc": "2026-08-29T00:00:00Z",
        "approver": {"identity": "reviewer", "review": "PR #1"},
        "engine": "sglang",
        "engine_revision": "1cf2b8c54d",
        "target": "sm120-rtxpro6000-server",
        "workload_profile": "d256-24x4-446k",
    }
    record.update(overrides)
    return record


def _write(registry, record, name="att.json"):
    (registry / "promotion" / "attestations" / name).write_text(json.dumps(record))


def test_live_registry_registers_both_engines():
    report = validate(ROOT)
    assert not report.errors, _errors(report)
    assert set(report.engines) == {"sglang", "vllm"}


def test_planned_engine_needs_no_gate_ladder():
    """vllm is registered to hold the axis open, not to claim anything."""
    report = validate(ROOT)
    assert report.engines["vllm"]["status"] == "planned"
    assert report.engines["vllm"]["gates"] == []


def test_engine_attestation_is_accepted(registry):
    _write(registry, _attestation())
    report = validate(registry)
    assert not report.errors, _errors(report)


def test_application_attestation_must_name_an_engine(registry):
    _write(registry, _attestation(engine=None))
    report = validate(registry)
    assert any("must name an engine" in line for line in report.errors), _errors(report)


def test_unknown_engine_is_rejected(registry):
    _write(registry, _attestation(engine="tensorrt-llm"))
    report = validate(registry)
    assert any("unknown engine" in line for line in report.errors), _errors(report)


def test_engine_revision_must_be_recorded(registry):
    _write(registry, _attestation(engine_revision=None))
    report = validate(registry)
    assert any("engine_revision" in line for line in report.errors), _errors(report)


def test_gate_must_exist_in_that_engines_ladder(registry):
    """An SGLang attestation cannot cite a gate SGLang does not define."""
    _write(registry, _attestation(gates=[{"name": "made up", "result": "pass", "engine_gate_id": "S9"}]))
    report = validate(registry)
    assert any("not in the 'sglang' gate ladder" in line for line in report.errors), _errors(report)


def test_production_gate_cannot_pass_on_development_hardware(registry):
    """S6 (exact production geometry) needs the production card. Passing it on the
    4090 would be a claim the hardware physically cannot support -- 24 GiB does not
    hold the 446k pool."""
    _write(
        registry,
        _attestation(
            target="sm89-rtx4090-local",
            gates=[{"name": "exact production geometry", "result": "pass", "engine_gate_id": "S6"}],
        ),
    )
    report = validate(registry)
    assert any("requires production hardware" in line for line in report.errors), _errors(report)


def test_development_gate_may_pass_on_development_hardware(registry):
    """S1 (adapter and dispatch correctness) is exactly what the local tier is for."""
    _write(
        registry,
        _attestation(
            target="sm89-rtx4090-local",
            gates=[{"name": "adapter and dispatch correctness", "result": "pass", "engine_gate_id": "S1"}],
        ),
    )
    report = validate(registry)
    assert not report.errors, _errors(report)


def test_active_engine_must_declare_a_gate_ladder(registry):
    record = yaml.safe_load((registry / "engines" / "profiles" / "vllm.yaml").read_text())
    record["status"] = "active"
    (registry / "engines" / "profiles" / "vllm.yaml").write_text(yaml.safe_dump(record, sort_keys=False))
    report = validate(registry)
    assert any("must declare its gate ladder" in line for line in report.errors), _errors(report)


def test_open_engine_defect_must_name_the_gates_it_affects(registry):
    """A defect that changes what a gate RESULT MEANS is useless if the link is lost."""
    record = yaml.safe_load((registry / "engines" / "profiles" / "sglang.yaml").read_text())
    record["known_issues"][0]["affects_gates"] = []
    (registry / "engines" / "profiles" / "sglang.yaml").write_text(yaml.safe_dump(record, sort_keys=False))
    report = validate(registry)
    assert any("names no affected gates" in line for line in report.errors), _errors(report)


def test_defect_cannot_reference_a_gate_that_does_not_exist(registry):
    record = yaml.safe_load((registry / "engines" / "profiles" / "sglang.yaml").read_text())
    record["known_issues"][0]["affects_gates"] = ["S0", "S99"]
    (registry / "engines" / "profiles" / "sglang.yaml").write_text(yaml.safe_dump(record, sort_keys=False))
    report = validate(registry)
    assert any("affects unknown gates" in line for line in report.errors), _errors(report)


def test_manifest_cannot_integrate_a_planned_engine(registry, manifest):
    release = registry / "promotion" / "releases" / "d256-int8-fp8-v0.3.0"
    release.mkdir()
    record = manifest()
    record["engines"] = {
        "integrated": [
            {"engine": "vllm", "engine_revision": "abc123", "adapter_files_sha256": "1" * 64}
        ]
    }
    (release / "artifact-manifest.json").write_text(json.dumps(record))
    report = validate(registry)
    assert any("only an active engine can hold an integration" in line for line in report.errors), _errors(report)


def test_manifest_may_integrate_an_active_engine(registry, manifest):
    release = registry / "promotion" / "releases" / "d256-int8-fp8-v0.3.0"
    release.mkdir()
    record = manifest()
    record["engines"] = {
        "integrated": [
            {"engine": "sglang", "engine_revision": "1cf2b8c54d", "adapter_files_sha256": "1" * 64}
        ]
    }
    (release / "artifact-manifest.json").write_text(json.dumps(record))
    report = validate(registry)
    assert not report.errors, _errors(report)
