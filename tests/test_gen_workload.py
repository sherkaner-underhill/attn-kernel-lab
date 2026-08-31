# SPDX-License-Identifier: Apache-2.0
"""The protected schedule is data. These tests lock its published values."""
from __future__ import annotations

import pathlib

import yaml
from gen_workload import build_payload, digest, expand_chunked_prefill

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROFILE = ROOT / "workloads" / "profiles" / "d256-24x4-446k.yaml"


def _profile() -> dict:
    return yaml.safe_load(PROFILE.read_text(encoding="utf-8"))


def test_schedule_matches_the_documented_progression():
    cases = expand_chunked_prefill(total_tokens=446335, chunk_size=32768)
    assert len(cases) == 14
    assert [c["q_len"] for c in cases] == [32768] * 13 + [20351]
    assert cases[0]["k_len"] == 32768
    assert cases[12]["k_len"] == 425984
    assert cases[13]["prefix_len"] == 425984
    assert cases[13]["k_len"] == 446335


def test_expected_operator_calls_are_224():
    """14 chunks x 16 full-attention layers. Asserted by the Gate S0 dispatch counters."""
    profile = _profile()
    payload = build_payload(profile)
    assert payload["totals"]["operator_calls"] == 224
    assert payload["totals"]["operator_calls"] == profile["expected_calls"]["per_request"]


def test_total_flops_agree_with_the_independent_analysis():
    """~39.17 PFLOP was derived separately in the prefill research doc. Cross-check."""
    pflop = build_payload(_profile())["totals"]["flops_total"] / 1e15
    assert abs(pflop - 39.17) < 0.05


def test_prefix_dominates_the_attended_pairs():
    """~92.9% of attended pairs are fully-visible prefix; that ratio motivates
    the prefix/triangle decomposition experiment, so it should not drift silently."""
    cases = expand_chunked_prefill(total_tokens=446335, chunk_size=32768)
    prefix_pairs = sum(c["q_len"] * c["prefix_len"] for c in cases)
    total_pairs = sum(c["attended_pairs"] for c in cases)
    assert 0.925 < prefix_pairs / total_pairs < 0.935


def test_recorded_hash_is_current():
    profile = _profile()
    assert profile["schedule"]["cases_sha256"] == digest(build_payload(profile))


def test_generated_cases_file_matches_the_profile():
    profile = _profile()
    generated = ROOT / profile["schedule"]["cases_path"]
    assert generated.exists(), "run tools/gen_workload.py --write"
    import json

    on_disk = json.loads(generated.read_text(encoding="utf-8"))
    assert on_disk["cases"] == build_payload(profile)["cases"]


def test_hash_is_sensitive_to_geometry_not_only_schedule():
    """Two profiles with the same chunk schedule but different head counts are
    different workloads and must not share a hash."""
    profile = _profile()
    other = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    other["geometry"]["q_heads"] = 32
    assert digest(build_payload(profile)) != digest(build_payload(other))
