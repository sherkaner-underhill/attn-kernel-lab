# SPDX-License-Identifier: Apache-2.0
"""The Q-lane record rules, exercised from both sides.

Same posture as ``tests/test_registry_invariants.py``: a schema check proves a
record is well-formed, and these prove the rules that hold *between* records and
*inside* a metric block. The rules worth this much machinery are the ones a
scorecard can break while looking perfectly well-formed -- a `worst_layer` that
is not the worst layer, a ratio that does not divide, a `saturated: false` on a
metric whose control moved as far as the candidate did, a private capture
manifest that names where its tensors live.

Every record here is assembled through ``quality/fidelity_records.py`` -- the
same code the harness uses -- so a valid bundle is valid the way a real one is,
and each invalid case is one field away from it.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest
from validate_registry import validate

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "quality"))

import fidelity_records as rec  # noqa: E402

FIXTURE_SET_ID = "pub-test-v1"
SCORECARD_ID = "test-candidate-attention-output"
ENVELOPE_ID = "test-envelope-v1"

SLICE_SEMANTICS = {"layer": "fixture_case", "row": "query_row"}

ENVIRONMENT = {
    "device": "NVIDIA GeForce RTX 4090",
    "capability": "sm_89",
    "torch": "2.13.0+cu129",
    "cuda": "12.9",
    "allow_tf32_matmul": False,
    "cublas_workspace_config": ":4096:8",
}

# value / control triples per metric, chosen so the nested reduction holds:
# a coarser slice can never be worse than a finer one.
LEVELS = {
    "row_rel_l2": {
        "layers": [0.0100, 0.0120],
        "heads": [0.0130, 0.0200],
        "rows": [0.0300, 0.0500],
        "control": 0.0045,
    },
    "rel_l1": {
        "layers": [0.0090, 0.0110],
        "heads": [0.0120, 0.0180],
        "rows": [0.0250, 0.0400],
        "control": 0.0041,
    },
    "rmse": {
        "layers": [0.0020, 0.0025],
        "heads": [0.0030, 0.0040],
        "rows": [0.0060, 0.0090],
        "control": 0.0009,
    },
    "cos_sim": {
        "layers": [0.99950, 0.99900],
        "heads": [0.99930, 0.99850],
        "rows": [0.99900, 0.99700],
        "control": 0.999995,
    },
}


def _errors(report) -> str:
    return "\n".join(report.errors)


def _worst(entries, direction):
    pick = min if direction == "higher_is_better" else max
    return pick(entries, key=lambda entry: entry["value"])


def _metric(name: str) -> dict:
    direction, basis = rec.METRIC_FORM[name]
    level = LEVELS[name]
    control = level["control"]
    per_layer = [
        rec.slice_entry(basis, value, control, layer=i, layer_name=f"case{i}")
        for i, value in enumerate(level["layers"])
    ]
    heads = [
        rec.slice_entry(basis, value, control, layer=i, layer_name=f"case{i}", head=3 + i)
        for i, value in enumerate(level["heads"])
    ]
    rows = [
        rec.slice_entry(
            basis, value, control, layer=i, layer_name=f"case{i}", head=3 + i, row=11 + i
        )
        for i, value in enumerate(level["rows"])
    ]
    return rec.metric_block(
        name,
        mean=sum(level["layers"]) / len(level["layers"]),
        mean_control=control,
        per_layer=per_layer,
        worst_layer=_worst(per_layer, direction),
        worst_head=_worst(heads, direction),
        worst_row=_worst(rows, direction),
    )


def _fixture_set(**overrides) -> dict:
    record = {
        "schema_version": 1,
        "fixture_set_id": FIXTURE_SET_ID,
        "lane": "public_synth",
        "redistributable": True,
        "fixture_set_sha256": "",
        "provenance": {"generator": "tests", "media": "NONE: seeded synthetic tensors"},
        "geometry": {
            "head_dim": 256,
            "q_heads": 24,
            "kv_heads": 4,
            "page_size": 1,
            "mask": "bottom_right_causal",
            "mode": "extend",
            "q_len": 128,
        },
        "slice_semantics": SLICE_SEMANTICS,
        "cases": [
            {"layer": 0, "name": "case0", "depth": 4096, "prefix": 3968},
            {"layer": 1, "name": "case1", "depth": 4096, "prefix": 3968},
        ],
    }
    record.update(overrides)
    return rec.seal(record, "fixture_set_sha256")


def _scorecard(fixture_set: dict, **overrides) -> dict:
    record = {
        "schema_version": 1,
        "kind": "fidelity_scorecard",
        "scorecard_id": SCORECARD_ID,
        "boundary": "attention_output",
        "status": "draft",
        "recorded_utc": "2026-08-30T12:00:00Z",
        "subject": {
            "impl": "attn_kernel_lab.ops.prefill_extend",
            "mode": "declared_default_v1",
            "env_switches": {"qk_i8": True},
        },
        "reference": {"role": "fp32_reference", "impl": "fp32 sdpa"},
        "controls": [{"role": "implementation_swap", "impl": "bf16", "primary": True}],
        "fixture_set": {
            "id": fixture_set["fixture_set_id"],
            "fixture_set_sha256": fixture_set["fixture_set_sha256"],
            "lane": fixture_set["lane"],
            "layers": len(fixture_set["cases"]),
            "q_heads": 24,
            "slice_semantics": SLICE_SEMANTICS,
        },
        "environment": dict(ENVIRONMENT),
        "metrics": {
            **{name: _metric(name) for name in rec.GATE_METRICS},
            "norm_ratio": rec.reported_block(
                [{"layer": 0, "value": 0.9997}, {"layer": 1, "value": 1.0007}],
                note="reported, never gated",
            ),
            "nan_count": 0,
            "inf_count": 0,
        },
        "not_claimed": ["Not a model-quality statement."],
    }
    record.update(overrides)
    return record


def _envelope(fixture_set: dict, scorecard: dict, **overrides) -> dict:
    record = {
        "schema_version": 1,
        "kind": "fidelity_envelope",
        "envelope_id": ENVELOPE_ID,
        "status": "draft",
        "calibrated": False,
        "boundary": "attention_output",
        "recorded_utc": "2026-08-30T12:00:00Z",
        "subject": {"impl": "attn_kernel_lab.ops.prefill_extend", "mode": "declared_default_v1"},
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
                "scorecard_sha256": rec.canonical_digest(scorecard),
            }
        ],
        "decision_rule": {
            "aggregate_tolerance_pct": 10.0,
            "worst_slice_tolerance_pct": 0.0,
            "worst_slice_enforced": False,
            "hard_fails": ["nan or inf"],
        },
        "metrics": {
            name: {
                "direction": rec.METRIC_FORM[name][0],
                "ratio_basis": rec.METRIC_FORM[name][1],
                "mean_ratio": scorecard["metrics"][name]["ratio"],
                "worst_layer_ratio": scorecard["metrics"][name]["worst_layer"]["ratio"],
                "worst_head_ratio": scorecard["metrics"][name]["worst_head"]["ratio"],
                "worst_row_ratio": scorecard["metrics"][name]["worst_row"]["ratio"],
                "ratio_band": None,
            }
            for name in rec.GATE_METRICS
        },
        "not_claimed": ["Not a model-quality statement."],
    }
    record.update(overrides)
    return record


def _attestation(scorecard: dict, envelope: dict, fixture_set: dict, **overrides) -> dict:
    record = {
        "schema_version": 1,
        "kind": "fidelity",
        "artifact_manifest_sha256": "a" * 64,
        "release_id": "d256-int8-fp8-v0.3.0",
        "verdict": "pass",
        "recorded_utc": "2026-08-30T12:00:00Z",
        "approver": {"identity": "reviewer", "review": "PR #1"},
        "target": "sm89-rtx4090-local",
        "notes": ["Rests on a DRAFT envelope; the band is not frozen."],
        "fidelity": {
            "boundaries": ["attention_output"],
            "scorecard_sha256": [rec.canonical_digest(scorecard)],
            "envelope_id": envelope["envelope_id"],
            "envelope_sha256": rec.canonical_digest(envelope),
            "envelope_calibrated": envelope["calibrated"],
            "envelope_status": envelope["status"],
            "reference": {"role": "fp32_reference"},
            "control": {"role": "implementation_swap"},
            "fixture_sets": [
                {
                    "id": fixture_set["fixture_set_id"],
                    "fixture_set_sha256": fixture_set["fixture_set_sha256"],
                    "lane": fixture_set["lane"],
                }
            ],
            "worst": {
                "metric": "row_rel_l2",
                "worst_layer": scorecard["metrics"]["row_rel_l2"]["worst_layer"],
                "worst_head": scorecard["metrics"]["row_rel_l2"]["worst_head"],
                "worst_row": scorecard["metrics"]["row_rel_l2"]["worst_row"],
            },
            "spec_decode": "not_applicable",
            "boot_gate": {
                "applicable": False,
                "reason": "Q1 is engine-free; there is no boot to gate",
            },
        },
    }
    record.update(overrides)
    return record


@pytest.fixture
def qregistry(registry):
    """The registry fixture, plus the directories the Q lane records live in."""
    for parts in (("promotion", "envelopes"), ("quality", "scorecards"), ("quality", "fixtures")):
        registry.joinpath(*parts).mkdir(parents=True, exist_ok=True)
    return registry


@pytest.fixture
def bundle(qregistry):
    """Write one valid fixture set + scorecard + envelope + attestation, and hand
    back a mutator so a test can perturb exactly one field and rewrite."""
    state: dict = {}

    def _write(fixture_set=None, scorecard=None, envelope=None, attestation=..., **_):
        fixture_set = _fixture_set() if fixture_set is None else fixture_set
        scorecard = _scorecard(fixture_set) if scorecard is None else scorecard
        envelope = _envelope(fixture_set, scorecard) if envelope is None else envelope
        if attestation is ...:
            attestation = _attestation(scorecard, envelope, fixture_set)
        state.update(fixture_set=fixture_set, scorecard=scorecard, envelope=envelope)

        (qregistry / "quality" / "fixtures" / f"{fixture_set['fixture_set_id']}.json").write_text(
            json.dumps(fixture_set)
        )
        (qregistry / "quality" / "scorecards" / f"{scorecard['scorecard_id']}.json").write_text(
            json.dumps(scorecard)
        )
        suffix = ".draft" if envelope["status"] == "draft" else ""
        (
            qregistry / "promotion" / "envelopes" / f"{envelope['envelope_id']}{suffix}.json"
        ).write_text(json.dumps(envelope))
        if attestation is not None:
            (qregistry / "promotion" / "attestations" / "fidelity.json").write_text(
                json.dumps(attestation)
            )
        return validate(qregistry)

    _write.state = state
    return _write


# --- the valid bundle --------------------------------------------------------


def test_a_complete_fidelity_bundle_is_accepted(bundle):
    report = bundle()
    assert not report.errors, _errors(report)
    assert set(report.fixture_sets) == {FIXTURE_SET_ID}
    assert set(report.envelopes) == {ENVELOPE_ID}
    assert len(report.scorecards) == 1


def test_live_registry_still_validates_with_the_new_rules():
    """The published v0.3.0 records must stay valid: these rules are additive."""
    report = validate(ROOT)
    assert not report.errors, _errors(report)


def test_the_shipped_q1_public_records_are_present_and_valid():
    """Candidate zero's scorecard, its fixture set and the draft envelope are
    records, not scratch output; if they are in the tree they are validated."""
    scorecards = sorted((ROOT / "quality" / "scorecards").glob("*.json"))
    assert scorecards, "candidate zero's Q1-public scorecard is missing"
    report = validate(ROOT)
    assert not report.errors, _errors(report)
    assert len(report.scorecards) == len(scorecards)


# --- the four mandatory aggregations ----------------------------------------


@pytest.mark.parametrize("slice_name", ["worst_layer", "worst_head", "worst_row", "per_layer"])
def test_a_scorecard_missing_any_mandatory_aggregation_is_invalid(bundle, slice_name):
    """The worst-layer mandate and the addendum's worst-ROW mandate are enforced
    by the schema, not by discipline. An average-only scorecard would have passed
    a kernel whose worst layer was 56.40% cosine."""
    fixtures = _fixture_set()
    envelope = _envelope(fixtures, _scorecard(fixtures))
    card = _scorecard(fixtures)
    del card["metrics"]["row_rel_l2"][slice_name]
    report = bundle(fixture_set=fixtures, scorecard=card, envelope=envelope, attestation=None)
    assert any(slice_name in line and "schema" in line for line in report.errors), _errors(report)


def test_a_worst_layer_that_is_not_the_worst_layer_is_rejected(bundle):
    card = _scorecard(_fixture_set())
    card["metrics"]["row_rel_l2"]["worst_layer"] = card["metrics"]["row_rel_l2"]["per_layer"][0]
    report = bundle(scorecard=card, attestation=None)
    assert any("per_layer's worst is layer" in line for line in report.errors), _errors(report)


def test_a_worst_head_better_than_its_worst_layer_is_rejected(bundle):
    """They are nested reductions of one tensor; a coarser slice cannot be worse."""
    card = _scorecard(_fixture_set())
    block = card["metrics"]["row_rel_l2"]
    block["worst_head"] = dict(
        block["worst_head"], value=0.0001, ratio=0.0001 / 0.0045, saturated=True
    )
    report = bundle(scorecard=card, attestation=None)
    assert any("nested reduction" in line for line in report.errors), _errors(report)


def test_per_layer_must_cover_the_fixture_set(bundle):
    card = _scorecard(_fixture_set())
    card["metrics"]["rmse"]["per_layer"] = card["metrics"]["rmse"]["per_layer"][:1]
    report = bundle(scorecard=card, attestation=None)
    assert any("per_layer has 1 entries" in line for line in report.errors), _errors(report)


# --- the ratio and the saturation rule --------------------------------------


def test_a_ratio_that_does_not_divide_is_rejected(bundle):
    card = _scorecard(_fixture_set())
    card["metrics"]["row_rel_l2"]["ratio"] = 1.0
    report = bundle(scorecard=card, attestation=None)
    assert any("is not value/control" in line for line in report.errors), _errors(report)


def test_a_mislabelled_saturation_flag_is_rejected(bundle):
    """The rule is control >= 0.5 * value. Marking a null detector unsaturated is
    how a metric that cannot rank backends ends up ranking them."""
    card = _scorecard(_fixture_set())
    card["metrics"]["rel_l1"]["saturated"] = True
    report = bundle(scorecard=card, attestation=None)
    assert any("saturated=True but control" in line for line in report.errors), _errors(report)


def test_a_saturated_metric_must_say_so(bundle):
    """A control that has moved half as far as the candidate is saturated, and a
    scorecard claiming otherwise is rejected -- the reverse of the test above."""
    card = _scorecard(_fixture_set())
    block = card["metrics"]["rmse"]
    for entry in (
        [block]
        + block["per_layer"]
        + [block["worst_layer"], block["worst_head"], block["worst_row"]]
    ):
        entry["control"] = entry.get("mean", entry.get("value")) * 0.9
        value = entry.get("mean", entry.get("value"))
        entry["ratio"] = value / entry["control"]
        entry["saturated"] = False
    report = bundle(scorecard=card, attestation=None)
    assert any("saturated=False but control" in line for line in report.errors), _errors(report)


def test_a_similarity_anchored_on_the_wrong_basis_is_rejected(bundle):
    """Cosine 0.9985 over a control of 0.99997 is a fifty-fold loss of agreement;
    the direct quotient reports it as none, so the basis is pinned."""
    card = _scorecard(_fixture_set())
    card["metrics"]["cos_sim"]["ratio_basis"] = "direct"
    report = bundle(scorecard=card, attestation=None)
    assert any("higher_is_better/one_minus" in line for line in report.errors), _errors(report)


def test_norm_ratio_cannot_be_declared_gated(bundle):
    """Addendum item 2: reported, never gated. `gated` is a schema constant."""
    card = _scorecard(_fixture_set())
    card["metrics"]["norm_ratio"]["gated"] = True
    report = bundle(scorecard=card, attestation=None)
    assert any("norm_ratio/gated" in line for line in report.errors), _errors(report)


def test_exactly_one_control_must_be_primary(bundle):
    card = _scorecard(_fixture_set())
    card["controls"] = [dict(card["controls"][0]), dict(card["controls"][0], impl="bf16b")]
    report = bundle(scorecard=card, attestation=None)
    assert any("primary" in line for line in report.errors), _errors(report)


# --- determinism prerequisites ----------------------------------------------


def test_a_tf32_contaminated_reference_is_rejected(bundle):
    """A TF32-contaminated fp32 reference is not one (plan §4.2)."""
    card = _scorecard(_fixture_set())
    card["environment"]["allow_tf32_matmul"] = True
    report = bundle(scorecard=card, attestation=None)
    assert any("allow_tf32_matmul" in line for line in report.errors), _errors(report)


def test_an_unpinned_cublas_workspace_is_rejected(bundle):
    """Contract §3.1 makes the pin normative: split-K reduces with atomics, so
    without it the rotation -- and any envelope frozen from it -- is not
    reproducible run to run."""
    card = _scorecard(_fixture_set())
    card["environment"]["cublas_workspace_config"] = ":16:8"
    report = bundle(scorecard=card, attestation=None)
    assert any("cublas_workspace_config" in line for line in report.errors), _errors(report)


# --- fixture sets, and the privacy guard ------------------------------------


def test_an_edited_fixture_set_manifest_is_detected(bundle):
    fixtures = _fixture_set()
    fixtures["cases"][0]["depth"] = 8192
    fixtures["cases"][0]["prefix"] = 8064
    report = bundle(
        fixture_set=fixtures, scorecard=_scorecard(fixtures), envelope=None, attestation=None
    )
    assert any("stale fixture_set_sha256" in line for line in report.errors), _errors(report)


def test_a_private_fixture_set_may_not_name_where_its_tensors_live(qregistry):
    """The manifest exists so a private result is auditable WITHOUT the tensors,
    which is also what makes it the artifact that leaks if it may point at them."""
    fixtures = _fixture_set(lane="private_real", redistributable=False)
    fixtures["provenance"]["media"] = "PRIVATE:" + "b" * 64
    fixtures["cases"][0]["path"] = "/secure/activation-dumps/L0_chunk_0.pt"
    fixtures = rec.seal(fixtures, "fixture_set_sha256")
    (qregistry / "quality" / "fixtures" / f"{fixtures['fixture_set_id']}.json").write_text(
        json.dumps(fixtures)
    )
    report = validate(qregistry)
    assert any("name tensor paths" in line for line in report.errors), _errors(report)


def test_a_private_fixture_set_may_not_be_redistributable(qregistry):
    fixtures = rec.seal(_fixture_set(lane="private_real"), "fixture_set_sha256")
    (qregistry / "quality" / "fixtures" / f"{fixtures['fixture_set_id']}.json").write_text(
        json.dumps(fixtures)
    )
    report = validate(qregistry)
    assert any("may not be marked redistributable" in line for line in report.errors), _errors(
        report
    )


def test_a_case_whose_prefix_contradicts_its_depth_is_rejected(bundle):
    fixtures = _fixture_set()
    fixtures["cases"][1]["prefix"] = 100
    fixtures = rec.seal(fixtures, "fixture_set_sha256")
    report = bundle(
        fixture_set=fixtures, scorecard=_scorecard(fixtures), envelope=None, attestation=None
    )
    assert any("bottom-right causal diagonal" in line for line in report.errors), _errors(report)


# --- envelopes: a draft may not enforce, and only review may freeze ----------


def test_a_draft_envelope_may_not_enforce_its_worst_slice_thresholds(bundle):
    fixtures = _fixture_set()
    card = _scorecard(fixtures)
    envelope = _envelope(fixtures, card)
    envelope["decision_rule"]["worst_slice_enforced"] = True
    report = bundle(fixture_set=fixtures, scorecard=card, envelope=envelope, attestation=None)
    assert any("may not enforce" in line for line in report.errors), _errors(report)


def test_an_uncalibrated_envelope_may_not_enforce_its_worst_slice_thresholds(bundle):
    fixtures = _fixture_set()
    card = _scorecard(fixtures)
    envelope = _envelope(
        fixtures, card, status="frozen", approver={"identity": "reviewer", "review": "PR #2"}
    )
    envelope["decision_rule"]["worst_slice_enforced"] = True
    report = bundle(fixture_set=fixtures, scorecard=card, envelope=envelope, attestation=None)
    assert any("requires calibrated=true" in line for line in report.errors), _errors(report)


def test_freezing_an_envelope_requires_a_reviewer(bundle):
    fixtures = _fixture_set()
    card = _scorecard(fixtures)
    envelope = _envelope(fixtures, card, status="frozen")
    report = bundle(fixture_set=fixtures, scorecard=card, envelope=envelope, attestation=None)
    assert any("needs an approver" in line for line in report.errors), _errors(report)


def test_a_calibrated_frozen_envelope_may_enforce(bundle):
    """The rules gate the uncalibrated case, not the calibrated one."""
    fixtures = _fixture_set()
    card = _scorecard(fixtures)
    envelope = _envelope(
        fixtures,
        card,
        status="frozen",
        calibrated=True,
        approver={"identity": "reviewer", "review": "PR #2"},
    )
    envelope["decision_rule"]["worst_slice_enforced"] = True
    report = bundle(fixture_set=fixtures, scorecard=card, envelope=envelope, attestation=None)
    assert not report.errors, _errors(report)


def test_a_draft_envelope_is_named_as_one_on_disk(qregistry):
    fixtures = _fixture_set()
    card = _scorecard(fixtures)
    envelope = _envelope(fixtures, card)
    (qregistry / "promotion" / "envelopes" / f"{ENVELOPE_ID}.json").write_text(json.dumps(envelope))
    report = validate(qregistry)
    assert any("must be" in line and ".draft" in line for line in report.errors), _errors(report)


def test_a_scorecard_may_not_pass_against_a_draft_envelope(bundle):
    fixtures = _fixture_set()
    card = _scorecard(fixtures)
    card["envelope"] = {"envelope_id": ENVELOPE_ID, "comparison": "pass"}
    report = bundle(fixture_set=fixtures, scorecard=card, attestation=None)
    assert any("not yet a commitment" in line for line in report.errors), _errors(report)


# --- fidelity attestations ---------------------------------------------------


def test_a_fidelity_attestation_needs_its_fidelity_block(bundle):
    fixtures = _fixture_set()
    card = _scorecard(fixtures)
    envelope = _envelope(fixtures, card)
    attestation = _attestation(card, envelope, fixtures)
    del attestation["fidelity"]
    report = bundle(
        fixture_set=fixtures, scorecard=card, envelope=envelope, attestation=attestation
    )
    assert any("'fidelity' is a required property" in line for line in report.errors), _errors(
        report
    )


def test_a_fidelity_attestation_must_cite_evidence_that_is_in_the_tree(bundle):
    fixtures = _fixture_set()
    card = _scorecard(fixtures)
    envelope = _envelope(fixtures, card)
    attestation = _attestation(card, envelope, fixtures)
    attestation["fidelity"]["scorecard_sha256"] = ["c" * 64]
    report = bundle(
        fixture_set=fixtures, scorecard=card, envelope=envelope, attestation=attestation
    )
    assert any("is not in quality/scorecards" in line for line in report.errors), _errors(report)


def test_a_fidelity_attestation_may_not_misstate_its_envelope(bundle):
    fixtures = _fixture_set()
    card = _scorecard(fixtures)
    envelope = _envelope(fixtures, card)
    attestation = _attestation(card, envelope, fixtures)
    attestation["fidelity"]["envelope_calibrated"] = True
    report = bundle(
        fixture_set=fixtures, scorecard=card, envelope=envelope, attestation=attestation
    )
    assert any("records calibrated=False" in line for line in report.errors), _errors(report)


def test_resting_on_a_draft_envelope_must_be_said_out_loud(bundle):
    fixtures = _fixture_set()
    card = _scorecard(fixtures)
    envelope = _envelope(fixtures, card)
    attestation = _attestation(card, envelope, fixtures)
    attestation["notes"] = []
    report = bundle(
        fixture_set=fixtures, scorecard=card, envelope=envelope, attestation=attestation
    )
    assert any("DRAFT envelope must say so" in line for line in report.errors), _errors(report)


def test_private_activations_cannot_back_a_development_tier_result(bundle):
    """The 4090 is blocked from the private lane by data residency, not memory:
    a development-tier record derived from tensors that may not leave the
    production host
    could not have been produced."""
    fixtures = _fixture_set(lane="private_real", redistributable=False)
    fixtures["provenance"]["media"] = "PRIVATE:" + "b" * 64
    fixtures = rec.seal(fixtures, "fixture_set_sha256")
    card = _scorecard(fixtures)
    envelope = _envelope(fixtures, card)
    attestation = _attestation(card, envelope, fixtures, target="sm89-rtx4090-local")
    report = bundle(
        fixture_set=fixtures, scorecard=card, envelope=envelope, attestation=attestation
    )
    assert any("may not leave the production host" in line for line in report.errors), _errors(
        report
    )


def test_an_offline_result_may_not_claim_a_boot_gate(bundle):
    fixtures = _fixture_set()
    card = _scorecard(fixtures)
    envelope = _envelope(fixtures, card)
    attestation = _attestation(card, envelope, fixtures)
    attestation["fidelity"]["boot_gate"] = {"applicable": True, "accept_len": 4.36}
    report = bundle(
        fixture_set=fixtures, scorecard=card, envelope=envelope, attestation=attestation
    )
    assert any("no boot to gate" in line for line in report.errors), _errors(report)


def test_a_collapsed_boot_voids_an_engine_bound_result(bundle):
    """accept_len 2.40-2.82 is the collapsed spec graph. A scorecard measured
    through it is self-invalidating regardless of verdict -- the direct lesson of
    a downstream task score that had to be retracted."""
    fixtures = _fixture_set()
    card = _scorecard(fixtures)
    envelope = _envelope(fixtures, card)
    attestation = _attestation(
        card,
        envelope,
        fixtures,
        target="sm120-rtxpro6000-server",
        engine="sglang",
        engine_revision="1cf2b8c54d",
        gates=[{"name": "Q2 in-server fidelity", "result": "pass", "engine_gate_id": "Q2"}],
    )
    attestation["fidelity"]["boot_gate"] = {
        "applicable": True,
        "accept_len": 2.55,
        "capture_parity": True,
        "fallback_calls": 0,
    }
    report = bundle(
        fixture_set=fixtures, scorecard=card, envelope=envelope, attestation=attestation
    )
    assert any("below the healthy band floor" in line for line in report.errors), _errors(report)


def test_a_healthy_engine_bound_fidelity_result_is_accepted(bundle):
    fixtures = _fixture_set()
    card = _scorecard(fixtures)
    envelope = _envelope(fixtures, card)
    attestation = _attestation(
        card,
        envelope,
        fixtures,
        target="sm120-rtxpro6000-server",
        engine="sglang",
        engine_revision="1cf2b8c54d",
        gates=[{"name": "Q2 in-server fidelity", "result": "pass", "engine_gate_id": "Q2"}],
    )
    attestation["fidelity"]["boot_gate"] = {
        "applicable": True,
        "accept_len": 4.36,
        "capture_parity": True,
        "candidate_extend_calls": 224,
        "fallback_calls": 0,
    }
    report = bundle(
        fixture_set=fixtures, scorecard=card, envelope=envelope, attestation=attestation
    )
    assert not report.errors, _errors(report)


def test_a_fidelity_result_may_not_be_recorded_against_an_integration_gate(bundle):
    fixtures = _fixture_set()
    card = _scorecard(fixtures)
    envelope = _envelope(fixtures, card)
    attestation = _attestation(
        card,
        envelope,
        fixtures,
        target="sm120-rtxpro6000-server",
        engine="sglang",
        engine_revision="1cf2b8c54d",
        gates=[{"name": "borrowed", "result": "pass", "engine_gate_id": "S5"}],
    )
    attestation["fidelity"]["boot_gate"] = {
        "applicable": True,
        "accept_len": 4.36,
        "fallback_calls": 0,
    }
    report = bundle(
        fixture_set=fixtures, scorecard=card, envelope=envelope, attestation=attestation
    )
    assert any("may only cite a fidelity gate" in line for line in report.errors), _errors(report)


def test_a_fidelity_run_may_not_claim_a_workload_its_target_excludes(bundle):
    """Borrowing a workload's geometry is not running that workload."""
    fixtures = _fixture_set()
    card = _scorecard(fixtures)
    envelope = _envelope(fixtures, card)
    attestation = _attestation(card, envelope, fixtures, workload_profile="d256-24x4-446k")
    report = bundle(
        fixture_set=fixtures, scorecard=card, envelope=envelope, attestation=attestation
    )
    assert any("excludes workload" in line for line in report.errors), _errors(report)


def test_boundaries_must_match_the_cited_scorecards(bundle):
    fixtures = _fixture_set()
    card = _scorecard(fixtures)
    envelope = _envelope(fixtures, card)
    attestation = _attestation(card, envelope, fixtures)
    attestation["fidelity"]["boundaries"] = ["task"]
    report = bundle(
        fixture_set=fixtures, scorecard=card, envelope=envelope, attestation=attestation
    )
    assert any("do not match the cited scorecards" in line for line in report.errors), _errors(
        report
    )


# --- the engine profile ------------------------------------------------------


def test_the_sglang_ladder_places_the_q_rungs_where_the_plan_puts_them():
    """Q1 gates production-target entry (before S3), Q2 comes before S6, and Q3
    precedes downstream application tests. The ordered ladder is the record."""
    import yaml

    profile = yaml.safe_load((ROOT / "engines" / "profiles" / "sglang.yaml").read_text())
    order = [gate["id"] for gate in profile["gates"]]
    authority = {gate["id"]: gate["authority"] for gate in profile["gates"]}
    for gate_id in ("Q1", "Q2", "Q3"):
        assert authority[gate_id] == "fidelity", gate_id
    assert order.index("Q1") < order.index("S3")
    assert order.index("Q2") < order.index("S6")
    assert order.index("S6") < order.index("Q3")


def test_engine_profile_authority_enum_admits_fidelity():
    schema = json.loads((ROOT / "engines" / "schema" / "engine-profile.schema.json").read_text())
    enum = schema["properties"]["gates"]["items"]["properties"]["authority"]["enum"]
    assert "fidelity" in enum


def test_metric_form_agrees_between_the_harness_and_the_validator():
    """Two independent statements of the same rule; drift between them is the
    failure this test exists to catch."""
    import validate_registry

    assert rec.METRIC_FORM == validate_registry.METRIC_FORM
    assert set(rec.GATE_METRICS) == set(validate_registry.GATE_METRICS)
