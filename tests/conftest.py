# SPDX-License-Identifier: Apache-2.0
import json
import pathlib
import shutil
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402
from gen_workload import build_payload, digest  # noqa: E402

REGISTRIES = ("targets", "workloads", "promotion", "engines")


def pytest_terminal_summary(terminalreporter):
    """Say out loud that the GPU lane was dropped.

    ``tests/kernel/conftest.py`` sets ``collect_ignore_glob`` so its whole
    directory is not collected on a machine with no CUDA device. That is the
    intended behaviour, but silently: the run then looks like a full green pass
    that happens to contain no kernel correctness tests. This line is the
    difference between "the GPU lane passed" and "the GPU lane did not run".

    It lives HERE, not in ``tests/kernel/conftest.py``, because the release
    manifests bind ``source_tree_sha256`` over ``src`` and ``tests/kernel``;
    this file is outside that digest scope.
    """
    try:
        import torch

        gpu = torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        gpu = False
    if not gpu:
        n = len(sorted((ROOT / "tests" / "kernel").glob("test_*.py")))
        terminalreporter.write_line(
            f"tests/kernel: {n} GPU test files not collected (no CUDA device)"
        )


@pytest.fixture
def registry(tmp_path):
    """A minimal but valid registry copied from the live one.

    Deliberately built from the real profiles rather than hand-written stubs: a
    fixture that drifts from the records it stands in for tests nothing useful.
    """
    for sub in REGISTRIES:
        (tmp_path / sub / "schema").mkdir(parents=True)
        for schema in (ROOT / sub / "schema").glob("*.json"):
            shutil.copy(schema, tmp_path / sub / "schema" / schema.name)
    (tmp_path / "workloads" / "profiles").mkdir()
    (tmp_path / "engines" / "profiles").mkdir()
    (tmp_path / "promotion" / "releases").mkdir()
    (tmp_path / "promotion" / "attestations").mkdir()

    for target in ("sm120-rtxpro6000-server", "sm89-rtx4090-local"):
        shutil.copy(ROOT / "targets" / f"{target}.yaml", tmp_path / "targets")
    for engine in ("sglang", "vllm"):
        shutil.copy(ROOT / "engines" / "profiles" / f"{engine}.yaml", tmp_path / "engines" / "profiles")

    # Resolve the real profile's open discrepancy so BLOCKED does not mask the
    # invariant actually under test; that state has its own test.
    profile = yaml.safe_load(
        (ROOT / "workloads" / "profiles" / "d256-24x4-446k.yaml").read_text(encoding="utf-8")
    )
    profile["origin"]["unresolved"] = []
    profile["schedule"]["cases_sha256"] = digest(build_payload(profile))
    profile["schedule"]["cases_path"] = None
    (tmp_path / "workloads" / "profiles" / "d256-24x4-446k.yaml").write_text(
        yaml.safe_dump(profile, sort_keys=False), encoding="utf-8"
    )
    return tmp_path


@pytest.fixture
def manifest():
    """A valid artifact manifest, as a factory so a test can perturb one field."""

    def _make(**overrides):
        record = json.loads((ROOT / "tests" / "fixtures" / "manifest.json").read_text(encoding="utf-8"))
        record.update(overrides)
        return record

    return _make
