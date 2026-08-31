# SPDX-License-Identifier: Apache-2.0
"""The cross-compile gate, and the trap it exists to catch.

Skipped wherever nvcc is absent, so ordinary CPU CI stays green; run on the
development workstation these are the tests that keep the gate honest.
"""
from __future__ import annotations

import pathlib
import shutil

import pytest
import yaml
from probe_target import compile_gate

ROOT = pathlib.Path(__file__).resolve().parents[1]

needs_nvcc = pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc not on PATH")


@needs_nvcc
def test_gencode_form_can_emit_sm120a_without_an_sm120_device():
    """The capability the whole development tier rests on: production-architecture
    code can be compile-gated on hardware that cannot run it."""
    result = compile_gate("sm_120a")
    assert result["available"], result["stderr"]


@needs_nvcc
def test_bare_arch_form_silently_lowers_and_then_fails():
    """`-arch=sm_120a` lowers to sm_120, after which ptxas rejects the
    architecture-specific instruction. A trivial kernel would not catch this,
    which is why the probe source contains a real block-scaled MMA."""
    result = compile_gate("sm_120a", bare_arch=True)
    assert not result["available"], "the -arch trap did not reproduce; re-verify the probe source"
    assert "not supported on .target" in result["stderr"]
    assert "sm_120'" in result["stderr"]


@needs_nvcc
def test_trivial_source_would_not_have_caught_it():
    """Explicitly pins why the probe source is not a hello-world kernel."""
    assert compile_gate("sm_120a", bare_arch=True, arch_specific=False)["available"]


@needs_nvcc
def test_native_arch_builds():
    result = compile_gate("sm_89")
    assert result["probe"] == "trivial", "SM89 has no block-scaled MMA; probing for it proves nothing"
    assert result["available"], result["stderr"]


@needs_nvcc
def test_local_profile_declares_what_it_can_actually_cross_compile():
    profile = yaml.safe_load((ROOT / "targets" / "sm89-rtx4090-local.yaml").read_text(encoding="utf-8"))
    for arch in profile["toolchain"].get("cross_compile_verified") or []:
        assert compile_gate(arch)["available"], f"profile claims {arch} but nvcc cannot emit it"
