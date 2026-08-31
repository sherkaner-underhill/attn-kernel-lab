# SPDX-License-Identifier: Apache-2.0
"""The declared surface, and the pin check, tested from both sides."""
from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attn_kernel_lab import (  # noqa: E402
    BuildIdentity,
    CapabilityError,
    PinMismatch,
    check_supported,
    current_build,
    verify_pin,
)

PRODUCTION = dict(
    head_dim=256,
    q_heads=24,
    kv_heads=4,
    page_size=1,
    mode="extend",
    mask="bottom_right_causal",
    kv_dtype="bfloat16",
)


def test_production_shape_is_supported():
    check_supported(**PRODUCTION)


@pytest.mark.parametrize(
    "override, expected_field",
    [
        ({"head_dim": 128}, "head_dim"),
        ({"page_size": 16}, "page_size"),
        ({"mode": "decode"}, "mode"),
        ({"mask": "noncausal"}, "mask"),
        ({"kv_dtype": "float8_e4m3fn"}, "kv_dtype"),
    ],
)
def test_outside_the_surface_raises_a_typed_error(override, expected_field):
    with pytest.raises(CapabilityError) as excinfo:
        check_supported(**{**PRODUCTION, **override})
    assert excinfo.value.field == expected_field


def test_decode_is_refused_because_v1_is_extend_only():
    """The prefill-only posture is a contract fact, not a deployment flag: decode,
    target verification, and draft execution must never reach this operator."""
    with pytest.raises(CapabilityError):
        check_supported(**{**PRODUCTION, "mode": "decode"})


def test_quantized_kv_pool_is_refused():
    """An FP8 pool would double-quantize; refusing is the contract, not a limitation
    to be worked around by the caller."""
    with pytest.raises(CapabilityError):
        check_supported(**{**PRODUCTION, "kv_dtype": "float8_e4m3fn"})


def test_lse_request_is_accepted_since_the_writeback_landed():
    check_supported(**PRODUCTION, return_lse=True)


def test_lse_refused_against_a_capability_without_it():
    """The check itself stays: a capability declaring returns_lse=False (a
    future implementation before its own writeback) still refuses."""
    from attn_kernel_lab.capability import OperatorCapability, V1_CAPABILITY
    import dataclasses

    no_lse = dataclasses.replace(V1_CAPABILITY, returns_lse=False)
    with pytest.raises(CapabilityError) as excinfo:
        check_supported(**PRODUCTION, return_lse=True, capability=no_lse)
    assert excinfo.value.field == "return_lse"


def test_capability_error_is_catchable_as_valueerror():
    """A consuming framework catches exactly this class and nothing broader."""
    with pytest.raises(ValueError):
        check_supported(**{**PRODUCTION, "head_dim": 64})


def _identity(**overrides) -> BuildIdentity:
    base = dict(
        build_id="candidate-zero",
        release_id="d256-int8-fp8-v0.3.0",
        artifact_variant_id="cp312-torch2.13-cu129-sm120a",
        operator_contract_version=1,
        layout_version=1,
        package_api_version=1,
        binary_abi="cp312-torch2.13-cu129",
        cuda_arch_targets=("sm_120a",),
        source_tree_sha256="a" * 64,
    )
    base.update(overrides)
    return BuildIdentity(**base)


LOCK = dict(
    release_id="d256-int8-fp8-v0.3.0",
    artifact_variant_id="cp312-torch2.13-cu129-sm120a",
    source_tree_sha256="a" * 64,
)


def test_matching_pin_passes():
    verify_pin(_identity(), **LOCK)


@pytest.mark.parametrize(
    "override", [{"release_id": "d256-int8-fp8-v0.4.0"}, {"artifact_variant_id": "other"}, {"source_tree_sha256": "b" * 64}]
)
def test_pin_mismatch_is_fatal(override):
    with pytest.raises(PinMismatch):
        verify_pin(_identity(**override), **LOCK)


def test_wrong_architecture_is_caught_before_run_time():
    with pytest.raises(PinMismatch, match="compute capability"):
        verify_pin(
            _identity(),
            **LOCK,
            device_compute_capability=(8, 9),
            supported_targets={"sm120-rtxpro6000-server": (12, 0)},
        )


def test_right_architecture_passes():
    verify_pin(
        _identity(),
        **LOCK,
        device_compute_capability=(12, 0),
        supported_targets={"sm120-rtxpro6000-server": (12, 0)},
    )


def test_current_build_refuses_to_invent_an_identity():
    """A stub identity would let a pin check pass against nothing at all.

    Since A2 the honest behaviour is two-sided: with no packaged wheel
    importable (or the JIT development build loaded) ``current_build`` must
    still raise rather than invent fields; with a wheel available it must
    return that wheel's own populated identity.
    """
    from attn_kernel_lab import kernel as kernel_mod

    if kernel_mod.loaded_from() == "jit" or kernel_mod.wheel_build_info() is None:
        with pytest.raises(NotImplementedError):
            current_build()
    else:
        identity = current_build()
        assert identity.build_id
        assert identity.source_tree_sha256
        assert identity.package_api_version >= 1
        assert identity.cuda_arch_targets
