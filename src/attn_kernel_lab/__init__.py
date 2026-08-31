# SPDX-License-Identifier: Apache-2.0
"""Low-precision fused paged prefill attention.

The canonical kernel and transform sources live here (adapted from the
``origin-private`` implementation described in ``THIRD_PARTY_NOTICES.md``):

- ``csrc/fp8_prefill_attn.cu`` -- the mma.sync-family fused kernel.
- ``quant.py`` -- the normative preprocessing pipeline (contract section 3).
- ``kernel.py`` -- JIT loader (arch-aware gencode) and source build identity.
- ``ops.py`` -- the public entry point, which validates the declared surface
  before any device work and then drives the pipeline above.

Alongside them, the consumer-facing interfaces: the declared capability
surface, the typed error that separates "outside the declared surface" from
every other failure, and the pin check a deployment runs at boot.
"""

from .capability import (
    CapabilityError,
    GENERALIZATION_CAPABILITY,
    OperatorCapability,
    V1_CAPABILITY,
    check_supported,
)
from .identity import BuildIdentity, PinMismatch, current_build, verify_pin
from .kernel import gencode_for, load, source_build_id
from .ops import PrefillRequest, check_request, plan_workspace, prefill_extend

__all__ = [
    "CapabilityError",
    "GENERALIZATION_CAPABILITY",
    "OperatorCapability",
    "V1_CAPABILITY",
    "check_supported",
    "BuildIdentity",
    "PinMismatch",
    "current_build",
    "verify_pin",
    "gencode_for",
    "load",
    "source_build_id",
    "PrefillRequest",
    "check_request",
    "plan_workspace",
    "prefill_extend",
]

__version__ = "0.0.0"
