# SPDX-License-Identifier: Apache-2.0
"""Build identity, and the fail-closed pin check a consumer runs at boot.

A pin that nothing verifies is a comment. The consuming repository records a
``release_id``, ``artifact_manifest_sha256``, and ``artifact_variant_id``; the
running process must be able to state what it actually loaded, and refuse to
start when the two disagree.

Failing closed matters more than it looks. The alternative -- warn and continue
-- produces a server that is serving a different kernel from the one its evidence
describes, which is indistinguishable from correct until someone tries to
attribute a regression.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


class PinMismatch(RuntimeError):
    """The loaded artifact is not the pinned one. Always fatal."""


@dataclass(frozen=True)
class BuildIdentity:
    """What the loaded artifact reports about itself."""

    build_id: str
    release_id: str
    artifact_variant_id: str
    operator_contract_version: int
    layout_version: int
    package_api_version: int
    binary_abi: str
    cuda_arch_targets: tuple[str, ...]
    source_tree_sha256: str

    def as_dict(self) -> dict:
        record = asdict(self)
        record["cuda_arch_targets"] = list(self.cuda_arch_targets)
        return record


#: Every field a wheel must carry for its identity to be usable by a pin check.
_REQUIRED_INFO = (
    "build_id",
    "release_id",
    "artifact_variant_id",
    "operator_contract_version",
    "layout_version",
    "package_api_version",
    "binary_abi",
    "cuda_arch_targets",
    "source_tree_sha256",
)


def package_api_version() -> int:
    """The public API version this source tree exposes (contract section 2).

    Derived, never written down twice. Exposing the LSE output through the
    public API *is* the ``package_api_version`` bump, so the number follows the
    declared capability; a literal repeated in the build tool and in a release
    record is a claim that can go stale in one place without anyone noticing,
    which is the failure ``make_candidate_records.py`` already avoids for
    ``returns_lse``.
    """
    from .capability import V1_CAPABILITY

    return 2 if V1_CAPABILITY.returns_lse else 1


def current_build() -> BuildIdentity:
    """Identity of the packaged kernel artifact available to this process.

    Read from the ``BUILD_INFO`` a wheel embeds at build time
    (``tools/build_wheel.py``), so it costs no torch import, no CUDA context and
    no compiler: a consumer can run its pin check before it touches a device,
    which is where a wrong pin is cheapest to discover.

    Raises ``NotImplementedError`` in the two cases where no artifact identity
    exists, and it is deliberately not a placeholder value in either: a stub
    identity would let a consumer's pin check pass against nothing at all, which
    is worse than having no check.

    1. No packaged wheel is installed.
    2. A JIT source build is loaded. That build is real, but it has no release,
       no variant, no build container and no fixed code-generation targets --
       every field a pin compares would have to be invented here. The honest
       development identity is ``kernel.source_build_id()``, and a deployment
       that reaches this line is running a binary its records do not describe.
    """
    from . import kernel as kernel_mod

    if kernel_mod.loaded_from() == "jit":
        raise NotImplementedError(
            "a JIT source build is loaded: a development artifact with no "
            "release identity (no wheel, no build container, no pinned "
            "code-generation targets). Its source digest is "
            f"{kernel_mod.source_build_id()}; install a packaged wheel "
            "(tools/build_wheel.py) for an identity a pin check can verify."
        )

    info = kernel_mod.wheel_build_info()
    if info is None:
        raise NotImplementedError(
            f"no kernel artifact is available: no {kernel_mod.WHEEL_PACKAGE} "
            "distribution is installed and nothing has been loaded. Build and "
            "install one with tools/build_wheel.py, or use "
            "kernel.source_build_id() for the source-level identity the JIT "
            "development path can honestly report."
        )

    missing = [
        field
        for field in _REQUIRED_INFO
        if not info.get(field) or (field.endswith("_version") and int(info[field]) < 1)
    ]
    if missing:
        raise RuntimeError(
            f"{kernel_mod.WHEEL_PACKAGE} is installed but its BUILD_INFO is "
            f"missing or empty at {missing}: it was built by an incompatible "
            "tools/build_wheel.py and cannot state its own identity"
        )

    return BuildIdentity(
        build_id=str(info["build_id"]),
        release_id=str(info["release_id"]),
        artifact_variant_id=str(info["artifact_variant_id"]),
        operator_contract_version=int(info["operator_contract_version"]),
        layout_version=int(info["layout_version"]),
        package_api_version=int(info["package_api_version"]),
        binary_abi=str(info["binary_abi"]),
        cuda_arch_targets=tuple(str(arch) for arch in info["cuda_arch_targets"]),
        source_tree_sha256=str(info["source_tree_sha256"]),
    )


def verify_pin(
    identity: BuildIdentity,
    *,
    release_id: str,
    artifact_variant_id: str,
    source_tree_sha256: str,
    device_compute_capability: tuple[int, int] | None = None,
    supported_targets: dict[str, tuple[int, int]] | None = None,
) -> None:
    """Raise ``PinMismatch`` unless ``identity`` matches the consumer's lock.

    ``supported_targets`` maps target id to compute capability. When it and
    ``device_compute_capability`` are both given, the running device must match
    one of them -- the check that catches an artifact built for the wrong
    architecture before it silently falls back at run time.
    """
    for name, actual, expected in (
        ("release_id", identity.release_id, release_id),
        ("artifact_variant_id", identity.artifact_variant_id, artifact_variant_id),
        ("source_tree_sha256", identity.source_tree_sha256, source_tree_sha256),
    ):
        if actual != expected:
            raise PinMismatch(
                f"{name} mismatch: loaded {actual!r}, pinned {expected!r}"
            )

    if device_compute_capability is not None and supported_targets is not None:
        capabilities = set(supported_targets.values())
        if device_compute_capability not in capabilities:
            raise PinMismatch(
                f"device compute capability {device_compute_capability} is not among the "
                f"artifact's supported targets {sorted(capabilities)}"
            )
