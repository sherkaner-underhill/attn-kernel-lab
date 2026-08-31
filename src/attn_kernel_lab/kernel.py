# SPDX-License-Identifier: Apache-2.0
"""Extension loader -- packaged wheel first, JIT second -- and source identity.

Adapted from the ``origin-private`` backend loader; the consuming serving
deployment's framework adapter calls this implementation instead of owning a
second copy. See ``THIRD_PARTY_NOTICES.md`` for the alias definition.

Three deliberate properties:

- **The gencode trap is handled here, once.** Architectures >= 9.0 need the
  arch-specific suffix (``sm_120a``): a bare ``-arch=sm_120a`` executable build
  silently lowers to ``sm_120`` and ptxas then rejects SM120a instructions
  (claim K5, reproduced on two machines). Every caller goes through
  :func:`gencode_for` -- the JIT path below and ``tools/build_wheel.py`` alike --
  so nobody re-makes that mistake.
- **Source identity is computed, not asserted.** :func:`source_build_id` hashes
  the exact files the extension is built from. A packaged wheel embeds that same
  digest at build time, which is what lets a deployment compare the binary it
  loaded against the sources a record describes.
- **Which binary ran is a reported fact, not an assumption.** A silent fallback
  from a packaged artifact to a locally recompiled one would make a benchmark or
  a qualification describe a binary nobody shipped, so :func:`load` records its
  choice and :func:`loaded_from` states it. ``identity.current_build()`` and the
  consumer's pin check are both built on that answer.

Precedence, and the one knob that overrides it:

    wheel (an installed ``attn_kernel_lab_ext``)  ->  JIT source build

``ATTN_KERNEL_LAB_LOADER=wheel`` demands the packaged path and raises if no
wheel is installed; ``=jit`` forces the source build even where a wheel exists,
which is how the test suite exercises both halves on one machine. ``=auto``
(the default, and the value of an unset variable) is the precedence above.
"""
from __future__ import annotations

import hashlib
import importlib
import os
import pathlib

_PKG = pathlib.Path(__file__).resolve().parent
_SOURCES = (_PKG / "csrc" / "fp8_prefill_attn.cu", _PKG / "quant.py")

#: The top-level module a packaged build installs.
#:
#: Deliberately NOT a submodule of this package. A wheel that shipped
#: ``attn_kernel_lab._fp8_prefill_attn`` would have to ship a second copy of
#: every ``.py`` file beside it, and whichever copy won the ``sys.path`` race
#: would decide which loader -- and which contract validation -- actually ran.
#: Keeping the binary in its own distribution leaves exactly one copy of the
#: Python surface in any environment, and makes "is a wheel installed?" a
#: question with one honest answer.
WHEEL_PACKAGE = "attn_kernel_lab_ext"

#: Environment override for :func:`load`: ``auto`` (default), ``wheel``, ``jit``.
LOADER_ENV = "ATTN_KERNEL_LAB_LOADER"

_PREFERENCES = ("auto", "wheel", "jit")

_ext = None
_loaded_from: str | None = None


def source_build_id() -> str:
    """SHA-256 over the canonical source files, stable across machines."""
    digest = hashlib.sha256()
    for path in _SOURCES:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def source_file_digests() -> dict[str, str]:
    """Per-file SHA-256 of the same source set, in :func:`source_build_id` order.

    A packaged wheel embeds this alongside the combined ``build_id`` so that a
    mismatch can say *which* file moved, from the wheel's own metadata, without
    the repository being present.
    """
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in _SOURCES}


def gencode_for(major: int, minor: int) -> str:
    arch = f"{major}{minor}a" if (major, minor) >= (9, 0) else f"{major}{minor}"
    return f"-gencode=arch=compute_{arch},code=sm_{arch}"


def arch_tag_for(major: int, minor: int) -> str:
    """``(12, 0) -> "sm_120a"`` -- the code-generation target :func:`gencode_for` emits."""
    return f"sm_{major}{minor}a" if (major, minor) >= (9, 0) else f"sm_{major}{minor}"


def loader_preference() -> str:
    """The resolved value of ``ATTN_KERNEL_LAB_LOADER``.

    An unrecognised value raises rather than defaulting quietly: a typo in a
    deployment's environment must not silently select a different binary from
    the one the operator asked for.
    """
    value = os.environ.get(LOADER_ENV, "").strip().lower() or "auto"
    if value not in _PREFERENCES:
        raise ValueError(
            f"{LOADER_ENV}={value!r} is not one of {list(_PREFERENCES)}"
        )
    return value


def _wheel_package():
    """The installed ``attn_kernel_lab_ext`` package, or ``None`` if there is none.

    Only "the distribution is not installed" is swallowed. An installed wheel
    that fails to import is a defect -- a torch ABI mismatch, a truncated
    install -- and falling back to a JIT rebuild there would convert a loud
    packaging failure into a quiet substitution of a different binary.
    """
    try:
        return importlib.import_module(WHEEL_PACKAGE)
    except ModuleNotFoundError as exc:
        if exc.name == WHEEL_PACKAGE:
            return None
        raise


def wheel_build_info() -> dict | None:
    """Build metadata embedded in the installed wheel, or ``None`` if none is installed.

    Importing this costs no torch import and no CUDA context: the packaged
    ``__init__`` keeps its build info in a plain Python module and defers the
    extension import to :func:`load`. A consumer can therefore run its pin check
    on a machine with no GPU at all, which is where a wrong pin is cheapest to
    discover.
    """
    package = _wheel_package()
    if package is None:
        return None
    info = getattr(package, "BUILD_INFO", None)
    if not isinstance(info, dict):
        raise RuntimeError(
            f"{WHEEL_PACKAGE} is installed but exposes no BUILD_INFO dict "
            "(built by an incompatible tools/build_wheel.py?)"
        )
    return dict(info)


def loaded_from() -> str | None:
    """``"wheel"``, ``"jit"``, or ``None`` when nothing has been loaded yet."""
    return _loaded_from


def load(extra_cuda_cflags: tuple[str, ...] = (), verbose: bool = False):
    """Return the extension: an installed wheel's if there is one, else a JIT build.

    The first successful call wins and is cached for the process, as before --
    ``extra_cuda_cflags`` therefore only influences a cold load. Passing flags at
    all forces the JIT path even when a wheel is installed: a caller asking for
    ``-DBM_D`` or ``--fmad=false`` is asking for a *different binary*, and
    handing back the packaged one would answer a numerics question with the
    wrong build.
    """
    global _ext, _loaded_from
    if _ext is not None:
        return _ext

    preference = loader_preference()
    if preference == "wheel" and extra_cuda_cflags:
        raise RuntimeError(
            f"{LOADER_ENV}=wheel cannot honour extra_cuda_cflags "
            f"{list(extra_cuda_cflags)}: a packaged wheel is fixed at build time. "
            "Use the JIT path for flag experiments."
        )
    if preference != "jit" and not extra_cuda_cflags:
        package = _wheel_package()
        if package is not None:
            _ext = package.load_extension()
            _loaded_from = "wheel"
            return _ext
        if preference == "wheel":
            raise RuntimeError(
                f"{LOADER_ENV}=wheel but no {WHEEL_PACKAGE} distribution is "
                "installed. Build one with tools/build_wheel.py and pip install it, "
                f"or unset {LOADER_ENV} to fall back to the JIT source build."
            )

    import torch
    from torch.utils.cpp_extension import load as _load

    major, minor = torch.cuda.get_device_capability()
    _ext = _load(
        name="attn_kernel_lab_fp8_prefill",
        sources=[str(_SOURCES[0])],
        extra_cuda_cflags=["-O3", gencode_for(major, minor), *extra_cuda_cflags],
        verbose=verbose,
    )
    _loaded_from = "jit"
    return _ext


def _reset() -> None:
    """Drop the cached extension so the next :func:`load` re-resolves its source.

    For tests that exercise both halves of the precedence in one process. It does
    not unload the compiled module -- Python cannot -- so a re-load returns the
    same object from ``sys.modules`` or the JIT cache; what it resets is the
    *decision*.
    """
    global _ext, _loaded_from
    _ext = None
    _loaded_from = None
