# SPDX-License-Identifier: Apache-2.0
"""The packaged wheel: its identity, its architectures, and which one loaded.

Three questions, each of which has burned somebody:

1. **Is this binary built from these sources?** The wheel embeds
   ``build_id`` -- sha256 over the canonical source set with exactly
   ``kernel.source_build_id()``'s semantics -- so the answer is a comparison
   rather than a belief. v0.3.0 had to record the JIT source build as its own
   artifact precisely because no wheel could answer this.

2. **Which loader ran?** A silent fall back from a packaged artifact to a local
   recompile makes a benchmark or a qualification describe a binary nobody
   shipped. ``ATTN_KERNEL_LAB_LOADER=wheel|jit`` forces each half so both can be
   exercised on one machine, and ``kernel.loaded_from()`` has to agree.

3. **Did the fat wheel really get both architectures?** Claim K5's build-time
   half: a bare ``-arch=sm_120a`` lowers to ``sm_120`` and ptxas rejects the
   architecture-specific MMA, so the packaging path must never emit one. The
   flag check below needs no GPU and no compiler, which is where it is cheap
   enough to run on every commit; the SASS check itself lives in
   ``tools/build_wheel.py`` (cuobjdump, at build time).

Everything here degrades cleanly: the tests that need the wheel skip where no
wheel is installed, the ones that need a GPU or nvcc skip on CPU-only CI, and
what is left -- the gencode flags and the declared binding surface -- runs
anywhere, torch or no torch.
"""
from __future__ import annotations

import importlib
import importlib.util
import pathlib
import re
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "tools"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import build_wheel  # noqa: E402
from attn_kernel_lab import capability, identity  # noqa: E402
from attn_kernel_lab import kernel as kernel_mod  # noqa: E402

CU_SOURCE = ROOT / "src" / "attn_kernel_lab" / "csrc" / "fp8_prefill_attn.cu"

#: The binding the consuming backend calls, in order. Written out rather than
#: derived, because "the wheel exposes what the JIT exposes" is only worth
#: asserting against an independent statement of what that is.
BINDING_ARGS = (
    "q8", "k8", "vt8", "vb16", "o",
    "qscale", "kscale", "vscale", "vlog2r", "vinvr", "vmean", "pv8_mask",
    "n", "prefix", "any_pv8", "all_pv8", "qk_i8", "lse",
)

_HAVE_WHEEL = importlib.util.find_spec(kernel_mod.WHEEL_PACKAGE) is not None

try:
    import torch

    _HAVE_GPU = torch.cuda.is_available()
except Exception:  # noqa: BLE001 -- CPU-only CI has no torch at all
    _HAVE_GPU = False

needs_wheel = pytest.mark.skipif(
    not _HAVE_WHEEL,
    reason=f"no {kernel_mod.WHEEL_PACKAGE} distribution installed "
           "(build one with tools/build_wheel.py)",
)
needs_jit = pytest.mark.skipif(
    not (_HAVE_GPU and shutil.which("nvcc")),
    reason="the JIT path needs a CUDA device and nvcc",
)


@pytest.fixture
def loader(monkeypatch):
    """Force one loader path, and leave the module-level cache as it was found.

    The cache is process-global by design (one extension per process), so a test
    that forces a path has to put it back: ``current_build()`` answers from it,
    and a leaked "wheel is loaded" would silently change what a later test in
    the same session sees.
    """

    def _force(preference: str | None):
        if preference is None:
            monkeypatch.delenv(kernel_mod.LOADER_ENV, raising=False)
        else:
            monkeypatch.setenv(kernel_mod.LOADER_ENV, preference)
        kernel_mod._reset()
        return kernel_mod.load()

    yield _force
    kernel_mod._reset()


def _signature(ext) -> str:
    """The pybind11 signature line of the compiled binding."""
    doc = ext.fp8_prefill_attn.__doc__ or ""
    return doc.strip().splitlines()[0]


# =========================================================================
# No GPU, no compiler, no wheel required
# =========================================================================


def test_packaging_emits_both_gencodes_and_never_a_bare_arch():
    """Claim K5, at the packaging layer.

    The flags are generated from each registered target's declared compute
    capability, so this pins the whole chain: both tiers are present, both use
    the ``-gencode`` form, and the SM120 one keeps its ``a`` suffix on both
    halves of the pair (``compute_120a`` *and* ``sm_120a`` -- dropping either is
    the silent lowering).
    """
    targets = build_wheel.resolve_targets(build_wheel.DEFAULT_TARGETS)
    flags = [target["gencode"] for target in targets]

    assert flags == [
        "-gencode=arch=compute_89,code=sm_89",
        "-gencode=arch=compute_120a,code=sm_120a",
    ]
    assert not any(flag.startswith("-arch=") for flag in flags)
    assert [target["arch"] for target in targets] == ["sm_89", "sm_120a"]


def test_declared_binding_surface_matches_the_source():
    """The argument list the wheel must expose is the one the ``.cu`` declares.

    Parsed from the source, not compiled: a rename or a reordering in the pybind
    block changes the operator's call surface for every consumer, and this is
    the cheapest place to notice.
    """
    block = CU_SOURCE.read_text(encoding="utf-8").split("PYBIND11_MODULE")[-1]
    assert tuple(re.findall(r'py::arg\("([^"]+)"\)', block)) == BINDING_ARGS
    assert 'py::arg("lse") = py::none()' in block, "lse must stay optional"


def test_an_unknown_loader_value_is_refused(monkeypatch):
    """A typo in the environment must not quietly select a different binary."""
    monkeypatch.setenv(kernel_mod.LOADER_ENV, "whee1")

    with pytest.raises(ValueError, match="ATTN_KERNEL_LAB_LOADER"):
        kernel_mod.loader_preference()


def test_a_packaged_build_is_not_a_flag_experiment(monkeypatch):
    """``extra_cuda_cflags`` asks for a different binary, and a wheel is fixed at
    build time. Refused outright rather than honoured by ignoring the flags,
    which would answer a numerics question with the wrong build."""
    monkeypatch.setenv(kernel_mod.LOADER_ENV, "wheel")
    kernel_mod._reset()
    try:
        with pytest.raises(RuntimeError, match="extra_cuda_cflags"):
            kernel_mod.load(extra_cuda_cflags=("-DBM_D=64",))
    finally:
        kernel_mod._reset()


@pytest.mark.skipif(_HAVE_WHEEL, reason="a wheel is installed, so it can be honoured")
def test_forcing_the_wheel_without_one_fails_loudly(loader):
    """``=wheel`` is a demand, not a preference: no silent JIT substitution."""
    with pytest.raises(RuntimeError, match="no attn_kernel_lab_ext"):
        loader("wheel")


# =========================================================================
# Wheel installed
# =========================================================================


@needs_wheel
def test_embedded_build_id_is_this_tree_s_source_build_id():
    """The property the whole artifact story rests on.

    Same digest, same semantics, computed independently on both sides: if these
    differ, the installed binary was built from other sources and every number
    attributed to it is attributed wrongly.
    """
    info = kernel_mod.wheel_build_info()

    assert info["build_id"] == kernel_mod.source_build_id(), (
        "the installed wheel was built from different sources than this tree; "
        "rebuild it with tools/build_wheel.py"
    )
    assert info["source_files"] == kernel_mod.source_file_digests()


@needs_wheel
def test_identity_fields_are_populated_and_self_consistent():
    build = identity.current_build()

    assert re.fullmatch(r"[0-9a-f]{64}", build.build_id)
    assert re.fullmatch(r"[0-9a-f]{64}", build.source_tree_sha256)
    assert build.release_id and build.artifact_variant_id and build.binary_abi

    # The contract the Python surface implements and the one the binary was
    # built against are the same contract, or the preprocessing pipeline and the
    # kernel disagree about the operator.
    assert build.operator_contract_version == capability.V1_CAPABILITY.operator_contract_version
    assert build.layout_version == capability.V1_CAPABILITY.layout_version
    assert build.package_api_version == identity.package_api_version()

    # A fat wheel, and every target spelled the way it is compiled.
    assert set(build.cuda_arch_targets) == {"sm_89", "sm_120a"}
    for arch in build.cuda_arch_targets:
        assert re.fullmatch(r"sm_\d+a?", arch)

    # The variant id is what a consumer pins, so it has to name the ABI and the
    # architectures rather than being an opaque string.
    assert build.binary_abi in build.artifact_variant_id
    for arch in build.cuda_arch_targets:
        assert arch.replace("_", "") in build.artifact_variant_id
    assert f"cp{sys.version_info.major}{sys.version_info.minor}" in build.binary_abi


@needs_wheel
def test_identity_satisfies_the_pin_check_it_exists_to_feed():
    """End to end: what the artifact reports is what a deployment can lock to."""
    build = identity.current_build()
    lock = dict(
        release_id=build.release_id,
        artifact_variant_id=build.artifact_variant_id,
        source_tree_sha256=build.source_tree_sha256,
    )

    identity.verify_pin(build, **lock)

    with pytest.raises(identity.PinMismatch):
        identity.verify_pin(build, **{**lock, "source_tree_sha256": "b" * 64})


@needs_wheel
def test_the_wheel_is_what_the_default_precedence_loads(loader):
    ext = loader(None)

    assert kernel_mod.loaded_from() == "wheel"
    assert ext.__name__.startswith(kernel_mod.WHEEL_PACKAGE)


@needs_wheel
def test_forced_wheel_path_reports_wheel_and_keeps_the_lse_argument(loader):
    ext = loader("wheel")
    signature = _signature(ext)

    assert kernel_mod.loaded_from() == "wheel"
    assert tuple(re.findall(r"(\w+): ", signature)) == BINDING_ARGS
    assert re.search(r"lse: [^,)]*= None\)", signature), signature


@needs_wheel
@pytest.mark.skipif(shutil.which("cuobjdump") is None, reason="cuobjdump not on PATH")
def test_the_installed_binary_carries_sass_for_both_tiers():
    """The claim the fat wheel exists to make, read back off the binary itself.

    ``tools/build_wheel.py`` runs this check before the wheel leaves the build
    host; repeating it against the *installed* module is what catches a wheel
    that was built correctly and then installed from somewhere else.
    """
    package = importlib.import_module(kernel_mod.WHEEL_PACKAGE)
    compiled = sorted(pathlib.Path(package.__file__).parent.glob("*.so"))
    assert len(compiled) == 1, compiled

    listing = subprocess.run(
        [shutil.which("cuobjdump"), "--list-elf", str(compiled[0])],
        capture_output=True, text=True, check=True,
    ).stdout
    for arch in kernel_mod.wheel_build_info()["cuda_arch_targets"]:
        assert f".{arch}." in listing, listing


# =========================================================================
# JIT path (GPU + nvcc)
# =========================================================================


@needs_jit
def test_forced_jit_path_reports_jit_and_refuses_an_artifact_identity(loader):
    """A JIT build is real but unreleased: no wheel, no container, no pinned
    code-generation targets. Inventing an identity for it is exactly what would
    let a pin check pass against a binary nobody can reproduce."""
    loader("jit")

    assert kernel_mod.loaded_from() == "jit"
    with pytest.raises(NotImplementedError, match="JIT source build"):
        identity.current_build()


@needs_jit
def test_the_jit_error_names_the_source_digest_it_can_honestly_report(loader):
    loader("jit")

    with pytest.raises(NotImplementedError) as excinfo:
        identity.current_build()
    assert kernel_mod.source_build_id() in str(excinfo.value)


@needs_wheel
@needs_jit
def test_wheel_and_jit_expose_the_same_binding_surface(loader):
    """The packaged path is a packaging change, not an API change."""
    wheel_signature = _signature(loader("wheel"))
    jit_signature = _signature(loader("jit"))

    assert wheel_signature == jit_signature
