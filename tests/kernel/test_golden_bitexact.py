# SPDX-License-Identifier: Apache-2.0
"""Bit-exact golden regression harness for the fp8 prefill attention kernel.

=============================================================================
WHY THIS FILE EXISTS
=============================================================================
The kernel performance phase changes ONLY scheduling: an XOR smem swizzle in
place of the +16-byte padded rows, cp.async pipeline depth / wait-group
restructures, occupancy work (smem diet, launch bounds), ldmatrix patterns.
None of those touch the arithmetic.  The kv-tile loop order, the k-step order
inside each mma chain, the online-softmax rescale order and every fp32
accumulation order are fixed by the source and must stay fixed.

    THE IRON RULE
    A scheduling-only change must be BIT-EXACT -- identical output bytes --
    for every mode and every shape.  Numerics changes are only ever
    intentional and reviewed.

This file is what enforces the rule.  For a fixed matrix of modes and shapes
it hashes the exact bytes of the kernel output AND of every quantized
artifact that feeds the kernel, and compares against checked-in goldens.
Hashing the intermediates separately is what makes a failure actionable: if
``out`` moved and ``q8``/``k8``/``vt8``/``vb16``/the scale arrays did not,
the change is in the .cu; if the quantized inputs moved, it is in quant.py
and the output difference is merely downstream of it.

=============================================================================
REGENERATION POLICY -- READ BEFORE YOU TOUCH golden_bitexact.json
=============================================================================
Regenerate goldens ONLY for an INTENTIONAL, REVIEWED numerics change: a new
quantization scheme, a changed scale fold, different rounding, a new mma
shape, a compiler-flag change that alters fma contraction.

  * Regenerate in ITS OWN COMMIT.  That commit contains the numerics change
    and the new golden_bitexact.json and nothing else.
  * NEVER bundle a regeneration with a scheduling / performance change.  A
    perf commit that also moves the goldens has destroyed the only evidence
    that the perf change was safe, and there is no way to get it back.
  * The commit message must state which artifact digests moved and why.
  * If a change you believed was scheduling-only moves a digest, the change
    is NOT scheduling-only.  Find out why BEFORE regenerating anything.
    Candidates, in order of how often they are the answer: an accumulation
    order change smuggled in with a pipeline restructure; a changed
    ``extra_cuda_cflags`` (``-use_fast_math``, ``--fmad``, ``-maxrregcount``,
    a different ``-gencode``) which changes codegen and therefore rounding;
    a reordered epilogue; different ambient torch matmul precision
    (see ``_pinned_numerics`` below).

=============================================================================
RUNNING IT (needs an SM120-class GPU + CUDA toolchain for the JIT)
=============================================================================
    cd tests/kernel

    # 1. first ever run / after a REVIEWED numerics change: write goldens
    python3 -m pytest test_golden_bitexact.py -q --write-golden
    #    (equivalently: GOLDEN_WRITE=1 python3 -m pytest test_golden_bitexact.py -q)
    #    then `git add golden_bitexact.json` and commit it ON ITS OWN.

    # 2. every run thereafter: check
    python3 -m pytest test_golden_bitexact.py -q

    # 3. the whole safety suite before/after a scheduling change
    python3 -m pytest test_kernel_vs_sdpa.py test_divergence_hunting.py \
                     test_golden_bitexact.py -q

Goldens are machine-specific by construction (they encode cuBLAS algorithm
choice, the GPU's reduction shapes and the compiled SASS).  Regenerate them
if the qualification GPU, CUDA toolkit or torch version changes -- the recorded
``env`` block in the json is there so you can tell.  A changed environment
is a legitimate reason to regenerate; it is still its own commit.

=============================================================================
WHAT IS COVERED
=============================================================================
modes  : {qk_i8} x {rotate} x {center_k}  (all 8 combinations on the two
         structurally richest shapes) x {all-fp8 PV, all-bf16 PV, mixed mask}
shapes : the (128,1024,896) anchor, a ragged-everything (100,999,899), a
         single-ragged-sub-tile (64,63,0), the production GQA shape
         (H=24,KVH=4,grp=6, T=192 so M spans several CTAs), a multi-chunk
         composition with a ragged last chunk, and a scattered page_size-1
         pool (the gather index path).

The PV axis is also the CTA TILE-SHAPE axis.  The launcher picks the wide
tile (BM=128) only when every head is fp8-PV -- the bf16-PV V staging buffer
does not fit beside a 128-row Q tile in SM120's 99 KB opt-in -- so:

    pv=fp8    -> wide  (BM=128) path
    pv=bf16   -> narrow (BM=64)  path
    pv=mixed  -> narrow (BM=64)  path, with both PV branches live

Every shape is therefore run through both tile widths.  That matters: a
wider CTA sweeps kv tiles that are fully causal-masked for its early rows,
and bit-exactness depends on those contributing p == 0 / alpha == 1 exactly
(and on the V staging rows they touch being zero rather than NaN -- the
0 * NaN class of bug that already bit this kernel once).
"""

import datetime as _dt
import hashlib
import json
import math
import os
import platform
import sys
import warnings

import pytest
import torch

HEAD_DIM = 256
PKG = os.path.join(os.path.dirname(__file__), "..", "..", "src", "attn_kernel_lab")
sys.path.insert(0, os.path.abspath(PKG))
sys.path.insert(0, os.path.abspath(os.path.join(PKG, "..")))

import quant as q_mod  # noqa: E402  (the package's quant.py, imported standalone)
from attn_kernel_lab import kernel as kernel_mod  # noqa: E402


# =========================================================================
# Target-capability gate (lab import, 2026-08-29).
#
# These goldens are BIT-EXACT records pinned to one (device capability,
# toolchain) pair -- currently the SM120 RTX PRO 6000 they were generated on.
# On any other device the correct outcome is SKIP, never a tolerance widen
# and never a spurious red suite: bit-exactness is not portable across
# architectures by design (targets/README.md, Phase 0 inventory).
# Regeneration on a new pair is a deliberate, reviewed act via --write-golden.
# =========================================================================
def _golden_capability_gate():
    import json as _json

    path = os.path.join(os.path.dirname(__file__), "golden_bitexact.json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            recorded = _json.load(fh).get("env", {}).get("capability")
    except (OSError, ValueError):
        return  # absent/unreadable goldens: the store's own banner handles it
    if recorded is None or not torch.cuda.is_available():
        return
    dev = torch.cuda.get_device_properties(0)
    current = f"{dev.major}.{dev.minor}"
    if current != recorded:
        pytest.skip(
            f"golden_bitexact.json is pinned to capability {recorded}; this device "
            f"is {current}. Bit-exact goldens are not portable across architectures "
            f"-- run this lane on the pinned target, or regenerate deliberately "
            f"with --write-golden on the new one.",
            allow_module_level=True,
        )


_golden_capability_gate()

DEV = "cuda"
GOLDEN_PATH = os.environ.get(
    "GOLDEN_FILE", os.path.join(os.path.dirname(__file__), "golden_bitexact.json")
)
GOLDEN_SCHEMA = 1

# q-head indices routed to the bf16-PV path in the "mixed" PV mode.  Valid for
# every H in the matrix (min H is 8) and, at H=24/KVH=4/grp=6, both indices
# land in kv group 0, so the mixed case also exercises two different PV paths
# inside one kv head -- the configuration the per-head dial actually ships.
MIXED_BF16_HEADS = (2, 5)


# =========================================================================
# environment pinning
# =========================================================================

_PIN_ATTRS = (
    # (module path, attribute, pinned value)
    ("torch.backends.cuda.matmul", "allow_tf32", False),
    ("torch.backends.cudnn", "allow_tf32", False),
    ("torch.backends.cuda.matmul", "allow_fp16_reduced_precision_reduction", False),
    ("torch.backends.cuda.matmul", "allow_bf16_reduced_precision_reduction", False),
)


def _resolve(path):
    obj = torch
    for part in path.split(".")[1:]:
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


@pytest.fixture(scope="module", autouse=True)
def _pinned_numerics():
    """Pin every global torch knob the goldens depend on.

    ``quant.py`` runs an fp32 GEMM for the Hadamard rotation.  If TF32 is
    enabled globally (``allow_tf32``, or ``set_float32_matmul_precision`` at
    "high"/"medium") that GEMM silently drops to a 10-bit mantissa and every
    downstream digest moves -- from AMBIENT STATE, not from any code change.
    SGLang and other libraries do set these flags.  Pin them here and restore
    on teardown so the rest of the session is unaffected.
    """
    saved = []
    for path, attr, val in _PIN_ATTRS:
        obj = _resolve(path)
        if obj is not None and hasattr(obj, attr):
            saved.append((obj, attr, getattr(obj, attr)))
            setattr(obj, attr, val)
    prec = None
    if hasattr(torch, "get_float32_matmul_precision"):
        prec = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")
    yield
    for obj, attr, val in saved:
        setattr(obj, attr, val)
    if prec is not None:
        torch.set_float32_matmul_precision(prec)


def _cuda_cflags():
    """The exact nvcc flags the ``ext`` fixture builds with.

    Recorded in the golden env block on purpose.  Compiler flags are
    NUMERICS, not scheduling: -use_fast_math, --fmad, -maxrregcount and a
    different -gencode all change fma contraction and therefore output bits.
    The kernel also carries -DBM_D / -DNW_D tile-shape overrides, which
    change the binary without changing a single source file -- so without
    this record a golden mismatch after a `-DBM_D=64` A/B would look like an
    unexplained numerics move.

    Derived from ``kernel.py`` rather than restated here, because the fixture
    now builds through ``kernel_mod.load()``: this list is a claim about the
    binary the goldens were produced by, and a hand-copied duplicate of the
    loader's flags is a claim that can go stale without anyone noticing.  It
    must stay the exact flag list ``load()`` passes -- ``["-O3", <gencode>]``,
    the loader's default ``extra_cuda_cflags=()`` contributing nothing.  The
    string this returns is unchanged by the consolidation, so previously
    written goldens still compare equal.
    """
    major, minor = torch.cuda.get_device_capability()
    return ["-O3", kernel_mod.gencode_for(major, minor)]


def _env_fingerprint():
    have = torch.cuda.is_available()
    dev = torch.cuda.get_device_properties(0) if have else None
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device_name": dev.name if dev else None,
        "capability": f"{dev.major}.{dev.minor}" if dev else None,
        "multi_processor_count": dev.multi_processor_count if dev else None,
        "shared_mem_per_block_optin": (
            getattr(dev, "shared_memory_per_block_optin", None) if dev else None
        ),
        "extra_cuda_cflags": _cuda_cflags() if have else None,
        "kernel_src_sha256": _src_sha(),
        "python": platform.python_version(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "allow_tf32": bool(getattr(torch.backends.cuda.matmul, "allow_tf32", False)),
        "float32_matmul_precision": (
            torch.get_float32_matmul_precision()
            if hasattr(torch, "get_float32_matmul_precision")
            else None
        ),
    }


def _src_sha():
    """sha256 of the .cu and quant.py the goldens were written against, so a
    mismatch report can say whether the sources moved at all."""
    out = {}
    for name, path in (
        ("fp8_prefill_attn.cu", os.path.join(PKG, "csrc", "fp8_prefill_attn.cu")),
        ("quant.py", os.path.join(PKG, "quant.py")),
    ):
        try:
            with open(path, "rb") as f:
                out[name] = hashlib.sha256(f.read()).hexdigest()[:16]
        except OSError:
            out[name] = None
    return out


# =========================================================================
# JIT extension -- same fixture as test_kernel_vs_sdpa.py / test_divergence_hunting.py
# =========================================================================


@pytest.fixture(scope="module")
def ext():
    """The one extension the whole repo builds: ``attn_kernel_lab_fp8_prefill``.

    GOLDEN-RELEVANT, so read this before changing it.  The goldens are bits, and
    bits depend on the flags nvcc was given, not on which fixture asked for the
    build.  The extension NAME enters no digest -- but it does select the build
    directory (``cpp_extension`` keys that on the name alone), so the old
    private name meant the goldens were checked against a SEPARATE compile of
    the same source rather than against the binary the public path ships.  They
    now share one, which is strictly the more honest arrangement: the bits the
    goldens pin are the bits ``attn_kernel_lab.ops`` executes.

    ``_env_fingerprint`` records ``_cuda_cflags()`` precisely to catch an
    accidental codegen change, so that function is derived from ``kernel.py``'s
    own flags and MUST be kept equal to what ``kernel_mod.load()`` passes.  If a
    build flag is ever added to the loader, add it there too or the fingerprint
    starts lying.

    A test that needs DIFFERENT ``-D`` flags (``-DBM_D``, ``-DFP8PA_*``) must
    take its own distinct extension name -- ``legacy/kernel_bench.py`` shows the
    pattern, appending a build tag to the name.  Sharing this name across
    differing flags would serve one binary while claiming another, which is a
    golden-integrity failure, not a build inconvenience.
    """
    return kernel_mod.load()


# =========================================================================
# digests
# =========================================================================


def digest_tensor(t: torch.Tensor) -> str:
    """sha256 over the tensor's EXACT bytes.

    ``.cpu().contiguous()`` then raw bytes.  bfloat16 has no numpy dtype, so
    it is reinterpreted as uint8 first (a contiguous byte-preserving view --
    NOT a conversion).  Hashing bytes rather than comparing values is
    deliberate: -0.0 vs 0.0 and distinct NaN payloads are differences, and
    under the iron rule they are differences we want to see.
    """
    t = t.detach().cpu().contiguous()
    if t.numel() == 0:
        # e.g. the Q padding slice when T is already tile-aligned; hashing the
        # empty byte string is the right answer and sidesteps the dtype-view
        # edge case on zero-size tensors
        return hashlib.sha256(b"").hexdigest()
    if t.dtype != torch.uint8:
        t = t.view(torch.uint8)
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()


# Artifacts hashed for every case, in three classes:
#
#   quant  the quantized inputs over the REAL rows -- what the pipeline made
#          of the actual data.  Moves only for a numerics change.
#   pad    the Q compute-padding rows [T, Mpad).  Mpad is a CTA tile-shape
#          granularity (quant.py MPAD, kernel BM), so these move for a
#          legitimate SCHEDULING change (e.g. BM 64 -> 128 widened MPAD to
#          128) with no numerics change at all.  Kept separate precisely so
#          that case is diagnosable in one line instead of looking like a
#          quantization regression.  (K/V padding is BLK/BN-granular, i.e.
#          the quantization tile, so it is not split out the same way: a BN
#          change IS a numerics change.)
#   kernel the output bytes.  This is the class the IRON RULE is about.
#
# A digest is only taken for a buffer the pipeline FULLY WRITES in the case's
# configuration -- the workspace is torch.empty()-backed, so hashing a
# partially written buffer would hash uninitialized memory and be worthless
# as a golden.  (vt8/vscale/vlog2r/vinvr/vmean are stale when need_vt8 is
# False; vb16 is stale when need_vb16 is False.)
QUANT_ARTIFACTS = (
    "q8",
    "qscale",
    "k8",
    "kscale",
    "vt8",
    "vscale",
    "vlog2r",
    "vinvr",
    "vmean",
    "vb16",
)
PAD_ARTIFACTS = ("q8_pad", "qscale_pad")
KERNEL_ARTIFACTS = ("out",)


def artifact_base(name):
    """Strip the multi-chunk decoration: ``c0.q8`` -> ``q8``,
    ``c1.q8_pad`` -> ``q8_pad``, ``out.all`` -> ``out``, ``q8`` -> ``q8``."""
    if name == "out.all":
        return "out"
    if name.startswith("c") and "." in name:
        return name.split(".", 1)[1]
    return name


def classify_artifacts(names):
    """-> (quant_side, padding, kernel_side) split of artifact names."""
    quant, pad, kernel = [], [], []
    for n in names:
        b = artifact_base(n)
        if b in KERNEL_ARTIFACTS:
            kernel.append(n)
        elif b in PAD_ARTIFACTS:
            pad.append(n)
        else:
            quant.append(n)
    return quant, pad, kernel


# =========================================================================
# the mode / shape matrix
# =========================================================================

# (name, T, N, prefix, H, KVH, chunks, pool, seed)
#   chunks: None for a single shot, else a per-chunk token schedule run with a
#           growing prefix through ONE workspace, exactly as forward_extend does
#   pool:   "contiguous" (idx = arange) or "scattered" (page_size-1 pool with
#           the request's rows at random slots -- the gather index path)
SHAPES = (
    ("anchor", 128, 1024, 896, 8, 2, None, "contiguous", 8101),
    ("ragged", 100, 999, 899, 8, 2, None, "contiguous", 8102),
    ("subtile", 64, 63, 0, 8, 2, None, "contiguous", 8103),
    # T=192 -> Mpad 256 -> >1 CTA along M for BOTH tile widths (2 at BM=128,
    # 4 at BM=64), and T not a multiple of either width
    ("gqa24", 192, 640, 448, 24, 4, None, "contiguous", 8104),
    # the widest tested generalization ratio (grp=8); same CTA-spanning T as
    # gqa24. Added when the GQA kv=4 family was promoted; its goldens were
    # generated in that dedicated numerics update.
    ("gqa32", 192, 640, 448, 32, 4, None, "contiguous", 8107),
    ("multichunk", 568, 568, 0, 8, 2, (256, 192, 120), "contiguous", 8105),
    ("scatter", 64, 512, 448, 8, 2, None, "scattered", 8106),
)

# shapes that get the full {qk_i8 x rotate x center_k} cross product; the rest
# get the production mode and the all-off A/B mode.  Keeps the suite inside
# the runtime budget while still covering every quant switch on the two
# structurally richest shapes.
FULL_MATRIX_SHAPES = ("anchor", "ragged")
CORE_QUANT_MODES = ((True, True, True), (False, False, False))
PV_MODES = ("fp8", "bf16", "mixed")


def _quant_modes(shape_name):
    if shape_name in FULL_MATRIX_SHAPES:
        return [
            (i8, rot, cen) for i8 in (True, False) for rot in (True, False) for cen in (True, False)
        ]
    return list(CORE_QUANT_MODES)


def case_matrix():
    """-> list of case dicts.  Pure python; no torch, no device."""
    cases = []
    for name, T, N, prefix, H, KVH, chunks, pool, seed in SHAPES:
        for qk_i8, rotate, center_k in _quant_modes(name):
            for pv in PV_MODES:
                cid = (
                    f"{name}|qk{'i8' if qk_i8 else 'f8'}|rot{int(rotate)}|cen{int(center_k)}|pv{pv}"
                )
                cases.append(
                    {
                        "id": cid,
                        "shape": name,
                        "T": T,
                        "N": N,
                        "prefix": prefix,
                        "H": H,
                        "KVH": KVH,
                        "chunks": chunks,
                        "pool": pool,
                        "seed": seed,
                        "qk_i8": qk_i8,
                        "rotate": rotate,
                        "center_k": center_k,
                        "pv": pv,
                    }
                )
    return cases


CASES = case_matrix()


# =========================================================================
# deterministic test vectors (CPU-generated, then moved to device)
# =========================================================================


def make_qkv(T, N, H, KVH, seed, device=DEV):
    """Uniform [-1,1] bf16, drawn on the CPU from a seeded generator.

    CPU generation is what makes the vectors portable: the CUDA Philox stream
    depends on device and launch configuration, the CPU MT19937 stream does
    not.  Draw order (q, k, v) matches the existing suites' ``rand_qkv``.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = (torch.rand((T, H, HEAD_DIM), generator=g) * 2 - 1).to(device, torch.bfloat16)
    k = (torch.rand((N, KVH, HEAD_DIM), generator=g) * 2 - 1).to(device, torch.bfloat16)
    v = (torch.rand((N, KVH, HEAD_DIM), generator=g) * 2 - 1).to(device, torch.bfloat16)
    return q, k, v


def scatter_pool(k, v, seed):
    """Place the request's K/V rows at scattered slots of an oversized pool,
    mimicking a page_size-1 paged pool other requests also occupy."""
    N, KVH, D = k.shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    pool_n = int(N * 1.7) + 97
    slots = torch.randperm(pool_n, generator=g)[:N].to(k.device)
    kp = (torch.rand((pool_n, KVH, D), generator=g) * 2 - 1).to(k.device, k.dtype)
    vp = (torch.rand((pool_n, KVH, D), generator=g) * 2 - 1).to(k.device, k.dtype)
    kp[slots] = k
    vp[slots] = v
    return kp, vp, slots


def pv_mask(H, pv, device=DEV):
    mask = torch.ones(H, dtype=torch.uint8, device=device)
    if pv == "bf16":
        mask.zero_()
    elif pv == "mixed":
        for h in MIXED_BF16_HEADS:
            mask[h] = 0
    elif pv != "fp8":
        raise ValueError(pv)
    return mask


# =========================================================================
# the pipeline, with the twice-launched determinism gate
# =========================================================================


def run_chunk(ext, ws, q, k_pool, v_pool, idx, prefix, mask, *, qk_i8, rotate, center_k):
    """One forward_extend-equivalent step: gather+quantize, quantize Q, launch.

    Same call pattern as ``run_kernel`` in test_kernel_vs_sdpa.py, with the
    kernel launched TWICE into two distinct output buffers from identical
    inputs.  Returns (artifact digests, out[H,T,HD] bf16 copy, determinism
    ok flag, shape info).

    The double launch is not paranoia: goldens are worthless if the kernel is
    not run-to-run deterministic, so that property is checked FIRST, every
    case, every run.  A failure here is a critical finding in its own right
    (a race, an uninitialized smem read, an atomics-based reduction), and it
    invalidates the whole harness until it is fixed.
    """
    T, H, D = q.shape
    any_pv8 = bool(mask.max().item())
    all_pv8 = bool(mask.min().item())

    kv = q_mod.gather_quantize_kv(
        ws,
        k_pool,
        v_pool,
        idx.long(),
        need_vt8=any_pv8,
        need_vb16=not all_pv8,
        center_k=center_k,
        qk_i8=qk_i8,
        rotate=rotate,
    )
    q8, qscale, mpad = q_mod.quantize_q(ws, q, 1.0 / math.sqrt(D), qk_i8=qk_i8, rotate=rotate)

    o1 = ws.get("o", (H, mpad, HEAD_DIM), torch.bfloat16)
    o2 = torch.empty_like(o1)

    def launch(o):
        ext.fp8_prefill_attn(
            q8,
            kv["k8"],
            kv["vt8"],
            kv["vb16"],
            o,
            qscale,
            kv["kscale"],
            kv["vscale"],
            kv["vlog2r"],
            kv["vinvr"],
            kv["vmean"],
            mask,
            kv["n"],
            prefix,
            any_pv8,
            all_pv8,
            qk_i8,
        )

    launch(o1)
    launch(o2)
    torch.cuda.synchronize()
    # byte comparison over the FULL output buffer (padded rows included): a
    # scheduling bug that stomps padding is still a bug worth seeing.
    det_ok = torch.equal(o1.view(torch.uint8), o2.view(torch.uint8))

    # Q is split at the real/padding boundary: Mpad is a CTA tile granularity,
    # so the padding rows are allowed to move for a scheduling change while
    # the real rows are not.  See the artifact-class comment above.
    arts = {
        "q8": digest_tensor(q8[:, :T]),
        "q8_pad": digest_tensor(q8[:, T:]),
        "qscale": digest_tensor(qscale[:, :T]),
        "qscale_pad": digest_tensor(qscale[:, T:]),
        "k8": digest_tensor(kv["k8"]),
        "kscale": digest_tensor(kv["kscale"]),
    }
    if any_pv8:
        arts["vt8"] = digest_tensor(kv["vt8"])
        arts["vscale"] = digest_tensor(kv["vscale"])
        arts["vlog2r"] = digest_tensor(kv["vlog2r"])
        arts["vinvr"] = digest_tensor(kv["vinvr"])
        arts["vmean"] = digest_tensor(kv["vmean"])
    if not all_pv8:
        arts["vb16"] = digest_tensor(kv["vb16"])
    arts["out"] = digest_tensor(o1[:, :T])

    info = {
        "T": T,
        "mpad": int(mpad),
        "n": int(kv["n"]),
        "npad": int(kv["ntmax"]) * q_mod.BLK,
        "prefix": int(prefix),
        "tile": "wide(all-fp8-PV)" if all_pv8 else "narrow(has bf16-PV)",
    }
    return arts, o1[:, :T].clone(), det_ok, info


def run_case(ext, case):
    """-> (artifacts, det_failures, shape infos).  One fresh workspace per
    case, so a case is reproducible on its own (``-k <case-id>`` gives the
    same digests as a full run); the multi-chunk case deliberately reuses one
    workspace ACROSS its chunks, as the backend does."""
    T, N = case["T"], case["N"]
    H, KVH = case["H"], case["KVH"]
    q, k, v = make_qkv(T, N, H, KVH, case["seed"])
    if case["pool"] == "scattered":
        k_pool, v_pool, idx_all = scatter_pool(k, v, case["seed"] + 1)
    else:
        k_pool, v_pool = k, v
        idx_all = torch.arange(N, device=q.device)

    mask = pv_mask(H, case["pv"])
    cfg = dict(qk_i8=case["qk_i8"], rotate=case["rotate"], center_k=case["center_k"])
    ws = q_mod.FP8PrefillWorkspace(q.device)

    arts, det_fail, infos = {}, [], []
    if case["chunks"] is None:
        a, _, ok, info = run_chunk(ext, ws, q, k_pool, v_pool, idx_all, case["prefix"], mask, **cfg)
        arts.update(a)
        infos.append(info)
        if not ok:
            det_fail.append("single")
    else:
        outs, p = [], 0
        for i, t in enumerate(case["chunks"]):
            a, out, ok, info = run_chunk(
                ext, ws, q[p : p + t], k_pool, v_pool, idx_all[: p + t], p, mask, **cfg
            )
            arts.update({f"c{i}.{k_}": v_ for k_, v_ in a.items()})
            outs.append(out)
            infos.append(info)
            if not ok:
                det_fail.append(f"chunk{i}")
            p += t
        # the composed result: what the backend actually returns for the
        # whole sequence, and the thing a prefix-bookkeeping regression moves
        arts["out.all"] = digest_tensor(torch.cat(outs, dim=1))
    return arts, det_fail, infos


# =========================================================================
# golden storage
# =========================================================================


def load_golden(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def diff_case(golden_arts, got_arts):
    """-> (moved, missing, extra) artifact-name lists.  Pure python."""
    moved = sorted(n for n in got_arts if n in golden_arts and golden_arts[n] != got_arts[n])
    missing = sorted(n for n in golden_arts if n not in got_arts)
    extra = sorted(n for n in got_arts if n not in golden_arts)
    return moved, missing, extra


def format_mismatch(case_id, golden_arts, got_arts):
    """The failure message.  Says WHICH artifact moved and, from that, whether
    the change is quant-side or kernel-side.  Pure python (unit-testable)."""
    moved, missing, extra = diff_case(golden_arts, got_arts)
    quant_moved, pad_moved, kernel_moved = classify_artifacts(moved)
    lines = [f"GOLDEN MISMATCH  case={case_id}", ""]
    if quant_moved:
        lines.append(f"  quant-side artifacts that moved : {', '.join(quant_moved)}")
    if pad_moved:
        lines.append(f"  Q-PADDING artifacts that moved  : {', '.join(pad_moved)}")
    if kernel_moved:
        lines.append(f"  kernel-side artifacts that moved: {', '.join(kernel_moved)}")
    if missing:
        lines.append(f"  in golden but not produced now  : {', '.join(missing)}")
    if extra:
        lines.append(f"  produced now but not in golden  : {', '.join(extra)}")
    lines.append("")
    if kernel_moved and not quant_moved:
        lines += [
            "  VERDICT: KERNEL-SIDE. Every quantized input over the real rows is",
            "           byte-identical and the kernel still produced different",
            "           output bytes.",
            "           If this change was supposed to be scheduling-only, IT IS NOT.",
            "           Look at csrc/fp8_prefill_attn.cu (accumulation order, the",
            "           epilogue, mma operand order) and at extra_cuda_cflags.",
        ]
        if pad_moved:
            lines += [
                "           NOTE: the Q padding moved too, so Mpad (the CTA tile",
                "           granularity) changed. A wider CTA sweeps extra kv tiles",
                "           for its early rows; those are fully causal-masked and",
                "           must contribute p == 0 / alpha == 1 EXACTLY. If the",
                "           output moved, that invariant broke -- suspect a stale",
                "           or NaN V staging row reached a masked-P mma (0 * NaN).",
            ]
    elif pad_moved and not quant_moved and not kernel_moved:
        lines += [
            "  VERDICT: PADDING-ONLY. Real-row quantization and the kernel output",
            "           are both byte-identical; only the Q compute-padding rows",
            "           [T, Mpad) changed. That is a CTA tile-shape change",
            "           (quant.py MPAD / kernel BM), i.e. SCHEDULING, not numerics.",
            "           It is safe -- but the goldens still need regenerating, in",
            "           their own commit, with 'padding granularity changed' and",
            "           the old/new Mpad in the message.",
        ]
    elif quant_moved and not kernel_moved:
        lines += [
            "  VERDICT: QUANT-SIDE ONLY. The quantized inputs changed but the",
            "           output digest did not -- that is either a scale array the",
            "           kernel does not read in this mode, or (more likely) a",
            "           digest set mismatch. Look at quant.py.",
        ]
    elif quant_moved and kernel_moved:
        lines += [
            "  VERDICT: QUANT-SIDE. The quantized inputs moved, so the output was",
            "           always going to move too -- the kernel is not implicated.",
            "           Look at quant.py (scales, centering, rotation, sigma",
            "           permutation, the r_t fold), and at ambient torch matmul",
            "           precision (TF32 on the Hadamard GEMM).",
        ]
    else:
        lines += [
            "  VERDICT: artifact SET changed, no digest moved. The case's",
            "           configuration changed, not its numerics.",
        ]
    lines += ["", "  detail (golden -> got, first 16 hex):"]
    for n in moved:
        lines.append(f"    {n:<14s} {golden_arts[n][:16]} -> {got_arts[n][:16]}")
    lines += [
        "",
        "  Do NOT 'fix' this by rerunning with --write-golden unless the numerics",
        "  change was intentional and reviewed; see this file's header.",
    ]
    return "\n".join(lines)


class _GoldenStore:
    def __init__(self, path, write):
        self.path = path
        self.write = write
        self.doc = load_golden(path)
        self.produced = {}
        self.failed = []
        self.skipped = []
        self.overwritten = []
        self.nondet = []
        self._announced_absent = False
        self._announced_partial = False

    @property
    def have_goldens(self):
        return self.doc is not None and bool(self.doc.get("cases"))

    def record(self, case, arts, infos=None):
        self.produced[case["id"]] = {
            "meta": {
                k: case[k]
                for k in (
                    "shape",
                    "T",
                    "N",
                    "prefix",
                    "H",
                    "KVH",
                    "chunks",
                    "pool",
                    "seed",
                    "qk_i8",
                    "rotate",
                    "center_k",
                    "pv",
                )
            },
            # informational only -- never compared; there so a diff can be
            # read ("Mpad went 64 -> 128") without rerunning anything
            "derived": infos or [],
            "artifacts": arts,
        }

    def check(self, case, arts, infos=None):
        cid = case["id"]
        self.record(case, arts, infos)
        if self.write:
            old = (self.doc or {}).get("cases", {}).get(cid, {}).get("artifacts")
            if old is not None:
                moved, _, _ = diff_case(old, arts)
                if moved:
                    self.overwritten.append((cid, moved))
            return
        if not self.have_goldens:
            self.skipped.append(cid)
            if not self._announced_absent:
                self._announced_absent = True
                msg = (
                    f"\n{'!' * 74}\n"
                    f"NO GOLDENS: {self.path} does not exist (or is empty).\n"
                    f"THE BIT-EXACTNESS RULE IS NOT BEING ENFORCED RIGHT NOW.\n"
                    f"Every case in this module will skip.\n"
                    f"Create the goldens on the SM120 target with:\n"
                    f"    python3 -m pytest test_golden_bitexact.py -q --write-golden\n"
                    f"then COMMIT golden_bitexact.json on its own.\n"
                    f"{'!' * 74}"
                )
                warnings.warn(UserWarning(msg))
                print(msg)
            pytest.skip(
                "golden_bitexact.json absent -- run with --write-golden "
                "to create it (SEE THE BANNER ABOVE)"
            )
        gold = self.doc["cases"].get(cid)
        if gold is None:
            self.skipped.append(cid)
            if not self._announced_partial:
                self._announced_partial = True
                msg = (
                    f"\n{'!' * 74}\n"
                    f"GOLDEN FILE IS BEHIND THE MATRIX: no entry for {cid}\n"
                    f"(and possibly others). Those cases are UNPROTECTED.\n"
                    f"Regenerate with --write-golden, in its own commit.\n"
                    f"{'!' * 74}"
                )
                warnings.warn(UserWarning(msg))
                print(msg)
            pytest.skip(f"no golden entry for {cid}")
        moved, missing, extra = diff_case(gold["artifacts"], arts)
        if moved or missing or extra:
            self.failed.append((cid, moved, missing, extra))
            pytest.fail(format_mismatch(cid, gold["artifacts"], arts), pytrace=False)

    def merged_cases(self):
        """New digests over old ones.

        MERGE, not replace: a partial write (``-k``, ``-x``, a mid-run
        failure) must never silently DELETE goldens for cases it did not
        run -- that would quietly unprotect them.  Orphans (entries whose
        case no longer exists in the matrix) are pruned only on a complete
        run, when we know the matrix we are pruning against is the one that
        just executed.
        """
        existing = dict((self.doc or {}).get("cases", {}))
        merged = {**existing, **self.produced}
        valid = {c["id"] for c in CASES}
        complete = len(self.produced) == len(CASES)
        orphans = sorted(set(merged) - valid)
        if complete:
            merged = {k: v for k, v in merged.items() if k in valid}
        return dict(sorted(merged.items())), complete, orphans

    def flush(self):
        if not self.write:
            return
        cases, complete, orphans = self.merged_cases()
        doc = {
            "schema": GOLDEN_SCHEMA,
            "_policy": (
                "REGENERATE ONLY FOR INTENTIONAL, REVIEWED NUMERICS CHANGES, IN "
                "ITS OWN COMMIT, NEVER BUNDLED WITH A SCHEDULING/PERF CHANGE. "
                "See the header of test_golden_bitexact.py."
            ),
            "written_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "env": _env_fingerprint(),
            "expected_cases": len(CASES),
            "complete": complete,
            "cases": cases,
        }
        with open(self.path, "w") as f:
            json.dump(doc, f, indent=1, sort_keys=False)
            f.write("\n")
        print(
            f"\n[golden] wrote {len(self.produced)}/{len(CASES)} cases "
            f"({len(cases)} total in file) -> {self.path}"
        )
        if not complete:
            print(
                "[golden] !! INCOMPLETE RUN: only the cases that executed "
                "were rewritten; the rest were kept as they were. Re-run the "
                "FULL module before committing."
            )
            if orphans:
                print(
                    f"[golden] !! {len(orphans)} entry(ies) in the file are "
                    f"not in the current matrix: {', '.join(orphans[:5])}"
                    f"{' ...' if len(orphans) > 5 else ''}"
                )
        for cid, moved in self.overwritten:
            print(f"[golden] OVERWROTE {cid}: {', '.join(moved)}")
        if self.overwritten:
            print(
                f"[golden] {len(self.overwritten)} case(s) changed digests. "
                f"State which artifacts moved, and why, in the commit message."
            )


@pytest.fixture(scope="module")
def golden(request):
    write = (
        request.config.getoption("--write-golden", default=False)
        or os.environ.get("GOLDEN_WRITE") == "1"
    )
    store = _GoldenStore(GOLDEN_PATH, write)
    if write:
        print(f"\n[golden] WRITE MODE -- regenerating {GOLDEN_PATH}")
    yield store
    store.flush()
    if store.nondet:
        print(
            f"\n[golden] !! NONDETERMINISTIC KERNEL in {len(store.nondet)} "
            f"case(s): {', '.join(store.nondet[:8])}"
        )
    if store.failed:
        print(
            f"\n[golden] {len(store.failed)} case(s) moved. Union of moved "
            f"artifacts: {sorted({a for _, m, _, _ in store.failed for a in m})}"
        )
    if store.skipped:
        print(f"\n[golden] {len(store.skipped)} case(s) skipped for want of a golden entry.")


# =========================================================================
# the test
# =========================================================================


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_bitexact_golden(ext, golden, case):
    """Digest the full pipeline for one (mode, shape) and compare to golden.

    Order of business, per case:
      1. run the pipeline, launching the kernel TWICE from identical inputs;
      2. require the two launches to be byte-identical -- if not, the kernel
         is nondeterministic and no golden means anything (critical finding);
      3. compare every artifact digest to the golden, or record it in write
         mode, or skip loudly if no goldens exist yet.
    """
    arts, det_fail, infos = run_case(ext, case)
    if det_fail:
        golden.nondet.append(case["id"])
        pytest.fail(
            "\n".join(
                [
                    f"CRITICAL: NONDETERMINISTIC KERNEL  case={case['id']}",
                    f"  two launches from BYTE-IDENTICAL inputs produced different",
                    f"  output bytes (at: {', '.join(det_fail)}).",
                    "",
                    "  This is a finding, not a flake. Goldens cannot exist for a",
                    "  nondeterministic kernel and the whole bit-exactness rule is",
                    "  void until it is fixed. Look for: a race on shared memory",
                    "  (a missing __syncthreads() after a cp.async wait-group",
                    "  restructure), an uninitialized smem read, or any reduction",
                    "  that became atomics-based.",
                ]
            ),
            pytrace=False,
        )
    golden.check(case, arts, infos)
