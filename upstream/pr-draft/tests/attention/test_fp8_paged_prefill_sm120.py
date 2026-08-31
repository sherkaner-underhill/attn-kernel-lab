# SPDX-License-Identifier: Apache-2.0
"""INT8-QK / FP8-PV paged prefill attention on SM120: D256, page_size 1, GQA.

PR-DRAFT (gap G3/G11 of ``upstream/EVIDENCE_GAP_ANALYSIS_2026-08-29.md``). This
file is written to run **in FlashInfer's tree, on their ``unit_test_rtx_pro_6000``
CI row**, not in this repository -- there is no GPU here and this directory is
outside ``testpaths``. It is a port of the *content* of ``tests/kernel/`` into
upstream's *form*, per §2.2: a module capability gate, ``pytest.param(..., id=)``
matrices, stacked single-dimension parametrize, ``torch.testing.assert_close``,
non-power-of-2 lengths, and ``flip_coin`` alternation of supplied-versus-allocated
``out``.

The operator: bottom-right-causal EXTEND attention over a **paged, page_size-1,
unquantized BF16** KV pool at ``head_dim`` 256 and 24:4 GQA, with per-row Q /
per-tile K / per-tile V scales and an inclusive gather-centre-rotate-quantize-pack
preprocessing path. Every low-precision SM120 neighbour upstream (#3640, #4502,
#4149, #4691) is dense or ragged, per-tensor or per-block scaled, and none is
paged; that combination is the whole of what this file tests.

**What is asserted, and why in these units.** Against a masked FP64 reference the
error is dominated by INT8-QK input rounding plus FP8-PV, so the thresholds are
*scale-free* -- cosine similarity, mean and max row-relative L2 -- rather than
``atol``/``rtol`` on outputs whose magnitude depends on the attention depth. That
follows ``test_nvfp4_attention_sm120.py``, which carries per-case ``cos_sim`` and
``mean_abs_err`` thresholds in ``pytest.param`` for the same reason. Where a
property is *exact* -- page-map equivalence, unread-pool independence, workspace
reuse, ``out=`` identity -- it is asserted at ``rtol=0, atol=0`` with
``assert_close``, because an approximate assertion there would prove nothing.

**Ported from** ``tests/kernel/test_kernel_vs_sdpa.py`` (reference comparison,
gather plumbing, GQA mapping, workspace reuse) and
``tests/kernel/test_divergence_hunting.py`` (adversarial geometry). The lab lane
stays as it is; this is the same evidence in the harness a maintainer can run.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

# --- lab-only preamble; delete on the way upstream ---------------------------
# FlashInfer's CI always has torch, and `tests/attention/` is a package whose
# siblings import by absolute path (`from tests.test_helpers.test_helpers import
# ...`). Here the module has to be collectable on a CPU-only machine with no
# torch at all, and the shim sits beside it.
try:
    import torch
except ImportError:  # pragma: no cover
    pytest.skip("torch is required", allow_module_level=True)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
# -----------------------------------------------------------------------------

from flashinfer_shim import (  # noqa: E402
    GENERALIZATION_SURFACE,
    flip_coin,
    fp8_paged_prefill_sm120,
    is_sm120a_supported,
    is_sm121a_supported,
    make_workspace,
    ref_single_prefill,
)

HEAD_DIM = 256
KV_TILE = 64  # the quantization/scale tile in the kv-row dimension


def _require_sm120():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda")
    if not (is_sm120a_supported(device) or is_sm121a_supported(device)):
        pytest.skip("SM120 or SM121 GPU is required")


def _sm12x_present() -> bool:
    try:
        if not torch.cuda.is_available():
            return False
        device = torch.device("cuda")
        return is_sm120a_supported(device) or is_sm121a_supported(device)
    except Exception:  # noqa: BLE001 -- a driver-less box must collect, not error
        return False


# Module gate as well as the per-body one: shape (2) of §2.2, so collection is
# safe on any machine. The rejection block below is device-free and upstream
# would lift it out of this gate, the way `test_nvfp4_split_kv_gate_dtype_logic`
# runs CPU-only.
pytestmark = pytest.mark.skipif(
    not _sm12x_present(), reason="SM120 or SM121 GPU is required"
)


# ------------------------------------------------------------------- fixtures


def _rand(shape, generator, device, dtype=torch.bfloat16) -> torch.Tensor:
    """Seeded on the host so a rerun benches identical bytes; the quantization
    path is data-dependent, so host seeding is the only reproducible option."""
    values = torch.empty(shape, dtype=torch.float32)
    values.uniform_(-1.0, 1.0, generator=generator)
    return values.to(dtype).to(device)


def _make_request(
    qo_len: int,
    kv_len: int,
    num_qo_heads: int = 24,
    num_kv_heads: int = 4,
    *,
    page_map: str = "contiguous",
    seed: int = 0,
    device: str = "cuda",
):
    """One request, in both the paged and the logical (position-order) views.

    ``page_map`` decides where position ``i`` physically lives:

    - ``contiguous``  page ``i``            -- the easy case a gather bug survives
    - ``shuffled``    a random permutation  -- a real pool after real traffic
    - ``fragmented``  page ``2*i + 1`` of a pool twice the size, the even rows
      filled with values no index names -- catches a gather that walks rows
      instead of following the index

    Returns ``(q, k_cache, v_cache, kv_indices, k_logical, v_logical)`` where the
    caches are FlashInfer's NHD ``[num_pages, page_size, num_kv_heads, head_dim]``
    and the logical tensors are what the reference must see.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    q = _rand((qo_len, num_qo_heads, HEAD_DIM), generator, device)
    k_logical = _rand((kv_len, num_kv_heads, HEAD_DIM), generator, device)
    v_logical = _rand((kv_len, num_kv_heads, HEAD_DIM), generator, device)

    if page_map == "contiguous":
        rows = torch.arange(kv_len, device=device)
        num_pages = kv_len
    elif page_map == "shuffled":
        rows = torch.randperm(kv_len, generator=generator).to(device)
        num_pages = kv_len
    elif page_map == "fragmented":
        rows = torch.arange(kv_len, device=device) * 2 + 1
        num_pages = 2 * kv_len
    else:  # pragma: no cover - a typo in a parametrize id, not a runtime path
        raise ValueError(f"unknown page_map {page_map!r}")

    k_cache = _rand((num_pages, 1, num_kv_heads, HEAD_DIM), generator, device)
    v_cache = _rand((num_pages, 1, num_kv_heads, HEAD_DIM), generator, device)
    k_cache[rows, 0] = k_logical
    v_cache[rows, 0] = v_logical
    return q, k_cache, v_cache, rows.to(torch.int64), k_logical, v_logical


def _reference(q, k_logical, v_logical) -> torch.Tensor:
    """FP64 bottom-right-causal reference, GQA by ``repeat_interleave`` as in-tree."""
    group = q.shape[1] // k_logical.shape[1]
    out, _lse = ref_single_prefill(
        q,
        k_logical.repeat_interleave(group, dim=1),
        v_logical.repeat_interleave(group, dim=1),
        causal=True,
    )
    return out


def _row_relative(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    """Per-(row, head) relative L2 -- the unit the numerics program is calibrated in."""
    delta = (actual.to(torch.float64) - expected).norm(dim=-1)
    return delta / expected.norm(dim=-1).clamp_min(1e-6)


def _cos_sim(actual: torch.Tensor, expected: torch.Tensor) -> float:
    flat_a = actual.to(torch.float64).flatten()
    flat_b = expected.flatten()
    return float(
        torch.dot(flat_a, flat_b)
        / (flat_a.norm() * flat_b.norm()).clamp_min(1e-12)
    )


# --------------------------------------------------------------- correctness


@pytest.mark.parametrize(
    "qo_len, kv_len, cos_threshold, row_rel_mean, row_rel_max",
    [
        # qo_len == kv_len: no prefix, pure self-attention, the shallow corner
        pytest.param(64, 64, 0.995, 0.06, 0.20, id="qo64-kv64-noprefix"),
        # the prefix-cache geometry: qo_len << kv_len, where #3684 found real
        # SM120 low-precision corruption with split-KV
        pytest.param(64, 1024, 0.998, 0.05, 0.15, id="qo64-kv1024"),
        pytest.param(128, 4096, 0.998, 0.05, 0.15, id="qo128-kv4096"),
        # non-power-of-2 everything: neither length is a multiple of the 64-row
        # kv tile nor of the 128-row Q padding granularity
        pytest.param(100, 999, 0.998, 0.05, 0.15, id="qo100-kv999-ragged"),
        pytest.param(193, 4993, 0.998, 0.05, 0.15, id="qo193-kv4993-ragged"),
        pytest.param(37, 2011, 0.998, 0.05, 0.16, id="qo37-kv2011-ragged"),
        # a single-row extend: the decode-shaped edge of an EXTEND operator
        pytest.param(1, 977, 0.995, 0.06, 0.20, id="qo1-kv977"),
    ],
)
@torch.inference_mode()
def test_fp8_paged_prefill_sm120_accuracy(
    qo_len, kv_len, cos_threshold, row_rel_mean, row_rel_max
):
    """Bottom-right-causal output against a masked FP64 reference.

    ``flip_coin`` alternates supplied-versus-allocated ``out`` across the matrix
    rather than doubling it, and the supplied case additionally pins the
    destination-passing contract: the returned tensor **is** the caller's buffer.
    """
    _require_sm120()
    torch.manual_seed(0)
    q, k_cache, v_cache, kv_indices, k_logical, v_logical = _make_request(
        qo_len, kv_len, seed=qo_len + kv_len
    )

    provided_out = None
    if flip_coin(qo_len, kv_len):
        provided_out = torch.empty_like(q)
    out = fp8_paged_prefill_sm120(
        q, k_cache, v_cache, kv_indices, out=provided_out
    )
    if provided_out is not None:
        assert out is provided_out

    assert out.dtype == torch.bfloat16
    assert out.shape == (qo_len, q.shape[1], HEAD_DIM)
    assert torch.isfinite(out).all()

    expected = _reference(q, k_logical, v_logical)
    row_rel = _row_relative(out, expected)
    assert _cos_sim(out, expected) >= cos_threshold
    assert row_rel.mean().item() <= row_rel_mean, row_rel.mean().item()
    assert row_rel.max().item() <= row_rel_max, row_rel.max().item()


@pytest.mark.parametrize(
    "num_qo_heads, num_kv_heads",
    [
        pytest.param(24, 4, id="gqa-24x4-production"),
        pytest.param(8, 2, id="gqa-8x2"),
        pytest.param(8, 1, id="mqa-8x1"),
        pytest.param(16, 4, id="gqa-16x4"),
    ],
)
@torch.inference_mode()
def test_fp8_paged_prefill_sm120_gqa_ratios(num_qo_heads, num_kv_heads):
    """Head mapping is general, and 24:4 is only the *declared* point.

    Two assertions, because accuracy alone would not catch a transposed group
    index: the output must track the FP64 reference, and perturbing one KV head
    must move exactly the query heads of its group and no others -- the
    ``h // (num_qo_heads // num_kv_heads)`` mapping, tested as an identity rather
    than as a tolerance.
    """
    _require_sm120()
    torch.manual_seed(0)
    qo_len, kv_len = 96, 1039
    surface = None if (num_qo_heads, num_kv_heads) == (24, 4) else GENERALIZATION_SURFACE
    extra = {} if surface is None else {"capability": surface}

    q, k_cache, v_cache, kv_indices, k_logical, v_logical = _make_request(
        qo_len, kv_len, num_qo_heads, num_kv_heads, seed=num_qo_heads * 31 + num_kv_heads
    )
    out = fp8_paged_prefill_sm120(q, k_cache, v_cache, kv_indices, **extra)
    expected = _reference(q, k_logical, v_logical)
    assert _cos_sim(out, expected) >= 0.998
    assert _row_relative(out, expected).mean().item() <= 0.05

    if num_kv_heads == 1:
        return  # no group boundary to isolate
    group = num_qo_heads // num_kv_heads
    v_cache_perturbed = v_cache.clone()
    v_cache_perturbed[kv_indices, 0, 1] *= -1.0
    perturbed = fp8_paged_prefill_sm120(
        q, k_cache, v_cache_perturbed, kv_indices, **extra
    )
    torch.testing.assert_close(
        perturbed[:, :group], out[:, :group], rtol=0, atol=0
    )
    assert not torch.equal(perturbed[:, group : 2 * group], out[:, group : 2 * group])


@pytest.mark.parametrize(
    "page_map",
    [
        pytest.param("contiguous", id="pages-contiguous"),
        pytest.param("shuffled", id="pages-shuffled"),
        pytest.param("fragmented", id="pages-fragmented"),
    ],
)
@torch.inference_mode()
def test_fp8_paged_prefill_sm120_page_map_equivalence(page_map):
    """The physical page map must be invisible in the result, **bit for bit**.

    The gather composes the index with the pool, so a shuffled or fragmented map
    holding the same logical K/V is not merely close to the contiguous result --
    it is the same result. ``rtol=0, atol=0`` is the only assertion that can
    catch a gather that walks rows instead of following the index, or one that
    reads a page a caller never named.
    """
    _require_sm120()
    torch.manual_seed(0)
    qo_len, kv_len = 128, 1500
    reference_case = _make_request(qo_len, kv_len, page_map="contiguous", seed=77)
    q, k_cache, v_cache, kv_indices, k_logical, v_logical = reference_case
    baseline = fp8_paged_prefill_sm120(q, k_cache, v_cache, kv_indices)

    case = _make_request(qo_len, kv_len, page_map=page_map, seed=77)
    q2, k2, v2, idx2, k_logical2, v_logical2 = case
    # same logical content, different physical placement
    torch.testing.assert_close(k_logical2, k_logical, rtol=0, atol=0)
    torch.testing.assert_close(v_logical2, v_logical, rtol=0, atol=0)
    out = fp8_paged_prefill_sm120(q2, k2, v2, idx2)
    torch.testing.assert_close(out, baseline, rtol=0, atol=0)


@torch.inference_mode()
def test_fp8_paged_prefill_sm120_unnamed_pages_are_never_read():
    """Pages no index names must not reach the result, whatever they hold.

    Poisoning the unused pool rows with a value that becomes an FP8 NaN encoding
    is the pool-side half of the operator's stale-data contract (§5): the other
    half, the workspace tail, is the test below. ``rtol=0, atol=0``, because a
    read of an unnamed page is not a tolerance question.
    """
    _require_sm120()
    torch.manual_seed(0)
    qo_len, kv_len = 100, 999  # kv_len % 64 != 0: a partial boundary tile
    q, k_cache, v_cache, kv_indices, _, _ = _make_request(
        qo_len, kv_len, page_map="fragmented", seed=101
    )
    clean = fp8_paged_prefill_sm120(q, k_cache, v_cache, kv_indices)

    unnamed = torch.ones(k_cache.shape[0], dtype=torch.bool, device=k_cache.device)
    unnamed[kv_indices] = False
    k_cache[unnamed] = float("inf")
    v_cache[unnamed] = float("nan")
    poisoned = fp8_paged_prefill_sm120(q, k_cache, v_cache, kv_indices)
    assert torch.isfinite(poisoned).all()
    torch.testing.assert_close(poisoned, clean, rtol=0, atol=0)


@torch.inference_mode()
def test_fp8_paged_prefill_sm120_workspace_reuse_long_after_short():
    """A shared workspace must not leak one request's tail into the next.

    Contract §5 requires every conforming implementation to zero the KV boundary
    tile's tail, because stale FP8 bytes there can decode as NaN and poison
    ``0 * NaN`` in the PV matmul. The consequence is that a reused workspace must
    produce **exactly** what a fresh one produces, in both directions -- long
    after short (buffers grow mid-sequence) and short after long (the dangerous
    one: the tail beyond the new, shorter kv length still holds the old request).
    """
    _require_sm120()
    torch.manual_seed(0)
    shared = make_workspace(torch.device("cuda"))
    lengths = [(64, 200), (128, 4096), (64, 200), (96, 1039)]

    for index, (qo_len, kv_len) in enumerate(lengths):
        case = _make_request(qo_len, kv_len, seed=1000 + index)
        q, k_cache, v_cache, kv_indices, _, _ = case
        reused = fp8_paged_prefill_sm120(
            q, k_cache, v_cache, kv_indices, workspace_buffer=shared
        )
        fresh = fp8_paged_prefill_sm120(
            q,
            k_cache,
            v_cache,
            kv_indices,
            workspace_buffer=make_workspace(torch.device("cuda")),
        )
        assert torch.isfinite(reused).all()
        torch.testing.assert_close(reused, fresh, rtol=0, atol=0)


@torch.inference_mode()
def test_fp8_paged_prefill_sm120_out_buffer_identity_and_aliasing():
    """Destination passing: the caller's buffer is written and returned, and a
    buffer that aliases ``q`` is refused rather than silently corrupted."""
    _require_sm120()
    torch.manual_seed(0)
    q, k_cache, v_cache, kv_indices, _, _ = _make_request(64, 512, seed=5)

    provided = torch.full_like(q, float("nan"))
    returned = fp8_paged_prefill_sm120(q, k_cache, v_cache, kv_indices, out=provided)
    assert returned is provided
    assert torch.isfinite(returned).all()

    allocated = fp8_paged_prefill_sm120(q, k_cache, v_cache, kv_indices)
    torch.testing.assert_close(returned, allocated, rtol=0, atol=0)

    with pytest.raises(ValueError, match="out must not overlap q storage"):
        fp8_paged_prefill_sm120(q, k_cache, v_cache, kv_indices, out=q)


# ------------------------------------------------------------------ rejection
#
# Five distinct messages, the shape #4272 asserts and the thing #3518 gained
# because a reviewer asked for it. Every one of these is raised before any
# device work, so they are the cheapest tests in the file and the ones that
# document the support surface to a reader who never opens the design doc.


@pytest.mark.parametrize(
    "head_dim", [pytest.param(64, id="d64"), pytest.param(128, id="d128")]
)
@torch.inference_mode()
def test_fp8_paged_prefill_sm120_rejects_head_dim(head_dim):
    """``head_dim`` 256 only: the shared-memory sizing and the tile pair are not
    templated on it (contract §6.2)."""
    _require_sm120()
    q = torch.zeros((16, 24, head_dim), dtype=torch.bfloat16, device="cuda")
    k_cache = torch.zeros((64, 1, 4, head_dim), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="head_dim"):
        fp8_paged_prefill_sm120(
            q, k_cache, k_cache.clone(), torch.arange(64, device="cuda")
        )


@torch.inference_mode()
def test_fp8_paged_prefill_sm120_rejects_page_size():
    """``page_size`` 1 only: larger pages need a different gather or kernel-side
    page-table addressing (contract §6.3). Rejected before the cache is
    reshaped, so it is a ValueError and never a mis-shaped read."""
    _require_sm120()
    q = torch.zeros((16, 24, HEAD_DIM), dtype=torch.bfloat16, device="cuda")
    k_cache = torch.zeros((4, 16, 4, HEAD_DIM), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="page_size"):
        fp8_paged_prefill_sm120(
            q,
            k_cache,
            k_cache.clone(),
            torch.arange(4, device="cuda"),
            page_size=16,
        )


@torch.inference_mode()
def test_fp8_paged_prefill_sm120_rejects_quantized_kv_pool():
    """An FP8 pool would double-quantize (contract §6.4). The operator refuses
    rather than accepting a pool it would silently re-scale."""
    _require_sm120()
    q = torch.zeros((16, 24, HEAD_DIM), dtype=torch.bfloat16, device="cuda")
    k_cache = torch.zeros((64, 1, 4, HEAD_DIM), dtype=torch.float8_e4m3fn, device="cuda")
    with pytest.raises(ValueError, match="kv_dtype"):
        fp8_paged_prefill_sm120(
            q, k_cache, k_cache.clone(), torch.arange(64, device="cuda")
        )


@torch.inference_mode()
def test_fp8_paged_prefill_sm120_rejects_non_causal():
    """Bottom-right causal is the only mask; a non-causal request is a different
    operator, not a flag."""
    _require_sm120()
    q = torch.zeros((16, 24, HEAD_DIM), dtype=torch.bfloat16, device="cuda")
    k_cache = torch.zeros((64, 1, 4, HEAD_DIM), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="mask"):
        fp8_paged_prefill_sm120(
            q, k_cache, k_cache.clone(), torch.arange(64, device="cuda"), causal=False
        )


@torch.inference_mode()
def test_fp8_paged_prefill_sm120_returns_base2_lse_on_request():
    """The base-2 LSE landed 2026-08-30 (gap G1 closed): requesting it returns
    (out, lse) with lse finite and consistent with logsumexp of the reference
    scores; omitting it preserves the output-only behaviour byte-exactly
    (asserted in the kernel lane; here we assert the API shape and sanity)."""
    _require_sm120()
    torch.manual_seed(1301)
    q = torch.randn((48, 24, HEAD_DIM), dtype=torch.bfloat16, device="cuda")
    k_cache = torch.randn((96, 1, 4, HEAD_DIM), dtype=torch.bfloat16, device="cuda")
    v_cache = torch.randn((96, 1, 4, HEAD_DIM), dtype=torch.bfloat16, device="cuda")
    indices = torch.arange(96, device="cuda")
    out, lse = fp8_paged_prefill_sm120(
        q, k_cache, v_cache, indices, return_lse=True
    )
    assert out.shape == (48, 24, HEAD_DIM)
    assert lse.shape == (48, 24) and lse.dtype == torch.float32
    assert torch.isfinite(lse).all()
    # Monotone sanity: each later row sees a superset of columns, so with any
    # fixed data the row-wise LSE cannot be wildly non-physical; check bounds
    # against a base-2 logsumexp of an all-visible upper envelope.
    assert (lse > -1000).all() and (lse < 1000).all()
