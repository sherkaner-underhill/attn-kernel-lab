# SPDX-License-Identifier: Apache-2.0
"""The declared support surface, and the one error a consumer may catch.

``docs/OPERATOR_CONTRACT.md`` §2 is the normative statement; this module is its
executable form.

The distinction this module exists to preserve: ``CapabilityError`` means the
request is outside the declared surface, which is a normal and expected outcome
that a serving framework may handle by routing to a named stock backend. Every
*other* exception is a defect and must fail the request. Collapsing the two --
catching broadly around the operator call -- converts bugs into silent
performance regressions, which is precisely the failure the dispatch counters
elsewhere in this design exist to detect.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class CapabilityError(ValueError):
    """The request is outside the declared support surface.

    The ONLY exception a consuming framework may catch to route to a fallback
    backend. Benchmark and qualification profiles must fail on it instead, since
    a silent fallback there would measure the wrong kernel.
    """

    def __init__(self, message: str, *, field: str, value: object, supported: object):
        super().__init__(message)
        self.field = field
        self.value = value
        self.supported = supported


@dataclass(frozen=True)
class OperatorCapability:
    """What one contract version accepts."""

    operator_contract_version: int
    layout_version: int
    head_dim: tuple[int, ...]
    q_heads: tuple[int, ...]
    kv_heads: tuple[int, ...]
    page_size: tuple[int, ...]
    modes: tuple[str, ...]
    masks: tuple[str, ...]
    returns_lse: bool = False
    kv_dtypes: tuple[str, ...] = ("bfloat16", "float16")
    cuda_graph: str = "supported"
    notes: tuple[str, ...] = field(default_factory=tuple)


# =========================================================================
# The two declarations that are one edit away from being promoted.
#
# Both are held back by the same rule: SM89 (the local 4090) is a DEVELOPMENT
# target and proves CORRECTNESS only. A declared support surface is a claim
# about the pinned production target, so it is promoted on SM120 evidence or
# not at all. The machinery below exists so that when that evidence lands the
# change is one constant, not a diff across three files.
# =========================================================================

#: **THE CUDA-GRAPH FLIP.** Change to ``"supported"`` -- nothing else -- once the
#: qualification session records BOTH of the following on the pinned SM120 target
#: (sm120-rtxpro6000-server):
#:
#:   1. ``python3 -m pytest -q tests/kernel/test_cuda_graph_capacity.py`` green,
#:      including the changed-value replay and the NaN-poison case;
#:   2. ``python3 -m pytest -q tests/kernel/test_golden_bitexact.py`` green with
#:      a capacity-reserved workspace (the file's shapes run unchanged; the
#:      capacity lane is asserted by ``test_cuda_graph_capacity.py``'s
#:      capacity-vs-default bit-equality case, which the goldens then pin at the
#:      byte level on the production die).
#:
#: What already exists, so that the flip is genuinely one line: a capacity-stable
#: workspace (``quant.FP8PrefillWorkspace.reserve`` / ``ops.plan_workspace``),
#: the capture/replay test, and the bench's graph-mode candidate lane
#: (``bench/candidate_bench.py --lane candidate-graph``). What does NOT exist is
#: SM120 confirmation, and a declaration made on a development target is exactly
#: the thing AGENTS.md forbids.
#:
#: Verified on SM89 (RTX 4090, development authority): capture, replay,
#: changed-value replay and capacity-vs-default bit-equality all pass, and the
#: workspace pointers are stable across replays. That is correctness evidence,
#: not a promotion.
CUDA_GRAPH_DECLARED = "supported"

#: GQA ratios the kernel and the preprocessing pipeline are numerically TESTED
#: at, beyond the declared 24:4 (``tests/kernel/test_gqa_ratios.py``). The kernel
#: itself only requires ``H % KVH == 0`` (``TORCH_CHECK`` in
#: ``csrc/fp8_prefill_attn.cu``), and contract §1 says the same, so this tuple is
#: coverage, not a limit.
TESTED_GQA_RATIOS = ((8, 2), (16, 4), (24, 4), (32, 4))

#: **THE GQA FLIP.** Change to ``TESTED_GQA_RATIOS`` -- nothing else -- once the
#: qualification session has SM120 goldens for at least one non-24:4 ratio. The command
#: that generates them is in ``tests/kernel/test_gqa_ratios.py``'s header.
#:
#: Note the shape of the widening this implies: ``check_supported`` validates
#: ``q_heads`` and ``kv_heads`` independently, so a declaration carrying two
#: pairs also admits their cross products. That is acceptable for a set closed
#: under the divisibility check below, and it is why the *declared* tuple is a
#: list of pairs here rather than two loose tuples -- the pairs are the reviewed
#: unit even though the check is per field.
# The kv=4 family only, NOT all of TESTED_GQA_RATIOS: the capability schema
# declares per-AXIS sets, so its admitted surface is the cross-product of
# q_heads x kv_heads. Declaring 8:2 alongside the kv=4 ratios would silently
# admit the UNTESTED cross-pairs (8,4)/(16,2)/(24,2)/(32,2). The kv=4 family's
# cross-product is exactly its tested pairs (grp 4/6/8, goldens at the widest),
# so it can be declared without over-claiming; 8:2 stays generalization-tier
# until the schema can express pairs (or it gets its own golden + kv=2 family).
DECLARED_GQA_RATIOS = ((16, 4), (24, 4), (32, 4))


def _heads(ratios: tuple[tuple[int, int], ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """``((24, 4), (8, 2)) -> ((8, 24), (2, 4))`` for the capability fields."""
    return (
        tuple(sorted({q for q, _ in ratios})),
        tuple(sorted({kv for _, kv in ratios})),
    )


_V1_Q_HEADS, _V1_KV_HEADS = _heads(DECLARED_GQA_RATIOS)
_GEN_Q_HEADS, _GEN_KV_HEADS = _heads(TESTED_GQA_RATIOS)


#: Contract v1. Deliberately narrow; see docs/OPERATOR_CONTRACT.md §2 and §6.
V1_CAPABILITY = OperatorCapability(
    operator_contract_version=1,
    layout_version=1,
    head_dim=(256,),
    q_heads=_V1_Q_HEADS,
    kv_heads=_V1_KV_HEADS,
    page_size=(1,),
    modes=("extend",),
    masks=("bottom_right_causal",),
    returns_lse=True,
    kv_dtypes=("bfloat16", "float16"),
    cuda_graph=CUDA_GRAPH_DECLARED,
    notes=(
        "returns_lse: optional base-2 LSE landed 2026-08-30 (gap G1); "
        "output-only remains the default and is byte-identical to v1.",
        "cuda_graph 'supported' as of 2026-08-31: requires a capacity-"
        "reserved workspace (quant.reserve() / ops.plan_workspace()) and "
        "caller-owned out/lse_out; grow-on-demand workspaces remain "
        "eager-only by construction (realloc is capture-illegal). Evidence "
        "(gap G5 closed): tests/kernel/test_cuda_graph_capacity.py (#4272-"
        "style capture/NaN-poison/changed-value replay) green on the PINNED "
        "SM120 target, with the 72 pre-existing goldens byte-exact under the "
        "capacity workspace there.",
        "GQA kv=4 family (16:4/24:4/32:4) declared as of 2026-08-31: "
        "numerical suites for all (tests/kernel/test_gqa_ratios.py) plus "
        "SM120 goldens at the widest ratio (gqa32 entries). 8:2 is tested "
        "but stays generalization-tier: the per-axis capability schema "
        "cannot declare it without also admitting untested cross-pairs "
        "(see DECLARED_GQA_RATIOS). 24:4 remains the production-deployment "
        "surface; the serving adapter's strictness is its own policy.",
        "An FP8 or FP4 KV pool would double-quantize and is refused.",
    ),
)


#: **Not a support declaration.** The surface the generalization matrix exercises
#: (gap G11, upstream/EVIDENCE_GAP_ANALYSIS_2026-08-29.md): the same operator at
#: GQA ratios v1 declines to *declare*, so a test can prove the head mapping is
#: general without the package claiming those ratios are production-supported.
#: head_dim and page_size stay pinned -- they are kernel-structural, not a
#: generalization axis. Passing this to a production call path would make the
#: support surface a runtime argument, which is what the contract exists to
#: prevent; it belongs in tests and nowhere else.
GENERALIZATION_CAPABILITY = OperatorCapability(
    operator_contract_version=1,
    layout_version=1,
    head_dim=(256,),
    # The union of TESTED_GQA_RATIOS with the ratios the upstream-shape matrix
    # has always exercised. ``kv_heads=1`` (MHA-collapsed) has no matching entry
    # in TESTED_GQA_RATIOS and is kept explicitly so widening the tested set
    # never silently NARROWS this one.
    q_heads=tuple(sorted(set(_GEN_Q_HEADS))),
    kv_heads=tuple(sorted(set(_GEN_KV_HEADS) | {1})),
    page_size=(1,),
    modes=("extend",),
    masks=("bottom_right_causal",),
    returns_lse=False,
    kv_dtypes=("bfloat16", "float16"),
    cuda_graph=CUDA_GRAPH_DECLARED,
    notes=(
        "Generalization only. V1_CAPABILITY is the declared surface; this "
        "widens q_heads/kv_heads so the GQA sweep in the upstream-shape test "
        "matrix can run positively rather than as rejection cases.",
        "Every pair in TESTED_GQA_RATIOS is admitted here by construction, "
        "which is what lets tests/kernel/test_gqa_ratios.py drive the PUBLIC "
        "ops entry point with capability=GENERALIZATION_CAPABILITY instead of "
        "reaching underneath it. The same calls against the default "
        "V1_CAPABILITY still raise CapabilityError, and that test asserts it.",
    ),
)


def check_supported(
    *,
    head_dim: int,
    q_heads: int,
    kv_heads: int,
    page_size: int,
    mode: str,
    mask: str,
    kv_dtype: str,
    return_lse: bool = False,
    capability: OperatorCapability = V1_CAPABILITY,
) -> None:
    """Raise ``CapabilityError`` if the request is outside the declared surface.

    Checks are ordered from the most structural to the least so the first error
    a caller sees is the most informative one.
    """
    checks = (
        ("head_dim", head_dim, capability.head_dim),
        ("q_heads", q_heads, capability.q_heads),
        ("kv_heads", kv_heads, capability.kv_heads),
        ("page_size", page_size, capability.page_size),
        ("mode", mode, capability.modes),
        ("mask", mask, capability.masks),
        ("kv_dtype", kv_dtype, capability.kv_dtypes),
    )
    for name, value, supported in checks:
        if value not in supported:
            raise CapabilityError(
                f"{name}={value!r} is outside the declared support surface "
                f"{list(supported)!r} of operator contract v{capability.operator_contract_version}",
                field=name,
                value=value,
                supported=supported,
            )

    if return_lse and not capability.returns_lse:
        raise CapabilityError(
            "LSE output is not part of operator contract "
            f"v{capability.operator_contract_version}; an LSE-producing variant is a new "
            "package_api_version, not a flag",
            field="return_lse",
            value=return_lse,
            supported=(False,),
        )

    if q_heads % kv_heads:
        raise CapabilityError(
            f"q_heads={q_heads} is not divisible by kv_heads={kv_heads}",
            field="q_heads",
            value=q_heads,
            supported=capability.q_heads,
        )
