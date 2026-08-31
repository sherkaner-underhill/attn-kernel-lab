<!-- SPDX-License-Identifier: Apache-2.0 -->
# Operator contract v1 (draft)

- **Status:** Draft. Normative once Oracle A implements it and the current
  production kernel passes against it.
- **`operator_contract_version`:** 1
- **`layout_version`:** 1
- **Extracted from:** the `origin-private` implementation's `quant.py` and
  `csrc/fp8_prefill_attn.cu`, read as source rather than from documentation
  known to be stale. See `THIRD_PARTY_NOTICES.md` for that alias.

This document is the arch-independent definition of what the operator *is*. It
is the asset the repository split exists to have exactly one of.

Two consequences, both load-bearing:

- **Every implementation must satisfy this document**, whether it is the
  `mma_sync` mainloop for SM89/SM120 or a future `wgmma` mainloop for SM90. An
  implementation that satisfies a *different* contract is a different operator
  and needs a new `operator_contract_version`.
- **Oracle A tests this document, not an implementation.** It answers *did the
  implementation execute the intended quantized operator?* A failure against
  Oracle A can never be waived as expected quantization error. Fidelity to BF16
  attention is a separate question, answered by Oracle B, and the two must not
  be merged into one tolerance.

## 1. Mathematical operator

One rectangular request per invocation. Given post-RoPE BF16 `Q [T, Hq, D]`,
paged BF16 `K`/`V`, and a prefix of `P` already-cached positions, compute
bottom-right-causal attention:

> Query row `r` (0-indexed within the extend) attends to all `P` prefix
> positions plus positions `0..r` of the current chunk. Equivalently, over the
> `T x (P+T)` score rectangle, the causal diagonal has offset `K - Q = P`.

`torch.nn.functional.scaled_dot_product_attention(..., is_causal=True)` does
**not** implement this for `Q < K` and must never be used as the reference
without an explicitly constructed mask.

GQA maps query head `h` to KV head `h // (Hq / Hkv)`; v1 requires `Hq` divisible
by `Hkv`.

## 2. Declared support surface (v1)

| Property | v1 |
|---|---|
| `head_dim` | 256 only |
| `q_heads` : `kv_heads` | 16:4, 24:4, and 32:4 declared; 8:2 is tested as *generalization*, not production |
| `page_size` | 1 |
| Mode | `EXTEND` only |
| Mask | bottom-right causal |
| CUDA graphs | **supported** (declared 2026-08-31), CONDITIONAL on capacity mode: capture requires a capacity-reserved workspace (`quant.FP8PrefillWorkspace.reserve()` / `ops.plan_workspace()` — no alloc/sync on the per-call path; a plan miss raises `WorkspaceCapacityError`, never a silent realloc) and caller-owned `out`/`lse_out` destinations (§5 address stability). A grow-on-demand workspace remains eager-only by construction. Evidence: `tests/kernel/test_cuda_graph_capacity.py` (capture/NaN-poison/changed-value replay, the #4272 pattern) green on the pinned SM120 target, with the 72 pre-existing goldens byte-exact under capacity mode. |
| Q dtype in | BF16, contiguous `[T, Hq, 256]` |
| K/V dtype in | BF16 paged `[num_pages, 1, Hkv, 256]`, unquantized pool |
| Output | BF16, preallocated, shaped like Q |
| LSE | **Implemented 2026-08-30** as an optional trailing output (gap G1 closed at kernel level): pass a contiguous fp32 `[H, M]` buffer to receive the **base-2** log-sum-exp per (head, row) — `m·log2e + log2(l_true)`, the FA2/CUTLASS wrapper convention; under fp8-PV the 448 P-scale fold is subtracted back out, computed from the PRE-rounding `l`. Omitting the buffer preserves v1 output-only behaviour byte-exactly (proven: the 72 goldens pass unchanged on the LSE build, and a dedicated test asserts O bytes identical with/without the request). Validated against Oracle A and an independent `logsumexp` route on both tiers. Exposure through the public API is the `package_api_version` bump. |

Anything outside this surface raises a **typed capability error**. Benchmark and
qualification profiles fail on fallback; only a consuming framework may catch
that one declared error and route to a named stock backend, with a prominent
counter.

The v1 output-only choice has a consequence worth stating: FlashInfer-Bench's
stock paged-GQA definition returns output *and* FP32 LSE, so v1 needs its own
output-only definition rather than reusing the stock one. An LSE-producing
variant is a new `package_api_version`, not a flag.

## 3. Preprocessing contract

The transforms are **part of the operator**, not a benchmark preamble. The
inclusive path is the honest reusable result; a core-only number that excludes
these is a diagnostic.

Constants: `BLK = 64` (quantization/scale tile in both row dimensions),
`MPAD = 128` (Q row padding granularity, the widest CTA's `BM`), `SLAB = 32768`
(gather slab), `FP8_MAX = 448.0`, `INT8_MAX = 127.0`.

**`SLAB` is numerics-visible and normative** (pinned 2026-08-30, found by the
Q0 coverage audit): two operations depend on the slab structure in final-ulp
fp32 — the K/V **channel-mean accumulation** (per-slab partial sums, summed in
slab order) and the K **rotation GEMM batching** (one `[KVH, ≤SLAB, D] @ [D, D]`
per slab; cuBLAS selects its reduction by shape). `SLAB % BLK == 0` is required.
The one-shot mathematical forms agree to ~1e-5 but not bitwise;
`oracle_a.preprocess_kv(slab=...)` implements both semantics and
`tests/kernel/test_multislab.py` holds the byte-equality above one slab.

### 3.1 Shared rotation

Q and K are rotated by the **same orthonormal Hadamard** matrix of size `D`
before quantization (incoherent processing, FA3-style). Scores are exactly
invariant under a shared orthonormal rotation; within-row outliers — RoPE'd keys
carry large shared channel offsets — spread across dimensions, which is what
linear INT8 needs.

Scope of the claim (narrowed 2026-08-30): exactness under a shared orthonormal
rotation is unconditional; the *benefit* is model-dependent. SageAttention2's
published ablation measured QuaRot-style rotation as no better than no smoothing
on sampled model layers, while this project measured worst-head per-layer error
2.2% → 0.55% on this model's real keys — evidence currently recorded as prose,
to be reproduced as a committed artifact (gap G18). Do not assert the rotation
as generally beneficial upstream.

The rotation is computed in **true FP32**. cuBLAS selects its algorithm partly
from workspace size, and split-K variants reduce with atomics, so the workspace
must be pinned for the rotation to be reproducible. This is a normative
requirement, not a test detail. **The normative pin is
`CUBLAS_WORKSPACE_CONFIG=":4096:8"`**, set before the first cuBLAS handle
exists; `quant._true_fp32_matmul` warns on entry when the environment disagrees
(it is too late for the library to set it itself), and the 72 goldens pass on
two independent dies under this pin.

### 3.2 Q

- Rotate, then take `amax` over `D` **per row**, clamped below at `1e-8`.
- Scale `= amax / lim`, where `lim` is `INT8_MAX` (default) or `FP8_MAX`.
- **The softmax scale is folded into the Q scales**: the exported scale is
  `scale * sm_scale`. The kernel does not apply `sm_scale` separately.
- Rows in `[T, ceil(T, MPAD))` are compute padding: bytes zero, `amax` clamped
  to `1e-8` so the scale is small-positive rather than 0 or NaN, scores are
  all-zero, and the rows are sliced off by the caller.

Per-**row** Q scales are normative. Per-64-block scales let one outlier row
coarsen 63 neighbours, measured as the dominant term of the conservative mode's
excess error. Any documentation describing per-block Q scaling is stale.

### 3.3 K

- Channel mean per KV head over **all `N` gathered positions**; quantize
  `K - mean_channel(K)`.
- Rotate, then per-`BLK`-tile `amax` → INT8 (default) or E4M3.
- **No correction term is needed.** The dropped `q · mean` is constant across
  every KV column of a query row and softmax is shift-invariant per row, so
  centering is exact, not approximate.

### 3.4 V

- Channel mean per KV head over all `N`; centre by it.
- Per-`BLK`-tile `amax` of the **centred** values → `vs_t`.
- **P-underflow guard (normative):** floor `vs_t` at `vs_max / 16` before
  dividing by `FP8_MAX`. The kernel packs `P` as `p · 448 · (vs_t / vs_max)`;
  ratios below `1/16` push packed `P` into E4M3 subnormals exactly when a quiet-V
  tile carries high attention mass. This produced measured downstream-output
  damage and is a correctness requirement, not a tuning constant.
- Store as E4M3 in the **SIGMA-permuted, transposed, tile-major** `V^T` layout
  of §3.5.
- The channel mean is added back **exactly in the kernel epilogue**, valid
  because softmax weights sum to 1. This neutralises massive-activation V
  channels that would otherwise consume the FP8 range.

### 3.5 SIGMA layout (`layout_version: 1`)

Within each 32-row half of a `BLK`-row KV tile, V rows are permuted by

```
SIGMA32 = [ 0, 1, 8, 9, 2, 3,10,11, 4, 5,12,13, 6, 7,14,15,
           16,17,24,25,18,19,26,27,20,21,28,29,22,23,30,31]
SIGMA64[j] = (j // 32) * 32 + SIGMA32[j % 32]
```

Both sides of the PV matmul agree on the reordering, so **any bijection is
exact**; this specific one is chosen so each thread's own `S` accumulators pack
directly into its PV A-fragment registers with no cross-lane traffic. That
property is a consequence of the measured `mma.m16n8k32.e4m3` A-fragment byte
order (see `upstream/CLAIMS.md`), so it is tied to the `mma_sync` family. A
`wgmma` implementation will need its own layout and therefore a new
`layout_version`.

### 3.6 P and accumulation

- FP32 online softmax.
- Online softmax bounds `p <= 1`, so **448 is the exact per-row `amax` FP8 scale
  for P**, and it folds into the exponential: `p · 448 = ex2(fma(s, log2e,
  log2(448) - m · log2e))` — zero extra instructions. The per-tile V dequant
  ratio `r_t = vs_t / vs_max` folds into the same constant; the `l`-sum takes a
  one-FMUL-per-tile correction.
- QK accumulates in **INT32** (INT8 path) — exact, so the only QK error is input
  rounding. PV accumulates in **FP32**.
- Conditional alpha rescale: the `o_acc *= alpha` body runs only when a
  warp-uniform vote finds some `alpha != 1.0f`. **Exact by construction** —
  `x * 1.0f` is the identity and the vote compares bitwise, so correctness does
  not depend on `ex2.approx(0) == 1`; only the skip *rate* does. NaN alpha
  compares `!= 1` and still rescales.

## 4. Numerical semantics that Oracle A must pin

Oracle A implements this section in simple high-precision code and validates
intermediate boundaries — generated scales, quantized Q/K/V, means and
corrections, packed layouts — not only the final output.

- Reduction order where normative, and where it is explicitly *not*.
- Rounding, saturation, zero, NaN, Inf, and scale-floor behaviour.
- The `1e-8` amax floor and the `vs_max/16` V floor.
- Layout and transposition semantics.
- Bottom-right mask and prefix offset.
- GQA head mapping.
- Online-softmax rescaling semantics.

"Exact contract" does **not** mean bit-identical to hardware FP32 under a
different reduction order. Compare the final output with a contract-specific
tolerance; reserve *bit-exact* for the explicitly pinned target/compiler
regression lane, which is per-target and not portable.

## 5. Memory and workspace semantics

- One persistent workspace, grown monotonically, reused across layers and
  forwards. Steady state is gather + quantization traffic only.
- The caller owns workspace and destination output; their addresses and capacity
  must remain stable across graph replay.
- `run` enqueues on the caller's current stream and does **not** implicitly
  synchronise.
- **Stale-workspace hazards are contract-level, not implementation detail.**
  Rows beyond the current `N` may hold data from a longer earlier request. Stale
  K/Q is harmless (masked before `exp`, or sliced from the output). Stale V is
  **not**: it can hold FP8 NaN encodings that poison `0 * NaN` in the PV matmul,
  so the KV-boundary tile's tail columns must be explicitly zeroed on every
  build. Any conforming implementation must do this or prove it cannot observe
  stale V.
- Every conformance suite must include workspace poisoning, reuse with changed
  values, and long-after-short invocation.

## 6. Known contract-level limitations

1. **The quantized prefix is suffix-dependent.** The K/V channel means and
   `vs_max` are computed over all `N` gathered positions, so appending tokens
   changes the quantized representation of the *existing* prefix. This is what
   blocks a trivial append-only transformed KV cache. A stable design needs
   per-tile centering with a per-query `q · mean_tile` score correction, per-tile
   V centering with each tile's probability mass multiplying its V mean in the
   accumulator, and calibrated rather than suffix-dependent scale floors. That is
   `operator_contract_version: 2` territory.
2. **`head_dim` 256 only.** Generalizing needs HD-templated shared-memory sizing
   and a second tile configuration.
3. **`page_size` 1 only.** Larger pages need gather generalization or kernel-side
   page-table addressing.
4. **The KV pool must be unquantized** (BF16/FP16). An FP8 or FP4 pool would
   double-quantize.
5. ~~No LSE in v1~~ **Resolved 2026-08-30**: the optional base-2 LSE output
   landed (see the section 2 row), unblocking split-KV composition at the API
   level. The published v0.3.0 manifest correctly records `returns_lse: false`
   for the artifact as it was measured; the next release records `true` with
   the `package_api_version` bump.

## 7. Open items before this becomes normative

- [x] ~~Measure the true preprocessing cost~~ **Resolved 2026-08-30 by
      measurement**: 8.21 s per full schedule (the ~0.2–0.3 s claim was wrong by
      ~30x), then reduced to **6.19 s** by the bit-exact Tier A restructure —
      identical on both allocations. Claims K8/K10.
- [x] ~~The `_true_fp32_matmul` workspace pin~~ **Resolved**: the constant is
      now stated in section 3.1 and checked at runtime.
- [ ] Decide whether the `SGLANG_FP8_PREFILL_*` environment switches (`QK`,
      `K_CENTER`, `BF16_HEADS`, `ZERO_WS`, `MIN_TOKENS`) are contract variants,
      debug-only, or consumer configuration. Anything that changes numerics and
      survives into production is a contract variant and must be named in the
      manifest. **Still open.**
- [x] ~~The typed capability-error taxonomy~~ **Resolved by implementation**:
      `CapabilityError(ValueError)` is the single catchable class, raised by
      `ops.check_request` before any CUDA work, with 39 rejection tests pinning
      one match-string per refusal (`tests/test_ops_rejection.py`).
