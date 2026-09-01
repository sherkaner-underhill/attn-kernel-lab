<!-- SPDX-License-Identifier: Apache-2.0 -->
# Attention transform research brief — state, ceilings, and open leads

Written 2026-09-01 as a handoff: everything measured so far about the
quantization transforms in this operator, what those measurements rule OUT,
and the experiments worth doing next. A reader picking this up cold needs
this file plus three probe writeups:

- `probes/quality/RESULTS.md` — the original 25-cell attribution (claim K14)
- `probes/quality/FP8_POOL_EMULATION.md` — the closest neighbour's scheme
  emulated on the same fixtures
- `probes/quality/ADAPTIVE_TRANSFORMS.md` — the exact adaptive-transform sweep

## 1. The frame: what "exact" permits

The attention score is bilinear (`s_j = q·k_j`) and softmax admits one
invariance (per-row additive shifts). The space of transforms that change
NOTHING about the mathematics — only the coordinates being quantized — is
therefore precisely:

1. **Constants**: `q·(k−μ)` differs from `q·k` by a row-constant → softmax
   invariant. On V, `Σpⱼvⱼ = Σpⱼ(vⱼ−μᵥ) + μᵥ` since `Σp = 1`. First-moment
   centering is free and the operator uses it adaptively.
2. **Any shared invertible linear map**: `q·k = (A⁻ᵀq)·(Ak)`. Orthogonal
   rotations (the operator's fixed Hadamard-family transform) are the special
   case; diagonals (SmoothQuant-style equilibration) and full whitening are
   the general case. On V, any invertible `M` with `M⁻¹` in the fp32
   epilogue.
3. **Permutations** — orthogonal, and in EXTEND mode the entire prefix is
   visible to every query row, so prefix keys/values may be reordered freely
   (the current chunk's causal tail must stay in place).
4. **Forbidden**: nonlinear per-coordinate maps (gaussianization by quantile
   or power transforms) — they do not commute with a bilinear form. Higher
   moments can only enter through the NUMBER FORMAT (companding); E4M3 for P
   is already that, and INT8-QK's exact INT32 dots forbid nonuniform codes on
   the QK side.

## 2. What is measured, and the ceilings it establishes

All numbers below are at the deepest cells (N = 446,335), row-rel-L2 mean
against the fp32 reference, on the seeded fixture grid. Ratios are relative
to the operator's production scheme (`lab_p`).

**The transforms, not scale granularity, carry the quality** (K14): per-tile
scales WITHOUT centering/rotation are indistinguishable from per-tensor
(14.91% vs 14.96% on RoPE-structured keys). The closest in-flight neighbour
(FlashInfer #4714: per-tensor scales, FP8 pool, no transforms) lands in the
per-tensor error class on every distribution, inherits the worst-row
collapse (cosine 0.050 on heavy tails while its mean looks benign), and its
one distinctive choice — unscaled E4M3 P — costs only +0.007..0.25pp. The
gap is the absent transforms.

**The QK side is nearly saturated** (`qk_exact_floor`): with Q and K exact
(no quantization at all on that side, V/P unchanged), total error only drops
to **0.912×**. Every conceivable QK-side transform improvement combined is
worth at most ~9%. This single measurement retires most of the
second-moment agenda.

**Incoherence and spectrum adaptation are different jobs**: replacing the
fixed Hadamard with covariance-derived whitening (no Hadamard) COSTS
1.65–1.66× on heavy tails and drops worst-row cosine 0.954 → 0.65. The
rotation's job is spreading tails across coordinates, and adaptive
second-moment transforms do not do it.

**Error is conserved across the dot product**: channel equilibration on K
measurably moved the burden onto Q (`q_rel_step` 0.024 → 0.045 on the
channel-outlier control) and raised total error; the balanced split
exponent (¼) beat one-sided whitening (0.975× vs 0.996×) on both
anisotropy controls. Any future two-sided transform must optimize the
BALANCE, not one operand.

**Whitening is out on cost regardless**: the per-head `eigh` alone measures
6.0–6.7 s/schedule against the entire 6.19 s preprocessing budget.

**Practical wins on the table**: `lab_veq` (diagonal V equilibration through
the existing epilogue) is free and a small win on V-outlier fixtures only
(0.988×). `lab_perm_v` — see below.

## 3. The live lead: prefix permutation tiling (`lab_perm_v`)

Reorder PREFIX keys/values so each 64-token quantization tile groups tokens
of similar magnitude; queries and the causal tail untouched; exactness is
permutation-orthogonality plus prefix-wide visibility in EXTEND.

Measured: **0.935–0.958× on six of seven fixtures, improving monotonically
with depth** (1.004× at 4k → 0.935× at 446k) — the only transform whose
value GROWS with context length, which is the regime this operator exists
for. Standalone cost ~1.6 s/schedule for the sort statistic (≈0 if fused
into the existing centering pass) plus a 0.09 ms argsort; no kernel, scale,
or epilogue changes.

Honest caveats, from the probe's own ablation:
- The mechanism is only PART-attributed. V's reconstruction error is
  unchanged while V's contribution to output error falls ~12% — a placement
  effect, not an accuracy effect — and on heavy tails there is a P-path
  component the probe could not attribute (recorded, not hand-waved).
- One fixture regresses (v_outlier, 1.007×), suggesting the sort key should
  be gated on V channel spread.
- **The decisive unknown is paged-gather locality**: the probe reads a
  contiguous fixture and cannot see what permuted access does to a real
  page_size-1 gather's memory behaviour. This is the whole practicality
  risk and the first thing to measure.

## 4. Open experiments, in rough priority order

1. **Permutation × paged gather**: extend `bench/candidate_bench.py`'s
   preprocessing lane (or a standalone probe) to measure gather bandwidth
   with a magnitude-sorted prefix index against the natural order, at
   446k, on real hardware. If the locality cost is small, `lab_perm_v`
   graduates to a kernel-adjacent change (it composes with everything else).
2. **Mechanism of the heavy-tail P-path effect**: sweep the P-tile floor
   (the `vs_max/16` guard) and the per-tile `r_t` statistics under
   permutation to pin the unattributed component before any implementation.
3. **Sort-key gating**: condition the permutation key on V channel spread to
   remove the v_outlier regression; re-run the 7×5 grid (seconds).
4. **Split-exponent sweep**: only {¼, ½} were measured; a 5-point sweep on
   the two anisotropy fixtures closes the balanced-transform question
   (bounded above by the 0.912× floor, so cap effort accordingly).
5. **V-side full matrix `M`**: the algebra admits any invertible M with
   M⁻¹ in the epilogue; only diagonals were measured. A random-rotation and
   a PCA variant on the v_massive/v_outlier fixtures would bound the
   headroom (V-side is NOT capped by the QK floor).
6. **Real-activation check**: all of the above is on synthetic fixtures by
   design; any transform that graduates should be re-measured through the
   private capture lane before a production claim.

## 5. Ground rules for follow-up work

- `probes/quality/pertensor_vs_finegrained.py`, `quality/q1_public.py`,
  `quality/metrics.py`, and everything under `src/` and `tests/kernel/` are
  digest-sealed by published records: new work goes in NEW probe files that
  import the donor (see `fp8_pool_emulation.py` / `adaptive_transforms.py`
  for the pattern, including the scrubbed env fingerprint).
- The validity-control discipline is not optional: identity configurations
  must reproduce the donor bit-for-bit, and every transform's compensation
  must be verified at full precision against the fp32 reference BEFORE
  quantization enters. Both controls are recorded in the probe's JSON.
- Cells are cheap (the full 7×5 grid runs in ~90 s on a 24 GiB card):
  measure before designing, and prefer adding a cell to arguing.
- Related work to cite when writing any of this up: SmoothQuant, Outlier
  Suppression+, QuIP/QuIP#, QuaRot, SpinQuant, SageAttention 1–3. The
  prefix-permutation idea has no citation we know of for paged EXTEND;
  absence of a citation is not evidence of absence.
