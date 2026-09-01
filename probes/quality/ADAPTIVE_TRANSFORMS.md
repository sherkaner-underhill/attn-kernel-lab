<!-- SPDX-License-Identifier: Apache-2.0 -->
# Adaptive exact distributional transforms on the quality grid

`adaptive_transforms.py` extends the lab's quality instrument with **exact**
adaptive transforms beyond its current mean-centering + fixed Hadamard, and
measures whether they pay. Record: `adaptive_transforms.json`. Every fixture,
reference, metric and scheme convention is imported unchanged from the sealed
donor probe `pertensor_vs_finegrained.py`; the real `src/attn_kernel_lab/quant.py`
runs unmodified and the transforms are injected through the workspace's own
`hadamard` attribute (which `quant.py` reads at call time and which broadcasts
over a per-head batch of matrices) and through the gather index vector.

## The frame

The attention score is bilinear, so the space of output-preserving rewrites is
affine-linear and larger than the shipped scheme uses:

* **QK.** For any invertible `A` and offset `mu`, `q . k == (A^-T q) . (A(k-mu)) + q.mu`,
  and `q.mu` is constant across every kv column of a query row, so row-wise
  softmax removes it exactly.
* **PV.** For any invertible `M`, `out == M^-1 (sum_j p_j M v_j)` because
  `sum_j p_j == 1` — the identity the shipped epilogue already uses to add
  `vmean` back exactly.
* **Key order.** Attention sums over keys, and every prefix key is visible to
  every query row of an extend chunk, so any permutation of the **prefix**
  columns applied to K and V rows together is exact. The current chunk's tail
  keys carry the causal mask and keep their order.

The shipped operator uses the **first** moment adaptively and the **second**
moment only through a fixed, data-independent rotation plus per-64-tile local
scales. The cells below make the second moment adaptive.

| cell | transform (K side / Q side) | family |
|---|---|---|
| `lab_diag` | `diag(1/sigma_c)` / `diag(sigma_c)` after centering | SmoothQuant, Outlier-Suppression+ |
| `lab_whiten` | `Sigma^-1/2` / `Sigma^+1/2` | adaptive cousin of QuaRot / SpinQuant / FA3 incoherence |
| `lab_whiten_bal` | `Sigma^-1/4` / `Sigma^+1/4` | balanced split exponent |
| `lab_perm` | prefix keys sorted by per-token Linf of centered-rotated K | — |
| `lab_perm_v` | prefix keys sorted by per-token Linf of centered V | — |
| `lab_veq` | V channels equalized, inverse in the fp32 epilogue | SageAttention2 family |

`*_had` composes the transform with the shipped Hadamard, because the baseline
**has** that rotation and a variant that drops it is not comparable to it.

## Validity controls

| control | result |
|---|---|
| **Identity** — transform set to the plain Hadamard, no permutation, no V gain, must reproduce the donor's `lab` / `lab_p` cell | max metric delta across all 25 donor cells (both schemes, every metric field): **0.0** |
| **Exactness** — transform + compensation at FULL precision, no quantization anywhere, against the transform-free fp32 reference | worst `row_rel_l2_max` over 7 fixtures x 10 transforms: **1.69e-05** |

The identity control means the new runner is not merely close to the donor's
lab path, it *is* that path. The exactness control means the invariance algebra
holds before quantization enters: the residual is fp32 rounding only, and its
largest value sits on `heavy_t3` under `lab_whiten`, whose `Sigma^-1/2` has the
worst conditioning on that fixture.

## Two extra fixtures, and why

The donor's five all have **isotropic keys once centered** — gaussian, t3 and
the RoPE-offset case are iid per channel, so the centered key covariance is a
multiple of the identity. Measuring adaptive whitening only there cannot
distinguish "the transform does not work" from "the fixture has nothing to
transform". Two positive controls were added, in the spirit of the donor's own
`v_massive` ("EXTRA beyond the four requested"):

* **`k_aniso`** — key covariance with condition number 1e4 in a seeded **random**
  eigenbasis. Measured `sigma_cond` 1.00e4, measured per-channel spread ratio
  **1.61**.
* **`k_chan`** — the same spectrum on the **channel axes**. Measured
  `sigma_cond` 1.00e4, measured per-channel spread ratio **100.1**.

That pair is itself a result: at D=256 a random-basis anisotropy of condition
number 10,000 shows up as a per-channel variance spread of only 1.6x, because
each channel variance is an average of the whole spectrum under a random
rotation. **A diagonal cannot see random-basis anisotropy**, which is why the
two fixtures had to be separated to tell channel equilibration and adaptive
rotation apart.

## The ceiling: the QK grid is not the binding constraint

Before any transform is judged, the record measures what the operand groups are
worth. `qk_exact_floor` runs Q and K **exact** (the Hadamard is orthonormal, so
rotating an unquantized operand is exact) with V and P through the lab path.
It is the hard lower bound on `lab_diag`, `lab_whiten*` and `lab_perm`, which
touch the QK grid only.

Row-rel-L2 mean at N = 446,335:

| | gaussian | heavy_t3 | rope_like | v_outlier | v_massive | k_aniso | k_chan |
|---|---|---|---|---|---|---|---|
| bf16 anchor | 0.2911% | 0.7927% | 0.9315% | 0.3119% | 0.0004% | 0.2937% | 0.3005% |
| `lab_p` baseline | 4.2798% | 3.6049% | 4.4317% | 4.3224% | 0.0017% | 4.2794% | 4.3178% |
| **`qk_exact_floor`** | 4.1195% | 3.2877% | 4.2721% | 4.1296% | 0.0017% | 4.1024% | 4.1538% |
| ⇒ best possible QK ratio | **0.963x** | **0.912x** | **0.964x** | **0.955x** | 1.000x | **0.959x** | **0.962x** |
| `v_exact_floor` (indicative) | 2.8967% | 1.2772% | 3.0677% | 3.1154% | 0.0702% | 2.9160% | 2.9060% |
| ⇒ indicative V ratio | 0.677x | 0.354x | 0.692x | 0.721x | *n/a* | 0.681x | 0.673x |

**Making Q and K perfect buys between 3.6% and 8.8%.** That single row explains
every `1.000x` in the tables below: the QK operand path carries roughly 8% of
the squared output error on this grid, so an adaptive second-moment transform
confined to it is bounded at a few percent before it is even written.

*Caveat on `v_exact_floor`:* it is indicative, not a bound. Exact V also removes
the per-tile `r_t` fold from P and (on `v_massive`) the centered-V/exact-mean-
add-back path the fixture depends on, which is why its `v_massive` entry is
*worse* than the baseline and is marked *n/a*. The `qk_exact_floor` row has no
such coupling and is a genuine bound.

## Results at full depth (N = 446,335), `lab_p` ratio to the baseline

| scheme | gaussian | heavy_t3 | rope_like | v_outlier | v_massive | k_aniso | k_chan |
|---|---|---|---|---|---|---|---|
| `lab_diag` | 1.000x | **1.662x** | 1.000x | 0.998x | 0.999x | 0.998x | 1.034x |
| `lab_diag_had` | 0.999x | 0.993x | 1.001x | 0.993x | 1.001x | 0.998x | 1.000x |
| `lab_whiten` | 1.001x | **1.647x** | 1.000x | 0.994x | 1.000x | 0.996x | 1.033x |
| `lab_whiten_had` | 0.999x | 1.004x | 1.000x | 0.999x | 1.000x | 0.997x | 0.998x |
| `lab_whiten_bal` | 1.000x | **1.662x** | 1.001x | 0.999x | 0.999x | **0.976x** | 1.005x |
| `lab_whiten_bal_had` | 0.999x | 0.990x | 1.000x | 0.994x | 1.001x | **0.975x** | **0.979x** |
| `lab_perm` | 0.988x | 1.055x | 1.006x | 0.992x | 1.006x | 1.013x | 1.010x |
| **`lab_perm_v`** | **0.935x** | **0.937x** | **0.936x** | 1.007x | **0.958x** | **0.956x** | **0.958x** |
| `lab_veq` | 1.000x | 1.005x | 1.000x | **0.988x** | 1.000x | 1.000x | 1.000x |
| `lab_diag_veq_had` | 0.999x | 0.999x | 1.001x | **0.980x** | 1.001x | 0.998x | 1.000x |

Worst-row cosine at the same depth (the silent-failure metric):

| scheme | heavy_t3 | v_outlier | gaussian |
|---|---|---|---|
| baseline | 0.9542 | 0.9646 | 0.9986 |
| `lab_diag` (no Hadamard) | **0.6903** | 0.9628 | 0.9986 |
| `lab_whiten` (no Hadamard) | **0.6524** | 0.9769 | 0.9985 |
| `lab_diag_had` | 0.9631 | 0.9780 | 0.9985 |
| `lab_perm_v` | 0.9542 | 0.9757 | 0.9988 |
| `lab_veq` | 0.9542 | 0.9799 | 0.9986 |

Depth trend for the two cells that move, `lab_p` ratio:

| `lab_perm_v` | 4096 | 32768 | 131072 | 262144 | 446335 |
|---|---|---|---|---|---|
| gaussian | 1.004x | 0.998x | 0.995x | 0.972x | **0.935x** |
| heavy_t3 | 1.017x | 0.939x | 0.911x | 0.939x | **0.937x** |
| rope_like | 1.008x | 1.003x | 0.989x | 0.980x | **0.936x** |
| v_outlier | 1.056x | 1.000x | 0.998x | 1.003x | 1.007x |
| k_aniso | 1.004x | 0.994x | 0.987x | 0.979x | **0.956x** |

| `lab_whiten_bal_had` | 4096 | 32768 | 131072 | 262144 | 446335 |
|---|---|---|---|---|---|
| k_aniso | 0.973x | 0.972x | 0.974x | 0.975x | 0.975x |
| k_chan | 0.974x | 0.974x | 0.975x | 0.975x | 0.979x |
| donor's five | 0.974–1.012x, no trend | | | | |

## Attribution, per fixture

**gaussian / rope_like** (iid keys, iid V; `rope_like`'s offset is a first
moment the shipped centering already removes). Measured per-channel spread
1.007, `sigma_cond` 1.10 — there is no second moment to adapt to, and every
second-moment cell lands at 1.000x, as it must. The only movement is
`lab_perm_v`, which is a *tiling* transform and does not depend on any moment.

**heavy_t3.** The one fixture where dropping the Hadamard is catastrophic:
`lab_diag` and `lab_whiten` land at **1.66x** and their worst-row cosine
collapses from 0.954 to **0.65–0.69**. The operand diagnostics say why —
`k_rel_step` rises from 0.0361 to 0.2034 and `q_rel_step` from 0.0227 to 0.0503
when the rotation is replaced. Infinite kurtosis means the damage is a
*within-row outlier*, and only a dense rotation spreads that across dimensions;
an adaptive second moment is a per-channel or per-eigendirection statistic and
is blind to it. Composed with the rotation (`*_had`) the adaptive transforms
recover to 0.990–1.004x — that is, they add nothing, and subtract nothing.

**v_outlier** (4 V channels x50 variance). The only fixture where `lab_veq`
moves: 0.988x alone, 0.980x combined with the K diagonal, with worst-row cosine
0.965 → 0.980. Measured `v_gain_ratio` 64 (the power-of-two-rounded channel
spread) versus 1.0 on every zero-variance-spread fixture. `lab_perm_v` is
*counterproductive* here (1.007x): the V error is channel-driven, not
token-driven, so sorting tokens by their Linf sorts by which token happens to
have a large draw in the four loud channels and buys nothing while disturbing
the tile grouping.

**v_massive** (V channels offset by +50). Baseline 0.0017% — the exact
epilogue add-back already reduces this fixture to nothing, and no transform
here is relevant. The `v_gain_ratio` is measured at 1.0, correctly: the
equilibration reads the *centered* V second moment, and a channel mean is not
a channel variance.

**k_aniso** (random-basis anisotropy, cond 1e4). The designed test for adaptive
rotation, and it answers cleanly: full whitening gets 0.996x, the **balanced
split gets 0.975x**. Error is conserved across the dot product, so moving all
of the burden to the Q side is not the optimum; `Sigma^-1/4` against
`Sigma^+1/4` is measurably better than `Sigma^-1/2` against `Sigma^+1/2`, and
the diagonal gets 0.998x because — as designed — it cannot see this.

**k_chan** (axis-aligned, cond 1e4, per-channel spread 100x). The most direct
statement about the fixed rotation. Without the Hadamard, the adaptive diagonal
lands at **1.034x** — *worse than the baseline* — and the operand diagnostic
shows exactly why: `q_rel_step` rises 0.0239 → 0.0450 while `k_rel_step` stays
0.0326. **The diagonal moved the burden onto Q and the total went up.** The
Hadamard, which is not adaptive at all, already equalizes a channel-variance
spectrum to its mean, so `lab_diag_had` is 1.000x: there was nothing left for
the adaptive diagonal to take. Only the balanced whitening composed with the
rotation improves, at 0.979x.

## Where `lab_perm_v`'s win actually comes from

The obvious mechanism is wrong, and the record says so. The first hypothesis was
that sorting makes the per-tile V dequant ratios `r_t` friendlier so the packed
P grid gets finer. The recorded `r_t` statistics refute it: under `lab_perm_v`
the mean `r_t` *falls* (gaussian 0.720 → 0.673) and the 1/16 floor binds *more*
(heavy_t3 0.627 → 0.906 of tiles).

The `perm_ablation` section measures the decomposition instead (N = 446,335,
row-rel-L2 mean with one operand group quantized):

| fixture | cell | `v_recon_rel_fro` | `v_only` | `qk_only` |
|---|---|---|---|---|
| gaussian | identity | 0.026510 | 3.1354% | 1.1938% |
| gaussian | `lab_perm` | 0.026508 | 3.0869% | 1.1498% |
| gaussian | **`lab_perm_v`** | 0.026566 | **2.7474%** | 1.1943% |
| heavy_t3 | identity | 0.026020 | 2.6529% | 1.1566% |
| heavy_t3 | **`lab_perm_v`** | 0.026393 | 2.6517% | 1.1614% |
| v_outlier | identity | 0.026176 | 2.9649% | 1.2777% |
| v_outlier | `lab_perm_v` | 0.026374 | 3.0156% | 1.2972% |

Two distinct things are happening:

1. **On light-tailed V (gaussian, and by the depth trend `rope_like`,
   `k_aniso`, `k_chan`), the win is V-placement, not V-accuracy.** The V tensor's
   own reconstruction error is *unchanged* (0.026510 → 0.026566, marginally
   worse), while V's contribution to the output falls 12%. That is consistent:
   the Frobenius norm weights every token equally, but the output is a
   `p`-weighted sum over ~446k tokens, and an unsorted 64-token tile has its
   scale set by its single loudest member, so 63 quiet tokens carry error
   proportional to a neighbour they have nothing to do with. Sorting removes
   that contamination for the many quiet tokens without helping the few loud
   ones — the total stays put, the weighted total falls. This also explains the
   monotone depth trend (1.004x at 4k to 0.935x at 446k): the more tiles there
   are, the heavier the tail of "loudest member of a tile" and the more
   contamination there is to remove.
2. **On heavy_t3 the win is in the P path and is NOT explained here.** Both
   `v_only` and `qk_only` are flat (2.6529 → 2.6517, 1.1566 → 1.1614) while
   `lab_p` improves to 0.937x and the fp32-P `lab` form does not move (1.002x).
   The only remaining difference is P, and the `r_t` statistics rule out the
   simple "finer P grid" story. A plausible but **unmeasured** hypothesis: with
   91% of tiles pinned at the 1/16 floor rather than 63%, the packed-P fold
   `448*r_t` becomes nearly *uniform* across tiles instead of varying by 16x, so
   P's precision stops varying tile to tile. This is stated as a hypothesis, not
   a finding, and is the first open question below.

## Practicality against the 6.19 s/schedule preprocessing budget

The committed figure (K10) is **6.19 s** of preprocessing for the whole
14-chunk x 16-layer schedule. That schedule sums 3,428,223 kv tokens per layer,
so one terminal-depth call is 122.9 schedule-equivalents of gather traffic;
per-call costs that do **not** scale with N are multiplied by 224 calls instead.
Statistics passes are assumed to have the same access pattern as the existing
centering pass, as directed. Measured on the local machine at N = 446,335,
4 KV heads:

| | measured | schedule-equivalent | verdict |
|---|---|---|---|
| bare read of V + channel sum (what the shipped statistics gather already does) | 5.2 ms | 0.64 s | reference point |
| same read + per-token Linf | 12.8 ms | 1.57 s | the `lab_perm_v` statistic |
| `argsort` of 446,207 keys | 0.09 ms | 0.01 s | negligible |
| this probe's combined statistics pass (means + full covariance + Linf + V moments) | 91–100 ms | **11.2–12.3 s** | ~2x the entire budget |
| `Sigma^-1/2` build (eigh of 4 x 256x256, N-independent) | 26.7–30.1 ms | **6.0–6.7 s** (x224 calls) | ~1x the entire budget on its own |

Per cell:

| cell | statistics needed | extra passes over K/V | folds into | verdict |
|---|---|---|---|---|
| `lab_diag` | per-channel 2nd moment of centered K — obtainable as `E[k^2] - mu^2` from **one extra accumulator in the existing statistics gather** | **0** | the rotation matrix becomes `diag(1/sigma) @ H`: same GEMM shape, same pass | free, and buys nothing |
| `lab_whiten*` | full `Sigma` per KV head: an N x D x D outer-product accumulation (117 GMAC/call at terminal depth), then an `eigh` per call | 0 traffic, large arithmetic | the same GEMM shape once `A` exists | **does not fit.** The `eigh` alone is 6.0–6.7 s/schedule against a 6.19 s budget, before the covariance arithmetic |
| `lab_perm` | per-token Linf of centered **rotated** K — the rotation currently happens in the *quantization* pass, so the statistic needs its own rotated pass | 1 (expensive) | the gather is already index-driven, so applying the permutation is free | not worth it; it is a QK transform and the QK ceiling is 0.96x |
| `lab_perm_v` | per-token Linf of centered V — same read, different reduction axis, from the existing statistics gather | **0 if fused; +1.6 s/schedule as a standalone torch pass** | **nothing else changes.** The gather already takes an arbitrary `idx`; permuting it costs zero in the quantization pass, zero in the kernel, and zero in the epilogue | the only candidate |
| `lab_veq` | per-channel 2nd moment of centered V — one extra accumulator in the existing gather | **0** | gain into the fp32 V conversion the quantization pass already performs, inverse into the epilogue that already adds `vmean` back; power-of-two gains make both exponent adjusts | free, pays on `v_outlier` only |

Two honest costs that the timings do not show:

* **`lab_perm_v` destroys gather locality.** A sorted prefix turns the paged-pool
  gather into a random-ordered one. The gather is already index-based so it is
  *correct* at no cost, but the DRAM access pattern degrades, and this
  evaluation harness cannot measure that — it reads a contiguous fixture, not a
  paged pool. Bucketing tokens into magnitude deciles within a page window,
  rather than fully sorting, would keep most of the effect and most of the
  locality; that is untested.
* **The permutation must be exact in the *pairing*, not in the statistic.** Any
  permutation is exact as long as K and V rows move together, so the sort key
  may be stale, approximate, or computed from uncentered V without breaking
  correctness. That is what makes a fused single-pass implementation plausible.

## Related work, stated as hedging rather than novelty

What is measured here are known families, put in this operator's context:

* **Channel equilibration** (`lab_diag`) is the SmoothQuant / Outlier-
  Suppression+ migration move applied inside attention rather than to a linear
  layer's weights.
* **Adaptive rotation** (`lab_whiten*`) is the data-dependent cousin of the
  fixed/learned rotations in QuaRot, SpinQuant and FlashAttention-3's incoherent
  processing — the shipped Hadamard is already in that family, and the question
  asked here is only whether making it adaptive helps.
* **V-side channel smoothing** (`lab_veq`) is in the SageAttention2 family, as
  is the V mean-centering the operator already ships.
* **Prefix-key permutation tiling for paged extend** (`lab_perm`, `lab_perm_v`)
  is not something the author has found published. **Absence of a citation is
  not evidence of absence** and no priority is claimed; magnitude-ordered
  grouping is a standard idea in quantization generally, and the specific
  observation here is only that paged extend makes it *free to apply* because
  the gather is already index-driven and every prefix key is visible to every
  query row.

## Recommendation

**One cell merits a kernel-adjacent change, and it is not one of the
second-moment transforms.**

1. **`lab_perm_v` — worth prototyping.** 0.935–0.958x on six of seven fixtures at
   terminal depth, improving monotonically with depth (1.004x at 4k → 0.935x at
   446k), which is the regime this operator exists for. It needs one extra
   reduction axis in a pass the pipeline already runs and an `argsort` that
   costs 0.09 ms; nothing in the kernel, the scale scheme, or the epilogue
   changes. Two things must be settled before it is worth anything: the
   gather-locality cost on a real paged pool (unmeasured here), and the
   `v_outlier` regression (1.007x), which suggests the sort key should be gated
   on the V channel spread rather than applied unconditionally.
2. **`lab_veq` — cheap, narrow, defensible.** Free (one accumulator, one
   exponent adjust in each direction), 0.988x on `v_outlier` and exactly 1.000x
   everywhere else, so it is a no-op-or-win. It is a smaller effect than
   `lab_perm_v` and only fires on one failure mode, but it costs nothing and it
   composes (`lab_diag_veq_had` reaches 0.980x on that fixture).
3. **`lab_whiten*` — no.** The best variant reaches 0.975x, and only on
   constructed anisotropic fixtures the donor grid does not contain; the `eigh`
   alone would consume the entire preprocessing budget. If key covariance
   anisotropy is ever *measured* on real activations, revisit — and revisit with
   the **balanced** exponent, which beat full whitening 0.975x vs 0.996x.
4. **`lab_diag` — no.** It is free, and it buys nothing, because the fixed
   Hadamard already equalizes a channel-variance spectrum to its mean. On
   `k_chan` without the rotation it is actively *worse* (1.034x).
5. **Do not remove the Hadamard.** The strongest result in this record is
   negative: replacing the fixed rotation with an adaptive second moment costs
   **1.65–1.66x** on heavy-tailed keys and collapses the worst-row cosine from
   0.954 to 0.65. Incoherence and spectrum adaptation are different jobs.

The broader conclusion the ceiling row forces: **on this grid the QK operand
path is not where the error is.** Perfecting Q and K is worth at most 0.912x.
Anyone extending this instrument further should spend the effort on the V and P
paths, where the indicative headroom is 0.35–0.72x.

## Open questions

1. **Why does `lab_perm_v` help the online-P form on `heavy_t3` when neither
   `v_only` nor `qk_only` moves?** The uniform-`r_t` hypothesis above is
   untested. The clean experiment is to sweep the 1/16 P-underflow floor and
   watch whether the `lab_perm_v` advantage tracks the floor fraction.
2. **Does the permutation survive a real paged gather?** Everything here reads a
   contiguous fixture. The locality question is the whole practicality risk, and
   a windowed/bucketed variant should be measured against the full sort.
3. **Is real key covariance anisotropic, and in which basis?** `k_aniso` vs
   `k_chan` shows the answer decides between "adaptive rotation might pay" and
   "the Hadamard already did it". No real activation capture enters this
   repository, so this cannot be answered here.
4. **Is the balanced exponent 1/4 optimal?** Only `p in {1/4, 1/2}` was measured.
   The conservation argument suggests an optimum that depends on the Q and K
   spectra jointly, and `k_aniso` deliberately keeps Q isotropic; a fixture with
   co-anisotropic Q and K is untested.
5. **The V-side `M` was restricted to a diagonal.** The PV algebra admits any
   invertible `M`, and a rotation on V (with its inverse in the epilogue) is
   unmeasured — it would be the V-side analogue of incoherent processing, and
   the ceiling row says the V path is where the headroom is.

## Boundaries

Everything the donor probe declines to claim is declined here. No kernel
executes: `quant.py`'s real quantizer runs, but the attention itself is the
donor's materialized evaluation form, so tiling and accumulation order differ,
as they do for every scheme on this grid. Distributions are seeded synthetic
fixtures chosen to isolate failure modes, boundary (a) only; absolute levels do
not transfer to real activations, and `k_aniso`/`k_chan` are constructed
positive controls whose condition number was chosen, not measured. Preprocessing
costs are torch-level statistics passes on this harness, not a fused kernel's
preprocessing: they bound what a transform would cost, they do not predict it.
The whole run is 92 s on the local development tier and is diagnostic evidence,
not a promotable performance number.
