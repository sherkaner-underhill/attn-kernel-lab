<!-- SPDX-License-Identifier: Apache-2.0 -->
# Emulating FlashInfer #4714's FP8 scheme on the 25-cell grid

`fp8_pool_emulation.py` reproduces the quantization arithmetic of
flashinfer-ai/flashinfer#4714 (pinned at its head commit `004d1aea`) inside
the donor probe's oracle-checked evaluation, on the same seeded fixtures the
lab's own scheme is measured on. Record: `fp8_pool_emulation.json`.

**What was reproduced, from that PR's own source**: per-tensor scalar scales
for Q/K/V (`sm120_fmha.py`), an E4M3 KV pool, fp32 online softmax, and P
converted to E4M3 **without a scale** while the row-sum denominator keeps the
pre-conversion fp32 values (`fmha_prefill_fp8_tma.py`). No centering, no
rotation. The emulation grants the scheme best-case amax/448 calibration on
every operand (the PR's default scales are 1.0, which would saturate the
heavy-tailed fixtures; testing that default would be a straw man).

**Validity control**: a diagnostic cell (`pr4714_p448diag` — their operands,
the lab's 448-fold P handling) is arithmetically identical to the donor
probe's `pertensor_p` scheme and is computed here through an independent code
path. Across all 25 cells the two agree to a max delta of **0.0** in
row-rel-L2 mean, so the fold=1.0 cell differs from validated arithmetic only
in the one constant that defines it.

## Results at full depth (N = 446,335), row-rel-L2 mean vs the fp32 reference

| Distribution | #4714 emulation | per-tensor (cuDNN class) | lab (`lab_p`) | lab advantage |
|---|---|---|---|---|
| gaussian | 5.59% | 4.65% | 4.28% | 1.3× |
| heavy_t3 | 12.09% | 12.08% | 3.60% | **3.4×** |
| rope_like | 15.28% | 14.95% | 4.43% | **3.4×** |
| v_outlier | 5.92% | 4.92% | 4.32% | 1.4× |
| v_massive | 0.613% | 0.651% | 0.0017% | **~360×** |

Worst-row cosine at the same depth — the K14 failure mode:

| Distribution | #4714 emulation | per-tensor | lab |
|---|---|---|---|
| heavy_t3 | **0.050** | 0.049 | 0.954 |
| rope_like | 0.979 | 0.979 | 0.999 |
| v_outlier | 0.932 | 0.937 | 0.965 |

## Findings, stated fairly

1. **The scheme lands in the per-tensor error class.** On every distribution
   its mean error is statistically indistinguishable from the plain
   per-tensor cell (heavy_t3: 12.09% vs 12.08%). This is K14's central
   result — granularity and dtype plumbing are not the quality
   differentiator; the transforms are — now shown to cover the closest
   in-flight neighbour as well.
2. **The worst-row collapse transfers verbatim.** On heavy-tailed keys at
   depth the emulated scheme's worst row collapses to cosine 0.050 (row error
   136%) while its mean stays a mild-looking 12% — the exact silent failure
   mode K14 documents for per-tensor FP8, invisible to mean-based evals.
3. **The unscaled-P choice is NOT the problem.** Attributed cost of
   converting P to E4M3 without a scale: +0.007 to +0.25 percentage points
   across the deep cells — real but minor. This is worth saying plainly
   because it is exculpatory for #4714: its quality gap versus this lab's
   scheme comes almost entirely from the absent centering/rotation
   transforms, not from its P handling.
4. **The lab scheme's margin is distribution-shaped**: ~1.3× on unstructured
   gaussian operands, 3.4× on heavy tails and RoPE-like structure, and orders
   of magnitude on massive-activation V channels — consistent with the
   transform-attribution ablation in `RESULTS.md`.

## Boundaries

Everything the donor probe declines to claim, this probe declines too: no
kernel from either project executes here; distributions are synthetic and
chosen to isolate failure modes, so absolute levels do not transfer to real
activations; this is boundary (a) attention-output evidence, not a task or
model-quality statement. The emulation is of the PR at one pinned commit;
the scheme could change under review. And the storage difference is not
modelled beyond its arithmetic: #4714 quantizes the resident KV pool
(paying its error once per token at write time), while the lab quantizes
transiently from an unquantized pool — at this boundary the arithmetic is
what matters, and both are given the same seeded inputs.
