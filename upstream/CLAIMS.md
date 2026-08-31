<!-- SPDX-License-Identifier: Apache-2.0 -->
# Claims register

One row per thing this project believes it did that is not already available
upstream, with the evidence that supports it and how confident that evidence
makes us.

This file exists because of an asymmetry: the kernel source is stable and
version-controlled, but *the answer to "what exactly do we claim is new, and
which measurement supports each clause"* decays fast. Some decay has already
happened — the kernel's opening comment and parts of the package README describe
an older design. Reconstructing this from commit archaeology later is far more
expensive than maintaining it now, which is why it is a Phase 1 deliverable and
not a Phase 6 one.

**Confidence** is about the evidence, not the idea. `measured` means a number was
produced on the named hardware. `derived` means it follows from a measurement by
an argument that could be wrong. `asserted` means we believe it but have not
shown it. Reproduction on a second independent machine is meaningfully stronger
than one, and is noted where it applies.

**Status** tracks the upstream path only. Nothing here is ready to file.

## Upstream baseline as of 2026-08-30

Rewritten after the fetch-verified gap analysis
([`EVIDENCE_GAP_ANALYSIS_2026-08-29.md`](EVIDENCE_GAP_ANALYSIS_2026-08-29.md));
the previous version of this block was wrong in four places, each noted.

- FlashInfer **#3640** introduced the SM120 NVFP4 attention path; **#4502**
  (merged 2026-08-25) is a *performance* PR on top of it — the N64 score-slot
  reuse — at **D64/D128, non-paged**. (Correction: this block previously
  credited #4502 with introducing prequantized-GQA support.)
- **#4691** (open, unreviewed) is the nearest neighbour: an SM120 Sage-style
  **INT8-QK / FP8-PV port, K-centred, V-permuted** — occupying much of the
  technique space this project uses, dense/non-paged, and stalled partly on
  "does not compute LSE". **#4149** (open): MXFP8 ragged prefill SM120/121,
  34 days with zero human review. Both are cautionary process precedents as
  much as technical ones.
- **#3518** covers D256 (BF16-class); **#3485** FP8-KV gather with SM120-aware
  tile shrinking; **#4272** is the CUDA-graph test-pattern donor.
- cuDNN FROST: per-tensor E4M3 SM120 SDPA (**#509**), dtypes and **native head
  tiles to 256** (**#595**), FP8 split-KV on SM120 (**#768**) — per-tensor
  descales, no paged attention. (Correction: "D256 low-precision on SM120 does
  not exist" was too strong; #595's native 256 tiles exist. The *paged,
  fine-grained-scaled* combination still does not.)
- **CI correction, in our favour:** FlashInfer's NVIDIA-internal GitLab matrix
  includes **`unit_test_rtx_pro_6000`** (cu129/cu130), visibly running in bot
  tables on #4272/#4013/#4703; a maintainer triggers it with `/bot run`. Public
  GH Actions remains sm86/sm75/H100. (Correction: this block previously said no
  SM120 runner exists anywhere.) Consequence: tests written in FlashInfer's own
  harness get executed on our exact SKU by one maintainer comment — the
  highest-leverage form for our evidence.

**The precise gap this project occupies: paged page_size-1 + D256 + 24:4 GQA +
bottom-right-causal prefix/EXTEND + the transform pipeline (K/V mean-centering +
shared rotation, with per-row-Q / per-tile-K/V scales — K14: the transforms,
not the granularity, carry the quality) + an inclusive
gather/centre/rotate/quantize/pack path.** Every neighbour is dense
or ragged or block-sparse, per-tensor or per-block scaled, and none is paged.

## Kernel and dataflow claims

| # | Claim | Evidence | Confidence | Status |
|---|---|---|---|---|
| K1 | The `mma.m16n8k32.e4m3` A-fragment byte order on SM120 is: thread `(r=lane/4, c=lane%4)` holds `A[r][4c+b]` in reg0 byte `b`, `A[r+8]` in reg1, `+16` columns in regs 2/3. | On-device one-hot FP8 bytes against a bit-pattern B matrix. [Fragment-layout probe](../probes/fragment_layout/r2a_fragment_microtest.cu). | measured | draft |
| K2 | Given K1, a fixed 32-permutation of V's KV rows plus a transposed d-major tile-major FP8 store lets each thread's own S accumulators pack directly into its PV A-fragment registers — **zero cross-lane traffic**. Any bijection is exact because both sides agree. | Follows from K1; realised in the production kernel and in `quant.py`'s `SIGMA64`. **Independently corroborated:** cuDNN #509 needs `shfl.sync.idx` + `prmt.b32` for the same S→PV packing step that SIGMA64 makes free — evidence the shuffle-free property is real and non-obvious. | derived, corroborated | **lead claim for the PR** |
| K3 | Online softmax bounds `p <= 1`, so 448 is the *exact* per-row FP8 `amax` scale for P, and folds into the exponential for zero extra instructions. The per-tile V dequant ratio folds into the same constant. | Analytic, realised in the kernel. | derived | draft |
| K4 | Legacy E4M3 `mma.sync` already issues at full rate on SM120 + CUDA 12.9: legacy 934.4 TF/s, unity-scale block-scaled `kind::mxf8f6f4` 934.2 TF/s, INT8 934.5 TOP/s, FP4 `kind::mxf4` 1868.4 TF/s. **The widely-cited ~2x separation is the FP4-vs-FP8 ratio, not legacy-vs-block-scaled.** Unity-scale output is bit-identical to legacy (0/128 mismatch) on NaN-free operands. | [Instruction-rate probe](../probes/mxfp8/pv_mma_bench.cu), allocation 1, 2026-08-28. | measured | **strongest standalone contribution** |
| K5 | A bare `-arch=sm_120a` executable build silently lowers to `sm_120`, after which ptxas rejects the block-scaled MMA. `-gencode=arch=compute_120a,code=sm_120a` is required. | [Syntax probe](../probes/mxfp8/mx_syntax_probe.cu) on the RTX PRO 6000, 2026-08-28. **Independently reproduced 2026-08-29 on the RTX 4090 dev tier with a fresh nvcc 12.9.86 and no SM120 device present** (`tools/probe_target.py --compile-gate sm_120a --bare-arch`): ptxas fails with *"Instruction 'mma with block scale' not supported on .target 'sm_120'"*. Locked by [the toolchain test](../tests/test_toolchain_gate.py). | measured, reproduced on two machines | good short bug report or doc note |
| K6 | The conditional alpha-rescale skip is exact by construction and worth −7.8% kernel time in isolation. Stacking the fully-visible-tile score path on top **costs** ~1.2% once the dominant scalar body is gone. | Interleaved A/B, 3 rounds, median-of-medians; golden 72/72 with both on. | measured | draft |
| K7 | The production kernel sits at **63%** of the measured 934 TF/s issue roof (cuBLASLt FP8 GEMM roof is 73%), so the limiter is scalar work, dependencies, and movement — not tensor-pipe throughput. | K4 plus the DIRECT schedule-weighted measurement in K8 (previously derived from the 548–557 pre-skip figure). | measured | supporting |
| K8 | **Direct post-rescale deployment measurements** (2026-08-30, allocation 1, fully measured 14x16 = 224-call replay, full-size no-wrap pool, per-call distinct data): schedule-weighted core **587.1 TF/s (66.71 s)**; preprocessing **8.21 s**/schedule; inclusive **74.23 s (527.7 TF/s honest)**. Per-chunk 30-iter view agrees (585.4 / 8.19 / 526.4). Flat 584–587 TF/s across all depths — no deep-prefix cliff. | [Schedule record](../bench/results/120-20260830T003408Z-candidate-zero-schedule.json) (raw samples, environment, clocks). | measured | replaces every inferred figure |
| K10 | **Fruit-round measurements (2026-08-30, allocation 2).** Tier A preprocessing restructure (bit-exact, 72 goldens hold): **8.21 → 6.19 s** full-schedule, within 1.3% of the byte-count model's 6.11 prediction; honest inclusive rate 536.1 → **550.0 TF/s** on `card-B`. K1/K2 instruction diet (−608 SASS, bit-exact): **−0.18% weighted** over chunks {0,6,12} A/B (3 process pairs; +0.40% shallow, −0.2% deep) — the instruction-elasticity model overpredicted because the removed predicates were latency-hidden; kept as code-health. Methodological note: byte models of the DRAM-bound path predicted to ~1%; instruction models of latency-hidden scalar work did not. | [Preprocessing record](../bench/results/120-20260830T053422Z-fruit-newprep.json) and the adjacent `fruit-ab` JSON records in [the results directory](../bench/results/). | measured | informs the ladder |
| K9 | The **Hadamard rotation benefit is model-dependent** and must not be asserted generally. A published SageAttention2 ablation measured QuaRot-class rotation as no better than no smoothing on sampled real-model layers, while this project's counter-evidence is private and not published. Exactness under the shared orthonormal rotation is unconditional; this repository does not publish evidence for a general benefit claim. | SageAttention2 ablation; supporting private artifact not published. | asserted (benefit); derived (exactness) | narrow before PR |

| K11 | **No public engine runs FP8-class attention at D256 on SM120 at all.** cuDNN's FP8 AND block-scale MXFP8 SDPA engines are registered at `d_qk = 128` only — structural in the engine capability table (`frozenset({128})`, cudnn/sdpa/fwd/engines.py, frontend 1.27 / backend 9.20), not version-gated; the 16→256 head range belongs to the f16-class engines. Combined with FlashInfer's D64/D128 NVFP4 surface: our kernel has no same-precision competitor at its head dim, paged or dense. | `probes/cudnn_frost/probe1_pygraph_fp8.py` (refusal reproduced at every shape), engines.py table inspection, 2026-08-30. | measured | strengthens the gap statement |
| K12 | vs **NVIDIA's strongest D256-capable engine** (FROST DSL SM120 BF16 dense, `causal_bottom_right` verified against an explicit reference, pre-gathered inputs = conservative for us): FROST 269–281 TF/s at the production shapes → **2.213× [2.210, 2.216] schedule-weighted** (ABBA, 2 independent blocks, 14 paired cases each; per-block geomeans 2.18/2.20; `bench/results/FROST-CONTROL-20260830.md`). Note FROST-BF16-dense is slightly *slower* than FlashInfer's paged BF16 (~298 TF/s) here, so the 1.97× FlashInfer number remains the conservative headline. | `probes/cudnn_frost/probe3_dsl_sm120_bf16.py` + campaign JSONs. | measured | secondary PR-table row |

| K13 | **Per-tensor FP8-QK ported into NVIDIA's own FROST SM120 template runs correctly at D256 but does NOT beat their BF16.** v0 port (m16n8k32.e4m3 QK, descales folded into the softmax scale, V dequantized in-register, everything else donor-identical): numerics gate 3.2e-2 vs fp32 (the per-tensor error class), full 446k depth, **239–264 TF/s vs the same template's BF16 at 269–281** — the halved K bytes and doubled MMA depth bought nothing in their dataflow. Caveats: v0 uses correctness-first lds.32 B-loads and per-byte V dequant (ldmatrix headroom); a tuned FP8 path could differ. Read with K12: our 2.2× over their BF16 at the same precision class says **the win is the dataflow around the low-precision MMA — fused paged gather/quant, register-resident packing — not the dtype swap itself.** | `probes/cudnn_frost/prefill_fp8qk_sm120.py` + `probe4_fp8qk_port.py`, first-run 2026-08-30. | measured (v0-caveated) | PR-body argument |

| K14 | **Where the lab scheme's quality advantage actually comes from — and where per-tensor FP8 breaks.** Attention-output simulation across 25 (distribution × depth) cells to the full 446,335 tokens, oracle-checked to 2e-3 (`probes/quality/`): (a) per-tensor E4M3's failure at depth is a **worst-ROW phenomenon** — on heavy-tailed keys the worst-row cosine collapses 0.894 → 0.049 (worst-row error 136%) while the MEAN stays flat, vindicating the quality-gate plan's worst-slice reporting and adding worst-row to it; (b) **the advantage is the transforms, not tile granularity**: per-tile scales WITHOUT centering/rotation are indistinguishable from per-tensor on gaussian (4.63 vs 4.65%) and RoPE-structured keys (14.91 vs 14.96%) — K/V mean-centering plus the rotation buy the 4.2× (rope) and 500× (massive-V) gaps, so claims should say "transform pipeline with tile scales", never "fine-grained scales" alone; (c) P-scale policy is a non-differentiator (≤1.4% relative across all cells); (d) predicted port behaviour matched: the v0 port's 3.2e-2 sits in the instrument's per-tensor class, and a port reading ~3% on RoPE-like keys would indicate accidental centering (validity tripwire). Boundary (a) evidence only: synthetic, attention-output level, says nothing about tasks. | `probes/quality/RESULTS.md` + JSON (env, seeds, ablation, excluded-mechanism checks: the vs_max/16 floor never engaged, V-mean dilution depth-invariant). | measured (simulation) | reframes the quality argument |

## Framework bug claims (independently filable, no kernel dependency)

| # | Claim | Evidence | Confidence | Status |
|---|---|---|---|---|
| B1 | With DFlash speculative decoding, **any** FlashInfer-family `--prefill-attention-backend` corrupts the spec verify path through the spec CUDA graphs — accept_len collapses 3.45–3.88 → 2.15–2.82 and **wrong tokens are accepted**, violating losslessness. Occurs with zero custom code running. Root cause: the verify graph is captured in custom-mask mode with a 9-byte dummy mask that no replay refreshes, because the runtime always passes `custom_mask=None` and FlashInfer selects mask mode by buffer presence. Correct fix is the capture/replay mask invariant. | Three independent traces converge; the source traces are private and not published. The upstream report is not yet drafted; it is tracked as `sglang-dflash-verify-mask-capture/` in [`reports/README.md`](reports/README.md). | measured | **ready to draft; highest external value** |
| B3 | **cudnn-frontend pygraph silently ignores unknown SDPA kwargs, including `diagonal_alignment`** (generic kwarg capture): requesting BOTTOM_RIGHT on `sdpa`/`sdpa_fp8` at frontend 1.27 executes TOP_LEFT with no error — at Q<K that is a different operator (we measured 2580% output error and physically impossible flat-in-K timing before the numerics gate caught it). The DSL surface (`SdpaFwdDslSm120`, `causal_bottom_right=`) is correct. A silently-accepted mask parameter that changes which operator runs is a correctness footgun worth an upstream report. | `probes/cudnn_frost/probe2_pygraph_bf16_WRONG_MASK.py` (preserved verbatim, wrong results and all). | measured | filable vs NVIDIA/cudnn-frontend |
| B2 | The stock position-computation path in the engine's extend scheduling slices the anchor position table with no width check on session-fork extends, crashing fork-stack ensembles. | Ensembles validated 6/6 across seeds 7100/7101; the supporting application artifact is not published. | measured | ready to draft |

## Explicitly NOT claimed

Stating these is part of the contribution's credibility.

- **Not** a general D256 attention operator. The support surface is narrow and
  documented in `docs/OPERATOR_CONTRACT.md` §2.
- ~~The ~610 TF/s figure~~ **RESOLVED 2026-08-30 by direct measurement: 587.1
  TF/s** schedule-weighted core (K8). The inferred 610 was ~4% optimistic; never
  cite it again.
- ~~Preprocessing cost dispute~~ **RESOLVED 2026-08-30: 8.21 s** per full
  schedule, fully measured. The ~0.2–0.3 s claim was wrong by ~30x; 11% of
  inclusive attention time; a first-class optimization target.
- ~~Single-allocation performance~~ **RESOLVED 2026-08-30: second-allocation
  confirmation ran on distinct silicon** (`card-A` and `card-B`):
  node effect bounded at ~2.3%, goldens bit-exact on both dies. Quote absolute
  rates as **587–601 TF/s across allocations**, never a single-card point. Still
  one SKU and power envelope — other SM120 variants remain unqualified.
- **Not** statistically signed-off on quality. Existing private downstream task
  evidence is not published and cannot establish statistical sign-off. The
  ~0.17 mean input-logprob delta saturates at implementation-swap sensitivity on
  this NVFP4 model and **cannot rank backends** at all.
- **Not** original in technique. K centering, Hadamard rotation, and per-row Q
  scales are the SageAttention/FA3 family. The contribution is the SM120 D256
  paged realisation and the measurements, not the ideas.

## Evidence hygiene for anything that goes public

Private real-activation captures and restricted inputs may be used on trusted
infrastructure as an internal gate. They must never be committed here, attached
to public CI artifacts, or embedded in an upstream PR. Public evidence uses
deterministic synthetic or redistributable fixtures; result records cite fixture
hashes, never content.
