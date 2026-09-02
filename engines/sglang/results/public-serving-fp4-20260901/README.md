<!-- SPDX-License-Identifier: Apache-2.0 -->
# Weight-quantized (NVFP4) serving results — 2026-09-01

Second full run of the serving lane (`engines/sglang/bench/`), under the
bench README's "Weight-quantized (NVFP4) arm-pairs" protocol: the same paired
protocol, selections, and posture as the BF16 run of record
(`../public-serving-20260901/`), with one variable moved — both arms serve
public NVFP4 weight-quantized exports of the same checkpoints. One rented
RTX PRO 6000 Blackwell machine (SM120), full tier, greedy decoding, no
speculative decoding, fresh server per arm.

## Headline

**The narrow-margin regime the protocol predicted arrived, and the kernel
held statistical parity inside it.** 70 of 386 paired items now differ byte
for byte (BF16 run: 0 of 386); per-item deltas land in both directions;
**every paired bootstrap CI straddles zero**; and the kernel's depth-growing
speedup is intact and larger than in BF16 (**1.418x** at the 27B MRCR anchor
vs 1.291x).

| Model | Bench | n | Ref | Cand | Delta (CI95) | sha equal | e2e speedup |
|---|---|---|---|---|---|---|---|
| qwen3.8-27b-fp4 | MRCR 32k | 40 | .9561 | .9559 | -.0002 [-.0334, +.0331] | 32/40 | 1.008x |
| qwen3.8-27b-fp4 | MRCR 131k | 30 | .9313 | .9527 | +.0214 [-.0854, +.1288] | 26/30 | 1.080x |
| qwen3.8-27b-fp4 | MRCR anchor | 45 | .6178 | .6250 | +.0071 [-.0965, +.1070] | 35/45 | **1.418x** |
| qwen3.8-27b-fp4 | InfB kv_retrieval | 30 | 1.0000 | 1.0000 | 0.0000 | 7/30 | 1.318x |
| qwen3.8-27b-fp4 | InfB longbook | 48 | .8958 | .8958 | 0.0000 [-.0625, +.0625] | 46/48 | 1.327x |
| qwen3.5-9b-fp4 | MRCR 32k | 40 | .9804 | .9573 | -.0232 [-.0698, +.0004] | 36/40 | 0.998x |
| qwen3.5-9b-fp4 | MRCR 131k | 30 | .9947 | .9947 | 0.0000 | 30/30 | 1.060x |
| qwen3.5-9b-fp4 | MRCR anchor | 45 | .7784 | .6937 | -.0847 [-.2062, +.0332] | 30/45 | **1.358x** |
| qwen3.5-9b-fp4 | InfB kv_retrieval | 30 | 1.0000 | 1.0000 | 0.0000 | 26/30 | 1.299x |
| qwen3.5-9b-fp4 | InfB longbook | 48 | .8333 | .8333 | 0.0000 | 48/48 | 1.238x |

Paired bootstrap: 10,000 resamples, percentile method, seed 20260830, from
each `*-summary.json`.

## Flips concentrate where the baseline was indifferent

Flip = a paired item whose `response_sha256` differs between arms. Where the
first divergence is locatable in the margin records (MRCR; see posture note
on InfiniteBench margins):

| | flips | locatable | ref margin at first divergence min/median/max | median as percentile of the arm's margins | arm margin p50 |
|---|---|---|---|---|---|
| mrcr-27b-fp4 | 22 | 18 | 0.0 / **1.0** / 11.75 | **0.27th** | 12.5 |
| mrcr-9b-fp4 | 19 | 16 | 0.0 / **0.8125** / 1.75 | **0.05th** | 14.25 |

On the 9B every locatable flip sits at a reference margin <= 1.75 — the
bottom 0.13% of that arm's margin distribution. Flips are essentially
confined to positions where the baseline model was already nearly
indifferent, which is the mechanism the protocol predicted. (One 27B
exception is listed under caveats.)

**Per-step margin deltas, pre-divergence only.** Once an item flips, the two
arms generate different text and later positions compare unrelated
distributions; the BF16 run had zero flips, so only PRE-divergence positions
are comparable to it:

| | positions | median | p99 | max | % > 1.0 |
|---|---|---|---|---|---|
| BF16 run of record | 77,159 | 0.125 | 0.625 | 3.94 | 0.34% |
| **NVFP4, pre-divergence** | 68,676 | **0.5** | **4.0** | 8.5 | 24.3% |

The same kernel perturbs the NVFP4 checkpoints roughly **4x more per step**
than the BF16 ones, on models whose margins are simultaneously narrower —
the low tail moved most (27B reference p1: 7.0 in BF16 -> 3.875 here, with
genuinely zero-margin positions appearing that BF16 never produced). Both
effects push toward flips; the task scores above show what they amount to.

## The most instructive rows

- **kv_retrieval, 27B: 23 of 30 items differ byte for byte while all 30
  score 1.0 on both arms.** The buried key is recovered identically; only
  surrounding prose differs (the candidate is typically terser). Byte
  divergence without behavioural divergence.
- **longbook 27B's 0.0000 is a cancellation, not agreement**: one item
  regressed (correct -> wrong) and one improved (wrong -> correct). Stated so
  the zero is not read as byte-equivalence.

## Cross-run: what the weights cost (reference arm vs reference arm)

| Model | Line | BF16 ref | NVFP4 ref | change |
|---|---|---|---|---|
| 27B | MRCR 32k / 131k / anchor | .9953 / .9955 / .8936 | .9561 / .9313 / .6178 | -.039 / -.064 / **-.276** |
| 27B | InfB kv / longbook | 1.0000 / .9375 | 1.0000 / .8958 | 0 / -.042 |
| 9B | MRCR 32k / 131k / anchor | .9571 / .9947 / .8497 | .9804 / .9947 / .7784 | +.023 / 0 / **-.071** |
| 9B | InfB kv / longbook | 1.0000 / .8125 | 1.0000 / .8333 | 0 / +.021 |

NVFP4 weight quantization costs the 27B heavily at extreme depth (anchor
.894 -> .618: the BF16 model's occasional 290k-355k collapses become
frequent) and costs the 9B less; two 9B lines improve slightly, within
noise. This bill is paid identically by both arms — it attributes to the
weights, not to either attention path. Depth-resolved degradation of this
size is invisible to short-context evaluations; it is a finding about the
export recipe class, not only about this run.

## Speed (candidate vs reference, same items)

| Model | Line | e2e ref -> cand | speedup | BF16 run's speedup |
|---|---|---|---|---|
| 27B | MRCR 32k / 131k / anchor | 9.9->9.8 / 17.1->15.8 / 100.2->70.7 s | 1.008 / 1.080 / **1.418** | 1.004 / 1.05 / 1.291 |
| 27B | InfB kv / longbook | 33.3->25.2 / 9.9->7.4 s | 1.318 / 1.327 | 1.211 / 1.222 |
| 9B | MRCR 32k / 131k / anchor | 4.5->4.5 / 7.3->6.9 / 36.8->27.1 s | 0.998 / 1.060 / **1.358** | 0.987 / 1.034 / 1.276 |
| 9B | InfB kv / longbook | 11.2->8.6 / 3.8->3.0 s | 1.299 / 1.238 | 1.204 / 1.186 |

Every line beats its BF16 counterpart, as expected: FP4 weights shrink the
non-attention share of prefill, so the attention time the kernel saves is a
larger fraction of the total. The quantized-weights regime is where this
kernel matters most.

## Honest caveats

1. **The least comfortable number**: 9B MRCR 32k, delta -0.0232, CI95
   [-0.0698, **+0.0004**] — it straddles zero by four ten-thousandths. Four
   of forty items flipped and three of those dropped from ~0.99 to ~0.08;
   with a metric this heavy-tailed, n=40 cannot resolve a sub-0.05 effect.
   The 9B also leans negative overall (mean paired delta -0.041 vs the 27B's
   +0.008). Nothing here excludes zero; a longer tier on the 9B is the right
   next measurement before quoting "parity" for that model unqualified.
2. **One 27B flip started at a wide margin**: item `5b70a35684bf3494` (32k)
   first diverges where the reference margin was 11.75 — far above every
   other locatable flip point (all <= 9.375, median 1.0). A flip there means
   the candidate's logits at that position were not a small perturbation of
   the reference's. Recorded as an open anomaly; this lane cannot inspect
   logits directly.
3. **Flips by bin are depth-dominated but not monotone**: both models flip
   more at 32k than at 131k (9B: 4 vs 0; 27B: 8 vs 4). Unexplained.
4. **Cross-model comparisons carry a recipe caveat**: the two exports use
   the same numerical scheme (NVFP4, group 16, W4A4) but different
   serialization formats and differ by one quantized layer family. Within a
   pair both arms serve the identical file, so paired deltas are unaffected.

## Checkpoints (found public, config-inspected; no weights republished)

| Role | Export | Revision | Format |
|---|---|---|---|
| qwen3.8-27b-fp4 | `Inferact/Qwen3.8-27B-NVFP4` | `6128240ebaf4` | TensorRT Model Optimizer, `quant_algo: NVFP4`, group 16, `kv_cache_quant_algo: null` |
| qwen3.5-9b-fp4 | `ig1/Qwen3.5-9B-NVFP4` | `3b9e07b03283` | compressed-tensors `nvfp4-pack-quantized`, group 16, no KV scheme; ships its own recipe |

Both are tagged as quantizations of the exact base checkpoints, keep
`rope_parameters` byte-identical to the bases (no rope/YaRN posture change),
and quantize the attention and MLP projections while leaving `lm_head`,
embeddings, the vision tower, and (9B) the linear-attention family at source
precision. **Screening disqualified many popular exports because they bake
FP8 KV-cache scales into the checkpoint**, which would have violated this
lane's "KV stays BF16 at page size 1" rule and silently changed the
comparison. Serving-side confirmation that NVFP4 executed: load reports
`quant_algo=NVFP4` at the packed on-disk weight size (27B 24.21 GB), which a
silent BF16 dequantization could not produce. Full per-file sha256 manifests
for both checkpoints: `provenance/digests-*.txt`.

## Posture

Identical across arms except the backend, carried from the run of record:

```
python3 -m sglang.launch_server \
  --model-path <export> --served-model-name <qwen3.8-27b-fp4|qwen3.5-9b-fp4> \
  --quantization <modelopt_fp4|compressed-tensors> \
  --context-length 446335 --chunked-prefill-size 32768 --page-size 1 \
  --language-only --max-mamba-cache-size 8 --disable-prefill-cuda-graph \
  --mem-fraction-static 0.62
# candidate arm only: --prefill-attention-backend fp8_prefill
#                and: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# both arms: SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1, FLASHINFER_DISABLE_VERSION_CHECK=1
```

- **`--mem-fraction-static 0.62` is the one flag added relative to the run
  of record, and it is required by FP4 weights**: sglang's auto heuristic,
  seeing 24 GB of weights, sized the 27B KV pool at 328,611 tokens with
  ~48 GB idle — smaller than 18 of the 45 anchor prompts. Pinned identically
  on both arms and both models (capacity only, never numerics; pool 535,651
  tokens on the 27B). A partial arm computed under the wrong pool was
  discarded whole and re-run.
- The candidate arm's allocator setting changes free-memory accounting at
  pool sizing (540,963 vs 535,651 tokens); both pools exceed every prompt
  and the context length. The two arms' resolved server configurations were
  diffed field by field: **the only difference is the prefill attention
  backend.**
- **Selections are the run of record's, byte for byte** (sha256 equal to the
  published `selection.json` files), so items pair across regimes.
- MRCR records margins (runner default ON); InfiniteBench does not (runner
  default OFF, as in the run of record) — InfiniteBench flip positions come
  from response bytes, not margins.
- Served kernel: byte-identical to `src/attn_kernel_lab/` at repo commit
  `5fb9f2f` (`quant.py` sha256 `02c0d25bd221...`, `csrc` `b9062dd066bc...`),
  staged over the engine-integration copy exactly as in the run of record.
  The SGLang backend glue itself is not yet shipped in this repository;
  publishing it as a pinned patch set is planned so that this lane is
  reproducible from public material alone.

## Reproduction notes

1. No SGLang source patch was needed at pin `1cf2b8c54d`:
   `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` reduces the derived-context
   check to a warning.
2. `--disable-prefill-cuda-graph` is the pin's alias for
   `cuda-graph-backend-prefill=disabled`; verify in the resolved config.
3. After a detached checkout of the pin inside a prebuilt image,
   `sglang.__version__` may still report the image's build string; trust
   `git rev-parse HEAD` in the source tree.

## Contents

Per run (`mrcr-*/`, `infbench-*/`): `*-summary.json` (paired bins/tasks,
bootstrap CIs, speed), `selection.json`, `results/{reference,candidate}/`
per-item records (`score`, token counts, latency, `response_sha256`, numeric
top-2 margins where recorded) and `arm.json`. `response_text` and margin
token strings are stripped: completions are derivative of third-party
dataset content; digests bind the evidence and the datasets are public and
pinned (`openai/mrcr` @ `f4c69fae`, `xinrongzhang2022/InfiniteBench` @
`90f03943`). `provenance/` holds the checkpoint digest manifests.

Boundary: serving-level evidence on two public NVFP4 exports, one machine,
one allocation, greedy decoding, full tier. Statistical parity within wide
CIs is the claim; caveat 1 bounds it. Not a claim about sampled decoding,
other export recipes, or other architectures.
