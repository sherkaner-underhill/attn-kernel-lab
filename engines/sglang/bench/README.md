<!-- SPDX-License-Identifier: Apache-2.0 -->
# The serving-quality lane — paired long-context benchmarks

This directory holds the runners that answer one question, and only one:

> With the same model, the same items and the same instrument, does swapping
> the engine's attention backend from stock BF16 to the candidate kernel change
> what the server *answers*, and how long it takes to answer it?

Every design decision below exists to keep that question answerable. The
comparison is **paired** — both arms score the identical item set — because
per-item difficulty dominates the raw per-arm variance on long-context tasks,
and subtracting it is the only way a 30-item run says anything at all.

## What this lane is NOT

These scores live at the **serving boundary**. They are end-to-end facts about
a whole system: model weights, chat template, sampler, scheduler, prefix cache,
and the attention kernel. A per-task score cannot isolate the kernel, and this
lane never claims it does.

Kernel-level fidelity evidence — attention output versus BF16 per case, per
head, per row, with same-input controls — lives in `quality/`, and the
numerical and hardware probes behind it live in `probes/`. Read those first if
the question is "is the kernel correct". Read this lane if the question is
"does the kernel change the product".

A null result here (no measurable quality delta, a real speedup) is the
expected and useful outcome. It is an *upper bound on observable harm at the
serving boundary*, not a proof of numerical equivalence.

## The two instruments

| Runner | Task | Score | Why it is here |
|---|---|---|---|
| `mrcr_run.py` | `openai/mrcr` — reproduce one earlier assistant turn from a long synthetic conversation | continuous `SequenceMatcher` ratio | Long-context retrieval with a *continuous* score, so a paired difference is informative at modest N |
| `infbench_run.py` | InfiniteBench `kv_retrieval` | binary containment of the answer string | Every item is its own context: no sharing, full cold prefill each time |
| `infbench_run.py` | InfiniteBench `longbook_choice_eng` | binary, letter match | Book-length contexts with several questions each, so a shared prefix can be amortised |

Both runners share `serving_common.py`, which holds the endpoint client, the
posture guard, the warmup, the chat-template length measurement, and the
bootstrap.

## The isolation discipline, and why each rule exists

Each of these was paid for once. Do not remove one without replacing the
evidence it buys.

| Rule | Why |
|---|---|
| **Fresh server boot per arm; never reuse one server for both** | A prefix cache that survives an arm switch would serve the second arm answers the first arm computed, and nothing downstream can detect it. |
| **Posture guard: read the active prefill attention backend from the server and refuse on mismatch** | The most expensive possible failure is two arms that were secretly served by the same backend; the run must refuse before the first item, not be discovered afterwards. |
| **Speculative decoding refused outright** | It changes which tokens come from which code path, so a per-item delta would no longer isolate the attention kernel. |
| **`compare` refuses when both arms recorded the same backend** | Second line of the same defence, enforced from the records rather than from the operator's memory. |
| **One UNSCORED warmup request (~16k tokens), then a flush** | Absorbs allocator growth and any first-call kernel compile, so item 1 measures steady state instead of billing boot cost to the candidate arm. |
| **`/flush_cache` before every scored item** | No prompt content and no radix prefix from a previous item may survive into a measurement. |
| **…except between questions of the same book** | Within a book the prompt prefix is bit-identical, so radix reuse returns exactly the KV the cold path would have computed. It is exact, it is applied identically on both arms, and it buys ~4x the questions per hour. |
| **Length bins measured with the SERVED model's chat template** | The quantity a kernel's behaviour depends on is sequence length *at attention*. Character counts and other models' tokenizers are both wrong, quietly. |
| **The chat-template encoding is normalised to `input_ids` before `len()`** | Some tokenizers return a `BatchEncoding` or dict here; `len()` of that is the number of *keys*. Unnormalised, every bin silently selects zero items. |
| **Greedy decoding (`temperature: 0.0`) on both arms** | Sampling entropy would be measured as kernel noise. |
| **Thinking mode disabled identically on both arms** | On a thinking model, a short answer spends the whole generation budget inside the reasoning block and the graded content comes back empty — on *both* arms, zeroing the instrument rather than measuring it. Required for any thinking model; a no-op for templates without the key. |
| **Format gate on the reference arm** | If the baseline cannot follow the instruction on the easiest items, the run is measuring the model's instruction-following, not the kernel. It aborts loudly (exit 3) instead of scoring garbage. |
| **`response_sha256` on every item** | Two runs can score identically while differing token by token. The digest is what binds a published score to the exact bytes that produced it. |
| **Top-2 logprob margins recorded where the server reports them** | A near-zero margin is a token the model was almost indifferent about — exactly where a numerically different kernel flips an output. Free diagnostic context for any delta that does appear. |
| **End-to-end latency recorded per item, next to the score** | "Faster" and "no worse" are two different claims. Producing them from one paired run is what stops either from being quoted without the other. |
| **Dataset revision pinned and recorded** | MRCR's 2025-12-05 bugfix re-uploaded rows with corrected ground truth. A cached item is only trustworthy against fixed dataset bytes. |
| **Per-item result cache keyed by (revision, item, arm)** | A run interrupted at hour two resumes instead of restarting, and a tier upgrade reuses every item already paid for. |

## Public models for this lane

The runners are model-agnostic; the lane's REFERENCE models (survey of ~190
public configs, 2026-09-01) are the two Apache-2.0 checkpoints that land on
the operator's declared surface:

| Role | Model | Geometry | Why |
|---|---|---|---|
| Primary | `Qwen/Qwen3.8-27B` | D256, 24:4, 262k native | The declared production ratio; the committed kernel evidence (24:4 schedule, 14x16 call structure) transfers unchanged. ~79 GiB weights+KV at 446,335 tokens on a 96 GiB card. |
| Control | `Qwen/Qwen3.5-9B` | D256, 16:4, 262k native | A second declared ratio with ~63 GiB of headroom, so a memory-pressure confound on the primary can be ruled out by construction. |

Two protocol rules that come with them:

- **Depths beyond 262,144 need YaRN at `factor: 2.0`** (not the model card's
  default 4.0). Scaling is static in current engines, so the factor must be
  set identically on BOTH arms of a comparison and held constant across every
  depth in a sweep; native depths need no scaling and should be run without.
- **Measure the reference arm's retrieval ceiling per depth before reading
  any candidate number.** At extended depths the model's own ceiling, not the
  attention kernel, may be the binding constraint; a kernel comparison is
  meaningful only where the reference arm still scores.

## Running both arms

Nothing is baked in. The model is a **required** flag, the endpoint and token
come from flags or the environment, and output lands in `runs/` beside these
scripts unless `--out-dir` says otherwise.

```bash
export SGLANG_BASE_URL=http://127.0.0.1:8000   # default; override for a remote server
export SGLANG_API_KEY=...                      # only if the server requires one
MODEL=<the name your server answers to>
```

The bearer token is read from the environment and used only to build an
`Authorization` header. It is never logged, echoed, or written into a record.

**1. Select the items once** (no server needed; it only needs the tokenizer):

```bash
python3 mrcr_run.py prepare --model "$MODEL"
python3 infbench_run.py prepare --model "$MODEL"
```

`prepare` always selects the **full** item set and records the tier table with
it. The shorter tiers are ordered prefixes of that same selection, so a
30-minute run can be extended to a full one later without discarding a single
item already paid for. Add `--tokenizer` if the tokenizer source differs from
the served model name, and `--trust-remote-code` if its chat template needs it.

**2. Boot the server on the stock BF16 backend, then run the reference arm:**

```bash
python3 mrcr_run.py arm \
    --model "$MODEL" --tier 60min \
    --arm bf16 --reference \
    --expect-backend flashinfer --reject-backend fp8_prefill
```

**3. Shut that server down. Boot a new one on the candidate backend, then run
the candidate arm:**

```bash
python3 mrcr_run.py arm \
    --model "$MODEL" --tier 60min \
    --arm fp8-prefill \
    --expect-backend fp8_prefill
```

`--expect-backend` is a substring the server's reported active prefill
attention backend must contain; `--reject-backend` is repeatable and refuses
when the substring *is* present. Between the two, an arm cannot be run against
a server that is not in the state it claims to measure.

**4. Compare:**

```bash
python3 mrcr_run.py compare --tier 60min \
    --reference-arm bf16 --candidate-arm fp8-prefill
python3 infbench_run.py compare --tier 60min \
    --reference-arm bf16 --candidate-arm fp8-prefill
```

`compare` prints one line per bin or task and writes the full summary JSON
beside the selection:

```
32k: n=10 ref=0.8947 cand=0.8107 delta=-0.0840 CI95=[-0.2099,+0.0273] | e2e ref 12.3s cand 12.2s speedup 1.013x
```

The interval is a **percentile bootstrap over the paired per-item differences**
(10,000 resamples, fixed seed, resampling items). It answers "would another
draw of items from this pool show the same sign?" — it is not a significance
test, and a CI straddling zero at N=10 means the tier was too short to decide,
not that the arms are equal.

## Time-budget tiers

`--tier` selects how many items per bin an arm runs. `full` reproduces the
counts of the reference run recorded in the private lane; the shorter tiers are
sized against per-item planning estimates measured on a single high-end
workstation-class accelerator, at the stock BF16 prefill rate (the pessimistic
arm). Budgets admit to 95% of the nominal wall clock.

Per-item planning estimates used for the arithmetic:

| Line | Estimated seconds per item |
|---|---|
| MRCR 32k bin | 31.08 |
| MRCR 131k bin | 66.19 |
| MRCR anchor bin (262k–524k) | 178.78 |
| InfiniteBench `kv_retrieval` | 54.09 |
| InfiniteBench `longbook_choice_eng`, per book (~3.95 questions) | 96.93 |

### MRCR, 2-needle (the default config)

| Tier | 32k | 131k | anchor | Arithmetic | Wall clock per arm |
|---|---|---|---|---|---|
| `30min` | 6 | 5 | 6 | 186.5 + 331.0 + 1072.7 | 1,590 s ≈ 26.5 min |
| `60min` | 10 | 8 | 14 | 310.8 + 529.5 + 2502.9 | 3,343 s ≈ 55.7 min |
| `full` | 40 | 30 | 45 | 1243.2 + 1985.7 + 8045.1 | 11,274 s ≈ 3.13 h |

### MRCR, 4-needle and 8-needle (optional; anchor bin only)

| Tier | anchor | Wall clock per arm |
|---|---|---|
| `30min` | 8 | 1,430 s ≈ 23.8 min |
| `60min` | 12 | 2,145 s ≈ 35.8 min |
| `full` | 12 | 2,145 s ≈ 35.8 min |

The 4-needle lane's full count already fits inside a 60-minute arm, so those
two tiers coincide.

### InfiniteBench

| Tier | `kv_retrieval` | books (`longbook_choice_eng`) | Wall clock per arm |
|---|---|---|---|
| `30min` | 18 | 7 (~28 questions) | 1,652 s ≈ 27.5 min |
| `60min` | 30 | 12 (~48 questions) | 2,786 s ≈ 46.4 min |
| `full` | 30 | 12 (~48 questions) | 2,786 s ≈ 46.4 min |

The whole InfiniteBench lane fits inside a 60-minute arm, so `60min` and `full`
coincide. The headroom is deliberate: it is what lets the tier absorb a slower
server without overrunning.

Two honest caveats on the tiers. First, these are **fixed counts against
estimates**, not a greedy wall-clock admission loop — a server slower than the
estimates will overrun, and the fix is to drop a tier, not to trust the label.
Second, within a bin the shorter tiers take the *shortest* items, so a short
tier skews to the lower edge of each bin. That skew is identical on both arms
and therefore cannot move the paired difference, but it does mean a `30min`
anchor result is not interchangeable with a `full` anchor result.

## What a run writes

```
runs/<suite>/
  selection.json                 the item set, the pinned revision, the tier table
  results/<arm>/arm.json         backend, tier, model, server args, template kwargs
  results/<arm>/<item>.json      score, e2e seconds, token counts, response_sha256,
                                 response text, top-2 logprob margins, backend
  <suite>-summary.json           the paired comparison
```

Per-item records are run output, not repository content: they contain model
completions over third-party datasets and are not committed here.

Summary field names are generic (`reference_mean`, `candidate_mean`,
`delta_mean`, `delta_ci95`, `speed.speedup_e2e`) because the arms are named by
the operator. The structure and the statistics are otherwise those of the
private reference run, so summaries from the two are directly comparable field
for field.

## Datasets

Neither dataset is redistributed here; both are fetched from the Hugging Face
Hub at a recorded revision.

| Dataset | Repo | Revision handling | License |
|---|---|---|---|
| MRCR | `openai/mrcr` | pinned by default to `f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d`, overridable with `--revision` | MIT |
| InfiniteBench | `xinrongzhang2022/InfiniteBench` | resolved at `prepare` time and **recorded** in the selection; pin with `--revision` | see the dataset card; the upstream code repository is MIT |

The reference run used InfiniteBench revision
`90f0394333616266d9fe85824ceaf505093cbaa5`. Pin it with `--revision` to compare
against those numbers.

## Known deviations from the standard harnesses

Stated rather than hidden, because each is a place where a number here is not
directly comparable to a published leaderboard figure. All are applied
identically to both arms, so none of them affects the paired difference.

1. **MRCR grading strips leading whitespace** before the dataset card's
   random-string prefix rule. Chat-tuned models routinely open with a newline
   pair; strict application would zero the instrument on both arms.
2. **`longbook_choice_eng` is answered generatively** (the model emits the
   option letter) rather than by loglikelihood ranking, for harness simplicity.
3. **`longbook_choice_eng` shares a book's prefix across its questions.** The
   standard harness treats each question as an independent item.
4. **Items are selected by length bin**, not sampled from the full set, because
   the length ladder is the independent variable this lane cares about.
5. **The bootstrap is a plain percentile bootstrap** over paired differences,
   not BCa, and there is no McNemar test for the binary tasks. Reported as-is.
