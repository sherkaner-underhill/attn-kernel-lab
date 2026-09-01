<!-- SPDX-License-Identifier: Apache-2.0 -->
# Public serving-lane results — 2026-09-01

First full run of the serving benchmark lane (`engines/sglang/bench/`) on
public models: the operator's INT8-QK / FP8-PV prefill kernel against stock
BF16 FlashInfer attention, paired arm-for-arm on the same selections, full
tier, one rented RTX PRO 6000 Blackwell machine (SM120).

## Headline

**The candidate kernel was generation-equivalent to full-precision BF16
attention on every item, while 1.19–1.29× faster end-to-end.** Across 386
paired items (MRCR 115 + InfiniteBench 78, on each of two models), every
response was **byte-identical** between arms (`response_sha256` equal on all
386 pairs), so every quality delta is exactly 0.0. The arms provably ran
different computations: recorded top-2 logit margins differ numerically on
**115/115** margin-instrumented items (max |Δmargin| 3.94), and the candidate
arms show the kernel's depth-growing speedup signature.

| Model | Bench | n | Ref quality | Cand quality | Δ | e2e speedup |
|---|---|---|---|---|---|---|
| Qwen3.8-27B (24:4, declared) | MRCR 32k / 131k / anchor | 40/30/45 | .995 / .995 / .894 | identical | 0.0 | 1.22× overall, **1.29× anchor** |
| Qwen3.8-27B | InfiniteBench kv / longbook | 30/48 | 1.000 / .938 | identical | 0.0 | 1.21× / 1.22× |
| Qwen3.5-9B (16:4, declared) | MRCR 32k / 131k / anchor | 40/30/45 | .957 / .943 / .811 | identical | 0.0 | 1.21× overall, **1.28× anchor** |
| Qwen3.5-9B | InfiniteBench kv / longbook | 30/48 | 1.000 / .813 | identical | 0.0 | 1.20× / 1.19× |

## Why equivalence, mechanically

Generation was greedy, and the models' decision margins are large: the 5th
percentile top-2 logit margin across instrumented tokens is ≈7.6 (a ~2000×
probability ratio). The kernel's numerical perturbation — visible in the
margin deltas — never approached flipping an argmax over hundreds of
thousands of generated tokens. Two readings, both stated:

- **As a serving claim, this is the strong form of parity**: on these BF16
  checkpoints, replacing prefill attention with the quantized kernel changes
  *nothing observable* about the served behaviour, at a 1.2–1.3× e2e saving.
- **As a benchmark caveat, margins this wide limit discriminative power**: a
  worse kernel could hide under them too. The evidence that separates THIS
  kernel from worse ones is upstream of serving — the fidelity ladder in
  `quality/` and `probes/quality/` (worst-row behaviour, transform
  attribution), where sensitivity is by construction. Prior private-model
  runs on a weight-quantized checkpoint (narrower margins) showed nonzero
  per-item deltas at statistical parity, consistent with this picture.

## Reference-arm observations (the baseline's own ceiling)

- 27B MRCR anchor: five accuracy collapses at extreme depth (scores
  0.03–0.27 at 290k–355k tokens) drag the anchor mean to 0.894; the other 40
  anchor items score 0.98+. The candidate reproduces each of these items
  byte-for-byte — the failures are the model's ceiling, not the kernel's.
- 9B trails 27B everywhere, as expected; kv_retrieval is 30/30 on both
  models and both arms.

## Posture (identical across arms except the backend)

Pinned SGLang source (`1cf2b8c54d`), BF16 weights, BF16 page_size-1 KV pool,
eager chunked prefill (32768), no CUDA graphs for prefill, **no speculative
decoding**, greedy decoding, thinking disabled on both arms, fresh server
boot per arm, cache flushed per scored item, unscored 16k warmup per arm.
Candidate arm adds only `--prefill-attention-backend fp8_prefill` (+
`expandable_segments`); the runner's posture guard verified the active
backend per arm, and `compare` re-verified it from the records.

**The served kernel is this repository's kernel**: the backend's
`quant.py` and `csrc/fp8_prefill_attn.cu` were byte-identical to
`src/attn_kernel_lab/` at commit `e638aab` —
sha256 `02c0d25bd22157764cc0de97a7f7771e72bb8107bdc808f87786248da3a988eb`
and `b9062dd066bc3a89...` respectively (full digests reproducible from the
tree).

Context: `--context-length 446335` with
`SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`. **No YaRN/rope override**:
these checkpoints use an mRoPE parameterization with `rope_theta 1e7` and
extend natively; the reference arm's anchor scores above are the empirical
statement of how well that holds, and both arms share whatever degradation
exists. Selections cap prompts at 440k tokens (one 458k-token InfiniteBench
book is excluded by the cap).

## Reproduction notes (environmental fixes a reproducer will need)

1. `FLASHINFER_DISABLE_VERSION_CHECK=1` if the flashinfer package trio is
   version-skewed (release python wheel vs dev cubin/jit-cache).
2. The KV pool, not the rope config, is the context bottleneck for the 27B
   in BF16 on 96 GB: the default mamba/GDN state cache allocated 14 GB for
   190 states; `--max-mamba-cache-size 8` frees it for a single-request
   workload (pool 254k → 458k tokens).
3. `--language-only`: these are vision-wrapper checkpoints, and the
   multimodal CUDA-IPC feature transport dies on `pidfd_getfd` under
   restrictive container seccomp profiles; text-only benches don't need it.

## Contents

Per model (`mrcr-*/`, `infbench-*/`): `*-summary.json` (paired bins/tasks,
bootstrap CIs, speed), `selection.json` (binned item ids + token counts),
`results/{reference,candidate}/` per-item records (scores, token counts,
latency, `response_sha256`, numeric top-2 margins) and `arm.json`.
`response_text` and margin token-strings are stripped: completions are
derivative of third-party dataset content; the digests bind the evidence and
the datasets are public and pinned (`openai/mrcr` @ `f4c69fae`,
`xinrongzhang2022/InfiniteBench` @ `90f03943`).

Boundary: serving-level evidence on two public checkpoints, one machine,
one allocation, greedy decoding. Not a claim about sampled decoding, other
architectures, or models with narrower margins.
