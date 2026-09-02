<!-- SPDX-License-Identifier: Apache-2.0 -->
# Real-activation attention statistics via a recording attention function

- **Kind:** study
- **Status:** explored
- **Author(s):** @sherkaner-underhill (drafted by Claude Fable 5.1)
- **Opened:** 2026-09-02 · **Last updated:** 2026-09-02
- **Numerics impact (best guess):** none
- **Would touch (if realized):** `probes/` (a committed recorder and statistics probe); the fixture families in `probes/quality/pertensor_vs_finegrained.py` if a real-like family is derived from the statistics.

Evidence status: all numbers *(unrecorded estimate)*; captures are not committed (README rule 5) and the
statistics live in uncommitted scratch records.

## Idea

Register a custom attention function in transformers' attention interface that records post-RoPE Q (last T
rows), K and V for every softmax-attention layer during one long prefill, then delegates to SDPA. Offline, for
the last 128 query rows against all keys (this repository's EXTEND geometry), compute per (row, tile of 64
keys): effective token count, the mass carried by tokens within one E4M3 cell (0.118 nats) of a neighbour, the
mass within ln(448/432) of the tile maximum, the decoded-mass error of each P cast, the θ = 8 split-eligible
mass, and the closed-form representation errors of §`full-range-p-cast.md` with a per-tile-amax E4M3 V.
The captures also feed a kernel-level A/B through the production quantizer path.

## Findings *(unrecorded estimate)*

Qwen3.5-9B (head_dim 256, 16 query over 4 KV heads, eight softmax layers) and Qwen3-4B (head_dim 128,
36 layers), 32,768 tokens of the SGLang documentation and source:

- Regimes: the 9B's rows are ordinary-to-diffuse (effective token count median 190–1046; 36% of rows spread
  over more than 512 tokens); the 4B is sharper (median 5–355; 13% of rows sharp).
- Coherence is common (24–30% of a row's mass in tokens within one E4M3 cell of a neighbour; p99 60–95%),
  yet full-range P's row mass error exceeds 1% in 0.16% / 0.67% of rows and never 2.3%; the shipped cast's
  exceeds 1% in 8% / 25% of rows (its running-max reference pushes far-below-max tiles into subnormals).
- Full-range P is 0.78× the shipped cast in the simulator and 0.834× at kernel level through the production
  path (9B); the DC split's gate passes 3.6–6.1% of the mass and changes nothing; top-cell crowding carries
  0.2–0.8% of the mass.
- After full-range P the residual is V quantization: full-range P sits 6–8% above the exact-P floor.

## Why it might not

One text, two models, one depth, exact scores in the statistics (the kernel A/B used the INT8 path), a
simple V quantizer in the closed forms, no downstream model evaluation. Gemma-3 (head_dim 256) was not
available under the token used. A real-like synthetic family derived from these statistics would let the
committed fixture set cover the sharp/subnormal regime the gaussian packs under-represent.

## Prior art

Attention-sink and entropy statistics are widely reported (e.g. the StreamingLLM sink observation); the
decoded-mass and coherence statistics tied to an FP8 cast are specific to this operator. Not searched further.

## Cheapest decisive test

Commit the recorder and statistics probe (any CUDA with a 9B model: about ten seconds of prefill plus a
minute of statistics) and run it on the deployment's own prompts, which are the distribution that matters.

## Log

- 2026-09-02 — recorder written; Qwen3-4B and Qwen3.5-9B statistics; kernel-level A/B on the 9B captures.
