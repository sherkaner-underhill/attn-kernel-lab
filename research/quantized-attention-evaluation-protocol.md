<!-- SPDX-License-Identifier: Apache-2.0 -->
# Evaluation protocol for quantized-attention candidates: seeds, tails, balanced timing, thermal record

- **Kind:** other
- **Status:** explored
- **Author(s):** @sherkaner-underhill (Codex proposal, Claude Fable 5.1 additions)
- **Opened:** 2026-09-02 · **Last updated:** 2026-09-02
- **Numerics impact (best guess):** none
- **Would touch (if realized):** `probes/` conventions, `docs/ARTIFACT_LIFECYCLE.md` (what a quality or timing record must contain).

## Idea

What a quality or timing record of a candidate representation should report before it can be compared with
the shipped kernel:

1. The seed, not the row, is the independent unit: rows within a seed share the K/V pack and quantizer state.
   Report per seed the mean, p99, p99.9, maximum row-relative error and worst-row cosine; across seeds the
   median, p90, p95, maximum and the paired win count; pooled row quantiles are descriptive only, and any
   uncertainty is a cluster bootstrap over seeds.
2. Paired fraction of rows worsened, overall and within the baseline's worst 1%.
3. Fixture families × depths × score regimes (diffuse / ordinary / sharp by effective token count) × sink
   cases × mass-structured cases, plus NaN/Inf counts and norm-ratio drift.
4. Kernel timing in balanced Latin-square blocks (every variant in every ordinal position equally often), raw
   CUDA-event samples retained, paired slowdowns with a bootstrap interval, at the shipped query geometry
   and at saturation.
5. The GPU thermal record for every run (temperature, power, throttle flags), because a candidate measured
   while the card throttles is not measured.
6. For any change that touches the accumulator: the target's FP8 MMA microprobe record
   (`fp8-mma-accumulator-precision.md`).
7. Before a claim about deployment: the same statistics on real activations of the deployment's own prompts
   (`real-activation-attention-statistics.md`).

## Why it might not

It is heavier than a single mean-error number and will be skipped when it is inconvenient. The defence is
to make the probe harness emit it by default.

## Prior art

Ordinary experimental practice; nothing novel claimed.

## Cheapest decisive test

Adopt it in the next committed probe; the cost is a few hundred lines of harness once.

## Log

- 2026-09-02 — written down after the round-2/3 reviews found fixed-order timing bias, pooled-row
  overconfidence and an unmeasured thermal state in earlier scratch results.
