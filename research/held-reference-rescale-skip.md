<!-- SPDX-License-Identifier: Apache-2.0 -->
# Held cast reference + discrete V tile scales, so full-range P's rescale can be skipped by warp vote

- **Kind:** modification
- **Status:** explored
- **Author(s):** @sherkaner-underhill (drafted by Claude Fable 5.1)
- **Opened:** 2026-09-02 · **Last updated:** 2026-09-02
- **Numerics impact (best guess):** semantics-changing
- **Would touch (if realized):** `full-range-p-cast.md`'s reference policy, operator contract §3.4 (the V tile-scale grid), §3.6; quantizer and kernel.

Evidence status: all numbers *(unrecorded estimate)* from uncommitted scratch records.

## Idea

Full-range P's +8.5% on SM120 is its unconditional accumulator rescale. The shipped kernel skips its rescale
whenever a warp-uniform vote finds every multiplier exactly 1. Full-range P's multiplier is
`exp(R_{t−1} − R_t) · vs_{t−1}/vs_t`; it is exactly 1 only if the cast reference is held across tiles *and*
the V tile scale is unchanged. So: refresh the reference only when a tile maximum exceeds it (mandatory, no
saturation) or drifts more than X nats below it, and put V tile scales on a discrete grid so consecutive
tiles usually share one.

## Why it might work

Holding the reference costs almost nothing in quality with production V scales: 1.005× (X = 0.5) to 1.010×
(X = 1) of full-range P on the packs. With power-of-two scales the ratio is constant on 83% of tiles and the
warp skips 20–50% of rescales at X = 1–8 in the diffuse and ordinary regimes *(unrecorded estimate)*.

## Why it might not

Two measured failures. With production V scales the tile scale changes on 99.7% of tiles, so the multiplier
is never 1 and nothing is skipped. With power-of-two scales (rounded up, at most 2×) the half-mass
near-zero-V sink cells get 2.2–3.9× worse: the sink token's centred codes are tiny (0.006–3.5 in code units),
a 1.76× coarser scale pushes more of them into E4M3's subnormal range, its decoded error goes from 3.9% to
11.8%, and it carries half the row's mass; exact P shows the same loss, so it is a V effect *(unrecorded
estimate)* — the same subnormal mechanism that rules out FP4 V. The vote also fires whenever any of a warp's
16 rows refreshes, and in the sharp regime it fires on nearly every tile at every X.

## Prior art

Lazy rescaling by threshold is the standard trick of FlashAttention-style online softmax; the coupling to a
discrete V-scale grid is specific to carrying the V-scale ratio in the accumulator. Not searched further.

## Cheapest decisive test

The untested remaining variant: a quarter-octave V-scale grid (2^{k/4}, at most 19% headroom instead of 2×)
with the held reference — first the sink cells in the simulator (any CUDA, minutes), then one kernel flag and
a balanced timing run on SM120. Plausible outcome: about half the +8.5% at ~1% quality; or nothing.

## Log

- 2026-09-02 — simulator sweep (X = 0.5–8, production and pow2 scales) and the sink diagnosis; negative as
  designed, quarter-octave grid left open.
