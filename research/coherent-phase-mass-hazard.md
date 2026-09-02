<!-- SPDX-License-Identifier: Apache-2.0 -->
# Coherent-phase tiles: a mass-error hazard for any FP8 P cast, and the fixture that exposes it

- **Kind:** study
- **Status:** explored
- **Author(s):** @sherkaner-underhill (fixture by Codex; anatomy by Claude Fable 5.1)
- **Opened:** 2026-09-02 · **Last updated:** 2026-09-02
- **Numerics impact (best guess):** none
- **Would touch (if realized):** the fixture set in `probes/quality/` (a mass-structured fixture and a decoded-mass diagnostic), the evaluation protocol.

Evidence status: all numbers *(unrecorded estimate)* from uncommitted scratch records.

## Idea

A tile whose tokens share one logit (Codex's fixture: 63 tokens at local logit 0, one token with an exact
within-tile share ρ, the tile with an exact share τ of the row's mass) rounds all 63 values identically, so
the tile's decoded mass is off by up to one E4M3 half-step (about 6%) with a sign fixed by the lattice
phase. The numerator uses rounded P while the online-softmax denominator keeps the exact pre-cast mass, so
the output is mis-scaled by that amount times τ.

## Findings *(unrecorded estimate)*

- On the fixture, full-range P is worse than the shipped cast in 74 of 168 cells (worst 3.2×). A 64-point
  sweep of ρ splits 32/64 and 31/64: a coin flip, because the shipped cast's phase is random per row while
  full-range P's is fixed by ρ; both pay the same worst case.
- Jitter of 0.1 nats (one E4M3 cell) on the 63 logits removes the effect; the gated DC split tracks the exact
  result throughout because its tile mass is exact.
- A decoded-P denominator (normalising by the sum of the decoded codes, cheap in-kernel) repairs it fully
  when the mis-rounded mass is common-mode (an ordinary V on the selected token: one cell in 168 worse, at
  1.001×) and only halves it next to a near-zero-V sink; it adds tail risk on sink fixtures for the shipped
  cast, so it is a diagnostic, not the normalisation.
- On real activations the row mass error under full-range P never exceeded 2.3% (see
  `real-activation-attention-statistics.md`).

## Why it might not (matter)

Continuous scores randomise the per-tile phase; RoPE breaks up duplicated keys. The hazard needs many tokens
within ~0.05 nats carrying a material share of a row, which the two models measured did not produce beyond
2.3%. It stays as a correctness fixture rather than a design driver.

## Prior art

Not searched.

## Cheapest decisive test

Commit the fixture (any CUDA, seconds per cell) with the decoded-mass diagnostic, and run the recorder on the
deployment's prompts to count rows whose full-range decoded-mass error exceeds 1%.

## Log

- 2026-09-02 — fixture (Codex), reproduction and anatomy (Claude), real-activation frequency measured.
