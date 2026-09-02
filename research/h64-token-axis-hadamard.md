<!-- SPDX-License-Identifier: Apache-2.0 -->
# H64: a 64-token Hadamard on the token axis of P and V (retired)

- **Kind:** new-operator
- **Status:** retired
- **Author(s):** @sherkaner-underhill (proposed in the 2026-09-01 transform research; GPU test by Codex)
- **Opened:** 2026-09-02 · **Last updated:** 2026-09-02
- **Numerics impact (best guess):** semantics-changing
- **Would touch (if realized):** nothing now; it would have touched §3.5 and §3.6.

Evidence status: numbers *(unrecorded estimate)* from uncommitted scratch records.

## Idea

Apply a 64-point Hadamard along the token axis of each tile to both P and V before the casts (exact in
composition, since the PV product is an inner product over tokens), spreading a flat tile's mass across
coefficients so that E4M3 sees a better-conditioned operand.

## Why it was retired

It is a diffuse-regime specialist: 0.35× the shipped cast on flat tiles, neutral or worse elsewhere, and
catastrophic when an attention sink lies inside the rotated tile (45–52× worse), because the sink's mass is
smeared across all 64 coefficients and cancels against its near-zero V *(unrecorded estimate)*. The gated DC
split achieves the diffuse-regime gain without the sink hazard, and real activations rarely contain flat
tiles at all. Retired 2026-09-02; kept as the tombstone for "rotate the token axis".

## Prior art

Incoherent processing (Hadamard on the head dimension) in FlashAttention-3 and QuaRot rotates channels, not
tokens; the token-axis variant was not found in the literature. Not searched further.

## Cheapest decisive test

None needed; the sink fixture is decisive.

## Log

- 2026-09-01 — proposed and simulated; Codex GPU test on the production pack.
- 2026-09-02 — retired.
