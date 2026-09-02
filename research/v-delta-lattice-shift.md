<!-- SPDX-License-Identifier: Apache-2.0 -->
# V-delta: shift V before the cast and add the shift back (retired: it was the lattice phase)

- **Kind:** modification
- **Status:** retired
- **Author(s):** @sherkaner-underhill (2026-09-01 transform research; mechanism by Claude Fable 5.1; controls by Codex)
- **Opened:** 2026-09-02 · **Last updated:** 2026-09-02
- **Numerics impact (best guess):** semantics-changing
- **Would touch (if realized):** nothing now; it would have touched §3.4.

Evidence status: numbers *(unrecorded estimate)* from uncommitted scratch records.

## Idea

Subtract a per-tile or per-block offset from V before the E4M3 cast and add it back exactly in the epilogue,
on the hypothesis that a better-placed zero improves the cast.

## Why it was retired

The measured gains on gaussian V were an artifact of the BF16 input lattice: BF16 values plus a sub-lattice
fp32 shift and round-to-nearest-even E4M3 (whose ties are BF16-representable) produce a coherent per-channel
rounding bias, and the shift merely changed the phase. The gain is null with fp32 inputs, is reproduced by a
random per-tile lattice offset with exact add-back, and is captured directly by the decoded-mean correction
(`v-error-mean-correction.md`), which itself turned out to have no effect on real activations *(unrecorded
estimate)*. Retired 2026-09-02 as a general lead; a block-affine variant on structured V was never shown to
beat the direct correction.

## Prior art

Affine (zero-point) quantization; the lattice-phase explanation is the specific finding.

## Cheapest decisive test

None needed; the fp32-input control is decisive.

## Log

- 2026-09-01 — gains observed, mechanism found, controls run (Codex).
- 2026-09-02 — retired.
