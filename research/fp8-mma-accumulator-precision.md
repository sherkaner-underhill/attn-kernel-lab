<!-- SPDX-License-Identifier: Apache-2.0 -->
# Measure the FP8 `mma.sync` accumulator per target: exact on SM120, 14-bit truncating on SM89

- **Kind:** study
- **Status:** explored
- **Author(s):** @sherkaner-underhill (drafted by Claude Fable 5.1; reproduced independently by Codex on the 4090)
- **Opened:** 2026-09-02 · **Last updated:** 2026-09-02
- **Numerics impact (best guess):** none
- **Would touch (if realized):** `docs/TARGETS.md` (qualification: attach the microprobe record for every target), promotion records; the kernel only through target-scoped build options.

Evidence status: all numbers are from uncommitted scratch probes and records; each is *(unrecorded
estimate)* until the microprobe and its records are committed under `probes/`.

## Idea

A ~60-line microprobe: one warp per test vector; lanes with `lane % 4 == 0` place four E4M3 bytes into A
rows `groupID`/`groupID+8` (k = 0..3) and into B rows k = 0..3 of column `groupID`, so every one of the 128
D elements receives the same Σ_k a_k·b_k plus its own C element, and 128 accumulator bit patterns are tested
per MMA. The observed D is fitted against explicit models `fp32(R_k(C) + P)` and `R_k(fp32(C + P))` for
k = 10..24 with named rounding policies (truncate, RNE, ties-away, floor). A chained variant measures
CUDA-core refills, small-contribution survival and same-sign accumulation.

## Findings *(unrecorded estimate)*

| Test | RTX 4090 (SM89) | RTX PRO 6000 Blackwell Server Edition (SM120) |
|---|---|---|
| Pass-through, zero product, 8,192 patterns × 5 exponents, both signs | D = C truncated toward zero to 14 significant bits, 100% fit | bit-exact 100% |
| Smallest product that still registers against C | 2^-13 · C | 2^-23 · C |
| C = 1 + j·2^-23 with a product | `fp32(R14_trunc(C) + P)`, 100% | `fp32(C + P)`, 100% |
| Output grid | on the 14-bit grid of the largest addend | fp32, truncated (not rounded) at the 24th bit; mean −1e-8 per MMA |
| 2,000-step CUDA-core refill chain | −4.40e-5 per step (theory 2^-14/(2 ln 2) = 4.40e-5) | 0 |
| 4,096 same-sign products into a growing accumulator | −5.4% | 0 |

Consequences on SM89: a CUDA-core write to the accumulator (a rescale, a rank-1 update) is truncated toward
zero at the next MMA; contributions smaller than 2^-13 of the accumulator vanish; the shipped kernel itself
carries an accumulation floor of 1.66% row-relative error at 446k keys (0.44% at 32k, scaling as √tiles),
which is exactly √(4.58² − 4.28²) between the kernel and its fp32 simulator. Remedies measured there: pre-round
the rescaled accumulator to 14 bits (works; +2.7–3.6%), fold (1 + 4.4027e-5) into the multiplier (cancels the
bias at zero cost but over-corrects on sharp fixtures), or compute each k32 chunk from a zero C operand and
add in fp32 (`zero-c-flush-accumulation.md`). On SM120 every one of these is unnecessary and the compensation
is actively harmful (16–18% error at terminal depth).

## Why it might not

PTX specifies an fp32 accumulator interface but neither the internal precision nor the rounding of the C
operand, so none of this is portable; a future target may differ either way. The 4090 is a development
tier, not a qualified target, and its behaviour must not leak into the operator's numerics as a
representation choice. That is the point of the note: accumulator remedies are target-scoped build options,
and the microprobe record is the qualification evidence for choosing them.

## Prior art

The DeepSeek-V3 technical report (§3.3.2) reports approximately 14-bit FP8 accumulation on H800 and promotes
partial sums to CUDA cores at a fixed interval. This probe measures the boundary bit-exactly and finds SM120
exact; that result was not found published. Not searched further.

## Cheapest decisive test

Run the microprobe on each candidate target: seconds, any CUDA with an FP8 `mma.sync`. Commit it under
`probes/` with one record per target, and reference the record from `docs/TARGETS.md`.

## Log

- 2026-09-01 — 4090 fused prototype lost 3× quality; bisection to the accumulator rescale; pre-round fix.
- 2026-09-02 — microprobe written and fitted (SM89), Codex reproduction, SM120 run: exact.
