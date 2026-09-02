<!-- SPDX-License-Identifier: Apache-2.0 -->
# Full-range P: cast each (row, tile) with its own maximum at 448 and convert the accumulator instead

- **Kind:** new-operator
- **Status:** explored
- **Author(s):** @sherkaner-underhill (drafted by Claude Fable 5.1 from the 2026-09-01/02 sessions, reviewed by Codex)
- **Opened:** 2026-09-02 · **Last updated:** 2026-09-02
- **Numerics impact (best guess):** semantics-changing
- **Would touch (if realized):** operator contract §3.6 (the `448·r_t` fold, the running-max reference, the conditional alpha rescale), §3.4 (what `vlog2r`/`vinvr` mean to the kernel), §4 (Oracle A: online-softmax rescaling semantics), `operator_contract_version`; the kernel's softmax loop, PV loop and epilogue; every golden.

Evidence status: every number below comes from uncommitted scratch probes and records in a development
worktree, not from `probes/`; per README rule 1 each is *(unrecorded estimate)* until the probe and its
record are committed.

## Idea

The shipped kernel casts P relative to the running row maximum `M_t`, with the per-tile V-scale ratio folded
in: `E4M3(exp(s − M_t) · 448 · r_t)`. Two consequences: a tile whose maximum sits well below the running
maximum uses only the low part of E4M3 (subnormals beyond about ten nats), and `448 · r_t` is not exactly
representable, so even the tile's maximum is rounded.

Full-range P casts each (row, tile) relative to its own maximum, `E4M3(448 · exp(s − R_t))` with
`R_t = max(m_t, M_t − K)`, K = 16, and takes `r_t` out of the cast. Every (row, tile) maximum is then exactly
448 and all mantissas are aligned to it. The running accumulator is converted between tiles by
`β_t = exp(R_{t−1} − R_t) · vs_{t−1} / vs_t` before the tile's PV MMA; the l-sum takes `exp(R_t − M_t)`; the
epilogue is `O = A · exp(R_last − M) · r_last · vs_max / l + vmean`. Algebraically identical to the shipped
operator before the casts.

## Why it might work

- The gain is exact representability of each maximum plus mantissa alignment. It is scale-free (damping the
  cast maximum to 56 or to 7 changes nothing) and it is lost by power-of-two lazy references, so it is not a
  range effect *(unrecorded estimate)*.
- Production-pack simulator at terminal depth (446k keys): 0.885–0.896× the shipped cast's row-relative error.
  Fused build on the SM120 qualified target: 0.924× (gaussian), 0.918× (v_outlier), 0.900× (heavy_t3),
  0.908× at 32k, the kernel equal to its own simulator to 1.7e-3 *(unrecorded estimate)*.
- Real activations (Qwen3.5-9B, head_dim 256, 32k tokens of the SGLang documentation and source, last 128
  rows, through the production INT8-QK and quantizer path on SM120): 0.834× the shipped kernel (0.79–0.90 per
  layer), p99 3.11% → 2.20%, max row 4.38% → 3.03%, and a −0.3% output-norm loss on early layers repaired.
  The shipped cast's real weakness is the running-max reference: on sharp rows most tiles' values fall into
  E4M3 subnormals, and 8% (9B) to 25% (Qwen3-4B) of rows carry a decoded-mass error above 1%, against
  0.16–0.67% under full-range P *(unrecorded estimate)*.

## Why it might not

- **Price.** The accumulator conversion is an unconditional rescale (128 FMUL per thread per tile) where the
  shipped kernel skips its rescale by warp vote. On an RTX 4090 that costs +1.9–2.0%; on the SM120 target it
  costs +8.3–8.6%, because the faster tensor core exposes CUDA-core work *(unrecorded estimate)*. The
  obvious remedy, holding the reference so the vote can skip, fails on sink precision
  (`held-reference-rescale-skip.md`). Whether −17% error is worth +8.5% time is an owner decision.
- **Clamp.** Without K the closed form overflows (NaN on heavy-tailed scores). With K = 16 the accumulator
  stays below ~1.7e9 on ordinary rows and ~1.6e14 on sharp ones, and K binds on more than half the tiles of
  a heavy-tailed fixture without measurable quality effect *(unrecorded estimate)*.
- **Accumulator precision.** On a target whose FP8 MMA truncates the incoming fp32 accumulator (SM89 keeps
  14 significand bits, `fp8-mma-accumulator-precision.md`) the per-tile rescale becomes a bias: 13.5% error
  at 446k instead of 4.3% unless the rescaled accumulator is pre-rounded to the MMA's grid, which costs a
  further +2.7–3.6% there *(unrecorded estimate)*. SM120 accumulates exactly and needs nothing.
- **Coherent tiles.** Sixty-three tokens sharing one logit turn the cast into a phase lottery (half of an
  adversarial fixture's cells worse than the shipped cast, worst 3.2×); the same lottery applies to the
  shipped cast, 0.1 nats of logit spread erases it, and on real activations the row mass error never
  exceeded 2.3% *(unrecorded estimate)*. See `coherent-phase-mass-hazard.md`.
- **Top-cell crowding.** Values within ln(448/432) = 0.036 nats of their tile maximum all round up to 448, a
  one-sided bias for near-uniform tiles: +1.5% decoded mass at 0.01 nats of score spread, +0.01% at 0.05
  *(unrecorded estimate)*. Real tiles are rarely that flat.

## Prior art

Per-block scaling of P to the FP8 maximum appears in FlashAttention-3's FP8 path and in SageAttention2's
per-block P quantization; the exact overlap with an exact-maximum per (row, tile) reference, a clamped lazy
running maximum and the V-scale ratio carried in the accumulator conversion has not been verified against
those papers. The accumulator-precision interaction was not found discussed there. Not searched further.

## Cheapest decisive test

Already run in scratch and awaiting a committed form: the closed-form simulator on the production pack (any
CUDA), and the fused build (`-DFP8PA_FULLRANGE_P` on a copy of the shipped source) validated against its
simulator and timed in balanced Latin-square blocks at T = 128 and 4096 (SM120 for qualifying numbers; the
RTX 4090 needs the 14-bit pre-round). About an hour on the qualified target. The open question is the price:
a rescale that a warp vote can skip without losing the exact-maximum property.

## Log

- 2026-09-01 — simulator lead (Codex `full_tilemax`), recurrence, clamp and epilogue derived, fused
  prototype on the 4090 exposed the accumulator-precision issue.
- 2026-09-02 — hardened (bf16-PV guard, masked-half hold, RNE pre-round), SM120 validation and timing,
  real-activation kernel A/B; note opened.
