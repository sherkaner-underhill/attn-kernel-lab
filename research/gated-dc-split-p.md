<!-- SPDX-License-Identifier: Apache-2.0 -->
# Gated DC split: send a flat tile's mean through a rank-1 fp32 term and only its residual through FP8

- **Kind:** new-operator
- **Status:** explored
- **Author(s):** @sherkaner-underhill (Claude Fable 5.1; broad-grid and boundary review by Codex)
- **Opened:** 2026-09-02 · **Last updated:** 2026-09-02
- **Numerics impact (best guess):** semantics-changing
- **Would touch (if realized):** operator contract §3.6 (P representation and accumulation), §4 (Oracle A), `operator_contract_version`; the kernel's softmax/PV loop (residual cast, gate, rank-1 update, per-tile V column sums or 1 KiB of stored sums per tile).

Evidence status: all numbers *(unrecorded estimate)* from uncommitted scratch records.

## Idea

For one row and one 64-token tile, `Σ_j e_j V_j = Σ_j (e_j − ē) V_j + ē · Σ_j V_j` with `ē` the tile mean of
the unnormalised weights. The residual is signed and zero-mean and goes through the E4M3 PV MMA (cast with
scale 448/e_max on top of full-range P); the mean term is an exact rank-1 update `ē · colsum(decoded V)` on
CUDA cores. Exact before quantization; V unchanged. Gate: split only when `e_max ≤ θ·ē`, θ = 8 (the largest
token owns at most 12.5% of the tile), because on concentrated tiles the split creates 63 coherent large
residuals that cancel against the mean term. A ramp of the DC coefficient from 1 at concentration 8 to 0 at 12
makes the gate continuous through its boundary at the cost of 1% of the ordinary-regime gain.

## Why it might work

Production-pack simulator relative to full-range P, held-out seeds, eight distribution families, three
regimes, sink and no sink: diffuse 0.366× (worst 0.917), ordinary 0.949× (worst 0.998), sharp 1.000×; no
cell's mean worse; worst max-row ratio 1.077. On near-uniform tiles it is 17× better than the shipped cast
and essentially exact. It repairs the coherent-phase hazard on every tile it splits (its tile mass is exact)
*(unrecorded estimate)*.

## Why it might not

**Real attention has almost no split-eligible tiles.** On Qwen3.5-9B and Qwen3-4B (32k tokens, last 128
rows) the gate passes 3.6–6.1% of the attention mass and the split changes the error by 0.03–0.06%; rows are
"diffuse" by effective token count without having flat 64-token tiles *(unrecorded estimate)*. Its kernel cost
is unmeasured (a rank-1 FFMA on the accumulator plus column sums, on a register-capped kernel where any
extra work in the tile loop spilled badly, see `zero-c-flush-accumulation.md`). Max-row tails are not
uniformly monotone (1.077 on v_outlier). It is a specialist for activation distributions with flat tiles;
none seen so far has them.

## Prior art

Mean/residual (DC/AC) decompositions are common in signal quantization; not searched for attention P
specifically. Differs from smoothing terms in SageAttention2 (which shift K, not P) in that the composition is
exact and the compensating term uses a summary of V.

## Cheapest decisive test

A count of split-eligible mass on the target activation distribution (the recorder in
`real-activation-attention-statistics.md`, minutes on any CUDA). If it stays below ~10%, do not build the
kernel; if a distribution with flat tiles appears, the fused prototype's cost is the next question.

## Log

- 2026-09-01 — simulator (CPU then production pack), gate θ = 8 chosen, ungated variant retired (sink hazard).
- 2026-09-02 — held-out grid (Codex broad grid; Claude seed offsets 10–12), boundary fixture and 8–12 ramp,
  real-activation gate statistics: deprioritised.
