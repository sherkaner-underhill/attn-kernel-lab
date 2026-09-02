<!-- SPDX-License-Identifier: Apache-2.0 -->
# After full-range P, the residual error is V: candidates for the next V representation

- **Kind:** new-operator
- **Status:** idea
- **Author(s):** @sherkaner-underhill (drafted by Claude Fable 5.1)
- **Opened:** 2026-09-02 · **Last updated:** 2026-09-02
- **Numerics impact (best guess):** semantics-changing
- **Would touch (if realized):** operator contract §3.4 (V), §3.5 (layout) if the tile structure changes, §4, `operator_contract_version`; quantizer and kernel PV path.

Evidence status: the motivating numbers are *(unrecorded estimate)* from uncommitted scratch records.

## Idea

On real activations full-range P sits 6–8% above the exact-P floor, i.e. 92–94% of the remaining
row-relative error is the E4M3 V representation (per-tile amax scale, global channel mean) *(unrecorded
estimate)*. Candidates, none tried: (a) a per-channel scale inside each tile (or per group of channels), so
outlier channels stop setting the tile's scale; (b) an orthogonal rotation on V paired with the inverse on
the output (the K-side Hadamard trick applied to V, exact in composition); (c) a finer discrete tile-scale
grid combined with a held P reference (`held-reference-rescale-skip.md`) so the two ideas pay for each other;
(d) keeping V in E4M3 but storing a second low-rank correction for the channels that dominate the error.

## Why it might work

The V error is now the largest single term and it is a representation error, not an accumulation error
(SM120 accumulates exactly). Rotations and per-channel scales are exact before the cast.

## Why it might not

Every candidate adds metadata or an extra pass, on a kernel already at the register cap; a rotation on V
costs an epilogue transform per output tile; the near-zero-V sink is the tail to protect (it is what killed
FP4 V and coarse tile scales). The exact-P floor itself is only 6–8% away, so the ceiling is modest.

## Prior art

SmoothQuant-style per-channel scaling and QuaRot/SpinQuant rotations for activations; SageAttention2's
per-block V handling. Not searched for the attention-V-specific combination.

## Cheapest decisive test

The closed-form simulator on the real-activation captures with each candidate V representation and exact P
(any CUDA, minutes each); the winner then needs the sink fixture and a cost model.

## Log

- 2026-09-02 — opened from the real-activation floor measurement.
