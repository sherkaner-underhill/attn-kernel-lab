<!-- SPDX-License-Identifier: Apache-2.0 -->
# V error-mean correction: fold the mean of the decoded centred V into the stored channel mean

- **Kind:** modification
- **Status:** explored
- **Author(s):** @sherkaner-underhill (Claude Fable 5.1 mechanism and correction; Codex production-pack measurement; kernel-level real-activation A/B by Claude)
- **Opened:** 2026-09-02 · **Last updated:** 2026-09-02
- **Numerics impact (best guess):** semantics-changing
- **Would touch (if realized):** operator contract §3.4 (the channel mean that is added back), §4 (Oracle A: "means and corrections"), `operator_contract_version` or a quantizer revision, the quantizer, all goldens; no attention-kernel change.

Evidence status: all numbers *(unrecorded estimate)* from uncommitted scratch records.

## Idea

The quantizer centres each V channel by its mean over N and casts the residuals to E4M3. The decoded residuals
need not average to zero; with BF16 inputs the E4M3 rounding ties are themselves BF16-representable, so the
per-channel error mean can be coherent. Correct it by storing `vmean − mean_j(decoded_j)` instead of `vmean`:
exact under Σp = 1, zero metadata, no kernel change, +8% of the quantizer pass on the 4090 (about 0.6 s per
recorded schedule there) *(unrecorded estimate)*.

## Why it might work

On the production synthetic packs it removes 15–16% of the ordinary-fixture mean error (Codex), and on 24
new seeds per fixture every pooled quantile improves *(unrecorded estimate)*. The mechanism is real: the
gain disappears with fp32 quantizer inputs and reappears with a controlled lattice offset.

## Why it might not

**It does nothing on real activations.** Applied at kernel level (by replacing the `vmean` tensor the kernel
adds back, with everything else identical) on Qwen3.5-9B captures through the production path on SM120, it
changes the error by 0.9996–0.9999× on every layer; the per-channel deltas it removes are ~1e-4
*(unrecorded estimate)*. The 15% was a property of the synthetic gaussian V, whose BF16 values sit in a few
binades where E4M3 ties are coherent. It remains correct and cheap, but it is not a gain to version on its
own; if adopted at all it should ride along with a quantizer revision made for another reason.

## Prior art

Zero-point / bias correction after quantization is standard in weight quantization (e.g. the bias-correction
step of data-free quantization methods); the lattice-tie mechanism with BF16 sources is the specific
observation here. Not searched further.

## Cheapest decisive test

Done in scratch: the kernel-level A/B on real activations through the production path (SM120, minutes
once captures exist). To revive the idea, find a real V distribution where the decoded-mean delta is not
negligible relative to the output.

## Log

- 2026-09-01 — V-delta gains traced to the lattice-tie mechanism; the direct correction proposed and measured
  on the production pack (Codex) and on 24 seeds (Claude); cost measured.
- 2026-09-02 — kernel-level real-activation A/B: no effect. Recommendation: do not version on its own.
