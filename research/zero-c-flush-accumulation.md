<!-- SPDX-License-Identifier: Apache-2.0 -->
# Flush accumulation: never feed the running accumulator to the FP8 MMA

- **Kind:** modification
- **Status:** explored
- **Author(s):** @sherkaner-underhill (drafted by Claude Fable 5.1)
- **Opened:** 2026-09-02 · **Last updated:** 2026-09-02
- **Numerics impact (best guess):** bit-visible
- **Would touch (if realized):** the kernel's PV loop (a target-scoped build option); no contract section, since the operator is unchanged; goldens on the affected target only.

Evidence status: all numbers *(unrecorded estimate)* from uncommitted scratch builds and records.

## Idea

On a target whose FP8 MMA truncates the incoming accumulator (SM89, see
`fp8-mma-accumulator-precision.md`), compute each k32 chunk's P·V with a zero C operand (the compiler emits
`QMMA … RZ`) and add the result to the fp32 accumulator on CUDA cores. The running accumulator then never
enters the tensor core: no truncation of small contributions, no pre-round or compensation for a rescaled
accumulator, for the shipped representation and for full-range P alike. It is the attention-kernel form of
DeepSeek-V3's promotion to CUDA cores, at the finest granularity.

## Why it might work

On the 4090 at 446k keys, the shipped representation with flush lands within 0.003 points of its fp32
simulator (4.283% vs 4.280%, from 4.580%), full-range P reaches 3.946% (simulator 3.943%, from 4.275% with
the pre-round), i.e. 0.862× the shipped kernel with no pre-round, and the kernel-to-simulator distance stops
growing with depth (1.7e-3 at 32k and at 446k) *(unrecorded estimate)*.

## Why it might not

It costs +13–16% on the present kernel *(unrecorded estimate)*, an order of magnitude more than its
instruction count: the kernel sits at the 255-register cap and holding 64 MMA results per tile until their
FADDs raises spill traffic from 32 to 195 STL/LDL per tile. Two periodic variants (an fp32 partial in global
memory flushed every 16 or 64 tiles, in two schedulings) were worse still (+25% to +93%) for the same reason
*(unrecorded estimate)*. And on SM120 the accumulator is already exact, so there the FADDs buy nothing. It is
therefore a requirement on a future kernel structure with register headroom (or a wgmma/TMA pipeline) for
truncating targets, not a patch of the shipped mma.sync kernel.

## Prior art

DeepSeek-V3 technical report §3.3.2 (promotion of FP8 GEMM partial sums to CUDA cores every N_C = 128
elements). Not searched for attention-specific variants.

## Cheapest decisive test

Run the microprobe on the target first (seconds). If it truncates: the scratch build
(`-DFR_FLUSH`) validated against the simulator at 446k and timed in balanced blocks — about half an hour on
any CUDA target with the kernel. If it does not truncate, the idea is moot for that target.

## Log

- 2026-09-02 — three implementations (per-chunk, periodic v2, periodic v3) built, validated, timed on the
  4090; quality established, cost prohibitive; SM120 shows no need.
