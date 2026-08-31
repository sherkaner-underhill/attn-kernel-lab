<!-- SPDX-License-Identifier: Apache-2.0 -->
# Targets

## Why a target axis exists at all

The design this repository implements binds evidence into permanently immutable
records. Adding an axis to an immutable record later means a schema version bump
and a re-issue of everything published under the old one. The axis therefore
costs almost nothing now, while the set of published manifests is empty, and a
great deal later.

That is the mechanical reason. The substantive one is that the three GPUs in
scope are not three sizes of one thing.

## Implementation families

A **family** is a set of targets a single mainloop can serve. Crossing a family
boundary is a rewrite, not a port.

### `mma_sync` — SM89 (Ada), SM120 (Blackwell consumer/pro)

`mma.sync.m16n8k32` for E4M3 and INT8, `ldmatrix` for fragment loads,
`cp.async` for global→shared staging, XOR shared-memory swizzle, ~99–100 KB
opt-in shared memory per block. No warp-group async MMA, no TMA.

The whole current kernel lives here. The consequence people find surprising is
that **the local 4090 is a closer relative of the RTX PRO 6000 than the H200 is**
— the version numbers suggest otherwise, but the instruction set is what matters.

SM120 adds block-scaled MMA (`kind::mxf8f6f4`) and FP4 (`kind::mxf4`). Measured
on the production card, the block-scaled FP8 path issues at exactly the legacy
rate; **FP4 is the only instruction path above the ~934 TF/s roof**, and it does
not exist on SM89 at all.

### `wgmma` — SM90 (Hopper)

Hopper reaches FP8 throughput through warp-group async MMA plus TMA, typically
with warp specialisation and a producer/consumer pipeline, and has roughly 227 KB
of shared memory to work with. An SM90 implementation would reuse the operator
contract, the transforms, and both oracles — and share almost none of the
mainloop.

Registered as `planned` so that the codebase cannot silently re-acquire the
assumption that there is one GPU.

## Authority

| Authority | May qualify a release | Performance numbers |
|---|---|---|
| `production` | yes | promotion evidence |
| `development` | no | diagnostic only |
| `none` | no | none |

`development` is not a weaker `production`. It is a different *kind* of claim.
A performance number from development hardware is diagnostic and never
substitutes for one from the production card — not because the hardware is worse,
but because the claim is about a different machine.

This is enforced in `tools/validate_registry.py`, not left to discipline: a
performance attestation naming a development target is a validation error, and a
development target carrying promotion thresholds is a validation error. Both have
tests that assert the rejection.

## Development tier: what the local 4090 is for

**Development happens on the RTX PRO 6000.** That is where the kernel is written,
measured, and integrated. The 4090 exists between rented-target sessions, and
for CI.

**Can:** Oracle A exact-contract tests at small and medium shapes; Oracle B
fidelity comparisons; Compute Sanitizer `memcheck`/`initcheck`/`racecheck`/
`synccheck`; cross-compile gating of `sm_120a` (ptxas assembles SM120 code with
no SM120 device present); adversarial and boundary suites; graph capture/replay
tests. Free, always available, and viable on every commit through a self-hosted
runner.

**Cannot:** produce any number that qualifies a release; run the 446k protected
schedule at all — 24 GiB does not hold a 446k D256 paged KV pool, whose BF16 K+V
across 16 full-attention layers is ~27.2 GiB on its own; exercise FP4; or satisfy
the pinned bit-exact regression lane, which is SM120-and-toolchain specific by
construction.

So the honest scope is narrow: keep correctness work moving between rented-target sessions,
and keep the arch-independent gates running on every commit. The authority rule
matters more than the capability — once a second GPU is reachable at all,
something has to stop a number measured on it from drifting into a promotion
decision.

## Adding a target

1. Write `targets/<id>.yaml`; the `id` must match the filename.
2. Set `verification.state` honestly. `unverified` is correct for hardware never
   accessed, and a `production`-authority target may not be `unverified`.
3. List anything taken from vendor documentation rather than read from the device
   under `verification.must_verify`. Shared-memory ceilings especially: the
   production wide path uses ~98 KiB against a ~99 KB ceiling, so "it fits" must
   be established by compiling.
4. Leave `capabilities.measured_instruction_rates` **absent** until the
   instruction-rate probe has run on that hardware. Never estimate it from
   another target.
5. Set `workload_limits` so a workload the target cannot physically hold is
   refused by validation rather than discovered at run time.
6. Run `python3 tools/validate_registry.py`.

## Local toolchain (Phase 1b, done 2026-08-29)

The development workstation runs a dedicated conda environment
`attn-kernel-lab` that **mirrors the production-target pin**:

| | local dev tier | production target |
|---|---|---|
| nvcc | 12.9.86 | 12.9 |
| PyTorch | 2.13.0+cu129 | 2.13.0+cu129 |
| driver | 595.79 (CUDA 13.2 capable) | — |

Mirroring rather than tracking the newer toolkit the driver would allow is
deliberate. A development tier exists to make numerical results comparable to
production; a toolchain skew undermines exactly that. Catching
forward-compatibility breakage early is a real benefit, but it belongs in its own
lane rather than in the tier whose whole job is comparability.

The environment also carries **`compute-sanitizer`** and **`ncu`**, which the
correctness and profiling lanes require and which a PyTorch wheel alone would not
provide.

```bash
export PATH="$HOME/anaconda3/envs/attn-kernel-lab/bin:$PATH"
python tools/probe_target.py --check targets/sm89-rtx4090-local.yaml
python tools/probe_target.py --compile-gate sm_120a
```

### What the probe confirmed

Every declared field of `sm89-rtx4090-local` was read off the device and matches:
compute capability `(8, 9)`, 128 SMs, 23.99 GiB, and — the field that mattered —
**101376 bytes** of opt-in shared memory per block, read from
`cudaDevAttrMaxSharedMemoryPerBlockOptin` rather than `sharedMemPerBlock`, which
reports the 48 KiB default and would have made the target look unusable. The
production wide path's ~98 KiB (100352 B) fits, with about 1 KiB to spare. That is
thin enough that only a real compilation settles it, which is the profile's one
remaining `must_verify`.

The cross-compile gate works: nvcc emits SM120a block-scaled MMA on this machine
with no SM120 device present. And the recorded `-arch=sm_120a` trap **reproduces
here** — it lowers to `sm_120` and ptxas fails with *"Instruction 'mma with block
scale' not supported on .target 'sm_120'"* — independently confirming claim K5 on
a second machine and a fresh toolchain.

That last point is the clearest case for keeping this tier registered at all: a
production-architecture toolchain trap is now caught on hardware that cannot run
the architecture, at zero cost, in a test that runs on every commit — rather than
on a rented production target, at cost, when someone hits it.
