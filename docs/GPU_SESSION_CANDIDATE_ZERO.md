<!-- SPDX-License-Identifier: Apache-2.0 -->
# Rented-GPU session: candidate zero

> **EXECUTED 2026-08-30** on allocation 1 (Server Edition,
> driver 595.91.07, 600 W, sustained ~2370 MHz under load). Outcomes:
> correctness **120/120** (48 numerical + 72 goldens — zero toolchain drift);
> schedule replay fully measured (224 calls): **preprocessing 8.21 s** (the
> ~0.2–0.3 s claim was wrong by ~30x; the baseline JSON's interpolation was
> right), **core 66.71 s = 587.1 TF/s** (direct; retires the inferred ~610),
> **inclusive 74.23 s = 527.7 TF/s honest**. Records published under
> `promotion/`; steps 1–4 complete, step 5 (SGLang S0) not run — bare CUDA
> environment.

The plan for the first qualification session on the rented RTX PRO 6000. Goal:
the **current production kernel**, validated through the new benchmarking and
promotion path, ending in the first artifact manifest and — if the SGLang leg is
included — an integration-smoke attestation.

**Provisioning requires the user's explicit go-ahead. Nothing below authorises
creating a rented GPU instance.**

## What is already done (no rented GPU was needed)

| Item | State |
|---|---|
| Canonical sources imported (`src/attn_kernel_lab/{csrc,quant.py}`) | done, attribution in `THIRD_PARTY_NOTICES.md` |
| Stale header claims fixed (comment-only, SASS-identical) | done |
| Correctness suites imported and **passing on the SM89 dev tier: 48/48** | done — first pass on a second architecture |
| Golden suite capability-gated (skips off-SM120, never widens) | done, verified skipping on SM89 |
| Bench harness `bench/candidate_bench.py` (successor to the legacy bootstrap) | done, self-checked end-to-end on SM89 |
| 446,335 vs 446,464 resolved → workload profile unblocked | done, resolution on record |
| Toolchain parity (nvcc 12.9.86 / torch 2.13.0+cu129 = production-target pin) | done locally |

## Rented-GPU requirements

- **SKU:** RTX PRO 6000 Blackwell **Server Edition** — the protected target.
  A workstation/Max-Q SM120 variant cannot qualify anything
  (`targets/sm120-rtxpro6000-server.yaml`, `protected: true`).
- **Stack:** CUDA 12.9 toolkit, torch 2.13.0+cu129, g++, ninja — the same pin
  the local tier mirrors. **No model download and no SGLang are needed for the
  kernel lanes**; this can be a bare CUDA environment, much lighter than the
  serving environment.
- Record per-allocation facts at start AND end (the bench does this
  automatically): UUID, power limit, observed clocks, temperature, driver.

## Session sequence

Ordered so that everything downstream of a blocker comes after it.

### 1. Sanity (≈5 min)

```bash
git clone <lab repo> && cd attn-kernel-lab
pip install -r requirements-dev.txt
python tools/probe_target.py --check targets/sm120-rtxpro6000-server.yaml
python tools/validate_registry.py
```

Probe mismatches → fix the profile from the device (that is what `must_verify`
is for), commit, continue.

### 2. Correctness gate (≈10 min, JIT ≈1 min)

```bash
python -m pytest tests/kernel -q            # 48 numerical + 72 golden bit-exact
```

- The goldens now run (capability 12.0 matches) and must pass **72/72** — the
  imported source is comment-edited only, so SASS is unchanged; a golden
  failure here means toolchain drift and stops the session.
- Any numerical failure stops the session. Candidate zero is supposed to be the
  kernel we already trust; if it is not, that is the finding.

### 3. THE two blocker measurements (≈20 min)

```bash
python bench/candidate_bench.py --profile d256-24x4-446k \
    --label candidate-zero-schedule --layers 16
```

One command answers both open questions:

- **Preprocessing cost** (`schedule aggregate → preprocessing`): settles
  ~0.2–0.3 s claimed vs ~7–8 s implied. The SM89 self-check already makes the
  low claim implausible (0.73 s for just 2 shallow chunks × 16 layers), but the
  protected number is measured here.
- **Direct post-rescale deployment rate** (`core → schedule_tflops`): replaces
  the inferred ~610 TF/s with a measurement. Expectation from history:
  548–557 TF/s pre-skip, −7.8% kernel A/B suggests ~590–620. Whatever it is,
  it is the denominator from now on.

Also run `--layers 1 --iters 30` once for a higher-repetition per-chunk view.

### 4. Publish candidate zero (≈30 min)

- `tools/tree_digest.py --root . src tests` → `source_tree_sha256`.
- Assemble the v0.3.0 artifact manifest against
  `promotion/schema/artifact-manifest.schema.json` (v1: the JIT build
  identity from `attn_kernel_lab.source_build_id()` stands in for a wheel;
  record that honestly in `limitations` — wheel packaging is a follow-up).
- Write the correctness + kernel-performance attestations
  (`kind: correctness`, `kind: kernel_performance`, target
  `sm120-rtxpro6000-server`, workload `d256-24x4-446k`, lane
  `schedule_replay`, timing backend `cuda_events`), referencing the bench JSON
  by sha256.
- `python tools/validate_registry.py` must pass with the new records in place.

### 5. Optional same-session leg: SGLang smoke (S0)

Needs the serving stack (SGLang @ `1cf2b8c54d`), so only if the instance is
provisioned with it:

- Wheel/JIT loads inside the SGLang env; build ID reported.
- Dispatch sidecar on one uncached 446k request: **exactly 224 candidate EXTEND
  calls; zero in decode / verify / draft / GDN**.
- **Assert the observed 14 chunk geometries equal the workload profile's cases**
  — the residual check the token-count resolution delegated to S0.
- Record as an `integration_smoke` attestation (`engine: sglang`,
  `engine_revision` recorded).

### 6. Before terminating

- Re-record environment (clocks drift under load).
- Commit results + attestations, push.
- `nvidia-smi` final state into the session log.

## Explicitly out of scope for this session

- Any *new* kernel optimization (Steps 4–5 of the ladder, FP4). Candidate zero
  is a measurement and packaging session.
- Second-allocation confirmation (required before *production* qualification,
  not for publishing the first manifest — it is the next session).
- Cold-L2 / CUDA-graph / CUPTI timing lanes; FlashInfer-Bench integration.
- The originating deployment's pin update (follows once the manifest exists).

## Abort criteria

Stop and reassess rather than burn rented-GPU time if: the golden lane fails (toolchain
drift), correctness fails (the kernel is not what we thought), or the measured
preprocessing cost exceeds ~15 s per schedule (the inclusive story changes
qualitatively and the next move is profiling, not benchmarking).
