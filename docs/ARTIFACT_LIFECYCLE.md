<!-- SPDX-License-Identifier: Apache-2.0 -->
# Artifact lifecycle: from new kernel to qualified release

This is the runbook for adding a **new artifact** to the lab — a new kernel, a
new implementation of the existing operator, or a third-party kernel expressed
here for controlled comparison — and carrying it through the qualification
ladder. It documents the *procedure*; the record semantics it produces are
defined in [`promotion/README.md`](../promotion/README.md), and the conduct
rules in [`AGENTS.md`](../AGENTS.md) apply throughout.

Two facts shape everything below:

- **A release is a record, not a code drop.** The artifact's identity is an
  immutable `artifact-manifest.json`; everything learned later attaches as an
  append-only attestation referencing the manifest digest.
- **Partial qualification is a legitimate published state.** A release with a
  correctness attestation and no performance attestation is not unfinished —
  it is a documented handoff asking someone with production-authority hardware
  to run the next rung. See [Handoff states](#handoff-states).

## Step 0 — Decide what you are building

Answer three questions before writing code; they determine every version field
you will later put in the manifest.

1. **Same operator, or a new one?** The
   [operator contract](OPERATOR_CONTRACT.md) defines the current operator
   independently of any implementation. If your kernel computes *that*
   operator (same math, same normative numerics), you implement
   `operator_contract_version: 1` and Oracle A
   (`tests/kernel/test_oracle_a.py`) already tests you. If your operator
   differs — different quantization scheme, different mask family, page size,
   scale semantics — you owe a new contract section with its own
   `operator_contract_version` and an oracle for it. A failure against the
   contract oracle can never be waived as "expected quantization error."
2. **Same dataflow family, or a new one?** Register/layout choices tied to an
   MMA family (the SIGMA permutation is the worked example) are versioned as
   `layout_version`. A `wgmma`-based implementation is a new `layout_version`
   even when the operator is unchanged.
3. **First-class artifact, or study-tier probe?** A kernel you intend to
   qualify and have consumed goes through this full document. A kernel being
   *studied* — a competing implementation run under controls for comparison —
   can instead live as a probe (`probes/<name>/`, self-contained build + run
   instructions in the source, results as committed records; see
   `probes/cudnn_frost/` for the precedent). Probes produce claims and
   controls, not releases. Promoting a probe later means starting at Step 1
   like anything else.

## Step 1 — Source layout

- Kernel sources live under `src/attn_kernel_lab/csrc/`, one `.cu` per
  kernel, with the Python-side preprocessing/launch wrapper alongside the
  existing modules. A new kernel must not change the behaviour of an existing
  one: the existing goldens and rejection tests must stay green untouched.
- Study-tier/foreign kernels live under `probes/<name>/` instead.
- Every new file carries an SPDX header. Third-party source retains its
  original notices and is recorded in `THIRD_PARTY_NOTICES.md`; vendoring
  someone else's kernel wholesale is usually the wrong move — prefer building
  against their published artifact and recording the pin.
- The CPU tier must stay green at every commit:
  `python3 tools/validate_registry.py && python3 -m pytest -q`.

## Step 2 — Correctness before any measurement

- **Contract conformance.** Implementations of contract v1 run against Oracle
  A as-is. A new operator needs its oracle written first — simple
  high-precision code validating intermediate boundaries (scales, quantized
  tensors, means, layouts), not only final outputs, per contract §4.
- **Fidelity is a separate question.** Closeness to BF16 attention (Oracle B,
  `quality/` tooling) must never be merged with contract conformance into one
  tolerance.
- **Device suite.** GPU tests live under `tests/kernel/` and are collected
  only when a CUDA device is present. A qualifying kernel needs, at minimum:
  oracle conformance, capability rejection (out-of-surface requests raise the
  typed error), and the workspace-hazard trio from contract §5 — poisoning,
  reuse with changed values, long-after-short.
- **Goldens.** Bit-exact goldens are per (target, toolchain) and pin
  regressions, not portability. Record the environment they were captured on.

## Step 3 — Name the hardware: target profiles

Every measurement binds to a target profile in `targets/` (schema in
`targets/schema/`). If your GPU has no profile, add one:

- `id`, `status`, `access`, `device` block (model, compute capability, SM
  count, memory, power limit), `capabilities` block (implementation family,
  MMA families, shared-memory ceilings — read the annotated fields in the
  existing profiles; some values come from specific CUDA attributes and the
  comments say which).
- **`authority` is the load-bearing field.** `development` targets support
  correctness work and diagnostics; they cannot appear in a
  `kernel_performance` attestation and cannot qualify a release — the
  validator refuses it. `authority: production` is granted in review, not
  self-declared (see `CONTRIBUTING.md`).

## Step 4 — Name the work: workload profiles

- Reuse an existing profile in `workloads/profiles/` when your claim is
  comparative — identical hashed cases are what make two artifacts'
  measurements pairable.
- A new geometry means a new profile expanded through the sanctioned producer:
  `python3 tools/gen_workload.py workloads/profiles/<profile>.yaml`. The
  expansion is deterministic and hashed; attestations record
  `workload_cases_sha256`, and the validator rejects a record whose hash no
  longer matches the profile as it stands. Never expand a schedule by hand.

## Step 5 — Measure

- **Lanes.** `core`, `preprocessing`, and `upstream_comparison` are
  diagnostic. Only `inclusive` (transforms included — the honest reusable
  number) and `schedule_replay` (full-schedule replay) carry promotion
  authority.
- **Protocol.** Explicit timing backend (`cuda_events`, `cupti`, or
  `host_wall` — recorded, never assumed), stated L2 policy and graph mode,
  warmups counted, independent interleaved blocks for paired comparisons
  (ABBA), medians with percentile bounds, paired deltas with 95% CIs.
- **Raw samples are part of the record.** Summaries cite
  `raw_samples_sha256`; a summary whose raw samples were discarded does not
  qualify anything.
- **Environment fingerprint.** GPU identifier (the stable neutral labels of
  `docs/hardware-labels.md`, never a bare UUID), driver, CUDA, container
  digest, power limit, observed clocks and temperature.
- `bench/candidate_bench.py` is the worked example of a conforming harness;
  a new artifact may need its own runner, held to the same protocol.

## Step 6 — Author the release record

Create `promotion/releases/<release_id>/` containing:

- **`artifact-manifest.json`** — validated against
  `promotion/schema/artifact-manifest.schema.json`. It is authored by hand
  (there is deliberately no generator yet) and written **once**: identity
  fields (`release_id`, the three version axes, `binary_abi`), `source` block
  with `source_tree_sha256` from the sanctioned producer
  (`python3 tools/tree_digest.py`, over the declared `digest_paths`),
  `artifacts` block (wheel/cubin digests, build container digest),
  `toolchain`, `targets` (supported / qualified / excluded, with reasons),
  and the `contract` support surface. If a fact will change later, it does
  not belong in the manifest — it belongs in an attestation.
- `environment.json`, `correctness-summary.json`, `benchmark-summary.json`,
  `profile-summary.md`, `known-limitations.md` — the human-readable layer.
  `known-limitations.md` is not optional; an artifact with no known
  limitations has not been examined.

Release ids follow `<geometry>-<method>-v<major.minor.patch>` (existing:
`d256-int8-fp8-v0.4.0`). Version bumps follow the axes: new API →
`package_api_version` and a minor/major bump; re-measurement of the same
artifact is a new attestation, never a new release.

## Step 7 — Attach attestations

Each attestation is one JSON file in `promotion/attestations/<release_id>/`,
validated against `promotion/schema/qualification-attestation.schema.json`:

- `kind` ∈ correctness · kernel_performance · integration_smoke · application
  · production · upstream_readiness · fidelity, with `verdict` pass/fail,
  `artifact_manifest_sha256` binding it to the manifest, `recorded_utc`, and
  an `approver` (`identity` = who ran and vouches for it, `review` = what was
  reviewed).
- A `kernel_performance` attestation names all three of (target, workload
  profile, cases hash) plus the `environment` and `measurement` blocks —
  the qualification is a fact about that triple and generalises across none
  of it.
- `gates` is the checklist actually enforced, each entry with a result and
  evidence digest. **A skipped gate needs a reason, and a warning never
  becomes a pass.**
- Attestations are append-only. Superseding a bad one means adding a new one
  that says so; the old file stays.

## Step 8 — Register and validate

- Append the release and its attestation paths to `promotion/registry.yaml`.
  The registry is append-only and deliberately has **no floating aliases** —
  no `latest`, no `stable`. Consumers pin exact release ids.
- `python3 tools/validate_registry.py --verbose` must pass, and is the
  arbiter for this whole document: schema conformance, digest bindings,
  authority rules, lane rules, cross-record invariants. When this document
  and the validator disagree, the validator is right and this document has a
  bug — fix it.
- An artifact holding correctness + kernel_performance + build verification +
  integration_smoke is an **integration candidate** (see the registry file's
  trailer comment); application and production facts remain contextual
  attestations, never a global mutable status.

## Step 9 — Envelopes (only when claiming quality parity)

A non-inferiority claim against a baseline requires an envelope under
`promotion/envelopes/` (schema `fidelity-envelope.schema.json`). Envelopes are
born `.draft` and stay draft until a production-tier run confirms them and a
reviewer freezes — the validator enforces the status transition. Do not claim
parity from a draft envelope.

## Handoff states

| Release contains | Meaning | Legitimate next actor |
|---|---|---|
| manifest only | identity staked, nothing demonstrated | author: correctness first |
| + correctness | operator verified on some device | anyone with a production-authority target: performance |
| + kernel_performance (production target, promoting lane) | qualified kernel measurement | integrator: integration_smoke |
| + integration_smoke + build verification | integration candidate | consumer: application/production attestations in their own context |
| + fidelity with frozen envelope | quality parity claimed against the named baseline | downstream quality review |

Publishing an early rung and explicitly requesting the next one is the
intended collaboration mechanism — say so in the pull request.

## What stays out of Git

Wheels, cubins, raw profiler captures, large traces and tensors: hosted
elsewhere (release assets or content-addressed storage), never committed. The
manifest records digests and a `stable_locator`. Never a credential, signed
URL, third-party dataset, or activation capture — see `AGENTS.md`.

## Known thin spots

Stated so they are worked around rather than tripped over:

- The manifest has no generator tool; it is authored by hand against the
  schema and checked by the validator. Copying an existing release and
  editing every field is the honest procedure.
- Oracle A covers contract v1 only; a new operator's oracle is the largest
  unscaffolded cost of a new contract.
- This document predates the lab's first *externally authored* artifact. The
  first such expression should amend it where reality disagrees.
