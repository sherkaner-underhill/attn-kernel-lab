<!-- SPDX-License-Identifier: Apache-2.0 -->
# Decisions

Short records of choices that are expensive to revisit. Newest first.

## D3 — Release IDs carry no architecture (2026-08-29)

`d256-int8-fp8-v0.3.0`, not `sm120-d256-int8-fp8-v0.3.0`.

One release may support several targets and be qualified on a subset of them, so
an architecture baked into a permanent identifier becomes a lie the moment a
second target qualifies. The architecture lives in the artifact `variant_id`,
where it identifies the binary ABI and code-generation target, and in the
per-target qualification records. Enforced by the manifest schema's `release_id`
pattern.

## D2 — The repository is multi-target from day one (2026-08-29)

Named `attn-kernel-lab` rather than `sm120-attention-lab`, with a target
registry, a workload registry, and a `targets` block in every manifest — despite
exactly one production target existing today.

The design's central asset is a set of permanently immutable records. Adding an
axis afterwards means schema v2 plus a migration of every published release. The
axis is nearly free now and expensive later. Known near-term pressure on the
axis: a local RTX 4090, a possible H200 rental, and models other than
`<private-model-id>`.

Supersedes the naming and single-SKU assumption in the original architecture
proposal.

## D1 — Two repositories, immutable promotion manifests (2026-08-29)

Kernel qualification and release separate from application acceptance. A release
is a record binding source, binary, environment, contract, and evidence digests —
not a directory into which source is copied. Adoption is an explicit pin-update
pull request; rollback is a repin. There is no automatic `latest` edge.

Rejected alternatives, with the reason each fails:

- **Keep everything in the application repository.** Preserves mixed ownership
  and lets application state influence kernel results.
- **A permanent FlashInfer fork as the research system.** Rebase and CI cost
  during exploratory work; experimental history leaks into the patch series. A
  fork is the delivery vehicle, not the lab.
- **Copy promoted kernels into a release directory.** Looks auditable; creates
  duplicate source and lets source, binary, and evidence diverge.
- **A Git submodule as the consumption boundary.** Pins source but binds neither
  the built binary, the ABI, the toolchain, nor the evidence.
- **Follow the latest passing candidate.** Passing kernel tests is not passing
  SGLang tests and neither is downstream application quality. Destroys rollback
  and causal attribution.

## D4 — The serving engine is a fourth axis (2026-08-29)

`engines/profiles/` registers SGLang (active) and vLLM (planned), integration and
application attestations name an engine and its exact revision, and the manifest
carries an `engines.integrated` block instead of a bare `required_sglang_commit`.

Without this the framework leaks into places that outlive it. The original
proposal names SGLang in the consumer lock, in the gate ladder S0–S7, in the
integration-smoke definition, and in the backend-registration patch — so a second
engine would arrive as a schema migration rather than a new file, exactly the
failure the target axis was added to avoid.

The split is clean because the operator contract already exists. Engine-independent
by construction: the contract, both oracles, the transforms, and every
kernel-performance lane, which run framework-free. Engine-specific: the adapter,
every gate result, the dispatch-count assertions, and any assumption about KV pool
layout or block size.

vLLM is registered as `planned` with no gates and `verification.state: unverified`
— it holds the axis open and claims nothing. The two questions that decide whether
its integration is an adapter or a contract change are whether its paged KV layout
can present the page-size-1 view v1 requires, and how its position-encoding
handling compares.
