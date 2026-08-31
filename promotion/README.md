<!-- SPDX-License-Identifier: Apache-2.0 -->
# Promotion

A release is a **record**, not a directory into which source is copied.

```text
promotion/releases/<release_id>/
├── artifact-manifest.json      # permanently immutable identity
├── environment.json
├── correctness-summary.json
├── benchmark-summary.json
├── profile-summary.md
└── known-limitations.md

promotion/attestations/<release_id>/
├── correctness-<digest>.json
├── kernel-performance-<digest>.json
├── fidelity-<digest>.json
├── integration-smoke-<digest>.json
├── application-<digest>.json
└── production-<digest>.json

promotion/envelopes/
└── <candidate>-v<N>[.draft].json   # non-inferiority envelopes (quality gate
                                    # Q0); .draft until a production-tier run confirms
                                    # and a reviewer freezes (validator-enforced)
```

The manifest is written once. Everything learned later — that the artifact
loaded against a particular SGLang revision, that downstream application tests
passed, or that a deployment held up under load — is an **append-only attestation** referencing the
manifest's digest. Later results never rewrite the artifact's identity.

## Why attestations are per (target, workload)

A performance qualification is a fact about `(artifact, target profile, workload
profile)`. It generalises across none of the three, so the record names all
three, and `tools/validate_registry.py` refuses one that does not:

- A target with `authority: development` cannot appear in a
  `kernel_performance` attestation at all.
- The `workload_cases_sha256` must match the profile as it stands, so a schedule
  cannot drift away from the results citing it.
- Only the `inclusive` and `schedule_replay` lanes carry promotion authority;
  `core`, `preprocessing`, and `upstream_comparison` are diagnostic.
- The timing backend is recorded, never assumed.
- A skipped gate needs a reason, and a warning never turns an incomplete
  qualification into a pass.

## What stays out of Git

Wheels, cubins, container attestations, raw `.ncu-rep`/`.nsys-rep`, large
traces, large synthetic tensors, and any access-controlled activation evidence
live in versioned release assets or content-addressed storage. The manifest
records their digests and where to find them. Ordinary expiring CI artifacts may
be execution output but must never be the only retained copy.

Never store credentials, signed URLs, machine tokens, restricted third-party
data, or raw private activations in a manifest. Artifact references are stable IDs plus
digests; access is handled separately.
