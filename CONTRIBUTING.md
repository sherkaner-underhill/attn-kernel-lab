<!-- SPDX-License-Identifier: Apache-2.0 -->
# Contributing

The lab accepts contributions, including parallel kernel development by people
who have never spoken to the maintainer. The machinery was built for exactly
that: records are append-only, qualification states are explicit, and a pull
request is validated by CI before a human looks at it. This file is the
protocol; [`docs/ARTIFACT_LIFECYCLE.md`](docs/ARTIFACT_LIFECYCLE.md) is the
technical runbook it leans on, and [`AGENTS.md`](AGENTS.md) binds every
contributor, human or automated.

## What fits here

- **A new kernel or implementation**, carried as far up the qualification
  ladder as your hardware allows (see the lifecycle doc — publishing a
  partially qualified artifact and requesting the next rung is a normal,
  welcome state, not a half-finished PR).
- **Attestations for an existing artifact** run on your hardware: a
  correctness pass on a new device, a performance qualification on a
  production-authority target, an integration smoke against an engine
  revision.
- **A third-party kernel expressed as a study-tier probe** for controlled
  comparison, with its build/run instructions and committed records.
- **Target profiles** for hardware the lab has not seen.
- **Probes, analysis, and documentation fixes.**

What never enters this repository, from anyone: third-party datasets, real
activation captures, credentials or signed URLs, large binaries (wheels,
cubins, profiler captures — host them and record digests), or results whose
raw samples were discarded.

## The flow

1. Fork, branch, work. Nothing here requires coordination with the
   maintainer until review.
2. Keep the CPU tier green at every commit — it is what CI runs on your PR:

   ```bash
   pip install -r requirements-dev.txt
   python3 tools/validate_registry.py
   python3 -m pytest -q
   ```

3. One release (or one coherent set of attestations) per pull request. The
   registry and attestation directories are append-only, so parallel PRs
   conflict only on trivial registry lines.
4. Say in the PR description **which handoff state** you are publishing and
   what you are requesting next ("correctness attested on a development
   target; requesting kernel-performance on a production SM120 target" is a
   complete, mergeable contribution).
5. For anything larger than a fix, open an issue first with the intended
   `release_id`, contract/layout versions, and target — it reserves the
   namespace and catches contract misreadings before you burn GPU hours.

## Evidence and trust

The repository's standing rule is *read the evidence, not the author*, and it
applies symmetrically to contributors:

- Every attestation names its runner: `approver.identity` is the GitHub
  identity of whoever executed and vouches for the record, and the
  `environment` block fingerprints the machine. Anonymous numbers are not
  mergeable.
- Review verifies what is checkable from the record — schema and digest
  validity, protocol conformance, internal consistency, plausibility against
  known hardware — **not** the truth of the measurement. A merged external
  attestation is that contributor's claim, carried with their name on it.
- Independent reproduction upgrades a claim: a second attestation of the same
  kind from different hardware/operator is the strongest review there is, and
  the maintainer may re-run any submission before or after merge.
- Fabricated records get reverted-by-supersession (attestations are
  append-only) and end the contributor's trust here.

## Authority

`authority: production` on a target profile — the right to qualify releases —
is granted during review of the profile, not self-declared. Expect to be
asked how the device is provisioned, cooled, power-limited, and shared. New
hardware enters as `authority: development`, which is fully sufficient for
correctness attestations and diagnostics.

## Naming and identity

- Release ids: `<geometry>-<method>-v<major.minor.patch>`
  (`d256-int8-fp8-v0.4.0` is the pattern). Version-axis semantics are in the
  lifecycle doc; re-measurement is a new attestation, never a new release.
- Contributed kernels keep their own honest names in `variant_id`; a probe
  expressing someone else's kernel names the upstream pin (commit/PR) it was
  taken at.
- Licensing is Apache-2.0 inbound and outbound; every new file carries an
  SPDX header, and third-party material goes through `THIRD_PARTY_NOTICES.md`
  with its original notice intact.

## Ground rules, restated from AGENTS.md

- Never provision rented GPU hardware on someone's behalf or assume the
  maintainer will; state what hardware a request needs and let its owner
  decide.
- `promotion/releases/` and the registry are immutable/append-only. Digests
  come only from `tools/tree_digest.py` and `tools/gen_workload.py`.
- Development-authority numbers are diagnostic; the fix for wanting to
  promote one is production hardware, not a relaxed rule.
- A skipped gate needs a reason; a warning never becomes a pass.
