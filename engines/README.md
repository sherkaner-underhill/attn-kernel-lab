<!-- SPDX-License-Identifier: Apache-2.0 -->
# Engine profiles

One YAML file per inference engine the operator can be integrated into.

The engine is a **separate axis** from the GPU target and the workload. An
integration or application qualification is a fact about
`(artifact, engine, target, workload)` — four coordinates, generalising across
none of them.

Without this axis the framework leaks everywhere: the adapter, the gate ladder,
the dispatch-count assertions, and the consumer lock's `required_engine_commit`
all quietly become SGLang-shaped, and adding a second engine turns into a
refactor of the evidence schema rather than a new file.

## What crosses an engine boundary, and what does not

**Carries over unchanged:** the operator contract, both correctness oracles, the
transforms, and every kernel-performance lane. These are engine-independent by
construction — that is much of the point of having a contract at all.

**Does not carry over:** the adapter, every integration gate result, the
dispatch-count assertions, and any assumption about the KV pool's layout or
block size.

## Registered

| Engine | Status | Mechanism | Notes |
|---|---|---|---|
| `sglang` | active | pinned patch | Production path. Two known engine defects affect what its gates mean. |
| `vllm` | planned | unknown | Placeholder for the axis. Nothing built or run. |

## Adding an engine

1. Write `engines/profiles/<id>.yaml`; `id` must match the filename stem.
2. Name the **seams** exactly — the functions, registries, and CLI choice lists
   the integration touches. That list is what makes an engine version bump
   reviewable instead of exploratory.
3. Set `phase_scope` to the phases the operator may serve. Everything else keeps
   its stock backend and must assert zero candidate calls.
4. Record `known_issues` that change what a gate result *means*. A gate that
   passes only because a workaround is in place is not the same result as one
   that passes cleanly, and the distinction has to survive being written down.
5. Give each gate a `requires_target_authority`. A gate needing production
   hardware cannot be satisfied on the development tier, and validation says so.
