<!-- SPDX-License-Identifier: Apache-2.0 -->
# Workload profiles

One YAML file per model and request schedule. The schedule is **data**: the
profile declares parameters, `tools/gen_workload.py` expands them deterministically
into an explicit case list, and the SHA-256 of that canonical list is recorded in
the profile and carried into every result and manifest.

This is not bookkeeping. Loop arithmetic hidden inside benchmark code cannot be
hashed, so a result cannot prove which schedule produced it — and a schedule that
quietly changes silently invalidates every number that cites it.

```bash
python3 tools/gen_workload.py workloads/profiles/<id>.yaml           # inspect
python3 tools/gen_workload.py workloads/profiles/<id>.yaml --write   # regenerate
python3 tools/gen_workload.py workloads/profiles/<id>.yaml --check   # CI gate
```

Supporting another model means adding a file here. It should never mean editing
a harness.

## `status: protected`

A protected profile reproduces an accepted production run. Its parameters may
change only through a reviewed change that names the reason and preserves the old
record.

A protected profile with a non-empty `origin.unresolved` list is reported
**BLOCKED** by the validator and must not back a promotion decision. That is the
current state of `d256-24x4-446k`: the production anchor is 446,335 tokens
while an older hard-coded benchmark target was 446,464, and until that is settled
against the accepted run's own token/chunk metadata, the profile describes two
different workloads.

`generated/` is derived output. Regenerate it; do not hand-edit it.
