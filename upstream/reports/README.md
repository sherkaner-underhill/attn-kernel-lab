<!-- SPDX-License-Identifier: Apache-2.0 -->
# Filable reports

One directory per report that could be filed upstream, holding the minimal
public reproducer and the evidence in the form a maintainer needs — not our
internal investigation narrative.

| Planned report | Claim | Source dossier |
|---|---|---|
| `sglang-dflash-verify-mask-capture/` | B1 — DFlash verify graph captured in custom-mask mode with a dummy mask no replay refreshes | private |
| `sglang-fork-position-extend-width/` | B2 — anchor position table sliced without a width check on session-fork extends | private |
| `sm120-mma-instruction-rates/` | K4/K5 — instruction-rate map and the `sm_120a` gencode trap | private |
| `cudnn-frontend-silent-kwarg-mask/` | B3 — pygraph swallows `diagonal_alignment`, silently running TOP_LEFT for a BOTTOM_RIGHT request | [public probe](../../probes/cudnn_frost/probe2_pygraph_bf16_WRONG_MASK.py) |

A report is ready when it has: a reproducer that runs without this project's
restricted inputs or rented infrastructure, exact environment facts, the observed-versus-expected
statement, and — where a fix is proposed — the argument for why that fix is the
correct invariant rather than a workaround that happens to help.
