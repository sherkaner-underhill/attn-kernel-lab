<!-- SPDX-License-Identifier: Apache-2.0 -->
# Clean-clone reproduction session — 2026-08-31, `card-D`

A stranger's-eye check of this repository: a freshly rented RTX PRO 6000
Blackwell machine (SM120, a fourth physical device — `card-D` in
`docs/hardware-labels.md`) received exactly one artifact, a git bundle of this
repository at its initial commit, and ran the suites and benches from that
clone alone. No private materials, credentials, or prior session state were
placed on the machine.

Environment on arrival matched the qualification pin without intervention:
torch 2.13.0+cu129, CUDA 12.9 (nvcc build 36037853), driver 595.91.

## Stages and results

| Stage | Result | Duration |
|---|---|---|
| clone from bundle (single-commit history verified) | OK | <1 s |
| dependency install | OK | ~1 s (image-satisfied; see caveats) |
| `python3 -m pytest -q` (full suite, GPU lane collected) | **368 passed, 7 skipped** | 74 s |
| `tools/validate_registry.py` | 16 records validated | ~1 s |
| `quality/q1_public.py --no-write` (full 446,335-token depth) | OK | 21 s |
| eager schedule bench (`--layers 16`, fully measured 14×16) | OK | 19 m 36 s |
| CUDA-graph lane (recorded protocol, 10/30 replays) | OK | 12 m 57 s |

The 7 pytest skips are environment-gated (`cuobjdump` absent, wheel-loader
lane, per-shape golden guards); the entire GPU correctness lane ran, including
every golden-bitexact test — the goldens pinned on `card-A`/`card-B`/`card-C`
reproduced **bit-exact on a fourth die**.

## Numbers against the committed records

| Quantity | This session (`card-D`) | Committed record |
|---|---|---|
| Schedule-weighted core | **593.9 TF/s** (65.94 s) | 587–601 TF/s across `card-A`/`card-B` (K8, FROST summary) |
| Preprocessing / schedule | **6.17 s** | 6.19 s (K10, `card-B`) |
| Honest inclusive | **546.1 TF/s** (71.72 s) | 550.0 TF/s (K10, `card-B`); 527.7 (K8, `card-A`) |
| Graph-lane inclusive span | **71.71 s** (546.2 TF/s) | 71.47 s / 548.0 TF/s (`card-C`) |
| Graph vs eager parity | **0.01 s** apart | expected: parity |
| Q1 row-rel L2 mean / anchor / ratio | **3.31% / 0.50% / 6.65×** | 3.51% / 0.53% / 6.60× (`local-dev-gpu` scorecard) |
| Q1 NaN/Inf | 0 / 0 | 0 / 0 |

Every figure lands inside the committed cross-device band. The Q1 comparison is
cross-architecture (SM120 vs the scorecard's SM89 device) on the same seeded
fixtures; the anchored ratios are the comparable quantity and agree to within
1%. Per-row extremes differ per device (worst-ROW here 54.8% on `heavy_t3` at
16.7× its anchor slice, vs 29.9% at 6.1× on `local-dev-gpu`) — worst-slice
values are device-sensitive in a way anchored means are not, which is the
reason the records report both.

## Records from this session

- `120-20260831T230216Z-repro-cleanclone-schedule.json`
  (sha256 `a94e1a7a604e1750ce7115bbce1562c095faaa68690ebf83629d7164a6b7d5a1`)
- `120-20260831T231513Z-repro-cleanclone-graph.json`
  (sha256 `de83e7dfa133bb202fc1267b75c04eacef86d1bad58fe577622ba736f7a1ac6a`)

## Caveats, stated plainly

- **The environment was image-warm, not cold.** The machine image already satisfied
  `requirements-dev.txt`, and the CUDA extension compiled through the image's
  `ccache`; the cold `pip` resolution and cold `nvcc` compile paths were not
  exercised by this session.
- **One file deviated from the cloned commit, by design**: the bench stages ran
  `bench/candidate_bench.py` with the raw-GPU-UUID fingerprint removal that
  lands in the same commit as this document (the fix touches only recorded
  metadata, not any measured path). Stages before the benches ran the pure
  clone.
- The q1 stage ran `--no-write`: its evidence is the session log quoted above
  and in the committed scorecard's terms, not a new scorecard record.
- Bundle-cloning note: the bundle carried `refs/heads/main` without HEAD, so
  cloning it requires `-b main` (a bundle made as
  `git bundle create <f> HEAD main` avoids this).
