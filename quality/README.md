<!-- SPDX-License-Identifier: Apache-2.0 -->
# The Q lane — model-fidelity gate

This directory implements the engine-free attention-output boundary. It
compares the candidate with BF16 on seeded synthetic fixtures, reports mean and
worst-slice metrics against same-input controls, and cannot establish
application-level quality. The record schemas and validator rules live with the
rest of the promotion system.

| Rung | What it measures | Engine | Tier | State |
|---|---|---|---|---|
| **Q0** | schemas, cross-record rules, harness self-tests | none | CPU | **implemented** — `tests/test_fidelity_records.py`, `tools/validate_registry.py` |
| **Q1** | attention output vs BF16, per (case, head, row) | none | 4090 (public) / production target (private) | **public lane implemented**; the private lane needs a restricted capture |
| **Q2** | in-server residual stream and logits | SGLang | production target | not implemented |
| **Q3** | long-context retrieval probe | SGLang | production target | not implemented |

Q1 gates production-target entry (it is the precondition of S3), Q2 comes before
S6, and Q3 precedes downstream application-level tests. Those positions are recorded in
`engines/profiles/sglang.yaml`, whose ladder is ordered.

## Run it

```bash
# full production depth, 446,335 tokens, on the local 4090; ~40 s, 6.2 GiB peak
python3 quality/q1_public.py

python3 quality/q1_public.py --quick            # smoke at 4,096 tokens
python3 quality/q1_public.py --no-write         # measure, write nothing
python3 quality/fidelity_diff.py quality/scorecards/*.json
python3 tools/validate_registry.py              # validates every record it writes
```

It needs the CUDA environment (`conda activate attn-kernel-lab`), the same one
`tests/kernel` needs. Everything else here — the schemas, the validator rules,
the record assembly, the tests — is CPU-only and runs in the plain environment.

## What is where

```
quality/
  q1_public.py        the Q1 runner: reference, controls, candidate, records
  metrics.py          slicing on top of the metric donor; executes the candidate
  fidelity_records.py record assembly; the ratio basis and the saturation rule
  fidelity_diff.py    the table diff (plan §6.2)
  fixtures/           fixture-set manifests (hashes, seeds, geometry; no tensors)
  scorecards/         one record per (subject, mode, fixture set, boundary)
  runs/               raw per-case output, for attribution after the fact
promotion/
  schema/fidelity-scorecard.schema.json
  schema/fidelity-envelope.schema.json
  schema/fidelity-fixture-set.schema.json
  envelopes/          append-only, like promotion/releases/
```

## Four things this lane does on purpose

**It implements no metric.** `probes/quality/pertensor_vs_finegrained.py` is the
metric donor. Its `_metrics` defines row-relative L2, cosine similarity, relative
L1, RMSE, the output-norm ratio and the NaN/Inf counts for this project; its
`_fp32_reference` and `_run_scheme` are the reference and control paths; its
`_pv(..., "p_online", ...)` is the P-rounding form `--check-oracle` pins against
`oracle_a.attention`. `quality/metrics.py` calls those and adds slicing. A second
implementation would be a second definition, and the public numbers and the
instrument's numbers would stop being comparable.

**Every number is a ratio to a control, and the control is measured at the same
slice.** The anchor is the BF16 implementation swap — bf16 inputs, bf16 P, bf16
output — computed on the same tensors in the same process. That is the same
quantity FlashAttention's own suite asserts against, and it reproduces this
repository's recorded 0.31–0.45% attention-output control band without tuning. A
metric whose control has moved at least half as far as the candidate is marked
`saturated` and demoted to a smoke check; that rule is what keeps a null detector
from silently ranking backends, and the validator re-derives it independently.

**Four aggregations, and the fourth is the one that matters.** `mean`,
`worst_layer`, `worst_head` and `worst_row`, all schema-required. An average-only
scorecard would have passed a kernel with a 56.40% worst layer. A worst-head-only
scorecard would have passed the per-tensor scheme whose worst row lost its
direction entirely at depth while its mean and its worst-head aggregate were both
*non-monotone* — that is the addendum's finding, and this run reproduces the
shape: candidate zero's mean row-relative error is 3.5% and its worst row is
29.9%.

**`norm_ratio` is reported and never gated.** Addendum item 2. It drifts
informatively with depth and never leaves [0.98, 1.02], so a gate would never
fire while the trend line is the artefact. The schema makes `gated` a constant
false so the rule cannot be quietly reversed.

## What the public lane is, and is not

The fixtures are **public and redistributable by construction**: seeded synthetic
tensors with no restricted inputs, real activations, or model weights, reproducible
from the recorded seeds alone. They are the instrument's five distributions, each built to
isolate a failure mode one of the operator's transforms exists for — iid
Gaussian, Student-t(3) heavy tails, a RoPE-like shared channel offset, a
high-variance V channel and a massive-activation V channel. That is a sharper
test than the `torch.randn` the upstream bar is set at, and weaker evidence than
a real capture.

So the *method* travels from this lane and the *production numbers* do not. The
plan ranks a synthetic public lane honestly — "little" — and the reason it is
what runs here is that the public lane must contain nothing derived from restricted
third-party data and nothing private, and the alternative (real activations from an open-weights model)
cannot reproduce the D256 / 24:4 geometry unless such a model exists, which is
the plan's own biggest open question. **Absolute levels here do not transfer to
real activations.** Lane P, on the production target, owns the production numbers.

`quality/fixtures/*.json` is written so the private lane uses the same manifest
shape with `lane: private_real`, at which point the validator refuses to let it
name a path to a tensor or call itself redistributable.

## Reading a scorecard

- `fixture_set.slice_semantics.layer` says what a "layer" is. In this lane it is
  `fixture_case`, because there is no model; in the private lane it is
  `model_layer` over all sixteen full-attention layers. A worst-layer number here
  is a statement about a distribution.
- `controls[]` carries the anchor, the input-rounding control, the harness floor
  (a same-config repeat must be bit-identical) and the oracle chain (candidate ≈
  the donor's simulated pipeline ≈ `oracle_a.attention`).
- `not_claimed` is required, and it is the field that has to survive the
  copy-paste. Nothing here is a model-quality claim, a performance claim, or any
  form of "quality-neutral".

## Envelope discipline

`promotion/envelopes/` is append-only, exactly like `promotion/releases/`. An
envelope is never edited; superseding one means adding a record that names the
reason. A `draft` envelope may not enforce anything, and `worst_slice_enforced`
requires `calibrated: true`, which requires three independent fixture sets. The
validator enforces all of that, so the draft that candidate zero seeded cannot
quietly start gating anything.
