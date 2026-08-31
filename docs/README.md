<!-- SPDX-License-Identifier: Apache-2.0 -->
# Documentation

| Document | What it settles |
|---|---|
| [`OPERATOR_CONTRACT.md`](OPERATOR_CONTRACT.md) | What the operator *is*, independent of any GPU. Both correctness oracles test this. |
| [`TARGETS.md`](TARGETS.md) | Why there is a target axis, what an implementation family is, and what the development tier may and may not claim. |
| [`DECISIONS.md`](DECISIONS.md) | Choices that are expensive to revisit, with the alternatives and why each fails. |

Still to be written, in roughly this order:

- `ORACLES.md` — the exact-contract and BF16-fidelity references, and the rule
  that a failure against the first can never be waived as quantization error.
- `BENCHMARKING.md` — the lane ladder, timing protocol, and the statistics a
  promotion aggregate requires.
- `EVIDENCE.md` — what is retained in Git, what is content-addressed, and the
  hygiene rules for anything that may become public.
