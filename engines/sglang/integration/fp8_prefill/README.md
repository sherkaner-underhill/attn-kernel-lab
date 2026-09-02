<!-- SPDX-License-Identifier: Apache-2.0 -->
# SGLang integration: the `fp8_prefill` attention backend

The glue that serves this repository's kernel from SGLang, as used by both
published serving-lane result sets (`../results/public-serving-20260901/`,
`../results/public-serving-fp4-20260901/`). It is a reversible, pinned
patch set — not an upstream contribution; upstreaming properly is tracked
through the claims ledger and the upstream issue.

## What it is

- `backend.py` — `FP8PrefillAttnBackend`, subclassing SGLang's
  `FlashInferAttnBackend` and overriding `forward_extend` only. Qualifying
  pure-EXTEND forwards run the fused kernel; everything else (target-verify,
  draft-extend, mixed batches, cross-attention, sliding-window or logit-cap
  layers, non-256 head dims, quantized KV pools, small prefills) falls
  through to the inherited FlashInfer path with its metadata intact.
  Request geometry is snapshotted at metadata time (overlap-scheduler
  contract); the layer-time path reads only the snapshot.
- `install.py` — copies the package into a pinned SGLang tree and makes two
  marker-guarded edits (backend factory + hybrid-GDN allowlist in
  `attention_registry.py`; `ATTENTION_BACKEND_CHOICES` in `server_args.py`),
  with backups, `py_compile` checks, double-apply refusal, digest printing,
  and a clean `revert`.
- The kernel itself is NOT vendored here: `install.py` stages the canonical
  `src/attn_kernel_lab/quant.py` and `src/attn_kernel_lab/csrc/fp8_prefill_attn.cu`
  from this repository, so the served bytes are the repository's kernel by
  construction (`status` prints the digests; the published runs bind
  `02c0d25bd221...` and `b9062dd066bc...`).

## Use

```
git clone https://github.com/sgl-project/sglang && cd sglang
git checkout 1cf2b8c54d81802abc15dcf23a29b9cc687bc01e   # the pin
pip install -e python
python3 <this repo>/engines/sglang/integration/fp8_prefill/install.py apply --sglang .
python3 -m sglang.launch_server ... --prefill-attention-backend fp8_prefill
```

The kernel JIT-compiles on first engaged forward (~30-60 s, cached); an
unscored warmup request absorbs it in the bench lane. Decode stays on the
stock `--attention-backend`; the stock `HybridAttnBackend` composes the two.

## Engagement conditions (else transparent FlashInfer fallback)

EXTEND forward - page_size 1 - unquantized BF16 KV pool - head dim 256 -
per-request prefix+chunk >= `SGLANG_FP8_PREFILL_MIN_TOKENS` (default 8192).
Env knobs (numerics A/B, all default to the production posture):
`SGLANG_FP8_PREFILL_K_CENTER`, `SGLANG_FP8_PREFILL_HADAMARD`,
`SGLANG_FP8_PREFILL_QK` (int8|fp8), `SGLANG_FP8_PREFILL_BF16_HEADS`,
`SGLANG_FP8_PREFILL_MIN_EXTEND`, `SGLANG_FP8_PREFILL_DISABLE`,
`SGLANG_FP8_PREFILL_DEBUG`. The published runs set NONE of them.

## Pin and provenance

Written against SGLang `1cf2b8c54d81802abc15dcf23a29b9cc687bc01e`; the
installer refuses other commits unless `--allow-unpinned` is given and both
edit anchors are still unique. Verified on a clean checkout of the pin:
pre-apply digests `attention_registry.py 36bb36a0...` /
`server_args.py 71d0ad92...` (identical to the published runs' recorded
pre-apply state), apply -> py_compile green, revert -> byte-identical tree.

`backend.py` and `__init__.py` here are functionally identical to the
modules that served both published result sets; `backend.py` differs only
in comments and this package's install markers (the runs used an earlier
private-path variant of the installer, one path constant apart). The
kernel files are digest-bound above, which is what the results depend on.

Boundary: a development-tier integration for one pinned commit and the
declared operator surface. It makes no claim for other SGLang commits,
architectures, MLA models, or quantized KV pools (it refuses or falls back
on all of these).
