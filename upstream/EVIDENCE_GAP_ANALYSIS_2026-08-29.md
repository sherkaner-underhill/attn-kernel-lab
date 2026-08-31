<!-- SPDX-License-Identifier: Apache-2.0 -->
# Evidence gap analysis: this pipeline vs. what a FlashInfer PR actually needs

**Date:** 2026-08-29. **Scope:** does the qualification ladder in this repository
produce evidence strong enough to back a FlashInfer pull request for the D256 /
page_size-1 / bottom-right-causal INT8-QK + FP8-PV EXTEND kernel?

**Method note.** Everything marked **[V]** was verified by fetching upstream sources
on 2026-08-29/30:

- a sparse clone of `flashinfer-ai/flashinfer` at
  `44428003ba219c14b5473fefec8f7bfd4b72178e` (main, 2026-08-29 08:27 UTC,
  `version.txt` = `0.6.18`) — the source of every file path, signature, tolerance and
  CI fact in §2.1–2.4b and §2.7;
- `gh pr view --repo flashinfer-ai/flashinfer` for PRs #3485, #3518, #3640, #4149,
  #4272, #4502, #4691, and `gh pr view --repo NVIDIA/cudnn-frontend` for #509, #595,
  #768 — titles, states, merge dates, file lists, bodies and review threads;
- `raw.githubusercontent.com` for `Dao-AILab/flash-attention` `tests/test_flash_attn.py`.

No **[R]** (recalled-but-unverified) claims remain in the body; the two places where a
fact came from a subagent's fetch rather than mine are marked inline. Where a prior
statement in this repository is contradicted by a fetched fact, the contradiction is
called out explicitly.

**Our own state, for reference.** The candidate-zero qualification session executed and is on
record (`docs/GPU_SESSION_CANDIDATE_ZERO.md`,
`promotion/releases/d256-int8-fp8-v0.3.0/`,
`bench/results/120-20260830T003408Z-candidate-zero-schedule.json`): correctness
**120/120** (48 numerical + 72 goldens), schedule replay fully measured over 224 calls
— preprocessing **8.21 s**, core **66.71 s = 587.1 TF/s**, inclusive
**74.23 s = 527.7 TF/s**. The suites referenced throughout are
`tests/kernel/test_kernel_vs_sdpa.py` (15 tests), `test_divergence_hunting.py`
(33 tests) and `test_golden_bitexact.py` (72 golden cases).

---

## 1. Executive summary

The ladder is unusually strong where upstream is weak, and weak exactly where
upstream is strong. Our internal evidence integrity — hashed schedule-as-data,
immutable manifests with append-only attestations, a 72-case bit-exact golden lane
that pins compiler flags and ambient TF32 state, a 33-case adversarial suite built
around named failure hypotheses, and a lane taxonomy in which only `inclusive` and
`schedule_replay` carry promotion authority — exceeds anything FlashInfer asks of a
contributor and is worth surfacing in the PR. What we do **not** have is the thing
reviewers actually consume: an operator shaped like a FlashInfer op, tests written in
FlashInfer's own harness with FlashInfer's own reference and tolerance conventions, a
`benchmarks/bench_*.py` reporting through `flashinfer.testing.bench_gpu_time`, a
same-harness stock-backend control, and an LSE output. Three findings dominate.
First, **the premise that upstream CI cannot see our SKU is false**: `CONTRIBUTING.md`
documents an NVIDIA-internal GitLab matrix with a `unit_test_rtx_pro_6000` job on
"RTX PRO 6000 Blackwell" for cu129 and cu130, and CI-bot tables inside merged PRs show
that row passing — so tests written in their harness *will* be executed on our exact
part by a maintainer running `/bot run`, which converts "write tests their way" from a
politeness into the single highest-leverage action available. Second, **LSE is
effectively mandatory**: on PR #4502 a maintainer explicitly reversed the author's
attempt to make `return_lse=False` the default, and PR #4691 — an SM120 QK-INT8 /
PV-FP8 SageAttention port — is sitting unreviewed with "it does not compute LSE"
named in its own body as the reason it cannot be wired into a wrapper. Our v1 returns
no LSE at all. Third, **the novelty window is closing**: #4149 (MXFP8 ragged prefill,
SM120/121) and #4691 (SM120 Sage INT8-QK/FP8-PV with mean-centred K, per-group Q
scales and a physical V permutation for the PV operand layout) are both open right
now and both use our technique family.

The recommended posture is therefore not "prepare a PR" but "re-aim the claim, open a
tracked issue, and build the upstream-legible surface." FlashInfer's written review
rubric (`docs/code_review_guidance_human.md`) makes the priority unambiguous: kernel
internals are explicitly **out of scope for human reviewers** — "rely on passing unit
tests and benchmarks + fuzz testing as the backstop for kernel correctness" — so the
tests *are* the review, and effort spent on them dominates effort spent anywhere else.
The same document describes an **experimental track** "declared via a tracked issue",
which is the natural home for a 24:4-only, page-size-1-only, one-SKU operator and is
also the cheapest fix for the failure that actually stalls PRs here (#4149: 34 days,
zero human review; #4691: no reviewer). So: open that issue first; correct
`upstream/CLAIMS.md`'s upstream-baseline section (four of its statements are now wrong
or stale, including the CI claim and the cuDNN D256 framing); restate the novelty as
*paged page_size-1 + D256 + 24:4 GQA + bottom-right-causal prefix/extend +
per-row-Q/per-tile-K/per-tile-V scales + an inclusive gather/centre/rotate/quantize/pack
path* — every neighbour is dense, ragged or block-sparse, and none is paged; implement
Oracle A, which is the only piece of the designed correctness model that FlashInfer,
cuDNN and FlashAttention all have a direct analogue for and which we do not have at
all; add LSE; and port the adversarial suite into `tests/attention/test_<op>_sm120.py`
shape so that a maintainer's `/bot run` produces the hardware evidence for us. Most of
that is CPU-only work. The GPU session is needed only for regenerating goldens after
the LSE/API change, the comparability lane against a stock backend, and a CUDA-graph
lane that our manifest already claims (`cuda_graph: "supported"`) without any test
behind it. One reassessment cuts the other way and is worth stating plainly: the
*upstream* bar for quality evidence on lossy attention is far below the SageAttention
papers' bar — FlashInfer merged Sage-family quantization on "internal UT passed" plus
`atol=0.1` against `torch.randn`, and SGLang made Sage3 the sm_120 default on a single
wall-clock ratio — so our quality gap (G12) is a gap against our own standard and our
own production decision, not against what reviewers will ask for.

---

## 2. What upstream actually requires

### 2.1 Where a new operator lives, and what ships with it **[V]**

`CLAUDE.md` at repo root gives an 11-step checklist for "Adding a New Operation"
(verified in the clone; also referenced from `CONTRIBUTING.md` §"Code Contribution
Procedure"):

1. `include/flashinfer/<op>.cuh` — framework-agnostic, raw pointers, **no Torch headers**
2. `csrc/<op>.cu` — PyTorch tensor handling
3. `csrc/<op>_jit_binding.cu` — TVM-FFI bindings
4. (optional) Jinja template for type specialization
5. `flashinfer/jit/<op>.py` — JIT module generator
6. `flashinfer/<op>.py` — Python API, `@functools.cache`
7. `tests/` — tests
8. `flashinfer/aot.py` — AOT registration
9. `flashinfer/__init__.py` — export
10. **`flashinfer/trace/templates/<category>.py` — a `TraceTemplate`, wired via `@flashinfer_api(trace=...)`**
11. **an example call in `tests/trace/example.py`, regenerated `tests/trace/fi_trace_out/*.json`, committed**

Steps 10–11 are not optional in practice: `CLAUDE.md` states "Every public API
decorated with `@flashinfer_api` should also carry a `trace=` argument", and PR #4691
drew an automated finding for a missing `docs/api/*.rst` entry plus a CodeRabbit ask
to add the `trace=` template. `docs/api/attention.rst` is part of the plumbing set
(#4149 includes it; #4272 skipped it and got away with it).

Two API shapes are precedented for an SM120 low-precision attention op, and the
*standalone-function* one is the closer fit for us **[V]**:

- **Standalone function family** — `flashinfer/nvfp4_attention_sm120.py` exposes
  `nvfp4_attention_sm120_quantize_qkv(q, k, v, per_block_mean=True)` and
  `nvfp4_attention_sm120_fwd(q_fp4, k_fp4, v_fp4_t, q_scale, k_scale, v_scale_t,
  qk_correction, sm_scale=None, causal=False, per_block_mean=True, out=None,
  lse=None, out_dtype=torch.bfloat16, return_lse=True, unpadded_k_len=None)`,
  each decorated `@supported_compute_capability([120, 121])` and
  `@flashinfer_api(trace=...)`. This is a *preprocessing op + core op* split with
  destination passing — structurally identical to the split our architecture doc
  already specifies. #4149 and #4691 both copy this shape deliberately.
- **Wrapper plan/run** — `BatchPrefillWithPagedKVCacheWrapper(float_workspace_buffer,
  kv_layout="NHD", use_cuda_graph=False, ..., backend="auto")`, `.plan(qo_indptr,
  paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len, num_qo_heads,
  num_kv_heads, head_dim_qk, page_size, ..., q_data_type, kv_data_type, o_data_type,
  ..., disable_split_kv=False)`, `.run(q, paged_kv_cache, *, q_scale=None,
  k_scale=None, v_scale=None, out=None, lse=None, return_lse=False, enable_pdl=None,
  window_left=None, sinks=None, kv_cache_sf=None, ...)`. Note `q_scale`/`k_scale`/
  `v_scale` now accept `Union[float, torch.Tensor]` (device-resident scales), and
  NVFP4 paged KV is carried by `kv_cache_sf`. Validation raises `ValueError`.

**Distance from our API today [V, local]:** `src/attn_kernel_lab/csrc/fp8_prefill_attn.cu`
exposes one raw `PYBIND11_MODULE` symbol `fp8_prefill_attn(...)` taking 17 positional
tensors/flags, validated with `TORCH_CHECK`. There is no `plan`, no flat caller-owned
workspace (the Python `FP8PrefillWorkspace` owns named sub-buffers and grows them
itself), no `out=`/`lse=` destination arguments at the C++ boundary beyond `o`, no
`torch.library`/TVM-FFI registration, no trace template, no `@supported_compute_capability`.
`capability.check_supported` exists but is not on the call path.

### 2.2 Test conventions **[V]**

**Layout.** `tests/` is split by domain into 24 subdirectories (`attention/`,
`attn_scores/`, `gemm/`, `grouped_mm/`, `moe/`, `moe_ep/`, `comm/`, `norm/`,
`quantization/`, `utils/`, `jit/`, `trace/`, `trace_apply/`, `test_helpers/`, …) with
only eight loose files at the root. `tests/attention/` is **flat**: `__init__.py`,
`conftest.py`, and 75 modules, one file per op family — e.g.
`test_nvfp4_attention_sm120.py` (807 lines), `test_batch_prefill_kernels.py`
(2459 lines, ~6912 collected cases in its main grid), `test_fp8_prefill.py`,
`test_fmha_v2_prefill.py`, `test_blackwell_fmha.py`, `test_sparse_mla_sm120.py`,
`test_vsa_block_sparse_sm120.py`. `pytest.ini` sets `--import-mode=importlib` and
`norecursedirs = test_helpers`.

**Two packages, and they are not interchangeable.** `flashinfer/testing/` is a
**benchmarking** package — it exports `bench_gpu_time`,
`bench_gpu_time_with_{cuda_event,cupti,cudagraph}`, `attention_flops`,
`attention_tflops_per_sec`, `set_seed`, `sleep_after_kernel_run` and contains **no
assertion helpers and no attention reference**; only 2 of 77 attention test files
import it. The shared *correctness* helpers live in `tests/test_helpers/`
(`test_helpers.py`, `jit_utils.py`, `params.py`, `utils_fp4.py`,
`sink_attention_reference.py`, `rope_reference.py`, `alibi_reference.py`) and are
imported absolutely: `from tests.test_helpers.test_helpers import
assert_close_chunked, ref_single_prefill`.

**Neither conftest defines a fixture.** `tests/conftest.py` is hooks only: it
registers markers (`arch_blackwell`, `arch_hopper`, `long_running`, `solo`,
`shard_group`, `gpu_2/4/8`, `nvep`), auto-skips by GPU count and architecture, and —
importantly for anyone porting a heavy suite — **globally converts
`torch.cuda.OutOfMemoryError` and `flashinfer.jit.MissingJITCacheError` into
`pytest.skip`**. `tests/attention/conftest.py` is a JIT bulk-precompiler keyed by test
filename (`_PREBUILD_SPEC_COLLECTORS`), deriving the exact `JitSpec` set the surviving
parametrizations need, building them as one ninja graph, hardlinking the `.so`s into
the AOT dir, and printing a "collector drift" warning at session end for any module
that still compiled serially. **There is no global seed fixture** — 52 of 77 attention
files call `torch.manual_seed(0|42)` themselves.

**Hardware gating** is by helper, not by device-name string:
`flashinfer.utils.get_compute_capability`, `is_sm90a_supported`, `is_sm100a_supported`,
`is_sm100f_supported`, `is_sm110a_supported`, `is_sm120a_supported`,
`is_sm120f_supported`, `is_sm121a_supported`, `is_sm12x_supported`
(`flashinfer/utils.py` lines 290, 638–673). The SM120 test file's idiom:

Note `is_sm120a_supported` requires `major == 12 and minor == 0` with CUDA ≥ 12.8;
`is_sm121a_supported` is `minor == 1`, CUDA ≥ 12.9; `is_sm12x_supported` covers both.
Three gating shapes are used, in this order of frequency: (1) in-body `pytest.skip`
when the check depends on parametrized values; (2) module-level
`pytestmark = pytest.mark.skipif(not is_sm12x_supported(torch.device("cuda")), ...)`;
(3) named reusable marker constants. The SM120 file's idiom, called as the first line
of every test body:

```python
def _require_sm120():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda")
    if not (is_sm120a_supported(device) or is_sm121a_supported(device)):
        pytest.skip("SM120 or SM121 GPU is required")
```

On the API side the corresponding decorator is
`@supported_compute_capability([120, 121])` (used on the ops themselves, not in
tests). Every test in the SM120 file is also decorated `@torch.inference_mode()`.

**Shared reference.** `tests/test_helpers/test_helpers.py::ref_single_prefill(q, k, v,
causal=False)` is an **FP64** reference that already implements *our* mask:
`mask = k_pos[None, :] - (kv_len - qo_len) > q_pos[:, None]` — i.e. bottom-right
causal for `qo_len < kv_len` — returns `(out, lse)` with `-inf`/zero for fully masked
rows. We should use this rather than shipping our own `ref_attention`. Companion
helpers: `assert_close_chunked` (memory-frugal `assert_close` for multi-GiB operands,
chunking dim 0 by 4096 rows) and `assert_close_with_mismatch_tolerance(...,
max_mismatched_elements=N)`.

Preference order for the oracle, as practised in-tree: (1) FlashInfer against itself
(`single_prefill_with_kv_cache` per request, KV reconstructed from pages); (2) the
same kernel family in 16-bit run on the *dequantized* low-precision values — this is
the defining FP8 idiom in `test_fp8_prefill.py` and it removes any dependence on
scale plumbing; (3) `ref_single_prefill` (FP64); (4) a file-local FP32 einsum
`attention_ref`; (5) `F.scaled_dot_product_attention`. For GQA, the reference is fed
`repeat_interleave`d K/V.

**LSE base is not uniform, and this is a live upstream issue.** The FA2/CUTLASS
wrapper path is **base-2** — `test_blackwell_fmha.py`'s reference multiplies a natural-
log `logsumexp` by `math.log2(math.e)`, and `ref_single_prefill` returns
`row_max + log2(sum_exp)`. The SM120 standalone op is **natural log** —
`test_nvfp4_attention_sm120_lse` compares directly against `torch.logsumexp`. Open
issue #4485 ("Add caller selectable LSE base": *"Requiring the caller to infer the
convention from the selected backend is fragile"*) plus open PRs #4547/#4650/#4663 are
unifying this. Whichever we pick must be stated in the docstring and asserted.

**Tolerance conventions** are documented, not folklore:
`tests/trace/reference_correctness_standards.md` is a table of the accepted standard
per op. Representative rows: BF16 attention output `atol=1e-2, rtol=1e-2`; FP8 RoPE
quantization `atol=1e-2, rtol=2e-1`; FP4 round-trip mean relative error `< 0.5`;
NVFP4 packed-byte mismatch fraction `< 0.05`; BF16/FP8 GEMM `cos_sim > 0.99`; XQA FP8
"pass ratio ≥ 0.95 within `atol=0.05, rtol=0.05`". The programmatic default is
`flashinfer/trace/template.py::default_tolerances`: fp32 `1e-5`, fp16 `1e-3`, bf16
`1e-2`, `float8*` `1e-1`, `float4*`/fp4 `1.0`; and `standard_check(...,
max_mismatch_pct=0.0, min_cos_sim=1.0 - 1e-3)`. In-tree FP8 attention tests use
`torch.testing.assert_close(o_fp8.to(float16), o_ref, atol=1e-2, rtol=1e-2)` and, for
a calibration-scale path, `atol=1e-2, rtol=2e-1`
(`tests/attention/test_fp8_prefill.py`).

**The property set a low-precision SM120 attention test is expected to assert** — read
directly off `tests/attention/test_nvfp4_attention_sm120.py`, which is the file PR
#4502 grew by +410 lines and is the single best template for us:

| Test | What it pins |
|---|---|
| `test_..._accuracy` | 6 shapes (s128–s8192 × d64/d128 × causal) against an FP32 reference **that replays the op's own preprocessing** (K mean-centering + per-128-block Q mean + `qk_correction`), asserting `mean_abs_err <= 0.02..0.09` *and* `cos_sim >= 0.94..0.95`, plus `out.dtype`, `lse.dtype`, `lse.shape`, and `not isnan/isinf` on both |
| `test_..._gqa_matches_expanded_packed_oracle` | GQA result vs KV-expanded-to-Q-heads result **bit-exact** (`rtol=0, atol=0`) for both `out` and `lse`; also asserts every packed tensor's exact shape; also `cos_sim >= 0.94` vs `F.scaled_dot_product_attention` |
| `test_..._rejects_nonuniform_gqa_ratio` | `pytest.raises(ValueError, match="num_qo_heads")` |
| `test_..._rejects_kv_head_correction`, `test_..._rejects_expanded_correction` | `pytest.raises(ValueError, match="qk_correction")` — mis-shaped metadata must raise, not mis-address |
| `test_..._structured_q_correction` | structured (per-block-biased) Q, i.e. a regression test for a correction tensor the kernel once mis-addressed |
| `test_..._output_magnitude` | Q=K=0, V=1 ⇒ output must be 1.0 everywhere (`abs err <= 0.05`) and `lse == log(seq_len)` (`<= 0.02`). Explicit rationale in the docstring: "a mis-reduced row_sum shows up as a uniform scale error that cosine thresholds cannot see" |
| `test_..._lse` | LSE vs `torch.logsumexp` of the scaled, K-centred, masked scores: `mean diff <= 0.05`, `max <= 0.5` |
| `test_..._without_lse` | output-only and LSE-enabled specializations agree at `rtol=0, atol=5e-4` |
| `test_..._causal_mask_column_order` | V = identity ⇒ output row *is* the attention distribution; catches transposed/reversed column order |
| `test_..._rectangular` | 4 explicit `pytest.param` cases crossing `seq_len_q ≠ seq_len_k`, unaligned lengths (193/321, 385/193, 257/257), fp16/bf16 in, fp16/bf16 out, `provided_out` True/False, `return_lse` True/False |
| `test_..._rectangular_matches_square_padded_oracle` | rectangular result bit-exact vs a square-padded run |
| `test_..._unpadded_k_len_masks_tail_garbage` | fills the padded K/V/scale/correction tails with `0x77`/`1.0e4`, asserts `rtol=0, atol=0` vs clean; plus `pytest.raises(ValueError, match="unpadded_k_len")` for out-of-range lengths |
| `test_nvfp4_split_kv_gate_dtype_logic` | a **CPU-only** dtype-classification test guarding a plan-time gate |

Six more properties come from the neighbouring files and are worth copying wholesale:

- **CUDA graph with changed inputs.** `test_fmha_v2_prefill.py` (PR #4272): warm up on
  a side `torch.cuda.Stream()`, capture, **fill the output with NaN**, replay, assert
  `rtol=0, atol=0`. `test_batch_decode_kernels.py` replays 3–4 times after re-`plan()`
  with growing `kv_indptr`. `test_attention_ts_*.py` adds
  `test_cuda_graph_keeps_captured_plan_after_replan`.
- **Non-default stream semantics.** `test_attention_ts_context.py` runs inside
  `with torch.cuda.stream(worker_stream)`, records a `torch.cuda.Event`,
  `current_stream().wait_event(...)`, and verifies the result changed;
  `test_run_uses_callers_current_stream` is the minimal form.
- **Output-buffer aliasing rejection.** `pytest.raises(ValueError, match="out must not
  overlap q storage")`, plus `assert returned is shared_out` for `out=` identity.
- **Workspace out-of-bounds guard.** `test_trtllm_gen_attention_prefill.py` zeroes a
  1 MiB region of the workspace past the kernel's declared slab end and asserts it is
  still all-zero after the call — a direct regression test for "the context kernel
  wrote past the softmax slab".
- **Rejection tests as the reviewer's ask.** PR #3518 added
  `test_trtllm_fmha_v2_prefill_sm120_chunked_rejected`
  (`pytest.raises(ValueError, match="[Cc]hunked")`) purely because saltyminty asked
  "nor is it covered in the tests?"; #4272 asserts five distinct `ValueError` messages.
- **Cheap matrix tricks.** `flip_coin(*args)` (deterministic `hash(params) % 2`)
  alternates "caller supplies `out=`" vs "kernel allocates" without doubling the
  matrix; coupled dimensions go in a single tuple-parametrize
  (`(head_dim_qk, head_dim_vo, sm_scale)`); per-case thresholds ride in
  `pytest.param(..., cos_threshold, mean_abs_err_threshold, id="s256-d128-causal")`.
  Newer files (PrimTS) add a **relative-L2 bound**
  (`‖actual-expected‖ / ‖expected‖.clamp_min(1e-6) <= max_relative_l2`) alongside
  `assert_close`, and `assert_close_with_mismatch_tolerance(...,
  max_mismatched_elements=int(rate * total))` where a small mismatch rate is expected
  (10 % for NVFP4 KV, `1e-7` otherwise), with a cosine floor (`cos > 0.86`) as the
  structural check that "catches block-scale mismatches element-wise tolerances miss".

**Compute-sanitizer is not an upstream gate [V].** The only in-tree references are a
`_is_compute_sanitizer_active()` skip helper in `tests/attention/test_page.py` and
comments in `flashinfer/attention/prims_ts/context.py`. No CI job runs it.

### 2.3 Benchmark conventions **[V]**

`flashinfer/testing/utils.py` is the timing layer. `bench_gpu_time` signature and
defaults (line 1546):

```python
def bench_gpu_time(fn, dry_run_iters=None, repeat_iters=None,
                   dry_run_time_ms=25, repeat_time_ms=100,
                   l2_flush=None, l2_flush_size_mb=None, l2_flush_device=None,
                   sleep_after_run=False, enable_cupti=False,
                   use_cuda_graph=False, num_iters_within_graph=10,
                   input_args=(), input_kwargs=None,
                   cold_l2_cache=True, aggregate_op=max)
```

It returns a **list of per-iteration milliseconds** (callers take the median).
Backends in precedence order: CUPTI (`enable_cupti=True`), CUDA graphs
(`use_cuda_graph=True`), CUDA events (default). **`cold_l2_cache` defaults to `True`**
— L2 flush for CUPTI and plain events, rotating input buffers for events+graphs.
Siblings: `bench_gpu_time_with_cuda_event`, `bench_gpu_time_with_cupti`,
`bench_gpu_time_with_cudagraph`, plus `attention_flops`,
`attention_flops_with_actual_seq_lens`, `attention_tb_per_sec`, `set_seed`.

`benchmarks/bench_sm120_attention.py` is the model for a per-op bench, and it already
does two things our bench does not:

- an **`end_to_end` lane that includes `nvfp4_attention_sm120_quantize_qkv`** alongside
  the `attention_only` lane — i.e. upstream's own "inclusive" number;
- an **in-harness FP8 FMHAv2 control** (`fp8_attention_only`) and a reported
  `nvfp4_speedup_over_fp8` column.

Defaults: `--warmup 5`, `--repeat 20`, CUDA graph **on** for attention-only
(`--no-attention-cuda-graph` to disable), `cold_l2_cache = not attention_cuda_graph`,
`num_iters_within_graph=1`, `statistics.median`, CSV via `--save-results-to`. Its
FLOP formula is `factor * B * H * S * S * D` with `factor = 2 if causal else 4`.
`validate_config` rejects `head_dim not in (64, 128)` and `seq_len % 128 != 0`.

`benchmarks/flashinfer_benchmark.py` is the framework `CONTRIBUTING.md` names for
performance evidence ("report the observed performance improvement in the PR
description: before/after numbers from a reproducible benchmark (e.g.
`benchmarks/flashinfer_benchmark.py`), along with the GPU and problem sizes used").
Its relevant flags: `--backends` (space-separated, e.g. `fa2 fa3 cutlass cudnn
trtllm-gen cute-dsl prims-ts`), `--refcheck` / `--allow_output_mismatch` (cross-backend
output verification inside the same harness), `--num_iters` (default 30),
`--dry_run_iters` (default 5), `--no_cuda_graph`, `--use_cupti`,
`--generate_repro_command`, `--case_tag`, `--output_path` CSV, `--testlist`.
`benchmarks/routines/attention.py` is the shared attention driver.

### 2.4 CI reality — **this corrects `upstream/CLAIMS.md`** **[V]**

`upstream/CLAIMS.md` currently states: *"No public FlashInfer CI runner exercises
SM120, so contributor CI will not reproduce RTX PRO 6000 performance."* Half of that
is right and the important half is wrong.

- **Public GitHub Actions PR CI** (`.github/workflows/pr-test.yml`) uses self-hosted
  runners labelled `sm86` (A10G, 5 shards), `sm75` (T4), and `h100`. **No SM120 row.**
  `nightly-release.yml` also only reaches `sm86`. Correct as stated.
- **NVIDIA-internal GitLab CI**, triggered by commenting `/bot run` on the GitHub PR,
  has a documented matrix in `CONTRIBUTING.md` including **`unit_test_rtx_pro_6000` —
  "RTX PRO 6000 Blackwell" — cu129 and cu130**, alongside `unit_test_5090`,
  `unit_test_h100`, `unit_test_b200/b300/gb200/gb300`, `unit_test_spark`,
  `unit_test_thor`. The bot posts a pass/fail table back to the PR; on #4272 and
  #4703 the "RTX Pro 6000 Blackwell" row is visible and passing. `/bot run
  tests/<dir-or-file>` scopes it.
- **Neither CI runs performance.** The bot reports unit-test pass/fail only. Perf
  numbers in a PR body are always the contributor's own.
- **Public CI does not self-start.** It needs `@flashinfer-bot run` or the `run-ci`
  label, from someone who can label the PR or is in `ci-users`. In the precedent set,
  `run-ci` arrived 4 days late (#4502), 12 days late (#4272), 20 minutes (#3518,
  known contributor), and **never** (#4149 after 34 days, #4691 after 6 days).

**Consequence:** correctness tests written in FlashInfer's harness *are*
reproducible on our exact SKU by a maintainer. Performance is not, and never is for
anyone. So the correct split of effort is: put correctness in their harness; make
performance a self-contained, exactly-reproducible command with the environment
stated.

Branch protection requires **at least one approving review** to merge (visible on
#4013). Non-authorised users cannot even start CI: on #3877 the bot replied
*"@aws-jiadingg is not authorized to trigger this CI job. cc: @yzh119, @sricketts,
@yongwww."*

**What maintainers demonstrably accepted in lieu of in-CI performance numbers**
(across the precedent set): a named GPU with the timing method stated; a pasted
runnable repro command; the contributor's own pass counts; **published regressions
rather than hidden ones**; and explicitly labelled cross-run comparisons. The
"tested locally on hardware CI does not have" pattern is routine and merges:

- #3960 (MERGED) — *"Validated on NVIDIA GB10 (SM121, CUDA 13.0)"*; a maintainer then
  ran `/bot run tests/gdn`.
- #3739 (MERGED) — *"SM86 runtime correctness on RTX 3090: 14 large-head prefill/
  single-prefill tests passed in 734.19s"*, plus *"SM100/B200 results from @qsang-nv:
  `14 passed in 399.96s`"* — i.e. a **second contributor's** hardware counts.
- #3526 (MERGED) — *"Test results on an RTX 2080 Ti (sm_75)"*, with the author noting
  the sm_75 64 KiB smem cap "is not exercised by upstream CI"; reviewer wrote
  *"Approved conditional on CI"*.
- #4551 (OPEN) — *"Tested locally on an RTX 5070 Ti (SM120, CUDA 13.2, torch 2.8, JIT
  build from source)"*, with pass counts.
- #4013 (OPEN) — *"Confirmed by measurement on RTX 5080, RTX PRO 6000 Blackwell
  (CC 12.0) and GB10 / DGX Spark (CC 12.1)"*; the bot returned
  `[FAILED] Pipeline #63744341 — 15/16 executed test jobs passed`, the one failure
  labelled *"CI infrastructure failure — RTX Pro 6000 Blackwell / CUDA 12.9, Jobs:
  `unit_test_rtx-pro-6000-blackwell: [cu129]`"* — independent confirmation of the
  internal job name.
- #4577 (CLOSED) — the counter-example: *"Kept as a draft: none of the validation
  below has been run on real hardware."* Untested-on-hardware does not merge.

Locked clocks appear **exactly once** (#4502, 2430 MHz requested and monitored) and
`ncu` corroboration **exactly once** (#4149). Neither was demanded. #3518 merged in
three days with **zero** performance data. There is also evidence of an automated
PR-screening rubric (`flashinfer-pr-screen`) whose row **"C2.1 perf claim backed"**
was satisfied on #4310 by *"full before/after table (absolute µs + ratios), GPU named
(RTX PRO 6000 Blackwell SE)"* — a usable spec for our own table. The bar is honesty
and reproducibility, not rigour.

### 2.4b The written review rubric — read this before writing anything **[V]**

`CONTRIBUTING.md` links `docs/code_review_guidance_human.md` (and an agent-reviewer
twin, `docs/code_review_guidance.md`). It is short, explicit, and decides where our
effort should go. Its **focus** list:

- **Crash-prone coding style** — "OOB indexing, unchecked pointer/tensor math, integer
  overflow in offset/stride/size math (int32 vs int64), unvalidated shapes/dtypes/device
  assumptions, **under-allocated workspace/buffers on large problem sizes**, missing
  synchronization." Every one of those is live for us at 446 k positions.
- **Interface** — "Interfaces get **replicated**… review with 'how will this be copied?'
  in mind. Check argument order, **plan/run split**, wrapper patterns, and especially
  naming-convention adherence. Framework separation: no Torch headers under `include/`."
- **Testing surface** — "Does the change add/extend unit tests for the new behavior and
  edge cases? Are numerical references present (`--refcheck`)? Are architecture guards
  correct? Is the code testable at all?"
- **PR description hygiene** — keep the default template; "For a performance
  optimization, the PR description **must** report the observed performance improvement:
  before/after numbers from a reproducible benchmark… with the GPU and problem sizes
  used. **A perf claim without numbers is not reviewable — ask for them.**"
- **PR defendability** — "if the author, upon being asked, cannot walk through the
  rationale, we may reject the PR submission", and defendability matters most in
  "perf / kernel-selection logic, high-level interfaces, widely-used operations".

Its **non-focus** list is the more actionable half:

> **Kernel implementation details** — deprioritized for human reviewers… "Instead, rely
> on **passing unit tests and benchmarks + fuzz testing** as the backstop for kernel
> correctness."

A human reviewer will not line-audit our mainloop. The tests *are* the review. That is
the strongest possible argument for spending the budget on G2/G3 rather than on
commentary in the `.cu`.

Two more provisions we should use deliberately:

- **Design-doc-as-enforcement.** "Design principle → design doc as markdown → code owner
  / agent enforce. When a change embodies a design decision for durable code, capture the
  principle as a markdown design doc (see `docs/design_docs/`) so reviewers — human or
  agent — can enforce against a written rationale instead of re-deriving it each PR."
  `docs/OPERATOR_CONTRACT.md` is already exactly this artefact; it should ship as
  `docs/design_docs/<op>.md`. Note #4147 (the withdrawn predecessor of #4149) *had* a
  168-line design doc and #4149 dropped it — a plausible contributor to its silence.
- **Experimental track.** "FlashInfer does not gate PRs by size. Instead, an
  **experimental** track is being introduced: some PRs may be submitted on experimental
  terms — a separate lifecycle, workflow management, and quality bar — declared via a
  tracked issue." (Still TODO-marked in-tree, owner `@bkryu`.) A narrow, production-
  specific, 24:4-only operator is precisely the shape this track exists for. **Open a
  tracked issue first and ask whether the experimental track applies** — that is a
  cheaper first move than a PR, and it also solves the "who will review this" problem
  that stalled #4149 and #4691.
- **Fuzz testing** is named as part of the correctness backstop. `tests/test_helpers/
  fuzz_ledger.py` exists (a shared known-failure/quarantine ledger requiring every waiver
  to cite a `#NNNN` tracker) and its docstring names "future sampling / norm-RoPE /
  **attention** testers" — no attention fuzzer exists yet. Our 33-case adversarial suite
  plus hashed generators is the closest thing anyone has to one.

### 2.5 What the closest merged PRs actually shipped **[V]**

All rows verified via `gh pr view --repo flashinfer-ai/flashinfer`. Version context:
latest release **v0.6.18**, tagged 2026-08-29 (PyPI `flashinfer-python` 0.6.18 the same
day); `version.txt` on main reads `0.6.18`. **#4502 is not in v0.6.18** — its merge
commit `1ff1f79…` diverges from the `v0.6.18` tag, so the N64 score-slot work lands in
v0.6.19. #4272 is in v0.6.18, #3518 in v0.6.14, #3485 in v0.6.13. None of the five
carries a GitHub milestone; release mapping is by branch containment plus cherry-pick.

| PR | Title / state | Open→merge | Test files | Bench files | Perf evidence | Docs |
|---|---|---|---|---|---|---|
| **#3640** | *Add SM120 NVFP4 attention JIT path* — MERGED | 2026-06-15 → 06-17 (2 d) | `tests/attention/test_nvfp4_attention_sm120.py`, `tests/conftest.py` | `benchmarks/bench_nvfp4_attention_sm120.py` | — | — |
| **#3518** | *FMHAv2 on SM120 for head_dim 256/512 + sliding-window masks* — MERGED | 2026-06-05 → 06-09 (3 d) | `tests/attention/test_fmha_v2_prefill.py` (+121) | **none** | **none** | none |
| **#3485** | *Speed up FP8 KV-cache prefill (FA2 BatchPrefill) by repacking K/V to BF16 in smem* — MERGED (maintainer `bkryu`) | 2026-06-01 → 06-02 (**9 h 48 m**) | `test_batch_prefill_kernels.py`, `test_fp8_prefill.py` | `benchmarks/routines/attention.py` (workspace bump only) | 64 measured cells, RTX PRO 6000 + DGX Spark, TFLOP/s, **regressions published**; method only surfaced on request: `flashinfer_benchmark.py --refcheck --no_cuda_graph --use_cuda_events` | none |
| **#4272** | *Add SM120 FP8 FMHAv2 self-attention* — MERGED | 2026-07-30 → 08-14 (14 d) | `test_fmha_v2_prefill.py` (+419), `tests/trace/example.py`, `fi_trace_out/…json` | `benchmarks/bench_sm120_attention.py` (renamed from `bench_nvfp4_attention_sm120.py`) | RTX PRO 6000, CUDA-graph attention-only, 10 warmups / 100 iters; author published FP8 **slower** than NVFP4 in 3 of 4 cells and still merged | none |
| **#4502** | *perf(sm120): optimize NVFP4 attention with N64 score-slot reuse* — MERGED | 2026-08-13 → 08-25 (11 d) | `test_nvfp4_attention_sm120.py` (+410), `tests/trace/test_fi_trace.py` (+61) | `benchmarks/bench_sm120_attention.py` | RTX PRO 6000 Blackwell Server Edition (188 SMs), BF16 I/O, D=128, `return_lse=False`, **CUDA-graph attention-only, median of 100 iters after 10 warmups, quantization excluded, 2430 MHz requested and monitored**; baseline columns explicitly flagged as *not remeasured in the same run* | none |
| **#4149** | *feat(attention): MXFP8 ragged prefill attention for SM120/121* — **OPEN, 34 days, zero human review, no `run-ci`** | — | `test_mxfp8_attention_sm120.py`, `tests/trace/example.py`, 2 `fi_trace_out` fixtures | `benchmarks/bench_mxfp8_attention_sm120.py` (new) | RTX 5090; **min of 5 alternating A/B rounds × 100 iters**, cross-checked against `ncu --set full` `gpu__time_duration` (<3 % agreement); tensor-pipe-active % column; raw reps published externally | **`docs/api/attention.rst` (+15)** |
| **#4691** | *feat(sparse): add SM120 Sage (QK-INT8/PV-FP8 and ALL FP8) block-sparse attention* — **OPEN since 2026-08-23, zero human review** | — | `tests/attention/test_vsa_block_sparse_sm120_sage.py` | none | — | none (flagged by the docs bot) |

Three of these matter disproportionately.

**#4502 is the direct comparability target and the LSE precedent.** Its body's
measurement protocol is quotable and reproducible; its `return_lse` story is the
finding. The author made `return_lse=False` the default for speed; reviewer
`saltyminty` replied on `flashinfer/nvfp4_attention_sm120.py:327` that old callers
would break and **"I believe defaulting to `return_lse=True` is correct here"**, and
the default was restored. Today `nvfp4_attention_sm120_fwd(..., return_lse: bool =
True)` with the docstring "Defaults to `True` for compatibility with the legacy
`(out, lse)` return contract."

**#4149 is the structural template and the cautionary tale.** It deliberately follows
"the standalone-op structure of `nvfp4_attention_sm120`", ships the full plumbing set
including docs and trace fixtures, carries the most rigorous perf evidence of the
seven, returns `(out, lse)`, and has had **no human review in 34 days**. Contributor-
authored standalone SM120 attention ops are the slowest category in this repository.

**#4691 is the closest technical neighbour and the reason the novelty framing has to
change.** Verified body: it ports Block-Sparse-Attention's SM120 Sage kernel — "custom
INT8 warp MMA for QK, native FP8 warp MMA for PV, log2-domain online softmax" — with
"per-32-token-group Q, mean-centered per-K64-tile K, per-channel V with a 16-token
physical permutation for the PV MMA operand layout". That is our K-centering, our
group Q scales, and a V permutation chosen for the same reason as our SIGMA64. Its
own body names its limits: standalone (not wired into the wrapper), **MHA only**, and
**"It does not compute LSE"** — with LSE listed first among what wiring it up would
require.

### 2.6 LSE **[V]**

- Every paged/ragged prefill wrapper supports `return_lse` and an optional caller-
  supplied `lse=` buffer; `single_prefill_with_kv_cache_return_lse` exists as a
  convenience alias. Base convention differs per backend (see §2.2); tests pin it
  explicitly, and `test_trtllm_gen_attention_prefill.py` additionally pre-fills the
  caller's `lse` with NaN to catch a missed write, asserts `lse_out is provided_lse`,
  `dtype == float32`, the exact shape, and `torch.isfinite(...).all()`.
- Composition needs it: `flashinfer/cascade.py`'s `merge_state`/`merge_states` and
  `MultiLevelCascadeAttentionWrapper` consume `(V, S)` pairs, and split-KV merging is
  the same operation. There is an open issue asking for a caller-selectable LSE base
  (#4485) and open PRs unifying base semantics (#4547, #4650, #4663) — meaning LSE
  conventions are an *active* area, not a dormant one.
- The counter-evidence is weaker than it looks: several merged ops' *trace templates*
  declare no `lse` output (`trtllm_batch_context_trace`, `sparse_mla_sm120_paged_trace`,
  `xqa_batch_decode_trace`, the `trtllm_batch_decode_mla_*` family), and
  `trtllm_batch_context_with_kv_cache` has `return_lse: bool = False`. But those ops
  are *capable* of LSE; ours is not. The two data points about ops that structurally
  cannot produce LSE are #4691 (stalled, LSE named as the blocker) and open issue
  #3653 ("Support return_lse in the XQA MLA decode backend", labelled **P0**), whose
  body states plainly: *"LSE output is needed to merge partial attention across
  split-KV / context-parallel ranks… there is currently no LSE-capable MLA decode path
  on sm_120."*

`docs/OPERATOR_CONTRACT.md` §2 already records that v1 returns no LSE and §6.5 that
this "forecloses split-KV composition at this contract version". The evidence says
that is not a documented limitation upstream will accept for a *new* op; it is a
review-stopper.

### 2.7 flashinfer-bench, `fi_trace`, and `trace_apply` **[V]**

The Definition/Solution/Workload/Trace model is real and is now *inside* the main
repo, not only in the separate `flashinfer-bench` project:

- `flashinfer/trace/` provides `TraceTemplate`, `Var`, `Const`, `Tensor`, `Scalar`,
  `Solution`, `BuildSpec`, `default_check`, `standard_check`, `default_tolerances`.
  Each op's template declares `op_type`, `axes` (Const vs Var), `inputs`, `outputs`,
  `constraints`, `tags` (e.g. `["stage:prefill", "status:verified"]`) and a `reference`
  callable. `gqa_paged_prefill_trace` (`flashinfer/trace/templates/attention.py:1432`)
  is the stock paged-GQA definition: `op_type="gqa_paged"`, inputs `q [total_q,
  num_qo_heads, head_dim]`, `k_cache`/`v_cache [num_pages, page_size, num_kv_heads,
  head_dim]`, `qo_indptr`, `kv_indptr`, `kv_indices`, `sm_scale`, `return_lse`;
  outputs `output` **and optional `lse [total_q, num_qo_heads]` float32**. This
  confirms the architecture doc's statement that the stock definition returns LSE and
  that an output-only v1 needs its own definition.
- `flashinfer/fi_trace.py` generates **flashinfer-bench-compatible definition JSON**
  from any `@flashinfer_api(trace=...)` function, either by `FLASHINFER_TRACE_DUMP=1`
  + `FLASHINFER_TRACE_DUMP_DIR=…` auto-dump or by `fn.fi_trace(**kwargs)`.
- `flashinfer/trace_apply/` does **runtime kernel substitution**: `enable_apply({name:
  callable_or_Solution})`, or `FLASHINFER_TRACE_APPLY=1` + `FLASHINFER_TRACE_APPLY_PATH`.
  This is a first-class path to demonstrate our kernel against a stock definition
  *without* an upstream PR.
- `flashinfer-bench` (separate repo): pydantic Definition/Solution/Workload/Trace
  schemas laid out as `definitions/*.json`, `solutions/*.json`, `workloads/*.jsonl`,
  `traces/*.jsonl`, `blob/workloads/*.safetensors`; run as
  `flashinfer-bench run --local flashinfer-trace`. The HuggingFace dataset
  `flashinfer-ai/flashinfer-trace` carries **190 definitions, 22 of them
  `gqa_paged_prefill_causal_*`**, and their required output #2 is a **float32,
  log2-based LSE of shape `[total_q, num_qo_heads]`** — a third independent
  confirmation that the stock paged-GQA prefill contract includes LSE. Default
  evaluator tolerances are `rtol = atol = 1e-2`. The public surface is a
  **leaderboard** (bench.flashinfer.ai) ranking *LLM authors* over ~660 workloads
  with no submit-a-solution button, plus an MLSys 2026 contest driven by
  `flashinfer-bench-starter-kit` (arXiv:2601.00227). Reported state: v0.1.2, PyPI
  2026-02-13, "Beta", ~276 stars, last commit ~2026-05-01 **[V via subagent; I
  independently confirmed the README, the HF dataset link and the PyPI package, not
  the commit date]**. The one main-repo integration PR, **#2151** "[Flashinfer-Bench
  integration] HF end-to-end inference", is still a **draft**.

**Verdict on the architecture doc's question:** flashinfer-bench is *not* used as PR
evidence — none of the precedent PRs cites a flashinfer-bench trace or leaderboard
result, its only main-repo integration PR is a draft, and the repo itself has not
moved since roughly May while `flashinfer/trace` inside the main repo is actively
developed. What upstream requires from a PR is the **in-repo** trace template plus
regenerated `tests/trace/fi_trace_out/*.json`. flashinfer-bench is the surrounding
ecosystem (dataset, leaderboard, MLSys contest) and `trace_apply` is a
deployment/eval mechanism. **Treat in-repo `TraceTemplate` as mandatory-for-merge and
flashinfer-bench as optional reach** — and note that the architecture doc's extensive
flashinfer-bench guardrail list (pinning the executor commit, the trace-config
self-test, `required_matched_ratio` semantics, the D256/H24/Hkv4/page-1 preset) is
work that buys nothing toward a merge. Deprioritise it.

### 2.8 Contrast cases: cuDNN frontend, FlashAttention, SageAttention **[V]**

*(cuDNN frontend, FlashAttention and SageAttention detail is summarised here at lower
confidence; see the marks.)*

- **cuDNN frontend [V].** All three PR numbers in `upstream/CLAIMS.md` check out.
  **#509** "Add SM120 per-tensor FP8 (e4m3) SDPA-forward engine" (merged 2026-08-10)
  adds `sdpa_fwd_prefill_sm120_fp8`, opt-in behind
  `CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1`, lowering to
  `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`, with `descale_q*descale_k`
  folded into the softmax scale and `descale_v*scale_o` into an epilogue scalar.
  **#595** (merged 2026-08-17) widens it: e5m2 inputs, independent output dtype,
  attention sinks, per-batch `seq_len_q` trim (padded rows write `O := 0, LSE := -inf`),
  right-band widening, single-KV-tile shapes, THD token-major Stats, `dense_flex`
  layouts — **each with a declared `Capabilities` bit and tests**. **#768** (merged
  2026-08-26) is the split-KV port, touching `prefill_fp8_sm120.py` and
  `test_split_kv_heuristic.py`. Evidence lives in `test/python/sdpa/frost/`
  (`test_sdpa_fwd_fp8_sm120.py`, `test_sm120_tile_rule.py`,
  `test_split_kv_heuristic.py`).
  Two corrections and one differentiator follow. **Correction 1:** #595 states "Native
  tiles stay any multiple of 32 **up to 256**" and actual head dims any multiple of 16
  — so cuDNN's SM120 FP8 SDPA *does* reach head dim 256; "no D256 low-precision
  attention on SM120" is no longer accurate as a blanket statement. What remains true
  is that it is **per-tensor descale** (coarser than our per-row Q / per-64-tile K /
  per-tile V scales), **dense BSHD/THD, not paged**, and FP8-QK rather than INT8-QK.
  **Correction 2:** the "Capabilities bit + test per bit" convention is a better model
  for our declared support surface than anything in FlashInfer — it is exactly what
  `src/attn_kernel_lab/capability.py` was designed for and is not yet wired to.
  **Differentiator, now independently corroborated:** #509 describes needing
  `pack_f8x2_pairs` + two `shfl.sync.idx` + one `prmt.b32` to build each PV A-operand
  from the QK C-fragment, because "the QK C-fragment owns columns `2*(t%4)+{0,1}`
  while the k32 A-fragment wants four consecutive bytes". Our claim K2 is that a fixed
  V-row permutation (SIGMA64) removes that traffic entirely. An independent
  implementation choosing the shuffle route is the strongest external evidence we have
  that K2 is a real contribution — and it is checkable by a reviewer without a GPU.

  Their *evidence practice* is the best of the three contrast cases and several pieces
  are directly portable:
  - **Test-to-kernel line ratio ≈ 1:2.** #509 added `test_sdpa_fwd_fp8_sm120.py`
    (+582) and `test_sm120_tile_rule.py` (+113) beside a +1607-line kernel; #595 was
    +562/−124 on the same test file. It is now 1310 lines, ~45 test functions.
  - **A tolerance derived from the output dtype**, not hard-coded:
    `tol_o = max(5e-2, 3 * floor)` where
    `floor = (o_ref - o_ref.to(o_dtype).float()).abs().max()`. The same idea as FA3's
    `2 * (ref + 0.3 - 0.3 - ref).abs().max()`, arrived at independently.
  - **LSE asserted including its `-inf` pattern**:
    `assert torch.equal(torch.isfinite(lse), finite)` before any magnitude check —
    catching masking-structure bugs a tolerance cannot see. `Amax_O` is a checked
    output too (`abs(amax_o - amax_o_ref) <= 0.03`).
  - **A tiled online-softmax FP32 reference** (`test/python/sdpa/fp8_ref.py::compute_ref`
    returning `(o_quant, stats, o_amax)`), i.e. their Oracle A — a reference that
    replays the kernel's own algorithm, not a naive one. This is precisely what our
    Oracle A must be.
  - **Negative testing as a first-class half of the suite.** `_fp8_graph_offers_sm120(...)`
    builds a graph and asserts whether the engine *claims* it — "A capability rejection
    is the point, so nothing is executed" — swept over off-granule head dims
    `[24, 144, 288]` and unsupported MLA shapes. The house rule in `python/cudnn/AGENTS.md`
    is quotable: **"Silent wrong results are the worst failure mode; a silent slow path
    is the second worst."** A reviewer objection on #595 makes the design rule explicit:
    a per-kernel property must be a `Capabilities` bit, not an adapter class-name check.
  - **The performance heuristic itself is landed as device-free tests.**
    `test_sm120_tile_rule.py` asserts the tile-selection arithmetic
    (`assert tiles(4096, causal=True)[0] == 64`) with `SMS = 188  # RTX PRO 6000
    Blackwell, the part the rule was measured on`, and the docstring states the point:
    *"a change here is a claim about the kernels, not about the hardware."* This is the
    only durable form of performance evidence in either project.
  - **Perf-body discipline worth imitating**: #509 carried a three-baseline table
    (native FP8 fprop, the BF16 SM120 cell, a sibling internal FP8 kernel) measured
    with **CUPTI kernel time, a 6-second device warm-up and interleaved repeats**; a
    **retracted claim** ("those were the pre-shfl kernel on a different part, and the
    ratio does not transfer… the claim was wrong and has been removed"); a **"Measured
    and rejected"** section listing four optimizations that did not pay; and a stated
    **noise floor** (~1 % run-to-run at s ≥ 2048, **12 % at s = 512** until the whole
    device was warmed — "larger than every effect above"). Our own bench's warm-up of
    3 iterations should be read against that last number.
  - **The cautionary half:** none of that shipped. #509's merged file list contains no
    docs and no benchmark; the design doc carrying the perf evidence was cut before
    merge after a reviewer asked whether design decisions belonged in an API doc. The
    evidence now exists only in a PR description. If we want our measurement protocol
    to survive, it has to be a *test* (their tile-rule file) or a *design doc under the
    repo's own conventions*, not a PR body.
  - Helpers we would reuse: `test/python/sdpa/helpers.py::compare_tensors(actual,
    expected, rtol=1e-2, atol=1e-2, num_diffs=10)` and
    `time_execution_cupti(num_warmup=3, num_trials=10, num_drop=5, flush_l2=True)`.
    Gating is centralised in `frost_test_utils.py` (`requires_blackwell_geforce` =
    `120 <= SM <= 129`, `requires_dsl` checking a *version*), created after five files
    each carried their own copy pinned to exactly `(10, 0)`. Tests are marked `L0`
    (CI smoke, must stay fast) vs `L1` (deep sweeps). Their PR template mandates a
    `## Testing` section: *"List exact commands and results. If something was not
    tested, explain why."*
- **FlashAttention [V].** `Dao-AILab/flash-attention` `tests/test_flash_attn.py`
  (2688 lines) uses a **ratio** assertion, appearing ~20 times, with the comment
  *"Check that FlashAttention's numerical error is at most twice the numerical error
  of a Pytorch implementation"*:

  ```python
  assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item()
  ```

  where `out_ref = attention_ref(..., upcast=True)` (FP32) and
  `out_pt = attention_ref(..., upcast=False, reorder_ops=True)` — i.e. the *same*
  reference run in the target dtype with reordered ops, as a same-process control.
  The gradient assertions use the same form with `+ 1e-5`; dropout-probability and
  attention-matrix assertions use it too.

  This is strictly better than an absolute envelope for our **Oracle B**, and we
  already have the ingredients: the integration README records a measured 0.45 %
  bf16-SDPA implementation-swap yardstick, but `tests/kernel/test_kernel_vs_sdpa.py`
  hard-codes `rr.mean() < 0.06 / rr.max() < 0.15` instead of computing the control in
  the same process. The direct port is: compute `out_ref` in FP32 and `out_pt` as a
  BF16 SDPA run of the *same* mask, and assert a ratio bound calibrated once. Note
  the honest complication: our operator is far lossier than FP16 attention, so the
  multiplier will not be 2 — it should be measured and then frozen, which is exactly
  what the golden-regeneration policy is for.
  Three more FlashAttention details are worth copying:
  - **The multiplier is never one number.** 2× forward; 3× backward; 3× under
    softcap; `mult = 4` for FP8 output and `mult_mean = 3` on the *mean* error;
    `mult = 2 if not alibi else 8` for split-KV backward; 5× with `+1e-3` for the
    overflow test. Anyone quoting "within 2×" as a blanket bar is quoting something
    FlashAttention itself does not do.
  - **FA3/FA4 derive the additive epsilon from the data**, which removes the last
    hard-coded constant: `fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max()`
    — an add/sub round trip in the reference dtype, i.e. the worst-case quantization
    step over that tensor — factored in FA4 into
    `check_tensor_vs_ref(name, actual, ref, pt, rtol=2, atol=None)`.
  - **Paged-KV test hygiene we should copy verbatim:** a **3× over-allocated** block
    pool, a **`torch.randperm`-shuffled** page table, and a contiguous reference cache
    **gathered through the same table**, so reference and kernel see identical values
    by construction. Page sizes swept `[None, 1, 4, 128]`. Varlen forces zero-length
    sequences every fifth batch element and always the last. Head dims include
    non-multiples of 8 (40, 59, 111). Determinism is its own test class at 50/250/1000
    iterations, one of which allocates a 70 GiB dummy tensor "to simulate under memory
    load". There are **no CUDA-graph tests anywhere** in FlashAttention, and **no perf
    assertions** — its CI benchmark step only has to exit 0.

- **SageAttention family [V].** `upstream/CLAIMS.md` already concedes "**Not** original
  in technique." The papers' evidence framework is worth naming precisely, because it
  is the standard our *own* claims should meet even though upstream will not ask:
  per-layer **cosine similarity, relative L1 (`Σ|O−O′| / Σ|O|`) and RMSE**, reported
  as **average *and worst over all layers*, on real Q/K/V captured from model layers**
  rather than synthetic Gaussians. The spread is the point: INT8 PV averages **99.70 %
  CosSim but its worst real layer is 56.40 %** — which is exactly why SageAttention v1
  kept PV in FP16. Three consequences for us:

  1. **Perplexity is a null detector for this failure mode.** Removing K-smoothing
     moves WikiText perplexity 5.823 → 5.826 (0.05 %) while one generative-task
     score moves 163.33 → 267.06 and another task score halves. Quantized FA3 on the same table has
     fine perplexity and FID 394. At long context, FP8 FA3 matches full precision on
     PassKey and Number retrieval while **Retr.KV collapses 7.0 → 0.4**. A text-metric
     quality lane would certify a broken kernel. Our own `CLAIMS.md` already says the
     logprob delta "cannot rank backends"; this is the published version of the same
     finding.
  2. **Our Hadamard-rotation claim is in direct tension with SageAttention2's measured
     result.** Their smoothing ablation on sampled real-model layers puts Hadamard rotation
     ("HadmdAttn", i.e. QuaRot) at **79.77 % average / 4.85 % worst CosSim versus
     80.04 % / 4.83 % for no smoothing at all** — measured as *useless* on that model
     family — and they chose Smooth Q + Smooth K (99.46 % / 96.71 % worst) instead. Our
     `test_hadamard_rotation_exact_and_helpful` measures rotation helping by ≥40 % on
     *synthetic* within-row outliers, while private prose reports "worst-head
     per-layer error 2.2 % → 0.55 %" on real keys. That private measurement is the
     one that matters and is not published here. **Promote it to a
     reproducible artefact or drop the claim** — a reviewer who knows the literature
     will treat an unqualified "Hadamard rotation helps" as contradicted.
  3. **SpargeAttn's hyperparameter protocol is the transferable methodology**: choose
     per-layer thresholds by grid search maximising the objective subject to a
     **relative-L1 bound measured over five different model inputs**, then publish the
     per-model thresholds. That is the shape our `vs_max/16` P-underflow floor and any
     future per-head dial should be justified in.

  **But the upstream bar is far below the papers' bar.** Verified adoption history:
  **vLLM never landed SageAttention** — PR #10532 died when a maintainer asked "can you
  run some benchmarks (e.g. benchmark throughput for Llama 8B)?" and no end-to-end
  numbers were ever posted; the feature issue was closed as not planned. **SGLang
  landed it only in the diffusion path** — #14878 merged on CI-green plus a code-owner
  review with *no accuracy tables at all*, and #15382 made Sage3 the **default** on
  sm_120 on a **speed number alone** (Wan2.1 1.74× wall-clock), deferring accuracy by
  quoting the upstream repo's own caveat. **FlashInfer landed it via TensorRT-LLM**:
  #3982 ("Internal UT related to this PR has all passed") imports `sageQuant.cu`, and
  #4654 adds per-head per-channel K mean-subtraction claiming "improved quantization
  accuracy" with **no RMSE, no cosine similarity, no tolerance value and no citation**.
  The only numeric evidence anywhere in that lineage is in
  `tests/attention/test_trtllm_ragged_dit.py`: the Sage path is compared to a BF16
  reference at `assert_close(atol=0.1, rtol=0.1)` — *looser* than the `0.05/0.05` used
  for the non-Sage FP8 paths in the same file — plus per-element dequantization error
  bounds `{"q": 0.04, "k": 0.04, "v": 0.17}` on `torch.randn` inputs.

  **Net:** our existing evidence already exceeds what FlashInfer accepted for
  SageAttention-family quantization. The gap that matters is not the PR's; it is ours.
  And note the shape of the question that actually killed the vLLM attempt: **an
  end-to-end throughput number on a real model** — which is precisely the 209.5 s →
  146.5 s (1.43×) 446 k anchor we already have and upstream contributors mostly do not.

---

## 3. Gap table

Class: **BLOCKING** = a credible PR would likely stall or be rejected without it;
**STRENGTHENING** = materially reduces review rounds; **OPTIONAL** = low leverage.
Status: *missing* / *designed-but-unbuilt* (with the architecture-doc citation) /
*built-but-divergent*.

| # | Gap | Class | Status | Concrete action | Lands in |
|---|---|---|---|---|---|
| G1 | **No LSE output.** Contract §2 declares `LSE: not returned`; manifest `contract.returns_lse: false`. #4502 shows a maintainer reversing an author's `return_lse=False` *default*; #4691 stalls with "does not compute LSE"; #3653 is P0 for the same reason. | **BLOCKING** | missing (contract-level; §6.5 records it as a limitation, not a plan) | Add an LSE writeback to the epilogue — the online softmax already carries `m` and `l`, so this is `m*log2e + log2(l)` (base-2, matching the FA2/CUTLASS wrapper convention and `ref_single_prefill`) or its natural-log form; **state the base in the docstring** and follow issue #4485. Expose `return_lse: bool = True` and an optional `lse=` buffer, add `test_..._lse` (vs `torch.logsumexp` of the masked, K-centred scores) and `test_..._without_lse` at `rtol=0, atol=5e-4`. Bump `package_api_version`; regenerate goldens **in their own commit** per the golden policy. | `src/attn_kernel_lab/csrc/fp8_prefill_attn.cu`, `docs/OPERATOR_CONTRACT.md` §2/§6, `promotion/schema` `returns_lse`, new upstream test file |
| G2 | **Oracle A (exact quantized-contract reference) does not exist.** `docs/OPERATOR_CONTRACT.md` is Draft precisely because of this; the correctness attestation records only the two implementation-comparison suites. Every current test compares an implementation to FP32 attention or to itself. | **BLOCKING** | designed-but-unbuilt — architecture doc §"Correctness model / Oracle A"; contract §4 enumerates exactly what it must pin | Implement a pure-PyTorch FP64/FP32 reference of contract §3 (Hadamard, per-row Q amax + `sm_scale` fold, K channel mean, V mean + per-tile amax + `vs_max/16` floor, SIGMA64 permutation, packed layouts), validating **intermediate boundaries** (scales, `q8`/`k8`/`vt8` bytes, means) not just the output. This is the analogue of `_reference_attention` + `_preprocess_qkv_ref` in `test_nvfp4_attention_sm120.py`, which replays the op's own preprocessing. | new `src/attn_kernel_lab/oracle_a.py` + `tests/kernel/test_oracle_a.py` |
| G3 | **Tests are not in FlashInfer's harness or shape.** Ours are three standalone pytest files that JIT-build via `torch.utils.cpp_extension.load` and import `quant.py` by `sys.path` hack. Given that `unit_test_rtx_pro_6000` exists in the internal CI, this is the difference between "maintainer can run our evidence on our SKU" and "cannot". | **BLOCKING** | built-but-divergent | Produce `tests/attention/test_<op>_sm120.py` in upstream shape: `_require_sm120()` via `is_sm120a_supported`, `ref_single_prefill` from `tests/test_helpers`, `torch.testing.assert_close` / `assert_close_chunked`, `pytest.param(..., id=...)` matrices, and the property set in §2.2. Keep our adversarial suite as the *content*; change the *form*. | new `upstream/pr-draft/tests/attention/…`; keep `tests/kernel/` as the lab lane |
| G4 | **No `ValueError` rejection tests / typed capability surface is off the call path.** `capability.check_supported` exists but the kernel validates with `TORCH_CHECK`, which raises `RuntimeError`. #3518 gained a rejection test purely because a reviewer asked; #4272 asserts five distinct `ValueError` messages. | **BLOCKING** | built-but-divergent (`src/attn_kernel_lab/capability.py` exists; unused) | Route every declared-surface check through `CapabilityError(ValueError)` **before** the kernel launch, and add `pytest.raises(ValueError, match=...)` cases for: head_dim ≠ 256, page_size > 1, non-24:4 (or non-divisible) GQA, FP8/FP4 KV pool, non-EXTEND mode, mis-shaped scale/mean metadata. | `src/attn_kernel_lab/capability.py`, a thin Python `run()` wrapper, new tests |
| G4b | **No stream-semantics, output-aliasing, or workspace-guard tests.** Contract §5 states `run` enqueues on the caller's current stream and does not implicitly synchronise — nothing tests it. Upstream tests all three (`test_run_uses_callers_current_stream`; `pytest.raises(ValueError, match="out must not overlap q storage")`; zero a 1 MiB region past the declared workspace slab and assert it is still zero). | **STRENGTHENING** | missing (contract §5 states the semantics; no test) | Add the three tests. The workspace-guard test is the highest value for us because our workspace is a set of monotonically grown named buffers with no declared end. | new upstream test file; `src/attn_kernel_lab/quant.py` |
| G5 | **`cuda_graph: "supported"` is claimed with zero evidence.** Declared in `capability.V1_CAPABILITY`, in `tools/make_candidate_records.py`, and in the published `artifact-manifest.json`; the same manifest's `limitations` says the CUDA-graph lane was not run. | **BLOCKING** (a false capability claim in an immutable record is worse than a missing one) | designed-but-unbuilt — architecture doc §"Timing protocol / Measurement modes" mode 3, §"Test distributions" ("eager and CUDA-graph replay with pointer-stable but value-changing inputs") | Either add the #4272-style test (warm on a side stream, capture, **NaN-poison the output**, replay with changed values, `rtol=0, atol=0`) or change the declaration to `eager_only` until it exists. Note the workspace design is the risk: `FP8PrefillWorkspace.get()` reallocates on growth, which is illegal mid-graph. | `src/attn_kernel_lab/quant.py` (capacity-stable workspace), new test, manifest field |
| G6 | **No same-harness external control.** `bench/candidate_bench.py` measures only our three lanes. The architecture doc lists this as existing benchmark debt. Upstream's own `bench_sm120_attention.py` carries an in-harness FP8 control and a speedup column; `flashinfer_benchmark.py` has `--backends` + `--refcheck`. | **BLOCKING** for any speedup claim (a number with no denominator is not evidence) | designed-but-unbuilt — architecture doc §"Benchmarking stack" row *Upstream comparison*, and §"Current benchmark debt" ("no stock FlashInfer/cuDNN/FlashAttention control in the same harness"). The attestation schema already has `lane: upstream_comparison`. | Add a `--control flashinfer-bf16` lane to `candidate_bench.py` driving `BatchPrefillWithPagedKVCacheWrapper` at D256/page_size 1/24:4 on the identical hashed cases, same process, interleaved A-B-B-A. | `bench/candidate_bench.py` |
| G7 | **Timing protocol is not comparable to #4502.** Ours: `cuda_events`, warm-L2, eager, warmup 3 / iters 10. #4502: CUDA graph, attention-only, median of 100 after 10 warmups, quantization excluded, 2430 MHz requested and monitored. Upstream's `bench_gpu_time` defaults to `cold_l2_cache=True`. | **STRENGTHENING** (blocking only for a head-to-head claim) | designed-but-unbuilt — architecture doc §"Timing protocol" names this exact lane: *"Keep one exact secondary comparability lane matching the merged NVFP4 SM120 evidence in FlashInfer PR #4502… explicit 10 warm-ups and 100 repeats, with the PR's requested 2430 MHz"* | Add a fourth bench lane `upstream_comparability`: `use_cuda_graph=True`, `num_iters_within_graph=1`, `dry_run_iters=10`, `repeat_iters=100`, prequantized inputs, preallocated `out`, clocks requested and observed recorded. Record `timing_backend`/`l2_policy`/`graph_mode` as we already do. | `bench/candidate_bench.py`, attestation `measurement.lane` |
| G8 | **No `benchmarks/bench_*.py` in upstream form.** `CONTRIBUTING.md` asks for before/after from a reproducible benchmark; every merged perf PR shipped or edited one. | **STRENGTHENING** | missing | Write `bench_<op>_sm120.py` modelled on `bench_sm120_attention.py`: argparse shape lists, `flashinfer.testing.bench_gpu_time`, median, CSV, `attention_only` + `end_to_end` lanes, an FP8/BF16 control column. | `upstream/pr-draft/benchmarks/` |
| G9 | **No `TraceTemplate` + `fi_trace_out` fixtures.** `CLAUDE.md` steps 10–11; #4149 and #4272 both ship them; #4691 was flagged for omitting one. | **STRENGTHENING** (mechanically required for merge, cheap to do) | missing | Copy the shape of `flashinfer/trace/templates/nvfp4_attention_sm120.py`, which already models a **preprocessing op + core op pair**: `Const` axis `head_dim`, `Var` axes for lengths and packed/scale extents, explicit packed-layout outputs with `dtype="uint8"`/`"float8_e4m3fn"`, and a `constraints` list expressing the padding and packing algebra (`head_dim == 2 * packed_head_dim`, `padded_kv_len % 128 == 0`, …). Ours becomes `head_dim == 256`, `page_size == 1`, `num_qo_heads == 6 * num_kv_heads`, `npad % 64 == 0`, plus the SIGMA64 tile-major `vt8` extent. Add the `tests/trace/example.py` call and commit regenerated JSON. **Declare output dtypes explicitly** — CodeRabbit flagged #4149 for letting `out`/`lse` inherit the input's fp8 dtype. | `upstream/pr-draft/flashinfer/trace/templates/` |
| G10 | **API shape: no plan/run split, no caller-owned flat workspace, no destination-passing `out=`/`lse=` at the Python boundary, no TVM-FFI binding, raw pybind11 with 17 positional args.** | **STRENGTHENING** (the standalone-function precedent softens this; a wrapper would harden it) | designed-but-unbuilt — architecture doc §"Planning, workspace, stream, and failure semantics" specifies `plan`/`run`, caller-owned workspace and output, stable addresses across replay, no implicit sync | Follow the `nvfp4_attention_sm120` shape first: `<op>_quantize_kv(...)`, `<op>_quantize_q(...)`, `<op>_fwd(..., out=None, lse=None, return_lse=True, sm_scale=None)` with `@supported_compute_capability([120, 121])`. Add `plan()` only if targeting the paged wrapper. Replace the growing named-buffer workspace with a caller-supplied `float_workspace_buffer` + a `plan`-returned byte requirement. | `src/attn_kernel_lab/` public API, `docs/OPERATOR_CONTRACT.md` §5 |
| G11 | **Generalization matrix is one production point.** Goldens use H=8/KVH=2 plus one H=24/KVH=4 shape; the bench uses only 24:4. No non-24:4 GQA sweep, no explicit rejection cases, no page_size>1 rejection test, no `qo_len << kv_len` prefix-cache geometry. #3684 found real SM120 low-precision corruption exactly at `qo_len << kv_len` with split-KV. | **STRENGTHENING** (BLOCKING if the PR claims generic divisible GQA) | designed-but-unbuilt — architecture doc §"Generalization matrix" lists every one of these bullets | Add a parametrized upstream-shape matrix: GQA ∈ {24:4 (required), 8:2, 8:1 MQA, 16:4}, `qo_len`/`kv_len` ∈ {aligned, unaligned, `qo_len << kv_len`}, page maps ∈ {contiguous, shuffled, fragmented}, plus explicit `raises` cases for D64/D128 and page_size 16. | new upstream test file; `workloads/profiles/` for a generalization profile |
| G12 | **Quality evidence cannot support a lossy-attention claim.** `CLAIMS.md` already says so: the private downstream task evidence is only a smoke probe, and the ~0.17 logprob delta saturates at implementation-swap sensitivity and cannot rank backends. Restricted inputs and private activations can never appear in a public PR (`CLAIMS.md` §"Evidence hygiene"). **Reassessed downward for upstream:** FlashInfer merged SageAttention-family quantization (#3982, #4654) on "internal UT passed" and `atol=0.1` on `torch.randn`; SGLang made Sage3 the sm_120 default on a wall-clock number with zero measured accuracy. Upstream will not ask. | **OPTIONAL** for the PR itself; **STRENGTHENING** for the PR body; **BLOCKING** for any "quality-neutral" wording and for our own production decision | missing | Build a public, redistributable quality lane using the SageAttention metric set — per-layer **cosine similarity, relative L1, RMSE reported as average *and worst layer*, on real captured activations** — plus a task metric that is not perplexity (long-context retrieval / NIAH), since perplexity is a demonstrated null detector for this failure mode. Until it exists, claim *kernel* numerics only and say so. | new `upstream/quality/` lane; `CLAIMS.md` "Explicitly NOT claimed" already carries the disclaimer — keep it |
| G18 | **The Hadamard-rotation claim rests on synthetic data and is contradicted in the literature.** `tests/kernel/test_kernel_vs_sdpa.py::test_hadamard_rotation_exact_and_helpful` measures a ≥40 % improvement on *synthetically* boosted within-row outliers. SageAttention2 measured Hadamard rotation (QuaRot) at **79.77 % avg / 4.85 % worst CosSim vs 80.04 % / 4.83 % for no smoothing** on sampled real-model layers — i.e. no benefit — and chose Smooth Q + Smooth K instead. Our real-tensor counter-evidence ("worst-head per-layer error 2.2 % → 0.55 %") exists only as private prose. | **STRENGTHENING** (BLOCKING if the PR asserts the rotation is beneficial) | built-but-divergent (test exists; evidence class is wrong) | Reproduce the 2.2 % → 0.55 % measurement as a committed artefact on a redistributable activation fixture with the SageAttention metric set, or restate the claim narrowly as "exact under a shared orthonormal rotation; measured to help on this model's key distribution", and say plainly that it is model-dependent. | `tests/kernel/`, `upstream/CLAIMS.md`, `docs/OPERATOR_CONTRACT.md` §3.1 |
| G13 | **Repro presentation.** Manifests and hashed workloads are excellent internally but are not what a reviewer consumes. Precedent shows the accepted currency is a pasted command + named GPU + warmup/iteration counts. | **STRENGTHENING** | built-but-divergent | Emit a `--generate-repro-command`-style line from `candidate_bench.py` and put a 5-line "Reproduce" block at the top of the PR body; keep manifests as the linked appendix. | `bench/candidate_bench.py`, PR template in `upstream/reports/` |
| G14 | **Compute-sanitizer lanes skipped.** Recorded honestly as `"result": "skipped"` in the correctness attestation. | **OPTIONAL** for upstream (no FlashInfer CI job runs it), **STRENGTHENING** for us | designed-but-unbuilt — architecture doc §"Memory and concurrency safety" (memcheck/initcheck/racecheck/synccheck + guard canaries) | Run the four sanitizers over a reduced adversarial subset on the SM89 dev tier (it is a correctness-authority target, so this needs no rented GPU). Cheap, and it directly targets the `0 * NaN` stale-V class the contract §5 already names. | `tests/kernel/`, `docs/TARGETS.md` lane |
| G15 | **Missing upstream plumbing: `docs/api/attention.rst`, `flashinfer/aot.py` registration, `flashinfer/__init__.py` export, `csrc/*_jit_binding.cu`.** | **OPTIONAL** individually, collectively required to merge | missing | Mechanical; do it last, from the `CLAUDE.md` 11-step list. | `upstream/pr-draft/` |
| G16 | **`upstream/CLAIMS.md` upstream-baseline section is stale in four places.** (a) It describes #4502 as adding prequantized GQA support — it is a *perf* PR (N64 score-slot reuse); **#3640** added the SM120 NVFP4 path. (b) It omits #3640, #4149 (MXFP8 ragged prefill SM120/121, open) and **#4691** (SM120 Sage INT8-QK/FP8-PV, open). (c) It asserts no SM120 CI runner exists; `unit_test_rtx_pro_6000` is in the internal matrix and visibly passing. (d) cuDNN #595 reaches native head tiles to 256, so "D256 low-precision on SM120 does not exist" is too strong. | **BLOCKING** for our own decision quality (not for the PR) | built-but-wrong | Rewrite the "Upstream baseline as of …" block per §2.4/§2.5/§2.8. Restate the gap precisely as **paged page_size-1 + D256 + 24:4 GQA + bottom-right-causal prefix/EXTEND + per-row-Q/per-tile-K/per-tile-V scales + an inclusive gather/centre/rotate/quantize/pack path** — every one of the neighbours is dense or ragged or block-sparse, per-tensor or per-block, and none is paged. Keep claim K2 (zero cross-lane S→PV packing) prominent: cuDNN #509 independently documents needing shuffles for the same step. | `upstream/CLAIMS.md` |
| G17 | **head_dim 256 + shared-memory staging is a known SM12x cliff.** PR #3485 shipped SM120-aware tile shrinking for exactly this and *still* drew a post-merge report of `head_dim >= 256` failing on 128 KiB-smem parts through the persistent `BatchAttention` path. Our kernel already lives at 72.7 KB of a 99 KB opt-in and `TORCH_CHECK(smem <= optin, ...)`. | **STRENGTHENING** | missing (as an *upstream-facing* test) | Add an explicit smem-budget test at D256 on both tile widths that asserts the launcher's opt-in request and fails with a typed error rather than a CUDA error; state the budget in the PR body. | new test; `docs/OPERATOR_CONTRACT.md` §5 |

---

## 4. Ordered work plan

### Phase A — CPU-only, no GPU session (do all of this first)

| # | Task | Gap | Notes |
|---|---|---|---|
| A0 | **Open a tracked issue upstream** describing the operator, its narrow surface, the SM120 target and the evidence we hold; ask whether the **experimental track** applies and who would review it | — | Cheapest, highest-leverage single action. `docs/code_review_guidance_human.md` says the experimental track is "declared via a tracked issue" (owner `@bkryu`). It also pre-solves the failure mode that stalled #4149 (34 days) and #4691 (no reviewer). Do this before writing kernel code for upstream. |
| A1 | Rewrite `upstream/CLAIMS.md` "Upstream baseline" and the gap statement | G16 | Half a day. Blocks nothing but corrects what every later decision rests on. |
| A2 | Implement **Oracle A** as pure PyTorch (CPU-runnable at small shapes) | G2 | The largest single item and the one that makes the contract normative. Model it on cuDNN's `test/python/sdpa/fp8_ref.py::compute_ref` — a **tiled online-softmax** FP32 reference returning the intermediates (`o_quant, stats, o_amax`), not a naive one — and on `_preprocess_qkv_ref` in `test_nvfp4_attention_sm120.py`. Runs on CPU for small/medium shapes; only the *comparison against the kernel* needs a GPU. |
| A3 | Route the declared surface through `CapabilityError(ValueError)`; write the rejection tests | G4 | CPU-testable for the argument-validation half (`pytest.raises` before any launch). |
| A4 | Design the upstream API surface (`_quantize_q`, `_quantize_kv`, `_fwd` with `out=`/`lse=`/`return_lse`/`sm_scale`), and the caller-owned flat workspace + byte-requirement query | G10, G5 | Write it as a thin Python layer over the existing extension so the .cu change is deferred. The workspace change is a prerequisite for CUDA graphs. |
| A5 | Write the `TraceTemplate` and the `tests/trace/example.py` call | G9 | Fixtures need one GPU run to regenerate; the template itself is CPU work. |
| A6 | Port the adversarial + reference suites into upstream test shape; add the stream / aliasing / workspace-guard tests | G3, G4b, G11 | Content already exists; this is form conversion plus the generalization matrix. Use stacked single-dimension `parametrize`, tuple-parametrize for coupled dims, `pytest.param(..., id=...)` for per-case thresholds, `flip_coin` to alternate `out=` vs allocated, and non-power-of-2 lengths. Keep `tests/kernel/` unchanged as the lab lane. |
| A7 | Add the `upstream_comparability` and `control` lanes to `candidate_bench.py`; add repro-command emission | G6, G7, G13 | Code only; measurement is Phase B. |
| A8 | Write `benchmarks/bench_<op>_sm120.py` in upstream form | G8 | Depends on A4. |
| A9 | Amend `docs/OPERATOR_CONTRACT.md`: LSE plan, smem budget as normative, `cuda_graph` claim downgraded to `eager_only` until G5 lands; restate the Hadamard claim as model-dependent | G1, G5, G17, G18 | Also closes contract §7's open item on env-switch variants. The contract is already the "design doc as markdown" artefact the review guidance asks for — it ships as `docs/design_docs/<op>.md`. |
| A10 | Convert our timing protocol into a **device-free test**, the way cuDNN landed `test_sm120_tile_rule.py` | G7 | The tile-width selection rule (wide BM=128 only when every head is fp8-PV, because the bf16-PV V staging buffer does not fit beside a 128-row Q tile in SM120's 99 KB opt-in) is arithmetic, and asserting it makes the smem budget a reviewable claim rather than a comment. Pure CPU. |

### Phase B — needs a GPU session (SM89 dev tier where possible)

| # | Task | Gap | Tier |
|---|---|---|---|
| B1 | Run Oracle A against the kernel across the golden matrix; record intermediate-boundary agreement | G2 | **SM89 is sufficient** — this is correctness, and `sm89-rtx4090-local` has correctness authority. |
| B2 | Compute-sanitizer memcheck/initcheck/racecheck/synccheck on a reduced adversarial subset | G14 | **SM89**. `docs/TARGETS.md` already lists it as available there. |
| B3 | Implement + test the LSE writeback; regenerate goldens **in a dedicated commit** | G1 | SM89 for correctness; SM120 to re-pin the goldens (they are capability-gated to 12.0). |
| B4 | CUDA-graph capture / NaN-poisoned replay test on a capacity-stable workspace | G5 | **SM89** for the semantics; SM120 to confirm. |
| B5 | Regenerate `fi_trace_out` fixtures | G9 | Any CUDA GPU. |
| B6 | **SM120 qualification:** goldens re-pinned; `upstream_comparability` lane (CUDA graph, 10/100, clocks requested and observed); `control` lane vs FlashInfer BF16 paged prefill at D256/page 1/24:4, interleaved A-B-B-A, ≥10 independent blocks; second-allocation confirmation | G6, G7 | **SM120 only.** This is the session that produces the PR's perf table. |
| B7 | Public quality lane on an open model: per-layer CosSim / relative-L1 / RMSE, **average and worst layer**, on captured real activations; a non-perplexity task metric (NIAH / long-context retrieval) | G12 | Whatever GPU the model fits; no restricted third-party data. |
| B8 | Reproduce the "worst-head per-layer error 2.2 % → 0.55 % with rotation" measurement as a committed artefact, or narrow the claim | G18 | SM89 is enough; needs a redistributable activation fixture, which B7 produces. |

### Phase C — assemble

C1 clean commit series against a fresh fork (kernel/header → binding → Python API →
tests → benchmark → trace → docs), C2 the plumbing set (G15), C3 PR body with pasted
repro commands, named SKU, warmup/iteration counts, published regressions and an
explicit cross-run caveat if any baseline column is not remeasured in the same run
(the #4502 pattern), C4 ask for `run-ci` / `/bot run tests/attention` early and expect
to have to ask twice.

**Sequencing judgement:** do not open a PR before G1, G2, G3 and G4 exist. #4149 and
#4691 demonstrate that a technically strong standalone SM120 attention PR with
complete plumbing can sit for 34 days with zero human review; an incomplete one will
do worse. The cheapest path to maintainer attention is a PR whose tests a maintainer
can run on `unit_test_rtx_pro_6000` with one comment.

**Process facts worth planning around** (verified from PR timelines): the `op:
attention` reviewers who actually act are `saltyminty` (reviewed #4502, #3518, #3485),
`bkryu` (merged #3518, authored #3485), `jimmyzho` (reviewed and merged #4272), with
`aleozlx` routing (*"@saltyminty to review or assign"* on #4502) and `qsang-nv` and
`nvpohanh` also in the pool. Median contributor experience: `run-ci` arrives days
late and must be asked for; #4272's author had to write *"The tests passed but github
PR status says it's not. Do you know what else I can do to get this PR reviewed?"*
Every merged PR here went through **exactly one or two human review rounds** — the
volume is CodeRabbit's, the decisions are one human's. Budget for latency, not for
review depth.

---

## 5. What this pipeline already does better than upstream convention

These are not filler. Several of them are things FlashInfer reviewers have visibly
wanted and not had, and each is cheap to surface in a PR body or a linked appendix.

1. **Schedule-as-data with a hash carried into every result.**
   `workloads/profiles/*.yaml` declares parameters, `tools/gen_workload.py` expands
   them deterministically, and the SHA-256 of the canonical case list
   (`2621651e…7beb0`) is recorded in the profile, in the bench JSON
   (`workload_cases_sha256`) and in the attestation. Upstream benchmarks encode shapes
   as argparse lists and `DEFAULT_CONFIGS` tuples; a published number cannot be tied
   to a schedule. `tools/gen_workload.py --check` is a CI gate.
2. **Bit-exact golden regression with the numerics environment pinned.**
   72 cases across `{qk_i8} × {rotate} × {center_k} × {fp8/bf16/mixed PV}` and six
   shapes, hashing not only the output but `q8`, `k8`, `vt8`, `vb16` and every scale
   array separately — so a failure says *whether the move is in the .cu or in
   `quant.py`*. The stored `env` block records torch/CUDA/device/SM count/opt-in smem/
   `extra_cuda_cflags`/source SHAs/`CUBLAS_WORKSPACE_CONFIG`/`allow_tf32`/
   `float32_matmul_precision`. The `_pinned_numerics` fixture pins four TF32-related
   globals *and restores them*, because ambient state (SGLang sets these) silently
   moves every digest. Upstream has nothing equivalent for any kernel.
3. **A determinism check inside the golden test.** Every case launches the kernel
   twice from byte-identical inputs and fails loudly if the outputs differ, with a
   message that names the likely causes (missing `__syncthreads()` after a cp.async
   restructure, uninitialized smem, atomics-based reduction). "Goldens cannot exist
   for a nondeterministic kernel."
4. **A capability-gated golden lane that skips rather than widens.** Off SM120 the
   suite `pytest.skip(...)`s with an explanation instead of relaxing tolerances —
   verified working on the SM89 tier. Upstream's `allclose` thresholds are per-file
   constants that nothing prevents from drifting.
5. **A written regeneration policy for the goldens** (own commit, never bundled with a
   scheduling change, message names which digests moved and why) with the failure
   modes enumerated in the file header. This is exactly the discipline that would have
   caught the `-use_fast_math` / `--fmad` / `-gencode` class of silent numerics change.
6. **An adversarial suite organized by named failure hypothesis, not by API surface.**
   33 tests where each docstring states what a failure *confirms*: quantization-window
   leakage (H2), head-confusion permutation proof (H6), partition-of-unity on the
   softmax denominator (H3), duplicated-V-channel exactness, constant-V exact
   reproduction, V-scale linearity, structured distributions (Student-t, channel
   outliers, real rotate-half RoPE, two-magnitude heterogeneous layouts), flat-
   attention regime, chunked composition bit-exactness across three chunk schedules.
   Upstream's closest analogue is #4502's `test_..._structured_q_correction` and
   `test_..._output_magnitude` — two tests, same spirit, far less coverage.
7. **Explicit workspace-hazard tests.** `test_workspace_reuse_shrinking_n`
   (long-then-short with one workspace), `test_pool_gather_ignores_foreign_slots`,
   and a `SGLANG_FP8_PREFILL_ZERO_WS` discriminator that zero-fills every workspace
   allocation so an allocator-history-dependent bug becomes reproducible. The contract
   §5 states the stale-V `0 * NaN` hazard as *contract-level*, binding on any
   conforming implementation. Upstream's equivalent is #4502's
   `test_..._unpadded_k_len_masks_tail_garbage` — one test, one tensor family.
8. **Immutable manifests with append-only attestations**, versioning
   `operator_contract_version`, `layout_version`, `package_api_version` and
   `binary_abi` **separately**, and binding source tree digest, toolchain, target,
   workload hash, raw-sample digest and approver identity. `tools/validate_registry.py`
   mechanically refuses a performance attestation on a `development`-authority target,
   a workload hash that does not match its profile, or a warning promoted to a pass.
9. **A lane taxonomy with declared authority.** Only `inclusive` and `schedule_replay`
   can qualify; `core`, `preprocessing` and `upstream_comparison` are diagnostic. The
   timing backend is recorded, never assumed, and results from different backends are
   declared non-comparable. Upstream's #3485 had to be *asked* what timing method it
   used.
10. **Published negative results and an explicit "NOT claimed" register.** `CLAIMS.md`
    records that the ~610 TF/s figure is derived rather than measured, that the
    private downstream result is only a smoke probe, that the logprob metric cannot
    rank backends, and that the technique is not original. The rented-GPU session then *retired* the disputed
    preprocessing number honestly (claimed 0.2–0.3 s; measured **8.21 s** — wrong by
    ~30×) rather than quietly dropping it. #4272 publishing FP8-slower-than-NVFP4 in 3
    of 4 cells and merging anyway is the upstream precedent that this posture is
    rewarded, not punished.
11. **A full-size, non-wrapping, scattered page-1 pool and per-(chunk, layer) seeds**
    in the bench, so deep-prefix gather timing is real and no run can exploit repeated
    identical buffers. Upstream attention benchmarks reuse one buffer set across all
    iterations by construction.
12. **An end-to-end serving number on a real model.** The 446 k anchor
    (209.5 s → 146.5 s, 1.43×, with the schedule and its hash on record) is the exact
    artefact whose absence killed the vLLM SageAttention attempt: the maintainer asked
    *"can you run some benchmarks (e.g. benchmark throughput for Llama 8B)?"* and the
    PR died there. SGLang's Sage3 default-switch on sm_120 merged on one wall-clock
    ratio for one model. We have a stronger version of that evidence than either, and
    it should appear in the PR body — with the honest inclusive framing (preprocessing
    measured at 8.21 s of the schedule, not the 0.2–0.3 s once claimed).
13. **An explicit, machine-checked support surface** (`capability.py`'s
    `OperatorCapability` + one catchable `CapabilityError`) with the rule that a
    benchmark or qualification profile must **fail** on fallback while only a consuming
    framework may catch it. cuDNN arrived at the same design ("silent wrong results are
    the worst failure mode; a silent slow path is the second worst") and enforces it
    with capability-bit tests; FlashInfer has no equivalent. Ours is designed but not
    yet on the call path — see G4 — which makes wiring it up a cheap way to be *ahead*
    of both.

**How to surface these in a PR without bloating it:** one "Evidence" section with (a)
the pasted repro command, (b) the SKU and clock statement, (c) the perf table, (d)
three sentences on the golden/determinism lane with a link to this repository's
public evidence directory, and (e) the "not claimed" list. Precedent (#4502's cross-run
caveat, #3485's published regressions, #3518's declared scope limits) says the
honesty is read as competence.
