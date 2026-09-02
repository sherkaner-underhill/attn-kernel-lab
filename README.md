<!-- SPDX-License-Identifier: Apache-2.0 -->
# attn-kernel-lab

`attn-kernel-lab` qualifies a narrow D256/page-1/24:4 fused paged prefill
kernel at **550 TF/s inclusive** on RTX PRO 6000 Blackwell (SM120), with
**1.968×** over FlashInfer BF16 paged prefill and **2.213×** over cuDNN FROST
BF16 under recorded controls; every number below links to its evidence.

## Limitations

- EXTEND only: no decode or target-verify path.
- Head dimension 256 and page size 1 only. The declared GQA family is 16:4,
  24:4, and 32:4; 8:2 remains generalization-tier.
- The KV pool must be unquantized BF16 or FP16. An FP8 or FP4 pool would be
  double-quantized and is refused.
- CUDA Graph capture requires a capacity-reserved workspace and caller-owned
  output buffers. Grow-on-demand workspaces remain eager-only.
- Performance evidence covers one SKU and power envelope across two independent
  allocations.
- The FROST FP8 port is a correctness-first v0; a tuned port could differ.
- Quality evidence is synthetic attention-output simulation, not a downstream
  task result. These are kernel measurements, not serving or application
  qualification.

## Claims

[`upstream/CLAIMS.md`](upstream/CLAIMS.md) is the canonical register. This table
is a short index into its K1–K14 claims.

| # | Claim | Evidence | Confidence | Reproduce |
|---|---|---|---|---|
| <a id="k1"></a>[K1](upstream/CLAIMS.md#kernel-and-dataflow-claims) | The SM120 E4M3 A-fragment byte order is measured. | [Fragment-layout probe](probes/fragment_layout/r2a_fragment_microtest.cu) | measured | [Build/run instructions](probes/fragment_layout/r2a_fragment_microtest.cu) · SM120 only |
| <a id="k2"></a>[K2](upstream/CLAIMS.md#kernel-and-dataflow-claims) | SIGMA64 makes S-to-PV packing shuffle-free. | [Kernel dataflow](src/attn_kernel_lab/csrc/fp8_prefill_attn.cu) | derived, independently corroborated | `python3 -m pytest -q tests/kernel/test_golden_bitexact.py` · SM120 only |
| <a id="k3"></a>[K3](upstream/CLAIMS.md#kernel-and-dataflow-claims) | The exact online-softmax P scale is 448. | [Operator contract §3.4](docs/OPERATOR_CONTRACT.md#34-v) | derived | `python3 -m pytest -q tests/kernel/test_oracle_a.py` · SM120 only |
| <a id="k4"></a>[K4](upstream/CLAIMS.md#kernel-and-dataflow-claims) | Legacy and unity-scale block-scaled FP8 MMA issue at the same rate on SM120. | [Instruction-rate probe](probes/mxfp8/pv_mma_bench.cu) | measured | [Build/run instructions](probes/mxfp8/pv_mma_bench.cu) · SM120 only |
| <a id="k5"></a>[K5](upstream/CLAIMS.md#kernel-and-dataflow-claims) | Bare `-arch=sm_120a` silently lowers; explicit `-gencode` is required. | [Syntax probe](probes/mxfp8/mx_syntax_probe.cu) and [test](tests/test_toolchain_gate.py) | measured on two toolchains | `python3 tools/probe_target.py --compile-gate sm_120a --bare-arch` |
| <a id="k6"></a>[K6](upstream/CLAIMS.md#kernel-and-dataflow-claims) | The exact alpha-rescale skip helps; stacking the visible-tile path does not. | [Kernel A/B implementation](src/attn_kernel_lab/csrc/fp8_prefill_attn.cu) | measured | SM120 only |
| <a id="k7"></a>[K7](upstream/CLAIMS.md#kernel-and-dataflow-claims) | Scalar work and movement, not tensor-pipe issue rate, limit the kernel. | [Issue-rate probe](probes/mxfp8/pv_mma_bench.cu) and [campaign summary](bench/results/B6-SUMMARY-20260830.md) | measured | SM120 only |
| <a id="k8"></a>[K8](upstream/CLAIMS.md#kernel-and-dataflow-claims) | Direct full-schedule replay replaces the earlier inferred throughput. | [Allocation-1 schedule](bench/results/120-20260830T003408Z-candidate-zero-schedule.json) | measured | `python3 bench/candidate_bench.py --profile d256-24x4-446k --label repro --layers 16` |
| <a id="k9"></a>[K9](upstream/CLAIMS.md#kernel-and-dataflow-claims) | Shared rotation is exact; any quality benefit is model-dependent and not generally claimed. | [Contract scope and caveat](docs/OPERATOR_CONTRACT.md#31-shared-rotation) | derived exactness; benefit unsubstantiated | `python3 -m pytest -q tests/kernel/test_oracle_a.py` · SM120 only |
| <a id="k10"></a>[K10](upstream/CLAIMS.md#kernel-and-dataflow-claims) | The preprocessing restructure helped; the instruction diet was performance-neutral. | [Preprocessing record](bench/results/120-20260830T053422Z-fruit-newprep.json) and [A/B records](bench/results/) | measured | SM120 only |
| <a id="k11"></a>[K11](upstream/CLAIMS.md#kernel-and-dataflow-claims) | cuDNN's public FP8-class SDPA surface does not accept D256 on SM120. | [Selection probe](probes/cudnn_frost/probe1_pygraph_fp8.py) | measured | `python3 probes/cudnn_frost/probe1_pygraph_fp8.py` · SM120 only |
| <a id="k12"></a>[K12](upstream/CLAIMS.md#kernel-and-dataflow-claims) | The candidate beats the strongest D256-capable FROST BF16 path in this campaign. | [FROST control summary](bench/results/FROST-CONTROL-20260830.md) | measured | `python3 probes/cudnn_frost/probe3_dsl_sm120_bf16.py` · SM120 only |
| <a id="k13"></a>[K13](upstream/CLAIMS.md#kernel-and-dataflow-claims) | A per-tensor FP8-QK FROST port does not beat its BF16 donor. | [FROST FP8-QK port](probes/cudnn_frost/prefill_fp8qk_sm120.py) | measured, v0-caveated | `python3 probes/cudnn_frost/probe4_fp8qk_port.py` · SM120 only |
| <a id="k14"></a>[K14](upstream/CLAIMS.md#kernel-and-dataflow-claims) | The transform pipeline, not scale granularity alone, carries the synthetic fidelity advantage. | [Quality results](probes/quality/RESULTS.md) and [raw record](probes/quality/pertensor_vs_finegrained.json) | measured simulation | `python3 probes/quality/pertensor_vs_finegrained.py` · Any CUDA GPU |

## Reproduce it

| Tier | Commands | What it can establish |
|---|---|---|
| No GPU | `python3 tools/gen_workload.py workloads/profiles/d256-24x4-446k.yaml --check`<br>`python3 tools/validate_registry.py`<br>`python3 -m pytest -q` | Workload, record, schema, and CPU invariant consistency |
| Any supported CUDA GPU | `python3 quality/q1_public.py --quick --no-write`<br>`python3 tools/probe_target.py --compile-gate sm_120a --bare-arch` | Synthetic numerical smoke and the SM120a toolchain trap |
| SM120 only | `python3 -m pytest -q tests/kernel/`<br>`python3 bench/candidate_bench.py --profile d256-24x4-446k --label repro --layers 16` | Device goldens, contract checks, and a fresh schedule replay |

CI cannot verify any performance claim in this repository, and it has no GPU. It
runs the no-GPU tier above: the workload hashes are current, the registry
records validate against their schemas and against the cross-record invariants,
and the CPU invariant suites pass. The GPU correctness lane under
[`tests/kernel/`](tests/kernel/) needs a CUDA device and is not collected at all
without one — a CPU-only run reports it on the summary line rather than counting
it as passed. The toolchain trap in K5 likewise has no CI evidence; its test
skips without `nvcc`.

## The measured field

| Path | Result | Scope |
|---|---|---|
| [Candidate core across allocations](upstream/CLAIMS.md#kernel-and-dataflow-claims) | [587–601 TF/s](bench/results/B6-SUMMARY-20260830.md) | [Two distinct SM120 devices](docs/hardware-labels.md) |
| [Candidate, post-restructure inclusive](upstream/CLAIMS.md#kernel-and-dataflow-claims) | [550.0 TF/s](bench/results/120-20260830T053422Z-fruit-newprep.json) | [Protected 24:4 schedule](workloads/profiles/d256-24x4-446k.yaml) |
| [FlashInfer BF16 paged control](bench/results/B6-SUMMARY-20260830.md) | [1.968× (95% CI 1.967–1.970)](bench/results/B6-SUMMARY-20260830.json) | [Ten paired ABBA process blocks](bench/results/B6-SUMMARY-20260830.md) |
| [cuDNN FROST BF16 control](bench/results/FROST-CONTROL-20260830.md) | [2.213× (95% CI 2.210–2.216)](bench/results/FROST-CONTROL-20260830.json) | [Two paired ABBA process blocks](bench/results/FROST-CONTROL-20260830.md) |
| [FROST per-tensor FP8-QK v0](probes/cudnn_frost/prefill_fp8qk_sm120.py) | [239–264 TF/s](upstream/CLAIMS.md#kernel-and-dataflow-claims) | [Correctness-first port caveat](probes/cudnn_frost/probe4_fp8qk_port.py) |

## How it is organised

- The [operator contract](docs/OPERATOR_CONTRACT.md) defines the quantized
  operator independently of an implementation.
- [Target profiles](targets/) state hardware authority and implementation
  family. Development-target numbers cannot qualify a release.
- [Workload profiles](workloads/) expand deterministically into hashed cases,
  binding geometry and measurements.

[Engine profiles](engines/) add the serving-integration coordinate. A
kernel-performance qualification omits that coordinate because it runs
framework-free; an integration or application attestation must name it.

## Enforced rules

- A `development`-authority target cannot carry a performance attestation or
  qualify a release.
- Release manifests are immutable; later facts are append-only attestations
  bound to the manifest digest.
- Workload and source-tree digests come only from the sanctioned producers, and
  the validator rejects stale bindings.
- Skipped gates need reasons, warnings never become passes, and only
  `inclusive` or `schedule_replay` lanes carry promotion authority.
- Timing backends are explicit; results from different backends are not silently
  compared.
- Third-party datasets and real activation captures never enter this repository.

## Contributing

Parallel kernel development is supported, including publishing a partially
qualified artifact and requesting the next rung from someone with the right
hardware. [`CONTRIBUTING.md`](CONTRIBUTING.md) is the protocol;
[`docs/ARTIFACT_LIFECYCLE.md`](docs/ARTIFACT_LIFECYCLE.md) is the step-by-step
runbook from new kernel source to qualified release.

## Provenance and licence

The project is Apache-2.0 except for one MIT-derived NVIDIA probe; [`NOTICE`](NOTICE)
and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) record the mixed licence
and the privacy-preserving `origin-private` provenance alias.

## AI assistance, disclosed up front

The majority of the work in this repository — the kernel, the preprocessing
pipeline, the test suites, the benchmark harness, the qualification machinery,
and most of the documentation — was written by **Claude**, Anthropic's AI
assistant, across many working sessions. The repository owner is not a CUDA
developer: the project grew incidentally out of a private serving deployment,
and the owner's role has been direction, review, hardware, and the final call
on every published claim.

That provenance is precisely why the repository is built the way it is. No
claim here asks to be trusted on authorship: every number traces to a committed
record with raw samples and environment fingerprints, correctness is pinned by
bit-exact goldens reproduced on four distinct devices, digests are recomputed
by the validator rather than quoted, and a clean-clone session
([`bench/results/REPRO-CLEANCLONE-20260831.md`](bench/results/REPRO-CLEANCLONE-20260831.md))
reproduced the headline results on a machine that received nothing but this
repository. Read the evidence, not the author.

The owner reviews and stands behind what is published here.
