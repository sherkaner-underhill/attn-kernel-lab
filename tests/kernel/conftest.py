# SPDX-License-Identifier: Apache-2.0
"""Pytest configuration for the fp8 prefill attention test package.

Three jobs.  The first two serve the bit-exact golden harness
(``test_golden_bitexact.py``); the third guards a property of the
preprocessing path:

1. ``--write-golden`` -- the flag that (re)generates ``golden_bitexact.json``.
   ``GOLDEN_WRITE=1`` in the environment does the same thing, for callers
   that cannot pass pytest flags.

2. cuBLAS determinism.  ``quant.py`` runs an fp32 GEMM (the Hadamard
   rotation of Q and K).  cuBLAS picks its algorithm partly from the size of
   the workspace it is handed, and split-K variants reduce with atomics,
   which is run-to-run nondeterministic.  ``CUBLAS_WORKSPACE_CONFIG=:4096:8``
   is the documented setting that pins this down; it must be set BEFORE the
   first cuBLAS handle is created, so it is set here, before torch is
   imported anywhere. Export it in the SM120 shell too if you run any of this
   outside pytest -- ``bench/candidate_bench.py`` sets it the same way, and
   for the same reason at the same scope (its module import, not inside
   ``pin_numerics()``, which runs after ``import torch``).

3. A synchronisation tripwire -- see the ``set_sync_debug_mode`` block below.
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# This directory is the GPU correctness lane. In a CPU-only environment (the
# GitHub CI runner has neither torch nor a GPU) the correct outcome is that
# these files are not collected at all -- not an import error.
try:
    import torch  # noqa: F401

    _GPU = torch.cuda.is_available()
except Exception:  # noqa: BLE001
    _GPU = False

collect_ignore_glob = [] if _GPU else ["test_*"]

# The preprocessing path is INTENDED to be sync-free. The kernel/host audit
# grepped ops.py, kernel.py, capability.py and quant.py for `.item()`,
# `int(tensor)`, `bool(tensor)`, `.cpu()`, `.tolist()`, `.numpy()` and
# `torch.cuda.synchronize` and found zero hits; the one real device->host
# synchronisation it did find (`.nonzero()` on a CUDA tensor in the boundary-tile
# tail path) is not in that grep's vocabulary, which is exactly why a grep is not
# a guard and this is. "warn" makes any implicit sync announce itself on stderr
# instead of hiding inside a millisecond.
#
# WARNINGS PRINTED HERE ARE FINDINGS, not noise: something on the hot path
# serialised the host against the device. Chase them; do not silence them by
# on request. `ATTN_KERNEL_LAB_SYNC_DEBUG=1` arms it for the case
# where a known third-party sync would drown a run.
#
# The mode is global and covers assertions in the test bodies too, which DO sync
# legitimately (`.item()`, `.cpu()`), so expect a warning tally on every GPU run;
# what matters is warnings attributable to library code. `set_sync_debug_mode`
# calls `_lazy_init()`, so this creates the CUDA context at conftest import --
# harmless, and strictly after the CUBLAS_WORKSPACE_CONFIG pin above, which is
# what has to precede the first cuBLAS handle.
# REVIEWER INVERSION (2026-08-30): opt-IN, not opt-out. The mode is
# process-global and fires on every legitimate test-body sync (~700/run),
# drowning real findings; its original target — the one library-path sync,
# fruit report Q5 — is fixed. Arm deliberately when hunting for new ones:
if _GPU and os.environ.get("ATTN_KERNEL_LAB_SYNC_DEBUG", "") == "1":
    torch.cuda.set_sync_debug_mode("warn")


def pytest_addoption(parser):
    parser.addoption(
        "--write-golden",
        action="store_true",
        default=False,
        help="(re)generate tests/golden_bitexact.json instead of checking "
        "against it. ONLY for reviewed, intentional numerics changes -- "
        "see the header of test_golden_bitexact.py.",
    )
