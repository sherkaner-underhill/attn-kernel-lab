#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Metric extraction for the Q lane, on top of the instrument in ``probes/quality``.

**This module implements no metric.**  ``probes/quality/pertensor_vs_finegrained.py``
is the metric donor: its ``_metrics`` is the definition of row-relative L2,
cosine similarity, relative L1, RMSE, output-norm ratio and the NaN/Inf counts
for this project, its ``_pv(..., "p_online", ...)`` is the P-rounding form that
``--check-oracle`` pins against ``oracle_a.attention``, and its ``_fp32_reference``
/ ``_run_scheme`` are the reference and control paths whose bf16 yardstick already
reproduces the repository's recorded 0.31-0.45% implementation-swap band.  A
second implementation of any of those would be a second definition, and then the
public-lane numbers and the instrument's numbers would not be comparable -- which
is the entire point of having one.

What this module adds is **slicing**.  The donor reports one aggregate per cell
plus a worst head; the gate needs the same metrics reported four ways -- mean,
worst layer, worst head, worst ROW -- with each slice anchored to the control
evaluated at *that same slice*.  It gets them by calling the donor's ``_metrics``
on sub-tensors, with the donor's geometry constants temporarily narrowed to match
(``_metrics`` derives its head axis from module-level ``T_ROWS``/``Q_HEADS``).
That is reuse rather than reimplementation: every number below comes out of the
donor's own arithmetic.

The candidate is the real operator -- ``attn_kernel_lab.ops.prefill_extend``,
called on the declared surface (24:4 GQA, head_dim 256, page_size 1, BF16 pool,
bottom-right causal extend).  The donor deliberately never executes a kernel;
Q1's whole job is that it does.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import os
import pathlib
import sys

# Before torch: the true-FP32 rotation inside quant.py is reproducible only under
# the contract §3.1 workspace pin, and the variable must be set before the first
# cuBLAS handle exists.  Same reasoning as bench/candidate_bench.py and the donor.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = pathlib.Path(__file__).resolve().parents[1]
DONOR_PATH = ROOT / "probes" / "quality" / "pertensor_vs_finegrained.py"

_donor = None


def donor():
    """Import ``probes/quality/pertensor_vs_finegrained.py`` as a module, once.

    Imported by path rather than as a package because ``probes/`` is a tree of
    standalone instruments, not a library -- the same way ``tests/kernel``
    imports ``quant`` directly.
    """
    global _donor
    if _donor is None:
        spec = importlib.util.spec_from_file_location("pertensor_vs_finegrained", DONOR_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _donor = module
    return _donor


def donor_sha256() -> str:
    return hashlib.sha256(DONOR_PATH.read_bytes()).hexdigest()


@contextlib.contextmanager
def _geometry(t_rows: int, q_heads: int):
    """Narrow the donor's geometry constants for the duration of a sliced call.

    ``_metrics`` reshapes to ``[T_ROWS, Q_HEADS]`` to build its per-head view, so
    a [T, 1, D] or [1, 1, D] slice needs the constants to agree.  Restored on
    exit; nothing else in the donor is touched.
    """
    module = donor()
    saved = (module.T_ROWS, module.Q_HEADS)
    module.T_ROWS, module.Q_HEADS = t_rows, q_heads
    try:
        yield module
    finally:
        module.T_ROWS, module.Q_HEADS = saved


def metrics_all(out, ref) -> dict:
    """The donor's metric set over a whole [T, Hq, D] case."""
    module = donor()
    return module._metrics(out, ref)


def metrics_per_head(out, ref) -> list[dict]:
    """The donor's metric set, per query head."""
    heads = out.shape[1]
    with _geometry(out.shape[0], 1) as module:
        return [module._metrics(out[:, h : h + 1], ref[:, h : h + 1]) for h in range(heads)]


def metrics_per_row(out, ref) -> list[list[dict]]:
    """The donor's metric set for every (row, head).

    Exhaustive rather than sampled: it is a reduction over 24 x 128 slices of one
    already-materialised tensor, and the worst ROW is the slice the addendum
    exists to protect.  Locating it approximately would defeat the point.
    """
    rows, heads = out.shape[0], out.shape[1]
    with _geometry(1, 1) as module:
        return [
            [module._metrics(out[r : r + 1, h : h + 1], ref[r : r + 1, h : h + 1]) for h in range(heads)]
            for r in range(rows)
        ]


def metrics_at_row(out, ref, row: int, head: int) -> dict:
    with _geometry(1, 1) as module:
        return module._metrics(out[row : row + 1, head : head + 1], ref[row : row + 1, head : head + 1])


def metrics_at_head(out, ref, head: int) -> dict:
    with _geometry(out.shape[0], 1) as module:
        return module._metrics(out[:, head : head + 1], ref[:, head : head + 1])


# --------------------------------------------------------------------------
# case execution: reference, controls, candidate
# --------------------------------------------------------------------------


class Case:
    """One fixture case, generated and held on the device.

    Ordering is memory-driven: the fp32 master is 3.7 GiB at 446k and exists only
    long enough to build the reference, which is why the reference is computed
    first and the fp32 K/V dropped immediately -- the same discipline as the
    donor's ``run_cell``.
    """

    def __init__(self, dist: str, n: int, verbose: bool = True):
        import torch

        module = donor()
        cfg = module.DISTRIBUTIONS[dist]
        dev = torch.device("cuda")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        self.dist = dist
        self.n = n
        self.seed = cfg["seed"]
        self.doc = cfg["doc"]
        self.t_rows = module.T_ROWS
        self.prefix = n - module.T_ROWS
        self.npad = (n + module.BLK - 1) // module.BLK * module.BLK
        self.tail = module._tail_mask(dev)
        self.head_batch = module._head_batch_for(self.npad)

        q_f32 = module._gen_q(dist, dev, cfg)
        k_f32, v_f32 = module._fill_kv(dist, n, dev, cfg)
        self.q_bf, self.k_bf, self.v_bf = (
            x.to(torch.bfloat16) for x in (q_f32, k_f32, v_f32)
        )

        self.ref = module._fp32_reference(
            q_f32, k_f32, v_f32, n, self.npad, self.prefix, self.tail, self.head_batch
        )
        del k_f32, v_f32, q_f32
        torch.cuda.empty_cache()
        self._q_bff = self.q_bf.float()

    # -- the anchors --------------------------------------------------------

    def _kv_bf(self, kvh):
        module = donor()
        return (
            module._pad_head(self.k_bf, kvh, self.npad, self.n),
            module._pad_head(self.v_bf, kvh, self.npad, self.n),
            None,
        )

    def control_input_rounding(self):
        """Scheme 2: fp32 attention on bf16-rounded inputs.  Input rounding alone."""
        module = donor()
        return module._run_scheme(
            lambda h: self._q_bff[:, h] * module.SM_SCALE, self._kv_bf,
            self.n, self.npad, self.prefix, self.tail, "plain", lambda *_: {}, self.head_batch,
        )

    def control_implementation_swap(self):
        """Scheme 2b: bf16 inputs, bf16 P, bf16 output.

        The fuller implementation swap, and the one the ratios are taken against:
        it is the same reference computed in the target dtype with reordered ops,
        which is exactly what FlashAttention's own suite uses as ``out_pt``.
        """
        import torch

        module = donor()
        out = module._run_scheme(
            lambda h: self._q_bff[:, h] * module.SM_SCALE, self._kv_bf,
            self.n, self.npad, self.prefix, self.tail, "p_bf16", lambda *_: {}, self.head_batch,
        )
        return out.to(torch.bfloat16).float()

    def candidate(self, workspace=None, **switches):
        """The real operator, on the declared surface."""
        import torch

        from attn_kernel_lab import ops

        idx = torch.arange(self.n, device=self.q_bf.device, dtype=torch.int64)
        out = ops.prefill_extend(
            self.q_bf, self.k_bf, self.v_bf, idx, self.prefix,
            workspace=workspace, **switches,
        )
        return out.float()

    def free(self):
        import torch

        for name in ("q_bf", "k_bf", "v_bf", "ref", "_q_bff", "tail"):
            setattr(self, name, None)
        torch.cuda.empty_cache()


def sha256_tensor(tensor, rows_per_chunk: int = 32768) -> str:
    """SHA-256 over a tensor's raw bytes, streamed row-block by row-block.

    Chunked so hashing a 913 MiB K plane does not need a 913 MiB host spike.
    """
    import torch

    digest = hashlib.sha256()
    flat = tensor.contiguous()
    for start in range(0, flat.shape[0], rows_per_chunk):
        block = flat[start : start + rows_per_chunk].cpu().contiguous()
        digest.update(block.flatten().view(torch.uint8).numpy().tobytes())
        del block
    return digest.hexdigest()
