# SPDX-License-Identifier: Apache-2.0
"""FP8 fused prefill attention backend (SM120 PoC).

Select with ``--prefill-attention-backend fp8_prefill`` (decode stays on
whatever ``--attention-backend`` selects; the stock ``HybridAttnBackend``
composes the two).  This class subclasses ``FlashInferAttnBackend`` and
overrides ``forward_extend`` only: pure-EXTEND forwards whose every request
qualifies run the fused fp8 kernel; everything else — target-verify,
draft-extend, mixed, cross-attention, sliding-window or logit-cap layers,
non-256 head dims, quantized KV pools, small prefills — falls through to
the inherited FlashInfer implementation with its metadata intact.

Environment knobs:

  SGLANG_FP8_PREFILL_MIN_TOKENS   minimum per-request total tokens
                                  (prefix + chunk) to engage (default 8192)
  SGLANG_FP8_PREFILL_BF16_HEADS   comma list of q-head indices that use the
                                  bf16-PV fallback path (default empty);
                                  ``all`` forces bf16 PV everywhere (an
                                  fp8-QK-only mode for numerics A/B)
  SGLANG_FP8_PREFILL_K_CENTER     "0" disables K mean-centering (default
                                  on; softmax-shift-exact quantization
                                  smoothing, see quant.py)
  SGLANG_FP8_PREFILL_QK           "int8" (default) or "fp8": QK matmul
                                  operand type. int8 dots are exact in
                                  int32 (SageAttention design); fp8 kept
                                  for A/B
  SGLANG_FP8_PREFILL_DISABLE      "1" disables the fp8 path entirely
                                  (the backend then behaves exactly like
                                  ``flashinfer``)

The kernel is JIT-compiled on first use via ``torch.utils.cpp_extension``
(cached; ~30-60 s cold).  See ``csrc/fp8_prefill_attn.cu`` for the kernel
design, and the repository's ``upstream/CLAIMS.md`` / ``bench/results/`` /
``engines/sglang/results/`` for measured numerics and throughput.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.layers.attention.fp8_prefill.quant import (
    BLK,
    HEAD_DIM,
    FP8PrefillWorkspace,
    gather_quantize_kv,
    quantize_q,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

_ext = None


def _load_kernel():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load

        src = os.path.join(os.path.dirname(__file__), "csrc", "fp8_prefill_attn.cu")
        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}{minor}a" if (major, minor) >= (9, 0) else f"{major}{minor}"
        logger.info("[fp8_prefill] JIT-compiling %s for sm_%s ...", src, arch)
        _ext = load(
            name="sgl_fp8_prefill_attn",
            sources=[src],
            extra_cuda_cflags=[
                "-O3",
                f"-gencode=arch=compute_{arch},code=sm_{arch}",
            ],
            verbose=False,
        )
        logger.info("[fp8_prefill] kernel ready")
    return _ext


class FP8PrefillAttnBackend(FlashInferAttnBackend):
    """FlashInfer everywhere, except qualifying EXTEND forwards run the
    fused fp8 prefill attention kernel."""

    def __init__(self, model_runner: ModelRunner, *args, **kwargs):
        super().__init__(model_runner, *args, **kwargs)
        self._fp8_disable = os.environ.get("SGLANG_FP8_PREFILL_DISABLE") == "1"
        self._min_tokens = int(
            os.environ.get("SGLANG_FP8_PREFILL_MIN_TOKENS", "8192")
        )
        # minimum CURRENT-CHUNK length to engage (0 = no constraint). Lets
        # big prefill chunks run fp8 while short cached-prefix extends
        # take the unquantized fallback.
        self._min_extend = int(
            os.environ.get("SGLANG_FP8_PREFILL_MIN_EXTEND", "0")
        )
        self._bf16_heads_env = os.environ.get("SGLANG_FP8_PREFILL_BF16_HEADS", "")
        # K mean-centering (SageAttention smoothing; softmax-shift-exact).
        # Default ON; disable with SGLANG_FP8_PREFILL_K_CENTER=0 for A/B.
        self._center_k = os.environ.get("SGLANG_FP8_PREFILL_K_CENTER", "1") != "0"
        # QK operand type: int8 (exact int32 dots; ~0.4% input rounding)
        # vs e4m3 (~3%). int8 is the production default.
        self._qk_i8 = os.environ.get("SGLANG_FP8_PREFILL_QK", "int8") != "fp8"
        # FA3-style incoherent processing (Hadamard rotation of Q/K before
        # quantization; scores exactly invariant). Default on.
        self._rotate = os.environ.get("SGLANG_FP8_PREFILL_HADAMARD", "1") != "0"
        self._debug = os.environ.get("SGLANG_FP8_PREFILL_DEBUG") == "1"
        self._ws: Optional[FP8PrefillWorkspace] = None
        self._pv8_mask_cache: dict[tuple, tuple[torch.Tensor, bool, bool]] = {}
        self._page_size = model_runner.server_args.page_size
        self._kv_cache_quantized = (
            getattr(model_runner.token_to_kv_pool, "is_quantized_kv_cache", False)
        )
        # capability gate is evaluated per layer at forward time; this only
        # logs the configuration once
        if not self._fp8_disable:
            logger.info(
                "[fp8_prefill] enabled: min_tokens=%d bf16_heads=%r page_size=%d",
                self._min_tokens, self._bf16_heads_env or None, self._page_size,
            )

    # ---- metadata-time snapshot (overlap-scheduler contract) -------------
    # Invariant protected: req_to_token / the extend-length lists are
    # scheduler-owned buffers that the overlap scheduler may rewrite after
    # metadata time; reading them at LAYER time is outside the contract
    # window inherited from FlashInfer.  Snapshot ONCE here, inside the
    # window, into owned tensors; the layer-time path reads only the
    # snapshot.

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        ret = super().init_forward_metadata(forward_batch)
        self._fp8_snapshot = None
        if (
            not self._fp8_disable
            and forward_batch.forward_mode == ForwardMode.EXTEND
            and forward_batch.extend_prefix_lens_cpu is not None
            and forward_batch.extend_seq_lens_cpu is not None
        ):
            pfx = [int(x) for x in forward_batch.extend_prefix_lens_cpu]
            ext = [int(x) for x in forward_batch.extend_seq_lens_cpu]
            req_rows = forward_batch.req_pool_indices.tolist()
            r2t = self.req_to_token_pool.req_to_token
            idxs = [
                r2t[row, : p + e].to(torch.long).clone()
                for row, p, e in zip(req_rows, pfx, ext)
            ]
            self._fp8_snapshot = (pfx, ext, idxs)
        return ret

    # ---- eligibility -----------------------------------------------------

    def _layer_supported(self, layer: RadixAttention) -> bool:
        return (
            layer.head_dim == HEAD_DIM
            and getattr(layer, "v_head_dim", HEAD_DIM) == HEAD_DIM
            and layer.tp_q_head_num % max(layer.tp_k_head_num, 1) == 0
            and layer.logit_cap in (0, 0.0, None)
            and getattr(layer, "sliding_window_size", None) in (None, -1)
            and not layer.is_cross_attention
        )

    def _batch_qualifies(self, forward_batch: ForwardBatch) -> bool:
        if self._fp8_disable or self._kv_cache_quantized or self._page_size != 1:
            return False
        if forward_batch.forward_mode != ForwardMode.EXTEND:
            return False
        if getattr(self, "_fp8_snapshot", None) is None:
            return False
        mip = getattr(self.forward_metadata, "multi_item_params", None)
        if mip is not None and mip.is_enabled():
            return False
        pfx_list, ext_list, _ = self._fp8_snapshot
        for pfx, ext in zip(pfx_list, ext_list):
            if pfx + ext < self._min_tokens:
                return False
            if ext < self._min_extend:
                return False
        return True

    def _pv8_mask(self, num_q_heads: int,
                  device: torch.device) -> tuple[torch.Tensor, bool, bool]:
        cached = self._pv8_mask_cache.get((num_q_heads, device))
        if cached is not None:
            return cached
        mask = torch.ones(num_q_heads, dtype=torch.uint8, device=device)
        env = self._bf16_heads_env.strip()
        if env == "all":
            mask.zero_()
        elif env:
            for tok in env.split(","):
                mask[int(tok)] = 0
        any_pv8 = bool(mask.max().item())
        all_pv8 = bool(mask.min().item())
        self._pv8_mask_cache[(num_q_heads, device)] = (mask, any_pv8, all_pv8)
        return mask, any_pv8, all_pv8

    # ---- forward ---------------------------------------------------------

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        if not (self._batch_qualifies(forward_batch) and self._layer_supported(layer)):
            return super().forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache
            )

        ext = _load_kernel()
        if self._ws is None:
            self._ws = FP8PrefillWorkspace(q.device)

        # 1. write the chunk's K/V into the paged pool first (the fused
        #    kernel then reads prefix + chunk uniformly from the pool),
        #    mirroring the parent's paged-branch save.
        if k is not None and save_kv_cache:
            assert v is not None
            self.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v,
                *self._kv_write_scales(layer),
            )

        num_q_heads = layer.tp_q_head_num
        mask, any_pv8, all_pv8 = self._pv8_mask(num_q_heads, q.device)
        k_buffer = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_buffer = self.token_to_kv_pool.get_value_buffer(layer.layer_id)

        q3 = q.view(-1, num_q_heads, HEAD_DIM)
        out = torch.empty_like(q3)

        # 2. per request: gather+quantize K/V, quantize Q, run the kernel.
        # All indices/lengths come from the metadata-time snapshot ONLY.
        pfx_list, ext_list, idx_list = self._fp8_snapshot
        if self._debug and layer.layer_id == 3:
            try:
                live_pfx = [int(x) for x in (forward_batch.extend_prefix_lens_cpu or [])]
                live_ext = [int(x) for x in (forward_batch.extend_seq_lens_cpu or [])]
                live_rows = forward_batch.req_pool_indices.tolist()
                r2t = self.req_to_token_pool.req_to_token
                drift = []
                for i, (p_, e_) in enumerate(zip(pfx_list, ext_list)):
                    live_idx = r2t[live_rows[i], : p_ + e_].to(torch.long)
                    ne = int((live_idx != idx_list[i]).sum().item())
                    drift.append(ne)
                logger.info(
                    "[fp8_prefill DEBUG] L3 nreq=%d snap_pfx=%s snap_ext=%s "
                    "live_pfx=%s live_ext=%s q_rows=%d sum_ext=%d "
                    "idx_drift_per_req=%s cache_loc=%s",
                    len(pfx_list), pfx_list, ext_list, live_pfx, live_ext,
                    q.shape[0], sum(ext_list), drift,
                    forward_batch.out_cache_loc[:3].tolist()
                    if forward_batch.out_cache_loc is not None else None,
                )
            except Exception as exc:  # never break serving for debug
                logger.info("[fp8_prefill DEBUG] probe failed: %r", exc)
        row0 = 0
        for i, (pfx, ext_len) in enumerate(zip(pfx_list, ext_list)):
            idx = idx_list[i]

            kv = gather_quantize_kv(
                self._ws, k_buffer, v_buffer, idx,
                need_vt8=any_pv8, need_vb16=not all_pv8,
                center_k=self._center_k, qk_i8=self._qk_i8,
                rotate=self._rotate,
            )
            q_req = q3[row0:row0 + ext_len]
            q8, qscale, mpad = quantize_q(self._ws, q_req, layer.scaling,
                                          qk_i8=self._qk_i8,
                                          rotate=self._rotate)
            o = self._ws.get("o", (num_q_heads, mpad, HEAD_DIM), torch.bfloat16)

            ext.fp8_prefill_attn(
                q8, kv["k8"], kv["vt8"], kv["vb16"], o,
                qscale, kv["kscale"], kv["vscale"],
                kv["vlog2r"], kv["vinvr"], kv["vmean"], mask,
                kv["n"], pfx, any_pv8, all_pv8, self._qk_i8,
            )
            out[row0:row0 + ext_len] = o[:, :ext_len].permute(1, 0, 2)
            row0 += ext_len

        return out.view(-1, num_q_heads * HEAD_DIM)
