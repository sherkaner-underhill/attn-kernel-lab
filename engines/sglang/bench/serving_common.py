#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Shared machinery for the paired serving-quality runners in this directory.

Everything here exists to make ONE comparison honest: the same served model,
the same items, the same instrument, run twice -- once with the engine's stock
attention backend, once with the candidate backend -- so that a per-item score
difference can only come from the kernel under test.

The pieces below are the parts of that discipline that are not specific to a
single benchmark:

* :class:`Server` -- endpoint, optional bearer token, and the three engine
  calls the runners make (chat completion, cache flush, server info). The
  token is read from the environment and is never logged, echoed, or written
  into a result record.
* :func:`check_posture` -- the guard that refuses to measure an arm the server
  is not actually running.
* :func:`warmup` -- the unscored request that absorbs one-time boot cost.
* :func:`chat_template_token_len` -- exact prompt length under the SERVED
  model's chat template, which is what the length bins are defined in.
* :func:`paired_bootstrap` -- the paired-difference bootstrap used for every
  confidence interval the runners report.

Nothing here knows a model name, a host, or a filesystem layout outside this
repository: all three arrive as flags or environment variables.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import statistics
import time
import urllib.error
import urllib.request
from typing import Any, Iterable, Sequence

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_API_KEY_ENV = "SGLANG_API_KEY"
DEFAULT_BASE_URL_ENV = "SGLANG_BASE_URL"
DEFAULT_REQUEST_TIMEOUT = 1800
DEFAULT_FLUSH_TIMEOUT = 300

# Absorbs allocator growth and any first-call kernel JIT compile so that the
# first SCORED item measures steady state rather than warmup. 16384 tokens is
# long enough to take the engine down its extend/prefill path.
DEFAULT_WARMUP_TOKENS = 16384

# Thinking-style chat templates spend the whole generation budget inside the
# reasoning block and return empty content for short answers. Disabling it is
# the default because the instruments here grade the visible content; the value
# is sent verbatim, so a template without the key simply ignores it.
DEFAULT_CHAT_TEMPLATE_KWARGS = '{"enable_thinking": false}'

# Bootstrap defaults. The seed is fixed so a comparison is reproducible from
# the per-item records alone.
DEFAULT_BOOTSTRAP = 10000
DEFAULT_SEED = 20260830

TIERS = ("30min", "60min", "full")


class PostureError(RuntimeError):
    """The server is not in the state the requested arm claims to measure."""


def log(prefix: str, msg: str) -> None:
    print(f"[{prefix} {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Server:
    """Minimal client for the serving endpoint under measurement.

    ``api_key`` is read from an environment variable (default
    ``SGLANG_API_KEY``) and used only to build an ``Authorization`` header. It
    is never printed and never stored in a result record; :meth:`describe`
    reports only whether a token was found.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        api_key_env: str = DEFAULT_API_KEY_ENV,
        timeout: int = DEFAULT_REQUEST_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.timeout = timeout
        self._token = (os.environ.get(api_key_env) or "").strip()

    def describe(self) -> str:
        state = "with bearer token" if self._token else "no bearer token"
        return f"{self.base_url} ({state} from ${self.api_key_env})"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = "Bearer " + self._token
        return headers

    def post(self, path: str, payload: Any, timeout: int | None = None) -> dict[str, Any]:
        """POST JSON and decode the reply.

        A non-JSON body is returned as ``{"raw": ...}`` rather than raising:
        some engine control endpoints answer with a bare status string, and a
        cache flush must never be able to abort a run.
        """
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode() if payload is not None else b"",
            headers=self._headers(),
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
            body = response.read()
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {"raw": body.decode(errors="replace")}

    def get(self, path: str, timeout: int = 60) -> dict[str, Any]:
        request = urllib.request.Request(self.base_url + path, headers=self._headers())
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())

    def server_info(self) -> dict[str, Any]:
        return self.get("/get_server_info")

    def flush_cache(self, timeout: int = DEFAULT_FLUSH_TIMEOUT) -> None:
        """Drop the prefix cache. Never raises: an item is scored either way.

        Every SCORED item is preceded by a flush so that no prompt content --
        and no radix-tree prefix from a previous item -- survives into the
        measurement. The one deliberate exception is documented at its call
        site in ``infbench_run.py``.
        """
        try:
            self.post("/flush_cache", None, timeout)
        except (urllib.error.URLError, OSError, TimeoutError):
            pass

    def chat(
        self,
        model: str,
        messages: Sequence[dict[str, Any]],
        max_tokens: int,
        chat_template_kwargs: dict[str, Any] | None = None,
        logprobs: bool = False,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """One greedy chat completion.

        ``temperature`` is pinned to 0.0 on both arms: the comparison is of
        kernels, so any sampling entropy would be measured as kernel noise.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        if chat_template_kwargs:
            payload["chat_template_kwargs"] = chat_template_kwargs
        if logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = 2
        return self.post("/v1/chat/completions", payload, timeout)


def content_of(response: dict[str, Any]) -> str:
    return (response.get("choices") or [{}])[0].get("message", {}).get("content") or ""


def token_margins(response: dict[str, Any], limit: int = 400) -> list[list[Any]]:
    """Per-token top-1 minus top-2 logprob, where the server reports them.

    A margin near zero is a token the model was nearly indifferent about, which
    is where a numerically different kernel is most likely to flip an output.
    Servers that do not return ``top_logprobs`` yield an empty list rather than
    an error, so this is safe to request unconditionally.
    """
    entries = ((response.get("choices") or [{}])[0].get("logprobs") or {}).get("content") or []
    out: list[list[Any]] = []
    for entry in entries[:limit]:
        tops = entry.get("top_logprobs") or []
        margin = entry["logprob"] - tops[1]["logprob"] if len(tops) > 1 else None
        out.append([entry.get("token"), round(margin, 4) if margin is not None else None])
    return out


def check_posture(
    info: dict[str, Any],
    arm: str,
    expect_backend: str,
    reject_backend: Iterable[str] = (),
    allow_speculative: bool = False,
) -> str:
    """Refuse to measure an arm the server is not running.

    A paired comparison is worthless if both arms were served by the same
    backend, and nothing downstream of the request can detect that. So the
    active prefill attention backend is read from the server itself and matched
    against what the caller says it is measuring, before any item is scored.

    Speculative decoding is refused outright: it changes which tokens are
    produced by which path, so a per-item quality delta would no longer isolate
    the attention kernel.
    """
    backend = str(info.get("prefill_attention_backend") or info.get("attention_backend"))
    if expect_backend not in backend:
        raise PostureError(
            f"arm {arm!r} expects a backend containing {expect_backend!r} "
            f"but the server reports {backend!r}"
        )
    for forbidden in reject_backend:
        if forbidden in backend:
            raise PostureError(
                f"arm {arm!r} must not run {forbidden!r} but the server reports {backend!r}"
            )
    speculative = info.get("speculative_algorithm")
    if not allow_speculative and speculative not in (None, "", "NONE", "None"):
        raise PostureError(
            f"speculative decoding is ON ({speculative!r}); benchmark boots must be spec-off"
        )
    return backend


def warmup(server: Server, prefix: str, tokens: int = DEFAULT_WARMUP_TOKENS) -> None:
    """Issue one UNSCORED long request, then flush.

    This is not a measurement and is never recorded. It exists so that the
    first scored item is not paying for allocator growth or a one-time kernel
    compile on the candidate arm, which would otherwise show up as a quality-
    neutral but very real end-to-end regression at item 1.
    """
    if tokens <= 0:
        return
    log(prefix, f"warmup request (unscored, {tokens} tokens)...")
    try:
        server.post(
            "/generate",
            {
                "input_ids": [15000] * tokens,
                "sampling_params": {"temperature": 0.0, "max_new_tokens": 4},
            },
        )
    except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
        log(prefix, f"warmup failed ({exc!r}); continuing")
    server.flush_cache()


def load_tokenizer(tokenizer: str, trust_remote_code: bool = False):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=trust_remote_code)


def _normalise_encoding(encoded: Any) -> list[int]:
    """Reduce whatever ``apply_chat_template`` returned to one list of ids.

    Some tokenizers return a ``BatchEncoding`` or a plain dict here. ``len()``
    of those is the number of KEYS, not the number of tokens -- which silently
    puts every prompt in the smallest bin and selects nothing. Normalising
    before measuring is the fix, and it is why binning is done through this
    helper everywhere rather than inline.
    """
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    elif isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    if encoded and isinstance(encoded[0], (list, tuple)):
        encoded = encoded[0]
    return list(encoded)


def chat_template_token_len(tokenizer, messages: Sequence[dict[str, Any]]) -> int:
    """Exact prompt length under the SERVED model's chat template.

    The bins are defined in the tokens the server will actually process, not in
    characters and not in some other model's vocabulary, because the quantity
    the kernel's behaviour depends on is sequence length at attention.
    """
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    return len(_normalise_encoding(encoded))


def plain_token_len(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def paired_bootstrap(
    deltas: Sequence[float],
    iterations: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float]:
    """Percentile CI95 of the mean PAIRED difference, resampling items.

    Paired because both arms score the same items: the per-item difference
    removes item difficulty, which dominates the raw per-arm variance on these
    benchmarks. Resampling is over items, so the interval answers "would
    another draw of items from this pool show the same sign?".
    """
    rng = random.Random(seed)
    boots: list[float] = []
    for _ in range(iterations):
        sample = [deltas[rng.randrange(len(deltas))] for _ in deltas]
        boots.append(sum(sample) / len(sample))
    boots.sort()
    lo = boots[max(0, round(0.025 * iterations) - 1)]
    hi = boots[min(iterations - 1, round(0.975 * iterations) - 1)]
    return lo, hi


def speed_block(reference: Sequence[float], candidate: Sequence[float]) -> dict[str, Any]:
    """Mean end-to-end seconds per arm plus the ratio.

    Speed rides alongside quality in the same record on purpose: a kernel that
    is faster and a kernel that is more accurate are different claims, and
    reporting them from one paired run is what stops either from being quoted
    without the other.
    """
    ref_mean = statistics.mean(reference) if reference else None
    cand_mean = statistics.mean(candidate) if candidate else None
    speedup = None
    if ref_mean is not None and cand_mean:
        speedup = round(ref_mean / cand_mean, 3)
    return {
        "reference_mean_e2e_s": round(ref_mean, 1) if ref_mean is not None else None,
        "candidate_mean_e2e_s": round(cand_mean, 1) if cand_mean is not None else None,
        "speedup_e2e": speedup,
    }


def default_out_dir(suite: str) -> pathlib.Path:
    """Repository-relative default, so a run needs no path flags at all."""
    return pathlib.Path(__file__).resolve().parent / "runs" / suite


def add_server_args(parser: argparse.ArgumentParser, suite: str) -> None:
    parser.add_argument(
        "--base-url",
        default=os.environ.get(DEFAULT_BASE_URL_ENV, DEFAULT_BASE_URL),
        help=f"serving endpoint (env {DEFAULT_BASE_URL_ENV}; default {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--api-key-env",
        default=DEFAULT_API_KEY_ENV,
        help=(
            "environment variable holding the bearer token, if the server needs one "
            f"(default {DEFAULT_API_KEY_ENV}); the value is never logged or recorded"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=default_out_dir(suite),
        help="selection, per-item records, and summaries (default: runs/%s here)" % suite,
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=DEFAULT_REQUEST_TIMEOUT,
        help="per-request timeout in seconds (default %(default)s)",
    )


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        required=True,
        help="model name the server answers to, and the default tokenizer source",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="tokenizer id or directory for length binning (default: --model)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="allow the tokenizer to execute repository code (needed by some chat templates)",
    )


def add_tier_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tier",
        choices=TIERS,
        default="full",
        help="time budget for one arm; selects how many items per bin (default %(default)s)",
    )


def add_arm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--arm", required=True, help="label for this arm; names the record folder")
    parser.add_argument(
        "--expect-backend",
        required=True,
        help="substring the server's active prefill attention backend MUST contain",
    )
    parser.add_argument(
        "--reject-backend",
        action="append",
        default=[],
        help="substring the active backend must NOT contain (repeatable)",
    )
    parser.add_argument(
        "--reference",
        action="store_true",
        help="this arm is the reference/baseline arm of the pair",
    )
    parser.add_argument(
        "--chat-template-kwargs",
        default=DEFAULT_CHAT_TEMPLATE_KWARGS,
        help=(
            "JSON forwarded as chat_template_kwargs; the default disables thinking mode "
            "and MUST be identical on both arms. Pass '{}' to send none."
        ),
    )
    parser.add_argument(
        "--warmup-tokens",
        type=int,
        default=DEFAULT_WARMUP_TOKENS,
        help="length of the unscored warmup request; 0 disables it (default %(default)s)",
    )


def add_compare_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--reference-arm", required=True, help="arm label of the baseline")
    parser.add_argument("--candidate-arm", required=True, help="arm label of the kernel under test")
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)


def parse_template_kwargs(raw: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise SystemExit("--chat-template-kwargs must be a JSON object")
    return value


def build_server(args: argparse.Namespace) -> Server:
    return Server(
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        timeout=args.request_timeout,
    )
