#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Paired MRCR runner: one served model, two attention backends, same items.

MRCR (OpenAI's Multi-Round Co-reference Resolution set, ``openai/mrcr`` on the
Hugging Face Hub) asks a model to reproduce one specific earlier assistant turn
out of a long synthetic conversation, prefixed with a random string. It is used
here because it is a *long-context retrieval* task whose score is continuous,
which makes a paired per-item difference informative at modest item counts.

    prepare  download the pinned revision, bin items by their EXACT length
             under the served model's chat template, write a selection
    arm      run the selection against a live server, grade, cache per item
    compare  paired per-bin deltas with bootstrap CIs, plus speed

Run ``arm`` twice: once against a server booted on the stock backend, once
against a server booted on the candidate backend, never the same server. See
``README.md`` in this directory for why each rule below exists.

The dataset is public (MIT) and is not redistributed here -- only its revision
is pinned, so a selection is reproducible from the identifier alone.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import pathlib
import statistics
import sys
import time

import serving_common as sc

SUITE = "mrcr"
DATASET = "openai/mrcr"

# Pinned because MRCR's 2025-12-05 bugfix re-uploaded a chunk of the rows with
# corrected ground truth. A cached item is only trustworthy against a fixed
# revision, and a revision bump must invalidate the items it touched.
DEFAULT_REVISION = "f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d"

# (bin name, exclusive lower bound, inclusive upper bound) in chat-template
# tokens. The bins are the length ladder: the whole point of the instrument is
# that a kernel defect may appear only past some sequence length.
BINS = (
    ("32k", 16384, 32768),
    ("131k", 65536, 131072),
    ("anchor", 262144, 524288),
)

# Items admitted per bin per time budget. ``full`` reproduces the reference
# run's counts exactly; the shorter tiers are prefixes of the same ordered
# selection, so their per-item records are reusable by a later full run.
# The arithmetic behind each row is in README.md.
TIER_COUNTS: dict[str, dict[str, int]] = {
    "30min": {"32k": 6, "131k": 5, "anchor": 6},
    "60min": {"32k": 10, "131k": 8, "anchor": 14},
    "full": {"32k": 40, "131k": 30, "anchor": 45},
}
# Configs other than the 2-needle default run the anchor bin only.
TIER_COUNTS_ANCHOR_ONLY: dict[str, dict[str, int]] = {
    "30min": {"anchor": 8},
    "60min": {"anchor": 12},
    "full": {"anchor": 12},
}

# Format gate: if the REFERENCE arm cannot even follow the instruction on the
# easiest items, MRCR is measuring the model's instruction-following, not the
# kernel, and every later number would be noise dressed as a result.
FORMAT_GATE_N = 5
FORMAT_GATE_MIN = 0.05

DEFAULT_MAX_PROMPT_TOKENS = 500_000


def log(msg: str) -> None:
    sc.log(SUITE, msg)


def tier_counts(config: str, tier: str) -> dict[str, int]:
    table = TIER_COUNTS if config == "2needle" else TIER_COUNTS_ANCHOR_ONLY
    return table[tier]


def bins_for(config: str) -> tuple[tuple[str, int, int], ...]:
    if config == "2needle":
        return BINS
    return tuple(b for b in BINS if b[0] == "anchor")


def selection_path(out_dir: pathlib.Path, config: str) -> pathlib.Path:
    name = "selection.json" if config == "2needle" else f"selection-{config}.json"
    return out_dir / name


def results_root(out_dir: pathlib.Path, config: str) -> pathlib.Path:
    return out_dir / ("results" if config == "2needle" else f"results-{config}")


def parquet_files(config: str) -> tuple[str, ...]:
    return (f"{config}/{config}_0.parquet", f"{config}/{config}_1.parquet")


def grade(response: str, answer: str, random_string: str) -> float:
    """The dataset card's rule, with one documented tolerance.

    No random-string prefix means the model did not comply, and the score is 0.
    Otherwise the score is ``SequenceMatcher`` ratio against the expected
    ``random_string + answer``.

    The tolerance: leading whitespace is stripped first. Chat-tuned models
    routinely open with a newline pair before complying, and failing that
    strictly would zero the entire instrument on BOTH arms -- which measures
    the chat template, not the kernel. The strip is applied identically to
    both arms, so it cannot move the paired difference.
    """
    response = response.lstrip()
    if not response.startswith(random_string):
        return 0.0
    return difflib.SequenceMatcher(None, response, random_string + answer).ratio()


def _load_frame(config: str, revision: str):
    import pandas as pd
    from huggingface_hub import hf_hub_download

    frames = []
    for name in parquet_files(config):
        path = hf_hub_download(DATASET, name, repo_type="dataset", revision=revision)
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


def cmd_prepare(args: argparse.Namespace) -> int:
    """Select the FULL item set once; the tiers are prefixes of it.

    Selection is deterministic: candidates are ordered by ``n_chars`` and taken
    in order, so the 30-minute set is the head of the 60-minute set is the head
    of the full set. That is what lets a short run be extended into a long one
    without discarding a single already-paid-for item.
    """
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_frame(args.config, args.revision)
    log(f"config={args.config} rows={len(frame)} columns={list(frame.columns)}")
    tokenizer = sc.load_tokenizer(args.tokenizer or args.model, args.trust_remote_code)

    full_counts = tier_counts(args.config, "full")
    selection: list[dict[str, object]] = []
    for bin_name, lo, hi in bins_for(args.config):
        want = full_counts[bin_name]
        # ``n_chars`` is the dataset's own column -- characters, NOT tokens.
        # It is used only as a coarse prefilter to avoid tokenizing all 2,400
        # rows; the bin edges themselves are enforced on the exact token count
        # under the served model's chat template, below.
        candidates = frame[
            (frame["n_chars"] > lo * 3.0) & (frame["n_chars"] <= hi * 4.8)
        ].sort_values("n_chars")
        taken = 0
        for idx, row in candidates.iterrows():
            if taken >= want:
                break
            messages = json.loads(row["prompt"])
            try:
                n_tokens = sc.chat_template_token_len(tokenizer, messages)
            except Exception as exc:  # noqa: BLE001 - one bad row must not stop selection
                log(f"tokenize failed for row {idx}: {exc!r}; skipping")
                continue
            if not (lo < n_tokens <= hi) or n_tokens > args.max_prompt_tokens:
                continue
            item_id = hashlib.sha256(
                f"{args.revision}:{idx}:{row['random_string_to_prepend']}".encode()
            ).hexdigest()[:16]
            selection.append(
                {
                    "item_id": item_id,
                    "bin": bin_name,
                    "row_index": int(idx),
                    "n_chars": int(row["n_chars"]),
                    "n_tokens_chat_template": int(n_tokens),
                    "answer_chars": len(row["answer"]),
                    # Enough budget to reproduce the target turn, no more: a
                    # runaway generation would dominate the e2e comparison.
                    "max_tokens": min(4096, int(len(row["answer"]) / 3 * 1.5) + 128),
                }
            )
            taken += 1
        log(f"bin {bin_name}: selected {taken}/{want}")
        if taken < want:
            log(f"WARNING: bin {bin_name} short ({taken} < {want})")

    selection_path(args.out_dir, args.config).write_text(
        json.dumps(
            {
                "dataset": DATASET,
                "config": args.config,
                "revision": args.revision,
                "model": args.model,
                "tokenizer": args.tokenizer or args.model,
                "max_prompt_tokens": args.max_prompt_tokens,
                "tier_counts": (
                    TIER_COUNTS if args.config == "2needle" else TIER_COUNTS_ANCHOR_ONLY
                ),
                "items": selection,
            },
            indent=1,
        )
        + "\n"
    )
    log(f"selection -> {selection_path(args.out_dir, args.config)} ({len(selection)} items)")
    return 0


def tier_items(selection: dict, config: str, tier: str) -> list[dict]:
    """Head-of-bin slice for the requested budget, then cheapest-first order.

    Bins run in ladder order (short before long) so that a run interrupted by
    its budget still carries the format gate and the cheap end of the ladder.
    """
    counts = tier_counts(config, tier)
    per_bin: dict[str, list[dict]] = {}
    for item in selection["items"]:
        per_bin.setdefault(item["bin"], []).append(item)
    order = {name: i for i, (name, *_rest) in enumerate(bins_for(config))}
    chosen: list[dict] = []
    for bin_name, items in per_bin.items():
        chosen.extend(items[: counts.get(bin_name, 0)])
    return sorted(chosen, key=lambda it: (order[it["bin"]], it["n_tokens_chat_template"]))


def cmd_arm(args: argparse.Namespace) -> int:
    selection = json.loads(selection_path(args.out_dir, args.config).read_text())
    revision = selection["revision"]
    items = tier_items(selection, args.config, args.tier)
    if not items:
        raise SystemExit("no items selected; run `prepare` first")

    server = sc.build_server(args)
    log(f"server {server.describe()}")
    info = server.server_info()
    backend = sc.check_posture(info, args.arm, args.expect_backend, args.reject_backend)
    log(f"posture OK: arm={args.arm} backend={backend}")

    frame = _load_frame(args.config, revision)
    rows = {int(it["row_index"]): frame.loc[it["row_index"]] for it in items}

    out_dir = results_root(args.out_dir, args.config) / args.arm
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "arm.json").write_text(
        json.dumps(
            {
                "arm": args.arm,
                "reference": args.reference,
                "backend": backend,
                "tier": args.tier,
                "config": args.config,
                "revision": revision,
                "model": args.model,
                "server_args": info.get("server_args"),
                "chat_template_kwargs": sc.parse_template_kwargs(args.chat_template_kwargs),
            },
            indent=1,
        )
        + "\n"
    )

    template_kwargs = sc.parse_template_kwargs(args.chat_template_kwargs)
    sc.warmup(server, SUITE, args.warmup_tokens)

    gate_bin = bins_for(args.config)[0][0]
    gate_active = args.format_gate == "always" or (args.format_gate == "auto" and args.reference)
    gate_scores: list[float] = []

    for position, item in enumerate(items, 1):
        cache = out_dir / f"{item['item_id']}.json"
        if cache.exists():
            record = json.loads(cache.read_text())
            log(
                f"[{position}/{len(items)}] {item['bin']} {item['item_id']} "
                f"cached score={record['score']:.3f}"
            )
        else:
            row = rows[item["row_index"]]
            messages = json.loads(row["prompt"])
            # Flush before EVERY scored item: no prefix from a previous item,
            # and no warmup residue, may survive into a measurement.
            server.flush_cache()
            started = time.time()
            response = server.chat(
                args.model,
                messages,
                max_tokens=item["max_tokens"],
                chat_template_kwargs=template_kwargs,
                logprobs=args.logprobs,
            )
            elapsed = time.time() - started
            text = sc.content_of(response)
            score = grade(text, row["answer"], row["random_string_to_prepend"])
            usage = response.get("usage", {})
            record = {
                "item_id": item["item_id"],
                "bin": item["bin"],
                "arm": args.arm,
                "score": score,
                "elapsed_s": round(elapsed, 1),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                # STRICT compliance, deliberately unlike `score`, which grades
                # after an lstrip. A run where every score is high while
                # prefix_ok is uniformly False is a chat template that opens
                # with whitespace -- worth knowing, and invisible otherwise.
                "prefix_ok": text.startswith(row["random_string_to_prepend"]),
                # Binds the score to the exact bytes that produced it: two runs
                # can score identically while differing token by token, and
                # only the digest can tell those apart afterwards.
                "response_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "response_text": text,
                "token_margins": sc.token_margins(response),
                "revision": revision,
                "backend": backend,
            }
            cache.write_text(json.dumps(record) + "\n")
            log(
                f"[{position}/{len(items)}] {item['bin']} {item['item_id']} "
                f"score={score:.3f} prefix_ok={record['prefix_ok']} {elapsed:.0f}s "
                f"tok={usage.get('prompt_tokens')}"
            )

        if gate_active and item["bin"] == gate_bin and len(gate_scores) < FORMAT_GATE_N:
            gate_scores.append(record["score"])
            if len(gate_scores) == FORMAT_GATE_N:
                mean = sum(gate_scores) / len(gate_scores)
                if mean < args.format_gate_min:
                    log(
                        f"FORMAT-GATE FAIL: arm {args.arm} mean {mean:.3f} < "
                        f"{args.format_gate_min} over the first {FORMAT_GATE_N} "
                        f"{gate_bin} items -- this model is not following the MRCR "
                        "instruction, so the run would measure the fine-tune. Aborting."
                    )
                    return 3
                log(f"format gate PASS: mean {mean:.3f} over {len(gate_scores)} items")

    log(f"arm {args.arm} complete ({len(items)} items, tier {args.tier})")
    return 0


def _read_arm(root: pathlib.Path, arm: str, item_id: str) -> dict | None:
    path = root / arm / f"{item_id}.json"
    return json.loads(path.read_text()) if path.exists() else None


def cmd_compare(args: argparse.Namespace) -> int:
    selection = json.loads(selection_path(args.out_dir, args.config).read_text())
    root = results_root(args.out_dir, args.config)
    items = tier_items(selection, args.config, args.tier)

    backends = {}
    for arm in (args.reference_arm, args.candidate_arm):
        meta = root / arm / "arm.json"
        backends[arm] = json.loads(meta.read_text()).get("backend") if meta.exists() else None
    if (
        backends[args.reference_arm]
        and backends[args.reference_arm] == backends[args.candidate_arm]
    ):
        raise SystemExit(
            f"both arms record the same active backend "
            f"({backends[args.reference_arm]!r}); this is not a paired comparison"
        )

    summary: dict[str, object] = {
        "dataset": DATASET,
        "revision": selection["revision"],
        "config": args.config,
        "tier": args.tier,
        "reference_arm": args.reference_arm,
        "candidate_arm": args.candidate_arm,
        "reference_backend": backends[args.reference_arm],
        "candidate_backend": backends[args.candidate_arm],
        "bootstrap": {"resamples": args.bootstrap, "seed": args.seed, "method": "percentile"},
        "bins": {},
        "paired_items": 0,
    }
    bins_out: dict[str, object] = summary["bins"]  # type: ignore[assignment]

    for bin_name, *_edges in bins_for(args.config):
        pairs: list[tuple[float, float]] = []
        ref_times: list[float] = []
        cand_times: list[float] = []
        prompt_tokens: list[int] = []
        completion_tokens: list[int] = []
        for item in items:
            if item["bin"] != bin_name:
                continue
            ref = _read_arm(root, args.reference_arm, item["item_id"])
            cand = _read_arm(root, args.candidate_arm, item["item_id"])
            if ref is None or cand is None:
                continue
            pairs.append((ref["score"], cand["score"]))
            ref_times.append(ref.get("elapsed_s") or 0.0)
            cand_times.append(cand.get("elapsed_s") or 0.0)
            for record in (ref, cand):
                prompt_tokens.append(record.get("prompt_tokens") or 0)
                completion_tokens.append(record.get("completion_tokens") or 0)
        if not pairs:
            continue
        deltas = [cand - ref for ref, cand in pairs]
        lo, hi = sc.paired_bootstrap(deltas, args.bootstrap, args.seed)
        speed = sc.speed_block(ref_times, cand_times)
        speed["mean_prompt_tokens"] = int(statistics.mean(prompt_tokens))
        speed["mean_completion_tokens"] = int(statistics.mean(completion_tokens))
        bins_out[bin_name] = {
            "n": len(pairs),
            "reference_mean": statistics.mean(ref for ref, _ in pairs),
            "candidate_mean": statistics.mean(cand for _, cand in pairs),
            "delta_mean": statistics.mean(deltas),
            "delta_ci95": [lo, hi],
            "pairs": pairs,
            "speed": speed,
        }
        summary["paired_items"] += len(pairs)  # type: ignore[operator]

    name = (
        f"{SUITE}-summary.json"
        if args.config == "2needle"
        else f"{SUITE}-summary-{args.config}.json"
    )
    (args.out_dir / name).write_text(json.dumps(summary, indent=1) + "\n")
    for bin_name, block in bins_out.items():
        speed = block["speed"]
        print(
            f"{bin_name}: n={block['n']} ref={block['reference_mean']:.4f} "
            f"cand={block['candidate_mean']:.4f} delta={block['delta_mean']:+.4f} "
            f"CI95=[{block['delta_ci95'][0]:+.4f},{block['delta_ci95'][1]:+.4f}] "
            f"| e2e ref {speed['reference_mean_e2e_s']}s "
            f"cand {speed['candidate_mean_e2e_s']}s speedup {speed['speedup_e2e']}x"
        )
    print(f"summary -> {args.out_dir / name}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        default="2needle",
        choices=("2needle", "4needle", "8needle"),
        help="MRCR needle count (default %(default)s)",
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="pinned dataset revision (default: the reference run's pin)",
    )
    sc.add_server_args(parser, SUITE)
    sub = parser.add_subparsers(dest="cmd", required=True)

    prepare = sub.add_parser("prepare", help="select and bin items (no server needed)")
    sc.add_model_args(prepare)
    prepare.add_argument("--max-prompt-tokens", type=int, default=DEFAULT_MAX_PROMPT_TOKENS)
    prepare.set_defaults(fn=cmd_prepare)

    arm = sub.add_parser("arm", help="run one backend's arm against a live server")
    sc.add_model_args(arm)
    sc.add_tier_arg(arm)
    sc.add_arm_args(arm)
    arm.add_argument(
        "--format-gate",
        choices=("auto", "always", "never"),
        default="auto",
        help="auto = gate the reference arm only (default %(default)s)",
    )
    arm.add_argument("--format-gate-min", type=float, default=FORMAT_GATE_MIN)
    arm.add_argument(
        "--no-logprobs",
        dest="logprobs",
        action="store_false",
        help="skip top-2 logprob margins (they add a little per-request overhead)",
    )
    arm.set_defaults(fn=cmd_arm, logprobs=True)

    compare = sub.add_parser("compare", help="paired per-bin summary of two finished arms")
    sc.add_tier_arg(compare)
    sc.add_compare_args(compare)
    compare.set_defaults(fn=cmd_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except sc.PostureError as exc:
        raise SystemExit(f"posture guard: {exc}") from exc


if __name__ == "__main__":
    sys.exit(main())
