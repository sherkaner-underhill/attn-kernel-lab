#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Paired InfiniteBench runner: kv_retrieval + longbook_choice_eng.

Two long-context tasks with very different shapes, which is the point of
running both:

``kv_retrieval``
    A large JSON object and one key; the answer is a value buried in it. Every
    item has its own context, so nothing is shared and every item pays a full
    cold prefill. Scored by exact answer-string containment in the response,
    which is the rule the standard evaluation harness uses.

``longbook_choice_eng``
    Multiple choice over a whole book. Questions are GROUPED BY BOOK so that a
    book's prefill is paid once and reused by its (~4) questions.

    Two deliberate deviations, both applied identically to the two arms so the
    paired difference stays valid: the question is answered *generatively* (the
    model emits the option letter) rather than by loglikelihood ranking, for
    harness simplicity; and the prefix cache is flushed BETWEEN books only.

Subcommands mirror ``mrcr_run.py``: ``prepare`` / ``arm`` / ``compare``.
Run ``arm`` twice against two separately booted servers. See ``README.md``.

The dataset is public and is not redistributed here; the resolved revision is
recorded in the selection so a run is reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import statistics
import sys
import time

import serving_common as sc

SUITE = "infbench"
DATASET = "xinrongzhang2022/InfiniteBench"
KV_FILE = "kv_retrieval.jsonl"
MC_FILE = "longbook_choice_eng.jsonl"

# Items admitted per time budget. ``full`` reproduces the reference run's
# counts; the shorter tier is a prefix of the same ordered selection. The whole
# InfiniteBench lane already fits inside a 60-minute arm, so ``60min`` and
# ``full`` coincide here -- the headroom is real, not padding.
TIER_COUNTS: dict[str, dict[str, int]] = {
    "30min": {"kv": 18, "books": 7},
    "60min": {"kv": 30, "books": 12},
    "full": {"kv": 30, "books": 12},
}

DEFAULT_MAX_PROMPT_TOKENS = 500_000
KV_MAX_TOKENS = 128
MC_MAX_TOKENS = 16
LETTERS = "ABCD"

KV_PROMPT = (
    "Extract the value corresponding to the specified key in the JSON "
    "object below.\n\n{context}\n\nKey: {key}\nThe value associated with "
    "the specified key is:"
)
MC_PROMPT = (
    "Read the book below and answer the question.\n\n{context}\n\n"
    "Question: {question}\n\nOnly one of the following options is "
    "correct. Reply with the letter of the correct option and nothing "
    "else.\nA. {a}\nB. {b}\nC. {c}\nD. {d}\n\nAnswer:"
)


def log(msg: str) -> None:
    sc.log(SUITE, msg)


def selection_path(out_dir: pathlib.Path) -> pathlib.Path:
    return out_dir / "selection.json"


def _download(revision: str, name: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(DATASET, name, repo_type="dataset", revision=revision)


def cmd_prepare(args: argparse.Namespace) -> int:
    """Resolve and RECORD the dataset revision, then select the full item set.

    The revision is recorded rather than assumed: a cached item is only
    comparable against the same dataset bytes, and this dataset has no
    long-standing published pin to hard-code.
    """
    from huggingface_hub import HfApi

    args.out_dir.mkdir(parents=True, exist_ok=True)
    revision = args.revision or HfApi().dataset_info(DATASET).sha
    log(f"dataset {DATASET} revision {revision}")
    tokenizer = sc.load_tokenizer(args.tokenizer or args.model, args.trust_remote_code)

    full = TIER_COUNTS["full"]
    kv_items: list[dict[str, object]] = []
    with open(_download(revision, KV_FILE)) as handle:
        for line in handle:
            if len(kv_items) >= full["kv"]:
                break
            row = json.loads(line)
            prompt = KV_PROMPT.format(context=row["context"], key=row["input"])
            n_tokens = sc.plain_token_len(tokenizer, prompt)
            if n_tokens > args.max_prompt_tokens:
                continue
            kv_items.append({"id": str(row["id"]), "n_tokens": n_tokens})
    log(f"kv_retrieval: {len(kv_items)} items")

    by_book: dict[str, list[dict]] = {}
    with open(_download(revision, MC_FILE)) as handle:
        for line in handle:
            row = json.loads(line)
            # Books are identified by a digest of their opening characters:
            # the file carries one row per question, and grouping is what makes
            # the shared-prefix reuse possible at all.
            key = hashlib.sha256(row["context"][:4096].encode()).hexdigest()[:12]
            by_book.setdefault(key, []).append(row)

    books: list[dict[str, object]] = []
    for key, rows in by_book.items():
        if len(books) >= full["books"]:
            break
        n_tokens = sc.plain_token_len(tokenizer, rows[0]["context"])
        if n_tokens + 2048 > args.max_prompt_tokens:
            continue
        books.append(
            {
                "book": key,
                "n_tokens_context": n_tokens,
                "question_ids": [str(row["id"]) for row in rows],
            }
        )
        log(f"book {key}: {len(rows)} questions, ~{n_tokens} tokens")
    total_questions = sum(len(book["question_ids"]) for book in books)
    log(f"longbook_choice: {len(books)} books, {total_questions} questions")

    selection_path(args.out_dir).write_text(
        json.dumps(
            {
                "dataset": DATASET,
                "revision": revision,
                "model": args.model,
                "tokenizer": args.tokenizer or args.model,
                "max_prompt_tokens": args.max_prompt_tokens,
                "tier_counts": TIER_COUNTS,
                "kv": kv_items,
                "books": books,
            },
            indent=1,
        )
        + "\n"
    )
    log(f"selection -> {selection_path(args.out_dir)}")
    return 0


def tier_slice(selection: dict, tier: str) -> tuple[list[dict], list[dict]]:
    counts = TIER_COUNTS[tier]
    return selection["kv"][: counts["kv"]], selection["books"][: counts["books"]]


def _load_rows(revision: str, kv_ids: set[str], question_ids: set[str]):
    kv_rows: dict[str, dict] = {}
    mc_rows: dict[str, dict] = {}
    with open(_download(revision, KV_FILE)) as handle:
        for line in handle:
            row = json.loads(line)
            if str(row["id"]) in kv_ids:
                kv_rows[str(row["id"])] = row
    with open(_download(revision, MC_FILE)) as handle:
        for line in handle:
            row = json.loads(line)
            if str(row["id"]) in question_ids:
                mc_rows[str(row["id"])] = row
    return kv_rows, mc_rows


def _record(path: pathlib.Path, payload: dict) -> None:
    path.write_text(json.dumps(payload) + "\n")


def cmd_arm(args: argparse.Namespace) -> int:
    selection = json.loads(selection_path(args.out_dir).read_text())
    revision = selection["revision"]
    kv_items, books = tier_slice(selection, args.tier)
    if not kv_items and not books:
        raise SystemExit("no items selected; run `prepare` first")

    server = sc.build_server(args)
    log(f"server {server.describe()}")
    info = server.server_info()
    backend = sc.check_posture(info, args.arm, args.expect_backend, args.reject_backend)
    log(f"posture OK: arm={args.arm} backend={backend}")

    template_kwargs = sc.parse_template_kwargs(args.chat_template_kwargs)
    kv_rows, mc_rows = _load_rows(
        revision,
        {str(item["id"]) for item in kv_items},
        {qid for book in books for qid in book["question_ids"]},
    )

    out_dir = args.out_dir / "results" / args.arm
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "arm.json").write_text(
        json.dumps(
            {
                "arm": args.arm,
                "reference": args.reference,
                "backend": backend,
                "tier": args.tier,
                "revision": revision,
                "model": args.model,
                "server_args": info.get("server_args"),
                "chat_template_kwargs": template_kwargs,
            },
            indent=1,
        )
        + "\n"
    )
    sc.warmup(server, SUITE, args.warmup_tokens)

    def ask(prompt: str, max_tokens: int):
        return server.chat(
            args.model,
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            chat_template_kwargs=template_kwargs,
            logprobs=args.logprobs,
        )

    for position, item in enumerate(kv_items, 1):
        cache = out_dir / f"kv_{item['id']}.json"
        if cache.exists():
            continue
        row = kv_rows[str(item["id"])]
        # Every kv item has its own context, so a flush before each one costs
        # nothing that could have been reused and guarantees a cold prefill.
        server.flush_cache()
        started = time.time()
        response = ask(KV_PROMPT.format(context=row["context"], key=row["input"]), KV_MAX_TOKENS)
        elapsed = time.time() - started
        text = sc.content_of(response)
        answers = row["answer"] if isinstance(row["answer"], list) else [row["answer"]]
        score = float(any(str(answer) in text for answer in answers))
        usage = response.get("usage", {})
        _record(
            cache,
            {
                "task": "kv_retrieval",
                "id": str(item["id"]),
                "arm": args.arm,
                "score": score,
                "elapsed_s": round(elapsed, 1),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "response_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "response_text": text[:2000],
                "token_margins": sc.token_margins(response),
                "revision": revision,
                "backend": backend,
            },
        )
        log(
            f"kv [{position}/{len(kv_items)}] {item['id']} score={score:.0f} "
            f"{elapsed:.0f}s tok={usage.get('prompt_tokens')}"
        )

    for book in books:
        pending = [q for q in book["question_ids"] if not (out_dir / f"mc_{q}.json").exists()]
        if not pending:
            continue
        # Flush BETWEEN books only. Within a book the prompt prefix is
        # bit-identical across questions, so the engine's radix reuse is exact
        # -- it returns the same KV the cold path would have computed -- and it
        # amortises one book prefill over ~4 questions. Both arms reuse the
        # same way, so this cannot bias the paired difference; it only buys
        # more items per hour.
        server.flush_cache()
        for index, qid in enumerate(book["question_ids"]):
            cache = out_dir / f"mc_{qid}.json"
            if cache.exists():
                continue
            row = mc_rows[qid]
            options = list(row["options"])
            if len(options) != len(LETTERS):
                log(f"skipping {qid}: {len(options)} options, expected {len(LETTERS)}")
                continue
            started = time.time()
            response = ask(
                MC_PROMPT.format(
                    context=row["context"],
                    question=row["input"],
                    a=options[0],
                    b=options[1],
                    c=options[2],
                    d=options[3],
                ),
                MC_MAX_TOKENS,
            )
            elapsed = time.time() - started
            text = sc.content_of(response).strip()
            match = re.search(r"[ABCD]", text)
            picked = match.group(0) if match else "?"
            expected = row["answer"][0] if isinstance(row["answer"], list) else row["answer"]
            correct = LETTERS[options.index(expected)] if expected in options else "?"
            score = float(picked == correct)
            usage = response.get("usage", {})
            _record(
                cache,
                {
                    "task": "longbook_choice",
                    "id": qid,
                    "book": book["book"],
                    "arm": args.arm,
                    "score": score,
                    "picked": picked,
                    "correct": correct,
                    "elapsed_s": round(elapsed, 1),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "response_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    # True for the question that paid the book's cold prefill;
                    # the rest ran warm, and mixing them in a timing mean
                    # without this flag would be dishonest.
                    "cold_book": index == 0,
                    "token_margins": sc.token_margins(response),
                    "revision": revision,
                    "backend": backend,
                },
            )
            log(
                f"mc book {book['book']} [{index + 1}/{len(book['question_ids'])}] "
                f"{qid} score={score:.0f} {elapsed:.0f}s"
            )

    log(f"arm {args.arm} complete (tier {args.tier})")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    selection = json.loads(selection_path(args.out_dir).read_text())
    root = args.out_dir / "results"
    kv_items, books = tier_slice(selection, args.tier)

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

    groups = {
        "kv_retrieval": [f"kv_{item['id']}" for item in kv_items],
        "longbook_choice": [f"mc_{qid}" for book in books for qid in book["question_ids"]],
    }
    summary: dict[str, object] = {
        "dataset": DATASET,
        "revision": selection["revision"],
        "tier": args.tier,
        "reference_arm": args.reference_arm,
        "candidate_arm": args.candidate_arm,
        "reference_backend": backends[args.reference_arm],
        "candidate_backend": backends[args.candidate_arm],
        "bootstrap": {"resamples": args.bootstrap, "seed": args.seed, "method": "percentile"},
        "tasks": {},
    }
    tasks: dict[str, object] = summary["tasks"]  # type: ignore[assignment]

    for task, keys in groups.items():
        pairs: list[tuple[float, float]] = []
        ref_times: list[float] = []
        cand_times: list[float] = []
        for key in keys:
            ref_path = root / args.reference_arm / f"{key}.json"
            cand_path = root / args.candidate_arm / f"{key}.json"
            if not (ref_path.exists() and cand_path.exists()):
                continue
            ref = json.loads(ref_path.read_text())
            cand = json.loads(cand_path.read_text())
            pairs.append((ref["score"], cand["score"]))
            ref_times.append(ref.get("elapsed_s") or 0.0)
            cand_times.append(cand.get("elapsed_s") or 0.0)
        if not pairs:
            continue
        deltas = [cand - ref for ref, cand in pairs]
        lo, hi = sc.paired_bootstrap(deltas, args.bootstrap, args.seed)
        tasks[task] = {
            "n": len(pairs),
            "reference_mean": statistics.mean(ref for ref, _ in pairs),
            "candidate_mean": statistics.mean(cand for _, cand in pairs),
            "delta_mean": statistics.mean(deltas),
            "delta_ci95": [lo, hi],
            "speed": sc.speed_block(ref_times, cand_times),
        }

    (args.out_dir / f"{SUITE}-summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    for task, block in tasks.items():
        speed = block["speed"]
        print(
            f"{task}: n={block['n']} ref={block['reference_mean']:.4f} "
            f"cand={block['candidate_mean']:.4f} delta={block['delta_mean']:+.4f} "
            f"CI95=[{block['delta_ci95'][0]:+.4f},{block['delta_ci95'][1]:+.4f}] "
            f"| e2e ref {speed['reference_mean_e2e_s']}s "
            f"cand {speed['candidate_mean_e2e_s']}s speedup {speed['speedup_e2e']}x"
        )
    print(f"summary -> {args.out_dir / f'{SUITE}-summary.json'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--revision",
        default=None,
        help="pin the dataset revision (default: resolve the current one and record it)",
    )
    sc.add_server_args(parser, SUITE)
    sub = parser.add_subparsers(dest="cmd", required=True)

    prepare = sub.add_parser("prepare", help="select items and books (no server needed)")
    sc.add_model_args(prepare)
    prepare.add_argument("--max-prompt-tokens", type=int, default=DEFAULT_MAX_PROMPT_TOKENS)
    prepare.set_defaults(fn=cmd_prepare)

    arm = sub.add_parser("arm", help="run one backend's arm against a live server")
    sc.add_model_args(arm)
    sc.add_tier_arg(arm)
    sc.add_arm_args(arm)
    arm.add_argument(
        "--logprobs",
        action="store_true",
        help="also record top-2 logprob margins (adds per-request overhead; off by default "
        "so end-to-end timings stay comparable with the reference run)",
    )
    arm.set_defaults(fn=cmd_arm)

    compare = sub.add_parser("compare", help="paired per-task summary of two finished arms")
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
