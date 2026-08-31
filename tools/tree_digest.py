#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reproducible digest of a named set of source paths.

The promotion manifest binds a ``source_tree_sha256`` over explicitly named
paths.  "Digest of these paths" is not by itself reproducible -- it depends on
traversal order, on whether the mode bit is included, and on how the per-file
hashes are combined.  This module is the ONLY sanctioned producer of that field,
so the algorithm is pinned here:

    1. Collect every regular file under each named path, as a POSIX-style
       path relative to the repository root.
    2. Exclude the standard non-source noise (see ``EXCLUDE_DIRS``/``EXCLUDE_SUFFIXES``).
    3. Sort by that relative path, bytewise, in the C locale.
    4. For each file emit exactly ``"{mode} {sha256_hex} {relpath}\\n"`` where
       mode is ``100755`` if the owner-execute bit is set and ``100644``
       otherwise, encoded UTF-8.
    5. SHA-256 the concatenation of those lines.

Deliberately not a git tree hash: this must work on an exported tarball with no
``.git``, and must cover a chosen subset of paths rather than a whole tree.

Usage:
    tree_digest.py --root . src tests benchmarks
    tree_digest.py --root . --manifest src tests      # per-file listing too
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

EXCLUDE_DIRS = {"__pycache__", ".git", ".pytest_cache", ".ruff_cache", "build", "dist", ".venv"}
# Directory-name suffixes excluded wherever they appear. `pip install .` drops
# `src/<pkg>.egg-info/` INSIDE a digest path, and its PKG-INFO/SOURCES.txt would
# silently change source_tree_sha256 between a clean tree and one that has ever
# been pip-installed (found during A2 wheel verification: the same sources
# digested differently after the round-trip install).
EXCLUDE_DIR_SUFFIXES = (".egg-info",)
EXCLUDE_SUFFIXES = (".pyc", ".pyo", ".so", ".o", ".ncu-rep", ".nsys-rep")

BLOCK = 1 << 20


def _file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(BLOCK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_files(root: pathlib.Path, targets: list[str]):
    for target in targets:
        base = root / target
        if not base.exists():
            raise FileNotFoundError(f"digest path does not exist: {target}")
        candidates = [base] if base.is_file() else sorted(base.rglob("*"))
        for path in candidates:
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(root)
            if EXCLUDE_DIRS.intersection(relative.parts):
                continue
            if any(part.endswith(EXCLUDE_DIR_SUFFIXES) for part in relative.parts[:-1]):
                continue
            if relative.name.endswith(EXCLUDE_SUFFIXES):
                continue
            yield relative, path


def tree_digest(root: pathlib.Path, targets: list[str]) -> tuple[str, list[dict]]:
    """Return (digest_hex, per_file_records) for ``targets`` under ``root``."""
    records = []
    for relative, path in _iter_files(root, targets):
        mode = "100755" if path.stat().st_mode & 0o100 else "100644"
        records.append(
            {"path": relative.as_posix(), "mode": mode, "sha256": _file_sha256(path)}
        )
    records.sort(key=lambda record: record["path"].encode("utf-8"))

    digest = hashlib.sha256()
    for record in records:
        line = f"{record['mode']} {record['sha256']} {record['path']}\n"
        digest.update(line.encode("utf-8"))
    return digest.hexdigest(), records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("paths", nargs="+", help="paths to digest, relative to --root")
    parser.add_argument("--root", default=".", help="repository root (default: .)")
    parser.add_argument("--manifest", action="store_true", help="also print per-file records")
    args = parser.parse_args(argv)

    root = pathlib.Path(args.root).resolve()
    digest, records = tree_digest(root, list(args.paths))

    if args.manifest:
        json.dump(
            {"root_paths": sorted(args.paths), "source_tree_sha256": digest, "files": records},
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
    else:
        print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
