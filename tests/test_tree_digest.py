# SPDX-License-Identifier: Apache-2.0
"""The source-tree digest must be reproducible, and must notice everything it claims to cover."""
from __future__ import annotations

import pathlib

from tree_digest import tree_digest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _make_tree(base: pathlib.Path) -> None:
    (base / "src").mkdir(parents=True)
    (base / "src" / "a.cu").write_text("kernel a\n")
    (base / "src" / "b.py").write_text("transform b\n")
    (base / "src" / "nested").mkdir()
    (base / "src" / "nested" / "c.h").write_text("header c\n")


def test_digest_is_stable_across_calls(tmp_path):
    _make_tree(tmp_path)
    first, _ = tree_digest(tmp_path, ["src"])
    second, _ = tree_digest(tmp_path, ["src"])
    assert first == second


def test_digest_is_independent_of_filesystem_order(tmp_path):
    _make_tree(tmp_path)
    expected, records = tree_digest(tmp_path, ["src"])
    assert [r["path"] for r in records] == sorted(r["path"] for r in records)

    other = tmp_path / "replica"
    other.mkdir()
    (other / "src").mkdir()
    (other / "src" / "nested").mkdir()
    # Deliberately create in a different order.
    (other / "src" / "nested" / "c.h").write_text("header c\n")
    (other / "src" / "b.py").write_text("transform b\n")
    (other / "src" / "a.cu").write_text("kernel a\n")
    assert tree_digest(other, ["src"])[0] == expected


def test_digest_changes_on_content_change(tmp_path):
    _make_tree(tmp_path)
    before, _ = tree_digest(tmp_path, ["src"])
    (tmp_path / "src" / "a.cu").write_text("kernel a modified\n")
    assert tree_digest(tmp_path, ["src"])[0] != before


def test_digest_changes_on_mode_change(tmp_path):
    """A file becoming executable changes the build; the digest must see it."""
    _make_tree(tmp_path)
    before, _ = tree_digest(tmp_path, ["src"])
    (tmp_path / "src" / "b.py").chmod(0o755)
    assert tree_digest(tmp_path, ["src"])[0] != before


def test_digest_covers_only_named_paths(tmp_path):
    _make_tree(tmp_path)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "notes.md").write_text("not source\n")
    before, _ = tree_digest(tmp_path, ["src"])
    (tmp_path / "docs" / "notes.md").write_text("still not source\n")
    assert tree_digest(tmp_path, ["src"])[0] == before


def test_digest_ignores_build_noise(tmp_path):
    _make_tree(tmp_path)
    before, _ = tree_digest(tmp_path, ["src"])
    (tmp_path / "src" / "__pycache__").mkdir()
    (tmp_path / "src" / "__pycache__" / "b.cpython-311.pyc").write_bytes(b"\x00\x01")
    (tmp_path / "src" / "libkernel.so").write_bytes(b"\x7fELF")
    assert tree_digest(tmp_path, ["src"])[0] == before


def test_missing_path_is_an_error(tmp_path):
    _make_tree(tmp_path)
    try:
        tree_digest(tmp_path, ["src", "nonexistent"])
    except FileNotFoundError:
        return
    raise AssertionError("a missing digest path must fail loudly, not silently digest less")
