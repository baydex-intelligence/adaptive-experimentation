"""Tests for shared notebook bootstrap helpers."""

from __future__ import annotations

import sys
from pathlib import Path

from experiments.bootstrap import ensure_src_on_path, find_project_root


def test_find_project_root_walks_up_to_repo_root(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    nested = root / "experiments" / "nested"
    nested.mkdir(parents=True)
    (root / "pyproject.toml").write_text("[tool.poetry]\nname='x'\n", encoding="utf-8")
    (root / "src").mkdir()

    observed = find_project_root(nested)

    assert observed == root


def test_ensure_src_on_path_is_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    start = root / "experiments"
    start.mkdir(parents=True)
    (root / "pyproject.toml").write_text("[tool.poetry]\nname='x'\n", encoding="utf-8")
    src = root / "src"
    src.mkdir()

    before = list(sys.path)
    try:
        first = ensure_src_on_path(start)
        second = ensure_src_on_path(start)
        assert first == root
        assert second == root
        assert sys.path.count(str(src)) == 1
        assert str(src) in sys.path
    finally:
        sys.path[:] = before
