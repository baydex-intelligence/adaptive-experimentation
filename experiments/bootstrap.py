"""Shared bootstrap helpers for experiment notebooks.

Intent: keep notebook setup cells short while reliably locating the project root
and making src importable for local package imports.
"""

from __future__ import annotations

import sys
from pathlib import Path


def find_project_root(start: Path) -> Path:
    """Return the nearest ancestor that looks like the repository root."""
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
            return candidate
    raise RuntimeError("Could not find the project root from the current working directory.")


def ensure_src_on_path(start: Path | None = None) -> Path:
    """Ensure the repository src directory is available in sys.path and return the project root."""
    root = find_project_root(start or Path.cwd())
    src_dir = root / "src"
    src_dir_str = str(src_dir)
    if src_dir_str not in sys.path:
        sys.path.insert(0, src_dir_str)
    return root
