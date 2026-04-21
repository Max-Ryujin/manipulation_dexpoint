"""Utilities for running DexPoint scripts from a dedicated scripts directory."""

from pathlib import Path
import sys


SCRIPTS_DIR = Path(__file__).resolve().parent
DEXPOINT_ROOT = SCRIPTS_DIR.parent
REPO_ROOT = DEXPOINT_ROOT.parent


def ensure_script_imports() -> None:
    """Make repository-local imports available to moved standalone scripts."""
    for path in (DEXPOINT_ROOT, REPO_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


ensure_script_imports()
