"""Provenance for training runs: git state plus a fingerprint of the training code and data.

Used by ``ttucs train`` to tag MLflow runs (``git_sha``, ``git_branch``, ``git_dirty``,
``code_fingerprint``) and by ``scripts/autotrain.py`` to decide whether anything that shapes
the model changed since the current champion was trained.

The fingerprint covers the training and data-generation code, the UCS CSVs and the vocabulary
banks - not the configs (the caller hashes the effective config itself). Outside a git checkout
(an installed wheel) the git fields are ``unknown`` and the fingerprint hashes whatever of the
listed files exists; this is a development-repo tool.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]

#: Files that shape a trained model, relative to the repo root (configs excluded on purpose).
_CODE_FILES = (
    "src/ttucs/config.py",
    "src/ttucs/datasets/build.py",
    "src/ttucs/evaluate.py",
    "src/ttucs/models/__init__.py",
    "src/ttucs/models/base.py",
    "src/ttucs/models/bow.py",
    "src/ttucs/slate/gear.py",
    "src/ttucs/slate/normalize.py",
    "src/ttucs/slate/parser.py",
    "src/ttucs/synth/generator.py",
    "src/ttucs/taxonomy.py",
    "src/ttucs/text.py",
    "src/ttucs/train.py",
    "src/ttucs/resources/synth_banks.yaml",
    "src/ttucs/resources/catid_phrases.yaml",
)
_RESOURCE_GLOBS = ("src/ttucs/resources/ucs_v8*.csv",)


def code_fingerprint(root: Path = ROOT) -> str:
    """First 12 hex chars of the SHA-256 over the training code and resource files."""
    files = [root / rel for rel in _CODE_FILES]
    res = root / "src" / "ttucs" / "resources"
    for pattern in _RESOURCE_GLOBS:
        files += sorted(res.glob(pattern.rsplit("/", 1)[-1]))
    h = hashlib.sha256()
    for f in sorted(p for p in files if p.is_file()):
        h.update(f.as_posix().encode())
        h.update(b"\0")
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def git_info(root: Path = ROOT) -> dict[str, str]:
    """Short SHA, branch name and dirty flag; ``unknown`` where git or the repo is unavailable."""

    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(("git", *args),  # noqa: S603,S607
                                 cwd=root, capture_output=True, text=True, timeout=10, check=False)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    sha = git("rev-parse", "--short", "HEAD")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    status = git("status", "--porcelain")
    return {"sha": sha or "unknown", "branch": branch or "unknown",
            "dirty": "unknown" if status is None else ("true" if status else "false")}


def run_tags(root: Path = ROOT) -> dict[str, str]:
    """MLflow tags for a training run: git identity plus the code fingerprint."""
    info = git_info(root)
    return {"git_sha": info["sha"], "git_branch": info["branch"], "git_dirty": info["dirty"],
            "code_fingerprint": code_fingerprint(root)}


def changed_vs(champion: dict, root: Path = ROOT) -> bool:
    """True when the champion record does not match the current code fingerprint."""
    return champion.get("code_fingerprint") != code_fingerprint(root)
