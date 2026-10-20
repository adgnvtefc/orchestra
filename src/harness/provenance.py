"""Provenance: what produced a run, recorded so it can be filtered on later (evals, fine-tuning data).

A hash *identifies* content, it doesn't store it. The full agent definition is
already in every trace's run_start.config; agent_version is the key to find and
group runs by it. The harness code is recoverable from git by commit, but only
when the working tree was clean (dirty=False): uncommitted changes exist nowhere.
"""

import functools
import hashlib
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def agent_version(config: dict) -> str:
    """A 12-hex-char fingerprint of the agent definition: same content, same version.

    The file's name isn't part of it (renaming an agent doesn't change what it does).
    Keys are sorted first, so the order settings appear in the file doesn't matter.
    """
    content = {key: value for key, value in config.items() if key != "name"}
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


@functools.cache
def harness_version() -> dict:
    """The git commit of the harness code, and whether there were uncommitted changes.

    Cached: measured once per process, at the first run. That's the closest match to
    the code actually running, which was imported at startup; files edited on disk
    afterwards don't change this process's code.
    """
    return git_state(REPO_ROOT)


def git_state(repo: Path) -> dict:
    """{"commit": sha or None, "dirty": bool or None}. None means unknown (no git, not a repo)."""
    try:
        commit = _git(repo, "rev-parse", "HEAD")
        # --porcelain lists modified, staged, and untracked files; empty output means clean.
        # Ignored files (traces/, workspace/, .env) don't count.
        dirty = _git(repo, "status", "--porcelain") != ""
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, timeout=5, check=True)
    return result.stdout.strip()


def collect(config: dict) -> dict:
    """Everything recorded about where a run came from. Goes in run_start.provenance."""
    return {
        "agent_version": agent_version(config),
        "harness": harness_version(),
        # The model asked for is config["model"], right next to this in run_start; the model
        # the server says actually answered is in each model_response event.
    }
