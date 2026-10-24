"""Provenance: what produced a run, recorded so it can be filtered on later (evals, fine-tuning data).

A hash *identifies* content, it doesn't store it. The full agent definition is
already in every trace's run_start.config; agent_version is the key to find and
group runs by it. The harness code is recoverable from git by commit, but only
when the working tree was clean (dirty=False): uncommitted changes exist nowhere.

Which harness code ran depends on how it was installed, and that's checked, never
assumed: an installed copy sits inside *another* project's .venv, and running git there
would record that project's commit instead.
"""

import functools
import hashlib
import importlib.metadata
import json
import subprocess
import tomllib
from pathlib import Path


def agent_version(config: dict) -> str:
    """A 12-hex-char fingerprint of the agent definition: same content, same version.

    The file's name isn't part of it (renaming an agent doesn't change what it does).
    Keys are sorted first, so the order settings appear in the file doesn't matter.
    """
    content = {key: value for key, value in config.items() if key != "name"}
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


@functools.cache
def orchestra_version() -> dict:
    """Which harness code is running: its version, how it was installed, and its commit if known.

    Cached: measured once per process, at the first run. That's the closest match to
    the code actually running, which was imported at startup; files edited on disk
    afterwards don't change this process's code.
    """
    return describe_install(_package_version(), find_source_checkout(Path(__file__)), _direct_url())


def describe_install(version: str | None, checkout: Path | None, direct_url: dict | None) -> dict:
    """What to record, from the three facts we can know (pure: tested for each case).

        checkout   running from the harness's own source (developing it, or an editable
                   install): git there says exactly which commit, and whether it's dirty
        git        installed from a git URL: the installer recorded the commit (PEP 610's
                   direct_url.json); installed from a commit, so never dirty
        path       installed from a folder or wheel: a copy taken at install time, whose
                   commit can't be known; the source is recorded instead
    """
    if checkout is not None:
        return {"version": version, "installed": "checkout", **git_state(checkout)}
    if direct_url and direct_url.get("vcs_info", {}).get("commit_id"):
        return {"version": version, "installed": "git", "commit": direct_url["vcs_info"]["commit_id"],
                "dirty": False, "source": direct_url.get("url")}
    return {"version": version, "installed": "path", "commit": None, "dirty": None,
            "source": direct_url.get("url") if direct_url else None}


def find_source_checkout(code_file: Path) -> Path | None:
    """The harness's own source folder, if `code_file` is running from one.

    The nearest pyproject.toml above the code decides: in the harness repo it says
    name = "orchestra"; installed in another project's .venv, the nearest one is that
    project's, so this returns None and git is never asked about the wrong repo.
    """
    for folder in code_file.resolve().parents:
        pyproject = folder / "pyproject.toml"
        if pyproject.is_file():
            try:
                name = tomllib.loads(pyproject.read_text()).get("project", {}).get("name")
            except (OSError, tomllib.TOMLDecodeError):
                return None
            return folder if name == "orchestra" else None
    return None


def _package_version() -> str | None:
    try:
        return importlib.metadata.version("orchestra")
    except importlib.metadata.PackageNotFoundError:
        return None


def _direct_url() -> dict | None:
    """How the harness was installed, as the installer recorded it (direct_url.json), if at all."""
    try:
        text = importlib.metadata.distribution("orchestra").read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        return None
    return json.loads(text) if text else None


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
        "orchestra": orchestra_version(),
        # The model asked for is config["model"], right next to this in run_start; the model
        # the server says actually answered is in each model_response event.
    }
