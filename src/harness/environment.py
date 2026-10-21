"""Environments: what a workspace is set up with before the agent starts.

    environments/python-dev.yaml
        description: Python 3.12 with pytest
        files: templates/python-dev     # optional folder, copied into the workspace
        setup:                          # shell commands, run in the workspace, in order
          - uv venv --python 3.12 .venv
        env:                            # extra variables for the agent's commands
          PATH: "{workspace}/.venv/bin"  # PATH is prepended to; {workspace} is filled in
        prompt: |                       # added to the agent's system prompt automatically
          Your workspace has ...

The harness sets the environment up, not the agent: setup is deterministic and costs
the agent no steps. Setup commands run without approval, because they come from a
config file you wrote (trusted), not from the model (untrusted).

An environment belongs to the *workspace*: provisioning leaves a marker in
<workspace>/.harness/environment.json, so a workspace shared by several sessions is
set up once, and a workspace set up differently is never silently re-provisioned.
"""

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

from harness.tools.base import ToolContext

ALLOWED_KEYS = {"description", "files", "setup", "env", "prompt"}
MARKER = Path(".harness") / "environment.json"
SETUP_TIMEOUT = 600  # seconds per setup command; installs can be slow the first time


class EnvironmentSetupError(Exception):
    """An environment file is invalid, or setting a workspace up failed."""


def fill_workspace(variables: dict[str, str], workspace: Path) -> dict[str, str]:
    """Replace "{workspace}" in each value with the workspace's path. The one place this happens:
    setup commands and the agent's commands must always see the same values."""
    return {name: value.replace("{workspace}", str(workspace)) for name, value in variables.items()}


@dataclass(frozen=True)
class Environment:
    name: str
    version: str  # hash of the file *and* the template files: any change is a new version
    description: str = ""
    files: Path | None = None
    setup: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    prompt: str = ""

    def variables(self, workspace: Path) -> dict[str, str]:
        """The extra environment variables for agent commands, with {workspace} filled in."""
        return fill_workspace(self.env, workspace)


def load_environment(environments_dir: Path, name: str, project_root: Path) -> Environment:
    """Read and validate environments/<name>.yaml. Problems fail here, before anything is created."""
    path = environments_dir / f"{name}.yaml"
    if not path.is_file():
        available = sorted(p.stem for p in environments_dir.glob("*.yaml")) if environments_dir.is_dir() else []
        raise EnvironmentSetupError(f"no environment {name!r}. Available: {', '.join(available) or '(none)'}")

    text = path.read_text()
    spec = yaml.safe_load(text) or {}
    if not isinstance(spec, dict):
        raise EnvironmentSetupError(f"{path}: must be 'key: value' lines")
    unknown = set(spec) - ALLOWED_KEYS
    if unknown:  # a typo like "setpu:" would otherwise be silently ignored
        raise EnvironmentSetupError(f"{path}: unknown key(s) {sorted(unknown)}. Allowed: {sorted(ALLOWED_KEYS)}")

    setup = spec.get("setup") or []
    if not (isinstance(setup, list) and all(isinstance(c, str) for c in setup)):
        raise EnvironmentSetupError(f"{path}: setup must be a list of commands")
    env = spec.get("env") or {}
    if not (isinstance(env, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in env.items())):
        raise EnvironmentSetupError(f"{path}: env must map names to strings")

    files = None
    if spec.get("files"):
        files = (project_root / spec["files"]).resolve()
        if not files.is_dir():
            raise EnvironmentSetupError(f"{path}: files folder {spec['files']!r} doesn't exist")

    return Environment(
        name=name,
        version=_version(text, files),
        description=spec.get("description", ""),
        files=files,
        setup=tuple(setup),
        env=env,
        prompt=(spec.get("prompt") or "").strip(),
    )


def _version(spec_text: str, files: Path | None) -> str:
    """12 hex chars over the spec and every template file (path + bytes), in a fixed order."""
    digest = hashlib.sha256(spec_text.encode())
    if files is not None:
        for file in sorted(p for p in files.rglob("*") if p.is_file()):
            digest.update(str(file.relative_to(files)).encode() + b"\0" + file.read_bytes())
    return digest.hexdigest()[:12]


def provision(environment: Environment, workspace: Path) -> bool:
    """Set `workspace` up for `environment`, unless it already is. Returns True if setup ran.

    Setup commands run with exactly the environment the agent's commands will get
    (allowlisted variables plus the environment's own), so setup never sees the
    harness's secrets either.
    """
    marker = workspace / MARKER
    if marker.is_file():
        existing = json.loads(marker.read_text())
        if existing.get("version") == environment.version:
            return False  # already set up, e.g. by an earlier session sharing this workspace
        raise EnvironmentSetupError(
            f"workspace {workspace} is already set up as {existing.get('name')!r} "
            f"(version {existing.get('version')}), not {environment.name!r} ({environment.version}). "
            "Refusing to set it up again over the top."
        )

    if environment.files is not None:
        # dirs_exist_ok + copy only what's missing: never overwrite a file already in the workspace
        shutil.copytree(environment.files, workspace, dirs_exist_ok=True, copy_function=_copy_if_missing)

    env = ToolContext(workspace, environment.variables(workspace)).command_env()
    for command in environment.setup:
        try:
            result = subprocess.run(
                command, shell=True, cwd=workspace, env=env, capture_output=True, text=True,
                timeout=SETUP_TIMEOUT, stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            raise EnvironmentSetupError(f"setup command timed out after {SETUP_TIMEOUT}s: {command}") from None
        if result.returncode != 0:
            raise EnvironmentSetupError(
                f"setup command failed (exit {result.returncode}): {command}\n{(result.stdout + result.stderr).strip()}"
            )

    # Written last: a workspace only counts as set up once every command succeeded.
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({
        "name": environment.name,
        "version": environment.version,
        "provisioned": datetime.now().isoformat(timespec="seconds"),
    }, indent=2))
    return True


def _copy_if_missing(src: str, dst: str) -> None:
    if not Path(dst).exists():
        shutil.copy2(src, dst)
