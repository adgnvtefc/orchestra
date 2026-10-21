"""Sandbox: confine the agent's shell commands to their workspace.

On macOS this uses sandbox-exec ("Seatbelt"), the kernel's per-process sandbox: the same
mechanism Claude Code and the Codex CLI use there. It's built in, and since commands still
run on macOS, an environment's .venv works unchanged. (A container backend would be a
second class here with the same prepare() method.)

What a Seatbelt-sandboxed command can do:
    write   only inside its workspace (and /dev, for /dev/null etc.); temp files go to
            <workspace>/.tmp, via TMPDIR
    read    everything outside your home folder (system files, tools, Homebrew), but inside
            home only the workspace and toolchains it needs (TOOLCHAINS). So: no ~/.ssh,
            no shell history, no other sessions, no harness .env, and no listing folders.
            *Whether* a file exists (its metadata: size, dates) is still visible: tools like
            pytest and git walk up the folder tree asking "is there a config file here?", and
            "permission denied" crashes them where "no" doesn't. The trade-off: a command that
            guesses an exact path can learn the file exists, never what's in it.
    network none outbound, unless network=True

Allow-lists, not deny-lists, for the same reason as the environment variables: a new
secret is protected without anyone remembering to add it.

This confines *bash* only. read_file/write_file have their own path check (fs._resolve)
and run inside the harness process, which isn't sandboxed (it needs your API key).
"""

import sys
from dataclasses import dataclass
from pathlib import Path

SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")

# Inside your home folder, the only places (besides the workspace) commands may read:
# where toolchains live that a workspace's tools link into. uv-installed Pythons are what
# an environment's .venv points at.
TOOLCHAINS = (".local/share/uv",)


def default_sandbox() -> "Seatbelt | Unsandboxed":
    """Seatbelt where it's available (macOS), otherwise none."""
    return Seatbelt() if available() else Unsandboxed()


@dataclass(frozen=True)
class Unsandboxed:
    """Commands run as a plain shell, with your full permissions. Only what the approver allows runs."""

    name: str = "none"

    def prepare(self, command: str, workspace: Path) -> tuple[list[str], dict[str, str]]:
        """(argv to run, extra environment variables)"""
        return ["/bin/sh", "-c", command], {}


@dataclass(frozen=True)
class Seatbelt:
    """Commands run under macOS sandbox-exec, confined as described in the module docstring."""

    network: bool = False
    home: Path = Path.home()  # the folder whose contents are unreadable (a parameter so tests can use a stand-in)
    name: str = "seatbelt"

    def __post_init__(self):
        if not available():
            raise RuntimeError("the Seatbelt sandbox needs macOS and /usr/bin/sandbox-exec")

    def prepare(self, command: str, workspace: Path) -> tuple[list[str], dict[str, str]]:
        """(argv to run, extra environment variables): the command wrapped in sandbox-exec, and
        TMPDIR pointed into the workspace, the one place it can write."""
        workspace = workspace.resolve()  # rules match real paths (macOS: /var is really /private/var)
        tmp = workspace / ".tmp"
        tmp.mkdir(exist_ok=True)
        profile = seatbelt_profile(workspace, home=self.home.resolve(), network=self.network)
        return [str(SANDBOX_EXEC), "-p", profile, "/bin/sh", "-c", command], {"TMPDIR": str(tmp)}


def available() -> bool:
    return sys.platform == "darwin" and SANDBOX_EXEC.is_file()


def seatbelt_profile(workspace: Path, *, home: Path, network: bool) -> str:
    """The sandbox-exec rules, in its own Scheme-like language. When rules overlap, the last
    matching one wins: so each "deny broadly" is followed by its "allow these exceptions".

    Reads are split in two: file-read-data (contents, and listing a folder) is what's denied
    in home; file-read-metadata (does it exist, size, dates) stays allowed. See the module
    docstring for why."""
    readable_in_home = [workspace, *(home / t for t in TOOLCHAINS)]
    rules = [
        "(version 1)",
        "(allow default)",
        # writes: nowhere, except the workspace and /dev (/dev/null, /dev/tty, ...)
        "(deny file-write*)",
        f'(allow file-write* (subpath "{_q(workspace)}") (subpath "/dev"))',
        # reads: no contents (or folder listings) in home, except the workspace and toolchains
        f'(deny file-read-data (subpath "{_q(home)}"))',
        "(allow file-read-data " + " ".join(f'(subpath "{_q(p)}")' for p in readable_in_home) + ")",
    ]
    if not network:
        rules.append("(deny network-outbound (remote ip))")  # local sockets still work; the internet doesn't
    return "\n".join(rules) + "\n"


def _q(path: Path) -> str:
    """A path as the inside of a sandbox-profile string literal: backslashes and quotes escaped."""
    return str(path).replace("\\", "\\\\").replace('"', '\\"')
