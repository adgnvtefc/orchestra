"""Sandbox: confine the agent's shell commands to their workspace.

On macOS this uses sandbox-exec ("Seatbelt"), the kernel's per-process sandbox: the same
mechanism Claude Code and the Codex CLI use there. It's built in, and since commands still
run on macOS, an environment's .venv works unchanged. (A container backend would be a
second class here with the same prepare() method.)

What a Seatbelt-sandboxed command can do:
    write   only inside its workspace (and /dev, for /dev/null etc.); temp files go to
            <workspace>/.tmp, via TMPDIR
    read    everything outside your home folder (system files, tools, Homebrew), but inside
            home only the workspace and what the environment allows (rules.readable, e.g.
            uv's Pythons for python-dev). So: no ~/.ssh,
            no shell history, no other sessions, no harness .env, and no listing folders.
            *Whether* a file exists (its metadata: size, dates) is still visible: tools like
            pytest and git walk up the folder tree asking "is there a config file here?", and
            "permission denied" crashes them where "no" doesn't. The trade-off: a command that
            guesses an exact path can learn the file exists, never what's in it.
    network none outbound, unless the environment allows it (rules.network)

Allow-lists, not deny-lists, for the same reason as the environment variables: a new
secret is protected without anyone remembering to add it.

Who decides what: the interface chooses *whether* there's a sandbox (Seatbelt, or none);
the session's environment says *what its commands need* (SandboxRules, from the sandbox:
section of its environment file). build_toolset combines them with sandbox.with_rules().

This confines *bash* only. read_file/write_file have their own path check (fs._resolve)
and run inside the harness process, which isn't sandboxed (it needs your API key).
"""

import sys
from dataclasses import dataclass, replace
from pathlib import Path

SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")


@dataclass(frozen=True)
class SandboxRules:
    """What an environment's commands need beyond the workspace. The default needs nothing.

    The written form ({"readable": [paths], "network": bool}) is defined here and only here:
    environment files and session.json both go through from_dict() / to_dict(), so adding a
    field means changing this class, not every module that stores rules.
    """

    readable: tuple[Path, ...] = ()  # extra folders whose contents may be read (e.g. a toolchain in home)
    network: bool = False  # outbound internet

    @classmethod
    def from_dict(cls, data: object, relative_to: Path | None = None) -> "SandboxRules":
        """Validate and build. Paths may use ~; relative ones are taken from `relative_to`.
        Raises ValueError with a readable message on anything malformed."""
        if not isinstance(data, dict):
            raise ValueError("must be 'key: value' lines")
        unknown = set(data) - {"readable", "network"}
        if unknown:
            raise ValueError(f"unknown key(s) {sorted(unknown)}. Allowed: ['network', 'readable']")
        readable = data.get("readable") or []
        if not (isinstance(readable, list) and all(isinstance(p, str) for p in readable)):
            raise ValueError("readable must be a list of paths")
        network = data.get("network", False)
        if not isinstance(network, bool):
            raise ValueError("network must be true or false")
        base = relative_to or Path("/")
        return cls(readable=tuple((base / Path(p).expanduser()).resolve() for p in readable), network=network)

    def to_dict(self) -> dict:
        """The written form, for session.json; from_dict() reads it back unchanged."""
        return {"readable": [str(p) for p in self.readable], "network": self.network}


def default_sandbox() -> "Seatbelt | Unsandboxed":
    """Seatbelt where it's available (macOS), otherwise none."""
    return Seatbelt() if available() else Unsandboxed()


@dataclass(frozen=True)
class Unsandboxed:
    """Commands run as a plain shell, with your full permissions. Only what the approver allows runs."""

    name: str = "none"

    def with_rules(self, rules: SandboxRules) -> "Unsandboxed":
        """Nothing is confined, so there's nothing to loosen."""
        return self

    def prepare(self, command: str, workspace: Path) -> tuple[list[str], dict[str, str]]:
        """(argv to run, extra environment variables)"""
        return ["/bin/sh", "-c", command], {}


@dataclass(frozen=True)
class Seatbelt:
    """Commands run under macOS sandbox-exec, confined as described in the module docstring."""

    network: bool = False
    readable: tuple[Path, ...] = ()  # folders in home whose contents may be read, besides the workspace
    home: Path = Path.home()  # the folder whose contents are unreadable (a parameter so tests can use a stand-in)
    name: str = "seatbelt"

    def __post_init__(self):
        if not available():
            raise RuntimeError("the Seatbelt sandbox needs macOS and /usr/bin/sandbox-exec")

    def with_rules(self, rules: SandboxRules) -> "Seatbelt":
        """This sandbox, allowing what an environment's commands need."""
        return replace(self, network=rules.network, readable=rules.readable)

    def prepare(self, command: str, workspace: Path) -> tuple[list[str], dict[str, str]]:
        """(argv to run, extra environment variables): the command wrapped in sandbox-exec, and
        TMPDIR pointed into the workspace, the one place it can write."""
        workspace = workspace.resolve()  # rules match real paths (macOS: /var is really /private/var)
        tmp = workspace / ".tmp"
        tmp.mkdir(exist_ok=True)
        readable = tuple(p.resolve() for p in self.readable)
        profile = seatbelt_profile(workspace, home=self.home.resolve(), network=self.network, readable=readable)
        return [str(SANDBOX_EXEC), "-p", profile, "/bin/sh", "-c", command], {"TMPDIR": str(tmp)}


def available() -> bool:
    return sys.platform == "darwin" and SANDBOX_EXEC.is_file()


def seatbelt_profile(workspace: Path, *, home: Path, network: bool, readable: tuple[Path, ...] = ()) -> str:
    """The sandbox-exec rules, in its own Scheme-like language. When rules overlap, the last
    matching one wins: so each "deny broadly" is followed by its "allow these exceptions".

    Reads are split in two: file-read-data (contents, and listing a folder) is what's denied
    in home; file-read-metadata (does it exist, size, dates) stays allowed. See the module
    docstring for why."""
    readable_in_home = [workspace, *readable]
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
