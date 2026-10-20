"""What a tool is, plus helpers shared by every tool."""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from harness.policy import Permission

# ~10k chars is roughly 2.5k tokens: enough to be useful, small enough that one
# tool result can't eat a 32k context window.
MAX_OUTPUT_CHARS = 10_000


# The only environment variables the agent's commands get. Everything else in the
# harness's environment (API keys loaded from .env, tokens, credentials) stays out:
# a command that echoes a secret would put it into the trace and the model's context.
ALLOWED_ENV = ("PATH", "HOME", "USER", "SHELL", "LANG", "LC_ALL", "LC_CTYPE", "TERM", "TMPDIR")


def agent_env() -> dict[str, str]:
    """The environment for agent commands: the allowlisted variables, minus the harness's own Python.

    The harness runs inside its virtual environment (.venv), which puts .venv/bin first on
    PATH. Left there, the agent's `python`/`pytest`/`pip` would be the harness's own, and it
    could change the packages the harness itself runs on.
    """
    env = {name: os.environ[name] for name in ALLOWED_ENV if name in os.environ}
    harness_bin = Path(sys.prefix) / "bin"
    if sys.prefix != sys.base_prefix and "PATH" in env:  # only when the harness is in a venv
        env["PATH"] = os.pathsep.join(p for p in env["PATH"].split(os.pathsep) if Path(p) != harness_bin)
    return env


@dataclass(frozen=True)
class ToolContext:
    """What a tool call runs *in*, as opposed to what it was asked to do (the arguments).

    Toolset passes it as the first argument of every tool function. The model never
    sees it and can't set it: it isn't in any schema.
    """

    workspace: Path  # all file access and shell commands are confined to this directory
    env: dict[str, str] = field(default_factory=dict)  # extra variables from the session's environment

    def command_env(self) -> dict[str, str]:
        """The environment shell commands run with: agent_env() plus this context's variables.
        PATH is prepended to (so the environment's tools win); every other variable is set."""
        env = agent_env()
        for name, value in self.env.items():
            if name == "PATH" and env.get("PATH"):
                env["PATH"] = f"{value}{os.pathsep}{env['PATH']}"
            else:
                env[name] = value
        return env


# makes so cannot modify after creation
@dataclass(frozen=True)
class Tool:
    """A tool = the Python function that runs it + the JSON schema the model sees."""

    # Callable means function takes arguments, return str
    fn: Callable[..., str]
    schema: dict
    # Default permission. Required, with no default value: a new tool can't silently become "allow".
    permission: Permission

    # makes so that tool.name is a field, but is computed from schema
    @property
    def name(self) -> str:
        return self.schema["function"]["name"]


def truncate(text: str) -> str:
    """Keep the head and tail of long output; errors and summaries usually live at the end."""
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    half = MAX_OUTPUT_CHARS // 2
    omitted = len(text) - MAX_OUTPUT_CHARS
    return f"{text[:half]}\n\n... [{omitted} characters omitted] ...\n\n{text[-half:]}"
