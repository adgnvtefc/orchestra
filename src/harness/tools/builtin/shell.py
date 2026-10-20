"""Shell tool: runs commands in the workspace. Its permission is "ask": the Policy gets approval first."""

import os
import signal
import subprocess
import sys
from pathlib import Path

from harness.tools.base import Tool, ToolContext, truncate

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


def bash(ctx: ToolContext, command: str, timeout: int = 30) -> str:
    """Run `command` in a shell inside the workspace and return combined stdout/stderr + exit code.

    Approval is not this function's job: BASH's permission is "ask", so the Policy
    checks with the human before this ever runs. (Real sandboxing is Phase 2.)
    The command sees only agent_env(), not the harness's environment.
    Truncate huge output so one command can't flood the context window.
    """
    ctx.workspace.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        command,
        shell=True,
        cwd=ctx.workspace,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        stdin=subprocess.DEVNULL,  # a command waiting for input would otherwise hang until timeout
        env=agent_env(),  # allowlisted variables only: no API keys, not the harness's Python
        # The shell becomes leader of a new process group, and everything it starts joins
        # that group. On timeout we kill the whole group: killing just the shell would
        # leave its children (servers, test runners, `sleep`) running as orphans.
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)  # proc.pid is also the group id, since it leads the group
        proc.communicate()  # reap the killed shell and close its pipes
        return f"Command timed out after {timeout}s and was killed."

    return truncate(f"{stdout + stderr}\n(exit code {proc.returncode})".strip())


BASH = Tool(
    fn=bash,
    schema={
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Run a shell command in the workspace directory and return its output and exit code. "
                "The user must approve each command, and may decline. "
                "Use read_file/write_file for files instead of cat, echo, or heredocs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to run, e.g. 'python primes.py'.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Seconds before the command is killed. Defaults to 30; raise it for slow commands like test suites.",
                    },
                },
                "required": ["command"],  # timeout is optional: the Python default (30) applies when omitted
            },
        },
    },
    permission="ask",
)
