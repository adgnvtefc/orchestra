"""Shell tool: runs commands in the workspace. Its permission is "ask": the Policy gets approval first."""

import os
import signal
import subprocess

from orchestra.tools.base import Tool, ToolContext, truncate


def bash(ctx: ToolContext, command: str, timeout: int = 30) -> str:
    """Run `command` in a shell inside the workspace and return combined stdout/stderr + exit code.

    Approval is not this function's job: BASH's permission is "ask", so the Policy
    checks with the human before this ever runs. Confinement isn't either: ctx.sandbox
    prepares the command (Seatbelt wraps it so it can only touch the workspace).
    The command sees only ctx.command_env(): allowlisted variables plus the environment's.
    Truncate huge output so one command can't flood the context window.
    """
    ctx.workspace.mkdir(parents=True, exist_ok=True)
    argv, sandbox_env = ctx.sandbox.prepare(command, ctx.workspace)
    proc = subprocess.Popen(
        argv,
        cwd=ctx.workspace,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        stdin=subprocess.DEVNULL,  # a command waiting for input would otherwise hang until timeout
        # allowlisted variables only (no API keys, not the harness's Python); the sandbox's own
        # (TMPDIR into the workspace) win over the environment's, since they're confinement rules
        env={**ctx.command_env(), **sandbox_env},
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
