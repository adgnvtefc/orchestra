"""The bash tool: execution in the workspace, output handling, timeouts. (Approval is tested in test_policy.py.)"""

import subprocess
import sys
import time
import uuid
from pathlib import Path

from harness.tools.base import MAX_OUTPUT_CHARS
from harness.tools.builtin.shell import ALLOWED_ENV, agent_env, bash


class TestExecution:
    def test_runs_in_workspace(self, ctx, workspace):
        assert bash(ctx, "pwd").splitlines()[0] == str(workspace)

    def test_can_see_workspace_files(self, ctx, workspace):
        (workspace / "present.txt").write_text("")
        assert "present.txt" in bash(ctx, "ls")

    def test_reports_exit_code_zero(self, ctx):
        assert bash(ctx, "true").endswith("(exit code 0)")

    def test_reports_nonzero_exit_code(self, ctx):
        assert bash(ctx, "exit 3").endswith("(exit code 3)")

    def test_captures_stdout_and_stderr(self, ctx):
        out = bash(ctx, "echo to-out; echo to-err >&2")
        assert "to-out" in out and "to-err" in out

    def test_shell_features_work(self, ctx):
        assert bash(ctx, "echo a | tr a b && echo $((2 + 3))").startswith("b\n5")

    def test_stdin_is_closed(self, ctx):
        # A command that reads stdin must get EOF at once, not hang until the timeout.
        start = time.monotonic()
        out = bash(ctx, "cat", timeout=10)
        assert time.monotonic() - start < 5
        assert out.endswith("(exit code 0)")

    def test_long_output_is_truncated(self, ctx):
        out = bash(ctx, f"python3 -c \"print('x' * {MAX_OUTPUT_CHARS * 2})\"")
        assert "characters omitted" in out
        assert out.endswith("(exit code 0)")  # the tail, with the exit code, survives truncation

    def test_recreates_missing_workspace(self, ctx, workspace):
        workspace.rmdir()
        assert bash(ctx, "true").endswith("(exit code 0)")
        assert workspace.is_dir()


class TestTimeout:
    def test_slow_command_is_killed(self, ctx):
        start = time.monotonic()
        assert bash(ctx, "sleep 10", timeout=1) == "Command timed out after 1s and was killed."
        assert time.monotonic() - start < 5

    def test_timeout_kills_child_processes(self, ctx):
        marker = f"sleep 3{uuid.uuid4().int % 1000:03d}"  # unique, so we only ever find our own process
        try:
            bash(ctx, f"{marker}; echo done", timeout=1)
            time.sleep(0.3)
            survivors = subprocess.run(["pgrep", "-f", marker], capture_output=True, text=True).stdout
            assert survivors == ""
        finally:
            subprocess.run(["pkill", "-f", marker])


class TestNoLongerAsks:
    def test_runs_without_prompting(self, ctx, answer_prompt):
        # Approval moved to the Policy. Called directly, bash(ctx, ) just runs.
        prompts = answer_prompt("n")
        assert bash(ctx, "echo hi").startswith("hi")
        assert prompts == []


class TestEnvironment:
    """Agent commands see an allowlisted environment: no secrets, not the harness's own Python."""

    def test_secrets_are_not_visible(self, ctx, monkeypatch):
        monkeypatch.setenv("OMLX_API_KEY", "sk-should-never-leak")
        monkeypatch.setenv("SOME_TOKEN", "also-secret")
        out = bash(ctx, 'echo "[$OMLX_API_KEY][$SOME_TOKEN]"')
        assert out.startswith("[][]")
        assert "sk-should-never-leak" not in bash(ctx, "env")

    def test_only_allowlisted_variables(self, ctx, monkeypatch):
        monkeypatch.setenv("RANDOM_HARNESS_SETTING", "x")
        names = {line.split("=", 1)[0] for line in bash(ctx, "env").splitlines() if "=" in line}
        # the shell itself adds a few (PWD, SHLVL, _), nothing else from the harness
        assert names - set(ALLOWED_ENV) <= {"PWD", "OLDPWD", "SHLVL", "_"}

    def test_allowlisted_variables_pass_through(self, ctx, monkeypatch):
        monkeypatch.setenv("HOME", "/tmp/fake-home")
        assert bash(ctx, "echo $HOME").startswith("/tmp/fake-home")

    def test_harness_venv_is_not_on_path(self, ctx):
        harness_bin = str(Path(sys.prefix) / "bin")
        assert harness_bin not in bash(ctx, "echo $PATH").split("\n")[0].split(":")
        assert "VIRTUAL_ENV" not in bash(ctx, "env")

    def test_path_kept_when_harness_is_not_in_a_venv(self, monkeypatch):
        monkeypatch.setattr(sys, "prefix", sys.base_prefix)
        monkeypatch.setenv("PATH", "/a/bin:/b/bin")
        assert agent_env()["PATH"] == "/a/bin:/b/bin"

    def test_venv_entry_removed_from_path(self, monkeypatch):
        venv_bin = str(Path(sys.prefix) / "bin")
        monkeypatch.setenv("PATH", f"{venv_bin}:/usr/bin:/bin")
        if sys.prefix != sys.base_prefix:
            assert agent_env()["PATH"] == "/usr/bin:/bin"

    def test_missing_variables_are_simply_absent(self, monkeypatch):
        monkeypatch.delenv("LC_ALL", raising=False)
        assert "LC_ALL" not in agent_env()
