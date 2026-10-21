"""cli.py: the terminal interface. Command-line parsing, the y/N approver, and what gets printed."""

import pytest

from harness import config, orchestrator
from harness.interfaces import cli
from harness.policy import approve_all, deny_all
from harness.sandbox import Unsandboxed, default_sandbox
from tests.fakes import FakeClient, reply


@pytest.fixture
def agents_dir(tmp_path, monkeypatch):
    directory = tmp_path / "agents"
    directory.mkdir()
    monkeypatch.setattr(config, "AGENTS_DIR", directory)
    for name in ("base", "reader"):
        (directory / f"{name}.md").write_text("---\nmodel: m\ntemperature: 0.1\nmax_steps: 3\n---\nPrompt.\n")
    return directory


class TestTerminalApprover:
    @pytest.mark.parametrize("answer", ["y", "Y", " y ", "y\n"])
    def test_y_approves(self, answer_prompt, answer):
        answer_prompt(answer)
        assert cli.terminal_approver("bash", {"command": "ls"}) is None

    @pytest.mark.parametrize("answer", ["", "n", "N", "no", "yes", "yy", "sure"])
    def test_anything_else_declines(self, answer_prompt, answer):
        # Only an explicit "y" approves. Even "yes" declines: strict beats surprising.
        answer_prompt(answer)
        assert cli.terminal_approver("bash", {"command": "ls"}) == (
            "The user declined this bash call. Try a different approach or ask the user."
        )

    def test_no_terminal_declines(self, answer_prompt):
        # No terminal is the same situation as unattended, so it gets the same message.
        answer_prompt(EOFError)
        assert cli.terminal_approver("bash", {"command": "ls"}) == deny_all("bash", {"command": "ls"})

    def test_shows_tool_and_arguments_before_asking(self, approve, capsys):
        cli.terminal_approver("bash", {"command": "echo visible-command", "timeout": 60})
        out = capsys.readouterr().out
        assert "agent wants to run bash" in out
        assert "command: echo visible-command" in out
        assert "timeout: 60" in out
        assert approve == ["  allow? [y/N] "]

    def test_long_argument_values_are_shortened(self, approve, capsys):
        cli.terminal_approver("write_file", {"content": "x" * 5000})
        out = capsys.readouterr().out
        assert "x" * 300 + "..." in out
        assert "x" * 301 not in out


class TestTurnSummary:
    def end(self, steps=3, tool_calls=2, prompt=1904, cached=0, seconds=4.21):
        return {"event": "run_end", "steps": steps, "tool_calls": tool_calls,
                "tokens": {"prompt": prompt, "completion": 50, "cached": cached}, "seconds": seconds}

    def test_format(self):
        assert cli.turn_summary(self.end()) == "(3 model calls · 2 tool calls · 1,904 prompt tokens, 0 cached · 4.2s)"

    def test_singular(self):
        assert cli.turn_summary(self.end(steps=1, tool_calls=1)).startswith("(1 model call · 1 tool call ·")

    def test_no_token_figures_without_usage(self):
        assert cli.turn_summary(self.end(prompt=0)) == "(3 model calls · 2 tool calls · 4.2s)"


class TestShowStep:
    def tool_call(self, result="hello", **overrides):
        return {"event": "tool_call", "step": 2, "name": "echo", "arguments": '{"text": "hi"}', "result": result, **overrides}

    def test_prints_the_call_and_its_result(self, capsys):
        cli.show_step(self.tool_call())
        assert capsys.readouterr().out == '[step 2] echo({"text": "hi"})\n  -> hello\n'

    def test_long_results_are_previewed(self, capsys):
        cli.show_step(self.tool_call(result="z" * 500))
        result_line = capsys.readouterr().out.splitlines()[1]
        assert result_line == "  -> " + "z" * 200 + "..."

    @pytest.mark.parametrize("kind", ["run_start", "model_response", "run_end"])
    def test_other_events_print_nothing(self, capsys, kind):
        cli.show_step({"event": kind})
        assert capsys.readouterr().out == ""


class FakeConversation:
    def __init__(self, calls, **kwargs):
        self.calls, self.kwargs = calls, kwargs
        self.session = type("S", (), {"id": "fake-session", "trace_path": lambda self, n: __import__("pathlib").Path(f"turn-{n:03d}.jsonl")})()

    def turn(self, task, on_event=None):
        self.calls.append({"task": task, **self.kwargs})
        self.on_event = on_event
        return orchestrator.TurnResult("printed answer", None)


class TestMain:
    @pytest.fixture
    def run_calls(self, agents_dir, monkeypatch):
        """Replace new_conversation; record what main() asks for and what it runs."""
        calls = []

        def fake_new(agent_name, **kwargs):
            return FakeConversation(calls, agent_name=agent_name, **kwargs)

        monkeypatch.setattr(cli, "new_conversation", fake_new)
        return calls

    def test_joins_words_into_one_task(self, run_calls, capsys):
        cli.main(["list", "the", "files"])
        assert run_calls == [
            {"task": "list the files", "agent_name": "base", "approver": cli.terminal_approver,
             "sandbox": default_sandbox(), "workspace": None, "environment": None}
        ]
        assert capsys.readouterr().out.strip() == "printed answer"

    def test_steps_are_shown_through_events(self, monkeypatch, agents_dir):
        convs = []
        monkeypatch.setattr(cli, "new_conversation", lambda name, **kw: convs.append(FakeConversation([], **kw)) or convs[-1])
        cli.main(["hi"])
        assert convs[0].on_event is cli.show_step

    def test_quoted_task(self, run_calls):
        cli.main(["list the files"])
        assert run_calls[0]["task"] == "list the files"

    def test_agent_flag(self, run_calls):
        cli.main(["--agent", "reader", "hi"])
        assert run_calls[0]["agent_name"] == "reader"

    def test_unattended_means_deny_all(self, run_calls):
        cli.main(["--unattended", "hi"])
        assert run_calls[0]["approver"] is deny_all

    def test_yes_means_approve_all(self, run_calls):
        cli.main(["--yes", "hi"])
        assert run_calls[0]["approver"] is approve_all

    def test_yes_with_no_sandbox_is_refused(self, run_calls, capsys):
        with pytest.raises(SystemExit) as exit_info:
            cli.main(["--yes", "--no-sandbox", "hi"])
        assert exit_info.value.code == 2
        assert "needs the sandbox" in capsys.readouterr().err
        assert run_calls == []  # refused before anything was created

    def test_yes_where_no_sandbox_is_available_is_refused(self, run_calls, monkeypatch, capsys):
        monkeypatch.setattr(cli, "default_sandbox", Unsandboxed)  # e.g. not macOS
        with pytest.raises(SystemExit):
            cli.main(["--yes", "hi"])
        assert "needs the sandbox" in capsys.readouterr().err

    def test_yes_and_unattended_are_exclusive(self, run_calls, capsys):
        with pytest.raises(SystemExit):
            cli.main(["--yes", "--unattended", "hi"])
        assert "not allowed with argument" in capsys.readouterr().err

    def test_sandboxed_by_default(self, run_calls):
        cli.main(["hi"])
        assert run_calls[0]["sandbox"] == default_sandbox()  # Seatbelt on macOS

    def test_no_sandbox_flag(self, run_calls):
        cli.main(["--no-sandbox", "hi"])
        assert run_calls[0]["sandbox"] == Unsandboxed()

    def test_flags_after_task(self, run_calls):
        cli.main(["do", "it", "--unattended", "--agent", "reader"])
        assert (run_calls[0]["task"], run_calls[0]["agent_name"], run_calls[0]["approver"]) == ("do it", "reader", deny_all)

    def test_workspace_flag(self, run_calls, tmp_path):
        cli.main(["--workspace", str(tmp_path), "hi"])
        assert run_calls[0]["workspace"] == tmp_path

    def test_workspace_flag_expands_home(self, run_calls):
        from pathlib import Path
        cli.main(["--workspace", "~", "hi"])
        assert run_calls[0]["workspace"] == Path.home()

    def test_environment_flag(self, run_calls):
        cli.main(["--environment", "none", "hi"])
        assert run_calls[0]["environment"] == "none"

    def test_unknown_agent_lists_real_ones(self, run_calls, capsys):
        with pytest.raises(SystemExit) as exit_info:
            cli.main(["--agent", "ghost", "hi"])
        assert exit_info.value.code == 2  # argparse's "bad usage" exit code
        assert "invalid choice: 'ghost' (choose from base, reader)" in capsys.readouterr().err
        assert run_calls == []

    @pytest.mark.parametrize("argv", [[], [""], ["   "]])
    def test_no_task_is_a_usage_error(self, run_calls, capsys, argv):
        with pytest.raises(SystemExit) as exit_info:
            cli.main(argv)
        assert exit_info.value.code == 2
        assert "usage: harness" in capsys.readouterr().err
        assert run_calls == []

    def test_help_lists_commands(self, run_calls, capsys):
        with pytest.raises(SystemExit) as exit_info:
            cli.main(["--help"])
        assert exit_info.value.code == 0
        out = capsys.readouterr().out
        assert "run" in out and "chat" in out

    def test_command_help_lists_options(self, run_calls, capsys):
        with pytest.raises(SystemExit):
            cli.main(["run", "--help"])
        out = capsys.readouterr().out
        assert "--agent" in out and "--unattended" in out and "--environment" in out

    def test_explicit_run_command(self, run_calls):
        cli.main(["run", "list", "files"])
        assert run_calls[0]["task"] == "list files"

    def test_reads_real_command_line_by_default(self, run_calls, monkeypatch):
        monkeypatch.setattr("sys.argv", ["harness", "from", "argv"])
        cli.main()
        assert run_calls[0]["task"] == "from argv"


class TestRunOutput:
    """`harness "task"` end to end: the answer on stdout, everything else on stderr."""

    def test_steps_are_shown_while_the_turn_runs(self, agents_dir, sessions_dir, monkeypatch, capsys):
        from tests.fakes import call
        (config.AGENTS_DIR / "base.md").write_text(
            "---\nmodel: m\ntemperature: 0.1\nmax_steps: 3\ntools:\n  - write_file\n---\nPrompt.\n"
        )
        replies = [reply(tool_calls=[call("write_file", {"path": "a.txt", "content": "x"})]), reply("done")]
        monkeypatch.setattr(orchestrator, "make_client", lambda: FakeClient(replies))
        cli.main(["write", "it"])
        out = capsys.readouterr().out
        assert '[step 1] write_file({"path": "a.txt", "content": "x"})' in out
        assert "  -> Wrote 1 characters to a.txt." in out
        assert out.strip().endswith("done")  # the answer comes last

    def test_answer_stdout_rest_stderr(self, agents_dir, sessions_dir, monkeypatch, capsys):
        monkeypatch.setattr(orchestrator, "make_client", lambda: FakeClient([reply("the answer")]))
        cli.main(["say", "hi"])
        captured = capsys.readouterr()
        assert captured.out.strip() == "the answer"
        (session,) = [p for p in sessions_dir.iterdir() if p.is_dir()]
        assert f"session: {session.name}  trace: turn-001.jsonl" in captured.err
        assert "(1 model call · 0 tool calls ·" in captured.err

    def test_session_printed_even_after_a_crash(self, agents_dir, sessions_dir, monkeypatch, capsys):
        monkeypatch.setattr(orchestrator, "make_client", lambda: FakeClient([]))  # the first model call fails
        with pytest.raises(AssertionError):
            cli.main(["say", "hi"])
        assert "session:" in capsys.readouterr().err
