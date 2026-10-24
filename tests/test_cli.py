"""cli.py: the terminal interface. Command-line parsing, the y/N approver, and what gets printed."""

import pytest

from orchestra import config, orchestrator
from orchestra.interfaces import cli
from orchestra.config import Project
from orchestra.policy import approve_all, deny_all
from orchestra.sandbox import Unsandboxed, default_sandbox
from tests.fakes import FakeClient, reply


@pytest.fixture
def agents_dir(tmp_path, monkeypatch):
    directory = tmp_path / "agents"
    directory.mkdir()
    monkeypatch.chdir(tmp_path)  # a temp project: its agents/ is found by walking up from here
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
        captured = capsys.readouterr()
        assert "agent wants to run bash" in captured.err
        assert "command: echo visible-command" in captured.err
        assert "timeout: 60" in captured.err
        assert captured.err.endswith("  allow? [y/N] ")  # the question, last, without a newline
        assert captured.out == ""  # stdout is only for the answer
        assert len(approve) == 1  # asked once

    def test_long_argument_values_are_shortened(self, approve, capsys):
        cli.terminal_approver("write_file", {"content": "x" * 5000})
        err = capsys.readouterr().err
        assert "x" * 300 + "..." in err
        assert "x" * 301 not in err


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
        captured = capsys.readouterr()
        assert captured.err == '[step 2] echo({"text": "hi"})\n  -> hello\n'
        assert captured.out == ""

    def test_long_results_are_previewed(self, capsys):
        cli.show_step(self.tool_call(result="z" * 500))
        result_line = capsys.readouterr().err.splitlines()[1]
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

    def test_joins_words_into_one_task(self, run_calls, agents_dir, capsys):
        cli.main(["list", "the", "files"])
        assert run_calls == [
            {"task": "list the files", "agent_name": "base", "project": Project(agents_dir.parent.resolve()),
             "approver": cli.terminal_approver, "sandbox": default_sandbox(), "workspace": None, "environment": None}
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

    def test_config_root_found_from_a_subfolder(self, run_calls, agents_dir, monkeypatch):
        sub = agents_dir.parent / "deep" / "er"
        sub.mkdir(parents=True)
        monkeypatch.chdir(sub)
        cli.main(["hi"])
        assert run_calls[0]["project"].root == agents_dir.parent.resolve()

    def test_config_root_flag(self, run_calls, tmp_path, monkeypatch):
        other = tmp_path / "other-project"
        (other / "agents").mkdir(parents=True)
        (other / "agents" / "base.md").write_text("---\nmodel: m\n---\nP\n")
        monkeypatch.chdir(tmp_path.parent)  # somewhere with no agents/ of its own
        cli.main(["--config-root", str(other), "hi"])
        assert run_calls[0]["project"].root == other

    def test_no_agents_folder_anywhere(self, run_calls, tmp_path, monkeypatch, capsys):
        empty = tmp_path / "empty"
        empty.mkdir()
        monkeypatch.chdir(empty)
        def nothing_found(start, extra_tools=()):
            raise FileNotFoundError(f"no agents/ folder in {start}")

        monkeypatch.setattr(cli.Project, "find", staticmethod(nothing_found))
        with pytest.raises(SystemExit):
            cli.main(["hi"])
        assert "no agents/ folder" in capsys.readouterr().err
        assert run_calls == []

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
        assert "usage: orchestra" in capsys.readouterr().err
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
        monkeypatch.setattr("sys.argv", ["orchestra", "from", "argv"])
        cli.main()
        assert run_calls[0]["task"] == "from argv"


class TestApiKey:
    def test_cli_loads_dotenv(self, agents_dir, monkeypatch, no_dotenv):
        monkeypatch.setattr(orchestrator, "make_client", lambda: FakeClient([reply("ok")]))
        cli.main(["hi"])
        assert len(no_dotenv) == 1  # the CLI (the interface) loads .env, once

    def test_missing_key_is_a_clean_usage_error(self, agents_dir, sessions_dir, monkeypatch, capsys):
        monkeypatch.delenv("OMLX_API_KEY", raising=False)
        with pytest.raises(SystemExit) as exit_info:
            cli.main(["hi"])
        assert exit_info.value.code == 2
        assert "OMLX_API_KEY is not set" in capsys.readouterr().err
        assert not sessions_dir.exists() or not any(sessions_dir.iterdir())  # nothing was created


class TestRunOutput:
    """`orchestra "task"` end to end: the answer on stdout, everything else on stderr."""

    def test_steps_are_shown_while_the_turn_runs(self, agents_dir, sessions_dir, monkeypatch, capsys):
        from tests.fakes import call
        (agents_dir / "base.md").write_text(
            "---\nmodel: m\ntemperature: 0.1\nmax_steps: 3\ntools:\n  - write_file\n---\nPrompt.\n"
        )
        replies = [reply(tool_calls=[call("write_file", {"path": "a.txt", "content": "x"})]), reply("done")]
        monkeypatch.setattr(orchestrator, "make_client", lambda: FakeClient(replies))
        cli.main(["write", "it"])
        captured = capsys.readouterr()
        assert '[step 1] write_file({"path": "a.txt", "content": "x"})' in captured.err
        assert "  -> Wrote 1 characters to a.txt." in captured.err
        # The README's promise: with tool calls and all, stdout is the answer and nothing else.
        assert captured.out == "done\n"

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


class TestProjectTools:
    """The CLI runs agents that use a project's own tools, from the project's tools.py."""

    TOOLS_PY = (
        "from orchestra.tools import Tool\n"
        "def weather(ctx, city: str) -> str:\n"
        "    return f'sunny in {city}'\n"
        "TOOLS = [Tool(fn=weather, permission='allow', schema={'type': 'function', 'function': {\n"
        "    'name': 'weather', 'description': 'the weather',\n"
        "    'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}})]\n"
    )

    @pytest.fixture
    def project(self, tmp_path, monkeypatch):
        (tmp_path / "agents").mkdir()
        (tmp_path / "agents" / "base.md").write_text("---\nmodel: m\ntemperature: 0.1\nmax_steps: 3\ntools:\n  - weather\n---\nP\n")
        (tmp_path / "tools.py").write_text(self.TOOLS_PY)
        monkeypatch.chdir(tmp_path)
        return tmp_path

    def weather_then_answer(self, monkeypatch):
        from tests.fakes import call
        client = FakeClient([reply(tool_calls=[call("weather", {"city": "Paris"})]), reply("It's sunny.")])
        monkeypatch.setattr(orchestrator, "make_client", lambda: client)
        return client

    def test_run_uses_the_projects_tool(self, project, monkeypatch, capsys):
        client = self.weather_then_answer(monkeypatch)
        cli.main(["weather in Paris?"])
        assert capsys.readouterr().out.strip() == "It's sunny."
        assert client.requests[1]["messages"][-1]["content"] == "sunny in Paris"  # the project's tool ran

    def test_config_root_flag_loads_them_too(self, project, monkeypatch, tmp_path, capsys):
        client = self.weather_then_answer(monkeypatch)
        elsewhere = tmp_path.parent
        monkeypatch.chdir(elsewhere)
        cli.main(["--config-root", str(project), "weather in Paris?"])
        assert client.requests[1]["messages"][-1]["content"] == "sunny in Paris"

    def test_resume_finds_them(self, project, monkeypatch, typed):
        client = self.weather_then_answer(monkeypatch)
        typed("weather in Paris?")
        cli.main(["chat"])
        (session_dir,) = list(orchestrator.SESSIONS_DIR.iterdir())
        session_id = session_dir.name
        monkeypatch.chdir(project.parent)  # resume from somewhere else: the session knows its project
        from tests.fakes import call
        client2 = FakeClient([reply(tool_calls=[call("weather", {"city": "Rome"})]), reply("ok")])
        monkeypatch.setattr(orchestrator, "make_client", lambda: client2)
        typed("and Rome?")
        cli.main(["chat", "--resume", session_id])
        assert client2.requests[1]["messages"][-1]["content"] == "sunny in Rome"

    def test_broken_tools_py_is_a_usage_error(self, project, capsys):
        (project / "tools.py").write_text("TOOLS = 'nope'\n")
        with pytest.raises(SystemExit) as exit_info:
            cli.main(["hi"])
        assert exit_info.value.code == 2
        assert "must define TOOLS" in capsys.readouterr().err
