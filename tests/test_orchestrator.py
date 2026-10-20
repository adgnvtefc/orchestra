"""The orchestrator: building the client, wiring config -> toolset -> agent, and the CLI."""

import json
import re
from pathlib import Path

import pytest

from harness import config_loader, orchestrator
from harness.policy import deny_all, terminal_approver
from harness.tools import Toolset
from tests.fakes import FakeClient, reply


@pytest.fixture
def env(monkeypatch):
    """A controlled environment: the real .env is never read, and OMLX_* vars start unset."""
    monkeypatch.setattr(orchestrator, "load_dotenv", lambda: None)
    monkeypatch.delenv("OMLX_API_KEY", raising=False)
    monkeypatch.delenv("OMLX_BASE_URL", raising=False)
    return monkeypatch


class TestMakeClient:
    def test_missing_key_exits_with_help(self, env):
        with pytest.raises(SystemExit, match="OMLX_API_KEY is not set"):
            orchestrator.make_client()

    def test_empty_key_counts_as_missing(self, env):
        env.setenv("OMLX_API_KEY", "")
        with pytest.raises(SystemExit):
            orchestrator.make_client()

    def test_default_base_url(self, env):
        env.setenv("OMLX_API_KEY", "test-key")
        client = orchestrator.make_client()
        assert str(client.base_url).rstrip("/") == "http://127.0.0.1:8000/v1"
        assert client.api_key == "test-key"

    def test_base_url_from_env(self, env):
        env.setenv("OMLX_API_KEY", "test-key")
        env.setenv("OMLX_BASE_URL", "http://example.test:9999/v1")
        assert str(orchestrator.make_client().base_url).rstrip("/") == "http://example.test:9999/v1"

    def test_long_timeout_for_local_models(self, env):
        env.setenv("OMLX_API_KEY", "test-key")
        assert orchestrator.make_client().timeout == 600


@pytest.fixture
def agents_dir(tmp_path, monkeypatch):
    """A temp agents/ folder the orchestrator reads from, plus a FakeClient it builds."""
    directory = tmp_path / "agents"
    directory.mkdir()
    monkeypatch.setattr(config_loader, "AGENTS_DIR", directory)
    return directory


def write_agent(directory, name, tools=None, prompt="Test prompt."):
    tools_yaml = "" if tools is None else "tools:\n" + "".join(f"  - {t}\n" for t in tools)
    (directory / f"{name}.md").write_text(f"---\nmodel: m\ntemperature: 0.1\nmax_steps: 3\n{tools_yaml}---\n{prompt}\n")


@pytest.fixture
def captured_run(monkeypatch):
    """Replace make_client and run_agent; record what the orchestrator hands the agent."""
    seen = {}

    def fake_run_agent(task, session, config, client, toolset, tracer):
        seen.update(task=task, session=session, config=config, client=client, toolset=toolset, tracer=tracer)
        return "agent result"

    monkeypatch.setattr(orchestrator, "make_client", lambda: FakeClient([reply("unused")]))
    monkeypatch.setattr(orchestrator, "run_agent", fake_run_agent)
    return seen


class TestRun:
    def test_wires_config_client_and_toolset(self, agents_dir, captured_run):
        write_agent(agents_dir, "base", tools=["read_file", "bash"])
        assert orchestrator.run("a task") == "agent result"
        assert captured_run["task"] == "a task"
        assert captured_run["config"]["model"] == "m"
        assert captured_run["config"]["system_prompt"] == "Test prompt."
        assert isinstance(captured_run["client"], FakeClient)
        assert isinstance(captured_run["toolset"], Toolset)
        assert list(captured_run["toolset"].tools) == ["read_file", "bash"]
        assert captured_run["toolset"].policy.approver is terminal_approver  # a human answers "ask"

    def test_unattended_uses_deny_all(self, agents_dir, captured_run):
        write_agent(agents_dir, "base", tools=["bash"])
        orchestrator.run("t", unattended=True)
        assert captured_run["toolset"].policy.approver is deny_all

    def test_default_agent_is_base(self, agents_dir, captured_run):
        write_agent(agents_dir, "base", prompt="I am base.")
        orchestrator.run("t")
        assert captured_run["config"]["system_prompt"] == "I am base."

    def test_named_agent(self, agents_dir, captured_run):
        write_agent(agents_dir, "base")
        write_agent(agents_dir, "reader", tools=["read_file"], prompt="I am reader.")
        orchestrator.run("t", agent_name="reader")
        assert captured_run["config"]["system_prompt"] == "I am reader."
        assert list(captured_run["toolset"].tools) == ["read_file"]

    def test_no_tools_key_means_no_tools(self, agents_dir, captured_run):
        # Access is opt-in: forgetting `tools:` must not hand out bash.
        write_agent(agents_dir, "base", tools=None)
        orchestrator.run("t")
        assert captured_run["toolset"].tools == {}

    def test_unknown_agent(self, agents_dir, captured_run):
        with pytest.raises(FileNotFoundError):
            orchestrator.run("t", agent_name="ghost")

    def test_tool_typo_fails_before_any_model_call(self, agents_dir, captured_run):
        write_agent(agents_dir, "base", tools=["read_fle"])
        with pytest.raises(ValueError, match="read_fle"):
            orchestrator.run("t")
        assert captured_run == {}  # the agent never started

    def test_config_includes_agent_name(self, agents_dir, captured_run):
        write_agent(agents_dir, "reader")
        orchestrator.run("t", agent_name="reader")
        assert captured_run["config"]["name"] == "reader"

    def test_end_to_end_with_fake_model(self, agents_dir, monkeypatch):
        # Not mocking run_agent: the real loop runs, only the model is scripted.
        write_agent(agents_dir, "base", tools=["read_file"])
        client = FakeClient([reply("hello from the fake model")])
        monkeypatch.setattr(orchestrator, "make_client", lambda: client)
        assert orchestrator.run("say hi") == "hello from the fake model"
        assert client.requests[0]["messages"][0]["content"] == "Test prompt."


class TestMain:
    @pytest.fixture
    def run_calls(self, agents_dir, monkeypatch):
        """Replace orchestrator.run; record the keyword arguments main() passes it."""
        write_agent(agents_dir, "base")
        write_agent(agents_dir, "reader")
        calls = []

        def fake_run(task, **kwargs):
            calls.append({"task": task, **kwargs})
            return "printed answer"

        monkeypatch.setattr(orchestrator, "run", fake_run)
        return calls

    def test_joins_words_into_one_task(self, run_calls, capsys):
        orchestrator.main(["list", "the", "files"])
        assert run_calls == [
            {"task": "list the files", "agent_name": "base", "unattended": False, "workspace": None, "environment": None}
        ]
        assert capsys.readouterr().out.strip() == "printed answer"

    def test_quoted_task(self, run_calls):
        orchestrator.main(["list the files"])
        assert run_calls[0]["task"] == "list the files"

    def test_agent_flag(self, run_calls):
        orchestrator.main(["--agent", "reader", "hi"])
        assert run_calls[0]["agent_name"] == "reader"

    def test_unattended_flag(self, run_calls):
        orchestrator.main(["--unattended", "hi"])
        assert run_calls[0]["unattended"] is True

    def test_flags_after_task(self, run_calls):
        orchestrator.main(["do", "it", "--unattended", "--agent", "reader"])
        assert run_calls == [
            {"task": "do it", "agent_name": "reader", "unattended": True, "workspace": None, "environment": None}
        ]

    def test_workspace_flag(self, run_calls, tmp_path):
        orchestrator.main(["--workspace", str(tmp_path), "hi"])
        assert run_calls[0]["workspace"] == tmp_path

    def test_workspace_flag_expands_home(self, run_calls):
        orchestrator.main(["--workspace", "~", "hi"])
        assert run_calls[0]["workspace"] == Path.home()

    def test_environment_flag(self, run_calls):
        orchestrator.main(["--environment", "none", "hi"])
        assert run_calls[0]["environment"] == "none"

    def test_unknown_agent_lists_real_ones(self, run_calls, capsys):
        with pytest.raises(SystemExit) as exit_info:
            orchestrator.main(["--agent", "ghost", "hi"])
        assert exit_info.value.code == 2  # argparse's "bad usage" exit code
        assert "invalid choice: 'ghost' (choose from base, reader)" in capsys.readouterr().err
        assert run_calls == []

    @pytest.mark.parametrize("argv", [[], [""], ["   "]])
    def test_no_task_is_a_usage_error(self, run_calls, capsys, argv):
        with pytest.raises(SystemExit) as exit_info:
            orchestrator.main(argv)
        assert exit_info.value.code == 2
        assert "usage: harness" in capsys.readouterr().err
        assert run_calls == []

    def test_help(self, run_calls, capsys):
        with pytest.raises(SystemExit) as exit_info:
            orchestrator.main(["--help"])
        assert exit_info.value.code == 0
        out = capsys.readouterr().out
        assert "--agent" in out and "--unattended" in out

    def test_reads_real_command_line_by_default(self, run_calls, monkeypatch):
        monkeypatch.setattr("sys.argv", ["harness", "from", "argv"])
        orchestrator.main()
        assert run_calls[0]["task"] == "from argv"


class TestSessionsAndTraces:
    """Each `harness "task"` run is a new session with one turn: sessions/<id>/turn-001.jsonl."""

    @pytest.fixture
    def fake_model(self, agents_dir, monkeypatch):
        write_agent(agents_dir, "base", tools=["read_file"], prompt="Pinned prompt.")
        monkeypatch.setattr(orchestrator, "make_client", lambda: FakeClient([reply("the answer")]))

    def session_dirs(self, sessions_dir):
        return sorted(p for p in sessions_dir.iterdir() if p.is_dir())

    def read(self, path):
        return [json.loads(line) for line in path.read_text().splitlines()]

    def test_each_run_is_a_new_session(self, fake_model, sessions_dir, monkeypatch):
        orchestrator.run("t")
        monkeypatch.setattr(orchestrator, "make_client", lambda: FakeClient([reply("again")]))
        orchestrator.run("t")
        assert len(self.session_dirs(sessions_dir)) == 2

    def test_session_folder_contents(self, fake_model, sessions_dir):
        orchestrator.run("t")
        (session,) = self.session_dirs(sessions_dir)
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}_base_[0-9a-f]{4}", session.name)
        assert sorted(p.name for p in session.iterdir()) == ["messages.jsonl", "session.json", "turn-001.jsonl", "workspace"]

    def test_session_records_agent_and_version(self, fake_model, sessions_dir):
        orchestrator.run("t")
        (session,) = self.session_dirs(sessions_dir)
        meta = json.loads((session / "session.json").read_text())
        trace = self.read(session / "turn-001.jsonl")
        assert meta["agent"] == "base"
        assert meta["agent_version"] == trace[0]["provenance"]["agent_version"]  # same config, same hash

    def test_messages_saved(self, fake_model, sessions_dir):
        orchestrator.run("say hi")
        (session,) = self.session_dirs(sessions_dir)
        assert self.read(session / "messages.jsonl") == [
            {"role": "system", "content": "Pinned prompt."},
            {"role": "user", "content": "say hi"},
            {"role": "assistant", "content": "the answer"},
        ]

    def test_trace_contents(self, fake_model, sessions_dir):
        orchestrator.run("say hi")
        (session,) = self.session_dirs(sessions_dir)
        events = self.read(session / "turn-001.jsonl")
        assert [e["event"] for e in events] == ["run_start", "model_response", "run_end"]
        assert events[0]["task"] == "say hi"
        assert events[0]["config"]["name"] == "base"
        assert events[0]["approver"] == "terminal_approver"
        assert events[-1]["answer"] == "the answer"

    def test_unattended_is_visible_in_trace(self, fake_model, sessions_dir):
        orchestrator.run("t", unattended=True)
        (session,) = self.session_dirs(sessions_dir)
        assert self.read(session / "turn-001.jsonl")[0]["approver"] == "deny_all"

    def test_session_and_trace_printed_to_stderr(self, fake_model, sessions_dir, capsys):
        orchestrator.run("t")
        (session,) = self.session_dirs(sessions_dir)
        captured = capsys.readouterr()
        assert f"session: {session.name}  trace: turn-001.jsonl" in captured.err
        assert "session:" not in captured.out  # stdout stays clean for the answer

    def test_crashed_run_still_leaves_session_and_trace(self, agents_dir, sessions_dir, monkeypatch, capsys):
        write_agent(agents_dir, "base")
        monkeypatch.setattr(orchestrator, "make_client", lambda: FakeClient([]))  # first model call fails
        with pytest.raises(AssertionError):
            orchestrator.run("t")
        (session,) = self.session_dirs(sessions_dir)
        assert self.read(session / "turn-001.jsonl")[-1]["outcome"] == "crashed"
        assert self.read(session / "messages.jsonl")[-1] == {"role": "user", "content": "t"}  # saved before the crash
        assert "session:" in capsys.readouterr().err  # printed even after a crash


class TestWorkspaces:
    """Where the agent works: --workspace, then the agent file's `workspace:`, then a fresh folder."""

    @pytest.fixture
    def writes_a_file(self, agents_dir, monkeypatch):
        """A fake model that writes marker.txt, then answers."""
        from tests.fakes import call

        def install(workspace_line=""):
            (agents_dir / "base.md").write_text(
                f"---\nmodel: m\ntemperature: 0.1\nmax_steps: 3\ntools:\n  - write_file\n{workspace_line}---\nPrompt.\n"
            )
            replies = [reply(tool_calls=[call("write_file", {"path": "marker.txt", "content": "hi"})]), reply("done")]
            monkeypatch.setattr(orchestrator, "make_client", lambda: FakeClient(replies))

        return install

    def only_session(self, sessions_dir):
        (session,) = [p for p in sessions_dir.iterdir() if p.is_dir()]
        return session

    def test_default_fresh_folder_inside_the_session(self, writes_a_file, sessions_dir):
        writes_a_file()
        orchestrator.run("t")
        assert (self.only_session(sessions_dir) / "workspace" / "marker.txt").read_text() == "hi"

    def test_cli_workspace(self, writes_a_file, sessions_dir, tmp_path):
        writes_a_file()
        project = tmp_path / "project"
        project.mkdir()
        orchestrator.run("t", workspace=project)
        assert (project / "marker.txt").read_text() == "hi"
        meta = json.loads((self.only_session(sessions_dir) / "session.json").read_text())
        assert meta["workspace"] == str(project.resolve())

    def test_agent_file_workspace(self, writes_a_file, tmp_path):
        project = tmp_path / "from_config"
        project.mkdir()
        writes_a_file(f"workspace: {project}\n")
        orchestrator.run("t")
        assert (project / "marker.txt").read_text() == "hi"

    def test_cli_wins_over_agent_file(self, writes_a_file, tmp_path):
        from_config, from_cli = tmp_path / "from_config", tmp_path / "from_cli"
        from_config.mkdir(), from_cli.mkdir()
        writes_a_file(f"workspace: {from_config}\n")
        orchestrator.run("t", workspace=from_cli)
        assert (from_cli / "marker.txt").exists()
        assert not (from_config / "marker.txt").exists()

    def test_agent_file_relative_path_is_from_project_root(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config_loader, "PROJECT_ROOT", tmp_path)
        (tmp_path / "sub").mkdir()
        assert config_loader.resolve_workspace(None, "sub") == (tmp_path / "sub").resolve()

    def test_agent_file_home_path(self):
        assert config_loader.resolve_workspace(None, "~") == Path.home().resolve()

    def test_none_means_fresh(self):
        assert config_loader.resolve_workspace(None, None) is None

    @pytest.mark.parametrize("source", ["cli", "config"])
    def test_missing_folder_fails_before_anything_is_created(self, writes_a_file, sessions_dir, tmp_path, source):
        missing = tmp_path / "typo"
        writes_a_file(f"workspace: {missing}\n" if source == "config" else "")
        with pytest.raises(ValueError, match="not an existing folder"):
            orchestrator.run("t", workspace=missing if source == "cli" else None)
        assert not missing.exists()  # not quietly created
        assert not sessions_dir.exists() or not any(sessions_dir.iterdir())  # no session either

    def test_workspace_is_part_of_the_agent_version(self, tmp_path):
        from harness.provenance import agent_version
        base = {"model": "m", "system_prompt": "p"}
        assert agent_version(base) != agent_version({**base, "workspace": str(tmp_path)})


class TestEnvironments:
    """A session's workspace is set up from its environment before the first turn."""

    @pytest.fixture
    def setup_env(self, agents_dir, tmp_path, monkeypatch):
        """Write environments/greet.yaml and an agent using it; the fake model runs `echo $GREETING`."""
        from tests.fakes import call

        envs = tmp_path / "environments"
        envs.mkdir()
        monkeypatch.setattr(config_loader, "ENVIRONMENTS_DIR", envs)
        (envs / "greet.yaml").write_text(
            "setup:\n  - echo built > made.txt\nenv:\n  GREETING: hello\nprompt: |\n  You have made.txt.\n"
        )
        (envs / "broken.yaml").write_text("setup:\n  - exit 3\n")

        def install(environment_line="environment: greet\n", replies=None):
            (agents_dir / "base.md").write_text(
                f"---\nmodel: m\ntemperature: 0.1\nmax_steps: 3\ntools:\n  - bash\n{environment_line}---\nBase prompt.\n"
            )
            replies = replies or [reply(tool_calls=[call("bash", {"command": 'echo "[$GREETING]"'})]), reply("done")]
            client = FakeClient(replies)
            monkeypatch.setattr(orchestrator, "make_client", lambda: client)
            return client

        install.envs = envs
        return install

    def only_session(self, sessions_dir):
        (session,) = [p for p in sessions_dir.iterdir() if p.is_dir()]
        return session

    def test_workspace_is_set_up_before_the_first_turn(self, setup_env, sessions_dir, approve):
        setup_env()
        orchestrator.run("t")
        assert (self.only_session(sessions_dir) / "workspace" / "made.txt").read_text() == "built\n"

    def test_agent_commands_see_the_environment(self, setup_env, approve):
        client = setup_env()
        orchestrator.run("t")
        tool_result = [m for m in client.requests[1]["messages"] if m["role"] == "tool"][0]["content"]
        assert tool_result.startswith("[hello]")

    def test_prompt_is_added_to_the_system_prompt(self, setup_env, approve):
        client = setup_env()
        orchestrator.run("t")
        assert client.requests[0]["messages"][0]["content"] == "Base prompt.\n\nYou have made.txt."

    def test_recorded_in_session_and_trace(self, setup_env, sessions_dir, approve):
        setup_env()
        orchestrator.run("t")
        session = self.only_session(sessions_dir)
        meta = json.loads((session / "session.json").read_text())
        assert meta["environment"]["name"] == "greet"
        assert meta["environment"]["variables"] == {"GREETING": "hello"}
        start = json.loads((session / "turn-001.jsonl").read_text().splitlines()[0])
        assert start["config"]["environment"] == {"name": "greet", "version": meta["environment"]["version"]}

    def test_environment_content_changes_the_agent_version(self, setup_env, sessions_dir, approve):
        setup_env()
        orchestrator.run("t")
        (setup_env.envs / "greet.yaml").write_text("env:\n  GREETING: changed\n")
        setup_env()
        orchestrator.run("t")
        versions = {json.loads((s / "session.json").read_text())["agent_version"] for s in sessions_dir.iterdir()}
        assert len(versions) == 2  # same agent file, different environment content: a different agent

    def test_cli_none_overrides_agent_file(self, setup_env, sessions_dir, approve):
        client = setup_env()
        orchestrator.run("t", environment="none")
        assert not (self.only_session(sessions_dir) / "workspace" / "made.txt").exists()
        assert client.requests[0]["messages"][0]["content"] == "Base prompt."

    def test_cli_names_an_environment(self, setup_env, sessions_dir, approve):
        setup_env(environment_line="")
        orchestrator.run("t", environment="greet")
        assert (self.only_session(sessions_dir) / "workspace" / "made.txt").exists()

    def test_unknown_environment_fails_before_anything_is_created(self, setup_env, sessions_dir):
        setup_env(environment_line="environment: nope\n")
        with pytest.raises(Exception, match="no environment 'nope'"):
            orchestrator.run("t")
        assert not sessions_dir.exists() or not any(sessions_dir.iterdir())

    def test_failed_setup_leaves_no_session(self, setup_env, sessions_dir):
        setup_env(environment_line="environment: broken\n")
        with pytest.raises(Exception, match="exit 3"):
            orchestrator.run("t")
        assert not any(sessions_dir.iterdir())

    def test_tool_typo_fails_before_anything_is_created(self, setup_env, agents_dir, sessions_dir):
        (agents_dir / "base.md").write_text("---\nmodel: m\ntemperature: 0.1\nmax_steps: 3\ntools:\n  - bsh\n---\nP\n")
        with pytest.raises(ValueError, match="bsh"):
            orchestrator.run("t")
        assert not sessions_dir.exists() or not any(sessions_dir.iterdir())
