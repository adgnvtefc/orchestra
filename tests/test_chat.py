"""orchestra chat: many turns in one session, resuming, and surviving interrupts and errors."""

import json

import pytest

from orchestra import config, orchestrator
from orchestra.interfaces import cli
from orchestra.session import INTERRUPTED
from orchestra.tools import REGISTRY, Tool
from tests.fakes import FakeClient, call, reply, use_model
from tests.test_records import read_log


@pytest.fixture
def sessions_dir(orchestra_dir):
    """The CLI's sessions: it works for the app "orchestra-cli", so ~/.orchestra/orchestra-cli/sessions/."""
    return orchestra_dir / "orchestra-cli" / "sessions"


@pytest.fixture
def agents_dir(tmp_path, monkeypatch, fake_tools):
    directory = tmp_path / "agents"
    directory.mkdir()
    monkeypatch.chdir(tmp_path)  # a temp project: its agents/ is found by walking up from here
    write_agent(directory)
    return directory


def write_agent(directory, model="m", tools=("echo",)):
    tools_yaml = "".join(f"  - {t}\n" for t in tools)
    (directory / "base.md").write_text(f"---\nmodel: {model}\ntemperature: 0.1\nmax_steps: 3\ntools:\n{tools_yaml}---\nChat prompt.\n")


@pytest.fixture
def model(monkeypatch):
    """Script the model's replies for the whole chat; returns the FakeClient."""

    def install(*replies):
        client = FakeClient(list(replies))
        use_model(monkeypatch, client)
        return client

    return install


def only_session(sessions_dir):
    (session,) = [p for p in sessions_dir.iterdir() if p.is_dir()]
    return session


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def events(session, kind):
    """The session trace's events of one kind, in order."""
    return [e for e in read(session / "trace.jsonl") if e["event"] == kind]


class TestNewChat:
    def test_turns_share_one_session(self, agents_dir, sessions_dir, typed, model):
        typed("first question", "second question")
        client = model(reply("first answer"), reply("second answer"))
        cli.main(["chat"])
        session = only_session(sessions_dir)
        # one trace for the session, both turns in it
        assert [e["turn"] for e in events(session, "turn_start")] == [1, 2]
        # turn 2 sees all of turn 1
        assert client.requests[1]["messages"] == [
            {"role": "system", "content": "Chat prompt."},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second question"},
        ]

    def test_answers_are_printed(self, agents_dir, typed, model, capsys):
        typed("hi")
        model(reply("hello there"))
        cli.main(["chat"])
        out = capsys.readouterr().out
        assert "hello there" in out
        assert "orchestra chat --resume" in out  # tells you how to come back

    @pytest.mark.parametrize("command", ["/exit", "/quit"])
    def test_exit_command(self, agents_dir, sessions_dir, typed, model, command):
        typed(command, "never sent")
        model()  # no model calls allowed
        cli.main(["chat"])
        assert list(only_session(sessions_dir).glob("turn-*.jsonl")) == []

    @pytest.mark.parametrize("key", [EOFError, KeyboardInterrupt])
    def test_ctrl_d_or_ctrl_c_at_the_prompt_leaves(self, agents_dir, typed, model, key):
        typed(key, "never sent")
        model()
        cli.main(["chat"])  # returns normally, no traceback

    def test_blank_lines_are_ignored(self, agents_dir, typed, model):
        typed("", "   ", "real question")
        client = model(reply("ok"))
        cli.main(["chat"])
        assert len(client.requests) == 1

    def test_tool_turns_work(self, agents_dir, sessions_dir, typed, model):
        typed("use the tool")
        model(reply(tool_calls=[call("echo", {"text": "hi"})]), reply("done"))
        cli.main(["chat"])
        events = read(only_session(sessions_dir) / "trace.jsonl")
        assert [e["event"] for e in events] == ["turn_start", "model_call", "tool_run", "model_call", "turn_end"]


class TestSurvivingProblems:
    def test_ctrl_c_mid_turn_repairs_and_continues(self, agents_dir, sessions_dir, typed, model, monkeypatch, capsys):
        # A tool that gets "Ctrl-C'd" while running leaves its call unanswered in the history.
        def interrupted(ctx) -> str:
            raise KeyboardInterrupt

        monkeypatch.setitem(REGISTRY, "slow", Tool(fn=interrupted, schema=_schema("slow"), permission="allow"))
        write_agent(agents_dir, tools=("slow",))
        typed("start something slow", "never mind, say hi")
        client = model(reply(tool_calls=[call("slow", {}, id="c1")]), reply("hi"))
        cli.main(["chat"])

        assert "(turn interrupted)" in capsys.readouterr().out
        # Turn 2's request is valid: the unanswered call got an INTERRUPTED result first.
        sent = client.requests[1]["messages"]
        assert {"role": "tool", "tool_call_id": "c1", "content": INTERRUPTED} in sent
        session = only_session(sessions_dir)
        assert [e["outcome"] for e in events(session, "turn_end")] == ["interrupted", "answered"]

    def test_error_mid_turn_reports_and_continues(self, agents_dir, sessions_dir, typed, model, capsys):
        typed("first", "second")
        model(ConnectionError("model server went away"), reply("back again"))
        cli.main(["chat"])
        captured = capsys.readouterr()
        assert "(turn failed: ConnectionError: model server went away)" in captured.err
        assert "back again" in captured.out
        session = only_session(sessions_dir)
        assert [e["outcome"] for e in events(session, "turn_end")] == ["crashed", "answered"]


class TestResume:
    def start_chat(self, typed, model, sessions_dir, *lines):
        typed(*lines)
        model(*[reply(f"answer to {line}") for line in lines])
        cli.main(["chat"])
        return only_session(sessions_dir)

    def test_continues_the_conversation(self, agents_dir, sessions_dir, typed, model):
        session = self.start_chat(typed, model, sessions_dir, "first")
        typed("second")
        client = model(reply("second answer"))
        cli.main(["chat", "--resume", session.name])
        assert client.requests[0]["messages"][-3:] == [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer to first"},
            {"role": "user", "content": "second"},
        ]
        assert [e["turn"] for e in events(session, "turn_start")] == [1, 2]  # numbering continues

    def test_announces_the_resume(self, agents_dir, sessions_dir, typed, model, capsys):
        session = self.start_chat(typed, model, sessions_dir, "first")
        capsys.readouterr()
        typed()
        model()
        cli.main(["chat", "--resume", session.name])
        assert "resumed, 1 turns so far" in capsys.readouterr().out

    def test_keeps_the_pinned_system_prompt(self, agents_dir, sessions_dir, typed, model):
        session = self.start_chat(typed, model, sessions_dir, "first")
        (agents_dir / "base.md").write_text("---\nmodel: m\ntemperature: 0.1\nmax_steps: 3\n---\nEDITED prompt.\n")
        typed("second")
        client = model(reply("ok"))
        cli.main(["chat", "--resume", session.name])
        assert client.requests[0]["messages"][0] == {"role": "system", "content": "Chat prompt."}

    def test_uses_the_agent_files_current_model(self, agents_dir, sessions_dir, typed, model, orchestra_dir):
        session = self.start_chat(typed, model, sessions_dir, "first")
        write_agent(agents_dir, model="newer-model")
        typed("second")
        client = model(reply("ok"))
        cli.main(["chat", "--resume", session.name])
        assert client.requests[0]["model"] == "newer-model"
        assert read_log(orchestra_dir, "orchestra-cli")[-1]["request"]["model"] == "newer-model"  # the record says so

    def test_keeps_the_workspace(self, agents_dir, sessions_dir, typed, model):
        write_agent(agents_dir, tools=("write_file",))
        typed("write it")
        model(reply(tool_calls=[call("write_file", {"path": "note.txt", "content": "hi"})]), reply("ok"))
        cli.main(["chat"])
        session = only_session(sessions_dir)
        typed("read it")
        write_agent(agents_dir, tools=("read_file",))
        client = model(reply(tool_calls=[call("read_file", {"path": "note.txt"})]), reply("ok"))
        cli.main(["chat", "--resume", session.name])
        assert client.requests[1]["messages"][-1]["content"] == "hi"  # same workspace, same file

    def test_unknown_session(self, agents_dir, sessions_dir, capsys):
        with pytest.raises(SystemExit):
            cli.main(["chat", "--resume", "no-such-session"])
        assert "no session 'no-such-session'" in capsys.readouterr().err

    @pytest.mark.parametrize("flag", [["--agent", "base"], ["--workspace", "."], ["--environment", "none"]])
    def test_fixed_settings_cannot_change(self, agents_dir, sessions_dir, typed, model, flag, capsys):
        session = self.start_chat(typed, model, sessions_dir, "first")
        with pytest.raises(SystemExit):
            cli.main(["chat", "--resume", session.name, *flag])
        assert "can't be changed when resuming" in capsys.readouterr().err

    def test_resume_session_records_environment_in_config(self, agents_dir, sessions_dir, typed, model, monkeypatch):
        session = self.start_chat(typed, model, sessions_dir, "first")
        meta = json.loads((session / "session.json").read_text())
        meta["environment"] = {"name": "py", "version": "abc123", "variables": {}}
        (session / "session.json").write_text(json.dumps(meta))
        agent = orchestrator.resume_conversation(session.name).agent
        assert agent["environment"] == {"name": "py", "version": "abc123"}  # from the session, not the file


def _schema(name):
    return {"type": "function", "function": {"name": name, "description": "test", "parameters": {"type": "object", "properties": {}}}}
