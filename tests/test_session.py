"""Sessions: creating, appending, reloading, turn numbering, and repairing interrupted histories."""

import json
import re
from pathlib import Path

import pytest

from harness.environment import Environment
from harness.sandbox import SandboxRules
from harness.session import INTERRUPTED, MESSAGES_FILE, SESSION_FILE, Session, new_session_id


@pytest.fixture
def root(tmp_path):
    return tmp_path / "sessions"


@pytest.fixture
def session(root):
    return Session.create(root, agent="base", system_prompt="You are a test agent.", agent_version="abc123def456")


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def assistant_calling(*ids):
    """An assistant message asking for one tool call per id."""
    return {
        "role": "assistant",
        "tool_calls": [{"id": i, "type": "function", "function": {"name": "echo", "arguments": "{}"}} for i in ids],
    }


def tool_result(call_id, content="ok"):
    return {"role": "tool", "tool_call_id": call_id, "content": content}


class TestNewSessionId:
    def test_format(self):
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}_base_[0-9a-f]{4}", new_session_id("base"))

    def test_unique_within_a_second(self):
        assert len({new_session_id("base") for _ in range(50)}) == 50


class TestCreate:
    def test_makes_folder_and_files(self, session, root):
        assert session.dir.parent == root
        assert (session.dir / SESSION_FILE).is_file()
        assert (session.dir / MESSAGES_FILE).is_file()

    def test_meta(self, session):
        meta = json.loads((session.dir / SESSION_FILE).read_text())
        assert meta["id"] == session.id == session.dir.name
        assert meta["agent"] == session.agent == "base"
        assert meta["agent_version"] == "abc123def456"
        assert "created" in meta
        assert set(meta) == {"id", "agent", "agent_version", "workspace", "config_root", "environment", "created"}
        assert meta["config_root"] is None  # not given here
        assert meta["environment"] is None

    def test_system_prompt_is_message_zero(self, session):
        assert session.messages == ({"role": "system", "content": "You are a test agent."},)
        assert lines(session.dir / MESSAGES_FILE) == list(session.messages)

    def test_two_sessions_never_share_a_folder(self, root):
        a = Session.create(root, "base", "p", "v")
        b = Session.create(root, "base", "p", "v")
        assert a.dir != b.dir

    def test_next_turn_starts_at_one(self, session):
        assert session.next_turn == 1


class TestWorkspace:
    def test_default_is_a_fresh_folder_inside_the_session(self, session):
        assert session.workspace == (session.dir / "workspace").resolve()
        assert session.workspace.is_dir()
        assert list(session.workspace.iterdir()) == []

    def test_sessions_never_share_a_default_workspace(self, root):
        a = Session.create(root, "base", "p", "v")
        b = Session.create(root, "base", "p", "v")
        assert a.workspace != b.workspace

    def test_given_folder_is_used_and_not_copied(self, root, tmp_path):
        project = tmp_path / "project"
        project.mkdir()
        session = Session.create(root, "base", "p", "v", workspace=project)
        assert session.workspace == project.resolve()
        assert not (session.dir / "workspace").exists()  # no unused default folder

    def test_recorded_and_kept_on_resume(self, root, tmp_path):
        project = tmp_path / "project"
        project.mkdir()
        session = Session.create(root, "base", "p", "v", workspace=project)
        assert json.loads((session.dir / SESSION_FILE).read_text())["workspace"] == str(project.resolve())
        assert Session.load(root, session.id).workspace == project.resolve()


ENV = Environment(
    name="py", version="abc", env={"PATH": "{workspace}/.venv/bin", "X": "1"},
    sandbox=SandboxRules(readable=(Path("/opt/toolchain"),), network=True),
)


class TestEnvironmentRecord:
    def test_recorded_with_workspace_unfilled(self, root):
        session = Session.create(root, "base", "p", "v", environment=ENV)
        assert json.loads((session.dir / SESSION_FILE).read_text())["environment"] == {
            "name": "py", "version": "abc", "variables": {"PATH": "{workspace}/.venv/bin", "X": "1"},
            "sandbox": {"readable": ["/opt/toolchain"], "network": True},
        }

    def test_variables_filled_in_on_read(self, root):
        session = Session.create(root, "base", "p", "v", environment=ENV)
        assert session.environment_variables == {"PATH": f"{session.workspace}/.venv/bin", "X": "1"}
        assert Session.load(root, session.id).environment_variables == session.environment_variables

    def test_sandbox_rules_survive_reload(self, root):
        session = Session.create(root, "base", "p", "v", environment=ENV)
        assert Session.load(root, session.id).sandbox_rules == SandboxRules(readable=(Path("/opt/toolchain"),), network=True)

    def test_older_sessions_keep_what_they_used_to_get(self, root):
        # Sessions recorded before sandbox rules existed must still resume with a working .venv.
        session = Session.create(root, "base", "p", "v", environment=ENV)
        meta = json.loads((session.dir / SESSION_FILE).read_text())
        del meta["environment"]["sandbox"]
        (session.dir / SESSION_FILE).write_text(json.dumps(meta))
        rules = Session.load(root, session.id).sandbox_rules
        assert rules == SandboxRules(readable=(Path.home() / ".local" / "share" / "uv",))

    def test_no_environment_no_sandbox_rules(self, session):
        assert session.sandbox_rules == SandboxRules()

    def test_no_environment_no_variables(self, session):
        assert session.environment_variables == {}
        assert session.environment is None

    def test_environment_name_and_version(self, root):
        session = Session.create(root, "base", "p", "v", environment=ENV)
        assert session.environment == {"name": "py", "version": "abc"}

    def test_session_and_environment_fill_the_same_way(self, root):
        # Setup commands use Environment.variables(), the agent's commands use the session's:
        # one shared function, so they can't disagree.
        session = Session.create(root, "base", "p", "v", environment=ENV)
        assert session.environment_variables == ENV.variables(session.workspace)


class TestAppend:
    def test_messages_cannot_be_changed_directly(self, session):
        # Appending to the list directly would update memory but not disk. Make it impossible.
        with pytest.raises(AttributeError):
            session.messages.append({"role": "user", "content": "sneaky"})
        assert len(session.messages) == 1

    def test_in_memory_and_on_disk_immediately(self, session):
        session.append({"role": "user", "content": "hi"})
        assert session.messages[-1] == {"role": "user", "content": "hi"}
        assert lines(session.dir / MESSAGES_FILE)[-1] == {"role": "user", "content": "hi"}  # no close/flush needed

    def test_one_line_per_message_even_with_newlines(self, session):
        session.append({"role": "user", "content": "line one\nline two"})
        assert len((session.dir / MESSAGES_FILE).read_text().splitlines()) == 2  # system + this

    def test_unicode_stays_readable(self, session):
        session.append({"role": "user", "content": "日本語 ✓"})
        assert "日本語 ✓" in (session.dir / MESSAGES_FILE).read_text()  # not \\u escapes

    def test_tool_call_messages_round_trip(self, session):
        session.append(assistant_calling("c1"))
        session.append(tool_result("c1", "result"))
        assert lines(session.dir / MESSAGES_FILE)[1:] == [assistant_calling("c1"), tool_result("c1", "result")]


class TestLoad:
    def test_round_trip(self, session, root):
        session.append({"role": "user", "content": "hi"})
        session.append({"role": "assistant", "content": "hello"})
        loaded = Session.load(root, session.id)
        assert loaded.messages == session.messages
        assert loaded.meta == session.meta
        assert loaded.dir == session.dir

    def test_unknown_session(self, root):
        with pytest.raises(FileNotFoundError):
            Session.load(root, "no-such-session")

    def test_loaded_session_keeps_appending_to_the_same_file(self, session, root):
        loaded = Session.load(root, session.id)
        loaded.append({"role": "user", "content": "after resume"})
        assert Session.load(root, session.id).messages[-1]["content"] == "after resume"

    def test_system_prompt_is_pinned(self, session, root):
        # Loading never re-reads the agent file: the session keeps the prompt it started with.
        assert Session.load(root, session.id).messages[0]["content"] == "You are a test agent."


class TestTurns:
    """The trace files are the only record of turns: next_turn counts them, however each ended."""

    def write_trace(self, session, turn, *events):
        session.trace_path(turn).write_text("".join(json.dumps(e) + "\n" for e in events))

    def test_trace_path(self, session):
        assert session.trace_path(1) == session.dir / "turn-001.jsonl"
        assert session.trace_path(42) == session.dir / "turn-042.jsonl"

    def test_numbering_follows_trace_files(self, session):
        self.write_trace(session, 1, {"event": "run_start"}, {"event": "run_end", "outcome": "answered"})
        assert session.next_turn == 2
        self.write_trace(session, 2, {"event": "run_start"}, {"event": "run_end", "outcome": "answered"})
        assert session.next_turn == 3

    @pytest.mark.parametrize("ending", [
        [{"event": "run_end", "outcome": "crashed"}],
        [{"event": "run_end", "outcome": "interrupted"}],
        [],  # kill -9 or power loss: no run_end was ever written
    ])
    def test_a_turn_counts_however_it_ended(self, session, ending):
        # The case the old session.json index got wrong: the turn's trace exists, so it counts.
        self.write_trace(session, 1, {"event": "run_start"}, *ending)
        assert session.next_turn == 2
        assert not session.trace_path(session.next_turn).exists()  # the next turn never collides

    def test_survives_reload(self, session, root):
        self.write_trace(session, 1, {"event": "run_start"})
        assert Session.load(root, session.id).next_turn == 2

    def test_session_json_never_changes(self, session):
        before = (session.dir / SESSION_FILE).read_text()
        session.append({"role": "user", "content": "hi"})
        self.write_trace(session, 1, {"event": "run_start"})
        assert (session.dir / SESSION_FILE).read_text() == before

    def test_other_files_are_not_turns(self, session):
        (session.dir / "notes.jsonl").write_text("")
        assert session.next_turn == 1

    def test_no_temp_file_left_behind(self, session):
        # session.json is written via a temp file + rename; the temp file never lingers.
        assert sorted(p.name for p in session.dir.iterdir()) == [MESSAGES_FILE, SESSION_FILE, "workspace"]


class TestRepairOnLoad:
    """A turn interrupted after the model asked for tools leaves calls without results.
    The API rejects that history, so load() answers each missing call with INTERRUPTED."""

    def test_complete_history_is_untouched(self, session, root):
        session.append(assistant_calling("c1"))
        session.append(tool_result("c1"))
        before = session.messages
        assert Session.load(root, session.id).messages == before

    def test_no_tool_calls_at_all(self, session, root):
        session.append({"role": "user", "content": "hi"})
        assert Session.load(root, session.id).messages == session.messages

    def test_unanswered_call_gets_interrupted_result(self, session, root):
        session.append({"role": "user", "content": "do it"})
        session.append(assistant_calling("c1"))  # Ctrl-C here, before the result was saved
        loaded = Session.load(root, session.id)
        assert loaded.messages[-1] == tool_result("c1", INTERRUPTED)

    def test_only_missing_parallel_calls_are_filled(self, session, root):
        session.append(assistant_calling("c1", "c2", "c3"))
        session.append(tool_result("c1", "real result"))  # interrupted after the first of three
        loaded = Session.load(root, session.id)
        assert loaded.messages[-3:] == (
            tool_result("c1", "real result"),
            tool_result("c2", INTERRUPTED),
            tool_result("c3", INTERRUPTED),
        )

    def test_repair_is_saved(self, session, root):
        session.append(assistant_calling("c1"))
        Session.load(root, session.id)
        assert lines(session.dir / MESSAGES_FILE)[-1] == tool_result("c1", INTERRUPTED)

    def test_repair_happens_once(self, session, root):
        session.append(assistant_calling("c1"))
        once = Session.load(root, session.id).messages
        twice = Session.load(root, session.id).messages
        assert once == twice

    def test_earlier_calls_are_checked_by_id(self, session, root):
        # Two complete tool rounds, then an interrupted third: only the third is repaired.
        for call_id in ("a", "b"):
            session.append(assistant_calling(call_id))
            session.append(tool_result(call_id))
        session.append(assistant_calling("c"))
        loaded = Session.load(root, session.id)
        assert [m["tool_call_id"] for m in loaded.messages if m["role"] == "tool"] == ["a", "b", "c"]

    def test_gap_in_the_middle_is_corruption(self, session, root):
        # Can't be fixed by appending (results must follow their call), so refuse loudly.
        session.append(assistant_calling("lost"))
        session.append({"role": "user", "content": "next"})
        session.append(assistant_calling("c2"))
        session.append(tool_result("c2"))
        with pytest.raises(ValueError, match="unanswered tool calls \\['lost'\\]"):
            Session.load(root, session.id)
