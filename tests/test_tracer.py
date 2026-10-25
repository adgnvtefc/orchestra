"""Tracer: a session's trace, JSONL events written and flushed one line at a time."""

import json
from pathlib import Path

import pytest

from orchestra.provenance import agent_version
from orchestra.tracer import Tracer, count_turns


def read_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


class TestInMemory:
    def test_records_events(self):
        tracer = Tracer()
        tracer.event("a", x=1)
        tracer.event("b", y="two")
        assert [(e["event"], e.get("x"), e.get("y")) for e in tracer.events] == [("a", 1, None), ("b", None, "two")]

    def test_every_event_has_a_timestamp(self):
        tracer = Tracer()
        tracer.event("a")
        assert "T" in tracer.events[0]["time"]  # ISO 8601, e.g. 2026-10-08T14:03:22.123

    def test_no_file(self):
        tracer = Tracer()
        tracer.event("a")
        tracer.close()  # closing with no file is fine
        assert tracer.path is None


class TestFile:
    def test_one_json_object_per_line(self, tmp_path):
        path = tmp_path / "t.jsonl"
        with Tracer(path) as tracer:
            tracer.event("a", n=1)
            tracer.event("b", n=2)
        assert [(e["event"], e["n"]) for e in read_lines(path)] == [("a", 1), ("b", 2)]

    def test_file_matches_memory(self, tmp_path):
        path = tmp_path / "t.jsonl"
        with Tracer(path) as tracer:
            tracer.event("a", nested={"k": [1, 2]})
        assert read_lines(path) == tracer.events

    def test_each_line_is_on_disk_immediately(self, tmp_path):
        # If the process dies mid-run, everything recorded so far must already be in the file.
        path = tmp_path / "t.jsonl"
        tracer = Tracer(path)
        tracer.event("a")
        assert len(read_lines(path)) == 1  # readable before close()
        tracer.close()

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "deep" / "er" / "t.jsonl"
        with Tracer(path) as tracer:
            tracer.event("a")
        assert path.is_file()

    def test_appends_to_an_existing_trace(self, tmp_path):
        # every turn of a session appends to its one trace; nothing before is ever lost
        path = tmp_path / "t.jsonl"
        path.write_text('{"event": "earlier"}\n')
        with Tracer(path) as tracer:
            tracer.event("later")
        assert [e["event"] for e in read_lines(path)] == ["earlier", "later"]

    def test_unencodable_values_become_strings(self, tmp_path):
        path = tmp_path / "t.jsonl"
        with Tracer(path) as tracer:
            tracer.event("a", where=Path("/x/y"), things={1, 2} - {2})
        line = read_lines(path)[0]
        assert line["where"] == "/x/y"
        assert line["things"] == "{1}"

    def test_unicode_and_newlines_stay_on_one_line(self, tmp_path):
        path = tmp_path / "t.jsonl"
        with Tracer(path) as tracer:
            tracer.event("a", text="line one\nline two ✓")
        assert len(path.read_text().splitlines()) == 1
        assert read_lines(path)[0]["text"] == "line one\nline two ✓"

    def test_closes_even_when_the_run_raises(self, tmp_path):
        path = tmp_path / "t.jsonl"
        with pytest.raises(RuntimeError):
            with Tracer(path) as tracer:
                tracer.event("a")
                raise RuntimeError("boom")
        assert tracer._file is None
        assert len(read_lines(path)) == 1


class TestListeners:
    def test_called_with_each_event_in_order(self):
        seen = []
        tracer = Tracer(listeners=[seen.append])
        tracer.event("a", n=1)
        tracer.event("b", n=2)
        assert seen == tracer.events

    def test_several_listeners(self):
        first, second = [], []
        Tracer(listeners=[first.append, second.append]).event("a")
        assert len(first) == len(second) == 1

    def test_called_after_the_line_is_on_disk(self, tmp_path):
        path = tmp_path / "t.jsonl"
        lines_when_called = []
        tracer = Tracer(path, listeners=[lambda e: lines_when_called.append(len(read_lines(path)))])
        tracer.event("a")
        tracer.close()
        assert lines_when_called == [1]  # a listener that crashes can't lose the event

    def test_no_listeners_by_default(self):
        assert Tracer().listeners == []


CONFIG = {"name": "base", "model": "m", "temperature": 0.5, "max_steps": 7, "system_prompt": "p"}


class TestTurnLifecycle:
    """start() writes turn_start; exactly one turn_end follows, from finish() or from __exit__."""

    def kinds(self, tracer):
        return [e["event"] for e in tracer.events]

    def test_start_holds_only_what_no_record_has(self):
        tracer = Tracer(turn=3)
        tracer.start("task", CONFIG, "deny_all", sandbox="seatbelt")
        start = tracer.events[0]
        assert set(start) == {"event", "time", "turn", "task", "agent_version", "max_steps", "approver", "sandbox"}
        assert (start["event"], start["turn"], start["task"], start["max_steps"]) == ("turn_start", 3, "task", 7)
        assert start["agent_version"] == agent_version(CONFIG)
        assert (start["approver"], start["sandbox"]) == ("deny_all", "seatbelt")

    def test_sandbox_defaults_to_none(self):
        tracer = Tracer()
        tracer.start("t", CONFIG, "a")
        assert tracer.events[0]["sandbox"] == "none"

    def test_finish_writes_turn_end_immediately(self):
        tracer = Tracer(turn=2)
        tracer.start("t", CONFIG, "a")
        tracer.finish("max_steps")
        assert tracer.events[-1] == {"event": "turn_end", "time": tracer.events[-1]["time"], "turn": 2, "outcome": "max_steps"}

    def test_totals_are_given_live_not_stored(self, tmp_path):
        # derivable from the records, so the trace doesn't store them; listeners and summary get them
        seen = []
        with Tracer(tmp_path / "t.jsonl", listeners=[seen.append]) as tracer:
            tracer.start("t", CONFIG, "a")
            tracer.steps = 3  # model_call() sets this; tested end to end in test_agent.py
            tracer.finish("answered")
        stored = read_lines(tmp_path / "t.jsonl")[-1]
        assert "steps" not in stored and "tokens" not in stored
        assert seen[-1]["steps"] == tracer.summary["steps"] == 3
        assert tracer.summary["outcome"] == "answered"
        assert tracer.summary["seconds"] >= 0

    def test_no_summary_until_the_turn_ends(self):
        tracer = Tracer()
        tracer.start("t", CONFIG, "a")
        assert tracer.summary is None

    def test_exit_after_finish_adds_nothing(self):
        with Tracer() as tracer:
            tracer.start("t", CONFIG, "a")
            tracer.finish("answered")
        assert self.kinds(tracer) == ["turn_start", "turn_end"]

    def test_exit_without_finish(self):
        with Tracer() as tracer:
            tracer.start("t", CONFIG, "a")
        assert tracer.events[-1]["outcome"] == "unfinished"

    def test_crash_recorded_and_not_swallowed(self):
        with pytest.raises(ValueError, match="bad"):
            with Tracer() as tracer:
                tracer.start("t", CONFIG, "a")
                raise ValueError("bad")
        end = tracer.events[-1]
        assert (end["outcome"], end["error"]) == ("crashed", "ValueError: bad")

    def test_interrupt_recorded_and_not_swallowed(self):
        with pytest.raises(KeyboardInterrupt):
            with Tracer() as tracer:
                tracer.start("t", CONFIG, "a")
                raise KeyboardInterrupt
        assert tracer.events[-1]["outcome"] == "interrupted"

    def test_crash_after_finish_keeps_the_real_outcome(self):
        # The turn itself finished; something after it failed. Still exactly one turn_end.
        with pytest.raises(RuntimeError):
            with Tracer() as tracer:
                tracer.start("t", CONFIG, "a")
                tracer.finish("answered")
                raise RuntimeError
        assert self.kinds(tracer) == ["turn_start", "turn_end"]
        assert tracer.events[-1]["outcome"] == "answered"

    def test_exit_before_start_writes_nothing(self):
        # e.g. setup failed before the run began: there's no run to end
        with pytest.raises(RuntimeError):
            with Tracer() as tracer:
                raise RuntimeError
        assert tracer.events == []

    def test_crash_end_reaches_the_file(self, tmp_path):
        path = tmp_path / "t.jsonl"
        with pytest.raises(ValueError):
            with Tracer(path) as tracer:
                tracer.start("t", CONFIG, "a")
                raise ValueError("bad")
        assert read_lines(path)[-1]["outcome"] == "crashed"


class TestCountTurns:
    def test_no_trace_yet(self, tmp_path):
        assert count_turns(tmp_path / "trace.jsonl") == 0

    def test_counts_turn_starts_however_they_ended(self, tmp_path):
        path = tmp_path / "trace.jsonl"
        for turn, ending in ((1, "finish"), (2, "crash"), (3, "none")):
            try:
                with Tracer(path, turn=turn) as tracer:
                    tracer.start("t", CONFIG, "a")
                    if ending == "finish":
                        tracer.finish("answered")
                    elif ending == "crash":
                        raise ValueError
            except ValueError:
                pass
        assert count_turns(path) == 3

    def test_older_sessions_turn_files_count_too(self, tmp_path):
        # sessions from before one trace per session had turn-NNN.jsonl files
        (tmp_path / "turn-001.jsonl").write_text("{}\n")
        (tmp_path / "turn-002.jsonl").write_text("{}\n")
        with Tracer(tmp_path / "trace.jsonl", turn=3) as tracer:
            tracer.start("t", CONFIG, "a")
        assert count_turns(tmp_path / "trace.jsonl") == 3
