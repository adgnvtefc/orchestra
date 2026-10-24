"""Tracer: JSONL events, written and flushed one line at a time."""

import json
from pathlib import Path

import pytest

from orchestra.tracer import Tracer


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

    def test_never_overwrites_an_existing_trace(self, tmp_path):
        path = tmp_path / "t.jsonl"
        path.write_text("precious\n")
        with pytest.raises(FileExistsError):
            Tracer(path)
        assert path.read_text() == "precious\n"

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


class TestRunLifecycle:
    """start() writes run_start; exactly one run_end follows, from finish() or from __exit__."""

    def kinds(self, tracer):
        return [e["event"] for e in tracer.events]

    def test_start_fields(self):
        tracer = Tracer()
        tracer.start("task", {"model": "m"}, [{"t": 1}], "deny_all")
        start = tracer.events[0]
        assert (start["event"], start["task"], start["config"], start["tools"], start["approver"]) == (
            "run_start", "task", {"model": "m"}, [{"t": 1}], "deny_all",
        )

    def test_start_records_the_sandbox(self):
        tracer = Tracer()
        tracer.start("t", {}, [], "a", sandbox="seatbelt")
        tracer.start("t", {}, [], "a")
        assert [e["sandbox"] for e in tracer.events] == ["seatbelt", "none"]

    def test_start_records_provenance(self):
        tracer = Tracer()
        tracer.start("t", {"model": "m", "system_prompt": "p"}, [], "a")
        record = tracer.events[0]["provenance"]
        assert set(record) == {"agent_version", "orchestra"}

    def test_finish_writes_run_end_immediately(self):
        tracer = Tracer()
        tracer.start("t", {}, [], "a")
        tracer.finish("max_steps", "gave up")
        end = tracer.events[-1]
        assert (end["event"], end["outcome"], end["answer"]) == ("run_end", "max_steps", "gave up")
        assert end["seconds"] >= 0

    def test_steps_counts_model_responses(self):
        tracer = Tracer()
        tracer.start("t", {}, [], "a")
        tracer.steps = 3  # model_response() sets this; tested end to end in test_agent.py
        tracer.finish("answered", "x")
        assert tracer.events[-1]["steps"] == 3

    def test_exit_after_finish_adds_nothing(self):
        with Tracer() as tracer:
            tracer.start("t", {}, [], "a")
            tracer.finish("answered", "x")
        assert self.kinds(tracer) == ["run_start", "run_end"]

    def test_exit_without_finish(self):
        with Tracer() as tracer:
            tracer.start("t", {}, [], "a")
        assert tracer.events[-1]["outcome"] == "unfinished"

    def test_crash_recorded_and_not_swallowed(self):
        with pytest.raises(ValueError, match="bad"):
            with Tracer() as tracer:
                tracer.start("t", {}, [], "a")
                raise ValueError("bad")
        end = tracer.events[-1]
        assert (end["outcome"], end["error"]) == ("crashed", "ValueError: bad")
        assert "answer" not in end

    def test_interrupt_recorded_and_not_swallowed(self):
        with pytest.raises(KeyboardInterrupt):
            with Tracer() as tracer:
                tracer.start("t", {}, [], "a")
                raise KeyboardInterrupt
        assert tracer.events[-1]["outcome"] == "interrupted"

    def test_crash_after_finish_keeps_the_real_outcome(self):
        # The run itself finished; something after it failed. Still exactly one run_end.
        with pytest.raises(RuntimeError):
            with Tracer() as tracer:
                tracer.start("t", {}, [], "a")
                tracer.finish("answered", "x")
                raise RuntimeError
        assert self.kinds(tracer) == ["run_start", "run_end"]
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
                tracer.start("t", {}, [], "a")
                raise ValueError("bad")
        assert read_lines(path)[-1]["outcome"] == "crashed"
