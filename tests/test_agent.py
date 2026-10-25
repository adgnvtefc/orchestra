"""The agent loop, driven by a scripted FakeClient instead of a real model."""

import pytest

import json

from orchestra.sandbox import Unsandboxed
from orchestra.agent import run_agent
from orchestra.session import MESSAGES_FILE, Session
from orchestra.tools import ToolContext, Toolset
from orchestra.tracer import Tracer
from tests.fakes import NO_HUMAN, FakeClient, always, call, reply
from tests.test_records import read_log
from orchestra.provenance import agent_version

CONFIG = {"model": "test-model", "temperature": 0.3, "max_steps": 5, "system_prompt": "You are a test agent."}

_sessions_root = None  # set per test by _setup below


def new_session(system_prompt=CONFIG["system_prompt"]) -> Session:
    return Session.create(_sessions_root, "test", system_prompt, "v", app="orchestra-test")


def run(replies, task="do the thing", tools=("echo",), tracer=None, session=None, **config_overrides):
    """Run one turn against scripted replies. Returns (final answer, client) for inspection.

    Pass your own in-memory `tracer` or `session` to inspect them afterwards; by default
    each call gets a fresh session.
    """
    client = FakeClient(replies)
    answer = run_agent(
        task,
        session or new_session(),
        {**CONFIG, **config_overrides},
        client,
        Toolset(list(tools), NO_HUMAN, ToolContext(workspace=_sessions_root.parent / "workspace", sandbox=Unsandboxed())),
        tracer or Tracer(),
    )
    return answer, client


@pytest.fixture(autouse=True)
def _setup(fake_tools, tmp_path):
    """Every test here may use the echo/boom/count test tools, and gets its own sessions folder."""
    global _sessions_root
    _sessions_root = tmp_path / "sessions"


class TestFinishing:
    def test_answer_without_tools_returns_immediately(self):
        answer, client = run([reply("done")])
        assert answer == "done"
        assert len(client.requests) == 1

    def test_none_content_becomes_empty_string(self):
        answer, _ = run([reply(None)])
        assert answer == ""

    def test_tool_then_answer(self):
        answer, client = run([reply(tool_calls=[call("echo", {"text": "hi"})]), reply("final")])
        assert answer == "final"
        assert len(client.requests) == 2

    def test_stops_at_max_steps(self):
        answer, client = run(always(lambda: reply(tool_calls=[call("echo", {"text": "again"})])), max_steps=3)
        assert answer == "Stopped after 3 steps without a final answer."
        assert len(client.requests) == 3  # exactly max_steps model calls, no more

    def test_max_steps_one(self):
        answer, client = run(always(lambda: reply(tool_calls=[call("echo", {"text": "x"})])), max_steps=1)
        assert answer.startswith("Stopped after 1 steps")
        assert len(client.requests) == 1


class TestRequests:
    def test_first_request_is_system_then_task(self):
        _, client = run([reply("ok")], task="list files")
        assert client.requests[0]["messages"] == [
            {"role": "system", "content": "You are a test agent."},
            {"role": "user", "content": "list files"},
        ]

    def test_model_and_temperature_come_from_config(self):
        _, client = run([reply("ok")], model="some-model", temperature=0.9)
        assert client.requests[0]["model"] == "some-model"
        assert client.requests[0]["temperature"] == 0.9

    def test_sends_only_the_toolsets_schemas(self):
        _, client = run([reply("ok")], tools=("echo", "count"))
        assert [t["function"]["name"] for t in client.requests[0]["tools"]] == ["echo", "count"]

    def test_omits_tools_key_when_toolset_empty(self):
        # Some servers reject "tools": [], so the key must be absent, not empty.
        _, client = run([reply("ok")], tools=())
        assert "tools" not in client.requests[0]

    def test_tools_identical_every_turn(self):
        _, client = run([reply(tool_calls=[call("echo", {"text": "a"})]), reply("ok")])
        assert client.requests[0]["tools"] == client.requests[1]["tools"]

    def test_history_is_append_only(self):
        # Each request extends the previous one without editing it: that's what lets the
        # server reuse its prompt cache instead of re-reading the whole conversation.
        _, client = run(
            [
                reply(tool_calls=[call("echo", {"text": "1"})]),
                reply(tool_calls=[call("echo", {"text": "2"})]),
                reply("done"),
            ]
        )
        sent = [r["messages"] for r in client.requests]
        for earlier, later in zip(sent, sent[1:]):
            assert later[: len(earlier)] == earlier
            assert len(later) > len(earlier)


class TestToolRoundTrip:
    def test_assistant_turn_then_tool_result(self):
        tc = call("echo", {"text": "hi"}, id="call_abc")
        _, client = run([reply(tool_calls=[tc]), reply("done")])
        history = client.requests[1]["messages"]

        assistant, tool_result = history[2], history[3]
        assert assistant["role"] == "assistant"
        assert assistant["tool_calls"][0]["id"] == "call_abc"
        assert assistant["tool_calls"][0]["function"] == {"name": "echo", "arguments": '{"text": "hi"}'}
        assert tool_result == {"role": "tool", "tool_call_id": "call_abc", "content": "hi"}

    def test_every_tool_call_gets_exactly_one_reply_in_order(self):
        calls = [call("echo", {"text": t}, id=f"id_{t}") for t in ("a", "b", "c")]
        _, client = run([reply(tool_calls=calls), reply("done")])
        results = [m for m in client.requests[1]["messages"] if m["role"] == "tool"]
        assert [(m["tool_call_id"], m["content"]) for m in results] == [("id_a", "a"), ("id_b", "b"), ("id_c", "c")]

    def test_assistant_message_has_no_null_fields(self):
        # A tool-only turn has content=None; some servers reject "content": null next to tool_calls.
        _, client = run([reply(None, tool_calls=[call("echo", {"text": "x"})]), reply("done")])
        assistant = client.requests[1]["messages"][2]
        assert "content" not in assistant
        assert None not in assistant.values()

    def test_reasoning_is_dropped_from_history(self):
        _, client = run([reply(tool_calls=[call("echo", {"text": "x"})], reasoning="secret thoughts"), reply("done")])
        assert "reasoning_content" not in client.requests[1]["messages"][2]
        assert "secret thoughts" not in str(client.requests[1]["messages"])

    def test_final_text_alongside_reasoning(self):
        answer, _ = run([reply("the answer", reasoning="thinking...")])
        assert answer == "the answer"


class TestModelMistakesDoNotCrashTheLoop:
    """Small models get tool calls wrong. Each mistake must come back as text the model can fix."""

    def last_tool_result(self, client):
        return [m for m in client.requests[-1]["messages"] if m["role"] == "tool"][-1]["content"]

    def test_invented_tool(self):
        _, client = run([reply(tool_calls=[call("hack_the_planet", {})]), reply("sorry")])
        assert self.last_tool_result(client).startswith("Error: unknown tool 'hack_the_planet'")

    def test_tool_not_in_this_agents_toolset(self, approve):
        _, client = run([reply(tool_calls=[call("bash", {"command": "rm -rf /"})]), reply("ok")], tools=("echo",))
        assert self.last_tool_result(client).startswith("Error: unknown tool 'bash'")
        assert approve == []  # never even asked the human

    def test_malformed_json_arguments(self):
        _, client = run([reply(tool_calls=[call("echo", "{text: oops")]), reply("ok")])
        assert "not valid JSON" in self.last_tool_result(client)

    def test_tool_raises(self):
        answer, client = run([reply(tool_calls=[call("boom")]), reply("recovered")], tools=("boom",))
        assert self.last_tool_result(client) == "Error: RuntimeError: kaboom"
        assert answer == "recovered"


class TestNoOutput:
    def test_the_loop_prints_nothing(self, capsys):
        # Showing progress is the interface's job (it listens to the tracer's events).
        run([reply(tool_calls=[call("echo", {"text": "hello"})]), reply("done")])
        assert capsys.readouterr() == ("", "")

    def test_listeners_see_every_event_as_it_happens(self):
        seen = []
        run([reply(tool_calls=[call("echo", {"text": "hello"})]), reply("done")], tracer=Tracer(listeners=[seen.append]))
        assert [e["event"] for e in seen] == ["turn_start", "model_call", "tool_run", "model_call", "turn_end"]
        assert seen[2]["result"] == "hello"


class TestTracing:
    """What run_agent records. Each run must produce: run_start, then per step a model_call
    and one tool_call per call, then exactly one run_end with the outcome."""

    def kinds(self, tracer):
        return [e["event"] for e in tracer.events]

    def test_answer_only(self):
        tracer = Tracer()
        run([reply("done")], tracer=tracer)
        assert self.kinds(tracer) == ["turn_start", "model_call", "turn_end"]

    def test_tool_round_trip(self):
        tracer = Tracer()
        run([reply(tool_calls=[call("echo", {"text": "a"}), call("echo", {"text": "b"})]), reply("done")], tracer=tracer)
        assert self.kinds(tracer) == ["turn_start", "model_call", "tool_run", "tool_run", "model_call", "turn_end"]

    def test_turn_start_records_the_setup(self, orchestra_dir):
        tracer = Tracer()
        run([reply("ok")], task="list files", tools=("echo", "count"), tracer=tracer)
        start = tracer.events[0]
        assert start["task"] == "list files"
        assert start["agent_version"] == agent_version(CONFIG)
        assert start["max_steps"] == 5
        assert start["approver"] == "deny_all"  # NO_HUMAN's approver: tells you if this was an unattended run
        assert start["sandbox"] == "none"
        # the model, system prompt, and tools are in the model call's record, not copied here
        (record,) = read_log(orchestra_dir, "orchestra-test")
        assert record["request"]["model"] == "test-model"
        assert record["request"]["messages"][0] == {"role": "system", "content": "You are a test agent."}
        assert [t["function"]["name"] for t in record["request"]["tools"]] == ["echo", "count"]

    def test_model_call_points_at_its_record(self, orchestra_dir):
        # chat() records the whole call in the session's app's log; the trace only says which record.
        tracer, session = Tracer(), new_session()
        usage = {"prompt_tokens": 120, "completion_tokens": 8, "total_tokens": 128}
        run([reply("hi", reasoning="let me think", usage=usage)], tracer=tracer, session=session)
        event = tracer.events[1]
        assert set(event) == {"event", "time", "step", "record"}
        (record,) = read_log(orchestra_dir, "orchestra-test")
        assert event["record"] == record["id"]
        assert (record["session"], record["app"]) == (session.id, session.app)
        assert record["request"]["model"] == "test-model"
        assert record["response"]["content"] == "hi"
        assert record["response"]["reasoning"] == "let me think"  # kept here, dropped from the conversation
        assert record["usage"]["prompt"] == 120

    def test_tokens_add_up_over_the_turn(self):
        tracer = Tracer()
        usage = {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105, "prompt_tokens_details": {"cached_tokens": 60}}
        run([reply(tool_calls=[call("echo", {"text": "a"})], usage=usage), reply("done", usage=usage)], tracer=tracer)
        assert tracer.summary["tokens"] == {"prompt": 200, "completion": 10, "cached": 120}

    def test_tool_run_points_at_its_record(self, orchestra_dir):
        tracer, session = Tracer(), new_session()
        run([reply(tool_calls=[call("echo", {"text": "hey"}, id="call_9")]), reply("done")], tracer=tracer, session=session)
        event = tracer.events[2]
        assert set(event) == {"event", "time", "step", "record"}  # a reference, no copies
        asked, tool, answered = read_log(orchestra_dir, "orchestra-test")
        assert [r["kind"] for r in (asked, tool, answered)] == ["chat", "tool", "chat"]
        assert event["record"] == tool["id"]
        assert (tool["parent"], tool["session"], tool["tool_call_id"]) == (asked["id"], session.id, "call_9")
        assert tool["result"] == "hey"
        assert tool["seconds"] >= 0
        # the name and arguments live in the model call that asked for the tool, not in the tool record
        (wanted,) = asked["response"]["tool_calls"]
        assert (wanted["id"], wanted["function"]["name"]) == ("call_9", "echo")
        assert "arguments" not in tool and "name" not in tool

    def test_listeners_get_tool_details_the_trace_doesnt_store(self):
        seen = []
        tracer = Tracer(listeners=[seen.append])
        run([reply(tool_calls=[call("echo", {"text": "hey"})]), reply("done")], tracer=tracer)
        shown, stored = seen[2], tracer.events[2]
        assert (shown["name"], shown["arguments"], shown["result"]) == ("echo", '{"text": "hey"}', "hey")
        assert "result" not in stored
        assert shown["record"] == stored["record"]

    def test_tool_records_say_who_approved(self, orchestra_dir):
        # NO_HUMAN: "allow" tools run (auto-allow), "ask" tools are refused (auto-deny)
        run([reply(tool_calls=[call("echo", {"text": "a"}), call("guarded", {"target": "x"})]), reply("ok")],
            tools=("echo", "guarded"))
        _, echo, guarded, _ = read_log(orchestra_dir, "orchestra-test")
        assert (echo["approval"], guarded["approval"]) == ("auto-allow", "auto-deny")

    def test_failed_tool_calls_are_recorded_too(self, orchestra_dir):
        run([reply(tool_calls=[call("echo", "{broken")]), reply("ok")])
        asked, tool, _ = read_log(orchestra_dir, "orchestra-test")
        assert asked["response"]["tool_calls"][0]["function"]["arguments"] == "{broken"  # exactly as the model wrote it
        assert "not valid JSON" in tool["result"]
        assert tool["approval"] is None  # rejected before anyone was asked

    def test_run_end_answered(self):
        tracer = Tracer()
        run([reply(tool_calls=[call("echo", {"text": "a"})]), reply("final")], tracer=tracer)
        end = tracer.events[-1]
        assert (end["event"], end["outcome"]) == ("turn_end", "answered")
        assert "answer" not in end  # it's the last model call's response
        assert (tracer.summary["steps"], tracer.summary["seconds"] >= 0) == (2, True)

    def test_run_end_max_steps(self):
        tracer = Tracer()
        run(always(lambda: reply(tool_calls=[call("echo", {"text": "x"})])), max_steps=2, tracer=tracer)
        assert (tracer.events[-1]["outcome"], tracer.summary["steps"]) == ("max_steps", 2)

    def test_crash_is_traced_then_reraised(self):
        # The model call blows up on step 2: the trace still ends with what happened.
        # The tracer's owner wraps it in `with`; that's what records the crash.
        with pytest.raises(AssertionError, match="more times than scripted"):
            with Tracer() as tracer:
                run([reply(tool_calls=[call("echo", {"text": "a"})])], tracer=tracer)
        end = tracer.events[-1]
        assert end["event"] == "turn_end"
        assert end["outcome"] == "crashed"
        assert end["error"].startswith("AssertionError: model called more times")
        # steps = model responses actually received: step 1 completed, step 2 crashed before replying
        assert tracer.summary["steps"] == 1
        assert sum(e["event"] == "model_call" for e in tracer.events) == tracer.summary["steps"]

    def test_interrupt_is_traced_then_reraised(self):
        def interrupting():
            raise KeyboardInterrupt
            yield  # makes this a generator, like the other reply streams

        with pytest.raises(KeyboardInterrupt):
            with Tracer() as tracer:
                run(interrupting(), tracer=tracer)
        assert tracer.events[-1]["outcome"] == "interrupted"


class TestSession:
    """The session owns the conversation: the model sees exactly what the session holds."""

    def saved(self, session):
        return [json.loads(line) for line in (session.dir / MESSAGES_FILE).read_text().splitlines()]

    def test_saved_history_is_exactly_what_the_model_saw(self):
        # Memory and disk can't drift: the last request plus the final answer is the whole file.
        session = new_session()
        run([reply(tool_calls=[call("echo", {"text": "a"})]), reply("final")], session=session)
        assert self.saved(session) == list(session.messages)
        assert session.messages[-1] == {"role": "assistant", "content": "final"}

    def test_last_request_matches_saved_history(self):
        session = new_session()
        _, client = run([reply(tool_calls=[call("echo", {"text": "a"})]), reply("final")], session=session)
        assert client.requests[-1]["messages"] + [{"role": "assistant", "content": "final"}] == self.saved(session)

    def test_second_turn_continues_the_conversation(self):
        session = new_session()
        run([reply("first answer")], task="first task", session=session)
        _, client = run([reply("second answer")], task="second task", session=session)
        assert client.requests[0]["messages"] == [
            {"role": "system", "content": "You are a test agent."},
            {"role": "user", "content": "first task"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second task"},
        ]

    def test_system_prompt_comes_from_the_session_not_the_config(self):
        # The session pinned its prompt at creation; editing the agent file later doesn't change it.
        session = new_session(system_prompt="pinned prompt")
        _, client = run([reply("ok")], session=session, system_prompt="edited prompt")
        assert client.requests[0]["messages"][0] == {"role": "system", "content": "pinned prompt"}

    def test_crash_keeps_everything_up_to_the_crash(self):
        session = new_session()
        with pytest.raises(AssertionError):
            run([reply(tool_calls=[call("echo", {"text": "a"})])], session=session)  # no reply for step 2
        roles = [m["role"] for m in self.saved(session)]
        assert roles == ["system", "user", "assistant", "tool"]


class TestTotals:
    """The turn's totals: given live and in tracer.summary, not stored (they're sums over the records)."""

    def usage(self, prompt, completion, cached=0):
        return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion,
                "prompt_tokens_details": {"cached_tokens": cached}}

    def test_counts_and_sums(self):
        tracer = Tracer()
        run(
            [
                reply(tool_calls=[call("echo", {"text": "a"}), call("echo", {"text": "b"})], usage=self.usage(100, 10)),
                reply(tool_calls=[call("echo", {"text": "c"})], usage=self.usage(150, 5, cached=100)),
                reply("done", usage=self.usage(180, 20, cached=150)),
            ],
            tracer=tracer,
        )
        end = tracer.summary
        assert end["steps"] == 3  # model calls
        assert end["tool_calls"] == 3  # two in one response, then one
        assert end["tokens"] == {"prompt": 430, "completion": 35, "cached": 250}

    def test_no_usage_reported(self):
        tracer = Tracer()
        run([reply("done")], tracer=tracer)
        assert tracer.summary["tokens"] == {"prompt": 0, "completion": 0, "cached": 0}

    def test_totals_reset_per_run(self):
        tracer = Tracer()
        run([reply(tool_calls=[call("echo", {"text": "a"})], usage=self.usage(10, 1)), reply("x")], tracer=tracer)
        run([reply("y", usage=self.usage(5, 1))], tracer=tracer)
        end = tracer.summary
        assert (end["steps"], end["tool_calls"], end["tokens"]["prompt"]) == (1, 0, 5)
