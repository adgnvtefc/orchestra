"""Tracer: a session's history, in order, one JSON object per line (JSONL).

This file *is* the trace format. The agent reports what happened (start,
model_call, tool_run, finish); the methods here decide what gets recorded.
Anything that reads traces later only needs to look here.

A trace is the ground truth of what happened in a session and in what order. It holds
only references and facts no record has (why: docs/decisions.md, 2026-10-24):

    turn_start   turn, task, agent_version, max_steps, approver, sandbox
    model_call   step, record   the chat record holds the request, response, usage
    tool_run     step, record   the tool record holds approval, result, timing
    turn_end     turn, outcome (and error, if it crashed)

Anything derivable isn't stored: token totals are sums over the records, a turn's time
is turn_end's time minus turn_start's, its answer is the last model call's response.

One file per session (sessions/<id>/trace.jsonl), appended to by each turn. One Tracer =
one turn. Whoever creates the tracer owns its lifetime with `with Tracer(path) as tracer:`,
which also guarantees the turn ends with exactly one turn_end, even if it crashes or is
interrupted.

Each line is written and flushed immediately, so a turn that crashes or is killed still
leaves its trace up to the moment it died. The turn's events are also kept in memory
(`tracer.events`), which is what tests read.
"""

import json
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from orchestra.provenance import agent_version

if TYPE_CHECKING:  # only for type hints; never imported at runtime
    from orchestra.calls import Reply

TRACE_FILE = "trace.jsonl"


def count_turns(path: Path) -> int:
    """How many turns a session's trace at `path` has started, however they ended.

    Sessions from before one trace per session (2026-10-24) have a turn-NNN.jsonl file per
    turn instead; those count too, so a resumed old session numbers its next turn right.
    """
    older = len(list(path.parent.glob("turn-*.jsonl")))
    if not path.exists():
        return older
    with path.open() as file:
        return older + sum(json.loads(line)["event"] == "turn_start" for line in file if line.strip())


class Tracer:
    def __init__(self, path: Path | None = None, listeners: list[Callable[[dict], None]] | None = None,
                 turn: int = 1):
        """Append turn `turn`'s events to `path` (parent folders are created). With no path,
        keep them in memory only.

        `listeners` are called with each event as it's recorded: how an interface shows
        what's happening (the terminal prints tool calls; a web UI could stream them),
        without the agent loop knowing any interface exists.
        """
        self.path = path
        self.listeners = listeners or []
        self.turn = turn
        self.events: list[dict] = []
        # Running totals for this turn, for summary(): derivable from the records, so not stored.
        self.steps = 0  # model calls answered so far
        self.tool_calls = 0  # tool calls made so far
        self.tokens = {"prompt": 0, "completion": 0, "cached": 0}  # summed over every model call
        self.summary: dict | None = None  # set when the turn ends: turn_end plus the totals
        self._started: float | None = None  # time.monotonic() at start(); None until the turn starts
        self._file = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            # "a": every turn of a session appends to its one trace; nothing is ever overwritten
            self._file = path.open("a")

    def event(self, kind: str, live: dict | None = None, **fields) -> None:
        """Record one event: {"event": kind, "time": ..., **fields}.

        `live`: details listeners see but the trace doesn't store, because a record already
        holds them or they can be derived (the terminal shows a tool's name and result as it
        finishes; the trace only points at the tool's record).
        """
        record = {"event": kind, "time": datetime.now().isoformat(timespec="milliseconds"), **fields}
        self.events.append(record)
        if self._file is not None:
            # default=str: anything JSON can't encode is written as its str(), never a crash
            self._file.write(json.dumps(record, default=str) + "\n")
            self._file.flush()
        for listener in self.listeners:  # after it's safely on disk
            listener({**record, **(live or {})})

    # ---- The trace format: one method per kind of event ----

    def start(self, task: str, config: dict, approver: str, sandbox: str = "none") -> None:
        """turn_start: what this turn was asked and set up with, that no record holds.

        The model, temperature, system prompt, tools, and code version are in every model
        call's record. What only the turn knows: the task as given, the agent definition's
        version (a hash of the agent file) and its max_steps, and who approved tools
        (approver) inside what (sandbox): sandboxed and unsandboxed runs aren't comparable data.
        """
        self._started = time.monotonic()
        self.steps, self.tool_calls = 0, 0
        self.tokens = {"prompt": 0, "completion": 0, "cached": 0}
        self.summary = None
        self.event("turn_start", turn=self.turn, task=task, agent_version=agent_version(config),
                   max_steps=config["max_steps"], approver=approver, sandbox=sandbox)

    def model_call(self, step: int, reply: "Reply") -> None:
        """A model call finished. chat() already recorded all of it (request, response,
        reasoning, usage) in the app's log; the trace just says which record it was."""
        self.steps = step
        # Every call re-sends the whole conversation, so prompt tokens add up fast; cached
        # ones are those the server didn't have to process again.
        for kind in self.tokens:
            self.tokens[kind] += reply.usage[kind]
        self.event("model_call", step=step, record=reply.id)

    def tool_run(self, step: int, record_id: str, call: dict, result: str) -> None:
        """A tool ran. Its record (written by the agent loop) holds the approval, result, and
        timing; the model call that asked for it holds the name and arguments. The trace just
        points at it."""
        self.tool_calls += 1
        name, arguments = call["function"]["name"], call["function"]["arguments"]
        self.event("tool_run", step=step, record=record_id, live={"name": name, "arguments": arguments, "result": result})

    def finish(self, outcome: str) -> None:
        """turn_end for a turn that ended normally: outcome is "answered" or "max_steps"."""
        self._end(outcome=outcome)

    def _end(self, **fields) -> None:
        """Write the one turn_end. Listeners (and summary) also get the turn's totals: model
        calls (steps), tool calls, tokens, and time, which the trace doesn't store."""
        seconds = round(time.monotonic() - self._started, 3) if self._started is not None else None
        totals = {"steps": self.steps, "tool_calls": self.tool_calls, "tokens": dict(self.tokens), "seconds": seconds}
        self.event("turn_end", live=totals, turn=self.turn, **fields)
        self.summary = {**self.events[-1], **totals}

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    # `with Tracer(path) as tracer:` runs __enter__ at the start of the block and __exit__
    # at the end, *however* the block ends: normally, by return, by exception, or by Ctrl-C.
    def __enter__(self) -> "Tracer":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        # A turn that started but never reached finish(): record how it ended instead.
        if self._started is not None and self.summary is None:
            if exc_type is None:
                self._end(outcome="unfinished")
            elif issubclass(exc_type, KeyboardInterrupt):  # Ctrl-C isn't an Exception subclass
                self._end(outcome="interrupted")
            else:
                self._end(outcome="crashed", error=f"{exc_type.__name__}: {exc}")
        self.close()
        # returning None (falsy) means: don't swallow the exception, let it propagate
