"""Tracer: records what happened in a run, one JSON object per line (JSONL).

This file *is* the trace format. The agent reports what happened (start,
model_response, tool_call, finish); the methods here decide what gets recorded.
Anything that reads traces later only needs to look here.

One Tracer = one run = one file. Whoever creates the tracer owns its lifetime with
`with Tracer(path) as tracer:`, which also guarantees the trace ends with exactly
one run_end, even if the run crashes or is interrupted.

Each line is written and flushed immediately, so a run that crashes or is killed
still leaves a trace up to the moment it died. Events are also kept in memory
(`tracer.events`), which is what tests and end-of-run summaries read.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from harness import provenance

if TYPE_CHECKING:  # only for type hints; never imported at runtime
    from openai.types.chat import ChatCompletion, ChatCompletionMessageToolCall


class Tracer:
    def __init__(self, path: Path | None = None):
        """Write events to `path` (parent folders are created). With no path, keep them in memory only."""
        self.path = path
        self.events: list[dict] = []
        self.steps = 0  # model responses received so far; run_end reports it
        self.tool_calls = 0  # tool calls made so far; run_end reports it
        self.tokens = {"prompt": 0, "completion": 0, "cached": 0}  # summed over every model call
        self._started: float | None = None  # time.monotonic() at start(); None until the run starts
        self._ended = False  # True once run_end is written; there is only ever one
        self._file = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            # "x": create a new file, never append to or overwrite an existing trace
            self._file = path.open("x")

    def event(self, kind: str, **fields) -> None:
        """Record one event: {"event": kind, "time": ..., **fields}."""
        record = {"event": kind, "time": datetime.now().isoformat(timespec="milliseconds"), **fields}
        self.events.append(record)
        if self._file is not None:
            # default=str: anything JSON can't encode is written as its str(), never a crash
            self._file.write(json.dumps(record, default=str) + "\n")
            self._file.flush()

    # ---- The trace format: one method per kind of event ----

    def start(self, task: str, config: dict, tools: list[dict], approver: str) -> None:
        """run_start: everything needed to know what this run was set up to do, and what produced it."""
        self._started = time.monotonic()
        self.steps, self.tool_calls = 0, 0
        self.tokens = {"prompt": 0, "completion": 0, "cached": 0}
        self.event(
            "run_start",
            task=task,
            config=config,
            tools=tools,
            approver=approver,
            provenance=provenance.collect(config),
        )

    def model_response(self, step: int, response: "ChatCompletion", seconds: float) -> None:
        self.steps = step
        if response.usage:
            # Every call re-sends the whole conversation, so prompt tokens add up fast; cached
            # ones are those the server didn't have to process again.
            details = response.usage.prompt_tokens_details
            self.tokens["prompt"] += response.usage.prompt_tokens or 0
            self.tokens["completion"] += response.usage.completion_tokens or 0
            self.tokens["cached"] += (details.cached_tokens or 0) if details else 0
        choice = response.choices[0]
        msg = choice.message
        # creates the event with the info extracted from message
        self.event(
            "model_response",
            step=step,
            message=msg.model_dump(exclude_none=True, exclude={"reasoning_content"}),
            # dropped from the conversation history, but kept here: it's *why* the model did what it did
            reasoning=getattr(msg, "reasoning_content", None),
            finish_reason=choice.finish_reason,
            model=response.model,  # what the server says answered; compare with run_start.config.model
            usage=response.usage.model_dump() if response.usage else None,
            seconds=seconds,
        )

    def tool_call(self, step: int, call: "ChatCompletionMessageToolCall", result: str, seconds: float) -> None:
        self.tool_calls += 1
        # creates event from tool call
        self.event(
            "tool_call",
            step=step,
            id=call.id,
            name=call.function.name,
            arguments=call.function.arguments,  # the raw string the model wrote, broken JSON included
            result=result,
            seconds=seconds,
        )

    def finish(self, outcome: str, answer: str) -> None:
        """run_end for a run that ended normally: outcome is "answered" or "max_steps"."""
        self._end(outcome=outcome, answer=answer)

    def _end(self, **fields) -> None:
        """Write the one run_end, with the totals every run_end carries: model calls (steps),
        tool calls, tokens, and time. Anything reading traces gets these without recounting."""
        seconds = round(time.monotonic() - self._started, 3) if self._started is not None else None
        self.event(
            "run_end", **fields, steps=self.steps, tool_calls=self.tool_calls, tokens=dict(self.tokens), seconds=seconds
        )
        self._ended = True

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    # `with Tracer(path) as tracer:` runs __enter__ at the start of the block and __exit__
    # at the end, *however* the block ends: normally, by return, by exception, or by Ctrl-C.
    def __enter__(self) -> "Tracer":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        # A run that started but never reached finish(): record how it ended instead.
        if self._started is not None and not self._ended:
            if exc_type is None:
                self._end(outcome="unfinished")
            elif issubclass(exc_type, KeyboardInterrupt):  # Ctrl-C isn't an Exception subclass
                self._end(outcome="interrupted")
            else:
                self._end(outcome="crashed", error=f"{exc_type.__name__}: {exc}")
        self.close()
        # returning None (falsy) means: don't swallow the exception, let it propagate
