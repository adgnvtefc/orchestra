"""Sessions: a conversation that persists across turns and can be resumed from disk.

A session is the agent's *memory*: exactly the messages the model will see next
turn. A trace is the *record* of what happened. They're kept apart on purpose:
compaction (later) will rewrite a session's history, while traces never change.

On disk, one folder per session:

    sessions/<id>/session.json      id, agent, agent_version, workspace, created: written once, never changed
    sessions/<id>/messages.jsonl    the conversation, one message per line, appended as it happens
    sessions/<id>/turn-001.jsonl    the trace of turn 1, turn-002.jsonl of turn 2, ...
    sessions/<id>/workspace/        where the agent works, unless the session was given another folder

A turn is one instruction plus everything the agent does about it: one run of the
loop, one trace file. The trace files are the *only* record of which turns happened.
Creating its trace is the first thing a turn does, so a turn that crashes or is
killed still counts, and numbering can never collide.

This module only reads and writes files. It knows nothing about models or tools.
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path

SESSION_FILE = "session.json"
MESSAGES_FILE = "messages.jsonl"

# What a tool call that never got its result is answered with on resume.
INTERRUPTED = "Interrupted: the session stopped before this tool call ran, so it has no result. Run it again if you still need it."


def new_session_id(agent: str) -> str:
    """2026-10-20T14-03-22_base_1a2b: sorts by time; the suffix keeps same-second sessions apart."""
    return f"{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}_{agent}_{uuid.uuid4().hex[:4]}"


class Session:
    def __init__(self, directory: Path, meta: dict, messages: list[dict]):
        """Use Session.create() or Session.load(), not this directly."""
        self.dir = directory
        self.meta = meta
        self._messages = messages  # private: only append() may change it, so memory and disk can't drift

    @property
    def messages(self) -> tuple[dict, ...]:
        """The history, read-only. A tuple has no .append(), so the only way to add a message
        is Session.append(), which also writes it to disk."""
        return tuple(self._messages)

    @property
    def id(self) -> str:
        return self.meta["id"]

    @property
    def agent(self) -> str:
        return self.meta["agent"]

    @property
    def workspace(self) -> Path:
        """The folder this session's tools work in. Fixed at creation, so a resumed session
        keeps working on the same files."""
        return Path(self.meta["workspace"])

    @property
    def next_turn(self) -> int:
        """One past the number of turn traces on disk, however those turns ended."""
        return len(list(self.dir.glob("turn-*.jsonl"))) + 1

    def trace_path(self, turn: int) -> Path:
        """Where turn `turn`'s trace goes: sessions/<id>/turn-002.jsonl."""
        return self.dir / f"turn-{turn:03d}.jsonl"

    @classmethod
    def create(
        cls, root: Path, agent: str, system_prompt: str, agent_version: str, workspace: Path | None = None
    ) -> "Session":
        """Start a new session. The system prompt becomes message 0 and is fixed from here on:
        editing the agent file later doesn't change sessions that already exist.

        `workspace`: an existing folder to work in. None (the default) gives the session its
        own empty folder, sessions/<id>/workspace/, so runs never see each other's files.
        """
        session_id = new_session_id(agent)
        directory = root / session_id
        directory.mkdir(parents=True)  # no exist_ok: a session folder is never reused
        if workspace is None:
            workspace = directory / "workspace"
            workspace.mkdir()
        meta = {
            "id": session_id,
            "agent": agent,
            "agent_version": agent_version,  # what the session started with; each turn's trace has its own provenance
            "workspace": str(workspace.resolve()),
            "created": _now(),
        }
        session = cls(directory, meta, [])
        session._save_meta()
        session.append({"role": "system", "content": system_prompt})
        return session

    @classmethod
    def load(cls, root: Path, session_id: str) -> "Session":
        """Reopen a saved session. If it was interrupted mid-turn, repair the history first."""
        directory = root / session_id
        meta = json.loads((directory / SESSION_FILE).read_text())
        messages = [json.loads(line) for line in (directory / MESSAGES_FILE).read_text().splitlines() if line]
        session = cls(directory, meta, messages)
        session._repair_unanswered_tool_calls()
        return session

    def append(self, message: dict) -> None:
        """Add a message to the history, in memory and on disk at once.

        One line per message, written and flushed immediately, like the tracer: a crash
        loses nothing that was already appended.
        """
        self._messages.append(message)
        with (self.dir / MESSAGES_FILE).open("a") as file:
            file.write(json.dumps(message, ensure_ascii=False, default=str) + "\n")

    def _save_meta(self) -> None:
        # Write a temp file, then rename it into place. A rename is atomic: a crash leaves
        # either no session.json or a complete one, never a half-written file.
        path = self.dir / SESSION_FILE
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.meta, indent=2, ensure_ascii=False))
        os.replace(tmp, path)

    def _repair_unanswered_tool_calls(self) -> None:
        """Answer tool calls left without results, so the history is valid to send again.

        The API rejects a history where an assistant message's tool_calls aren't each
        followed by a tool result. That happens when a turn is interrupted (Ctrl-C, crash)
        after the model asked for tools but before all their results were saved. Only the
        end of the history can be in that state, since results are saved as they arrive.
        """
        last_call = max((i for i, m in enumerate(self._messages) if m.get("tool_calls")), default=None)
        if last_call is None:
            return

        answered = {m.get("tool_call_id") for m in self._messages if m.get("role") == "tool"}
        for i, message in enumerate(self._messages[:last_call]):
            missing = [c["id"] for c in message.get("tool_calls", []) if c["id"] not in answered]
            if missing:  # can't be fixed by appending: results must directly follow their call
                raise ValueError(f"session {self.id}: message {i} has unanswered tool calls {missing}; history is corrupt")

        for call in self._messages[last_call]["tool_calls"]:
            if call["id"] not in answered:
                self.append({"role": "tool", "tool_call_id": call["id"], "content": INTERRUPTED})


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
