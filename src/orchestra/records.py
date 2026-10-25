"""Records: the one log of everything orchestra does for an app, one JSON object per line.

This file *is* the record format, and the only code that writes it. Every record has
the same envelope; what's left depends on its kind:

    "chat"   a model call (calls.chat): request, response, usage
    "tool"   a tool run (agent._run_tool): tool_call_id, approval (policy.Approval, or null
             if it never got that far), result. Its parent is the chat record that asked
             for it, which holds the tool's name and arguments.

The envelope:

    v          format version: when the format changes, old and new records stay distinguishable
    id         "<kind>_<time>_<random>", e.g. "chat_2026-10-24T01-23-51_8a6f3c": unique, and says its day
    parent     the id of the record that caused this one (a tool run, say), or null
    session    the session it belongs to, or null
    app        who asked: "jewels", "chaos"; orchestra's own are "orchestra-<name>" ("orchestra-cli")
    kind       "chat" or "tool"
    time       when it started (local time)
    seconds    how long it took
    error      null, or what went wrong ("TimeoutError: ...")
    metadata   the caller's own data, stored exactly as given and never read by orchestra
    orchestra  which orchestra code ran (provenance.orchestra_version())

Everything orchestra keeps is grouped by app, one folder each (app_dir); records go in
its log, by day:

    ~/.orchestra/<app>/log/2026-10-24.jsonl
    ~/.orchestra/<app>/sessions/            (orchestrator.py)
"""

import json
import re
import secrets
import threading
from datetime import datetime
from pathlib import Path

from orchestra import provenance

FORMAT_VERSION = 1
ORCHESTRA_DIR = Path.home() / ".orchestra"
# Sessions from before they were grouped by app are in ~/.orchestra/sessions/, so no app
# may be called that.
_RESERVED = {"sessions"}

# App names become folder names, so they're kept to plain ones: no "/", no "..".
_APP_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
# One process (jewels' server, say) may save from several threads at once; without this,
# two long lines could interleave in the file.
_write_lock = threading.Lock()


def start(kind: str, *, app: str, metadata: dict | None = None, parent: str | None = None,
          session: str | None = None) -> dict:
    """A new record's envelope, before the work it records starts.

    Checks the caller's input here, so a bad app name or metadata that can't be stored
    fails before any model time is spent. The caller adds its kind's fields, then fills
    in `seconds` and `error` and calls save().
    """
    check_app(app)
    metadata = {} if metadata is None else metadata
    if not isinstance(metadata, dict):
        raise ValueError(f"metadata must be a dict, not {type(metadata).__name__}")
    try:
        json.dumps(metadata)
    except (TypeError, ValueError) as error:
        raise ValueError(f"metadata must be storable as JSON: {error}") from None
    now = datetime.now()
    return {
        "v": FORMAT_VERSION,
        "id": f"{kind}_{now:%Y-%m-%dT%H-%M-%S}_{secrets.token_hex(3)}",
        "parent": parent,
        "session": session,
        "app": app,
        "kind": kind,
        "time": now.isoformat(timespec="milliseconds"),
        "seconds": None,
        "error": None,
        "metadata": metadata,
        "orchestra": provenance.orchestra_version(),
    }


def check_app(app: str) -> None:
    """Raise ValueError unless `app` can be an app name: it becomes a folder name."""
    if not isinstance(app, str) or not _APP_NAME.fullmatch(app):
        raise ValueError(f"app must be a plain name like 'jewels' (letters, digits, . _ -), not {app!r}")
    if app in _RESERVED:
        raise ValueError(f"app can't be called {app!r}: ~/.orchestra/{app}/ holds sessions from before apps")


def app_dir(app: str) -> Path:
    """The folder everything of `app`'s lives in: ~/.orchestra/<app>/."""
    check_app(app)
    return ORCHESTRA_DIR / app


def save(record: dict) -> Path:
    """Append `record` to its app's log for the day it started. Returns the file."""
    path = app_dir(record["app"]) / "log" / f"{record['time'][:10]}.jsonl"
    # default=str: the model's time is already spent, so anything JSON can't encode is
    # written as its str() rather than losing the record
    line = json.dumps(record, default=str) + "\n"
    with _write_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as file:
            file.write(line)
    return path
