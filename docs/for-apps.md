# Using orchestra from an app

How another project (jewels, chaos, anything new) gets model calls through orchestra.
Orchestra works like a service provider for your apps: an app says who it is and asks
for a call; orchestra makes the call and keeps a record of it. Every call any app makes
ends up in one place, `~/.orchestra/`, one folder per app, in one format.

## 1. Add orchestra to the app

In the app's `pyproject.toml` (uv):

```toml
[project]
dependencies = ["orchestra"]

[tool.uv.sources]
orchestra = { path = "../../harness", editable = true }   # path to this repo from the app's folder
```

Then `uv sync`. Editable means the app always runs this repo's current code.

Orchestra only **reads** `OMLX_API_KEY` (and optionally `OMLX_BASE_URL`) from the
environment; loading them is the app's job. For example, jewels loads this repo's `.env`:
`load_dotenv(Path("~/projects/harness/.env").expanduser())`.

## 2. Pick an app name

Every call needs `app=`: it's how records are grouped, and the folder they go in
(`~/.orchestra/<app>/`). Use the project's short name: `"jewels"`, `"chaos"`.
Plain names only (letters, digits, `.`, `_`, `-`). Names starting with `orchestra-` are
orchestra's own (`orchestra-cli`, `orchestra-test`); don't use them for an app.

## 3. Make calls

**One model call, no agent:** `chat()`.

```python
from orchestra import chat

reply = chat(
    app="jewels",
    model="Qwen3.8-27B-MLX-8bit",
    messages=[{"role": "system", "content": "..."}, {"role": "user", "content": idea}],
    temperature=0.4,                         # optional; also max_tokens, tools
    metadata={"job": job_id, "idea": idea},  # optional: your own data, kept with the record
)
reply.content        # the answer (None if the model only asked for tools)
reply.reasoning      # the model's thinking, if the server split it out
reply.tool_calls     # [{"id", "type", "function": {"name", "arguments"}}], or []
reply.finish_reason  # "stop", "tool_calls", or "length" (cut off: raise max_tokens)
reply.usage          # {"prompt", "completion", "cached"} tokens
reply.id             # this call's record id
reply.message        # the reply as an assistant message, to append for the next call
```

An app running **its own loop** (its own tools, like chaos) calls `chat()` each step and
appends `reply.message`, then a `{"role": "tool", ...}` message per tool call. Pass
`parent=` or `session=` (any id string of yours) if you want records linked.

**An orchestra agent** (an `agents/<name>.md` in the app, with orchestra's loop, tools,
sessions, and sandbox): `new_conversation()`.

```python
from orchestra.config import Project
from orchestra.orchestrator import new_conversation

conv = new_conversation("artist", app="jewels", project=Project.at(APP_ROOT))  # APP_ROOT holds agents/
answer = conv.turn(idea).answer
conv.session.id      # the session; every model call it makes is recorded with this id
```

## 4. What `metadata` is for

Anything about the call that only your app knows and you'd want later: the job or
request id, the user's original input, which feature asked, the candidate number. It
must be a dict that can be stored as JSON. Orchestra stores it exactly as given and
never reads it.

What *isn't* about a model call (your HTTP requests, your job queue, your own progress)
stays in your app's own logs. To connect the two, write `reply.id` into your own log
line: `print(f"job {job_id}: {reply.content!r} (call {reply.id})")`.

## 5. Where calls go, and what a record holds

```text
~/.orchestra/<app>/log/2026-10-24.jsonl     one line per call, appended
~/.orchestra/<app>/sessions/<id>/            an agent's sessions (new_conversation)
```

Each record has the full request (exactly what was sent), the full response (reasoning
included), token usage, timing, your `metadata`, and which orchestra code ran. A call
that fails is recorded too, with its `error`. An agent's tool runs are records too, saying
who approved each one (`auto-allow`, `auto-deny`, `human-allow`, `human-deny`), and its
session's `trace.jsonl` lists every record id in order. The format is defined in one place:
[`src/orchestra/records.py`](../src/orchestra/records.py).

## 6. Errors

| When | What you get |
|---|---|
| bad `app` name, `metadata` not a JSON-storable dict | `ValueError`, before any model time is spent; nothing recorded |
| `OMLX_API_KEY` not set | `orchestra.providers.MissingAPIKey`, recorded, then raised |
| the server fails, times out, or you press Ctrl-C | the original exception, recorded, then raised |

Calls are safe from several threads at once.

## 7. Not there yet

- **No retries,** and the timeout is the client's 600 seconds. Catch errors and fall back
  (jewels uses the idea as written). Retries and a shorter timeout are planned.
- **No way to read records back** except opening the files; an `orchestra log` command is planned.
- **Text models only.** Image generation (`image()`) is planned, recorded the same way.

## Checklist for a new app

1. Add the dependency (section 1) and load `OMLX_API_KEY` into the environment.
2. Choose its `app` name.
3. Use `chat()` for single calls, `new_conversation()` for agents; pass `metadata` with your ids.
4. Log `reply.id` (or `conv.session.id`) next to your own records of the same work.
5. When this guide changes the API, check the apps that use it (today: jewels).
