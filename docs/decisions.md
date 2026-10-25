# Design decisions

Decisions worth revisiting later, with why they were made and what was rejected. Newest first.

---

## 2026-10-24: What is stored where: records, sessions, traces

**Status:** accepted; being implemented in steps (below). Marked for revisiting.

### The goal

Orchestra should be the **one place every model call is recorded**, for every app of
mine (jewels, chaos, the CLI), whether or not the app uses orchestra's agents. The records
should be complete enough to train on later (SFT), so anything that can't be reconstructed
later has to be captured when it happens.

### The problem it solved

Before this, a model call was only recorded as part of an agent turn, and the same
information was written up to three times:

- the conversation: in `messages.jsonl`, and again in each trace;
- each model reply: in the trace, and in `messages.jsonl`;
- tool results: in the trace, and in `messages.jsonl`.

An app that only wanted one model call (jewels) had to create a whole session and an
empty workspace to get it recorded. An app with its own loop (chaos) couldn't be recorded at all.

### The decision: three stores, one job each

| Store | Job | Holds | Changes? |
|---|---|---|---|
| **Records** (`records.py`) | the **content** of each thing that happened | model calls (full request, full response incl. reasoning, usage, timing); tool runs (who approved, result, timing) | never: append-only |
| **Session** (`session.py`) | the **current state**, to continue from | `session.json`, `messages.jsonl` (what the model sees next), `workspace/` | yes (compaction will rewrite it) |
| **Trace** (`tracer.py`) | the **ground truth of a session's history**: what happened, in what order | references to records by id, plus facts that exist nowhere else | never: append-only |

**The rule for the trace:** every entry is either a *reference* (`record=chat_…`) or a
*fact no record holds*: the task, the approver and sandbox a turn ran with, how a turn
ended, resumes, compactions. No copies of content. Anything derivable (token totals) is
derived, not stored.

**Records are made where the work is done:** `chat()` makes the model call *and* writes its
record, so no call can go unrecorded. Any app can use `chat()` without sessions or agents.

**Layout, app first:** each app is a "customer" with its own space:

```text
~/.orchestra/<app>/log/<day>.jsonl     records
~/.orchestra/<app>/sessions/<id>/      sessions
~/.orchestra/sessions/<id>/            sessions from before apps (found on resume; never written to)
```

No `apps/` level in between: anything of orchestra's own is just an app too
(`orchestra-cli` for the command). The one name no app may take is `sessions`.

- `app` is **required** on every call. Orchestra's own callers are named `orchestra-<name>`
  (`orchestra-cli`, `orchestra-test`).
- Apps attach their own data as `metadata` (any JSON dict), which orchestra stores as given
  and never reads. Events that aren't model calls stay in the app's own logs; the app links
  them by logging the record id (`reply.id`).
- Every record has a format version `v`, so old and new formats stay distinguishable.

### Overlaps kept on purpose

- **Every record holds its full request.** So each call's request repeats the conversation
  before it, and a tool's result appears in its tool record *and* in the next call's request.
  This is the price of every model-call record being a complete training example on its own.
  If storage grows too much: compress old day files (repeated prefixes compress very well).
- **The task text** is in the trace's `turn_start` and in the next request. Messages have no
  ids to reference, and their positions change once compaction rewrites `messages.jsonl`.
- **Records carry `session` and `parent` back-references,** which overlap with the trace's
  structure. They let a record be found and understood without reading the trace, and give
  apps without sessions (their own loops) a way to link records.

### Alternatives rejected

1. **The trace copies everything (the original design).** Readable on its own, but it stored
   everything two or three times, and only agent turns were recorded at all.
2. **Apps record with the tracer directly.** The tracer needs a session and a turn, and
   recording would be a separate step each app could forget. Recording has to be inside the call.
3. **A logging proxy in front of oMLX.** Catches every app with no code changes, but it
   doesn't know which app, agent, or job made a call, which is exactly what training and
   tracking need. Could still be added later as a catch-all.
4. **No trace; one "turn" record per turn with only the unique facts** (proposed and
   dropped the same day). It loses the explicit ordering, has no place for session-level
   events (resume, compaction), and a killed process loses the whole turn's record. The
   argument for it ("a trace of references needs lookups to read") didn't hold: the trace
   already needed lookups once model calls became references.
5. **Each app keeps its call logs in its own repo.** Then there's no central record, and a
   training export would have to know where every app keeps its logs.
6. **`log/` and `sessions/` as separate top-level folders.** Sessions belong to an app just
   like records do, so the app is the top-level folder.
7. **An `apps/` folder holding the app folders** (`~/.orchestra/apps/<app>/`), to leave room
   for things that aren't per-app. Dropped: orchestra's own things are an app too.

### Questions to revisit

- **Storage growth** from full requests: how big does a chaos night get? When to compress?
- **Tool records don't repeat the tool's name and arguments** (they're in the model call
  that asked for it, found via `parent` + `tool_call_id`). "Every bash run" therefore needs a
  lookup. Worth adding the name to tool records for querying?
- **Approval labels from a pipe:** a `y` piped into the terminal approver (`yes | orchestra …`)
  is recorded as `human-allow`; the approver can't tell a script's answer from a person's.
  (No terminal at all is `auto-deny`, correctly.) Worth distinguishing if scripted approvals
  start mattering for training data.
- **Trace ordering vs `parent` links:** two encodings of structure. Is one enough?
- **Labels for training** ("tests passed", "the user picked image 2"): a separate append-only
  file keyed by record id. Not designed yet.
- **Images and other model kinds:** `image()` beside `chat()`, same envelope, files saved
  next to the day's log. If several apps load the same image model, orchestra needs to
  become a background service that holds the models.
- **One trace per session vs one per turn** (decided: one per session). Revisit if session
  traces get unwieldy.

### Steps

1. ✅ `records.py` + `chat()`: any app's model call, made and recorded together.
2. 2a. ✅ The agent loop calls the model through `chat()`; the trace stores only `call_id`.
   2b. ✅ Tool runs become records; the trace's `tool_run` references them.
   2c. ✅ The trace holds references and turn facts only (`turn_start`/`turn_end` instead
   of `run_start`/`run_end`); one `trace.jsonl` per session.
   2d. ✅ Tool records say who approved each call: `auto-allow`, `auto-deny`, `human-allow`,
   `human-deny` (`null` if the call was rejected before anyone was asked).
3. ✅ Sessions move under `<app>/sessions/`; resume finds a session by id in any app, or
   in the old `~/.orchestra/sessions/`.
4. jewels on `chat()` with its job in `metadata`; retries/timeouts/`max_tokens` in `chat()`; then chaos.
5. `orchestra log`: list records, show a session's trace with its records filled in, totals.
