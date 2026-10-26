# How orchestra works

A picture of the agent as it exists today, kept up to date as the code changes.
Diagrams are [Mermaid](https://mermaid.js.org/), which GitHub renders automatically.

**Last updated:** 2026-10-26, only the provider holds a model client; nothing passes one around

---

## 1. The big picture

One command builds everything the agent needs, then hands it over. The orchestrator
wires things together; the agent loop does the work; every tool call passes through
one checkpoint.

```mermaid
flowchart TD
    user(["You<br/>orchestra 'task'  (one turn)<br/>orchestra chat  /  chat --resume ID"])

    subgraph iface["Interface: interfaces/cli.py (all terminal input/output)"]
        direction LR
        cli["main(): run / chat commands<br/>terminal_approver: the y/N prompt"]
    end

    subgraph control["Control plane: orchestrator.py (no terminal I/O)"]
        direction LR
        run["new_conversation() / resume_conversation()<br/>then Conversation.turn() per turn"]
    end

    cli -->|"approver + options"| run

    subgraph inputs["Config files"]
        direction LR
        agentmd["agents/base.md<br/>model, temperature, max_steps,<br/>tools, system prompt"]
        env[".env<br/>OMLX_API_KEY"]
        envyaml["environments/python-dev.yaml<br/>setup commands, env vars, prompt"]
    end

    subgraph built["Built fresh for each run"]
        direction LR
        cfg["config dict"]
        policy["Policy<br/>uses the approver the<br/>interface handed in"]
        toolset["Toolset<br/>only the tools this agent lists"]
        session["Session<br/>owns the conversation;<br/>system prompt pinned at creation"]
        tracer["Tracer<br/>appends the turn to<br/>the session's trace.jsonl"]
        policy --> toolset
    end

    agentloop["Agent loop<br/>agent.py: run_agent()"]
    omlx[("oMLX server<br/>Qwen3.8-27B")]
    ws[("the session's workspace<br/>~/.orchestra/&lt;app&gt;/sessions/&lt;id&gt;/workspace/ by default,<br/>or --workspace / workspace: in the agent file")]
    sessions[("~/.orchestra/&lt;app&gt;/sessions/&lt;id&gt;/<br/>session.json, messages.jsonl,<br/>trace.jsonl")]

    user --> cli
    run -->|reads| inputs
    inputs -->|"check everything first, then build"| built
    envyaml -.->|"provision: setup runs once,<br/>before turn 1"| ws
    built -->|handed to| agentloop
    agentloop -->|"each model call"| chatfn["calls.chat()<br/>makes the call + records it"]
    chatfn <-->|"POST /v1/chat/completions"| omlx
    chatfn -->|"one record per call"| applog[("~/.orchestra/&lt;app&gt;/log/&lt;day&gt;.jsonl")]
    agentloop <-->|"every tool call"| toolset
    toolset <--> ws
    agentloop -->|"every message: append"| session
    agentloop -->|"every step: events"| tracer
    session --> sessions
    tracer --> sessions
    tracer -.->|"each event, as it happens<br/>(listener: show_step)"| cli
```

---

## 2. One turn, step by step

`orchestra "task"` is a new session with one turn. `orchestra chat` loops: read a line, run a
turn, print the answer, repeat; every turn adds to the same session. `chat --resume ID`
reloads a saved session (repairing an interrupted history) and keeps going. The diagram
shows one turn of a new session; a resumed or later turn skips the creation steps.

The model is stateless: every turn sends the **whole** conversation. The loop ends
when the model replies without asking for a tool, or when `max_steps` runs out.

```mermaid
sequenceDiagram
    autonumber
    actor You
    participant C as interfaces/cli.py
    participant O as Orchestrator
    participant A as Agent loop
    participant M as Model (oMLX)
    participant T as Toolset
    participant P as Policy
    participant F as Tool function
    participant R as Tracer
    participant S as Session

    You->>C: orchestra "write primes.py and run it"
    C->>O: new_conversation(agent, approver=terminal_approver)
    O->>O: check config, workspace, environment, tool names<br/>(any problem fails here, before anything is created)
    O->>S: Session.create: new folder,<br/>system prompt (+ environment prompt) = message 0
    O->>O: provision: run the environment's setup in the workspace<br/>(fails: the session is removed)
    O->>R: Tracer(session.trace_path, turn=next turn)<br/>appends to the session's trace.jsonl
    O->>A: run_agent(task, session, config, toolset, tracer)
    A->>R: start: turn_start (turn, task, agent_version,<br/>max_steps, approver, sandbox)
    A->>S: append {user: task}

    loop each step, up to max_steps
        A->>M: chat(app, model, messages + tool schemas, session)
        M-->>A: Reply (chat() has recorded the whole call<br/>in the app log, APP/log/DAY.jsonl)
        A->>R: model_call (step, record id)
        A->>S: append reply.message<br/>(reasoning left out, it is in the record)
        alt no tool_calls
            A->>R: finish: turn_end (outcome: answered)
            A-->>O: final text answer
            O-->>C: TurnResult(answer, summary: outcome + totals)
            C-->>You: answer on stdout, summary line on stderr
        else has tool_calls
            loop each tool call, in order
                A->>T: execute(name, arguments JSON)
                T->>P: check(tool, parsed args)
                alt permission is allow
                    P-->>T: None (go)
                else permission is ask
                    P->>You: approver: cli.py's y/N prompt<br/>(or deny_all: no prompt)
                    P-->>T: None, or a refusal message
                end
                opt approved
                    T->>F: fn(**arguments)
                    F-->>T: result text
                end
                T-->>A: result or error, always a string
                A->>A: write a tool record (approval, result, seconds,<br/>parent = the model call that asked)
                A->>R: tool_run (step, record id)
                R-->>C: listener show_step: prints [step N] tool(args) -> result
                A->>S: append {role: tool, tool_call_id, content}
            end
        end
    end
    Note over A,R: max_steps reached: finish, turn_end (outcome: max_steps)<br/>crash or Ctrl-C: the orchestrator's `with Tracer` writes<br/>turn_end (crashed / interrupted), then the exception continues
```

---

## 3. Inside `Toolset.execute`: the one checkpoint

Every tool call goes through these checks in order. **Nothing here raises:** each
failure becomes text the model reads next turn, so it can fix its own mistake.
Malformed calls are rejected *before* a human is ever asked.

It returns a `ToolRun`: the result, and **who decided** (`approval`, shown on the arrows
below), which the tool's record keeps. A call rejected before the policy (unknown tool,
bad JSON) has `approval: null`: nobody was asked.

```mermaid
flowchart TD
    start(["execute(name, arguments_json)"]) --> known{"name in this<br/>agent's toolset?"}
    known -->|no| e1["Error: unknown tool<br/>+ list of available tools"]
    known -->|yes| parse{"arguments are<br/>valid JSON?"}
    parse -->|no| e2["Error: not valid JSON<br/>+ example of valid JSON"]
    parse -->|yes| obj{"a JSON object?"}
    obj -->|no| e3["Error: must be a JSON object"]
    obj -->|yes| perm{"tool.permission"}
    perm -->|"allow: auto-allow"| runfn
    perm -->|ask| approver{"approver"}
    approver -->|"terminal: you typed y<br/>human-allow"| runfn
    approver -->|"terminal: anything else<br/>human-deny"| r1["The user declined this call.<br/>Try a different approach."]
    approver -->|"deny_all (--unattended)<br/>or no terminal: auto-deny"| r2["Unavailable in this run.<br/>Don't retry; say what you couldn't do."]
    approver -->|"approve_all (--yes,<br/>sandbox required): auto-allow"| runfn
    runfn["tool.fn(**arguments)"] --> ok{"raised?"}
    ok -->|no| result["result as a string"]
    ok -->|TypeError| e4["Error: bad arguments"]
    ok -->|other exception| e5["Error: ExceptionType: message"]

    classDef err fill:#fde2e2,stroke:#c0392b,color:#000
    classDef refuse fill:#fff3cd,stroke:#b7950b,color:#000
    classDef good fill:#d5f5e3,stroke:#1e8449,color:#000
    class e1,e2,e3,e4,e5 err
    class r1,r2 refuse
    class result good
```

---

## 4. The tools

| Tool | Permission | What it does | Guardrails |
|---|---|---|---|
| `read_file` | allow | Return a file's text | Paths can't leave the workspace; binary/empty files say so; long output truncated (head + tail) |
| `write_file` | allow | Create or overwrite a file | Same path rule; creates parent folders |
| `bash` | **ask** | Run a shell command in the workspace | Approval first; **runs in the macOS sandbox** (section 6); **allowlisted environment only** (no API keys, not the harness's own Python); stdin closed; timeout kills the whole process group; output truncated; exit code always reported |

Every tool function takes a `ToolContext` first (today: just the workspace path). `Toolset`
passes it on every call; it isn't in any schema, so the model can neither see nor set it.

**Which workspace:** `--workspace PATH` wins, then `workspace:` in the agent file (relative to
the project root), else a fresh empty folder inside the session. A given folder must already
exist. The session records its workspace when created, so resuming works on the same files.

Tools come from the built-ins plus the project's own: its `tools.py` (defining `TOOLS`, loaded
by `Project.find`/`Project.at`, so the CLI picks them up) and any passed from Python. A project
adds tools without editing the harness. An agent only gets the tools its `.md` file lists. A tool it wasn't given can't run,
even if the model invents a call to it.

---

## 5. Environments: how a workspace is set up

An environment is a config file the **harness** applies before the agent starts, so setup is
deterministic and costs the agent no steps. Setup commands run without approval: they come
from a file you wrote (trusted), not from the model (untrusted).

```yaml
# environments/python-dev.yaml          (agents/base.md has: environment: python-dev)
description: Python 3.12 virtualenv with pytest
setup:                                  # run in the workspace, in order; any failure stops
  - uv venv --quiet --python 3.12 .venv
  - uv pip install --quiet --python .venv pytest
env:                                    # added to the agent's commands; PATH is prepended
  PATH: "{workspace}/.venv/bin"
  VIRTUAL_ENV: "{workspace}/.venv"
prompt: |                               # appended to the system prompt automatically
  Your workspace has a Python 3.12 virtualenv in .venv with pytest installed. ...
```

- **Which one:** `--environment NAME` (or `none`) wins over `environment:` in the agent file.
- **It belongs to the workspace:** provisioning writes `<workspace>/.orchestra/environment.json`
  (name + version) last, after every command succeeded. Same version already there: skipped
  (a shared workspace is set up once). A different version: refused, never set up over the top.
- **Same environment for commands and setup:** both run with `ToolContext.command_env()`, the
  allowlisted variables plus the environment's, so neither sees the harness's secrets.
- **Provenance:** the resolved config includes `environment: {name, version}`, where the version
  hashes the spec *and* its template files, so changing an environment changes `agent_version`
  (which each turn's `turn_start` records).

---

## 6. The sandbox: what bash commands can touch

`bash` asks `ctx.sandbox` to `prepare()` each command. On macOS the default is **Seatbelt**:
the command runs under `sandbox-exec` with rules generated for its workspace, and the
operating system enforces them. (`--no-sandbox` runs commands unconfined.)

| | Allowed | Blocked |
|---|---|---|
| **Write** | the workspace, `/dev` | everywhere else (`/tmp`, your home folder, the harness) |
| **Read contents** | everything outside your home folder; in home only the workspace and the environment's `sandbox.readable` (python-dev: uv's Pythons) | the rest of home: `~/.zshrc`, `~/.ssh`, the harness `.env`, other sessions, folder listings |
| **Check existence** | everywhere (tools like pytest and git walk up the tree looking for config files) | |
| **Network** | local sockets; the internet if the environment sets `sandbox.network: true` | the internet, by default |

**Who decides what:** the interface decides *whether* there's a sandbox (Seatbelt, or none with
`--no-sandbox`); the session's environment decides *what its commands need* (the `sandbox:`
section of its file, recorded in `session.json` when the session is created). `build_toolset`
combines them: `sandbox.with_rules(session.sandbox_rules)`.

Temp files go to `<workspace>/.tmp` (`TMPDIR`). Each turn's `turn_start` records `sandbox`.

**`--yes` (`approve_all`) is only allowed with the sandbox on**: with nobody approving, the
sandbox is the safety net. `require_sandbox_for_auto_approval()` in `policy.py` is the one
place that rule lives; the CLI calls it for a clear error, and the orchestrator calls it
again before creating anything, so code that skips the CLI can't bypass it.
`read_file`/`write_file` don't use the sandbox (they run inside the harness); `_resolve()`'s
path check confines them instead.

**Why sessions live in `~/.orchestra/<app>/sessions/`, outside this project:** tools walk *up*
from their folder looking for config. A workspace inside the harness project made git treat it
as part of the harness repository, and pytest read the harness's `pyproject.toml`. Outside, each
workspace is a clean folder. (Sessions from before apps are in `~/.orchestra/sessions/`;
`~/projects/orchestra-sessions` is a symlink to that old folder.)

---

## 7. Chat: many turns, one session

```text
$ orchestra chat                        $ orchestra chat --resume 2026-10-20T18-55-51_base_66ec
you> Create fruits.txt ...            you> which one did you add last time?
you> Add a fourth fruit to it ...     -> "mango", remembered from the earlier process
```

- **What's fixed when the session is created:** system prompt, workspace, environment.
  Resuming can't change them (`--agent`/`--workspace`/`--environment` with `--resume` is an error).
- **What comes from the agent file each time:** model, temperature, max_steps, tools; each
  turn's trace records exactly what it used.
- **Ctrl-C during a turn** stops only that turn: `session.repair()` answers any tool call left
  without a result (the API would reject the next request otherwise), and the prompt comes back.
  **An error** (e.g. the model server going away) does the same. Ctrl-C/Ctrl-D/`/exit` at the
  prompt leaves, printing the command to resume.

---

## 8. What the model sees: the conversation grows

From a real run of `"Write primes.py that prints the first 10 primes, run it"`:

```text
request 1: [system] [user]
request 2: [system] [user] [assistant: write_file] [tool: Wrote 381 characters]
request 3: [system] [user] [assistant: write_file] [tool: ...] [assistant: bash] [tool: [2, 3, 5, ... 29] (exit code 0)]
answer:    "Done! I created primes.py and ran it. The output was ..."
```

History is append-only: each request starts with the previous one, unchanged. That's
what lets oMLX reuse its prompt cache instead of re-reading the whole conversation.

The history lives in the **session**, which owns it: `session.messages` is read-only, and
`session.append()` is the only way to add to it, saving each message to `messages.jsonl`
as it happens. A second turn in the same session starts from all of the first:

```text
turn 2, request 1: [system] [user: turn 1 task] ... [assistant: turn 1 answer] [user: turn 2 task]
```

**Session vs. trace vs. records:** the session is the *current state* (exactly what the model
sees next; later compaction may rewrite it); the trace is the *ground truth of the session's
history* (what happened, in order, never rewritten); records hold the *content* of each model
call and tool run (section 10). Why: [docs/decisions.md](docs/decisions.md).

```text
~/.orchestra/orchestra-cli/sessions/2026-10-20T14-13-59_base_8569/
  session.json      id, app, agent, agent_version, workspace, created: written once, never changed
  messages.jsonl    the conversation, one message per line
  trace.jsonl       the session's history; its turn_start lines are the only record of which turns happened
  workspace/        where the agent works (unless the session was given another folder)
```

---

## 9. What a trace records

**The trace is also how interfaces see progress.** The agent loop never prints: it reports
each event to the tracer, which writes it to the file and then calls any *listeners* with it.
`Conversation.turn(task, on_event=...)` is how an interface attaches one; the terminal's
`show_step` prints each `tool_run` as it finishes (listeners also get the tool's name,
arguments, and result live, which the trace file doesn't store). A web UI would attach a listener that
streams events to the browser, with no change to the agent loop.

One JSONL file per session, appended to by each turn: `~/.orchestra/<app>/sessions/<id>/trace.jsonl`.
The format is defined in one place, `tracer.py`: the agent reports what happened
(`start`, `model_call`, `tool_run`, `finish`), and the tracer decides what's recorded.
**The trace holds references and facts no record has, never copies:** model calls and tool
runs are records (section 10), and the trace's `model_call` / `tool_run` events hold only
their record ids, in order. Anything derivable isn't stored: token totals are sums over the
records, a turn's time is `turn_end`'s time minus `turn_start`'s, its answer is the last
model call's response. Why this split, and what was rejected: [docs/decisions.md](docs/decisions.md).
One tracer = one turn. The orchestrator owns it with `with Tracer(path) as tracer:`,
so even a crash or Ctrl-C ends the turn with exactly one `turn_end`.
Each line is flushed as it happens, so a crashed turn still leaves its trace.
From a real run:

```text
turn_start  turn=1  task="Use bash to run: echo approved-run..."  agent_version=0915a4a3ac62
            max_steps=20  approver=terminal_approver  sandbox=seatbelt
model_call  step 1  record=chat_2026-10-24T17-07-23_e5985c   (asks for bash)
tool_run    step 1  record=tool_2026-10-24T17-07-26_05b42f   (human-allow, "approved-run (exit code 0)")
model_call  step 2  record=chat_2026-10-24T17-07-26_65f572   (the final answer)
turn_end    turn=1  outcome=answered
```

| Event | When | Fields |
|---|---|---|
| `turn_start` | once per turn, first | turn, task, `agent_version`, max_steps, approver, sandbox |
| `model_call` | every model call | step, **record**: the chat record has the request, response, reasoning, usage |
| `tool_run` | every tool call | step, **record**: the tool record has approval, result, seconds; its `parent` (the model call) has the name and raw arguments |
| `turn_end` | once per turn, last | turn, outcome: `answered` / `max_steps` / `crashed` / `interrupted` (+ error if crashed) |

Listeners get two things the file doesn't store: a `tool_run`'s name, arguments, and result
(the terminal prints them), and `turn_end`'s totals (steps, tool calls, tokens, seconds: the
terminal's summary line). `TurnResult.summary` has the same totals.

**Provenance**: what produced a turn, so turns can be filtered later (evals, fine-tuning data).
A hash *identifies* content; it doesn't store it.

| Field | Where | Meaning | Recoverable? |
|---|---|---|---|
| `agent_version` | `turn_start` | 12-char hash of the agent definition (name excluded): same content, same version | Yes: its parts are in the records (model, temperature, system prompt, tools), `turn_start` (max_steps), and `session.json` |
| `orchestra.version` / `.installed` | every record | the package's version; how it got there: `checkout` (its own source), `git` (a git install), or `path` (a folder or wheel) | |
| `orchestra.commit` / `.dirty` | every record | the commit: from git in a checkout (with `dirty`), from the installer's record for a git install (never dirty), `null` for a path install | Via `git checkout`, only if `dirty` is false |
| `request.model` / `response.model` | every chat record | the model asked for, and the one the server says answered | Only if the same weights still exist |

---

## 10. Recorded model calls: `chat()` and records

Any app can call a model through orchestra, and every call is recorded in the same
function that makes it, so none can go unrecorded. Orchestra acts as a service: the
app says who it is (`app`) and can attach its own data (`metadata`); orchestra keeps
both with the record and never reads them.

```mermaid
flowchart LR
    app(["an app<br/>chat(app=, model=, messages=, metadata=)"]) --> start["records.start<br/>check app + metadata<br/>(fails before any model time)"]
    start --> send["providers.send()<br/>talks to oMLX, returns plain dicts"]
    send -->|reply| ok["record: request, response<br/>(reasoning too), usage"]
    send -->|error, Ctrl-C| bad["record: request + error"]
    ok --> save["records.save<br/>&lt;app&gt;/log/&lt;day&gt;.jsonl"]
    bad --> save
    save -->|Reply| app
    bad -.->|then raised| app
```

- **One record = one model call,** holding the full request (exactly what was sent), so each
  record is a complete example on its own: for tracking now, for training later.
- **`records.py` owns the format** and is the only code that writes it. Every record has
  the same envelope (`v`, `id`, `parent`, `session`, `app`, `kind`, `time`, `seconds`,
  `error`, `metadata`, `orchestra`). `kind: "chat"` adds `request`, `response`, `usage`;
  `kind: "tool"` (written by the agent loop) adds `tool_call_id`, `approval` (auto-allow /
  auto-deny / human-allow / human-deny, or null if never asked), and `result`, with the
  model call that asked for it as its `parent`.
- **Grouped by app, then day:** `~/.orchestra/jewels/log/2026-10-24.jsonl`. Each app's folder
  also holds its sessions (`~/.orchestra/jewels/sessions/`); `records.app_dir` is the one place
  that says where an app's folder is.
  Orchestra's own callers are named `orchestra-<name>` (the CLI will be `orchestra-cli`).
- **The agent loop uses it too:** each step is `chat(app=session.app, session=session.id, ...)`,
  so an agent's calls land in its app's log like any other, and the trace only points at them.
  A session records its `app` in `session.json`, so a resumed session keeps it.
- **How other apps use it:** [docs/for-apps.md](docs/for-apps.md).

---

## 11. How the code is organized

```text
src/orchestra/
  interfaces/   cli.py          where a human talks to the harness (terminal now; a web UI later)
  orchestrator.py               wiring: Conversation, new/resume, no terminal I/O
  agent.py                      the loop
  calls.py                      chat(): one model call for any app, made and recorded
  records.py                    the record format, the only writer of <app>/log/, and app_dir()
  providers/    omlx.py         the model server: client (cached), send (the only code that has a client or knows oMLX's responses)
  tools/        base.py, toolset.py, builtin/{fs,shell}.py
  policy.py  config.py  environment.py  session.py  tracer.py  provenance.py  sandbox.py
```

Arrows mean "imports". They only point one way: nothing imports an interface,
nothing but the interface imports the orchestrator,
tools never import the machinery around them, and `policy.py` imports nothing at runtime.

```mermaid
flowchart TD
    climod["interfaces/cli.py<br/>terminal interface:<br/>commands, chat loop, y/N prompt"] --> orch
    orch["orchestrator.py<br/>control plane: Conversation,<br/>no terminal I/O"] --> agent["agent.py<br/>the loop"]
    orch --> loader["config.py<br/>Project (root + its tools); reads + resolves<br/>agents/ and environments/ into a RunConfig"]
    orch --> providers["providers/omlx.py<br/>client (cached; key check), send"]
    base --> sandboxmod["sandbox.py<br/>Seatbelt: confine bash"]
    agent --> calls
    loader --> envmod
    loader --> tools_api
    orch --> sessionmod["session.py<br/>conversation on disk"]
    orch --> envmod["environment.py<br/>load + provision"]
    envmod --> base
    agent --> sessionmod
    sessionmod --> envmod
    orch --> policy["policy.py<br/>Policy, approvers"]
    orch --> tools_api["tools/__init__.py<br/>public API"]
    orch --> tracer["tracer.py<br/>JSONL events"]
    agent --> tracer
    calls["calls.py<br/>chat(), Reply"] --> records["records.py<br/>record format, app folders"]
    calls --> providers
    records --> prov
    tracer --> prov["provenance.py<br/>agent hash, git state"]
    agent --> tools_api
    tools_api --> toolset["tools/toolset.py<br/>REGISTRY + Toolset"]
    toolset --> policy
    toolset --> builtin["tools/builtin/<br/>fs.py, shell.py"]
    toolset --> base["tools/base.py<br/>Tool, ToolContext, agent_env, truncate"]
    builtin --> base
    base -.->|"Permission type only"| policy
```

---

## Not built yet

What's planned, in rough order. Each becomes a diagram change when it lands.

- **jewels on `chat()`,** with its job id in `metadata`; then chaos, once `chat()` has retries and timeouts.
- **Reading records:** `orchestra log` to list records, show a session as a tree, and total tokens by app.
- **Reading traces:** `orchestra show <trace>` to print a past run readably, and a summary line (steps, tokens, cache hits, time) at the end of each run.
- **More tools:** `list_dir`, `edit`, eventually `web_search` and MCP servers.
- **Context management:** compaction when the conversation nears the 32k window.
- **Provider adapters:** oMLX and native Anthropic behind one interface.
