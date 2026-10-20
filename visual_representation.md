# How the harness works

A picture of the agent as it exists today, kept up to date as the code changes.
Diagrams are [Mermaid](https://mermaid.js.org/), which GitHub renders automatically.

**Last updated:** 2026-10-20, sessions: every run is a turn of a saved conversation (on top of commit `c84f069`)

---

## 1. The big picture

One command builds everything the agent needs, then hands it over. The orchestrator
wires things together; the agent loop does the work; every tool call passes through
one checkpoint.

```mermaid
flowchart TD
    user(["You<br/>uv run harness 'task' --agent base --unattended"])

    subgraph control["Control plane: orchestrator.py"]
        direction LR
        cli["main()<br/>parses flags"] --> run["run()"]
    end

    subgraph inputs["Config files"]
        direction LR
        agentmd["agents/base.md<br/>model, temperature, max_steps,<br/>tools, system prompt"]
        env[".env<br/>OMLX_API_KEY"]
    end

    subgraph built["Built fresh for each run"]
        direction LR
        cfg["config dict"]
        client["OpenAI client"]
        policy["Policy<br/>terminal_approver, or<br/>deny_all if --unattended"]
        toolset["Toolset<br/>only the tools this agent lists"]
        session["Session<br/>owns the conversation;<br/>system prompt pinned at creation"]
        tracer["Tracer<br/>one file per turn"]
        policy --> toolset
    end

    agentloop["Agent loop<br/>agent.py: run_agent()"]
    omlx[("oMLX server<br/>Qwen3.8-27B")]
    ws[("workspace/<br/>the only folder tools can touch")]
    sessions[("sessions/&lt;id&gt;/<br/>session.json, messages.jsonl,<br/>turn-001.jsonl, turn-002.jsonl, ...")]

    user --> cli
    run -->|reads| inputs
    inputs -->|"load_agent_config, make_client"| built
    built -->|handed to| agentloop
    agentloop <-->|"POST /v1/chat/completions"| omlx
    agentloop <-->|"every tool call"| toolset
    toolset <--> ws
    agentloop -->|"every message: append"| session
    agentloop -->|"every step: events"| tracer
    session --> sessions
    tracer --> sessions
```

---

## 2. One run, step by step

The model is stateless: every turn sends the **whole** conversation. The loop ends
when the model replies without asking for a tool, or when `max_steps` runs out.

```mermaid
sequenceDiagram
    autonumber
    actor You
    participant O as Orchestrator
    participant A as Agent loop
    participant M as Model (oMLX)
    participant T as Toolset
    participant P as Policy
    participant F as Tool function
    participant R as Tracer
    participant S as Session

    You->>O: harness "write primes.py and run it"
    O->>O: load config, make client,<br/>build Policy + Toolset
    O->>S: Session.create: new folder,<br/>system prompt = message 0
    O->>R: Tracer(session.trace_path(next turn))<br/>turn-001.jsonl exists from here on
    O->>A: run_agent(task, session, config, client, toolset, tracer)
    A->>R: start: run_start (task, config, tools, approver, provenance)
    A->>S: append {user: task}

    loop each step, up to max_steps
        A->>M: messages + tool schemas
        M-->>A: response: message, finish_reason, usage
        A->>R: model_response (message, reasoning,<br/>usage, seconds)
        A->>S: append assistant message<br/>(reasoning dropped here, kept in the trace)
        alt no tool_calls
            A->>R: finish: run_end (outcome: answered)
            A-->>O: final text answer
            O-->>You: print answer
        else has tool_calls
            loop each tool call, in order
                A->>T: execute(name, arguments JSON)
                T->>P: check(tool, parsed args)
                alt permission is allow
                    P-->>T: None (go)
                else permission is ask
                    P->>You: approver: y/N prompt<br/>(or deny_all: no prompt)
                    P-->>T: None, or a refusal message
                end
                opt approved
                    T->>F: fn(**arguments)
                    F-->>T: result text
                end
                T-->>A: result or error, always a string
                A->>R: tool_call (name, raw arguments,<br/>result, seconds)
                A->>S: append {role: tool, tool_call_id, content}
            end
        end
    end
    Note over A,R: max_steps reached: finish, run_end (outcome: max_steps)<br/>crash or Ctrl-C: the orchestrator's `with Tracer` writes<br/>run_end (crashed / interrupted), then the exception continues
```

---

## 3. Inside `Toolset.execute`: the one checkpoint

Every tool call goes through these checks in order. **Nothing here raises:** each
failure becomes text the model reads next turn, so it can fix its own mistake.
Malformed calls are rejected *before* a human is ever asked.

```mermaid
flowchart TD
    start(["execute(name, arguments_json)"]) --> known{"name in this<br/>agent's toolset?"}
    known -->|no| e1["Error: unknown tool<br/>+ list of available tools"]
    known -->|yes| parse{"arguments are<br/>valid JSON?"}
    parse -->|no| e2["Error: not valid JSON<br/>+ example of valid JSON"]
    parse -->|yes| obj{"a JSON object?"}
    obj -->|no| e3["Error: must be a JSON object"]
    obj -->|yes| perm{"tool.permission"}
    perm -->|allow| runfn
    perm -->|ask| approver{"approver"}
    approver -->|"terminal: you typed y"| runfn
    approver -->|"terminal: anything else"| r1["The user declined this call.<br/>Try a different approach."]
    approver -->|"deny_all (--unattended)<br/>or no terminal"| r2["Unavailable in this run.<br/>Don't retry; say what you couldn't do."]
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
| `read_file` | allow | Return a file's text | Paths can't leave `workspace/`; binary/empty files say so; long output truncated (head + tail) |
| `write_file` | allow | Create or overwrite a file | Same path rule; creates parent folders |
| `bash` | **ask** | Run a shell command in `workspace/` | Approval first; stdin closed; timeout kills the whole process group; output truncated; exit code always reported |

An agent only gets the tools its `.md` file lists. A tool it wasn't given can't run,
even if the model invents a call to it.

---

## 5. What the model sees: the conversation grows

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

**Session vs. trace:** the session is the agent's *memory* (exactly what the model sees,
and later compaction may rewrite it); the trace is the *record* (everything, never rewritten).

```text
sessions/2026-10-20T14-13-59_base_8569/
  session.json      id, agent, agent_version, created: written once, never changed
  messages.jsonl    the conversation, one message per line
  turn-001.jsonl    trace of turn 1; the trace files are the only record of which turns happened
```

---

## 6. What a trace records

One JSONL file per turn, inside its session: `sessions/<id>/turn-001.jsonl` (gitignored).
The format is defined in one place, `tracer.py`: the agent reports what happened
(`start`, `model_response`, `tool_call`, `finish`), and the tracer decides what's recorded.
One tracer = one turn = one file. The orchestrator owns it with `with Tracer(path) as tracer:`,
so even a crash or Ctrl-C ends the trace with exactly one `run_end`.
Each line is flushed as it happens, so a crashed run still leaves its trace.
From the real primes run above:

```text
run_start       agent=base  approver=terminal_approver  tools=[read_file, write_file, bash]
model_response  step 1  7.0s  prompt=623 tokens  cached=0  out=204  -> write_file
                reasoning: "The user is asking me to write a primes.py ..."
tool_call       step 1  write_file -> "Wrote 381 characters to primes.py."
model_response  step 2  2.1s  prompt=816  cached=0  out=43  -> bash
tool_call       step 2  bash -> "[2, 3, 5, 7, 11, 13, 17, 19, 23, 29] (exit code 0)"
model_response  step 3  3.9s  prompt=903  cached=0  out=101  -> final answer
run_end         outcome=answered  steps=3  13.0s
```

| Event | When | Key fields |
|---|---|---|
| `run_start` | once, first | task, full config (incl. system prompt), tool schemas, approver, **provenance** (below) |
| `model_response` | every model call | message, **reasoning** (dropped from history, kept here), finish_reason, model (as reported by the server), usage (tokens, cache, oMLX timings), seconds |
| `tool_call` | every tool call | id, name, **raw** arguments string, result, seconds |
| `run_end` | once, last | outcome: `answered` / `max_steps` / `crashed` / `interrupted`; answer or error; steps; seconds |

**Provenance**: what produced a run, so runs can be filtered later (evals, fine-tuning data).
A hash *identifies* content; it doesn't store it.

| Field | Meaning | Recoverable? |
|---|---|---|
| `agent_version` | 12-char hash of the agent definition (name excluded): same content, same version | Yes: the full definition is in `run_start.config` |
| `harness.commit` / `harness.dirty` | git commit of the harness code; whether there were uncommitted changes | Via `git checkout`, only if `dirty` is false |
| `config.model` (+ `model` per response) | the model asked for, and the one the server says answered | Only if the same weights still exist |

---

## 7. How the code is organized

Arrows mean "imports". They only point one way: nothing imports the orchestrator,
tools never import the machinery around them, and `policy.py` imports nothing at runtime.

```mermaid
flowchart TD
    orch["orchestrator.py<br/>control plane, CLI"] --> agent["agent.py<br/>the loop"]
    orch --> loader["config_loader.py<br/>reads agents/*.md"]
    orch --> sessionmod["session.py<br/>conversation on disk"]
    agent --> sessionmod
    orch --> policy["policy.py<br/>Policy, approvers"]
    orch --> tools_api["tools/__init__.py<br/>public API"]
    orch --> tracer["tracer.py<br/>JSONL events"]
    agent --> tracer
    tracer --> prov["provenance.py<br/>agent hash, git state"]
    agent --> tools_api
    tools_api --> toolset["tools/toolset.py<br/>REGISTRY + Toolset"]
    toolset --> policy
    toolset --> builtin["tools/builtin/<br/>fs.py, shell.py"]
    toolset --> base["tools/base.py<br/>Tool, WORKSPACE, truncate"]
    builtin --> base
    base -.->|"Permission type only"| policy
```

---

## Not built yet

What's planned, in rough order. Each becomes a diagram change when it lands.

- **`harness chat` / resume:** many turns in one session, continued later (the next step).
- **Reading traces:** `harness show <trace>` to print a past run readably, and a summary line (steps, tokens, cache hits, time) at the end of each run.
- **More tools:** `list_dir`, `edit`, eventually `web_search` and MCP servers.
- **Sandbox:** `bash` runs in a container; then a `--yes` mode can auto-approve safely.
- **Context management:** compaction when the conversation nears the 32k window.
- **Provider adapters:** oMLX and native Anthropic behind one interface.
