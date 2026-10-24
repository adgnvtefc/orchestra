# orchestra

This is a learning project: an agent harness built from scratch, running local models on
Apple Silicon. It runs an agent loop (model ↔ tools) with sessions, traces, environments,
and a macOS sandbox. For how it works inside, see [visual_representation.md](visual_representation.md).

## Requirements

- **macOS** (the sandbox uses macOS's `sandbox-exec`; on other systems bash runs unsandboxed)
- **[uv](https://docs.astral.sh/uv/)** for Python and dependencies
- **A local model server speaking the OpenAI chat-completions API**, by default oMLX
  at `http://127.0.0.1:8000/v1`, with the model named in the agent file loaded
  (`agents/base.md` uses `Qwen3.8-27B-MLX-8bit`)

## Setup

```bash
uv sync                      # install dependencies
cp .env.example .env         # then put your oMLX API key in .env
uv run orchestra "What is 6 times 7?"
```

`.env` holds:

| Variable | Required | Meaning |
|---|---|---|
| `OMLX_API_KEY` | yes | the model server's API key |
| `OMLX_BASE_URL` | no | default `http://127.0.0.1:8000/v1` |

The agent's commands never see these: bash runs with an allowlisted environment.

## Using it

There are two commands. A plain `orchestra "task"` is short for `orchestra run "task"`.

```bash
orchestra "task"                       # one task: a new session with one turn, then exit
orchestra chat                         # a conversation: type a task, get an answer, repeat
orchestra chat --resume SESSION_ID     # continue a saved conversation
```

(Inside this repo, prefix with `uv run`, e.g. `uv run orchestra chat`.)

**Where config comes from:** the harness looks for the nearest folder, at or above where you
run it, that has an `agents/` folder (the way git finds `.git`). That folder is the *config
root*: agents come from its `agents/`, environments from its `environments/`. So run it from
this repo (or any folder inside it), or from another project with its own `agents/`, or pass
`--config-root PATH`. From a folder with no `agents/` above it, it tells you so.

### Options

Both `run` and `chat` take these:

| Option | Default | Meaning |
|---|---|---|
| `--agent NAME` | `base` | which agent in `agents/` to run |
| `--config-root PATH` | nearest folder with `agents/` | where `agents/` and `environments/` are |
| `--unattended` | off | never prompt; tool calls that need approval are **declined** |
| `--yes` | off | never prompt; tool calls are **approved**. Only allowed with the sandbox on |
| `--no-sandbox` | sandboxed | run bash commands with your full permissions |
| `--workspace PATH` | a fresh folder | an existing folder for the agent to work in, instead of a new empty one |
| `--environment NAME` | the agent file's | an environment from `environments/` to set the workspace up with, or `none` |

`chat` also takes `--resume SESSION_ID`. A resumed session keeps the agent (read from the
project it was created in, wherever you resume from), workspace, and environment it started
with, so `--agent`, `--config-root`, `--workspace`, and `--environment` can't be used with
it; `--unattended`/`--yes`/`--no-sandbox` can, since they apply per run.

`--unattended` and `--yes` can't be combined. `--yes` with `--no-sandbox` is refused.

### Examples

```bash
# Interactive: you approve each bash command (y/N)
uv run orchestra "Write primes.py that prints the first 10 primes and run it"

# Fully unattended, safe because bash is sandboxed to the session's workspace
uv run orchestra --yes "Create dates.py with a parse_date function, write pytest tests, run them"

# Unattended, no bash at all (file tools still work)
uv run orchestra --unattended "Create notes.md listing three facts about primes"

# Work on an existing folder (the agent can edit its files)
uv run orchestra --workspace ~/projects/some-app "Find and fix the failing test"

# A plain workspace, no Python environment set up
uv run orchestra --environment none "What files are in the workspace?"

# A conversation, continued later
uv run orchestra chat
uv run orchestra chat --resume 2026-10-21T18-55-51_base_66ec

# Capture only the answer in a script (everything else goes to stderr)
answer=$(uv run orchestra --yes "What is 17 * 23? Just the number." 2>/dev/null)
```

### What you see

- **stdout:** the final answer, and nothing else (so scripts can capture it).
- **stderr:** each tool call as it finishes (`[step N] tool(args)` and the start of its
  result), a summary line per turn (`(3 model calls · 2 tool calls · 1,904 prompt tokens,
  650 cached · 4.2s)`), and for one-off runs `session: <id>  trace: turn-001.jsonl`.
- **Approval prompts** (without `--yes`/`--unattended`): bash asks `allow? [y/N]`. Only `y` approves.
- **In chat:** `/exit`, Ctrl-D, or Ctrl-C at the prompt leaves and prints the resume command.
  Ctrl-C *during* a turn stops only that turn.
- **Errors:** bad arguments exit with code 2 and a usage message.

## Safety

| | Runs without asking | Confined by |
|---|---|---|
| `read_file`, `write_file` | yes | a path check: nothing outside the workspace |
| `bash` | no: asks y/N (or `--yes` / `--unattended`) | the **sandbox** (on by default on macOS) |

The sandbox lets bash commands write only inside the workspace, read nothing in your home
folder except the workspace (plus what the session's environment lists under `sandbox:`,
e.g. uv's Pythons), and use the network only if the environment allows it. You decide
*whether* there's a sandbox (`--no-sandbox`); the environment decides *what's allowed* in it. Details: section 6 of
[visual_representation.md](visual_representation.md).

## Config files

### Agents: `agents/<name>.md`

YAML settings between `---` lines, then the system prompt:

```markdown
---
model: Qwen3.8-27B-MLX-8bit      # required: model name on the server
temperature: 0.6                 # required
max_steps: 20                    # required: model calls per turn before giving up
tools:                           # optional: tools this agent may use (none if absent)
  - read_file
  - write_file
  - bash
environment: python-dev          # optional: set the workspace up from environments/python-dev.yaml
workspace: ~/scratch/research    # optional: always work in this existing folder
---
You are a helpful AI agent.      # the system prompt (everything below the second ---)
```

- Available tools: `read_file`, `write_file`, `bash`. An unknown name is an error before anything runs.
- A `workspace:` path may use `~` and is relative to this project's root.
- `uv run pytest tests/test_agent_files.py` checks every agent file.

### Environments: `environments/<name>.yaml`

How a session's workspace is set up before the agent starts. Setup runs without approval:
it comes from your config, not the model.

```yaml
description: Python 3.12 virtualenv with pytest     # optional
files: templates/python-dev                          # optional: a folder copied into the workspace (never overwrites)
setup:                                               # optional: commands run in the workspace, in order
  - uv venv --quiet --python 3.12 .venv
  - uv pip install --quiet --python .venv pytest
env:                                                 # optional: extra variables for the agent's commands
  PATH: "{workspace}/.venv/bin"                      #   PATH is prepended to; {workspace} is filled in
  VIRTUAL_ENV: "{workspace}/.venv"
prompt: |                                            # optional: appended to the agent's system prompt
  Your workspace has a Python 3.12 virtualenv in .venv with pytest installed.
sandbox:                                             # optional: what its commands need inside the sandbox
  readable: [~/.local/share/uv]                      #   folders in your home they may read (here: .venv's Python)
  network: false                                     #   outbound internet (default false)
```

Only these keys are allowed (a typo is an error). A workspace already set up with a
different environment is refused, never set up again over the top.

### Project tools: `tools.py`

A project can add its own tools by putting a `tools.py` next to its `agents/` folder that
defines `TOOLS`, a list of `orchestra.tools.Tool`. The CLI (and `Project.find` / `Project.at`
from Python) load it automatically, so an agent can list those tools by name like built-ins:

```python
# tools.py, in your project's root
from orchestra.tools import Tool

def stock_price(ctx, ticker: str) -> str:
    return lookup_price(ticker)

TOOLS = [Tool(fn=stock_price, permission="allow", schema={"type": "function", "function": {
    "name": "stock_price", "description": "Latest price for a ticker.",
    "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]}}})]
```

- A tool function takes `ctx` (its workspace, environment, sandbox) first, then the arguments the model sends.
- `permission="ask"` makes the CLI ask y/N before each call (or follow `--yes` / `--unattended`).
- A name already used by a built-in is an error. A broken `tools.py` is reported with its path.
- **`tools.py` is code you trust,** like the agent files: it runs inside the harness (the
  sandbox confines bash only), the way pytest runs a `conftest.py`. Keep it self-contained or
  import installed packages; the project folder isn't added to Python's import path.

## Where things go

Sessions live in `~/.orchestra/sessions/` (also reachable as `~/projects/orchestra-sessions`),
deliberately outside this project:

```text
~/.orchestra/sessions/2026-10-21T18-55-51_base_66ec/
  session.json       id, agent, agent version, workspace, environment, created
  messages.jsonl     the conversation: exactly what the model sees each turn
  turn-001.jsonl     the trace of turn 1: every model call and tool call, with timings and tokens
  turn-002.jsonl     ...
  workspace/         where the agent works (unless --workspace / workspace: chose another folder)
```

Each trace's first line records what produced it: the agent version (a hash of its
config), which orchestra code ran (its version, how it was installed, and its git commit when
knowable: exact for a source checkout or a git install), the approver,
and the sandbox.

## Using it from Python, with your own tools

Another project (or a script) can drive the harness directly and add its own tools,
without changing anything here. Tools are `Tool` objects: a function taking the context
first, its JSON schema (what the model sees), and a permission. Agents list them by name.

```python
from pathlib import Path
from orchestra.config import Project
from orchestra.orchestrator import new_conversation
from orchestra.policy import approve_all
from orchestra.tools import Tool

def weather(ctx, city: str) -> str:
    return f"sunny in {city}"

WEATHER = Tool(
    fn=weather,
    schema={"type": "function", "function": {
        "name": "weather", "description": "Today's weather in a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    permission="allow",
)

# The project: where agents/ and environments/ are, plus its tools: the ones in its tools.py
# (loaded automatically) and any passed here. Project.find walks up from a folder;
# Project.at(Path(...), [WEATHER]) names the folder directly.
project = Project.find(Path.cwd(), extra_tools=[WEATHER])

conv = new_conversation(
    "forecaster",                         # agents/forecaster.md, with `tools: [weather]`
    project=project,                      # optional: default is Project.find(Path.cwd())
    approver=approve_all,                 # optional: default is deny_all; approve_all needs the sandbox (on by default)
)
print(conv.turn("What's the weather in Paris?").answer)
print(conv.turn("And tomorrow?").answer)  # same session: it remembers the first turn
```

An extra tool can't reuse a built-in tool's name (that's an error, not a replacement).

As a library, the harness only *reads* `OMLX_API_KEY`/`OMLX_BASE_URL` from the environment;
loading a `.env` file is the CLI's job, so set them however your program likes. A missing
key raises `orchestra.providers.MissingAPIKey`, an ordinary exception your program can catch.
`resume_conversation(session_id, project=project)` continues a saved session (its agent file
always comes from the folder the session was created in).

## Development

```bash
uv run pytest                 # the test suite (no model needed; never touches real folders)
uv run pytest -m live         # also run real tasks against the model server
```

## Changelog

- 10/22: added functionality to import harness as library to external projects. NOTE: functionality as library has not been explicitly human tested yet. NOTE 2: TO NUKE CHANGES ROLLBACK TO LAST NON LIBRARY RELATED COMMIT.
- 10/21: added sessions, traces, chats, refactors for scalability
- 10/7: small refactor for scalable design in tools. added tests (fully ai generated, trusting that they work)
- 10/6: got harness running with minimal features. can answer simple queries.
