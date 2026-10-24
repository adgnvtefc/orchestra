"""Control plane: load config, build what the agent needs, hand it over, run.

The orchestrator wires things together and does no work itself. It never reads from or
prints to the terminal: interfaces (interfaces/cli.py now, maybe a web UI later) drive it through
new_conversation() / resume_conversation() and Conversation.turn(), and supply the
approver, which is how *they* ask a human. Nothing here imports an interface.
"""

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from orchestra.agent import run_agent
from orchestra.config import Project, RunConfig, resolve_agent, resolve_run_config
from orchestra.environment import provision
from orchestra.policy import Approver, Policy, deny_all, require_sandbox_for_auto_approval
from orchestra.provenance import agent_version
from orchestra.sandbox import Seatbelt, Unsandboxed, default_sandbox
from orchestra.providers import make_client
from orchestra.session import Session
from orchestra.tools import ToolContext, Toolset
from orchestra.tracer import Tracer

# Where sessions (conversations, traces, default workspaces) are stored. Deliberately *outside*
# this project: tools that walk up looking for their config (git, pytest, uv) would otherwise
# find the harness's own .git and pyproject.toml from inside every workspace.
# (~/projects/orchestra-sessions is a symlink here, for opening it in an editor.)
SESSIONS_DIR = Path.home() / ".orchestra" / "sessions"


@dataclass(frozen=True)
class TurnResult:
    answer: str
    run_end: dict | None  # the trace's run_end: outcome and totals (steps, tool_calls, tokens, seconds)


class Conversation:
    """A session plus everything its turns need. What an interface talks to."""

    def __init__(self, session: Session, agent: dict, client: OpenAI, toolset: Toolset):
        self.session = session
        self.agent = agent  # the resolved agent config each turn runs with
        self.client = client
        self.toolset = toolset

    def turn(self, task: str, on_event: Callable[[dict], None] | None = None) -> TurnResult:
        """Run one turn. Its trace file is created first, so the turn counts even if it crashes.

        `on_event` is called with each trace event as it happens: how the interface shows
        progress (the same events the trace file gets, nothing extra).
        """
        # creates this turn's trace file: sessions/<id>/turn-001.jsonl
        listeners = [on_event] if on_event else []
        with Tracer(self.session.trace_path(self.session.next_turn), listeners) as tracer:
            answer = run_agent(task, self.session, self.agent, self.client, self.toolset, tracer)
        last = tracer.events[-1] if tracer.events else None
        return TurnResult(answer, last if last and last["event"] == "run_end" else None)

    def repair(self) -> None:
        """After a turn was interrupted: make the history valid to send again."""
        self.session.repair()


def new_conversation(
    agent_name: str = "base",
    *,
    project: Project | None = None,
    approver: Approver = deny_all,
    sandbox: Seatbelt | Unsandboxed | None = None,
    workspace: Path | None = None,
    environment: str | None = None,
) -> Conversation:
    """Start a new session for `agent_name`, set its workspace up, and get it ready for turns.

    `approver`: how "ask" tools get answered. The default, deny_all, is the safe one: an
    interface that can ask a human passes its own (interfaces/cli.py passes terminal_approver).
    approve_all is refused unless sandboxed.
    `sandbox`: how bash commands are confined; None means default_sandbox() (Seatbelt on macOS).
    `workspace` / `environment`: override the agent file's; see config.
    `project`: where agents/ and environments/ are, plus the project's own tools; None means
    Project.find() from the current directory, the way git finds .git.
    """
    # Check everything first: a bad combination or a typo should fail before any folder is created.
    sandbox = sandbox or default_sandbox()
    require_sandbox_for_auto_approval(approver, sandbox)
    project = project or Project.find(Path.cwd())
    rc = resolve_run_config(agent_name, project, workspace=workspace, environment=environment)
    client = make_client()
    session = start_session(rc, project)
    toolset = build_toolset(session, rc.agent.get("tools", []), approver, sandbox, project)
    return Conversation(session, rc.agent, client, toolset)


def resume_conversation(
    session_id: str,
    *,
    project: Project | None = None,
    approver: Approver = deny_all,
    sandbox: Seatbelt | Unsandboxed | None = None,
) -> Conversation:
    """Reopen a saved session and get it ready for more turns.

    The session keeps what it was created with (system prompt, workspace, environment);
    model, temperature, max_steps, and tools come from the agent file as it is *now*,
    and each turn's trace records exactly what that turn used.

    `project` supplies the project's own tools. The agent file is always read from the
    folder the session was created in (wherever you resume from); only sessions from before
    that was recorded fall back to `project`'s root, or Project.find() from here.
    """
    sandbox = sandbox or default_sandbox()
    require_sandbox_for_auto_approval(approver, sandbox)
    session = Session.load(SESSIONS_DIR, session_id)  # also repairs an interrupted history
    caller = project or Project.find(session.config_root or Path.cwd())
    project = Project(session.config_root or caller.root, caller.extra_tools)
    agent = resolve_agent(session.agent, project)
    agent["environment"] = session.environment  # what the workspace was set up with, not today's file
    client = make_client()
    toolset = build_toolset(session, agent.get("tools", []), approver, sandbox, project)
    return Conversation(session, agent, client, toolset)


def session_exists(session_id: str) -> bool:
    return (SESSIONS_DIR / session_id / "session.json").is_file()


def run(task: str, agent_name: str = "base", **options) -> str:
    """One task as a new one-turn session; returns the answer. `options` as for new_conversation."""
    return new_conversation(agent_name, **options).turn(task).answer


def start_session(rc: RunConfig, project: Project) -> Session:
    """Create a session for `rc` and set its workspace up. The system prompt is pinned here.

    If setup fails, the session is removed: one that never got set up has nothing worth keeping.
    """
    session = Session.create(
        SESSIONS_DIR, rc.agent["name"], rc.system_prompt, agent_version(rc.agent), rc.workspace, rc.environment,
        config_root=project.root,
    )
    if rc.environment:
        try:
            provision(rc.environment, session.workspace)
        except Exception:
            shutil.rmtree(session.dir)
            raise
    return session


def build_toolset(
    session: Session, tools: list[str], approver: Approver, sandbox: Seatbelt | Unsandboxed, project: Project
) -> Toolset:
    """The tools for a turn of `session`: confined to its workspace, with its environment's variables.

    Built from the session (not the config) so a resumed session works on the same folder
    with the same variables it started with.
    """
    # the interface chose whether to sandbox; the session's environment says what its commands need
    context = ToolContext(
        workspace=session.workspace, env=session.environment_variables, sandbox=sandbox.with_rules(session.sandbox_rules)
    )
    return Toolset(tools, Policy(approver), context, project.tools())
