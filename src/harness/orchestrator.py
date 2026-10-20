"""Control plane: load config, build what the agent needs, hand it over, run.

The orchestrator wires things together and does no work itself. Everything else
is imported here; nothing imports the orchestrator.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from harness.agent import run_agent
from harness import config_loader
from harness.config_loader import resolve_run_config
from harness.environment import provision
from harness.policy import Policy, deny_all, terminal_approver
from harness.provenance import agent_version
from harness.session import Session
from harness.tools import ToolContext, Toolset
from harness.tracer import Tracer

# Where sessions (conversations, traces, default workspaces) are stored.
SESSIONS_DIR = config_loader.PROJECT_ROOT / "sessions"


def make_client() -> OpenAI:
    """Load .env and return an OpenAI client pointed at oMLX (OMLX_BASE_URL, OMLX_API_KEY)."""
    # reads .env, copies each KEY=value into process env, os.environ can see all
    load_dotenv()

    # get the api key
    api_key = os.environ.get("OMLX_API_KEY")
    if not api_key:
        raise SystemExit("OMLX_API_KEY is not set. Copy .env.example to .env and fill it in.")
    return OpenAI(
        base_url=os.environ.get("OMLX_BASE_URL", "http://127.0.0.1:8000/v1"),
        api_key=api_key,
        timeout=600,
    )


def run(
    task: str,
    agent_name: str = "base",
    unattended: bool = False,
    workspace: Path | None = None,
    environment: str | None = None,
) -> str:
    """Load the named agent's config, build what it needs, and run `task` as turn 1 of a new session.

    `unattended`: nobody is at the terminal, so every "ask" is answered no.
    `workspace`: a folder to work in, overriding the agent file's `workspace:`. With
    neither, the session gets its own fresh folder.
    `environment`: an environment name (or "none"), overriding the agent file's `environment:`.
    """
    # 1. Resolve and check all config first: a typo should fail before any folder is created.
    rc = resolve_run_config(agent_name, workspace=workspace, environment=environment)
    client = make_client()

    # 2. Create the session (the system prompt is pinned here) and set its workspace up.
    env = rc.environment
    session = Session.create(
        SESSIONS_DIR, agent_name, rc.system_prompt, agent_version(rc.agent), rc.workspace,
        environment={**rc.agent["environment"], "variables": env.env} if env else None,  # {workspace} unfilled
    )
    if env:
        try:
            provision(env, session.workspace)
        except Exception:
            shutil.rmtree(session.dir)  # a session that never got set up has nothing worth keeping
            raise

    # 3. Build what the turn needs and run it.
    policy = Policy(approver=deny_all if unattended else terminal_approver)
    context = ToolContext(workspace=session.workspace, env=session.environment_variables)
    toolset = Toolset(rc.agent.get("tools", []), policy, context)
    return run_turn(task, session, rc.agent, client, toolset)


def run_turn(task: str, session: Session, config: dict, client: OpenAI, toolset: Toolset) -> str:
    """Run one turn of `session`. Its trace file is created first, so the turn counts even if it crashes."""
    # creates this turn's trace file: sessions/<id>/turn-001.jsonl
    with Tracer(session.trace_path(session.next_turn)) as tracer:
        try:
            return run_agent(task, session, config, client, toolset, tracer)
        finally:
            # stderr, so stdout stays just the answer (useful when scripts capture it)
            print(f"session: {session.id}  trace: {tracer.path.name}", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: `uv run harness [--agent NAME] [--unattended] your task here`.

    `argv` defaults to the real command line; tests pass their own list.
    """
    agents = sorted(path.stem for path in config_loader.AGENTS_DIR.glob("*.md"))

    # this block is ai generated and i am trusting it works
    parser = argparse.ArgumentParser(prog="harness", description="Run an agent on a task.")
    # nargs="+" collects every remaining word, so quotes around the task are optional
    parser.add_argument("task", nargs="+", help="what the agent should do")
    # choices: a typo'd agent name fails right here, with the list of real ones
    parser.add_argument("--agent", default="base", choices=agents, help="which agent in agents/ to run (default: base)")
    parser.add_argument(
        "--unattended",
        action="store_true",  # a flag: present means True, absent means False
        help="never prompt; tools that would ask for approval are declined (for scripts and evals)",
    )
    parser.add_argument(
        "--workspace",
        help="an existing folder for the agent to work in (default: a fresh folder inside the session)",
    )
    parser.add_argument(
        "--environment",
        help="environment from environments/ to set the workspace up with, or 'none' (default: the agent file's)",
    )
    args = parser.parse_args(argv)

    task = " ".join(args.task).strip()
    if not task:
        parser.error("the task is empty")  # prints usage and exits, like any other bad argument
    workspace = Path(args.workspace).expanduser() if args.workspace else None
    print(run(task, agent_name=args.agent, unattended=args.unattended, workspace=workspace, environment=args.environment))
