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
from harness.config_loader import PROJECT_ROOT, RunConfig, list_agents, resolve_agent, resolve_run_config
from harness.environment import provision
from harness.policy import Policy, deny_all, terminal_approver
from harness.provenance import agent_version
from harness.session import Session
from harness.tools import ToolContext, Toolset
from harness.tracer import Tracer

# Where sessions (conversations, traces, default workspaces) are stored.
SESSIONS_DIR = PROJECT_ROOT / "sessions"


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
    # Resolve and check all config first: a typo should fail before any folder is created.
    rc = resolve_run_config(agent_name, workspace=workspace, environment=environment)
    client = make_client()
    session = start_session(rc)
    toolset = build_toolset(session, rc.agent.get("tools", []), unattended)
    try:
        return run_turn(task, session, rc.agent, client, toolset)
    finally:
        # stderr, so stdout stays just the answer (useful when scripts capture it); printed
        # even after a crash, so you can find the trace
        print(f"session: {session.id}  trace: {session.trace_path(1).name}", file=sys.stderr)


def start_session(rc: RunConfig) -> Session:
    """Create a session for `rc` and set its workspace up. The system prompt is pinned here.

    If setup fails, the session is removed: one that never got set up has nothing worth keeping.
    """
    session = Session.create(
        SESSIONS_DIR, rc.agent["name"], rc.system_prompt, agent_version(rc.agent), rc.workspace, rc.environment
    )
    if rc.environment:
        try:
            provision(rc.environment, session.workspace)
        except Exception:
            shutil.rmtree(session.dir)
            raise
    return session


def build_toolset(session: Session, tools: list[str], unattended: bool) -> Toolset:
    """The tools for a turn of `session`: confined to its workspace, with its environment's variables.

    Built from the session (not the config) so a resumed session works on the same folder
    with the same variables it started with.
    """
    policy = Policy(approver=deny_all if unattended else terminal_approver)
    context = ToolContext(workspace=session.workspace, env=session.environment_variables)
    return Toolset(tools, policy, context)


def run_turn(task: str, session: Session, config: dict, client: OpenAI, toolset: Toolset) -> str:
    """Run one turn of `session`. Its trace file is created first, so the turn counts even if it crashes."""
    # creates this turn's trace file: sessions/<id>/turn-001.jsonl
    with Tracer(session.trace_path(session.next_turn)) as tracer:
        answer = run_agent(task, session, config, client, toolset, tracer)
        if tracer.events and tracer.events[-1]["event"] == "run_end":
            print(turn_summary(tracer.events[-1]), file=sys.stderr)  # stderr: stdout stays just the answer
        return answer


def turn_summary(run_end: dict) -> str:
    """One line about what a turn cost: (3 model calls · 2 tool calls · 1,904 prompt tokens, 0 cached · 4.2s)"""
    parts = [_count(run_end["steps"], "model call"), _count(run_end["tool_calls"], "tool call")]
    tokens = run_end["tokens"]
    if tokens["prompt"]:  # a server that reports no usage gets no token figures
        parts.append(f"{tokens['prompt']:,} prompt tokens, {tokens['cached']:,} cached")
    parts.append(f"{run_end['seconds']:.1f}s")
    return f"({' · '.join(parts)})"


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


def resume_session(session_id: str) -> tuple[Session, dict]:
    """Reopen a saved session, and the agent settings its next turns run with.

    The session keeps what it was created with (system prompt, workspace, environment);
    model, temperature, max_steps, and tools come from the agent file as it is *now*,
    and each turn's trace records exactly what that turn used.
    """
    session = Session.load(SESSIONS_DIR, session_id)  # also repairs an interrupted history
    agent = resolve_agent(session.agent)
    agent["environment"] = session.environment  # what the workspace was set up with, not today's file
    return session, agent


def chat(session: Session, agent: dict, client: OpenAI, toolset: Toolset) -> None:
    """Talk to the agent turn by turn, until /exit, Ctrl-D, or Ctrl-C at the prompt.

    Ctrl-C *during* a turn stops just that turn: the history is repaired (an unanswered
    tool call would make the next request fail) and you get the prompt back. So does an
    error, like the model server going away.
    """
    turns = session.next_turn - 1
    print(f"session {session.id}  ({'resumed, ' + str(turns) + ' turns so far' if turns else 'new'})")
    print(f"workspace {session.workspace}")
    print("Type /exit or press Ctrl-D to leave.\n")
    while True:
        try:
            task = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):  # Ctrl-D / Ctrl-C at the prompt: leave
            print()
            break
        if not task:
            continue
        if task in ("/exit", "/quit"):
            break
        try:
            answer = run_turn(task, session, agent, client, toolset)
        except KeyboardInterrupt:
            session.repair()
            print("\n(turn interrupted)\n")
            continue
        except Exception as e:  # keep the conversation alive; the trace has the details
            session.repair()
            print(f"\n(turn failed: {type(e).__name__}: {e})\n", file=sys.stderr)
            continue
        print(f"\n{answer}\n")
    print(f"Session saved. Continue it with: harness chat --resume {session.id}")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point.

        harness "task"                  one task, as a new one-turn session (same as `harness run "task"`)
        harness chat                    a conversation, turn by turn
        harness chat --resume <id>      continue a saved conversation

    `argv` defaults to the real command line; tests pass their own list.
    """
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] not in ("run", "chat", "-h", "--help"):
        argv = ["run", *argv]  # plain `harness "task"` keeps working

    # Options for setting up a new session, shared by both commands.
    new_session = argparse.ArgumentParser(add_help=False)
    # choices: a typo'd agent name fails right here, with the list of real ones
    new_session.add_argument("--agent", choices=list_agents(), help="which agent in agents/ to run (default: base)")
    new_session.add_argument(
        "--unattended",
        action="store_true",  # a flag: present means True, absent means False
        help="never prompt; tools that would ask for approval are declined (for scripts and evals)",
    )
    new_session.add_argument(
        "--workspace",
        help="an existing folder for the agent to work in (default: a fresh folder inside the session)",
    )
    new_session.add_argument(
        "--environment",
        help="environment from environments/ to set the workspace up with, or 'none' (default: the agent file's)",
    )

    # this block is ai generated and i am trusting it works
    parser = argparse.ArgumentParser(prog="harness", description="Run an agent.")
    commands = parser.add_subparsers(dest="command", required=True)
    run_cmd = commands.add_parser("run", parents=[new_session], help="run one task as a new session")
    # nargs="+" collects every remaining word, so quotes around the task are optional
    run_cmd.add_argument("task", nargs="+", help="what the agent should do")
    chat_cmd = commands.add_parser("chat", parents=[new_session], help="talk to an agent, turn by turn")
    chat_cmd.add_argument("--resume", metavar="SESSION_ID", help="continue a saved session")
    args = parser.parse_args(argv)

    workspace = Path(args.workspace).expanduser() if args.workspace else None
    if args.command == "run":
        task = " ".join(args.task).strip()
        if not task:
            run_cmd.error("the task is empty")  # prints usage and exits, like any other bad argument
        print(run(task, agent_name=args.agent or "base", unattended=args.unattended, workspace=workspace,
                  environment=args.environment))
        return

    if args.resume:
        # A session's agent, workspace, and environment are fixed when it's created.
        fixed = [flag for flag, value in (("--agent", args.agent), ("--workspace", workspace),
                                          ("--environment", args.environment)) if value]
        if fixed:
            chat_cmd.error(f"{', '.join(fixed)} can't be changed when resuming: the session keeps what it started with")
        if not (SESSIONS_DIR / args.resume / "session.json").is_file():
            chat_cmd.error(f"no session {args.resume!r} in {SESSIONS_DIR}")
        session, agent = resume_session(args.resume)
    else:
        rc = resolve_run_config(args.agent or "base", workspace=workspace, environment=args.environment)
        session, agent = start_session(rc), rc.agent
    client = make_client()
    chat(session, agent, client, build_toolset(session, agent.get("tools", []), args.unattended))
