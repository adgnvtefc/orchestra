"""Terminal interface: parse the command line, chat turn by turn, ask the human y/N.

Everything that reads from or prints to the terminal lives here. The orchestrator does
neither: this module drives it through new_conversation() / resume_conversation() and
Conversation.turn(), and hands it terminal_approver as the way to ask a human.

    orchestra "task"                  one task, as a new one-turn session (same as `orchestra run "task"`)
    orchestra chat                    a conversation, turn by turn
    orchestra chat --resume <id>      continue a saved conversation
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from orchestra.config import Project, ProjectToolsError
from orchestra.orchestrator import Conversation, new_conversation, resume_conversation, session_exists
from orchestra.policy import approve_all, deny_all, require_sandbox_for_auto_approval
from orchestra.providers import MissingAPIKey
from orchestra.sandbox import Unsandboxed, default_sandbox


# this approver is currently only called from within the policy class
def terminal_approver(name: str, arguments: dict) -> str | None:
    """Ask the human at the terminal. Only an explicit "y" approves.

    Everything goes to stderr, prompt included (input()'s own prompt would go to stdout):
    stdout is only the answer, so `answer=$(orchestra ...)` still shows you the question.
    """
    print(f"\n  agent wants to run {name}", file=sys.stderr)
    for key, value in arguments.items():
        text = str(value)
        print(f"    {key}: {text if len(text) <= 300 else text[:300] + '...'}", file=sys.stderr)
    print("  allow? [y/N] ", end="", file=sys.stderr, flush=True)
    try:
        answer = input().strip().lower()
    except EOFError:  # no terminal attached (e.g. piped input): same situation as unattended
        return deny_all(name, arguments)
    if answer == "y":
        return None
    return f"The user declined this {name} call. Try a different approach or ask the user."


def show_step(event: dict) -> None:
    """Print each tool call as it finishes: what the agent ran, and the start of what it got back.
    To stderr: progress, not the answer.

    Listens to the turn's trace events (Conversation.turn's on_event); everything else
    (model responses, run_start/run_end) isn't shown step by step.
    """
    if event["event"] != "tool_call":
        return
    print(f"[step {event['step']}] {event['name']}({event['arguments']})", file=sys.stderr)
    result = event["result"]
    print(f"  -> {result if len(result) <= 200 else result[:200] + '...'}", file=sys.stderr)


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


def chat(conv: Conversation) -> None:
    """Talk to the agent turn by turn, until /exit, Ctrl-D, or Ctrl-C at the prompt.

    Ctrl-C *during* a turn stops just that turn: the history is repaired (an unanswered
    tool call would make the next request fail) and you get the prompt back. So does an
    error, like the model server going away.
    """
    session = conv.session
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
            result = conv.turn(task, on_event=show_step)
        except KeyboardInterrupt:
            conv.repair()
            print("\n(turn interrupted)\n")
            continue
        except Exception as e:  # keep the conversation alive; the trace has the details
            conv.repair()
            print(f"\n(turn failed: {type(e).__name__}: {e})\n", file=sys.stderr)
            continue
        if result.run_end:
            print(turn_summary(result.run_end), file=sys.stderr)
        print(f"\n{result.answer}\n")
    print(f"Session saved. Continue it with: orchestra chat --resume {session.id}")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point (see the module docstring). `argv` defaults to the real command line;
    tests pass their own list."""
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] not in ("run", "chat", "-h", "--help"):
        argv = ["run", *argv]  # plain `orchestra "task"` keeps working

    # Options for setting up a new session, shared by both commands.
    new_session = argparse.ArgumentParser(add_help=False)
    new_session.add_argument("--agent", help="which agent in agents/ to run (default: base)")
    new_session.add_argument(
        "--config-root",
        metavar="PATH",
        help="the folder with agents/ and environments/ (default: the nearest one at or above the current folder)",
    )
    # How "ask" tools get answered: a y/N prompt by default, or one of these (not both).
    approval = new_session.add_mutually_exclusive_group()
    approval.add_argument(
        "--unattended",
        action="store_true",  # a flag: present means True, absent means False
        help="never prompt; tools that would ask for approval are declined",
    )
    approval.add_argument(
        "--yes",
        action="store_true",
        help="never prompt; approve every tool call (only with the sandbox: it's what keeps you safe then)",
    )
    new_session.add_argument(
        "--no-sandbox",
        action="store_true",
        help="run bash commands unconfined, with your full permissions (default: sandboxed to the workspace on macOS)",
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
    parser = argparse.ArgumentParser(prog="orchestra", description="Run an agent.")
    commands = parser.add_subparsers(dest="command", required=True)
    run_cmd = commands.add_parser("run", parents=[new_session], help="run one task as a new session")
    # nargs="+" collects every remaining word, so quotes around the task are optional
    run_cmd.add_argument("task", nargs="+", help="what the agent should do")
    chat_cmd = commands.add_parser("chat", parents=[new_session], help="talk to an agent, turn by turn")
    chat_cmd.add_argument("--resume", metavar="SESSION_ID", help="continue a saved session")
    args = parser.parse_args(argv)
    # The terminal's way of supplying the API key: a .env file (here or above). The harness
    # itself only reads environment variables, so programs using it as a library choose their own.
    load_dotenv()

    # The terminal decides how to ask a human; the orchestrator just uses whatever it's given.
    approver = deny_all if args.unattended else approve_all if args.yes else terminal_approver
    sandbox = Unsandboxed() if args.no_sandbox else default_sandbox()
    try:
        require_sandbox_for_auto_approval(approver, sandbox)  # fail here, before anything is created
    except ValueError as e:
        (run_cmd if args.command == "run" else chat_cmd).error(str(e))
    workspace = Path(args.workspace).expanduser() if args.workspace else None
    command = run_cmd if args.command == "run" else chat_cmd

    if not getattr(args, "resume", None):  # a new session: find its config, check the agent name
        try:
            project = Project.at(Path(args.config_root).expanduser()) if args.config_root else Project.find(Path.cwd())
        except FileNotFoundError as e:
            command.error(f"{e}. Run from a project with agents/, or pass --config-root")
        except ProjectToolsError as e:
            command.error(str(e))
        agents = project.agents()
        if (args.agent or "base") not in agents:  # a typo fails here, with the list of real ones
            command.error(f"argument --agent: invalid choice: {args.agent or 'base'!r} (choose from {', '.join(agents)})")
        new = dict(project=project, approver=approver, sandbox=sandbox, workspace=workspace, environment=args.environment)

    if args.command == "run":
        task = " ".join(args.task).strip()
        if not task:
            run_cmd.error("the task is empty")  # prints usage and exits, like any other bad argument

    if getattr(args, "resume", None):
        # A session's agent, workspace, and environment are fixed when it's created.
        fixed = [flag for flag, value in (("--agent", args.agent), ("--workspace", workspace),
                                          ("--environment", args.environment), ("--config-root", args.config_root)) if value]
        if fixed:
            chat_cmd.error(f"{', '.join(fixed)} can't be changed when resuming: the session keeps what it started with")
        if not session_exists(args.resume):
            chat_cmd.error(f"no session {args.resume!r}")

    # The one place a conversation is opened; problems become usage errors, not tracebacks.
    try:
        if getattr(args, "resume", None):
            conv = resume_conversation(args.resume, approver=approver, sandbox=sandbox)  # per run, not fixed
        else:
            conv = new_conversation(args.agent or "base", **new)
    except (MissingAPIKey, ProjectToolsError) as e:  # ProjectToolsError: a resumed session's project tools.py
        command.error(str(e))

    if args.command == "run":
        try:
            result = conv.turn(task, on_event=show_step)
        finally:
            # stderr, so stdout stays just the answer (useful when scripts capture it); printed
            # even after a crash, so you can find the trace
            print(f"session: {conv.session.id}  trace: {conv.session.trace_path(1).name}", file=sys.stderr)
        if result.run_end:
            print(turn_summary(result.run_end), file=sys.stderr)
        print(result.answer)
        return
    chat(conv)
