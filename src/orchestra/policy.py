"""Policy: decides whether a tool call may run. Toolset.execute is where it's enforced.

Each tool declares a permission: "allow" runs, "ask" goes to the approver. The
approver is chosen per run, by the interface: interfaces/cli.py's terminal_approver asks a human
at the terminal; deny_all (here) says no when nobody's there.

Every check returns a Decision saying who decided and how, one of four Approval values:
auto-allow (the tool is "allow", or approve_all), auto-deny (deny_all, or a human approver
that found nobody to ask), human-allow, human-deny. Tool records keep it: whether a person
approved a call is data you can't reconstruct later.
Whether a risky tool is safe depends on where it runs, so that choice belongs to
the run, not to the agent. (To keep an agent away from a tool, don't list it.)

This module imports nothing from the rest of the harness at runtime, so anything
can import it without creating an import cycle.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal

if TYPE_CHECKING:  # only for type hints; never imported at runtime
    from orchestra.tools.base import Tool

# allow: run without asking.  ask: the approver decides.
Permission = Literal["allow", "ask"]

# (tool name, parsed arguments) -> None if approved, or a refusal message for the model.
# The approver writes the message because only it knows *why* it said no: a human
# declined, or nobody was there to ask. One that asks a person is marked @asks_human.
Approver = Callable[[str, dict], str | None]

# Who decided whether a tool call ran, and what they decided.
Approval = Literal["auto-allow", "auto-deny", "human-allow", "human-deny"]


@dataclass(frozen=True)
class Decision:
    approval: Approval
    refusal: str | None  # None: the call may run. Otherwise the message the model gets instead.


class NobodyToAsk(Exception):
    """Raised by an @asks_human approver that finds no person to ask (no terminal attached).
    Policy then refuses the call the way deny_all would, and records it as auto-deny."""


def asks_human(approver: Approver) -> Approver:
    """Mark `approver` as one that asks a person, so its answers are recorded as human-allow /
    human-deny. Unmarked approvers (deny_all, approve_all, a script's own) count as automatic."""
    approver.asks_human = True
    return approver


def deny_all(name: str, arguments: dict) -> str | None:
    """Unattended runs: there's nobody to ask, so the answer is no.

    The message tells the model the situation won't change, so it shouldn't retry
    or offer to ask anyone, and should be upfront about what it couldn't do.
    """
    return (
        f"{name} is unavailable in this run: it needs human approval and nobody is available to give it. "
        f"Don't retry it. Finish without it if you can, and say clearly what you couldn't do."
    )


def approve_all(name: str, arguments: dict) -> str | None:
    """--yes: approve every call without asking. Only allowed with a sandbox (see
    require_sandbox_for_auto_approval): the sandbox, not a person, is then the safety net."""
    return None


def require_sandbox_for_auto_approval(approver: Approver, sandbox) -> None:
    """Refuse approve_all without a sandbox: nothing would stand between the model and your machine.

    The one place this rule lives; the CLI calls it for a friendly error before anything
    happens, and the orchestrator calls it again so code that skips the CLI can't bypass it.
    (`sandbox` is anything with a .name, "none" meaning unsandboxed: no import needed here.)
    """
    if approver is approve_all and sandbox.name == "none":
        raise ValueError("auto-approval (--yes) needs the sandbox; it can't be combined with no sandbox")


class Policy:
    def __init__(self, approver: Approver, name: str | None = None):
        """`name` labels the policy in traces; it defaults to the approver function's name
        ("terminal_approver", "deny_all"). Pass one explicitly for an approver without a
        meaningful name, like a lambda."""
        self.approver = approver
        self.name = name or approver.__name__

    def check(self, tool: "Tool", arguments: dict) -> Decision:
        """May this call run, and who decided? Decision.refusal is None if it may."""
        if tool.permission == "allow":
            return Decision("auto-allow", None)
        human = getattr(self.approver, "asks_human", False)
        try:
            refusal = self.approver(tool.name, arguments)
        except NobodyToAsk:
            human, refusal = False, deny_all(tool.name, arguments)
        return Decision(f"{'human' if human else 'auto'}-{'deny' if refusal else 'allow'}", refusal)
