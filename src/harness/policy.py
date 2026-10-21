"""Policy: decides whether a tool call may run. Toolset.execute is where it's enforced.

Each tool declares a permission: "allow" runs, "ask" goes to the approver. The
approver is chosen per run: a human at the terminal, or "no" when nobody's there.
Whether a risky tool is safe depends on where it runs, so that choice belongs to
the run, not to the agent. (To keep an agent away from a tool, don't list it.)

This module imports nothing from the rest of the harness at runtime, so anything
can import it without creating an import cycle.
"""

from typing import TYPE_CHECKING, Callable, Literal

if TYPE_CHECKING:  # only for type hints; never imported at runtime
    from harness.tools.base import Tool

# allow: run without asking.  ask: the approver decides.
Permission = Literal["allow", "ask"]

# (tool name, parsed arguments) -> None if approved, or a refusal message for the model.
# Same convention as Policy.check. The approver writes the message because only it
# knows *why* it said no: a human declined, or nobody was there to ask.
Approver = Callable[[str, dict], str | None]


# this approver is currently only called from within the policy class
def terminal_approver(name: str, arguments: dict) -> str | None:
    """Ask the human at the terminal. Only an explicit "y" approves."""
    print(f"\n  agent wants to run {name}")
    for key, value in arguments.items():
        text = str(value)
        print(f"    {key}: {text if len(text) <= 300 else text[:300] + '...'}")
    try:
        answer = input("  allow? [y/N] ").strip().lower()
    except EOFError:  # no terminal attached (e.g. piped input): same situation as unattended
        return deny_all(name, arguments)
    if answer == "y":
        return None
    return f"The user declined this {name} call. Try a different approach or ask the user."


def deny_all(name: str, arguments: dict) -> str | None:
    """Unattended runs: there's nobody to ask, so the answer is no.

    The message tells the model the situation won't change, so it shouldn't retry
    or offer to ask anyone, and should be upfront about what it couldn't do.
    """
    return (
        f"{name} is unavailable in this run: it needs human approval and nobody is available to give it. "
        f"Don't retry it. Finish without it if you can, and say clearly what you couldn't do."
    )


class Policy:
    def __init__(self, approver: Approver, name: str | None = None):
        """`name` labels the policy in traces; it defaults to the approver function's name
        ("terminal_approver", "deny_all"). Pass one explicitly for an approver without a
        meaningful name, like a lambda."""
        self.approver = approver
        self.name = name or approver.__name__

    def check(self, tool: "Tool", arguments: dict) -> str | None:
        """Return None if the call may run, or a refusal message for the model."""
        if tool.permission == "allow":
            return None
        return self.approver(tool.name, arguments)
