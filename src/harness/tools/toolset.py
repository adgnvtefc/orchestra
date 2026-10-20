"""Tool machinery: the registry of every tool, and the per-agent Toolset that dispatches calls.

Nothing in here knows what any particular tool does. Tool implementations live in
builtin/ and only depend on base.py.
"""

import json

from harness.policy import Policy
from harness.tools.base import Tool, ToolContext
from harness.tools.builtin import BUILTIN_TOOLS

# Every tool that exists, by name. Later, other sources (MCP servers, plugins) add to this too.
REGISTRY: dict[str, Tool] = {t.name: t for t in BUILTIN_TOOLS}


class Toolset:
    def __init__(self, names: list[str], policy: Policy, context: ToolContext):
        """Pick tools from REGISTRY by name. Unknown names fail now, at startup, not mid-run.

        `policy` decides whether each call may run; execute() checks it before every call.
        `context` (the workspace, ...) is passed to every tool as its first argument.
        """
        unknown = [n for n in names if n not in REGISTRY]
        if unknown:
            raise ValueError(f"Unknown tool(s) {unknown}. Available tools: {', '.join(REGISTRY)}.")
        # define an internal registry of all the tools we have
        self.tools = {n: REGISTRY[n] for n in names}

        # define a policy object for approval
        self.policy = policy
        self.context = context

    def schemas(self) -> list[dict]:
        """What the model sees: only this agent's tools."""
        return [t.schema for t in self.tools.values()]

    # executes a tool
    def execute(self, name: str, arguments_json: str) -> str:
        """Parse the model's JSON arguments, call the matching tool, return its result as a string.

        This must never raise. Unknown tool, invalid JSON, wrong argument names, and
        exceptions from the tool itself all become an error string that goes back to
        the model, so it can see what went wrong and try again.
        """
        # Every failure is *returned*, not raised: the error text becomes the tool result,
        # and the model reads it on its next turn and can fix its own mistake.
        # Lookup is in THIS toolset, not REGISTRY: a tool this agent wasn't given can't run.
        tool = self.tools.get(name)
        if tool is None:
            return f"Error: unknown tool {name!r}. Available tools: {', '.join(self.tools) or '(none)'}."

        try:
            # Some servers send "" instead of "{}" for a call with no arguments.
            arguments = json.loads(arguments_json or "{}")
        except json.JSONDecodeError as e:
            return f"Error: tool arguments were not valid JSON ({e}). Send a JSON object like {{\"path\": \"notes.txt\"}}."
        if not isinstance(arguments, dict):
            return f"Error: tool arguments must be a JSON object, got {type(arguments).__name__}."

        # The one enforcement point: every call passes here, after parsing (so a human
        # approving it sees the real arguments) and before the tool runs.
        refusal = self.policy.check(tool, arguments)
        if refusal:
            return refusal

        try:
            # ** unpacks the dict as keyword args: {"path": "a.txt"} -> fn(path="a.txt")
            # the context goes first, positionally: the model's arguments can't replace it
            return str(tool.fn(self.context, **arguments))
        except TypeError as e:  # missing / unexpected argument names
            return f"Error: bad arguments for {name}: {e}"
        except Exception as e:  # anything the tool itself raised (PermissionError, FileNotFoundError, ...)
            return f"Error: {type(e).__name__}: {e}"
