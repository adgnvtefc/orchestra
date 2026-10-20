"""What a tool is, plus helpers shared by every tool."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from harness.policy import Permission

# ~10k chars is roughly 2.5k tokens: enough to be useful, small enough that one
# tool result can't eat a 32k context window.
MAX_OUTPUT_CHARS = 10_000


@dataclass(frozen=True)
class ToolContext:
    """What a tool call runs *in*, as opposed to what it was asked to do (the arguments).

    Toolset passes it as the first argument of every tool function. The model never
    sees it and can't set it: it isn't in any schema.
    """

    workspace: Path  # all file access and shell commands are confined to this directory


# makes so cannot modify after creation
@dataclass(frozen=True)
class Tool:
    """A tool = the Python function that runs it + the JSON schema the model sees."""

    # Callable means function takes arguments, return str
    fn: Callable[..., str]
    schema: dict
    # Default permission. Required, with no default value: a new tool can't silently become "allow".
    permission: Permission

    # makes so that tool.name is a field, but is computed from schema
    @property
    def name(self) -> str:
        return self.schema["function"]["name"]


def truncate(text: str) -> str:
    """Keep the head and tail of long output; errors and summaries usually live at the end."""
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    half = MAX_OUTPUT_CHARS // 2
    omitted = len(text) - MAX_OUTPUT_CHARS
    return f"{text[:half]}\n\n... [{omitted} characters omitted] ...\n\n{text[-half:]}"
