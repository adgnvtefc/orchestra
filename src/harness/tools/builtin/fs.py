"""Filesystem tools. All paths are relative to the workspace and can't escape it."""

from pathlib import Path

from harness.tools.base import Tool, ToolContext, truncate


def _resolve(ctx: ToolContext, path: str) -> Path:
    """Turn a model-supplied path into an absolute path inside the workspace, or raise.

    resolve() collapses ".." and follows symlinks *before* the check, so
    "../../.ssh/id_rsa" and "link_to_home/.ssh" are both caught. The workspace is
    resolved too, so a workspace reached through a symlink (macOS: /var -> /private/var)
    still contains its own files.
    """
    workspace = ctx.workspace.resolve()
    full = (workspace / path).resolve()
    if not full.is_relative_to(workspace):
        raise PermissionError(f"Path {path!r} is outside the workspace. Use paths relative to the workspace.")
    return full


def read_file(ctx: ToolContext, path: str) -> str:
    """Return the contents of `path` (relative to the workspace).

    Refuse paths that resolve outside the workspace (think: "../../.ssh/id_rsa").
    """
    full = _resolve(ctx, path)
    try:
        text = full.read_text()
    except UnicodeDecodeError:
        return f"{path} is a binary file and can't be shown as text."
    if not text:
        return f"{path} is empty."
    return truncate(text)


def write_file(ctx: ToolContext, path: str, content: str) -> str:
    """Write `content` to `path` (relative to the workspace), creating parent dirs.

    Same path rule as read_file. Return a short confirmation the model can read.
    """
    full = _resolve(ctx, path)
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content)
    return f"Wrote {len(content)} characters to {path}."


READ_FILE = Tool(
    fn=read_file,
    schema={
        "type": "function",                 # always "function"; the only type that exists
        "function": {
            "name": "read_file",            # the name agents list in their .md file
            "description": "Read a text file from the workspace and return its contents.",
            "parameters": {                 # a JSON Schema for the arguments object
                "type": "object",           # arguments are always one object, i.e. keyword args
                "properties": {
                    "path": {               # must match the Python parameter name exactly
                        "type": "string",
                        "description": "Path relative to the workspace, e.g. 'src/main.py'.",
                    },
                },
                "required": ["path"],       # which properties the model must provide
            },
        },
    },
    permission="allow",
)

WRITE_FILE = Tool(
    fn=write_file,
    schema={
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file in the workspace, creating parent directories. Overwrites the file if it already exists.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path relative to the workspace, e.g. 'src/main.py'.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    permission="allow",
)
