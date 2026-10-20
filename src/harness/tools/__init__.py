"""Tools package. Public API: what the rest of the harness imports from here.

Layout:
  base.py     the contract every tool is written against (Tool, ToolContext, truncate)
  toolset.py  machinery: REGISTRY of all tools + Toolset, the subset one agent gets
  builtin/    tool implementations; they import base.py and nothing else from here

To add a tool: write it in a builtin/ module, then add it to BUILTIN_TOOLS in builtin/__init__.py.
"""

from harness.tools.base import Tool, ToolContext
from harness.tools.toolset import REGISTRY, Toolset

__all__ = ["REGISTRY", "Tool", "ToolContext", "Toolset"]
