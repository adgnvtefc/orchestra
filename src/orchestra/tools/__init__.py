"""Tools package. Public API: what the rest of the harness imports from here.

Layout:
  base.py     the contract every tool is written against (Tool, ToolContext, truncate)
  toolset.py  machinery: REGISTRY of all tools + Toolset, the subset one agent gets
  builtin/    tool implementations; they import base.py and nothing else from here

To add a built-in tool: write it in a builtin/ module, then add it to BUILTIN_TOOLS in builtin/__init__.py.
To add a tool without touching the harness: a tools.py in your project defining TOOLS, or
Project.at(root, extra_tools=[...]) from Python (see config.py).
"""

from orchestra.tools.base import Tool, ToolContext
from orchestra.tools.toolset import REGISTRY, ToolRun, Toolset, available_tools

__all__ = ["REGISTRY", "Tool", "ToolContext", "ToolRun", "Toolset", "available_tools"]
