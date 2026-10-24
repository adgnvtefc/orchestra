"""Built-in tool implementations. Each module defines functions plus their Tool objects.

BUILTIN_TOOLS is the catalog the registry is built from; a tool not listed here doesn't exist.
"""

from orchestra.tools.builtin.fs import READ_FILE, WRITE_FILE
from orchestra.tools.builtin.shell import BASH

BUILTIN_TOOLS = [READ_FILE, WRITE_FILE, BASH]
