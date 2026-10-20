"""Shared fixtures. pytest loads this file automatically for every test in this folder."""

import pytest

from harness import orchestrator
from harness.tools import REGISTRY, Tool
from harness.tools import base
from harness.tools.builtin import fs, shell


@pytest.fixture(autouse=True)
def workspace(tmp_path, monkeypatch):
    """Point every tool at a fresh temp directory, for every test (autouse).

    No test can ever touch the real workspace/. WORKSPACE is imported *by name* into
    each tool module, so each module's copy has to be patched, not just base's.
    .resolve() matters on macOS: tmp dirs live under /var, a symlink to /private/var.
    """
    ws = tmp_path.resolve() / "workspace"
    ws.mkdir()
    for module in (base, fs, shell):
        monkeypatch.setattr(module, "WORKSPACE", ws)
    return ws


@pytest.fixture(autouse=True)
def sessions_dir(tmp_path, monkeypatch):
    """Put every session (and its traces) in a temp directory, for every test (autouse):
    tests never fill the real sessions/."""
    directory = tmp_path / "sessions"
    monkeypatch.setattr(orchestrator, "SESSIONS_DIR", directory)
    return directory


@pytest.fixture
def answer_prompt(monkeypatch):
    """Script the human's answers to y/N prompts. Returns the list of prompts shown.

    Usage: prompts = answer_prompt("y")  or  answer_prompt(EOFError) for "no terminal".
    """
    prompts = []

    def install(answer):
        def fake_input(prompt=""):
            prompts.append(prompt)
            if isinstance(answer, type) and issubclass(answer, BaseException):
                raise answer
            return answer

        monkeypatch.setattr("builtins.input", fake_input)
        return prompts

    return install


@pytest.fixture
def approve(answer_prompt):
    """The human says yes to every prompt."""
    return answer_prompt("y")


@pytest.fixture
def fake_tools(monkeypatch):
    """Register test-only tools in REGISTRY for the duration of one test.

    echo:  returns its `text` argument; has an optional `suffix` with a default.
    boom:  always raises RuntimeError, to test that tool exceptions don't escape.
    count: returns an int (not a str), to test that results are stringified.
    guarded: permission "ask"; records each call in `guarded.calls`, to test enforcement.
    """

    def echo(text: str, suffix: str = "") -> str:
        return text + suffix

    def boom() -> str:
        raise RuntimeError("kaboom")

    def count() -> int:
        return 42

    def guarded(target: str) -> str:
        guarded.calls.append(target)
        return f"did {target}"

    guarded.calls = []

    tools = {
        "echo": Tool(fn=echo, schema=_schema("echo", {"text": "string", "suffix": "string"}, ["text"]), permission="allow"),
        "boom": Tool(fn=boom, schema=_schema("boom", {}, []), permission="allow"),
        "count": Tool(fn=count, schema=_schema("count", {}, []), permission="allow"),
        "guarded": Tool(fn=guarded, schema=_schema("guarded", {"target": "string"}, ["target"]), permission="ask"),
    }
    for name, tool in tools.items():
        monkeypatch.setitem(REGISTRY, name, tool)
    return tools


def _schema(name: str, props: dict[str, str], required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"test tool {name}",
            "parameters": {
                "type": "object",
                "properties": {p: {"type": t} for p, t in props.items()},
                "required": required,
            },
        },
    }
