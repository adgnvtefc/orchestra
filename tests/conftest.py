"""Shared fixtures. pytest loads this file automatically for every test in this folder."""

import pytest

from orchestra import records
from orchestra.providers import omlx
from orchestra.sandbox import Unsandboxed
from orchestra.tools import REGISTRY, Tool, ToolContext


@pytest.fixture
def workspace(tmp_path):
    """A fresh temp workspace, for tests that call tools directly.

    (Runs through the orchestrator get their own workspace inside their session, and
    sessions go to a temp folder, so no test ever touches a real folder.) .resolve()
    matters on macOS: tmp dirs live under /var, a symlink to /private/var.
    """
    ws = tmp_path.resolve() / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def ctx(workspace):
    """The ToolContext for calling tools directly: confined to the test's temp workspace, no
    sandbox (sandboxed bash has its own tests)."""
    return ToolContext(workspace=workspace, sandbox=Unsandboxed())


@pytest.fixture(autouse=True)
def orchestra_dir(tmp_path, monkeypatch):
    """A temp ~/.orchestra for every test (autouse): records and sessions never touch the real one."""
    directory = tmp_path / "orchestra"
    monkeypatch.setattr(records, "ORCHESTRA_DIR", directory)
    return directory


@pytest.fixture(autouse=True)
def no_model_client(monkeypatch):
    """Every test (autouse) starts with no oMLX client made yet, and gets the real omlx.client
    back afterwards. A test that needs a model scripts one with fakes.use_model(...); without
    that, a model call needs OMLX_API_KEY, which tests never load."""
    real = omlx.client
    real.cache_clear()
    monkeypatch.setattr(omlx, "client", real)  # undone after the test, whatever replaced it
    yield
    real.cache_clear()


@pytest.fixture
def sessions_dir(orchestra_dir):
    """Where tests' sessions go: the test app's, ~/.orchestra/orchestra-test/sessions/.
    (CLI tests override this with the CLI's app, "orchestra-cli".)"""
    return orchestra_dir / "orchestra-test" / "sessions"


@pytest.fixture(autouse=True)
def no_dotenv(monkeypatch):
    """The CLI loads .env at startup; in tests that would pull your real API key into the
    test process. Every test (autouse) gets a no-op instead. Returns the list of calls."""
    calls = []
    monkeypatch.setattr("orchestra.interfaces.cli.load_dotenv", lambda *args, **kwargs: calls.append(args))
    return calls


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
def typed(monkeypatch):
    """Script what the human types at `you>`. When the script runs out, it's Ctrl-D."""

    def install(*lines):
        remaining = list(lines)

        def fake_input(prompt=""):
            if not remaining:
                raise EOFError
            item = remaining.pop(0)
            if isinstance(item, type) and issubclass(item, BaseException):
                raise item
            return item

        monkeypatch.setattr("builtins.input", fake_input)

    return install


@pytest.fixture
def fake_tools(monkeypatch):
    """Register test-only tools in REGISTRY for the duration of one test.

    echo:  returns its `text` argument; has an optional `suffix` with a default.
    boom:  always raises RuntimeError, to test that tool exceptions don't escape.
    count: returns an int (not a str), to test that results are stringified.
    guarded: permission "ask"; records each call in `guarded.calls`, to test enforcement.
    """

    def echo(ctx, text: str, suffix: str = "") -> str:
        return text + suffix

    def boom(ctx) -> str:
        raise RuntimeError("kaboom")

    def count(ctx) -> int:
        return 42

    def guarded(ctx, target: str) -> str:
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
