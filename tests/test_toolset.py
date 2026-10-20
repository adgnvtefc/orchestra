"""The registry and Toolset: what an agent can see, what it can run, and how failures come back."""

import inspect
import re

import pytest

from harness.policy import Policy
from harness.tools import REGISTRY, ToolContext, Toolset
from tests.fakes import NO_HUMAN


def model_params(tool):
    """The function's parameters the model fills in: all but the first, which is the ToolContext."""
    params = dict(inspect.signature(tool.fn).parameters)
    first = next(iter(params))
    assert first == "ctx", f"{tool.name}: first parameter must be ctx, got {first!r}"
    del params[first]
    return params


class TestRegistry:
    def test_builtins_registered(self):
        assert {"read_file", "write_file", "bash"} <= set(REGISTRY)

    def test_keys_match_tool_names(self):
        assert all(key == tool.name for key, tool in REGISTRY.items())

    @pytest.mark.parametrize(("name", "permission"), [("read_file", "allow"), ("write_file", "allow"), ("bash", "ask")])
    def test_builtin_default_permissions(self, name, permission):
        assert REGISTRY[name].permission == permission

    def test_every_permission_is_valid(self):
        assert all(tool.permission in ("allow", "ask") for tool in REGISTRY.values())


@pytest.mark.parametrize("name", sorted(REGISTRY))
class TestSchemaMatchesFunction:
    """Every built-in tool's schema must agree with its Python function.

    The model only sees the schema, and execute() calls fn(**arguments). If they drift,
    the tool fails only when the model calls it, so catch it here instead.
    """

    def test_schema_shape(self, name):
        schema = REGISTRY[name].schema
        assert schema["type"] == "function"
        fn = schema["function"]
        assert re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", fn["name"])  # what model APIs accept
        assert fn["description"].strip()
        assert fn["parameters"]["type"] == "object"

    def test_properties_match_parameters(self, name):
        tool = REGISTRY[name]
        params = model_params(tool)
        assert set(tool.schema["function"]["parameters"]["properties"]) == set(params)

    def test_required_matches_parameters_without_defaults(self, name):
        tool = REGISTRY[name]
        params = model_params(tool)
        no_default = {p for p, v in params.items() if v.default is inspect.Parameter.empty}
        assert set(tool.schema["function"]["parameters"].get("required", [])) == no_default

    def test_every_property_is_documented(self, name):
        for prop, spec in REGISTRY[name].schema["function"]["parameters"]["properties"].items():
            assert spec.get("type"), f"{name}.{prop} has no type"
            assert spec.get("description", "").strip(), f"{name}.{prop} has no description"


class TestConstruction:
    def test_picks_named_tools(self, ctx):
        assert set(Toolset(["read_file", "bash"], NO_HUMAN, ctx).tools) == {"read_file", "bash"}

    def test_empty(self, ctx):
        toolset = Toolset([], NO_HUMAN, ctx)
        assert toolset.tools == {}
        assert toolset.schemas() == []

    def test_unknown_name_fails_at_startup(self, ctx):
        with pytest.raises(ValueError, match=r"Unknown tool\(s\) \['bsh'\]") as err:
            Toolset(["read_file", "bsh"], NO_HUMAN, ctx)
        assert "read_file, write_file, bash" in str(err.value)  # tells you the valid names

    def test_reports_every_unknown_name(self, ctx):
        with pytest.raises(ValueError, match=r"\['nope', 'nada'\]"):
            Toolset(["nope", "read_file", "nada"], NO_HUMAN, ctx)

    def test_duplicates_collapse(self, ctx):
        # Model APIs reject duplicate tool names, so listing a tool twice must not send it twice.
        assert len(Toolset(["bash", "bash"], NO_HUMAN, ctx).schemas()) == 1

    def test_independent_of_each_other(self, ctx):
        a, b = Toolset(["read_file"], NO_HUMAN, ctx), Toolset(["bash"], NO_HUMAN, ctx)
        assert set(a.tools) == {"read_file"} and set(b.tools) == {"bash"}


class TestSchemas:
    def test_only_own_tools(self, ctx):
        names = [s["function"]["name"] for s in Toolset(["read_file"], NO_HUMAN, ctx).schemas()]
        assert names == ["read_file"]

    def test_order_follows_config(self, ctx):
        # A stable order keeps the request byte-identical across runs (prompt caching).
        names = [s["function"]["name"] for s in Toolset(["bash", "read_file", "write_file"], NO_HUMAN, ctx).schemas()]
        assert names == ["bash", "read_file", "write_file"]

    def test_are_the_registry_schemas(self, ctx):
        assert Toolset(["bash"], NO_HUMAN, ctx).schemas() == [REGISTRY["bash"].schema]


class TestExecute:
    def test_calls_tool_with_keyword_args(self, ctx, fake_tools):
        assert Toolset(["echo"], NO_HUMAN, ctx).execute("echo", '{"text": "hi", "suffix": "!"}') == "hi!"

    def test_optional_argument_uses_default(self, ctx, fake_tools):
        assert Toolset(["echo"], NO_HUMAN, ctx).execute("echo", '{"text": "hi"}') == "hi"

    def test_result_is_stringified(self, ctx, fake_tools):
        assert Toolset(["count"], NO_HUMAN, ctx).execute("count", "{}") == "42"

    @pytest.mark.parametrize("arguments", ["", None])
    def test_empty_arguments_mean_no_arguments(self, ctx, fake_tools, arguments):
        assert Toolset(["count"], NO_HUMAN, ctx).execute("count", arguments) == "42"

    def test_real_tool_end_to_end(self, ctx, workspace):
        toolset = Toolset(["read_file", "write_file"], NO_HUMAN, ctx)
        assert toolset.execute("write_file", '{"path": "a.txt", "content": "hi"}') == "Wrote 2 characters to a.txt."
        assert toolset.execute("read_file", '{"path": "a.txt"}') == "hi"


class TestContext:
    """Each toolset carries its own ToolContext; the model can't see or replace it."""

    def test_each_toolset_works_in_its_own_workspace(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        a.mkdir(), b.mkdir()
        Toolset(["write_file"], NO_HUMAN, ToolContext(a)).execute("write_file", '{"path": "f.txt", "content": "A"}')
        Toolset(["write_file"], NO_HUMAN, ToolContext(b)).execute("write_file", '{"path": "f.txt", "content": "B"}')
        assert (a / "f.txt").read_text() == "A"
        assert (b / "f.txt").read_text() == "B"

    @pytest.mark.parametrize("arguments", ['{"ctx": "/", "path": "x"}', '{"workspace": "/", "path": "x"}'])
    def test_model_cannot_supply_the_context(self, ctx, arguments):
        # ctx is passed positionally first; a model-sent "ctx" collides and is refused.
        result = Toolset(["read_file"], NO_HUMAN, ctx).execute("read_file", arguments)
        assert result.startswith("Error: bad arguments for read_file")


class TestExecuteRefusesToolsNotGiven:
    """The security property: the registry has a tool, but this agent wasn't given it."""

    def test_registered_but_not_given(self, ctx, workspace):
        result = Toolset(["read_file"], NO_HUMAN, ctx).execute("write_file", '{"path": "x.txt", "content": "pwned"}')
        assert result == "Error: unknown tool 'write_file'. Available tools: read_file."
        assert not (workspace / "x.txt").exists()

    def test_bash_not_given_never_prompts(self, ctx, answer_prompt, workspace):
        prompts = answer_prompt("y")
        result = Toolset(["read_file"], NO_HUMAN, ctx).execute("bash", '{"command": "touch pwned"}')
        assert result.startswith("Error: unknown tool 'bash'")
        assert prompts == []  # refused before it ever got to ask
        assert not (workspace / "pwned").exists()

    def test_empty_toolset_runs_nothing(self, ctx):
        assert Toolset([], NO_HUMAN, ctx).execute("read_file", '{"path": "a"}') == (
            "Error: unknown tool 'read_file'. Available tools: (none)."
        )

    def test_invented_tool(self, ctx):
        assert Toolset(["read_file"], NO_HUMAN, ctx).execute("delete_everything", "{}").startswith("Error: unknown tool")


class TestExecuteEnforcesPolicy:
    """Toolset.execute is the enforcement point: the policy is checked before every call."""

    @staticmethod
    def policy(approve: bool):
        asked = []

        def approver(name, arguments):
            asked.append((name, arguments))
            return None if approve else f"test declined {name}"

        policy = Policy(approver)
        policy.asked = asked
        return policy

    def test_approved_call_runs(self, ctx, fake_tools):
        policy = self.policy(True)
        assert Toolset(["guarded"], policy, ctx).execute("guarded", '{"target": "x"}') == "did x"
        assert fake_tools["guarded"].fn.calls == ["x"]

    def test_declined_call_does_not_run(self, ctx, fake_tools):
        result = Toolset(["guarded"], self.policy(False), ctx).execute("guarded", '{"target": "x"}')
        assert result == "test declined guarded"
        assert fake_tools["guarded"].fn.calls == []  # the tool's function was never called

    def test_approver_sees_parsed_arguments(self, ctx, fake_tools):
        policy = self.policy(True)
        Toolset(["guarded"], policy, ctx).execute("guarded", '{"target": "x"}')
        assert policy.asked == [("guarded", {"target": "x"})]

    def test_allow_tools_never_ask(self, ctx, fake_tools):
        policy = self.policy(False)
        assert Toolset(["echo"], policy, ctx).execute("echo", '{"text": "hi"}') == "hi"
        assert policy.asked == []

    @pytest.mark.parametrize("arguments", ["{bad json", "[1]"])
    def test_malformed_calls_are_rejected_before_asking(self, ctx, fake_tools, arguments):
        # Never bother the human about a call that couldn't run anyway.
        policy = self.policy(True)
        assert Toolset(["guarded"], policy, ctx).execute("guarded", arguments).startswith("Error:")
        assert policy.asked == []

    def test_unknown_tool_is_rejected_before_asking(self, ctx, fake_tools):
        policy = self.policy(True)
        Toolset(["echo"], policy, ctx).execute("guarded", '{"target": "x"}')
        assert policy.asked == []

    def test_real_bash_respects_policy(self, ctx, workspace):
        result = Toolset(["bash"], self.policy(False), ctx).execute("bash", '{"command": "touch pwned"}')
        assert result == "test declined bash"
        assert not (workspace / "pwned").exists()

    def test_real_bash_runs_when_approved(self, ctx, workspace):
        result = Toolset(["bash"], self.policy(True), ctx).execute("bash", '{"command": "touch made"}')
        assert result.endswith("(exit code 0)")
        assert (workspace / "made").exists()


class TestExecuteReturnsErrorsInsteadOfRaising:
    """execute() must never raise: every failure becomes text the model can read and react to."""

    def test_invalid_json(self, ctx, fake_tools):
        result = Toolset(["echo"], NO_HUMAN, ctx).execute("echo", "{text: hi}")
        assert result.startswith("Error: tool arguments were not valid JSON")
        assert '{"path": "notes.txt"}' in result  # shows the model what valid looks like

    @pytest.mark.parametrize(("arguments", "kind"), [("[1, 2]", "list"), ('"hi"', "str"), ("3", "int"), ("null", "NoneType")])
    def test_json_that_is_not_an_object(self, ctx, fake_tools, arguments, kind):
        assert Toolset(["echo"], NO_HUMAN, ctx).execute("echo", arguments) == f"Error: tool arguments must be a JSON object, got {kind}."

    def test_missing_required_argument(self, ctx, fake_tools):
        result = Toolset(["echo"], NO_HUMAN, ctx).execute("echo", "{}")
        assert result.startswith("Error: bad arguments for echo:")
        assert "text" in result

    def test_unexpected_argument(self, ctx, fake_tools):
        result = Toolset(["echo"], NO_HUMAN, ctx).execute("echo", '{"text": "a", "colour": "red"}')
        assert result.startswith("Error: bad arguments for echo:")
        assert "colour" in result

    def test_tool_exception(self, ctx, fake_tools):
        assert Toolset(["boom"], NO_HUMAN, ctx).execute("boom", "{}") == "Error: RuntimeError: kaboom"

    def test_real_tool_exceptions_are_named(self, ctx):
        toolset = Toolset(["read_file"], NO_HUMAN, ctx)
        assert toolset.execute("read_file", '{"path": "nope.txt"}').startswith("Error: FileNotFoundError:")
        assert toolset.execute("read_file", '{"path": "../x"}').startswith("Error: PermissionError:")

    @pytest.mark.parametrize(
        ("name", "arguments"),
        [
            ("", ""),
            ("echo", "}{"),
            ("echo", '{"text": 123}'),  # wrong type: echo does str + str, which raises TypeError
            ("echo", '{"text": "a"' * 100),
            ("echo", "\x00\xff"),
            ("boom", '{"unexpected": true}'),
            ("../../bash", "{}"),
            ("read_file", '{"path": null}'),
            ("read_file", '{"path": ["a", "b"]}'),
        ],
    )
    def test_never_raises_on_garbage(self, ctx, fake_tools, name, arguments):
        result = Toolset(["echo", "boom", "read_file"], NO_HUMAN, ctx).execute(name, arguments)
        assert isinstance(result, str) and result.startswith("Error:")
