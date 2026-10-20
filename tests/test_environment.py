"""Environments: loading and validating specs, versions, and provisioning workspaces."""

import json

import pytest

from harness import environment as environment_module
from harness.environment import MARKER, EnvironmentSetupError, load_environment, provision
from harness.tools import ToolContext


@pytest.fixture
def envs(tmp_path):
    """A temp environments/ folder. Write specs with envs.write(name, yaml_text)."""
    directory = tmp_path / "environments"
    directory.mkdir()

    class Envs:
        root = tmp_path
        dir = directory

        def write(self, name, text):
            (directory / f"{name}.yaml").write_text(text)

        def load(self, name):
            return load_environment(directory, name, tmp_path)

    return Envs()


SIMPLE = """\
description: test env
setup:
  - echo built > made.txt
env:
  PATH: "{workspace}/bin"
  GREETING: hello
prompt: |
  You have made.txt.
"""


class TestLoad:
    def test_fields(self, envs):
        envs.write("simple", SIMPLE)
        env = envs.load("simple")
        assert env.name == "simple"
        assert env.description == "test env"
        assert env.setup == ("echo built > made.txt",)
        assert env.env == {"PATH": "{workspace}/bin", "GREETING": "hello"}
        assert env.prompt == "You have made.txt."
        assert env.files is None

    def test_minimal_spec(self, envs):
        envs.write("empty", "description: nothing\n")
        env = envs.load("empty")
        assert (env.setup, env.env, env.prompt) == ((), {}, "")

    def test_unknown_environment_lists_available(self, envs):
        envs.write("a", SIMPLE)
        envs.write("b", SIMPLE)
        with pytest.raises(EnvironmentSetupError, match="no environment 'nope'. Available: a, b"):
            envs.load("nope")

    def test_typo_in_key_is_an_error(self, envs):
        envs.write("typo", "setpu:\n  - echo hi\n")
        with pytest.raises(EnvironmentSetupError, match="unknown key"):
            envs.load("typo")

    @pytest.mark.parametrize("text", ["setup: echo hi\n", "setup:\n  - 3\n"])
    def test_setup_must_be_a_list_of_strings(self, envs, text):
        envs.write("bad", text)
        with pytest.raises(EnvironmentSetupError, match="setup must be a list"):
            envs.load("bad")

    def test_env_values_must_be_strings(self, envs):
        envs.write("bad", "env:\n  X: 3\n")
        with pytest.raises(EnvironmentSetupError, match="env must map names to strings"):
            envs.load("bad")

    def test_not_a_mapping(self, envs):
        envs.write("bad", "- a\n- b\n")
        with pytest.raises(EnvironmentSetupError, match="key: value"):
            envs.load("bad")

    def test_missing_files_folder(self, envs):
        envs.write("files", "files: templates/missing\n")
        with pytest.raises(EnvironmentSetupError, match="doesn't exist"):
            envs.load("files")

    def test_variables_fill_in_workspace(self, envs, tmp_path):
        envs.write("simple", SIMPLE)
        assert envs.load("simple").variables(tmp_path)["PATH"] == f"{tmp_path}/bin"


class TestVersion:
    def test_stable(self, envs):
        envs.write("simple", SIMPLE)
        assert envs.load("simple").version == envs.load("simple").version

    def test_changes_with_the_spec(self, envs):
        envs.write("simple", SIMPLE)
        before = envs.load("simple").version
        envs.write("simple", SIMPLE.replace("hello", "hi"))
        assert envs.load("simple").version != before

    def test_changes_with_template_file_contents(self, envs):
        template = envs.root / "templates" / "t"
        template.mkdir(parents=True)
        (template / "conftest.py").write_text("# v1\n")
        envs.write("withfiles", "files: templates/t\n")
        before = envs.load("withfiles").version
        (template / "conftest.py").write_text("# v2\n")
        assert envs.load("withfiles").version != before


class TestProvision:
    def test_runs_setup_in_the_workspace(self, envs, workspace):
        envs.write("simple", SIMPLE)
        assert provision(envs.load("simple"), workspace) is True
        assert (workspace / "made.txt").read_text() == "built\n"

    def test_writes_marker_last(self, envs, workspace):
        envs.write("simple", SIMPLE)
        env = envs.load("simple")
        provision(env, workspace)
        marker = json.loads((workspace / MARKER).read_text())
        assert (marker["name"], marker["version"]) == ("simple", env.version)

    def test_second_time_is_skipped(self, envs, workspace):
        # e.g. a second session sharing this workspace
        envs.write("simple", SIMPLE)
        env = envs.load("simple")
        provision(env, workspace)
        (workspace / "made.txt").unlink()
        assert provision(env, workspace) is False
        assert not (workspace / "made.txt").exists()  # setup didn't run again

    def test_refuses_a_workspace_set_up_differently(self, envs, workspace):
        envs.write("a", SIMPLE)
        envs.write("b", SIMPLE.replace("hello", "hi"))
        provision(envs.load("a"), workspace)
        with pytest.raises(EnvironmentSetupError, match="already set up as 'a'"):
            provision(envs.load("b"), workspace)

    def test_setup_sees_the_environment_variables(self, envs, workspace):
        envs.write("vars", 'setup:\n  - echo "$GREETING" > greeting.txt\nenv:\n  GREETING: hello\n')
        provision(envs.load("vars"), workspace)
        assert (workspace / "greeting.txt").read_text() == "hello\n"

    def test_setup_never_sees_secrets(self, envs, workspace, monkeypatch):
        monkeypatch.setenv("OMLX_API_KEY", "sk-secret")
        envs.write("leak", 'setup:\n  - echo "[$OMLX_API_KEY]" > out.txt\n')
        provision(envs.load("leak"), workspace)
        assert (workspace / "out.txt").read_text() == "[]\n"

    def test_failed_command_raises_with_output_and_no_marker(self, envs, workspace):
        envs.write("broken", "setup:\n  - echo first > ok.txt\n  - echo oops >&2; exit 7\n  - echo never > never.txt\n")
        with pytest.raises(EnvironmentSetupError, match="exit 7") as err:
            provision(envs.load("broken"), workspace)
        assert "oops" in str(err.value)
        assert not (workspace / "never.txt").exists()  # stops at the first failure
        assert not (workspace / MARKER).exists()  # not marked as set up

    def test_timeout(self, envs, workspace, monkeypatch):
        monkeypatch.setattr(environment_module, "SETUP_TIMEOUT", 0.5)
        envs.write("slow", "setup:\n  - sleep 5\n")
        with pytest.raises(EnvironmentSetupError, match="timed out"):
            provision(envs.load("slow"), workspace)

    def test_copies_template_files_without_overwriting(self, envs, workspace):
        template = envs.root / "templates" / "t"
        (template / "sub").mkdir(parents=True)
        (template / "README.md").write_text("from template\n")
        (template / "sub" / "conf.txt").write_text("conf\n")
        (workspace / "README.md").write_text("the user's own\n")
        envs.write("withfiles", "files: templates/t\n")
        provision(envs.load("withfiles"), workspace)
        assert (workspace / "README.md").read_text() == "the user's own\n"  # never overwritten
        assert (workspace / "sub" / "conf.txt").read_text() == "conf\n"


class TestCommandEnv:
    """ToolContext.command_env(): what both setup commands and the agent's bash run with."""

    def test_path_is_prepended(self, workspace, monkeypatch):
        monkeypatch.setenv("PATH", "/usr/bin:/bin")
        env = ToolContext(workspace, {"PATH": "/ws/.venv/bin"}).command_env()
        assert env["PATH"].startswith("/ws/.venv/bin:")
        assert env["PATH"].endswith("/usr/bin:/bin")

    def test_other_variables_are_set(self, workspace):
        assert ToolContext(workspace, {"VIRTUAL_ENV": "/ws/.venv"}).command_env()["VIRTUAL_ENV"] == "/ws/.venv"

    def test_still_allowlisted(self, workspace, monkeypatch):
        monkeypatch.setenv("OMLX_API_KEY", "sk-secret")
        assert "OMLX_API_KEY" not in ToolContext(workspace, {"X": "1"}).command_env()

    def test_no_variables_is_plain_agent_env(self, workspace):
        from harness.tools.base import agent_env
        assert ToolContext(workspace).command_env() == agent_env()
