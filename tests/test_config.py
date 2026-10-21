"""Config: parsing agent files, and resolving agent + environment + command-line overrides into a RunConfig."""

from pathlib import Path

import pytest
import yaml

from harness import config
from harness.config import load_agent_config, resolve_environment, resolve_run_config, resolve_workspace
from harness.environment import EnvironmentSetupError


@pytest.fixture
def agent_file(tmp_path):
    """Write `text` to a temp .md file and return its path."""

    def write(text: str):
        path = tmp_path / "agent.md"
        path.write_text(text)
        return path

    return write


class TestValidFiles:
    def test_settings_and_prompt(self, agent_file):
        path = agent_file("---\nmodel: m\ntemperature: 0.6\nmax_steps: 20\n---\nYou are helpful.\n")
        assert load_agent_config(path) == {
            "model": "m",
            "temperature": 0.6,
            "max_steps": 20,
            "system_prompt": "You are helpful.",
        }

    def test_yaml_types_are_converted(self, agent_file):
        config = load_agent_config(agent_file("---\nt: 0.6\nn: 20\nflag: true\ntools:\n  - a\n  - b\n---\n"))
        assert config["t"] == 0.6 and isinstance(config["t"], float)
        assert config["n"] == 20 and isinstance(config["n"], int)
        assert config["flag"] is True
        assert config["tools"] == ["a", "b"]

    def test_accepts_str_path(self, agent_file):
        assert load_agent_config(str(agent_file("---\nmodel: m\n---\nhi")))["model"] == "m"

    def test_empty_settings_block(self, agent_file):
        assert load_agent_config(agent_file("---\n---\nprompt")) == {"system_prompt": "prompt"}

    def test_empty_body_gives_empty_prompt(self, agent_file):
        assert load_agent_config(agent_file("---\nmodel: m\n---\n"))["system_prompt"] == ""

    def test_prompt_is_stripped_but_keeps_inner_formatting(self, agent_file):
        body = "\n\n# Role\n\nLine one.\n  indented\n\n"
        assert load_agent_config(agent_file(f"---\nmodel: m\n---{body}"))["system_prompt"] == (
            "# Role\n\nLine one.\n  indented"
        )

    def test_dashes_inside_prompt_are_kept(self, agent_file):
        # Only the first closing '---' ends the settings; later ones belong to the prompt.
        config = load_agent_config(agent_file("---\nmodel: m\n---\nabove\n---\nbelow"))
        assert config == {"model": "m", "system_prompt": "above\n---\nbelow"}

    def test_windows_line_endings(self, agent_file):
        config = load_agent_config(agent_file("---\r\nmodel: m\r\n---\r\nhi\r\n"))
        assert config == {"model": "m", "system_prompt": "hi"}

    def test_opening_line_tolerates_trailing_whitespace(self, agent_file):
        assert load_agent_config(agent_file("---  \nmodel: m\n---\nhi"))["model"] == "m"

    def test_closing_line_tolerates_trailing_whitespace(self, agent_file):
        assert load_agent_config(agent_file("---\nmodel: m\n---  \nhi"))["model"] == "m"


class TestInvalidFiles:
    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_agent_config(tmp_path / "nope.md")

    @pytest.mark.parametrize("text", ["", "\n", "model: m\n---\nhi", "hello\n---\nmodel: m\n---\n"])
    def test_must_start_with_dashes(self, agent_file, text):
        with pytest.raises(ValueError, match="must start with a '---' line"):
            load_agent_config(agent_file(text))

    def test_missing_closing_dashes(self, agent_file):
        with pytest.raises(ValueError, match="no closing '---' line"):
            load_agent_config(agent_file("---\nmodel: m\nYou are helpful."))

    @pytest.mark.parametrize("settings", ["- a\n- b", "just a string", "42"])
    def test_settings_must_be_a_mapping(self, agent_file, settings):
        with pytest.raises(ValueError, match="settings must be 'key: value' lines"):
            load_agent_config(agent_file(f"---\n{settings}\n---\nhi"))

    def test_error_names_the_file(self, agent_file):
        path = agent_file("no frontmatter")
        with pytest.raises(ValueError, match=str(path)):
            load_agent_config(path)

    def test_malformed_yaml(self, agent_file):
        with pytest.raises(yaml.YAMLError):
            load_agent_config(agent_file("---\nmodel: [unclosed\n---\nhi"))

    def test_yaml_cannot_construct_python_objects(self, agent_file):
        # safe_load refuses python/* tags; plain yaml.load would execute this.
        with pytest.raises(yaml.constructor.ConstructorError):
            load_agent_config(agent_file("---\nx: !!python/object/apply:os.system ['echo pwned']\n---\nhi"))


@pytest.fixture
def config_dirs(tmp_path, monkeypatch):
    """Temp agents/ and environments/ folders, with the project root at tmp_path."""
    agents, envs = tmp_path / "agents", tmp_path / "environments"
    agents.mkdir(), envs.mkdir()
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "AGENTS_DIR", agents)
    monkeypatch.setattr(config, "ENVIRONMENTS_DIR", envs)
    (envs / "py.yaml").write_text("setup:\n  - echo hi\nprompt: |\n  You have Python.\n")

    def agent(name, extra=""):
        (agents / f"{name}.md").write_text(f"---\nmodel: m\ntemperature: 0.1\nmax_steps: 3\ntools:\n  - bash\n{extra}---\nBase prompt.\n")

    return agent


class TestListAgents:
    def test_sorted_names(self, config_dirs):
        config_dirs("zeta")
        config_dirs("alpha")
        assert config.list_agents() == ["alpha", "zeta"]


class TestResolveWorkspace:
    def test_cli_wins(self, tmp_path):
        assert resolve_workspace(tmp_path, "somewhere/else") == tmp_path.resolve()

    def test_agent_file_path_is_from_project_root(self, config_dirs, tmp_path):
        (tmp_path / "sub").mkdir()
        assert resolve_workspace(None, "sub") == (tmp_path / "sub").resolve()

    def test_agent_file_home_path(self):
        assert resolve_workspace(None, "~") == Path.home().resolve()

    def test_none_means_fresh(self):
        assert resolve_workspace(None, None) is None

    def test_must_exist(self, tmp_path):
        with pytest.raises(ValueError, match="not an existing folder"):
            resolve_workspace(tmp_path / "typo", None)


class TestResolveEnvironment:
    def test_from_agent_file(self, config_dirs):
        assert resolve_environment(None, "py").name == "py"

    def test_cli_wins(self, config_dirs):
        assert resolve_environment("none", "py") is None

    def test_neither(self, config_dirs):
        assert resolve_environment(None, None) is None

    def test_unknown(self, config_dirs):
        with pytest.raises(EnvironmentSetupError, match="no environment 'nope'"):
            resolve_environment("nope", None)


class TestResolveRunConfig:
    def test_without_environment(self, config_dirs):
        config_dirs("base")
        rc = resolve_run_config("base")
        assert rc.agent["name"] == "base"
        assert rc.agent["environment"] is None
        assert rc.system_prompt == "Base prompt."
        assert rc.workspace is None and rc.environment is None

    def test_environment_is_resolved_into_the_config(self, config_dirs):
        config_dirs("base", "environment: py\n")
        rc = resolve_run_config("base")
        assert rc.agent["environment"] == {"name": "py", "version": rc.environment.version}
        assert rc.system_prompt == "Base prompt.\n\nYou have Python."

    def test_cli_overrides(self, config_dirs, tmp_path):
        config_dirs("base", "environment: py\n")
        rc = resolve_run_config("base", workspace=tmp_path, environment="none")
        assert rc.workspace == tmp_path.resolve()
        assert rc.environment is None
        assert rc.system_prompt == "Base prompt."

    def test_unknown_tool(self, config_dirs, tmp_path):
        (config.AGENTS_DIR / "typo.md").write_text("---\nmodel: m\ntools:\n  - bsh\n---\nP\n")
        with pytest.raises(ValueError, match="bsh"):
            resolve_run_config("typo")

    def test_unknown_agent(self, config_dirs):
        with pytest.raises(FileNotFoundError):
            resolve_run_config("ghost")

    def test_creates_nothing(self, config_dirs, tmp_path):
        config_dirs("base", "environment: py\n")
        before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
        resolve_run_config("base")
        assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before
