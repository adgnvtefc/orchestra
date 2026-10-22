"""Config: parsing agent files, and resolving agent + environment + command-line overrides into a RunConfig."""

from pathlib import Path

import pytest
import yaml

from harness import config
from harness.config import Project, load_agent_config, resolve_environment, resolve_run_config, resolve_workspace
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

    def test_yaml_cannot_construct_python_objects(self, agent_file, tmp_path):
        # safe_load refuses python/* tags; plain yaml.load would execute this.
        with pytest.raises(yaml.constructor.ConstructorError):
            load_agent_config(agent_file("---\nx: !!python/object/apply:os.system ['echo pwned']\n---\nhi"))


@pytest.fixture
def config_dirs(tmp_path, monkeypatch):
    """Temp agents/ and environments/ folders, with the project root at tmp_path."""
    agents, envs = tmp_path / "agents", tmp_path / "environments"
    agents.mkdir(), envs.mkdir()
    (envs / "py.yaml").write_text("setup:\n  - echo hi\nprompt: |\n  You have Python.\n")

    def agent(name, extra=""):
        (agents / f"{name}.md").write_text(f"---\nmodel: m\ntemperature: 0.1\nmax_steps: 3\ntools:\n  - bash\n{extra}---\nBase prompt.\n")

    return agent


class TestListAgents:
    def test_sorted_names(self, config_dirs, tmp_path):
        config_dirs("zeta")
        config_dirs("alpha")
        assert Project(tmp_path).agents() == ["alpha", "zeta"]


class TestResolveWorkspace:
    def test_cli_wins(self, tmp_path):
        assert resolve_workspace(tmp_path, "somewhere/else", Project(tmp_path)) == tmp_path.resolve()

    def test_agent_file_path_is_from_project_root(self, config_dirs, tmp_path):
        (tmp_path / "sub").mkdir()
        assert resolve_workspace(None, "sub", Project(tmp_path)) == (tmp_path / "sub").resolve()

    def test_agent_file_home_path(self, tmp_path):
        assert resolve_workspace(None, "~", Project(tmp_path)) == Path.home().resolve()

    def test_none_means_fresh(self, tmp_path):
        assert resolve_workspace(None, None, Project(tmp_path)) is None

    def test_must_exist(self, tmp_path):
        with pytest.raises(ValueError, match="not an existing folder"):
            resolve_workspace(tmp_path / "typo", None, Project(tmp_path))


class TestResolveEnvironment:
    def test_from_agent_file(self, config_dirs, tmp_path):
        assert resolve_environment(None, "py", Project(tmp_path)).name == "py"

    def test_cli_wins(self, config_dirs, tmp_path):
        assert resolve_environment("none", "py", Project(tmp_path)) is None

    def test_neither(self, config_dirs, tmp_path):
        assert resolve_environment(None, None, Project(tmp_path)) is None

    def test_unknown(self, config_dirs, tmp_path):
        with pytest.raises(EnvironmentSetupError, match="no environment 'nope'"):
            resolve_environment("nope", None, Project(tmp_path))


class TestResolveRunConfig:
    def test_without_environment(self, config_dirs, tmp_path):
        config_dirs("base")
        rc = resolve_run_config("base", Project(tmp_path))
        assert rc.agent["name"] == "base"
        assert rc.agent["environment"] is None
        assert rc.system_prompt == "Base prompt."
        assert rc.workspace is None and rc.environment is None

    def test_environment_is_resolved_into_the_config(self, config_dirs, tmp_path):
        config_dirs("base", "environment: py\n")
        rc = resolve_run_config("base", Project(tmp_path))
        assert rc.agent["environment"] == {"name": "py", "version": rc.environment.version}
        assert rc.system_prompt == "Base prompt.\n\nYou have Python."

    def test_cli_overrides(self, config_dirs, tmp_path):
        config_dirs("base", "environment: py\n")
        rc = resolve_run_config("base", Project(tmp_path), workspace=tmp_path, environment="none")
        assert rc.workspace == tmp_path.resolve()
        assert rc.environment is None
        assert rc.system_prompt == "Base prompt."

    def test_unknown_tool(self, config_dirs, tmp_path):
        (tmp_path / "agents" / "typo.md").write_text("---\nmodel: m\ntools:\n  - bsh\n---\nP\n")
        with pytest.raises(ValueError, match="bsh"):
            resolve_run_config("typo", Project(tmp_path))

    def test_unknown_agent(self, config_dirs, tmp_path):
        with pytest.raises(FileNotFoundError):
            resolve_run_config("ghost", Project(tmp_path))

    def test_creates_nothing(self, config_dirs, tmp_path):
        config_dirs("base", "environment: py\n")
        before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
        resolve_run_config("base", Project(tmp_path))
        assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before



class TestProjectFind:
    def test_the_folder_itself(self, tmp_path):
        (tmp_path / "agents").mkdir()
        assert Project.find(tmp_path).root == tmp_path.resolve()

    def test_walks_up_from_a_subfolder(self, tmp_path):
        (tmp_path / "agents").mkdir()
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        assert Project.find(deep).root == tmp_path.resolve()

    def test_nearest_wins(self, tmp_path):
        (tmp_path / "agents").mkdir()
        inner = tmp_path / "inner"
        (inner / "agents").mkdir(parents=True)
        assert Project.find(inner).root == inner.resolve()

    def test_none_found(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "parents", property(lambda self: []))  # don't let it find a real agents/ above tmp
        with pytest.raises(FileNotFoundError, match="no agents/ folder"):
            Project.find(tmp_path)


class TestExtraToolNames:
    def test_agent_may_list_an_extra_tool(self, config_dirs, tmp_path):
        from harness.tools import Tool
        weather = Tool(fn=lambda ctx: "", schema={"type": "function", "function": {"name": "weather"}}, permission="allow")
        (tmp_path / "agents" / "forecaster.md").write_text("---\nmodel: m\ntools:\n  - weather\n---\nP\n")
        assert resolve_run_config("forecaster", Project(tmp_path, (weather,))).agent["tools"] == ["weather"]
        with pytest.raises(ValueError, match="weather"):
            resolve_run_config("forecaster", Project(tmp_path))  # without it, it's unknown



class TestProject:
    def test_dirs(self, tmp_path):
        project = Project(tmp_path)
        assert (project.agents_dir, project.environments_dir) == (tmp_path / "agents", tmp_path / "environments")

    def test_find_keeps_extra_tools(self, tmp_path):
        from harness.tools import Tool
        (tmp_path / "agents").mkdir()
        weather = Tool(fn=lambda ctx: "", schema={"type": "function", "function": {"name": "weather"}}, permission="allow")
        project = Project.find(tmp_path, [weather])
        assert project.extra_tools == (weather,)
        assert "weather" in project.tools() and "bash" in project.tools()


TOOLS_PY = '''
from harness.tools import Tool

def weather(ctx, city: str) -> str:
    return f"sunny in {city}"

TOOLS = [Tool(fn=weather, permission="allow", schema={"type": "function", "function": {
    "name": "weather", "description": "the weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}})]
'''


class TestProjectToolsFile:
    """A project's tools.py, defining TOOLS, is how the CLI (and Project.find) get its own tools."""

    def project(self, tmp_path, tools_py=TOOLS_PY):
        (tmp_path / "agents").mkdir(exist_ok=True)
        if tools_py is not None:
            (tmp_path / "tools.py").write_text(tools_py)
        return tmp_path

    def test_find_loads_tools_py(self, tmp_path):
        project = Project.find(self.project(tmp_path))
        assert [t.name for t in project.extra_tools] == ["weather"]
        assert "weather" in project.tools() and "bash" in project.tools()
        assert project.tools()["weather"].fn(None, "Paris") == "sunny in Paris"

    def test_at_loads_tools_py(self, tmp_path):
        assert [t.name for t in Project.at(self.project(tmp_path)).extra_tools] == ["weather"]

    def test_plain_constructor_loads_nothing(self, tmp_path):
        # Project(root) is a value; only at()/find() read the project's files.
        assert Project(self.project(tmp_path)).extra_tools == ()

    def test_no_tools_py(self, tmp_path):
        assert Project.find(self.project(tmp_path, tools_py=None)).extra_tools == ()

    def test_passed_tools_are_added(self, tmp_path):
        from harness.tools import Tool
        extra = Tool(fn=lambda ctx: "", schema={"type": "function", "function": {"name": "extra"}}, permission="allow")
        assert [t.name for t in Project.find(self.project(tmp_path), [extra]).extra_tools] == ["weather", "extra"]

    @pytest.mark.parametrize(("tools_py", "error"), [
        ("x = 1\n", "must define TOOLS"),
        ("TOOLS = ['not a tool']\n", "must define TOOLS"),
        ("import not_a_real_module_xyz\n", "couldn't be loaded: ModuleNotFoundError"),
        ("def broken(:\n", "couldn't be loaded: SyntaxError"),
    ])
    def test_problems_name_the_file(self, tmp_path, tools_py, error):
        from harness.config import ProjectToolsError
        with pytest.raises(ProjectToolsError, match=error) as err:
            Project.find(self.project(tmp_path, tools_py))
        assert "tools.py" in str(err.value)

    def test_agent_can_use_them(self, tmp_path):
        root = self.project(tmp_path)
        (root / "agents" / "forecaster.md").write_text("---\nmodel: m\ntools:\n  - weather\n---\nP\n")
        assert resolve_run_config("forecaster", Project.find(root)).agent["tools"] == ["weather"]
