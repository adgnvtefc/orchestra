"""Config: read agent and environment files, and resolve them into what one run uses.

Reads files and checks folders exist; builds nothing (no sessions, clients, or tools).
Everything a run is configured by goes through resolve_run_config(), so a typo in any
of it fails before anything is created.

A Project is where config comes from: a folder with agents/ (and optionally
environments/, and a tools.py defining TOOLS), plus any tools passed in. Project.find()
locates the folder by walking up from where you are, the way git finds .git. So any
project can have its own.
"""

import importlib.util
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from harness.environment import Environment, load_environment
from harness.tools import Tool, Toolset, available_tools


PROJECT_TOOLS_FILE = "tools.py"


class ProjectToolsError(ValueError):
    """A project's tools.py couldn't be loaded, or doesn't define TOOLS properly."""


@dataclass(frozen=True)
class Project:
    """Everything that comes from the project you're running in, as one value.

    Functions take a Project instead of its parts, so a new per-project setting is added
    here, not to every signature on the way to where it's used.
    """

    root: Path  # has agents/ (and optionally environments/)
    extra_tools: tuple[Tool, ...] = ()  # the project's own tools, alongside the built-ins

    @classmethod
    def at(cls, root: Path, extra_tools: Sequence[Tool] = ()) -> "Project":
        """The project in `root`: its tools.py's TOOLS (if it has one) plus `extra_tools`.

        (Project(root) itself loads nothing: it's a plain value. This is the one that reads
        the project's files.)
        """
        root = root.resolve()
        return cls(root, load_project_tools(root) + tuple(extra_tools))

    @classmethod
    def find(cls, start: Path, extra_tools: Sequence[Tool] = ()) -> "Project":
        """The nearest folder at or above `start` that has an agents/ folder, loaded with Project.at()."""
        start = start.resolve()
        for folder in (start, *start.parents):
            if (folder / "agents").is_dir():
                return cls.at(folder, extra_tools)
        raise FileNotFoundError(f"no agents/ folder in {start} or any folder above it")

    @property
    def agents_dir(self) -> Path:
        return self.root / "agents"

    @property
    def environments_dir(self) -> Path:
        return self.root / "environments"

    def agents(self) -> list[str]:
        """Names of the agents defined in agents/, sorted."""
        return sorted(path.stem for path in self.agents_dir.glob("*.md"))

    def tools(self) -> dict:
        """The built-in tools plus this project's own, by name (a reused name is an error)."""
        return available_tools(self.extra_tools)


def load_project_tools(root: Path) -> tuple[Tool, ...]:
    """The TOOLS list from <root>/tools.py, or () if there's no such file.

    This runs the project's code, the way pytest runs a conftest.py: project tools are as
    trusted as the project's agent files, and run inside the harness process (not in the
    sandbox, which confines bash only). Keep tools.py self-contained, or import only
    installed packages: the project folder isn't added to Python's import path.
    """
    path = root / PROJECT_TOOLS_FILE
    if not path.is_file():
        return ()
    spec = importlib.util.spec_from_file_location(f"_harness_project_tools_{abs(hash(str(path)))}", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise ProjectToolsError(f"{path}: couldn't be loaded: {type(e).__name__}: {e}") from e
    tools = getattr(module, "TOOLS", None)
    if not isinstance(tools, (list, tuple)) or not all(isinstance(t, Tool) for t in tools):
        raise ProjectToolsError(f"{path}: must define TOOLS, a list of harness.tools.Tool")
    return tuple(tools)


@dataclass(frozen=True)
class RunConfig:
    """Everything a run is configured by, resolved and checked."""

    agent: dict  # the resolved agent config: file + name + environment {name, version}; what agent_version hashes
    system_prompt: str  # the agent's prompt, plus the environment's
    workspace: Path | None  # an existing folder, or None for a fresh per-session one
    environment: Environment | None


def resolve_agent(agent_name: str, project: Project) -> dict:
    """Read agents/<name>.md and check its tool names against the project's tools. Just the
    agent's own settings: a resumed session uses this, since its workspace and environment
    are already fixed."""
    agent = {"name": agent_name, **load_agent_config(project.agents_dir / f"{agent_name}.md")}
    Toolset.check_names(agent.get("tools", []), project.tools())  # no `tools:` key means no tools: access is opt-in
    return agent


def resolve_run_config(
    agent_name: str, project: Project, *, workspace: Path | None = None, environment: str | None = None
) -> RunConfig:
    """Read the agent file and apply the command line's overrides; check everything.

    `workspace` / `environment` come from the command line and win over the agent file's.
    """
    agent = resolve_agent(agent_name, project)
    folder = resolve_workspace(workspace, agent.get("workspace"), project)
    env = resolve_environment(environment, agent.get("environment"), project)

    # The resolved config records what the run actually used: the environment's name *and*
    # version, so agent_version changes whenever the environment's content does.
    agent["environment"] = {"name": env.name, "version": env.version} if env else None
    system_prompt = agent["system_prompt"] + (f"\n\n{env.prompt}" if env and env.prompt else "")
    return RunConfig(agent=agent, system_prompt=system_prompt, workspace=folder, environment=env)


def resolve_workspace(from_cli: Path | None, from_config: str | None, project: Project) -> Path | None:
    """Which folder to work in: the command line wins, then the agent file; None means a fresh one.

    A path in an agent file may use ~ and is relative to the project's root, so it means
    the same thing wherever you run from. The folder must already exist: a typo should
    fail here, not quietly create a new empty folder somewhere.
    """
    if from_cli is not None:
        path = from_cli
    elif from_config is not None:
        path = project.root / Path(from_config).expanduser()  # an absolute path replaces the root
    else:
        return None
    if not path.is_dir():
        raise ValueError(f"workspace {str(path)!r} is not an existing folder")
    return path.resolve()


def resolve_environment(from_cli: str | None, from_config: str | None, project: Project) -> Environment | None:
    """Which environment: the command line wins, then the agent file. "none" (or neither) means none."""
    name = from_cli if from_cli is not None else from_config
    if name is None or name == "none":
        return None
    return load_environment(project.environments_dir, name, project.root)


def load_agent_config(path: str | Path) -> dict:
    """Read an agent file: YAML settings between '---' lines, then the system prompt.

        ---
        model: Qwen3.8-27B-MLX-8bit     <- settings (YAML)
        temperature: 0.6
        ---
        You are a coding agent...       <- everything below is the system prompt

    Returns the settings as a dict, plus a "system_prompt" key holding the body.
    """

    # reads a path and each line
    lines = Path(path).read_text().splitlines()

    # The settings block must open on the very first line and close on a later "---" line.
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"{path}: must start with a '---' line")
    try:
        # strip() so trailing whitespace on the closing line is tolerated, same as the opening one
        end = [line.strip() for line in lines].index("---", 1)
    except ValueError:
        raise ValueError(f"{path}: no closing '---' line after the settings") from None

    # safe_load only builds plain data (dicts, lists, strings, numbers); plain yaml.load
    # can construct arbitrary Python objects, which you never want from a file.
    
    # loads everything between "---" as settings dict
    settings = yaml.safe_load("\n".join(lines[1:end])) or {}
    if not isinstance(settings, dict):
        raise ValueError(f"{path}: settings must be 'key: value' lines")

    # loads all lines after the last "---" as systems prompt
    settings["system_prompt"] = "\n".join(lines[end + 1 :]).strip()
    return settings
