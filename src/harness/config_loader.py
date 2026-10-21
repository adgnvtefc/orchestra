"""Config: read agent and environment files, and resolve them into what one run uses.

Reads files and checks folders exist; builds nothing (no sessions, clients, or tools).
Everything a run is configured by goes through resolve_run_config(), so a typo in any
of it fails before anything is created.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

from harness.environment import Environment, load_environment
from harness.tools import Toolset

# Where config lives, relative to this file, so `uv run harness` works from any directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENTS_DIR = PROJECT_ROOT / "agents"
ENVIRONMENTS_DIR = PROJECT_ROOT / "environments"


@dataclass(frozen=True)
class RunConfig:
    """Everything a run is configured by, resolved and checked."""

    agent: dict  # the resolved agent config: file + name + environment {name, version}; what agent_version hashes
    system_prompt: str  # the agent's prompt, plus the environment's
    workspace: Path | None  # an existing folder, or None for a fresh per-session one
    environment: Environment | None


def list_agents() -> list[str]:
    """Names of the agents defined in agents/, sorted."""
    return sorted(path.stem for path in AGENTS_DIR.glob("*.md"))


def resolve_agent(agent_name: str) -> dict:
    """Read agents/<name>.md and check its tool names. Just the agent's own settings: a
    resumed session uses this, since its workspace and environment are already fixed."""
    agent = {"name": agent_name, **load_agent_config(AGENTS_DIR / f"{agent_name}.md")}
    Toolset.check_names(agent.get("tools", []))  # no `tools:` key means no tools: access is opt-in
    return agent


def resolve_run_config(agent_name: str, *, workspace: Path | None = None, environment: str | None = None) -> RunConfig:
    """Read the agent file and apply the command line's overrides; check everything.

    `workspace` / `environment` come from the command line and win over the agent file's.
    """
    agent = resolve_agent(agent_name)
    folder = resolve_workspace(workspace, agent.get("workspace"))
    env = resolve_environment(environment, agent.get("environment"))

    # The resolved config records what the run actually used: the environment's name *and*
    # version, so agent_version changes whenever the environment's content does.
    agent["environment"] = {"name": env.name, "version": env.version} if env else None
    system_prompt = agent["system_prompt"] + (f"\n\n{env.prompt}" if env and env.prompt else "")
    return RunConfig(agent=agent, system_prompt=system_prompt, workspace=folder, environment=env)


def resolve_workspace(from_cli: Path | None, from_config: str | None) -> Path | None:
    """Which folder to work in: the command line wins, then the agent file; None means a fresh one.

    A path in an agent file may use ~ and is relative to the project root, so it means
    the same thing wherever you run from. The folder must already exist: a typo should
    fail here, not quietly create a new empty folder somewhere.
    """
    if from_cli is not None:
        path = from_cli
    elif from_config is not None:
        path = PROJECT_ROOT / Path(from_config).expanduser()  # an absolute path replaces PROJECT_ROOT
    else:
        return None
    if not path.is_dir():
        raise ValueError(f"workspace {str(path)!r} is not an existing folder")
    return path.resolve()


def resolve_environment(from_cli: str | None, from_config: str | None) -> Environment | None:
    """Which environment: the command line wins, then the agent file. "none" (or neither) means none."""
    name = from_cli if from_cli is not None else from_config
    if name is None or name == "none":
        return None
    return load_environment(ENVIRONMENTS_DIR, name, PROJECT_ROOT)


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
