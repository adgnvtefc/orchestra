"""Lint the real agent definitions in agents/, so a broken config fails here instead of at runtime."""

from pathlib import Path

import pytest

from harness.sandbox import Unsandboxed
from harness.config import AGENTS_DIR, load_agent_config
from harness.tools import ToolContext, Toolset
from tests.fakes import NO_HUMAN

AGENT_FILES = sorted(AGENTS_DIR.glob("*.md"))
REQUIRED = {"model": str, "temperature": (int, float), "max_steps": int, "system_prompt": str}


def test_agents_dir_has_agents():
    assert AGENT_FILES, f"no agent files found in {AGENTS_DIR}"


def test_default_agent_exists():
    assert (AGENTS_DIR / "base.md").is_file()


@pytest.mark.parametrize("path", AGENT_FILES, ids=lambda p: p.stem)
class TestAgentFile:
    def test_loads(self, path):
        load_agent_config(path)

    def test_has_required_settings_with_right_types(self, path):
        config = load_agent_config(path)
        for key, expected_type in REQUIRED.items():
            assert key in config, f"{path.name} is missing {key!r}"
            assert isinstance(config[key], expected_type), f"{path.name}: {key} should be {expected_type}"

    def test_values_in_range(self, path):
        config = load_agent_config(path)
        assert 0 <= config["temperature"] <= 2
        assert config["max_steps"] >= 1
        assert config["system_prompt"].strip()

    def test_tools_exist(self, path):
        tools = load_agent_config(path).get("tools", [])
        assert isinstance(tools, list)
        Toolset(tools, NO_HUMAN, ToolContext(workspace=Path("."), sandbox=Unsandboxed()))  # raises on any unknown tool name
