"""Provenance: agent version hashes, harness git state, and what collect() records."""

import re
import subprocess

import pytest

from harness import provenance
from harness.provenance import agent_version, collect, git_state

CONFIG = {"name": "base", "model": "m", "temperature": 0.6, "max_steps": 20, "tools": ["bash"], "system_prompt": "Hi."}


class TestAgentVersion:
    def test_is_12_hex_chars(self):
        assert re.fullmatch(r"[0-9a-f]{12}", agent_version(CONFIG))

    def test_same_content_same_version(self):
        assert agent_version(CONFIG) == agent_version(dict(CONFIG))

    def test_key_order_does_not_matter(self):
        assert agent_version(CONFIG) == agent_version(dict(reversed(list(CONFIG.items()))))

    def test_name_does_not_matter(self):
        # Renaming an agent file doesn't change what the agent does.
        assert agent_version(CONFIG) == agent_version({**CONFIG, "name": "renamed"})

    @pytest.mark.parametrize(
        "change",
        [
            {"system_prompt": "Hi!"},  # one character of the prompt
            {"temperature": 0.7},
            {"model": "other"},
            {"tools": ["bash", "read_file"]},
            {"max_steps": 21},
        ],
    )
    def test_any_content_change_is_a_new_version(self, change):
        assert agent_version(CONFIG) != agent_version({**CONFIG, **change})

    def test_tool_order_matters(self):
        # Tool order changes the request the model sees (and the prompt cache), so it's a different agent.
        assert agent_version({**CONFIG, "tools": ["a", "b"]}) != agent_version({**CONFIG, "tools": ["b", "a"]})

    def test_unicode_prompt(self):
        assert re.fullmatch(r"[0-9a-f]{12}", agent_version({**CONFIG, "system_prompt": "日本語 ✓"}))


def git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    """A real, throwaway git repo with one commit."""
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "test@example.com")
    git(tmp_path, "config", "user.name", "test")
    (tmp_path / "code.py").write_text("x = 1\n")
    (tmp_path / ".gitignore").write_text("ignored/\n")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


class TestGitState:
    def test_clean_repo(self, repo):
        state = git_state(repo)
        assert re.fullmatch(r"[0-9a-f]{40}", state["commit"])
        assert state["dirty"] is False

    def test_modified_file_is_dirty(self, repo):
        (repo / "code.py").write_text("x = 2\n")
        assert git_state(repo)["dirty"] is True

    def test_untracked_file_is_dirty(self, repo):
        # A new, uncommitted module changes the running code just as much as an edit.
        (repo / "new_tool.py").write_text("")
        assert git_state(repo)["dirty"] is True

    def test_ignored_files_do_not_count(self, repo):
        (repo / "ignored").mkdir()
        (repo / "ignored" / "trace.jsonl").write_text("{}")
        assert git_state(repo)["dirty"] is False

    def test_commit_tracks_head(self, repo):
        before = git_state(repo)["commit"]
        (repo / "code.py").write_text("x = 3\n")
        git(repo, "commit", "-qam", "change")
        after = git_state(repo)
        assert after["commit"] != before
        assert after["dirty"] is False

    def test_not_a_repo_is_unknown(self, tmp_path):
        assert git_state(tmp_path) == {"commit": None, "dirty": None}

    def test_no_git_installed_is_unknown(self, repo, monkeypatch):
        monkeypatch.setenv("PATH", "")
        assert git_state(repo) == {"commit": None, "dirty": None}


class TestCollect:
    def test_fields(self):
        record = collect(CONFIG)
        assert set(record) == {"agent_version", "harness"}
        assert record["agent_version"] == agent_version(CONFIG)
        assert set(record["harness"]) == {"commit", "dirty"}

    def test_harness_version_is_measured_once_per_process(self, monkeypatch):
        provenance.harness_version.cache_clear()
        calls = []
        monkeypatch.setattr(provenance, "git_state", lambda repo: calls.append(repo) or {"commit": "c", "dirty": False})
        collect(CONFIG)
        collect(CONFIG)
        assert len(calls) == 1
        provenance.harness_version.cache_clear()  # don't leak the fake value into other tests

    def test_this_repo_is_measured(self):
        # The real harness repo: a 40-char commit, and a dirty flag that's a real bool.
        harness = collect(CONFIG)["harness"]
        assert re.fullmatch(r"[0-9a-f]{40}", harness["commit"])
        assert isinstance(harness["dirty"], bool)
