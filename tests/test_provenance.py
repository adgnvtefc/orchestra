"""Provenance: agent version hashes, and which orchestra code is running."""

import re
import subprocess

import pytest

from orchestra import provenance
from orchestra.provenance import agent_version, describe_install, find_source_checkout, git_state

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


class TestOrchestraVersion:
    def test_fields(self):
        assert {"version", "installed", "commit", "dirty"} <= set(provenance.orchestra_version())

    def test_measured_once_per_process(self, monkeypatch):
        provenance.orchestra_version.cache_clear()
        calls = []
        monkeypatch.setattr(provenance, "git_state", lambda repo: calls.append(repo) or {"commit": "c", "dirty": False})
        provenance.orchestra_version()
        provenance.orchestra_version()
        assert len(calls) == 1
        provenance.orchestra_version.cache_clear()  # don't leak the fake value into other tests

    def test_this_repo_is_measured(self):
        # The real repo: a 40-char commit, and a dirty flag that's a real bool.
        version = provenance.orchestra_version()
        assert re.fullmatch(r"[0-9a-f]{40}", version["commit"])
        assert isinstance(version["dirty"], bool)


class TestDescribeInstall:
    """Which harness code ran, for each way it can be installed."""

    def test_source_checkout_uses_git_there(self, repo):
        record = describe_install("0.1.0", repo, None)
        assert record["installed"] == "checkout"
        assert record["version"] == "0.1.0"
        assert re.fullmatch(r"[0-9a-f]{40}", record["commit"]) and record["dirty"] is False

    def test_git_install_uses_the_recorded_commit(self):
        direct = {"url": "https://github.com/x/harness", "vcs_info": {"vcs": "git", "commit_id": "a" * 40}}
        assert describe_install("0.2.0", None, direct) == {
            "version": "0.2.0", "installed": "git", "commit": "a" * 40, "dirty": False,
            "source": "https://github.com/x/harness",
        }

    def test_path_install_admits_the_commit_is_unknown(self):
        direct = {"url": "file:///some/folder", "dir_info": {}}
        assert describe_install("0.1.0", None, direct) == {
            "version": "0.1.0", "installed": "path", "commit": None, "dirty": None, "source": "file:///some/folder",
        }

    def test_nothing_known(self):
        assert describe_install(None, None, None)["commit"] is None


class TestFindSourceCheckout:
    def write_project(self, folder, name):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "pyproject.toml").write_text(f'[project]\nname = "{name}"\n')

    def test_inside_the_harness_repo(self, tmp_path):
        self.write_project(tmp_path, "orchestra")
        code = tmp_path / "src" / "orchestra" / "provenance.py"
        code.parent.mkdir(parents=True)
        code.write_text("")
        assert find_source_checkout(code) == tmp_path.resolve()

    def test_installed_inside_another_project_is_not_a_checkout(self, tmp_path):
        # The bug this exists to prevent: an installed copy inside another project's .venv
        # must never be mistaken for the harness repo (git there gives *that* project's commit).
        self.write_project(tmp_path, "consumer")
        code = tmp_path / ".venv" / "lib" / "python3.12" / "site-packages" / "orchestra" / "provenance.py"
        code.parent.mkdir(parents=True)
        code.write_text("")
        assert find_source_checkout(code) is None

    def test_unreadable_pyproject(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("this is [ not toml")
        code = tmp_path / "x.py"
        code.write_text("")
        assert find_source_checkout(code) is None

    def test_this_repo(self):
        from pathlib import Path
        import orchestra.provenance as module
        root = Path(module.__file__).resolve().parents[2]
        assert find_source_checkout(Path(module.__file__)) == root


class TestInstallFacts:
    def test_no_pyproject_anywhere_above(self):
        from pathlib import Path
        assert find_source_checkout(Path("/not-a-project-file.py")) is None

    def test_package_not_installed(self, monkeypatch):
        import importlib.metadata
        def missing(name):
            raise importlib.metadata.PackageNotFoundError(name)
        monkeypatch.setattr(importlib.metadata, "version", missing)
        monkeypatch.setattr(importlib.metadata, "distribution", missing)
        assert provenance._package_version() is None
        assert provenance._direct_url() is None

    def test_installed_without_a_record(self, monkeypatch):
        import importlib.metadata
        fake = type("Dist", (), {"read_text": lambda self, name: None})()
        monkeypatch.setattr(importlib.metadata, "distribution", lambda name: fake)
        assert provenance._direct_url() is None

    def test_installed_with_a_record(self, monkeypatch):
        import importlib.metadata
        fake = type("Dist", (), {"read_text": lambda self, name: '{"url": "file:///x", "dir_info": {}}'})()
        monkeypatch.setattr(importlib.metadata, "distribution", lambda name: fake)
        assert provenance._direct_url() == {"url": "file:///x", "dir_info": {}}
