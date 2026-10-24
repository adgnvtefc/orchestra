"""sandbox.py: what a Seatbelt-sandboxed command can and can't do. Runs real sandbox-exec (macOS only)."""

import subprocess
from pathlib import Path

import pytest

from orchestra import sandbox
from orchestra.sandbox import SandboxRules, Seatbelt, Unsandboxed, seatbelt_profile

macos_only = pytest.mark.skipif(not sandbox.available(), reason="needs macOS sandbox-exec")


@pytest.fixture
def layout(tmp_path):
    """A stand-in home folder with a workspace inside it (like sessions/<id>/workspace in the
    real home), a secret next to it, and a toolchain the workspace may read."""
    root = tmp_path.resolve()
    home = root / "home"
    workspace = home / "project" / "sessions" / "s1" / "workspace"
    workspace.mkdir(parents=True)
    (home / "secret.txt").write_text("TOP SECRET\n")
    (home / "project" / ".env").write_text("OMLX_API_KEY=sk-x\n")
    toolchain = home / ".local" / "share" / "uv"
    toolchain.mkdir(parents=True)
    (toolchain / "python.txt").write_text("toolchain\n")
    (root / "outside").mkdir()
    return type("Layout", (), {"root": root, "home": home, "workspace": workspace, "toolchain": toolchain})


def run(box, command, workspace):
    """Run `command` the way bash will: prepared by the sandbox, in the workspace."""
    argv, extra_env = box.prepare(command, workspace)
    result = subprocess.run(
        argv, cwd=workspace, env={"PATH": "/usr/bin:/bin", **extra_env}, capture_output=True, text=True, timeout=30
    )
    return result.returncode, (result.stdout + result.stderr).strip()


@macos_only
class TestWrites:
    def test_inside_workspace(self, layout):
        code, out = run(Seatbelt(home=layout.home), "echo ok > f.txt && cat f.txt", layout.workspace)
        assert (code, out) == (0, "ok")

    def test_subfolders_of_the_workspace(self, layout):
        code, _ = run(Seatbelt(home=layout.home), "mkdir -p a/b && echo x > a/b/c.txt", layout.workspace)
        assert code == 0 and (layout.workspace / "a" / "b" / "c.txt").exists()

    @pytest.mark.parametrize("target", ["../../../../secret_copy.txt", "../sibling.txt"])
    def test_elsewhere_in_home_is_blocked(self, layout, target):
        code, out = run(Seatbelt(home=layout.home), f"echo pwned > {target}", layout.workspace)
        assert code != 0 and "Operation not permitted" in out
        assert not (layout.workspace / target).resolve().exists()

    def test_outside_home_is_blocked_too(self, layout):
        # e.g. /tmp: writes are confined to the workspace, not just kept out of home
        code, out = run(Seatbelt(home=layout.home), f"echo pwned > {layout.root / 'outside' / 'x.txt'}", layout.workspace)
        assert code != 0 and "Operation not permitted" in out

    def test_dev_null(self, layout):
        assert run(Seatbelt(home=layout.home), "echo hidden > /dev/null && echo ok", layout.workspace) == (0, "ok")

    def test_temp_files_go_to_the_workspace(self, layout):
        code, out = run(Seatbelt(home=layout.home), 'python3 -c "import tempfile; print(tempfile.mkdtemp())"', layout.workspace)
        assert code == 0 and out.startswith(str(layout.workspace / ".tmp"))


@macos_only
class TestReads:
    def test_workspace(self, layout):
        (layout.workspace / "mine.txt").write_text("mine\n")
        assert run(Seatbelt(home=layout.home), "cat mine.txt", layout.workspace) == (0, "mine")

    @pytest.mark.parametrize("relative", ["secret.txt", "project/.env"])
    def test_rest_of_home_is_blocked(self, layout, relative):
        code, out = run(Seatbelt(home=layout.home), f"cat {layout.home / relative}", layout.workspace)
        assert code != 0 and "Operation not permitted" in out

    def test_other_sessions_are_blocked(self, layout):
        other = layout.home / "project" / "sessions" / "s2"
        other.mkdir(parents=True)
        (other / "messages.jsonl").write_text("{}\n")
        code, out = run(Seatbelt(home=layout.home), f"cat {other / 'messages.jsonl'}", layout.workspace)
        assert code != 0 and "Operation not permitted" in out

    def test_listing_home_is_blocked(self, layout):
        # The old rules allowed this by accident (an exception for the home folder itself).
        code, out = run(Seatbelt(home=layout.home), f"ls {layout.home}", layout.workspace)
        assert code != 0 and "Operation not permitted" in out and "secret.txt" not in out

    def test_listing_other_sessions_is_blocked(self, layout):
        sessions = layout.home / "project" / "sessions"
        (sessions / "someone-elses-session").mkdir(parents=True)
        code, out = run(Seatbelt(home=layout.home), f"ls {sessions}", layout.workspace)
        assert code != 0 and "someone-elses-session" not in out

    def test_existence_is_visible_contents_are_not(self, layout):
        # The documented trade-off: metadata (exists, size) yes; contents no.
        secret = layout.home / "secret.txt"
        assert run(Seatbelt(home=layout.home), f"test -e {secret} && echo exists", layout.workspace) == (0, "exists")
        code, out = run(Seatbelt(home=layout.home), f"cat {secret}", layout.workspace)
        assert code != 0 and "TOP SECRET" not in out

    def test_tools_can_look_for_config_up_the_tree(self, layout):
        # pytest, git, uv... walk up asking "is there a config file here?". The answer must be
        # an ordinary "no", not PermissionError, or they crash (what happened before this fix).
        script = (
            "import pathlib; p = pathlib.Path.cwd()\n"
            "print(all(not (d / 'conftest.py').is_file() for d in [p, *p.parents]))"
        )
        assert run(Seatbelt(home=layout.home), f'python3 -c "{script}"', layout.workspace) == (0, "True")

    def test_git_sees_no_repository(self, layout):
        code, out = run(Seatbelt(home=layout.home), "git status", layout.workspace)
        assert "not a git repository" in out

    def test_toolchains_need_to_be_allowed(self, layout):
        # Nothing in home is readable by default, toolchains included...
        code, out = run(Seatbelt(home=layout.home), f"cat {layout.toolchain / 'python.txt'}", layout.workspace)
        assert code != 0 and "Operation not permitted" in out
        # ...until the environment's rules allow it.
        allowed = Seatbelt(home=layout.home).with_rules(SandboxRules(readable=(layout.toolchain,)))
        assert run(allowed, f"cat {layout.toolchain / 'python.txt'}", layout.workspace) == (0, "toolchain")

    def test_system_tools_work(self, layout):
        code, out = run(Seatbelt(home=layout.home), 'python3 -c "print(6 * 7)" && git --version', layout.workspace)
        assert code == 0 and out.startswith("42")

    def test_real_home_is_protected_by_default(self, tmp_path):
        # With the default home (yours), files there, like this project's own, are unreadable.
        project_file = Path(__file__).resolve().parents[1] / "pyproject.toml"
        if not project_file.is_relative_to(Path.home().resolve()):
            pytest.skip("project isn't inside the home folder")
        code, out = run(Seatbelt(), f"cat {project_file}", tmp_path.resolve())
        assert code != 0 and "Operation not permitted" in out


@macos_only
class TestNetwork:
    CONNECT = "python3 -c \"import socket; socket.create_connection(('1.1.1.1', 443), timeout=3)\""

    def test_blocked_by_default(self, layout):
        code, out = run(Seatbelt(home=layout.home), self.CONNECT, layout.workspace)
        # EPERM: refused by the sandbox (offline would be a different error, like ENETUNREACH)
        assert code != 0 and "PermissionError" in out

    def test_profile_allows_when_asked(self, layout):
        assert "network" not in seatbelt_profile(layout.workspace, home=layout.home, network=True)

    def test_rules_turn_network_on(self, layout):
        allowed = Seatbelt(home=layout.home).with_rules(SandboxRules(network=True))
        assert allowed.network is True


class TestProfile:
    def test_last_rule_wins_order(self, tmp_path):
        # each broad deny comes before its exceptions; sandbox-exec applies the last match
        rules = seatbelt_profile(tmp_path / "ws", home=tmp_path, network=False).splitlines()
        assert rules.index("(deny file-write*)") < next(i for i, r in enumerate(rules) if r.startswith("(allow file-write*"))
        assert next(i for i, r in enumerate(rules) if r.startswith("(deny file-read-data")) < next(
            i for i, r in enumerate(rules) if r.startswith("(allow file-read-data")
        )

    def test_quotes_and_backslashes_are_escaped(self, tmp_path):
        tricky = tmp_path / 'we"ird\\name'
        profile = seatbelt_profile(tricky, home=tmp_path, network=False)
        assert 'we\\"ird\\\\name' in profile

    @macos_only
    def test_workspace_with_quote_in_name_works(self, tmp_path):
        workspace = tmp_path.resolve() / 'a "quoted" folder'
        workspace.mkdir()
        assert run(Seatbelt(home=tmp_path.resolve()), "echo ok > f.txt && cat f.txt", workspace) == (0, "ok")


class TestRules:
    def test_default_needs_nothing(self):
        assert SandboxRules() == SandboxRules(readable=(), network=False)

    def test_written_form_round_trips(self, tmp_path):
        rules = SandboxRules(readable=(tmp_path.resolve() / "tool",), network=True)
        assert rules.to_dict() == {"readable": [str(tmp_path.resolve() / "tool")], "network": True}
        assert SandboxRules.from_dict(rules.to_dict()) == rules

    def test_from_dict_paths(self, tmp_path):
        rules = SandboxRules.from_dict({"readable": ["~/.local/share/uv", "rel/bin"]}, relative_to=tmp_path)
        assert rules.readable == ((Path.home() / ".local/share/uv").resolve(), (tmp_path / "rel" / "bin").resolve())

    def test_from_dict_empty_means_nothing(self):
        assert SandboxRules.from_dict({}) == SandboxRules()

    @pytest.mark.parametrize(("data", "error"), [
        ("yes", "must be 'key: value' lines"),
        ({"netwrok": True}, "unknown key"),
        ({"readable": "~/x"}, "readable must be a list"),
        ({"network": "sometimes"}, "network must be true or false"),
    ])
    def test_from_dict_rejects(self, data, error):
        with pytest.raises(ValueError, match=error):
            SandboxRules.from_dict(data)

    @macos_only
    def test_with_rules_returns_a_new_sandbox(self, tmp_path):
        base = Seatbelt(home=tmp_path)
        loosened = base.with_rules(SandboxRules(readable=(tmp_path / "tool",), network=True))
        assert (base.network, base.readable) == (False, ())  # the original is unchanged (frozen)
        assert (loosened.network, loosened.readable, loosened.home) == (True, (tmp_path / "tool",), tmp_path)

    def test_unsandboxed_ignores_rules(self):
        assert Unsandboxed().with_rules(SandboxRules(network=True)) == Unsandboxed()


class TestUnsandboxed:
    def test_plain_shell(self, tmp_path):
        assert Unsandboxed().prepare("echo hi", tmp_path) == (["/bin/sh", "-c", "echo hi"], {})

    def test_name(self):
        assert (Unsandboxed().name, Seatbelt.name) == ("none", "seatbelt")


def test_seatbelt_refuses_where_unavailable(monkeypatch):
    monkeypatch.setattr(sandbox, "available", lambda: False)
    with pytest.raises(RuntimeError, match="needs macOS"):
        Seatbelt()
