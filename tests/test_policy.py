"""Policy: "allow" runs, "ask" goes to the approver; plus the approvers themselves."""

import pytest

from harness.policy import Policy, deny_all, terminal_approver
from harness.tools import Tool


def make_tool(name="t", permission="allow") -> Tool:
    return Tool(fn=lambda: "", schema={"type": "function", "function": {"name": name}}, permission=permission)


DECLINED = "declined by test approver"


def recording_approver(approve: bool):
    """An approver that approves or not, and records every (name, arguments) it was asked about."""
    asked = []

    def approver(name, arguments):
        asked.append((name, arguments))
        return None if approve else DECLINED

    approver.asked = asked
    return approver


class TestCheck:
    def test_allow_runs_without_asking(self):
        approver = recording_approver(False)
        assert Policy(approver).check(make_tool(permission="allow"), {}) is None
        assert approver.asked == []

    def test_ask_approved(self):
        approver = recording_approver(True)
        assert Policy(approver).check(make_tool(permission="ask"), {"x": 1}) is None

    def test_ask_declined_passes_on_the_approvers_message(self):
        # The approver knows why it said no, so its message is what the model sees.
        assert Policy(recording_approver(False)).check(make_tool("bash", "ask"), {}) == DECLINED

    def test_approver_sees_name_and_arguments(self):
        approver = recording_approver(True)
        Policy(approver).check(make_tool("bash", "ask"), {"command": "ls"})
        assert approver.asked == [("bash", {"command": "ls"})]


class TestName:
    def test_defaults_to_the_approvers_name(self):
        assert Policy(deny_all).name == "deny_all"
        assert Policy(terminal_approver).name == "terminal_approver"

    def test_explicit_name_for_an_anonymous_approver(self):
        assert Policy(lambda name, args: None, name="auto_approve").name == "auto_approve"


class TestDenyAll:
    def test_always_declines(self):
        assert deny_all("bash", {"command": "ls"}) is not None

    def test_message_explains_the_situation(self):
        # Tells the model nobody is around, so it won't retry or offer to ask the user.
        message = deny_all("bash", {"command": "ls"})
        assert message.startswith("bash is unavailable in this run")
        assert "nobody is available" in message
        assert "Don't retry" in message


class TestTerminalApprover:
    @pytest.mark.parametrize("answer", ["y", "Y", " y ", "y\n"])
    def test_y_approves(self, answer_prompt, answer):
        answer_prompt(answer)
        assert terminal_approver("bash", {"command": "ls"}) is None

    @pytest.mark.parametrize("answer", ["", "n", "N", "no", "yes", "yy", "sure"])
    def test_anything_else_declines(self, answer_prompt, answer):
        # Only an explicit "y" approves. Even "yes" declines: strict beats surprising.
        answer_prompt(answer)
        assert terminal_approver("bash", {"command": "ls"}) == (
            "The user declined this bash call. Try a different approach or ask the user."
        )

    def test_no_terminal_declines(self, answer_prompt):
        # No terminal is the same situation as unattended, so it gets the same message.
        answer_prompt(EOFError)
        assert terminal_approver("bash", {"command": "ls"}) == deny_all("bash", {"command": "ls"})

    def test_shows_tool_and_arguments_before_asking(self, approve, capsys):
        terminal_approver("bash", {"command": "echo visible-command", "timeout": 60})
        out = capsys.readouterr().out
        assert "agent wants to run bash" in out
        assert "command: echo visible-command" in out
        assert "timeout: 60" in out
        assert approve == ["  allow? [y/N] "]

    def test_long_argument_values_are_shortened(self, approve, capsys):
        terminal_approver("write_file", {"content": "x" * 5000})
        out = capsys.readouterr().out
        assert "x" * 300 + "..." in out
        assert "x" * 301 not in out
