"""Policy: "allow" runs, "ask" goes to the approver; plus the approvers themselves."""

import pytest

from orchestra.interfaces.cli import terminal_approver
from orchestra.policy import (
    Decision, NobodyToAsk, Policy, approve_all, asks_human, deny_all, require_sandbox_for_auto_approval,
)
from orchestra.sandbox import Unsandboxed
from orchestra.tools import Tool


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
        assert Policy(approver).check(make_tool(permission="allow"), {}) == Decision("auto-allow", None)
        assert approver.asked == []

    def test_ask_approved(self):
        approver = recording_approver(True)
        assert Policy(approver).check(make_tool(permission="ask"), {"x": 1}) == Decision("auto-allow", None)

    def test_ask_declined_passes_on_the_approvers_message(self):
        # The approver knows why it said no, so its message is what the model sees.
        assert Policy(recording_approver(False)).check(make_tool("bash", "ask"), {}) == Decision("auto-deny", DECLINED)

    def test_approver_sees_name_and_arguments(self):
        approver = recording_approver(True)
        Policy(approver).check(make_tool("bash", "ask"), {"command": "ls"})
        assert approver.asked == [("bash", {"command": "ls"})]


class TestWhoDecided:
    """Every decision says who made it: auto-allow, auto-deny, human-allow, human-deny."""

    @pytest.mark.parametrize(("approver", "approval"), [(approve_all, "auto-allow"), (deny_all, "auto-deny")])
    def test_automatic_approvers(self, approver, approval):
        assert Policy(approver).check(make_tool("bash", "ask"), {}).approval == approval

    @pytest.mark.parametrize(("approves", "approval"), [(True, "human-allow"), (False, "human-deny")])
    def test_an_approver_that_asks_a_person(self, approves, approval):
        approver = asks_human(recording_approver(approves))
        assert Policy(approver).check(make_tool("bash", "ask"), {}).approval == approval

    def test_allow_tools_are_automatic_even_with_a_human_there(self):
        # nobody is asked about an "allow" tool, so nobody decided
        approver = asks_human(recording_approver(False))
        assert Policy(approver).check(make_tool(permission="allow"), {}).approval == "auto-allow"

    def test_nobody_to_ask_is_an_automatic_refusal(self):
        @asks_human
        def no_terminal(name, arguments):
            raise NobodyToAsk

        decision = Policy(no_terminal).check(make_tool("bash", "ask"), {})
        assert decision == Decision("auto-deny", deny_all("bash", {}))

    def test_terminal_approver_asks_a_person(self):
        assert terminal_approver.asks_human is True


class TestName:
    def test_defaults_to_the_approvers_name(self):
        assert Policy(deny_all).name == "deny_all"
        assert Policy(terminal_approver).name == "terminal_approver"

    def test_explicit_name_for_an_anonymous_approver(self):
        assert Policy(lambda name, args: None, name="auto_approve").name == "auto_approve"


class TestApproveAll:
    def test_always_approves(self):
        assert approve_all("bash", {"command": "rm -rf /"}) is None

    def test_name_in_traces(self):
        assert Policy(approve_all).name == "approve_all"


class TestAutoApprovalNeedsASandbox:
    class Sandboxed:
        name = "seatbelt"

    def test_refused_without_a_sandbox(self):
        with pytest.raises(ValueError, match="needs the sandbox"):
            require_sandbox_for_auto_approval(approve_all, Unsandboxed())

    def test_allowed_with_a_sandbox(self):
        require_sandbox_for_auto_approval(approve_all, self.Sandboxed())  # no error

    @pytest.mark.parametrize("approver", [deny_all, terminal_approver])
    def test_other_approvers_never_need_one(self, approver):
        require_sandbox_for_auto_approval(approver, Unsandboxed())  # no error


class TestDenyAll:
    def test_always_declines(self):
        assert deny_all("bash", {"command": "ls"}) is not None

    def test_message_explains_the_situation(self):
        # Tells the model nobody is around, so it won't retry or offer to ask the user.
        message = deny_all("bash", {"command": "ls"})
        assert message.startswith("bash is unavailable in this run")
        assert "nobody is available" in message
        assert "Don't retry" in message
