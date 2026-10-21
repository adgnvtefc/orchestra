"""tools/base.py: the Tool dataclass and truncate()."""

import dataclasses

import pytest

from harness.tools.base import MAX_OUTPUT_CHARS, Tool, truncate


class TestTool:
    def test_name_comes_from_schema(self):
        tool = Tool(fn=lambda: "", schema={"type": "function", "function": {"name": "abc"}}, permission="allow")
        assert tool.name == "abc"

    def test_is_frozen(self):
        tool = Tool(fn=lambda: "", schema={"function": {"name": "abc"}}, permission="allow")
        with pytest.raises(dataclasses.FrozenInstanceError):
            tool.fn = lambda: "other"

    def test_permission_is_required(self):
        # No default value: a tool that forgets to declare one fails at definition time.
        with pytest.raises(TypeError, match="permission"):
            Tool(fn=lambda: "", schema={"function": {"name": "abc"}})


class TestToolContext:
    def test_sandbox_is_required(self, tmp_path):
        # No default: forgetting it is an error, not a silently unsandboxed bash.
        from harness.tools import ToolContext
        with pytest.raises(TypeError, match="sandbox"):
            ToolContext(workspace=tmp_path)


class TestTruncate:
    def test_short_text_unchanged(self):
        assert truncate("hello") == "hello"

    def test_empty(self):
        assert truncate("") == ""

    def test_exactly_at_limit_unchanged(self):
        text = "x" * MAX_OUTPUT_CHARS
        assert truncate(text) == text

    def test_one_over_limit_is_truncated(self):
        assert truncate("x" * (MAX_OUTPUT_CHARS + 1)) != "x" * (MAX_OUTPUT_CHARS + 1)

    def test_keeps_head_and_tail(self):
        # Errors and summaries live at the end of long output, so the tail must survive.
        text = "HEAD" + "x" * (MAX_OUTPUT_CHARS * 3) + "TAIL"
        out = truncate(text)
        assert out.startswith("HEAD")
        assert out.endswith("TAIL")

    def test_reports_omitted_count(self):
        out = truncate("x" * (MAX_OUTPUT_CHARS + 1234))
        assert "[1234 characters omitted]" in out

    def test_output_is_bounded(self):
        out = truncate("x" * 1_000_000)
        assert len(out) < MAX_OUTPUT_CHARS + 100  # the limit plus the short marker
