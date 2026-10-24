"""Filesystem tools: read_file, write_file, and the workspace boundary they share.

The `workspace` fixture (conftest.py, autouse) points every test at a temp directory.
"""

import os

import pytest

from orchestra.tools.base import MAX_OUTPUT_CHARS
from orchestra.tools.builtin.fs import _resolve, read_file, write_file

# Paths a model might send, that must never reach outside the workspace.
ESCAPES = [
    "..",
    "../secret.txt",
    "../../etc/passwd",
    "/etc/passwd",
    "sub/../../secret.txt",
    "./../secret.txt",
    "sub/./../../secret.txt",
]


class TestResolve:
    @pytest.mark.parametrize("path", ["a.txt", "sub/a.txt", "./a.txt", "sub/../a.txt", "a/b/c/d.txt"])
    def test_paths_inside_are_allowed(self, ctx, workspace, path):
        assert _resolve(ctx, path).is_relative_to(workspace)

    def test_workspace_itself_is_allowed(self, ctx, workspace):
        assert _resolve(ctx, ".") == workspace

    @pytest.mark.parametrize("path", ESCAPES)
    def test_paths_outside_are_blocked(self, ctx, path):
        with pytest.raises(PermissionError, match="outside the workspace"):
            _resolve(ctx, path)

    def test_symlink_pointing_outside_is_blocked(self, ctx, workspace, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("secret")
        (workspace / "link").symlink_to(outside)
        with pytest.raises(PermissionError):
            _resolve(ctx, "link/secret.txt")

    def test_symlink_pointing_inside_is_allowed(self, ctx, workspace):
        (workspace / "real").mkdir()
        (workspace / "alias").symlink_to(workspace / "real")
        assert _resolve(ctx, "alias/x.txt") == workspace / "real" / "x.txt"

    def test_prefix_sibling_directory_is_blocked(self, ctx, workspace):
        # "workspace_evil" starts with the string "workspace"; a naive startswith() check would allow it.
        (workspace.parent / "workspace_evil").mkdir()
        with pytest.raises(PermissionError):
            _resolve(ctx, "../workspace_evil/x.txt")


class TestReadFile:
    def test_reads_text(self, ctx, workspace):
        (workspace / "a.txt").write_text("hello\nworld")
        assert read_file(ctx, "a.txt") == "hello\nworld"

    def test_reads_nested(self, ctx, workspace):
        (workspace / "sub").mkdir()
        (workspace / "sub" / "a.txt").write_text("nested")
        assert read_file(ctx, "sub/a.txt") == "nested"

    def test_unicode(self, ctx, workspace):
        (workspace / "u.txt").write_text("héllo ✓ 日本")
        assert read_file(ctx, "u.txt") == "héllo ✓ 日本"

    def test_empty_file_says_so(self, ctx, workspace):
        (workspace / "empty.txt").write_text("")
        assert read_file(ctx, "empty.txt") == "empty.txt is empty."

    def test_binary_file_says_so(self, ctx, workspace):
        (workspace / "img.bin").write_bytes(bytes(range(256)))
        assert read_file(ctx, "img.bin") == "img.bin is a binary file and can't be shown as text."

    def test_long_file_is_truncated(self, ctx, workspace):
        (workspace / "big.txt").write_text("x" * (MAX_OUTPUT_CHARS * 2))
        assert "characters omitted" in read_file(ctx, "big.txt")

    def test_missing_file_raises(self, ctx):
        with pytest.raises(FileNotFoundError):
            read_file(ctx, "nope.txt")

    def test_directory_raises(self, ctx, workspace):
        (workspace / "sub").mkdir()
        with pytest.raises(IsADirectoryError):
            read_file(ctx, "sub")

    @pytest.mark.parametrize("path", ESCAPES)
    def test_cannot_read_outside(self, ctx, path):
        with pytest.raises(PermissionError):
            read_file(ctx, path)

    def test_cannot_read_real_file_outside(self, ctx, workspace):
        # Not just "the path looks bad": a real file next to the workspace stays unreadable.
        (workspace.parent / "secret.txt").write_text("TOP SECRET")
        with pytest.raises(PermissionError):
            read_file(ctx, "../secret.txt")


class TestWriteFile:
    def test_writes_and_confirms(self, ctx, workspace):
        assert write_file(ctx, "a.txt", "hello") == "Wrote 5 characters to a.txt."
        assert (workspace / "a.txt").read_text() == "hello"

    def test_creates_parent_dirs(self, ctx, workspace):
        write_file(ctx, "a/b/c.txt", "deep")
        assert (workspace / "a" / "b" / "c.txt").read_text() == "deep"

    def test_overwrites_existing(self, ctx, workspace):
        (workspace / "a.txt").write_text("old content that is longer")
        write_file(ctx, "a.txt", "new")
        assert (workspace / "a.txt").read_text() == "new"

    def test_empty_content(self, ctx, workspace):
        assert write_file(ctx, "e.txt", "") == "Wrote 0 characters to e.txt."
        assert (workspace / "e.txt").read_text() == ""

    def test_special_characters_round_trip(self, ctx):
        # The reason write_file exists: no shell quoting, so content arrives byte-for-byte.
        content = 'print("it\'s $5")\n`cmd` \\n $(rm -rf /) "EOF"\n'
        write_file(ctx, "tricky.py", content)
        assert read_file(ctx, "tricky.py") == content

    @pytest.mark.parametrize("path", ESCAPES[1:])  # ".." alone is a directory, not a file target
    def test_cannot_write_outside(self, ctx, workspace, path):
        before = set(os.listdir(workspace.parent))
        with pytest.raises(PermissionError):
            write_file(ctx, path, "pwned")
        assert set(os.listdir(workspace.parent)) == before  # nothing was created next to the workspace

    def test_blocked_write_creates_no_dirs(self, ctx, workspace):
        with pytest.raises(PermissionError):
            write_file(ctx, "../newdir/x.txt", "pwned")
        assert not (workspace.parent / "newdir").exists()
