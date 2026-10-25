"""Records: the envelope every record has, checking the caller's input, and where records go."""

import json
import re
import threading

import pytest

from orchestra import provenance, records


def read_log(orchestra_dir, app):
    """Every record saved for `app`, oldest first."""
    lines = []
    for path in sorted((orchestra_dir / app / "log").glob("*.jsonl")):
        lines += path.read_text().splitlines()
    return [json.loads(line) for line in lines]


class TestStart:
    def test_envelope(self):
        record = records.start("chat", app="jewels", metadata={"job": "3f2a"}, parent="tool_x", session="s1")
        assert record["v"] == records.FORMAT_VERSION
        assert re.fullmatch(r"chat_\d{4}-\d\d-\d\dT\d\d-\d\d-\d\d_[0-9a-f]{6}", record["id"])
        assert (record["app"], record["kind"], record["parent"], record["session"]) == ("jewels", "chat", "tool_x", "s1")
        assert record["metadata"] == {"job": "3f2a"}
        assert (record["seconds"], record["error"]) == (None, None)  # filled in when the work ends
        assert record["orchestra"] == provenance.orchestra_version()

    def test_id_says_its_day(self):
        record = records.start("chat", app="jewels")
        assert record["id"].split("_")[1][:10] == record["time"][:10]

    def test_defaults(self):
        record = records.start("chat", app="jewels")
        assert (record["metadata"], record["parent"], record["session"]) == ({}, None, None)

    def test_ids_are_unique(self):
        assert len({records.start("chat", app="a")["id"] for _ in range(200)}) == 200

    @pytest.mark.parametrize("app", ["jewels", "orchestra", "my_app.v2", "A1"])
    def test_plain_app_names(self, app):
        assert records.start("chat", app=app)["app"] == app

    @pytest.mark.parametrize("app", ["", "../escape", "a/b", ".hidden", "-x", "has space", None, 3])
    def test_app_names_that_cant_be_folders(self, app):
        with pytest.raises(ValueError, match="app must be a plain name"):
            records.start("chat", app=app)

    def test_sessions_is_not_an_app_name(self):
        # ~/.orchestra/sessions/ holds sessions from before apps
        with pytest.raises(ValueError, match="can't be called 'sessions'"):
            records.start("chat", app="sessions")

    def test_app_dir(self, orchestra_dir):
        assert records.app_dir("jewels") == orchestra_dir / "jewels"
        with pytest.raises(ValueError):
            records.app_dir("../escape")

    @pytest.mark.parametrize("metadata", [["job"], "job=1"])
    def test_metadata_must_be_a_dict(self, metadata):
        with pytest.raises(ValueError, match="metadata must be a dict"):
            records.start("chat", app="jewels", metadata=metadata)

    @pytest.mark.parametrize("metadata", [{"when": object()}, {"tags": {1, 2}}])
    def test_metadata_must_be_storable(self, metadata):
        # checked up front: better to fail now than after the model's time is spent
        with pytest.raises(ValueError, match="storable as JSON"):
            records.start("chat", app="jewels", metadata=metadata)


class TestSave:
    def test_goes_to_the_apps_log_for_its_day(self, orchestra_dir):
        record = records.start("chat", app="jewels")
        path = records.save(record)
        assert path == orchestra_dir / "jewels" / "log" / f"{record['time'][:10]}.jsonl"
        assert read_log(orchestra_dir, "jewels") == [record]

    def test_appends(self, orchestra_dir):
        first, second = records.start("chat", app="jewels"), records.start("chat", app="jewels")
        records.save(first)
        records.save(second)
        assert [r["id"] for r in read_log(orchestra_dir, "jewels")] == [first["id"], second["id"]]

    def test_apps_are_kept_apart(self, orchestra_dir):
        records.save(records.start("chat", app="jewels"))
        records.save(records.start("chat", app="chaos"))
        assert len(read_log(orchestra_dir, "jewels")) == len(read_log(orchestra_dir, "chaos")) == 1

    def test_anything_unencodable_is_written_as_text(self, orchestra_dir):
        record = records.start("chat", app="jewels")
        record["response"] = {"odd": object()}
        records.save(record)
        assert read_log(orchestra_dir, "jewels")[0]["response"]["odd"].startswith("<object object")

    def test_threads_never_interleave_lines(self, orchestra_dir):
        # Guards the result, not the lock: on macOS this passes even without records' lock,
        # because the OS appends each write whole. The lock is for when Python splits a write.
        def save_many():
            for _ in range(20):
                record = records.start("chat", app="jewels")
                record["response"] = "x" * 50_000  # long lines are the ones that could interleave
                records.save(record)

        threads = [threading.Thread(target=save_many) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(read_log(orchestra_dir, "jewels")) == 160  # every line parses as one whole record
