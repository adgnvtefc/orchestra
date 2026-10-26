"""chat(): one model call, made and recorded together."""

import pytest

from orchestra.providers import omlx
from orchestra import chat
from orchestra.providers import MissingAPIKey
from tests.fakes import FakeClient, call, reply
from tests.test_records import read_log

MESSAGES = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "6 times 7?"}]
USAGE = {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25,
         "prompt_tokens_details": {"cached_tokens": 12}}


def ask(client, **options):
    """chat() against `client`, a scripted FakeClient, as the shared model client."""
    omlx.client = lambda: client  # the autouse no_model_client fixture puts the real one back afterwards
    return chat(app=options.pop("app", "jewels"), model="m", messages=options.pop("messages", MESSAGES), **options)


class TestReply:
    def test_answer(self):
        result = ask(FakeClient([reply("42", reasoning="6*7 is 42", usage=USAGE)]))
        assert (result.content, result.reasoning, result.finish_reason) == ("42", "6*7 is 42", "stop")
        assert result.model == "fake-model"
        assert result.usage == {"prompt": 20, "completion": 5, "cached": 12}
        assert result.tool_calls == []
        assert result.seconds >= 0

    def test_tool_calls_are_plain_dicts(self):
        wanted = call("bash", {"command": "ls"}, id="call_1")
        result = ask(FakeClient([reply(tool_calls=[wanted])]))
        assert result.tool_calls == [wanted]
        assert result.finish_reason == "tool_calls"

    def test_no_usage_reported(self):
        assert ask(FakeClient([reply("hi")])).usage == {"prompt": 0, "completion": 0, "cached": 0}

    def test_reply_id_is_the_records_id(self, orchestra_dir):
        result = ask(FakeClient([reply("hi")]))
        assert read_log(orchestra_dir, "jewels")[0]["id"] == result.id


class TestRequest:
    def test_sends_only_what_was_set(self):
        client = FakeClient([reply("hi")])
        ask(client)
        assert client.requests == [{"model": "m", "messages": MESSAGES}]  # the server's defaults apply

    def test_options(self):
        tools = [{"type": "function", "function": {"name": "bash"}}]
        client = FakeClient([reply("hi")])
        ask(client, temperature=0, max_tokens=64, tools=tools)
        assert client.requests[0] == {"model": "m", "messages": MESSAGES, "temperature": 0, "max_tokens": 64, "tools": tools}

    def test_an_empty_tool_list_is_not_sent(self):
        client = FakeClient([reply("hi")])
        ask(client, tools=[])
        assert "tools" not in client.requests[0]

    def test_messages_may_be_a_tuple_and_are_not_changed(self):
        client = FakeClient([reply("hi")])
        ask(client, messages=tuple(MESSAGES))
        assert client.requests[0]["messages"] == MESSAGES

    def test_a_bad_app_fails_before_the_model_is_called(self):
        client = FakeClient([])
        with pytest.raises(ValueError, match="app must be"):
            ask(client, app="../x")
        assert client.requests == []

    def test_the_client_comes_from_the_environment(self, monkeypatch):
        # nothing scripted: oMLX's client is made from OMLX_* on first use
        client = FakeClient([reply("hi")])
        monkeypatch.setattr(omlx, "make_client", lambda: client)
        assert chat(app="jewels", model="m", messages=MESSAGES).content == "hi"
        assert omlx.client() is client  # made once, then reused


class TestRecord:
    def test_full_request_and_response(self, orchestra_dir):
        client = FakeClient([reply("42", reasoning="6*7", usage=USAGE)])
        ask(client, temperature=0.4)
        (record,) = read_log(orchestra_dir, "jewels")
        assert record["kind"] == "chat"
        assert record["request"] == client.requests[0]  # exactly what was sent
        assert record["response"] == {"content": "42", "reasoning": "6*7", "tool_calls": [],
                                      "finish_reason": "stop", "model": "fake-model"}
        assert record["error"] is None
        assert record["seconds"] >= 0

    def test_usage_keeps_the_servers_own_block(self, orchestra_dir):
        ask(FakeClient([reply("42", usage=USAGE)]))
        usage = read_log(orchestra_dir, "jewels")[0]["usage"]
        assert (usage["prompt"], usage["completion"], usage["cached"]) == (20, 5, 12)
        assert usage["server"]["total_tokens"] == 25

    def test_metadata_parent_and_session(self, orchestra_dir):
        ask(FakeClient([reply("hi")]), metadata={"job": "3f2a", "idea": "autumn"}, parent="tool_x", session="s1")
        record = read_log(orchestra_dir, "jewels")[0]
        assert record["metadata"] == {"job": "3f2a", "idea": "autumn"}
        assert (record["parent"], record["session"]) == ("tool_x", "s1")

    @pytest.mark.parametrize("failure", [ConnectionError("server gone"), KeyboardInterrupt()])
    def test_a_failed_call_is_recorded_then_raised(self, orchestra_dir, failure):
        with pytest.raises(type(failure)):
            ask(FakeClient([failure]))
        (record,) = read_log(orchestra_dir, "jewels")
        assert record["error"].startswith(type(failure).__name__)
        assert record["request"]["messages"] == MESSAGES  # what was being asked is still kept
        assert "response" not in record

    def test_missing_api_key_is_recorded(self, orchestra_dir, monkeypatch):
        monkeypatch.delenv("OMLX_API_KEY", raising=False)
        with pytest.raises(MissingAPIKey):
            chat(app="jewels", model="m", messages=MESSAGES)
        assert read_log(orchestra_dir, "jewels")[0]["error"].startswith("MissingAPIKey")
